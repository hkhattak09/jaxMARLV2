"""MACA-Role critic with role-specific V/Q heads.

Supports:
  - Exp 2 & 4: Role-specific heads on shared transformer

Key design:
  - Shared transformer encoder + attention (unchanged)
  - Shared z_intermediate = Dense(256)(flatten(H))
  - Per-role z_k = Dense(128)→ReLU→LN→Dense(64)
  - Per-role V_k = MLPHead(64→1) on z_k
  - Shared linear sa_encoder on concat(z_k, action_flat)
  - Per-role Q_k = Dense(1) on zsa_k (linear preserves marginalization)
  - GAE targets = mean across roles
  - Per-agent baseline uses Q_{r_i}
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import ones, zeros, xavier_uniform, normal, orthogonal

from .transformer import (
    Encoder,
    MLPHead,
    LayerNorm,
    get_active_func,
    dense_init_kwargs,
    compute_joint_attention,
)


class RoleEncoder(nn.Module):
    """Transformer encoder with role-specific V/Q heads.

    Mirrors the existing Encoder but adds per-role projections and heads.
    """

    args: dict
    obs_space: Any
    act_space: Any
    n_roles: int = 6

    @nn.compact
    def __call__(
        self,
        obs,
        action,
        policy_prob,
        rnn_states,
        resets,
        output_attentions: bool = False,
        deterministic: bool = True,
        role_ids: jnp.ndarray = None,
    ):
        """Forward pass returning role-specific and shared outputs.

        Args:
            obs: ``(batch, n_agents, obs_dim)``
            action: one-hot actions ``(batch, n_agents, action_dim)``
            policy_prob: policy probabilities ``(batch, n_agents, action_dim)``
            rnn_states: ``(batch, n_agents, n_embd)``
            resets: bool/0-1 tensor
            output_attentions: whether to compute attention rollout
            deterministic: dropout flag
            role_ids: ``(batch, n_agents)`` integer role IDs

        Returns:
            Tuple of (all_v, all_q, all_eq, vq, vq_coma, baseline_weights,
                      final_attentions, zs, zsa, rnn_states, z_k_embs)
            where all_v/all_q/all_eq have shape (n_roles, batch, 1).
        """
        cfg = self.args["transformer"]
        n_embd = cfg["n_embd"]
        zs_dim = cfg["zs_dim"]
        bias = cfg["bias"]
        dropout = cfg["dropout"]
        n_encode_layer = cfg["n_encode_layer"]
        n_decode_layer = cfg["n_decode_layer"]
        head_aggr = cfg.get("aggregation", "mean")
        att_sigma = min(cfg["att_sigma"] / obs.shape[-2], 1.0)
        vq_bsln_coef = cfg["vq_bsln_coef"]
        vq_coma_bsln_coef = cfg["vq_coma_bsln_coef"]
        init_kwargs = dense_init_kwargs(cfg)
        active_fn = get_active_func(cfg["active_fn"])

        # ---- Feature normalization (shared) --------------------------------
        if self.args["use_feature_normalization"]:
            obs = LayerNorm(obs.shape[-1], bias=bias, name="feature_norm")(obs)

        # ---- Observation encoder (shared) -----------------------------------
        s_rep = nn.Dense(n_embd, use_bias=bias, name="obs_enc_0", **init_kwargs)(obs)
        s_rep = active_fn(s_rep)
        s_rep = LayerNorm(n_embd, bias=bias, name="obs_enc_1")(s_rep)
        s_rep = nn.Dropout(rate=dropout)(s_rep, deterministic=deterministic)

        batch_size, n_agents, _ = s_rep.shape

        # ---- Optional recurrent layer (shared) ------------------------------
        if self.args["use_naive_recurrent_policy"] or self.args["use_recurrent_policy"]:
            flat_rep = s_rep.reshape(batch_size * n_agents, n_embd)
            flat_state = rnn_states.reshape(batch_size * n_agents, n_embd)
            flat_resets = resets.reshape(batch_size * n_agents).astype(bool)
            flat_state = jnp.where(flat_resets[:, None], jnp.zeros_like(flat_state), flat_state)
            flat_state, flat_rep = nn.GRUCell(features=n_embd, name="rnn_cell")(
                flat_state, flat_rep
            )
            flat_rep = nn.LayerNorm(epsilon=1e-5, name="rnn_norm")(flat_rep)
            s_rep = flat_rep.reshape(batch_size, n_agents, n_embd)
            rnn_states = flat_state.reshape(batch_size, n_agents, n_embd)

        # ---- Transformer encoder blocks (shared) ----------------------------
        all_self_attns = []
        for idx in range(n_encode_layer):
            from .transformer import EncodeBlock
            s_rep, self_attn = EncodeBlock(cfg, name=f"block_{idx}")(
                s_rep, output_attentions, deterministic
            )
            all_self_attns.append(self_attn)
        s_rep = LayerNorm(n_embd, bias=bias, name="ln1")(s_rep)

        # ---- Shared state projection z_intermediate -------------------------
        zs = s_rep.reshape(batch_size, -1)
        zs = nn.Dense(zs_dim, use_bias=bias, name="s_enc_0", **init_kwargs)(zs)
        zs = active_fn(zs)
        zs = LayerNorm(zs_dim, bias=bias, name="s_enc_1")(zs)

        # ---- Role-conditioned or per-role projections z_k -----------------------
        role_conditioning = self.args.get("ROLE_CONDITIONING", "heads")
        z_k_dims = self.args.get("role_z_k_dims", [128, 64])

        if role_conditioning == "embedding":
            emb_dim = self.args.get("ROLE_EMB_DIM", 16)
            z_k_dim = z_k_dims[-1] if z_k_dims else zs.shape[-1]

            role_emb = self.param(
                "role_emb", orthogonal(), (self.n_roles, emb_dim)
            )
            z_shared = nn.Dense(
                z_k_dim, use_bias=bias, name="z_shared_proj", **init_kwargs
            )(zs)
            role_offsets = nn.Dense(
                z_k_dim, use_bias=bias, name="role_offset_proj", **init_kwargs
            )(role_emb)  # (n_roles, z_k_dim)

            z_k_embs = z_shared[None, :, :] + role_offsets[:, None, :]

            if z_k_dims:
                for idx, dim in enumerate(z_k_dims[:-1] if len(z_k_dims) > 1 else []):
                    z_k_embs = nn.Dense(
                        dim, use_bias=bias, name=f"z_enc_mlp_{idx}", **init_kwargs
                    )(z_k_embs)
                    z_k_embs = active_fn(z_k_embs)
                    z_k_embs = LayerNorm(
                        dim, bias=bias, name=f"z_enc_norm_{idx}"
                    )(z_k_embs)
                z_k_embs = nn.Dense(
                    z_k_dims[-1], use_bias=bias, name=f"z_enc_mlp_{len(z_k_dims)-1}", **init_kwargs
                )(z_k_embs)
        else:
            z_k_embs = []
            for k in range(self.n_roles):
                z_k = zs
                for idx, dim in enumerate(z_k_dims):
                    z_k = nn.Dense(dim, use_bias=bias, name=f"z_enc_{k}_dense_{idx}", **init_kwargs)(z_k)
                    if idx < len(z_k_dims) - 1:
                        z_k = active_fn(z_k)
                        z_k = LayerNorm(dim, bias=bias, name=f"z_enc_{k}_norm_{idx}")(z_k)
                z_k_embs.append(z_k)
            z_k_embs = jnp.stack(z_k_embs, axis=0)

        # ---- V-heads (shared in embedding mode, per-role in heads mode) ----------
        v_head_dims = self.args.get("role_v_head_dims", [64])
        if role_conditioning == "embedding":
            all_v = z_k_embs
            for idx, dim in enumerate(v_head_dims):
                all_v = nn.Dense(
                    dim, use_bias=bias, name=f"v_head_dense_{idx}", **init_kwargs
                )(all_v)
                all_v = active_fn(all_v)
                all_v = LayerNorm(dim, bias=bias, name=f"v_head_norm_{idx}")(all_v)
            all_v = nn.Dense(1, use_bias=bias, name="v_head_out", **init_kwargs)(all_v)
        else:
            all_v = []
            for k in range(self.n_roles):
                v = z_k_embs[k]
                for idx, dim in enumerate(v_head_dims):
                    v = nn.Dense(
                        dim,
                        use_bias=bias,
                        name=f"v_head_{k}_dense_{idx}",
                        **init_kwargs,
                    )(v)
                    v = active_fn(v)
                    v = LayerNorm(dim, bias=bias, name=f"v_head_{k}_norm_{idx}")(v)
                v = nn.Dense(1, use_bias=bias, name=f"v_head_{k}_out", **init_kwargs)(v)
                all_v.append(v)
            all_v = jnp.stack(all_v, axis=0)

        # ---- Q/EQ heads (shared in embedding mode, per-role in heads mode) ------
        sa_encoder = nn.Dense(zs_dim, use_bias=bias, name="sa_encoder", **init_kwargs)

        all_zsa = []
        all_zspi = []

        for k in range(self.n_roles):
            zsa_k = sa_encoder(
                jnp.concatenate([z_k_embs[k], action.reshape(batch_size, -1)], axis=-1)
            )
            zspi_k = sa_encoder(
                jnp.concatenate([z_k_embs[k], policy_prob.reshape(batch_size, -1)], axis=-1)
            )
            all_zsa.append(zsa_k)
            all_zspi.append(zspi_k)

        all_zsa = jnp.stack(all_zsa, axis=0)    # (n_roles, batch, zs_dim)
        all_zspi = jnp.stack(all_zspi, axis=0)  # (n_roles, batch, zs_dim)

        if role_conditioning == "embedding":
            q_head = nn.Dense(1, use_bias=bias, name="q_head_shared", **init_kwargs)
            all_q = q_head(all_zsa)               # (n_roles, batch, 1)
            all_eq = q_head(all_zspi)             # (n_roles, batch, 1)
        else:
            all_q = []
            all_eq = []
            for k in range(self.n_roles):
                q_head = nn.Dense(1, use_bias=bias, name=f"q_head_{k}", **init_kwargs)
                all_q.append(q_head(all_zsa[k]))
                all_eq.append(q_head(all_zspi[k]))
            all_q = jnp.stack(all_q, axis=0)
            all_eq = jnp.stack(all_eq, axis=0)

        # ---- Attention-derived baseline (shared) ----------------------------
        if output_attentions and n_encode_layer > 0:
            all_self_attns = jnp.stack(all_self_attns, axis=1)
            if head_aggr == "mean":
                all_self_attns = jnp.mean(all_self_attns, axis=2)
            elif head_aggr == "max":
                all_self_attns = jnp.max(all_self_attns, axis=2)
            else:
                raise ValueError(f"Unsupported attention aggregation: {head_aggr}")

            joint_attentions = compute_joint_attention(
                all_self_attns, add_residual=cfg.get("att_roll_res", False)
            )
            mix_a_pi, coma_a_pi = self._get_mixed_action_pi(
                action, policy_prob, joint_attentions, att_sigma
            )
            baseline_weights = self._get_baseline_weights(
                joint_attentions, vq_bsln_coef, vq_coma_bsln_coef, att_sigma
            )
        else:
            joint_attentions = None
            mix_a_pi = None
            coma_a_pi = None
            baseline_weights = None

        # ---- VQ / VQ-COMA per role ------------------------------------------
        vq_per_role = []
        vq_coma_per_role = []

        if output_attentions and joint_attentions is not None:
            if role_conditioning == "embedding":
                # Vectorized: compute all roles at once with shared heads
                # mix_a_pi: (batch, n_agents, n_agents, action_dim)
                # coma_a_pi: (batch, n_agents, n_agents, action_dim)
                # For each role k, we encode: sa_encoder(concat(z_k_embs[k] tiled, mixed_actions))
                mix_flat = mix_a_pi.reshape(batch_size * n_agents, n_agents, -1)
                coma_flat = coma_a_pi.reshape(batch_size * n_agents, n_agents, -1)
                mix_flat = jnp.tile(mix_flat, (self.n_roles, 1, 1))
                coma_flat = jnp.tile(coma_flat, (self.n_roles, 1, 1))
                mix_flat = mix_flat.reshape(self.n_roles * batch_size * n_agents, -1)
                coma_flat = coma_flat.reshape(self.n_roles * batch_size * n_agents, -1)

                # Tile z_k_embs per agent: (n_roles, batch, z_k_dim) -> (n_roles, batch, n_agents, z_k_dim)
                zs_rep_all = jnp.repeat(z_k_embs[:, :, None, :], n_agents, axis=2)
                zs_rep_all = zs_rep_all.reshape(self.n_roles * batch_size * n_agents, -1)

                zsa_mix_all = sa_encoder(jnp.concatenate([zs_rep_all, mix_flat], axis=-1))
                zsa_coma_all = sa_encoder(jnp.concatenate([zs_rep_all, coma_flat], axis=-1))
                zsa_mix_all = active_fn(zsa_mix_all)
                zsa_mix_all = LayerNorm(zs_dim, bias=bias, name="vq_mix_ln")(zsa_mix_all)
                zsa_coma_all = active_fn(zsa_coma_all)
                zsa_coma_all = LayerNorm(zs_dim, bias=bias, name="vq_coma_ln")(zsa_coma_all)

                vq_head = nn.Dense(1, use_bias=bias, name="vq_head_shared", **init_kwargs)
                vq_coma_head = nn.Dense(1, use_bias=bias, name="vq_coma_head_shared", **init_kwargs)
                vq_all = vq_head(zsa_mix_all).reshape(self.n_roles, batch_size, n_agents, 1)
                vq_coma_all = vq_coma_head(zsa_coma_all).reshape(self.n_roles, batch_size, n_agents, 1)
                vq_per_role = vq_all
                vq_coma_per_role = vq_coma_all
            else:
                for k in range(self.n_roles):
                    mix_k = mix_a_pi.reshape(batch_size * n_agents, n_agents, -1)
                    coma_k = coma_a_pi.reshape(batch_size * n_agents, n_agents, -1)

                    zs_rep = jnp.repeat(z_k_embs[k][:, None, :], n_agents, axis=1)
                    zs_rep = zs_rep.reshape(batch_size * n_agents, -1)

                    def encode_mixed(m):
                        m = m.reshape(batch_size * n_agents, -1)
                        return sa_encoder(jnp.concatenate([zs_rep, m], axis=-1))

                    zsa_mix = encode_mixed(mix_k)
                    zsa_mix = active_fn(zsa_mix)
                    zsa_mix = LayerNorm(zs_dim, bias=bias, name=f"vq_mix_ln_{k}")(zsa_mix)

                    zsa_coma = encode_mixed(coma_k)
                    zsa_coma = active_fn(zsa_coma)
                    zsa_coma = LayerNorm(zs_dim, bias=bias, name=f"vq_coma_ln_{k}")(zsa_coma)

                    vq_k = nn.Dense(1, use_bias=bias, name=f"vq_head_{k}", **init_kwargs)(zsa_mix)
                    vq_coma_k = nn.Dense(1, use_bias=bias, name=f"vq_coma_head_{k}", **init_kwargs)(zsa_coma)

                    vq_per_role.append(vq_k.reshape(batch_size, n_agents, -1))
                    vq_coma_per_role.append(vq_coma_k.reshape(batch_size, n_agents, -1))

                vq_per_role = jnp.stack(vq_per_role, axis=0)
                vq_coma_per_role = jnp.stack(vq_coma_per_role, axis=0)
        else:
            vq_per_role = None
            vq_coma_per_role = None

        # ---- Gather per-agent VQ/VQ-COMA using role_ids ---------------------
        if vq_per_role is not None and role_ids is not None:
            # role_ids: (batch, n_agents)
            # vq_per_role: (n_roles, batch, n_agents, 1)
            vq = self._gather_by_role(vq_per_role, role_ids)       # (batch, n_agents, 1)
            vq_coma = self._gather_by_role(vq_coma_per_role, role_ids)
        else:
            vq = None
            vq_coma = None

        if output_attentions and joint_attentions is not None:
            final_attentions = joint_attentions[:, -1]
        else:
            final_attentions = None

        # Return ALL role-specific values + gathered per-agent values
        return (
            all_v,           # (n_roles, batch, 1)
            all_q,           # (n_roles, batch, 1)
            all_eq,          # (n_roles, batch, 1)
            vq,              # (batch, n_agents, 1) or None
            vq_coma,         # (batch, n_agents, 1) or None
            baseline_weights,
            final_attentions,
            zs,              # (batch, zs_dim) shared
            all_zsa,         # (n_roles, batch, zs_dim)
            rnn_states,
            z_k_embs,        # (n_roles, batch, 64)
        )

    # -----------------------------------------------------------------------
    # Helpers (copied from Encoder for standalone use)
    # -----------------------------------------------------------------------

    def _get_mixed_action_pi(self, action, pi, joint_attentions, sigma, layer=-1):
        if joint_attentions.ndim == 4:
            attn = joint_attentions[:, layer]
        else:
            attn = joint_attentions
        batch_size, n_agents, action_dim = action.shape
        pi_rep = jnp.repeat(pi[:, None, :, :], n_agents, axis=1)
        action_rep = jnp.repeat(action[:, None, :, :], n_agents, axis=1)
        mixed = jnp.where(attn[..., None] >= sigma, pi_rep, action_rep)
        diag = jnp.eye(n_agents, dtype=action.dtype).reshape(1, n_agents, n_agents, 1)
        mixed = jnp.where(diag >= sigma, pi_rep, mixed)
        coma = jnp.where(diag >= sigma, pi_rep, action_rep)
        return mixed.reshape(batch_size, n_agents, n_agents, action_dim), coma.reshape(
            batch_size, n_agents, n_agents, action_dim
        )

    def _get_baseline_weights(self, joint_attentions, vq_coef, vq_coma_coef, sigma=None, layer=-1):
        cfg = self.args["transformer"]
        if joint_attentions.ndim == 4:
            attn = joint_attentions[:, layer]
        else:
            attn = joint_attentions
        n_agents = attn.shape[-1]
        if sigma is None:
            sigma = min(cfg["att_sigma"] / n_agents, 1.0)

        self_weights = jnp.diagonal(attn, axis1=-2, axis2=-1)[..., None]
        group_weights = jnp.where(attn >= sigma, attn, jnp.zeros_like(attn))
        diag = jnp.eye(n_agents, dtype=attn.dtype).reshape(1, n_agents, n_agents)
        group_weights = jnp.where(diag >= sigma, attn, group_weights)
        group_weights = jnp.sum(group_weights, axis=-1, keepdims=True) - self_weights
        joint_weights = jnp.sum(attn, axis=-1, keepdims=True)

        self_weights = vq_coef * self_weights
        group_weights = vq_coma_coef * group_weights
        joint_weights = jnp.clip(joint_weights - self_weights - group_weights, 0.0, 1.0)
        return jnp.concatenate([self_weights, group_weights, joint_weights], axis=-1)

    @staticmethod
    def _gather_by_role(all_values, role_ids):
        """Gather per-role values using role_ids.

        Args:
            all_values: ``(n_roles, batch, n_agents, ...)``
            role_ids: ``(batch, n_agents)`` integer indices.
        """
        n_roles, batch, n_agents, *remaining = all_values.shape
        batch_idx = jnp.arange(batch)[:, None]   # (batch, 1)
        agent_idx = jnp.arange(n_agents)[None, :]  # (1, n_agents)
        gathered = all_values[role_ids, batch_idx, agent_idx]
        return gathered


class RoleTransVCritic(nn.Module):
    """Flax port of MACA's TransVCritic with role-specific heads.

    Wraps RoleEncoder and provides convenience methods for:
    - Env-level value computation (mean across roles)
    - Per-agent baseline computation
    - Diversity penalty
    """

    config: Dict[str, Any]
    share_obs_space: Any
    obs_space: Any
    act_space: Any
    num_agents: int
    state_type: str
    n_roles: int = 6

    def _one_hot_actions(self, action):
        action = jnp.asarray(action)
        if (
            action.ndim >= 3
            and action.shape[-1] == getattr(self.act_space, "n", action.shape[-1])
            and jnp.issubdtype(action.dtype, jnp.floating)
        ):
            return action.astype(jnp.float32)
        if action.shape[-1] == 1 and action.ndim >= 3:
            action = jnp.squeeze(action, axis=-1)
        n_actions = getattr(self.act_space, "n", None)
        if n_actions is None:
            # Fallback: infer from action values
            n_actions = int(jnp.max(action)) + 1
        return jax.nn.one_hot(action.astype(jnp.int32), n_actions)

    def _encoder_step(self, encoder, carry, inputs, output_attentions, deterministic):
        obs_t, action_t, policy_t, reset_t, role_ids_t = inputs
        out = encoder(
            obs_t,
            action_t,
            policy_t,
            carry,
            reset_t,
            output_attentions,
            deterministic,
            role_ids_t,
        )
        # out[9] is rnn_states (carry), out[10] is z_k_embs
        carry_out = out[9]
        outputs = out[:9] + out[10:]
        return carry_out, outputs

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def _scan_encoder_attn_det(self, carry, inputs):
        encoder = RoleEncoder(
            args=self.config,
            obs_space=self.obs_space,
            act_space=self.act_space,
            n_roles=self.n_roles,
            name="role_encoder",
        )
        return self._encoder_step(encoder, carry, inputs, True, True)

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def _scan_encoder_no_attn_det(self, carry, inputs):
        encoder = RoleEncoder(
            args=self.config,
            obs_space=self.obs_space,
            act_space=self.act_space,
            n_roles=self.n_roles,
            name="role_encoder",
        )
        return self._encoder_step(encoder, carry, inputs, False, True)

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False, "dropout": True},
    )
    @nn.compact
    def _scan_encoder_attn_train(self, carry, inputs):
        encoder = RoleEncoder(
            args=self.config,
            obs_space=self.obs_space,
            act_space=self.act_space,
            n_roles=self.n_roles,
            name="role_encoder",
        )
        return self._encoder_step(encoder, carry, inputs, True, False)

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False, "dropout": True},
    )
    @nn.compact
    def _scan_encoder_no_attn_train(self, carry, inputs):
        encoder = RoleEncoder(
            args=self.config,
            obs_space=self.obs_space,
            act_space=self.act_space,
            n_roles=self.n_roles,
            name="role_encoder",
        )
        return self._encoder_step(encoder, carry, inputs, False, False)

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        policy_prob: jnp.ndarray,
        rnn_states: jnp.ndarray,
        resets: jnp.ndarray,
        role_ids: jnp.ndarray,
        output_attentions: bool = True,
        deterministic: bool = True,
    ) -> Tuple:
        action_onehot = self._one_hot_actions(action)

        if obs.ndim == 4:
            if output_attentions:
                scan_fn = (
                    self._scan_encoder_attn_det
                    if deterministic
                    else self._scan_encoder_attn_train
                )
            else:
                scan_fn = (
                    self._scan_encoder_no_attn_det
                    if deterministic
                    else self._scan_encoder_no_attn_train
                )
            final_carry, outputs = scan_fn(
                rnn_states,
                (obs, action_onehot, policy_prob, resets, role_ids),
            )
            (
                all_v,
                all_q,
                all_eq,
                vq,
                vq_coma,
                baseline_weights,
                final_attentions,
                zs,
                all_zsa,
                z_k_embs,
            ) = outputs
            return (
                all_v,
                all_q,
                all_eq,
                vq,
                vq_coma,
                baseline_weights,
                final_attentions,
                zs,
                all_zsa,
                final_carry,
                z_k_embs,
            )

        encoder = RoleEncoder(
            args=self.config,
            obs_space=self.obs_space,
            act_space=self.act_space,
            n_roles=self.n_roles,
            name="role_encoder",
        )
        return encoder(
            obs,
            action_onehot,
            policy_prob,
            rnn_states,
            resets,
            output_attentions,
            deterministic,
            role_ids,
        )

    # -----------------------------------------------------------------------
    # Convenience methods for training
    # -----------------------------------------------------------------------

    def get_env_level_values(self, params, obs, action, policy_prob, rnn_states, resets):
        """Return V_env, Q_env, EQ_env = mean across roles."""
        all_v, all_q, all_eq, _, _, _, _, _, _, _, _ = self.apply(
            params, obs, action, policy_prob, rnn_states, resets,
            jnp.zeros((obs.shape[0], self.num_agents), dtype=jnp.int32),
            True, True,
        )
        return (
            jnp.mean(all_v, axis=0),    # (batch, 1)
            jnp.mean(all_q, axis=0),    # (batch, 1)
            jnp.mean(all_eq, axis=0),   # (batch, 1)
        )

    def get_all_role_values(self, params, obs, action, policy_prob, rnn_states, resets):
        """Return all V_k values: (n_roles, batch, 1)."""
        all_v, _, _, _, _, _, _, _, _, _, _ = self.apply(
            params, obs, action, policy_prob, rnn_states, resets,
            jnp.zeros((obs.shape[0], self.num_agents), dtype=jnp.int32),
            True, True,
        )
        return all_v

    def compute_per_agent_baseline(
        self, params, obs, action, policy_prob, rnn_states, resets, role_ids
    ):
        """Compute MACA baseline per agent using role-specific Q heads."""
        _, _, all_eq, vq, vq_coma, baseline_weights, _, _, _, _, _ = self.apply(
            params, obs, action, policy_prob, rnn_states, resets, role_ids, True, True
        )

        if vq is None or baseline_weights is None:
            raise ValueError("output_attentions must be True for baseline computation")

        # baseline_weights: (batch, n_agents, 3) = [self, group, joint]
        eq_env = jnp.mean(all_eq, axis=0)[:, None, :]  # (batch, 1, 1)
        baseline = (
            baseline_weights[..., 0:1] * vq_coma +
            baseline_weights[..., 1:2] * vq +
            baseline_weights[..., 2:3] * eq_env
        )
        return baseline.squeeze(-1)

    def compute_diversity_penalty(self, params, obs, action, policy_prob, rnn_states, resets):
        """Activation-space diversity: -sum of pairwise L2 distances between z_k means."""
        if obs.ndim == 4:
            # Recurrent: (time, batch, n_agents, obs_dim)
            dummy_role_ids = jnp.zeros(
                (obs.shape[0], obs.shape[1], self.num_agents), dtype=jnp.int32
            )
        else:
            # Single-step: (batch, n_agents, obs_dim)
            dummy_role_ids = jnp.zeros((obs.shape[0], self.num_agents), dtype=jnp.int32)
        _, _, _, _, _, _, _, _, _, _, z_k_embs = self.apply(
            params, obs, action, policy_prob, rnn_states, resets,
            dummy_role_ids,
            False, True,
        )
        # z_k_embs: (n_roles, batch, dim)
        z_means = jnp.mean(z_k_embs, axis=1)  # (n_roles, dim)

        # Vectorized pairwise L2: diffs[i,j] = ||z_i - z_j||^2
        diffs = z_means[:, None, :] - z_means[None, :, :]  # (n_roles, n_roles, dim)
        dists = jnp.sum(jnp.square(diffs), axis=-1)  # (n_roles, n_roles)
        penalty = jnp.sum(jnp.triu(dists, k=1))  # sum over unique pairs
        count = self.n_roles * (self.n_roles - 1) // 2

        return -penalty / (count + 1e-8)

    def compute_embedding_decorrelation(self, params) -> jnp.ndarray:
        """Barlow-Twins-style decorrelation loss on critic role embeddings."""

        def _find_role_emb(p):
            if isinstance(p, dict):
                if "role_emb" in p:
                    return p["role_emb"]
                for v in p.values():
                    if isinstance(v, dict):
                        r = _find_role_emb(v)
                        if r is not None:
                            return r
            return None

        p = params.get("params", params) if isinstance(params, dict) else params
        role_emb = _find_role_emb(p)
        if role_emb is None:
            raise KeyError("role_emb not found in critic parameter tree")

        n = role_emb.shape[0]
        norm_emb = role_emb / (jnp.linalg.norm(role_emb, axis=-1, keepdims=True) + 1e-8)
        corr = norm_emb @ norm_emb.T
        off_diag = corr - jnp.eye(n)
        return jnp.sum(jnp.square(off_diag)) / (n * (n - 1) + 1e-8)

    def compute_eq_for_role(self, params, k, obs, policy_prob, rnn_states, resets):
        """Compute EQ_k(s, π) for a specific role k."""
        _, _, all_eq, _, _, _, _, _, _, _, _ = self.apply(
            params, obs, jnp.zeros_like(policy_prob), policy_prob, rnn_states, resets,
            jnp.zeros((obs.shape[0], self.num_agents), dtype=jnp.int32),
            False, True,
        )
        return all_eq[k]

    def compute_all_q_for_role(self, params, k, obs, rnn_states, resets):
        """Compute Q_k(s, a) for all possible actions (marginalization check)."""
        # This is a simplified version for testing
        # Full implementation would enumerate all actions
        action_dim = self.act_space.n
        batch = obs.shape[0]
        n_agents = self.num_agents

        # Create all possible action combinations (simplified: single-agent)
        # For full joint action space, this is action_dim^n_agents which is huge
        # We return Q for a dummy action instead
        dummy_action = jnp.zeros((batch, n_agents, action_dim))
        dummy_action = dummy_action.at[:, :, 0].set(1.0)

        _, all_q, _, _, _, _, _, _, _, _, _ = self.apply(
            params, obs, dummy_action, dummy_action, rnn_states, resets,
            jnp.zeros((batch, n_agents), dtype=jnp.int32),
            False, True,
        )
        return all_q[k]
