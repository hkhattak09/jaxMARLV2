"""Transformer critic components for MAPPO-T.

The module mirrors MACA/harl/models/base/transformer.py while using Flax
modules and JAX array updates.  The critic treats agents as the transformer
tokens, and optional recurrent state is applied independently per agent before
the transformer blocks, matching MACA's RNNLayer usage.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
from flax.linen.initializers import ones, zeros, xavier_uniform, normal


def get_active_func(name: str) -> Callable:
    if name == "relu":
        return nn.relu
    if name == "gelu":
        return nn.gelu
    if name == "tanh":
        return jnp.tanh
    raise ValueError(f"Unsupported activation function: {name}")


def get_dense_initializer(weight_init: str, gain: float = 1.0):
    """Get Dense layer initializer based on weight_init config.
    
    Args:
        weight_init: One of "default", "tfixup", "nanogpt".
        gain: Gain factor for scaling (used by tfixup).
        
    Returns:
        Tuple of (kernel_init, bias_init) callables.
    """
    if weight_init == "tfixup":
        # T-Fixup: Xavier uniform for kernel, zero for bias
        kernel_init = xavier_uniform()
        bias_init = zeros
    elif weight_init == "nanogpt":
        # nanoGPT: normal with std=0.02, zero bias
        kernel_init = normal(0.02)
        bias_init = zeros
    else:
        # Default: use whatever Flax default is (typically lecun_normal)
        return None, None
    return kernel_init, bias_init


def dense_init_kwargs(cfg):
    kernel_init, bias_init = get_dense_initializer(cfg.get("weight_init", "default"))
    if kernel_init is None:
        return {}
    return {"kernel_init": kernel_init, "bias_init": bias_init}


def apply_tfixup_scaling(variables, cfg):
    """Apply T-Fixup scaling to initialized transformer variables.
    
    This function applies the post-init scaling described in the T-Fixup paper:
    - Encoder blocks: scale mlp.0, mlp.2, attn.proj by 0.67 * n_encode_layer^-0.25
    - Encoder attn.value: same scale * sqrt(2)
    - Decoder blocks (if n_decode_layer > 0): scale by (9 * n_decode_layer)^-0.25
    - Decoder attn.value: same scale * sqrt(2)
    
    Args:
        variables: Flax variables dict (params).
        cfg: Transformer config dict.
        
    Returns:
        Updated variables dict with T-Fixup scaling applied.
    """
    n_encode_layer = cfg.get("n_encode_layer", 0)
    n_decode_layer = cfg.get("n_decode_layer", 0)
    
    if n_encode_layer == 0 and n_decode_layer == 0:
        return variables
    
    mutable = unfreeze(variables)
    params = mutable["params"]
    encoder_params = params.get("transformer_encoder", params)
    
    # Encoder scaling factor
    if n_encode_layer > 0:
        enc_scale = 0.67 * (n_encode_layer ** -0.25)
        enc_value_scale = enc_scale * math.sqrt(2)
    else:
        enc_scale = 1.0
        enc_value_scale = 1.0
    
    # Decoder scaling factor
    if n_decode_layer > 0:
        dec_scale = (9 * n_decode_layer) ** -0.25
        dec_value_scale = dec_scale * math.sqrt(2)
    else:
        dec_scale = 1.0
        dec_value_scale = 1.0
    
    # Apply scaling to encoder blocks
    for idx in range(n_encode_layer):
        block_name = f"block_{idx}"
        if block_name in encoder_params:
            block = encoder_params[block_name]
            
            # Scale MLP weights: mlp.mlp_0 and mlp.mlp_2
            if "mlp" in block:
                if "mlp_0" in block["mlp"]:
                    block["mlp"]["mlp_0"]["kernel"] = block["mlp"]["mlp_0"]["kernel"] * enc_scale
                if "mlp_2" in block["mlp"]:
                    block["mlp"]["mlp_2"]["kernel"] = block["mlp"]["mlp_2"]["kernel"] * enc_scale
            
            # Scale attention weights
            if "attn" in block:
                if "proj" in block["attn"]:
                    block["attn"]["proj"]["kernel"] = block["attn"]["proj"]["kernel"] * enc_scale
                if "value" in block["attn"]:
                    block["attn"]["value"]["kernel"] = block["attn"]["value"]["kernel"] * enc_value_scale
    
    # Apply scaling to decoder blocks
    for idx in range(n_decode_layer):
        block_name = f"cross_block_{idx}"
        if block_name in encoder_params:
            block = encoder_params[block_name]
            
            # Scale MLP weights
            if "mlp" in block:
                if "mlp_0" in block["mlp"]:
                    block["mlp"]["mlp_0"]["kernel"] = block["mlp"]["mlp_0"]["kernel"] * dec_scale
                if "mlp_2" in block["mlp"]:
                    block["mlp"]["mlp_2"]["kernel"] = block["mlp"]["mlp_2"]["kernel"] * dec_scale
            
            # Scale attention weights (both attn1 and attn2)
            for attn_name in ["attn1", "attn2"]:
                if attn_name in block:
                    if "proj" in block[attn_name]:
                        block[attn_name]["proj"]["kernel"] = block[attn_name]["proj"]["kernel"] * dec_scale
                    if "value" in block[attn_name]:
                        block[attn_name]["value"]["kernel"] = block[attn_name]["value"]["kernel"] * dec_value_scale
    
    return freeze(mutable)


class LayerNorm(nn.Module):
    ndim: int
    bias: bool = True
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        weight = self.param("weight", ones, (self.ndim,))
        y = weight * (x - mean) / jnp.sqrt(var + self.eps)
        if self.bias:
            y = y + self.param("bias", zeros, (self.ndim,))
        return y


class SelfAttention(nn.Module):
    cfg: dict

    @nn.compact
    def __call__(
        self,
        query,
        key,
        value,
        output_attentions: bool = False,
        deterministic: bool = True,
    ):
        n_head = self.cfg["n_head"]
        n_embd = self.cfg["n_embd"]
        bias = self.cfg["bias"]
        dropout = self.cfg["dropout"]
        is_causal = self.cfg.get("is_causal", False)
        n_block = self.cfg.get("n_block", query.shape[-2])
        init_kwargs = dense_init_kwargs(self.cfg)

        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
        head_size = n_embd // n_head

        q = nn.Dense(n_embd, use_bias=bias, name="query", **init_kwargs)(query)
        k = nn.Dense(n_embd, use_bias=bias, name="key", **init_kwargs)(key)
        v = nn.Dense(n_embd, use_bias=bias, name="value", **init_kwargs)(value)

        q = q.reshape(q.shape[0], q.shape[1], n_head, head_size).transpose(0, 2, 1, 3)
        k = k.reshape(k.shape[0], k.shape[1], n_head, head_size).transpose(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], n_head, head_size).transpose(0, 2, 1, 3)

        att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_size)
        if is_causal:
            mask = jnp.tril(jnp.ones((n_block, n_block), dtype=att.dtype))
            mask = jnp.where(mask == 0, jnp.array(-1e10, dtype=att.dtype), 0.0)
            att = att + mask[None, None, : att.shape[-2], : att.shape[-1]]

        att = nn.softmax(att, axis=-1)
        att = nn.Dropout(rate=dropout)(att, deterministic=deterministic)

        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(y.shape[0], y.shape[2], n_embd)
        y = nn.Dense(n_embd, use_bias=bias, name="proj", **init_kwargs)(y)
        y = nn.Dropout(rate=dropout)(y, deterministic=deterministic)
        return (y, att) if output_attentions else (y, None)


class FeedForward(nn.Module):
    cfg: dict

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        n_embd = self.cfg["n_embd"]
        bias = self.cfg["bias"]
        active_fn = get_active_func(self.cfg["active_fn"])
        init_kwargs = dense_init_kwargs(self.cfg)
        x = nn.Dense(n_embd, use_bias=bias, name="mlp_0", **init_kwargs)(x)
        x = active_fn(x)
        x = nn.Dense(n_embd, use_bias=bias, name="mlp_2", **init_kwargs)(x)
        x = nn.Dropout(rate=self.cfg["dropout"])(x, deterministic=deterministic)
        return x


class EncodeBlock(nn.Module):
    cfg: dict

    @nn.compact
    def __call__(self, x, output_attentions: bool = False, deterministic: bool = True):
        n_embd = self.cfg["n_embd"]
        bias = self.cfg["bias"]

        z = LayerNorm(n_embd, bias=bias, name="ln1")(x)
        y, att = SelfAttention(self.cfg, name="attn")(
            z, z, z, output_attentions, deterministic
        )
        x = x + y
        z = LayerNorm(n_embd, bias=bias, name="ln2")(x)
        x = x + FeedForward(self.cfg, name="mlp")(z, deterministic)
        return x, att


class DecodeBlock(nn.Module):
    cfg: dict

    @nn.compact
    def __call__(self, x, src, deterministic: bool = True):
        n_embd = self.cfg["n_embd"]
        bias = self.cfg["bias"]

        z = LayerNorm(n_embd, bias=bias, name="ln1")(x)
        y, _ = SelfAttention(self.cfg, name="attn1")(z, z, z, False, deterministic)
        x = x + y

        z = LayerNorm(n_embd, bias=bias, name="ln3")(x)
        y, _ = SelfAttention(self.cfg, name="attn2")(z, src, src, False, deterministic)
        x = x + y

        z = LayerNorm(n_embd, bias=bias, name="ln2")(x)
        x = x + FeedForward(self.cfg, name="mlp")(z, deterministic)
        return x


class MLPHead(nn.Module):
    hidden_dim: int
    out_dim: int
    active_fn_name: str
    bias: bool = True
    with_norm: bool = True
    extra_hidden: bool = False
    weight_init: str = "default"

    @nn.compact
    def __call__(self, x):
        active_fn = get_active_func(self.active_fn_name)
        init_kwargs = dense_init_kwargs({"weight_init": self.weight_init})
        x = nn.Dense(self.hidden_dim, use_bias=self.bias, name="dense_0", **init_kwargs)(x)
        x = active_fn(x)
        if self.with_norm:
            x = LayerNorm(self.hidden_dim, bias=self.bias, name="norm_0")(x)
        if self.extra_hidden:
            x = nn.Dense(self.hidden_dim, use_bias=self.bias, name="dense_1", **init_kwargs)(x)
            x = active_fn(x)
            if self.with_norm:
                x = LayerNorm(self.hidden_dim, bias=self.bias, name="norm_1")(x)
        return nn.Dense(self.out_dim, use_bias=self.bias, name="out", **init_kwargs)(x)


class Encoder(nn.Module):
    """MAPPO-T transformer critic.

    Args:
        obs: ``(batch, n_agents, obs_dim)``
        action: one-hot actions ``(batch, n_agents, action_dim)``
        policy_prob: policy probabilities with the same shape as ``action``
        rnn_states: ``(batch, n_agents, n_embd)``
        resets: bool/0-1 tensor, True means reset recurrent state
    """

    args: dict
    obs_space: Any
    act_space: Any

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
    ):
        cfg = self.args["transformer"]
        hidden_sizes = self.args["hidden_sizes"]
        active_fn = get_active_func(cfg["active_fn"])
        n_embd = cfg["n_embd"]
        zs_dim = cfg["zs_dim"]
        n_block = cfg["n_block"] or obs.shape[-2]
        bias = cfg["bias"]
        dropout = cfg["dropout"]
        n_encode_layer = cfg["n_encode_layer"]
        n_decode_layer = cfg["n_decode_layer"]
        head_aggr = cfg.get("aggregation", "mean")
        att_sigma = min(cfg["att_sigma"] / n_block, 1.0)
        vq_bsln_coef = cfg["vq_bsln_coef"]
        vq_coma_bsln_coef = cfg["vq_coma_bsln_coef"]
        init_kwargs = dense_init_kwargs(cfg)

        if self.args["use_feature_normalization"]:
            obs = LayerNorm(obs.shape[-1], bias=bias, name="feature_norm")(obs)

        s_rep = nn.Dense(n_embd, use_bias=bias, name="obs_enc_0", **init_kwargs)(obs)
        s_rep = active_fn(s_rep)
        s_rep = LayerNorm(n_embd, bias=bias, name="obs_enc_1")(s_rep)
        s_rep = nn.Dropout(rate=dropout)(s_rep, deterministic=deterministic)

        batch_size, n_agents, _ = s_rep.shape
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

        all_self_attns = []
        for idx in range(n_encode_layer):
            s_rep, self_attn = EncodeBlock(cfg, name=f"block_{idx}")(
                s_rep, output_attentions, deterministic
            )
            all_self_attns.append(self_attn)
        s_rep = LayerNorm(n_embd, bias=bias, name="ln1")(s_rep)

        zs = s_rep.reshape(batch_size, -1)
        zs = nn.Dense(zs_dim, use_bias=bias, name="s_enc_0", **init_kwargs)(zs)
        zs = active_fn(zs)
        zs = LayerNorm(zs_dim, bias=bias, name="s_enc_1")(zs)

        v_loc = MLPHead(
            hidden_dim=n_embd,
            out_dim=1,
            active_fn_name=cfg["active_fn"],
            bias=bias,
            weight_init=cfg.get("weight_init", "default"),
            name="v_head",
        )(zs)

        if output_attentions:
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

        if n_decode_layer:
            act_dense = nn.Dense(n_embd, use_bias=bias, name="act_enc_0", **init_kwargs)
            act_norm = LayerNorm(n_embd, bias=bias, name="act_enc_1")
            cross_blocks = [
                DecodeBlock(cfg, name=f"cross_block_{idx}") for idx in range(n_decode_layer)
            ]
            ln2 = LayerNorm(n_embd, bias=bias, name="ln2")
            sa_dense = nn.Dense(zs_dim, use_bias=bias, name="sa_enc_0", **init_kwargs)
            sa_norm = LayerNorm(zs_dim, bias=bias, name="sa_enc_1")

            def act_encoder(x):
                x = act_dense(x)
                x = active_fn(x)
                return act_norm(x)

            def decode_actions(act_like, src):
                rep = act_encoder(act_like)
                for block in cross_blocks:
                    rep = block(rep, src, deterministic)
                rep = ln2(rep)
                rep = rep.reshape(rep.shape[0], -1)
                rep = sa_dense(rep)
                rep = active_fn(rep)
                return sa_norm(rep)

            zsa = decode_actions(action, s_rep)
            zspi = decode_actions(policy_prob, s_rep)
        else:
            sa_encoder = nn.Dense(zs_dim, use_bias=bias, name="sa_encoder", **init_kwargs)

            def encode_state_action(act_like):
                rep = jnp.concatenate([zs, act_like.reshape(batch_size, -1)], axis=-1)
                return sa_encoder(rep)

            zsa = encode_state_action(action)
            zspi = encode_state_action(policy_prob)

        if n_decode_layer:
            q_head = MLPHead(
                hidden_dim=n_embd,
                out_dim=1,
                active_fn_name=cfg["active_fn"],
                bias=bias,
                extra_hidden=True,
                weight_init=cfg.get("weight_init", "default"),
                name="q_head",
            )
        else:
            q_head = nn.Dense(1, use_bias=bias, name="q_head", **init_kwargs)
        q_loc = q_head(zsa)
        eq_loc = q_head(zspi)

        if output_attentions:
            if n_decode_layer:
                repeated_s = jnp.repeat(s_rep[:, None, :, :], n_agents, axis=1)
                repeated_s = repeated_s.reshape(batch_size * n_agents, n_agents, n_embd)

                def decode_mixed(act_like):
                    act_like = act_like.reshape(batch_size * n_agents, n_agents, -1)
                    return decode_actions(act_like, repeated_s)

                zsapi = decode_mixed(mix_a_pi)
                zsapi_coma = decode_mixed(coma_a_pi)
            else:
                zs_rep = jnp.repeat(zs[:, None, :], n_agents, axis=1)
                zsapi = sa_encoder(
                    jnp.concatenate(
                        [
                            zs_rep.reshape(batch_size * n_agents, -1),
                            mix_a_pi.reshape(batch_size * n_agents, -1),
                        ],
                        axis=-1,
                    )
                )
                zsapi_coma = sa_encoder(
                    jnp.concatenate(
                        [
                            zs_rep.reshape(batch_size * n_agents, -1),
                            coma_a_pi.reshape(batch_size * n_agents, -1),
                        ],
                        axis=-1,
                    )
                )

            vq_loc = q_head(zsapi).reshape(batch_size, n_agents, -1)
            vq_coma_loc = q_head(zsapi_coma).reshape(batch_size, n_agents, -1)
            final_attentions = joint_attentions[:, -1]
        else:
            vq_loc = None
            vq_coma_loc = None
            final_attentions = None

        return (
            v_loc,
            q_loc,
            eq_loc,
            vq_loc,
            vq_coma_loc,
            baseline_weights,
            final_attentions,
            zs,
            zsa,
            rnn_states,
        )

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

    def _get_baseline_weights(
        self,
        joint_attentions,
        vq_coef,
        vq_coma_coef,
        sigma: Optional[float] = None,
        layer=-1,
    ):
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


def compute_joint_attention(attn_layers, add_residual=True):
    """Compute MACA attention rollout.

    Args:
        attn_layers: ``(batch, layers, tokens, tokens)`` attention matrices.
        add_residual: add identity residual attention and row-normalize first.
    """
    _, n_layers, n_tokens, _ = attn_layers.shape
    if add_residual:
        residual = jnp.eye(n_tokens, dtype=attn_layers.dtype).reshape(
            1, 1, n_tokens, n_tokens
        )
        aug_attn = attn_layers + residual
        aug_attn = aug_attn / jnp.sum(aug_attn, axis=-1, keepdims=True)
    else:
        aug_attn = attn_layers

    rolled = [aug_attn[:, 0]]
    for idx in range(1, n_layers):
        rolled.append(aug_attn[:, idx] @ rolled[-1])
    return jnp.stack(rolled, axis=1)
