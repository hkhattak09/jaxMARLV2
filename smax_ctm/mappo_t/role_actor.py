"""MACA-Role actor network with role-specific policy heads.

Supports:
  - Exp 1 & 2: Post-GRU role heads only
  - Exp 3 & 4: Pre-GRU residual routes + post-GRU role heads

All parameters train end-to-end (no frozen backbone).
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from .transformer import get_active_func
from .actor import ExplicitGRUCell, ScannedRNN


class RoleActorTrans(nn.Module):
    """Transformer-compatible stochastic actor with role-specific heads.

    The shared backbone (base MLP + GRU) processes observations for all roles.
    Per-role heads sit after the GRU. Optionally, per-role pre-GRU residual
    routes can shift the GRU input in role-specific directions.
    """

    action_dim: int
    config: Dict[str, Any]
    use_pre_gru_routes: bool = False
    n_roles: int = 6

    @nn.compact
    def __call__(
        self,
        rnn_states: jnp.ndarray,
        x: Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]],
        role_ids: jnp.ndarray,
    ):
        """Return the next hidden state and action distribution.

        Args:
            rnn_states: ``(batch, hidden)`` actor recurrent state.
            x: tuple ``(obs, resets, available_actions)`` where ``obs`` is
                ``(time, batch, obs_dim)`` and ``resets`` is ``(time, batch)``.
            role_ids: ``(time, batch)`` integer role IDs in ``[0, n_roles)``.
        """
        obs, resets, available_actions = x
        cfg = self.config
        hidden_sizes = cfg["hidden_sizes"]
        activation_name = cfg.get("activation_func", cfg["transformer"]["active_fn"])
        active_fn = get_active_func(activation_name)

        if cfg["use_feature_normalization"]:
            obs = nn.LayerNorm(epsilon=1e-5, name="feature_norm")(obs)

        # ---- Shared base MLP ------------------------------------------------
        embedding = obs
        for idx, hidden_size in enumerate(hidden_sizes):
            embedding = nn.Dense(
                hidden_size,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
                name=f"base_{idx}",
            )(embedding)
            embedding = active_fn(embedding)
            embedding = nn.LayerNorm(epsilon=1e-5, name=f"base_norm_{idx}")(embedding)

        # ---- Optional pre-GRU residual routes (Exp 3/4) ---------------------
        if self.use_pre_gru_routes:
            embedding = self._add_residual_routes(obs, embedding, role_ids)

        # ---- Shared GRU ------------------------------------------------------
        if cfg["use_naive_recurrent_policy"] or cfg["use_recurrent_policy"]:
            rnn_states, embedding = ScannedRNN(name="rnn")(
                rnn_states, (embedding, resets)
            )

        # ---- Role-specific post-GRU heads -----------------------------------
        all_logits = self._compute_all_role_logits(embedding)
        logits = self._gather_role_logits(all_logits, role_ids)

        if available_actions is not None:
            if available_actions.ndim == logits.ndim - 1:
                available_actions = available_actions[None, ...]
            logits = jnp.where(
                available_actions > 0.5,
                logits,
                jnp.full_like(logits, -1e10),
            )

        return rnn_states, distrax.Categorical(logits=logits)

    # -----------------------------------------------------------------------
    # Pre-GRU residual routes (Exp 3/4)
    # -----------------------------------------------------------------------

    def _add_residual_routes(
        self,
        obs: jnp.ndarray,
        shared_embedding: jnp.ndarray,
        role_ids: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute per-role routes on raw obs and add residually to embedding."""
        out_dim = shared_embedding.shape[-1]  # Match base MLP output dim
        all_routes = []
        for k in range(self.n_roles):
            route = nn.Dense(
                128,
                kernel_init=orthogonal(0.1),
                bias_init=constant(0.0),
                name=f"route_{k}_dense_0",
            )(obs)
            route = nn.relu(route)
            route = nn.Dense(
                out_dim,
                kernel_init=orthogonal(0.1),
                bias_init=constant(0.0),
                name=f"route_{k}_dense_1",
            )(route)
            route = nn.relu(route)
            all_routes.append(route)

        all_routes = jnp.stack(all_routes, axis=0)  # (n_roles, time, batch, out_dim)
        route = self._gather_by_role(all_routes, role_ids)  # (time, batch, out_dim)
        return shared_embedding + route

    # -----------------------------------------------------------------------
    # Post-GRU role heads
    # -----------------------------------------------------------------------

    def _compute_all_role_logits(self, embedding: jnp.ndarray) -> jnp.ndarray:
        """Return logits for every role: (n_roles, time, batch, action_dim)."""
        all_logits = []
        for k in range(self.n_roles):
            h = nn.Dense(
                64,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
                name=f"head_{k}_dense_0",
            )(embedding)
            h = nn.relu(h)
            h = nn.Dense(
                32,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
                name=f"head_{k}_dense_1",
            )(h)
            h = nn.relu(h)
            logits = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(self.config.get("gain", 0.01)),
                bias_init=constant(0.0),
                name=f"head_{k}_dense_2",
            )(h)
            all_logits.append(logits)
        return jnp.stack(all_logits, axis=0)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _gather_by_role(
        all_values: jnp.ndarray, role_ids: jnp.ndarray
    ) -> jnp.ndarray:
        """Gather per-role values based on role_ids.

        Supports two shapes:
          - ``(n_roles, time, batch, ...features)`` with ``role_ids=(time, batch)``
          - ``(n_roles, batch, ...features)`` with ``role_ids=(batch,)`` (no time dim)

        Returns:
            Gathered tensor with role dim removed.
        """
        if all_values.ndim == 3:
            # No time dimension: (n_roles, batch, feat), role_ids=(batch,)
            n_roles, batch, *feat = all_values.shape
            gathered = all_values[role_ids, jnp.arange(batch)]
            return gathered

        # With time dimension: (n_roles, time, batch, feat)
        n_roles, t, b, *feat = all_values.shape
        all_flat = all_values.reshape(n_roles, t * b, *feat)
        role_ids_flat = role_ids.reshape(t * b)
        gathered = all_flat[role_ids_flat, jnp.arange(t * b)]
        return gathered.reshape(t, b, *feat)

    def _gather_role_logits(
        self, all_logits: jnp.ndarray, role_ids: jnp.ndarray
    ) -> jnp.ndarray:
        """Select logits for the active roles."""
        return self._gather_by_role(all_logits, role_ids)

    # -----------------------------------------------------------------------
    # KL diversity penalty
    # -----------------------------------------------------------------------

    def compute_kl_diversity(
        self,
        params: Any,
        rnn_states: jnp.ndarray,
        obs: jnp.ndarray,
        resets: jnp.ndarray,
        avail: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        """Mean pairwise KL between all role policies on the given obs."""
        # Handle non-recurrent (2D obs) vs recurrent (3D obs)
        if obs.ndim == 2:
            obs = obs[None, :]          # (1, batch, obs_dim)
            resets = resets[None, :]    # (1, batch)
            if avail is not None:
                avail = avail[None, :]  # (1, batch, action_dim)

        all_logits = []
        for k in range(self.n_roles):
            role_ids_k = jnp.full_like(obs[:, :, 0], k, dtype=jnp.int32)
            _, pi = self.apply(params, rnn_states, (obs, resets, avail), role_ids_k)
            all_logits.append(pi.logits)
        all_logits = jnp.stack(all_logits, axis=0)  # (n_roles, time, batch, action_dim)

        kl_sum = 0.0
        count = 0
        for i in range(self.n_roles):
            for j in range(i + 1, self.n_roles):
                pi_i = distrax.Categorical(logits=all_logits[i])
                pi_j = distrax.Categorical(logits=all_logits[j])
                kl = pi_i.kl_divergence(pi_j)
                kl_sum += jnp.mean(kl)
                count += 1

        return kl_sum / (count + 1e-8)

    @staticmethod
    def make_kl_schedule(total_steps: int, initial_weight: float = 0.001) -> Any:
        """Cosine decay from ``initial_weight`` to 0 over first 30% of steps."""
        decay_end = int(total_steps * 0.3)

        def schedule(step: int) -> jnp.ndarray:
            t = jnp.minimum(step, decay_end)
            # Cosine decay: 0.5 * (1 + cos(pi * t / decay_end))
            decay = 0.5 * (1.0 + jnp.cos(jnp.pi * t / decay_end))
            return initial_weight * decay

        return schedule

    # -----------------------------------------------------------------------
    # Convenience helpers matching ActorTrans API
    # -----------------------------------------------------------------------

    def get_actions(
        self,
        rnn_states: jnp.ndarray,
        obs: jnp.ndarray,
        resets: jnp.ndarray,
        available_actions: Optional[jnp.ndarray],
        role_ids: jnp.ndarray,
        rng: jnp.ndarray,
        deterministic: bool = False,
    ):
        """MACA-style helper returning actions, log-probs, probs, and state."""
        new_states, pi = self(rnn_states, (obs, resets, available_actions), role_ids)
        actions = jnp.argmax(pi.logits, axis=-1) if deterministic else pi.sample(seed=rng)
        action_log_probs = pi.log_prob(actions)
        return actions, action_log_probs, pi.probs, new_states

    def evaluate_actions(
        self,
        rnn_states: jnp.ndarray,
        obs: jnp.ndarray,
        resets: jnp.ndarray,
        actions: jnp.ndarray,
        role_ids: jnp.ndarray,
        available_actions: Optional[jnp.ndarray] = None,
        active_masks: Optional[jnp.ndarray] = None,
    ):
        """Evaluate old actions for PPO, matching MACA ACTLayer semantics."""
        _, pi = self(rnn_states, (obs, resets, available_actions), role_ids)
        action_log_probs = pi.log_prob(actions)
        entropy = pi.entropy()
        if active_masks is not None:
            entropy = jnp.sum(entropy * active_masks) / (jnp.sum(active_masks) + 1e-8)
        else:
            entropy = jnp.mean(entropy)
        return action_log_probs, entropy, pi
