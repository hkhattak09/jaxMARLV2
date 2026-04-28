"""MAPPO-T actor network.

This is the JAX/Flax counterpart of MACA's StochasticPolicyTrans.  The public
``__call__`` follows the existing JAX training scripts in this repository:
it consumes a recurrent state plus a sequence tuple and returns an updated
state and a Distrax categorical policy.
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


class ScannedRNN(nn.Module):
    """GRU layer scanned over the leading time axis.

    ``resets`` follows the convention used by the repo's JAX MAPPO scripts:
    True/1 means the hidden state should be reset before this step.
    """

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, None],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        y = nn.LayerNorm(epsilon=1e-5, name="rnn_norm")(y)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorTrans(nn.Module):
    """Transformer-compatible stochastic actor for discrete SMAX actions."""

    action_dim: int
    config: Dict[str, Any]

    @nn.compact
    def __call__(
        self,
        rnn_states: jnp.ndarray,
        x: Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]],
    ):
        """Return the next hidden state and action distribution.

        Args:
            rnn_states: ``(batch, hidden)`` actor recurrent state.
            x: tuple ``(obs, resets, available_actions)`` where ``obs`` is
                ``(time, batch, obs_dim)`` and ``resets`` is ``(time, batch)``.
        """
        obs, resets, available_actions = x
        cfg = self.config
        hidden_sizes = cfg["hidden_sizes"]
        activation_name = cfg.get("activation_func", cfg["transformer"]["active_fn"])
        active_fn = get_active_func(activation_name)

        if cfg["use_feature_normalization"]:
            obs = nn.LayerNorm(epsilon=1e-5, name="feature_norm")(obs)

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

        if cfg["use_naive_recurrent_policy"] or cfg["use_recurrent_policy"]:
            rnn_states, embedding = ScannedRNN(name="rnn")(rnn_states, (embedding, resets))

        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(cfg.get("gain", 0.01)),
            bias_init=constant(0.0),
            name="action_out",
        )(embedding)

        if available_actions is not None:
            if available_actions.ndim == logits.ndim - 1:
                available_actions = available_actions[None, ...]
            logits = logits - ((1.0 - available_actions) * 1e10)

        return rnn_states, distrax.Categorical(logits=logits)

    def get_actions(
        self,
        rnn_states: jnp.ndarray,
        obs: jnp.ndarray,
        resets: jnp.ndarray,
        available_actions: Optional[jnp.ndarray],
        rng: jnp.ndarray,
        deterministic: bool = False,
    ):
        """MACA-style helper returning actions, log-probs, probs, and state."""
        new_states, pi = self(rnn_states, (obs, resets, available_actions))
        actions = jnp.argmax(pi.logits, axis=-1) if deterministic else pi.sample(seed=rng)
        action_log_probs = pi.log_prob(actions)
        return actions, action_log_probs, pi.probs, new_states

    def evaluate_actions(
        self,
        rnn_states: jnp.ndarray,
        obs: jnp.ndarray,
        resets: jnp.ndarray,
        actions: jnp.ndarray,
        available_actions: Optional[jnp.ndarray] = None,
        active_masks: Optional[jnp.ndarray] = None,
    ):
        """Evaluate old actions for PPO, matching MACA ACTLayer semantics."""
        _, pi = self(rnn_states, (obs, resets, available_actions))
        action_log_probs = pi.log_prob(actions)
        entropy = pi.entropy()
        if active_masks is not None:
            entropy = jnp.sum(entropy * active_masks) / (jnp.sum(active_masks) + 1e-8)
        else:
            entropy = jnp.mean(entropy)
        return action_log_probs, entropy, pi
