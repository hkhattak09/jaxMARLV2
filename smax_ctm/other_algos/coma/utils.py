"""Utility functions for COMA training.

Contains batchify/unbatchify and other helpers ported from MAPPO-T utils.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple


def batchify(x: Dict[str, jnp.ndarray], agent_list: list, num_actors: int) -> jnp.ndarray:
    """Convert per-agent observation dict to batched array.

    Args:
        x: Dict mapping agent names to arrays of shape (num_envs, ...)
        agent_list: List of agent names in order
        num_actors: Total number of actors (num_envs * num_agents)

    Returns:
        Batched array of shape (num_actors, ...)
    """
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list: list, num_envs: int, num_actors: int) -> Dict[str, jnp.ndarray]:
    """Convert batched array back to per-agent dict.

    Args:
        x: Batched array of shape (num_actors, ...)
        agent_list: List of agent names in order
        num_envs: Number of parallel environments
        num_actors: Total number of actors

    Returns:
        Dict mapping agent names to arrays of shape (num_envs, ...)
    """
    x = x.reshape((len(agent_list), num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def compute_gae(
    traj_batch: Any,
    last_val: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation.

    Args:
        traj_batch: Transition batch with value, reward, done fields
        last_val: Value estimate for the last state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        (advantages, targets) tuple
    """
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = transition.global_done, transition.v_value, transition.reward
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.v_value


def normalize_advantages(advantages: jnp.ndarray) -> jnp.ndarray:
    """Normalize advantages for PPO training."""
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
