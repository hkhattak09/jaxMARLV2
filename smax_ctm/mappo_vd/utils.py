"""Utility functions for MAPPO-VD training.

Contains batchify/unbatchify, buffer implementations, and other utilities
ported from MACA framework.
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
        done, value, reward = transition.global_done, transition.value, transition.reward
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
    return advantages, advantages + traj_batch.value


def normalize_advantages(advantages: jnp.ndarray) -> jnp.ndarray:
    """Normalize advantages for PPO training."""
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def make_transition(
    global_done: jnp.ndarray,
    done: jnp.ndarray,
    action: jnp.ndarray,
    value: jnp.ndarray,
    reward: jnp.ndarray,
    log_prob: jnp.ndarray,
    obs: jnp.ndarray,
    world_state: jnp.ndarray,
    info: jnp.ndarray,
    avail_actions: jnp.ndarray,
) -> Dict:
    """Create a transition dict for the buffer."""
    return {
        "global_done": global_done,
        "done": done,
        "action": action,
        "value": value,
        "reward": reward,
        "log_prob": log_prob,
        "obs": obs,
        "world_state": world_state,
        "info": info,
        "avail_actions": avail_actions,
    }


class OnPolicyActorBuffer:
    """Buffer for storing actor training data."""
    
    def __init__(self, num_actors: int, obs_dim: int, action_dim: int, num_steps: int):
        self.num_actors = num_actors
        self.num_steps = num_steps
        self.obs = jnp.zeros((num_steps, num_actors, obs_dim))
        self.actions = jnp.zeros((num_steps, num_actors), dtype=jnp.int32)
        self.log_probs = jnp.zeros((num_steps, num_actors))
        self.avail_actions = jnp.ones((num_steps, num_actors, action_dim))
        self.masks = jnp.ones((num_steps, num_actors, 1))
        self.active_masks = jnp.ones((num_steps, num_actors, 1))
        
    def insert(
        self,
        step: int,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        log_probs: jnp.ndarray,
        masks: jnp.ndarray,
        active_masks: jnp.ndarray,
        avail_actions: jnp.ndarray,
    ):
        """Insert a transition at the given step."""
        self.obs = self.obs.at[step].set(obs)
        self.actions = self.actions.at[step].set(actions)
        self.log_probs = self.log_probs.at[step].set(log_probs)
        if avail_actions is not None:
            self.avail_actions = self.avail_actions.at[step].set(avail_actions)
        self.masks = self.masks.at[step].set(masks)
        self.active_masks = self.active_masks.at[step].set(active_masks)
    
    def reset(self):
        """Reset the buffer (clear all data)."""
        self.obs = jnp.zeros_like(self.obs)
        self.actions = jnp.zeros_like(self.actions)
        self.log_probs = jnp.zeros_like(self.log_probs)
        self.avail_actions = jnp.ones_like(self.avail_actions)
        self.masks = jnp.ones_like(self.masks)
        self.active_masks = jnp.ones_like(self.active_masks)


class OnPolicyCriticBufferEP:
    """Buffer for storing critic training data (Episodic, Parameter Sharing)."""
    
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        num_agents: int,
        share_obs_dim: int,
        obs_dim: int,
        action_dim: int,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        
        # State representations
        self.share_obs = jnp.zeros((num_steps, num_envs, share_obs_dim))
        self.obs = jnp.zeros((num_steps, num_envs, num_agents, obs_dim))
        self.actions = jnp.zeros((num_steps, num_envs, num_agents), dtype=jnp.int32)
        self.policy_probs = jnp.zeros((num_steps, num_envs, num_agents, action_dim))
        
        # Value predictions
        self.value_preds = jnp.zeros((num_steps + 1, num_envs, 1))
        self.q_value_preds = jnp.zeros((num_steps + 1, num_envs, num_agents, 1))
        self.eq_value_preds = jnp.zeros((num_steps + 1, num_envs, num_agents, 1))
        self.vq_value_preds = jnp.zeros((num_steps + 1, num_envs, num_agents, 1))
        self.vq_coma_value_preds = jnp.zeros((num_steps + 1, num_envs, num_agents, 1))
        
        # Baselines and attention
        self.bsln_weights = jnp.zeros((num_steps, num_envs, num_agents, 3))
        self.attn_weights = jnp.zeros((num_steps, num_envs, num_agents, num_agents))
        
        # Returns
        self.rewards = jnp.zeros((num_steps, num_envs, num_agents, 1))
        self.masks = jnp.ones((num_steps, num_envs, 1))
        self.bad_masks = jnp.ones((num_steps, num_envs, 1))
        self.returns = jnp.zeros_like(self.value_preds)
        self.q_returns = jnp.zeros_like(self.q_value_preds)
        self.eq_returns = jnp.zeros_like(self.eq_value_preds)
        self.vq_returns = jnp.zeros_like(self.vq_value_preds)
        self.vq_coma_returns = jnp.zeros_like(self.vq_coma_value_preds)
        
    def insert(
        self,
        step: int,
        share_obs: jnp.ndarray,
        rnn_states_critic: jnp.ndarray,
        values: jnp.ndarray,
        q_values: jnp.ndarray,
        eq_values: jnp.ndarray,
        vq_values: jnp.ndarray,
        vq_coma_values: jnp.ndarray,
        bsln_weights: jnp.ndarray,
        attn_weights: jnp.ndarray,
        rewards: jnp.ndarray,
        masks: jnp.ndarray,
        bad_masks: jnp.ndarray,
        obs: jnp.ndarray,
        rnn_states: jnp.ndarray,
        actions: jnp.ndarray,
        policy_probs: jnp.ndarray,
        active_masks: jnp.ndarray,
        avail_actions: jnp.ndarray,
    ):
        """Insert a transition at the given step."""
        self.share_obs = self.share_obs.at[step].set(share_obs)
        self.obs = self.obs.at[step].set(obs)
        self.actions = self.actions.at[step].set(actions)
        self.policy_probs = self.policy_probs.at[step].set(policy_probs)
        
        self.value_preds = self.value_preds.at[step].set(values)
        self.q_value_preds = self.q_value_preds.at[step].set(q_values)
        self.eq_value_preds = self.eq_value_preds.at[step].set(eq_values)
        self.vq_value_preds = self.vq_value_preds.at[step].set(vq_values)
        self.vq_coma_value_preds = self.vq_coma_value_preds.at[step].set(vq_coma_values)
        
        self.bsln_weights = self.bsln_weights.at[step].set(bsln_weights)
        self.attn_weights = self.attn_weights.at[step].set(attn_weights)
        
        self.rewards = self.rewards.at[step].set(rewards)
        self.masks = self.masks.at[step].set(masks)
        self.bad_masks = self.bad_masks.at[step].set(bad_masks)
    
    def compute_returns(
        self,
        next_value: jnp.ndarray,
        value_normalizer: Any = None,
    ):
        """Compute returns using GAE."""
        del value_normalizer
        self.value_preds = self.value_preds.at[-1].set(next_value)
        rewards = jnp.mean(self.rewards, axis=2)

        def scan_step(carry, step_data):
            gae, next_pred = carry
            reward, mask, pred = step_data
            delta = reward + 0.99 * next_pred * mask - pred
            gae = delta + 0.99 * 0.95 * mask * gae
            return (gae, pred), gae + pred

        _, returns = jax.lax.scan(
            scan_step,
            (jnp.zeros_like(next_value), next_value),
            (rewards, self.masks, self.value_preds[:-1]),
            reverse=True,
        )
        self.returns = self.returns.at[:-1].set(returns)
        self.returns = self.returns.at[-1].set(next_value)
