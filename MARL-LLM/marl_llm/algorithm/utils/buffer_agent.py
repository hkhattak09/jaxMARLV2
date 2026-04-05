"""
Multi-Agent Replay Buffer Module — Centralised Critic Version

Stores joint experience rows: one row per timestep containing ALL agents'
observations and actions concatenated. This is the correct structure for
a centralised critic that receives the full joint state during training.

Buffer shapes (max_steps rows):
  obs_buffs:      (max_steps, n_agents * obs_dim)
  ac_buffs:       (max_steps, n_agents * action_dim)
  ac_prior_buffs: (max_steps, n_agents * action_dim)
  rew_buffs:      (max_steps, 1)   — mean reward across agents
  next_obs_buffs: (max_steps, n_agents * obs_dim)
  done_buffs:     (max_steps, 1)   — max done flag across agents
"""

import numpy as np
from torch import Tensor
import torch


class ReplayBufferAgent(object):
    """
    Replay Buffer for centralised-critic MADDPG.

    Stores one joint row per environment timestep rather than per-agent rows.
    Each row concatenates all agents' observations / actions so the centralised
    critic can consume them directly from sampled batches.
    """

    def __init__(self, max_steps, num_agents, state_dim, action_dim,
                 start_stop_index=None):
        """
        Args:
            max_steps (int):   Maximum number of timesteps to store.
            num_agents (int):  Number of agents (used for joint row width).
            state_dim (int):   Per-agent observation dimension.
            action_dim (int):  Per-agent action dimension.
            start_stop_index:  Kept for API compatibility, not used.
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.total_length = max_steps  # one joint row per timestep

        self.obs_buffs      = np.zeros((max_steps, num_agents * state_dim),  dtype=np.float32)
        self.ac_buffs       = np.zeros((max_steps, num_agents * action_dim), dtype=np.float32)
        self.ac_prior_buffs = np.zeros((max_steps, num_agents * action_dim), dtype=np.float32)
        self.log_pi_buffs   = np.zeros((max_steps, 1),                       dtype=np.float32)
        self.rew_buffs      = np.zeros((max_steps, 1),                       dtype=np.float32)
        self.next_obs_buffs = np.zeros((max_steps, num_agents * state_dim),  dtype=np.float32)
        self.done_buffs     = np.zeros((max_steps, 1),                       dtype=np.float32)

        self.filled_i = 0  # total filled (capped at max_steps)
        self.curr_i   = 0  # current write index

    def __len__(self):
        return self.filled_i

    def push(self, observations_orig, actions_orig, rewards_orig,
             next_observations_orig, dones_orig, index=None,
             actions_prior_orig=None, log_pi_orig=None):
        """
        Add one or more joint-row transitions to the buffer.

        Args:
            observations_orig:      (obs_dim,  N*n_a)  numpy array
            actions_orig:           (act_dim,  N*n_a)
            rewards_orig:           (1,        N*n_a)
            next_observations_orig: (obs_dim,  N*n_a)
            dones_orig:             (1,        N*n_a)
            index:                  ignored (kept for API compatibility)
            actions_prior_orig:     (act_dim,  N*n_a)  optional
            log_pi_orig:            ignored
        """
        n_a = self.num_agents
        # Number of parallel envs in this batch (N=1 in standard setup)
        data_length = observations_orig.shape[1] // n_a

        obs_dim = observations_orig.shape[0]
        act_dim = actions_orig.shape[0]

        # (obs_dim, N*n_a).T → (N*n_a, obs_dim) → (N, n_a*obs_dim)
        obs_joint      = observations_orig.T.reshape(data_length, n_a * obs_dim)
        acs_joint      = actions_orig.T.reshape(data_length, n_a * act_dim)
        next_obs_joint = next_observations_orig.T.reshape(data_length, n_a * obs_dim)

        # (1, N*n_a) → (N, n_a) → mean/max → (N, 1)
        rews_joint  = rewards_orig.reshape(data_length, n_a).mean(axis=1, keepdims=True)
        dones_joint = dones_orig.reshape(data_length, n_a).max(axis=1, keepdims=True)

        if actions_prior_orig is not None:
            prior_joint = actions_prior_orig.T.reshape(data_length, n_a * act_dim)

        # Handle circular buffer wraparound
        if self.curr_i + data_length > self.total_length:
            rollover = data_length - (self.total_length - self.curr_i)
            self.curr_i -= rollover

        sl = slice(self.curr_i, self.curr_i + data_length)
        self.obs_buffs[sl]      = obs_joint
        self.ac_buffs[sl]       = acs_joint
        self.rew_buffs[sl]      = rews_joint
        self.next_obs_buffs[sl] = next_obs_joint
        self.done_buffs[sl]     = dones_joint

        if actions_prior_orig is not None:
            self.ac_prior_buffs[sl] = prior_joint

        self.curr_i += data_length
        if self.filled_i < self.total_length:
            self.filled_i += data_length
        if self.curr_i == self.total_length:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, is_prior=False, is_log_pi=False):
        """
        Sample N joint-row transitions.

        Returns:
            obs:       (N, n_agents * obs_dim)
            acs:       (N, n_agents * act_dim)
            rews:      (N, 1)
            next_obs:  (N, n_agents * obs_dim)
            dones:     (N, 1)
            prior_acs: (N, n_agents * act_dim) or None
            log_pis:   None (not stored)
        """
        inds = np.random.choice(self.filled_i, size=N, replace=False)

        cast = (lambda x: Tensor(x).requires_grad_(False).cuda()) if to_gpu \
               else (lambda x: Tensor(x).requires_grad_(False))

        prior = cast(self.ac_prior_buffs[inds]) if is_prior else None

        return (
            cast(self.obs_buffs[inds]),
            cast(self.ac_buffs[inds]),
            cast(self.rew_buffs[inds]),
            cast(self.next_obs_buffs[inds]),
            cast(self.done_buffs[inds]),
            prior,
            None,
        )

    def get_average_rewards(self, N):
        """Average reward over the last N stored timesteps."""
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[inds].mean()]
