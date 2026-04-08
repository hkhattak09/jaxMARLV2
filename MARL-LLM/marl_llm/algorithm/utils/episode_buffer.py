"""
Episode-Sequence Replay Buffer for Stateful Actor + Recurrent Critic Training

Stores complete episodes and samples contiguous sequences for temporal learning.
Each sequence is split into burn-in (no gradient, reconstructs hidden states) and
training (with gradient) portions by the consumer — the buffer just returns
contiguous chunks.

Storage format (per episode):
  obs:      (episode_length, n_agents * obs_dim)
  acs:      (episode_length, n_agents * action_dim)
  acs_prior:(episode_length, n_agents * action_dim)
  rews:     (episode_length, 1)
  next_obs: (episode_length, n_agents * obs_dim)
  dones:    (episode_length, 1)

Sample format (time-first for sequential processing):
  obs:      (sequence_length, num_sequences, n_agents * obs_dim)
  acs:      (sequence_length, num_sequences, n_agents * action_dim)
  rews:     (sequence_length, num_sequences, 1)
  next_obs: (sequence_length, num_sequences, n_agents * obs_dim)
  dones:    (sequence_length, num_sequences, 1)
  acs_prior:(sequence_length, num_sequences, n_agents * action_dim)
"""

import numpy as np
import torch


class EpisodeSequenceBuffer:
    """
    Episode-sequence replay buffer for stateful actor + recurrent critic.

    Stores complete episodes in pre-allocated arrays and samples contiguous
    sequences of configurable length. Designed for R2D2-style burn-in training
    where the consumer splits sequences into burn-in and training portions.
    """

    def __init__(self, max_episodes, episode_length, num_agents, obs_dim, action_dim,
                 sequence_length=32):
        """
        Args:
            max_episodes:    Maximum number of complete episodes to store.
            episode_length:  Steps per episode (fixed, e.g. 200).
            num_agents:      Number of agents (for joint row width).
            obs_dim:         Per-agent observation dimension.
            action_dim:      Per-agent action dimension.
            sequence_length: Length of contiguous sequences to sample.
        """
        self.max_episodes = max_episodes
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length

        if sequence_length > episode_length:
            raise ValueError(
                f"sequence_length ({sequence_length}) must be <= episode_length ({episode_length})"
            )

        joint_obs_dim = num_agents * obs_dim
        joint_act_dim = num_agents * action_dim

        # Pre-allocate storage for all episodes
        self.obs_buffs      = np.zeros((max_episodes, episode_length, joint_obs_dim), dtype=np.float32)
        self.acs_buffs      = np.zeros((max_episodes, episode_length, joint_act_dim), dtype=np.float32)
        self.acs_prior_buffs = np.zeros((max_episodes, episode_length, joint_act_dim), dtype=np.float32)
        self.rews_buffs     = np.zeros((max_episodes, episode_length, 1), dtype=np.float32)
        self.next_obs_buffs = np.zeros((max_episodes, episode_length, joint_obs_dim), dtype=np.float32)
        self.dones_buffs    = np.zeros((max_episodes, episode_length, 1), dtype=np.float32)

        self.filled_i = 0   # number of episodes stored (capped at max_episodes)
        self.curr_i = 0     # current write index (circular)

    def __len__(self):
        """Number of complete episodes stored."""
        return self.filled_i

    def push_episode(self, obs_episode, acs_episode, rews_episode,
                     next_obs_episode, dones_episode, acs_prior_episode=None):
        """
        Store one complete episode from the training loop's bulk transfer.

        Accepts column-major format (as produced by torch.stack(...).cpu().numpy()
        in the training loop) and converts to joint rows internally.

        Args:
            obs_episode:      (T, obs_dim, N*n_a) — column-major observations
            acs_episode:      (T, act_dim, N*n_a) — column-major actions
            rews_episode:     (T, N*n_a)          — per-agent rewards
            next_obs_episode: (T, obs_dim, N*n_a) — column-major next observations
            dones_episode:    (T, N*n_a)          — per-agent done flags
            acs_prior_episode:(T, act_dim, N*n_a) — column-major prior actions, or None
        """
        T = obs_episode.shape[0]
        n_a = self.num_agents
        obs_dim = self.obs_dim
        act_dim = self.action_dim

        # Detect number of parallel envs
        total_agents = obs_episode.shape[2]
        n_envs = total_agents // n_a

        for env_i in range(n_envs):
            agent_slice = slice(env_i * n_a, (env_i + 1) * n_a)

            # (T, obs_dim, n_a) → (T, n_a, obs_dim) → (T, n_a*obs_dim)
            obs_joint = obs_episode[:, :, agent_slice].transpose(0, 2, 1).reshape(T, n_a * obs_dim)
            next_obs_joint = next_obs_episode[:, :, agent_slice].transpose(0, 2, 1).reshape(T, n_a * obs_dim)

            # (T, act_dim, n_a) → (T, n_a, act_dim) → (T, n_a*act_dim)
            acs_joint = acs_episode[:, :, agent_slice].transpose(0, 2, 1).reshape(T, n_a * act_dim)

            # (T, n_a) → mean → (T, 1)
            rews_joint = rews_episode[:, agent_slice].mean(axis=1, keepdims=True)
            dones_joint = dones_episode[:, agent_slice].max(axis=1, keepdims=True)

            idx = self.curr_i
            self.obs_buffs[idx, :T] = obs_joint
            self.acs_buffs[idx, :T] = acs_joint
            self.rews_buffs[idx, :T] = rews_joint
            self.next_obs_buffs[idx, :T] = next_obs_joint
            self.dones_buffs[idx, :T] = dones_joint

            if acs_prior_episode is not None:
                prior_joint = acs_prior_episode[:, :, agent_slice].transpose(0, 2, 1).reshape(T, n_a * act_dim)
                self.acs_prior_buffs[idx, :T] = prior_joint

            self.curr_i = (self.curr_i + 1) % self.max_episodes
            self.filled_i = min(self.filled_i + 1, self.max_episodes)

    def sample(self, num_sequences, to_gpu=False):
        """
        Sample contiguous sequences from stored episodes.

        Returns time-first tensors for sequential processing by LSTM/CTM.
        The consumer (maddpg.py) splits into burn-in and training portions.

        Args:
            num_sequences: Number of sequences to sample.
            to_gpu: If True, return CUDA tensors; else CPU tensors.

        Returns:
            obs:       (sequence_length, num_sequences, n_agents*obs_dim)
            acs:       (sequence_length, num_sequences, n_agents*act_dim)
            rews:      (sequence_length, num_sequences, 1)
            next_obs:  (sequence_length, num_sequences, n_agents*obs_dim)
            dones:     (sequence_length, num_sequences, 1)
            acs_prior: (sequence_length, num_sequences, n_agents*act_dim)
        """
        if self.filled_i == 0:
            raise RuntimeError("Cannot sample from empty buffer")

        # Random episode indices (with replacement — some episodes may repeat)
        ep_indices = np.random.randint(0, self.filled_i, size=num_sequences)

        # Random start positions within each episode
        max_start = self.episode_length - self.sequence_length
        start_indices = np.random.randint(0, max_start + 1, size=num_sequences)

        # Gather sequences: build index arrays for efficient numpy indexing
        # Result shape: (num_sequences, sequence_length, feature_dim)
        seq_len = self.sequence_length
        obs_seqs = np.stack([
            self.obs_buffs[ep, start:start + seq_len]
            for ep, start in zip(ep_indices, start_indices)
        ])
        acs_seqs = np.stack([
            self.acs_buffs[ep, start:start + seq_len]
            for ep, start in zip(ep_indices, start_indices)
        ])
        rews_seqs = np.stack([
            self.rews_buffs[ep, start:start + seq_len]
            for ep, start in zip(ep_indices, start_indices)
        ])
        next_obs_seqs = np.stack([
            self.next_obs_buffs[ep, start:start + seq_len]
            for ep, start in zip(ep_indices, start_indices)
        ])
        dones_seqs = np.stack([
            self.dones_buffs[ep, start:start + seq_len]
            for ep, start in zip(ep_indices, start_indices)
        ])
        prior_seqs = np.stack([
            self.acs_prior_buffs[ep, start:start + seq_len]
            for ep, start in zip(ep_indices, start_indices)
        ])

        # Transpose to time-first: (num_sequences, seq_len, feat) → (seq_len, num_sequences, feat)
        obs_seqs = obs_seqs.transpose(1, 0, 2)
        acs_seqs = acs_seqs.transpose(1, 0, 2)
        rews_seqs = rews_seqs.transpose(1, 0, 2)
        next_obs_seqs = next_obs_seqs.transpose(1, 0, 2)
        dones_seqs = dones_seqs.transpose(1, 0, 2)
        prior_seqs = prior_seqs.transpose(1, 0, 2)

        if to_gpu:
            cast = lambda x: torch.tensor(x).cuda()
        else:
            cast = lambda x: torch.tensor(x)

        return (
            cast(obs_seqs),
            cast(acs_seqs),
            cast(rews_seqs),
            cast(next_obs_seqs),
            cast(dones_seqs),
            cast(prior_seqs),
        )

    def get_average_rewards(self, N):
        """Average reward over the last N stored episodes."""
        if self.filled_i == 0:
            return [0.0]
        n = min(N, self.filled_i)
        if self.filled_i == self.max_episodes:
            # Buffer is full, curr_i points to oldest
            indices = [(self.curr_i - 1 - i) % self.max_episodes for i in range(n)]
        else:
            indices = list(range(max(0, self.filled_i - n), self.filled_i))
        return [np.mean([self.rews_buffs[i].mean() for i in indices])]
