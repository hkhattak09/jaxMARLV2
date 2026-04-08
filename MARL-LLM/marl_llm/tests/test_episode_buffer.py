"""
Tests for EpisodeSequenceBuffer.

Tests cover:
  1. Construction and memory layout
  2. push_episode — column-major input, joint-row storage, circular overwrite
  3. sample — shape correctness, time-first format, contiguity within sequences
  4. Multi-env push (N > 1 rollout threads)
  5. Edge cases — full buffer wraparound, sequence_length == episode_length
  6. get_average_rewards

Run with:
    cd MARL-LLM/marl_llm
    python -m pytest tests/test_episode_buffer.py -v
or:
    python tests/test_episode_buffer.py
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_MARL_LLM = _THIS_DIR.parent
for p in [str(_MARL_LLM)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import pytest

from algorithm.utils.episode_buffer import EpisodeSequenceBuffer


# ── Test fixtures ──────────────────────────────────────────────────────────
NUM_AGENTS = 24
OBS_DIM = 192
ACT_DIM = 2
EPISODE_LENGTH = 200
SEQUENCE_LENGTH = 32
MAX_EPISODES = 10  # small for tests


def make_episode_data(episode_length=EPISODE_LENGTH, n_envs=1, value_offset=0.0):
    """Create fake episode data in column-major format (as training loop produces).

    obs:      (T, obs_dim, n_envs * n_a)
    acs:      (T, act_dim, n_envs * n_a)
    rews:     (T, n_envs * n_a)
    next_obs: (T, obs_dim, n_envs * n_a)
    dones:    (T, n_envs * n_a)
    prior:    (T, act_dim, n_envs * n_a)
    """
    total_agents = n_envs * NUM_AGENTS
    rng = np.random.RandomState(42)
    obs = rng.randn(episode_length, OBS_DIM, total_agents).astype(np.float32) + value_offset
    acs = rng.randn(episode_length, ACT_DIM, total_agents).astype(np.float32) + value_offset
    rews = rng.randn(episode_length, total_agents).astype(np.float32)
    next_obs = rng.randn(episode_length, OBS_DIM, total_agents).astype(np.float32) + value_offset
    dones = np.zeros((episode_length, total_agents), dtype=np.float32)
    dones[-1, :] = 1.0  # episode done at last step
    prior = rng.randn(episode_length, ACT_DIM, total_agents).astype(np.float32)
    return obs, acs, rews, next_obs, dones, prior


# ── 1. Construction ────────────────────────────────────────────────────────
class TestConstruction:
    def test_shapes(self):
        buf = EpisodeSequenceBuffer(MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, SEQUENCE_LENGTH)
        assert buf.obs_buffs.shape == (MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS * OBS_DIM)
        assert buf.acs_buffs.shape == (MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS * ACT_DIM)
        assert buf.rews_buffs.shape == (MAX_EPISODES, EPISODE_LENGTH, 1)
        assert len(buf) == 0

    def test_sequence_length_validation(self):
        with pytest.raises(ValueError):
            EpisodeSequenceBuffer(MAX_EPISODES, 10, NUM_AGENTS, OBS_DIM, ACT_DIM,
                                   sequence_length=20)


# ── 2. Push ────────────────────────────────────────────────────────────────
class TestPush:
    def test_single_episode_push(self):
        buf = EpisodeSequenceBuffer(MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, SEQUENCE_LENGTH)
        obs, acs, rews, next_obs, dones, prior = make_episode_data()
        buf.push_episode(obs, acs, rews, next_obs, dones, prior)
        assert len(buf) == 1
        assert buf.curr_i == 1

    def test_joint_row_conversion(self):
        """Verify column-major → joint-row conversion is correct."""
        buf = EpisodeSequenceBuffer(MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, SEQUENCE_LENGTH)
        obs, acs, rews, next_obs, dones, prior = make_episode_data()
        buf.push_episode(obs, acs, rews, next_obs, dones, prior)

        # Check first timestep: obs[:, :, 0] should be the first agent's obs
        # After conversion: buf.obs_buffs[0, 0, :OBS_DIM] should be agent 0's obs at t=0
        # In column-major: obs[0, :, 0] is agent 0's obs at t=0
        # After transpose: obs[0, :, 0:NUM_AGENTS].T → (NUM_AGENTS, OBS_DIM)
        # Then reshape → (NUM_AGENTS * OBS_DIM,)
        # So buf[0, 0, 0:OBS_DIM] should equal obs[0, :, 0]
        expected_agent0_obs = obs[0, :, 0]
        stored_agent0_obs = buf.obs_buffs[0, 0, :OBS_DIM]
        np.testing.assert_allclose(stored_agent0_obs, expected_agent0_obs, atol=1e-6)

    def test_rewards_are_mean_aggregated(self):
        buf = EpisodeSequenceBuffer(MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, SEQUENCE_LENGTH)
        obs, acs, rews, next_obs, dones, prior = make_episode_data()
        buf.push_episode(obs, acs, rews, next_obs, dones, prior)

        expected_mean_rew_t0 = rews[0, :NUM_AGENTS].mean()
        np.testing.assert_allclose(buf.rews_buffs[0, 0, 0], expected_mean_rew_t0, atol=1e-6)

    def test_dones_are_max_aggregated(self):
        buf = EpisodeSequenceBuffer(MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, SEQUENCE_LENGTH)
        obs, acs, rews, next_obs, dones, prior = make_episode_data()
        buf.push_episode(obs, acs, rews, next_obs, dones, prior)

        # Last timestep should have done=1 (max of all agents' done flags)
        assert buf.dones_buffs[0, -1, 0] == 1.0
        # First timestep should have done=0
        assert buf.dones_buffs[0, 0, 0] == 0.0

    def test_circular_overwrite(self):
        buf = EpisodeSequenceBuffer(3, EPISODE_LENGTH, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, SEQUENCE_LENGTH)
        for i in range(5):
            obs, acs, rews, next_obs, dones, prior = make_episode_data(value_offset=float(i))
            buf.push_episode(obs, acs, rews, next_obs, dones, prior)

        assert len(buf) == 3  # capped at max_episodes
        assert buf.curr_i == 2  # 5 % 3 = 2

    def test_multi_env_push(self):
        """N=2 rollout threads → 2 episodes pushed per call."""
        buf = EpisodeSequenceBuffer(MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, SEQUENCE_LENGTH)
        obs, acs, rews, next_obs, dones, prior = make_episode_data(n_envs=2)
        buf.push_episode(obs, acs, rews, next_obs, dones, prior)
        assert len(buf) == 2  # two episodes stored


# ── 3. Sample ──────────────────────────────────────────────────────────────
class TestSample:
    def _filled_buffer(self, n_episodes=5):
        buf = EpisodeSequenceBuffer(MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, SEQUENCE_LENGTH)
        for i in range(n_episodes):
            data = make_episode_data(value_offset=float(i))
            buf.push_episode(*data)
        return buf

    def test_sample_shapes(self):
        buf = self._filled_buffer()
        num_seq = 4
        obs, acs, rews, next_obs, dones, prior = buf.sample(num_seq)

        assert obs.shape == (SEQUENCE_LENGTH, num_seq, NUM_AGENTS * OBS_DIM)
        assert acs.shape == (SEQUENCE_LENGTH, num_seq, NUM_AGENTS * ACT_DIM)
        assert rews.shape == (SEQUENCE_LENGTH, num_seq, 1)
        assert next_obs.shape == (SEQUENCE_LENGTH, num_seq, NUM_AGENTS * OBS_DIM)
        assert dones.shape == (SEQUENCE_LENGTH, num_seq, 1)
        assert prior.shape == (SEQUENCE_LENGTH, num_seq, NUM_AGENTS * ACT_DIM)

    def test_sample_returns_tensors(self):
        buf = self._filled_buffer()
        obs, acs, rews, next_obs, dones, prior = buf.sample(2)
        assert isinstance(obs, torch.Tensor)
        assert obs.dtype == torch.float32

    def test_sample_contiguity(self):
        """Verify sampled sequences are contiguous within the stored episode."""
        buf = EpisodeSequenceBuffer(MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, sequence_length=4)
        # Use a deterministic pattern: obs at timestep t has value t in first feature
        obs_data = np.zeros((EPISODE_LENGTH, OBS_DIM, NUM_AGENTS), dtype=np.float32)
        for t in range(EPISODE_LENGTH):
            obs_data[t, 0, :] = float(t)  # first feature = timestep index

        acs = np.zeros((EPISODE_LENGTH, ACT_DIM, NUM_AGENTS), dtype=np.float32)
        rews = np.zeros((EPISODE_LENGTH, NUM_AGENTS), dtype=np.float32)
        dones = np.zeros((EPISODE_LENGTH, NUM_AGENTS), dtype=np.float32)

        buf.push_episode(obs_data, acs, rews, obs_data, dones)
        assert len(buf) == 1

        # Sample and check contiguity: consecutive timesteps should differ by 1
        np.random.seed(123)
        obs_seq, _, _, _, _, _ = buf.sample(1)
        # obs_seq: (4, 1, 24*192) → first agent's first feature across time
        first_feat = obs_seq[:, 0, 0].numpy()
        diffs = np.diff(first_feat)
        np.testing.assert_allclose(diffs, 1.0, atol=1e-5,
                                   err_msg="Sequence is not contiguous")

    def test_empty_buffer_raises(self):
        buf = EpisodeSequenceBuffer(MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, SEQUENCE_LENGTH)
        with pytest.raises(RuntimeError, match="empty buffer"):
            buf.sample(1)

    def test_sample_sequence_length_equals_episode_length(self):
        """Edge case: sequence_length == episode_length → only one possible start."""
        buf = EpisodeSequenceBuffer(MAX_EPISODES, 50, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, sequence_length=50)
        obs, acs, rews, next_obs, dones, prior = make_episode_data(episode_length=50)
        buf.push_episode(obs, acs, rews, next_obs, dones, prior)
        result = buf.sample(2)
        assert result[0].shape == (50, 2, NUM_AGENTS * OBS_DIM)


# ── 4. get_average_rewards ──────────────────────────────────────────────────
class TestAverageRewards:
    def test_empty_buffer(self):
        buf = EpisodeSequenceBuffer(MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, SEQUENCE_LENGTH)
        assert buf.get_average_rewards(5) == [0.0]

    def test_with_data(self):
        buf = EpisodeSequenceBuffer(MAX_EPISODES, EPISODE_LENGTH, NUM_AGENTS,
                                     OBS_DIM, ACT_DIM, SEQUENCE_LENGTH)
        obs, acs, rews, next_obs, dones, prior = make_episode_data()
        buf.push_episode(obs, acs, rews, next_obs, dones, prior)
        result = buf.get_average_rewards(1)
        assert len(result) == 1
        print(f"\n[DIAG test_with_data] result={result}, type={type(result[0])}, val={result[0]}")
        assert isinstance(result[0], (float, np.floating)), \
            f"Expected float, got {type(result[0])}: {result[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
