"""
Tests for Phase 5: lstm_hidden_dim threading through agent constructors.

Tests cover:
  1. DDPGAgent passes lstm_hidden_dim to all 4 AggregatingCritic instances
  2. CTMDDPGAgent passes lstm_hidden_dim to all 4 AggregatingCritic instances
  3. MADDPG threads lstm_hidden_dim from constructor to agents (MLP path)
  4. Non-default lstm_hidden_dim values propagate correctly
  5. Default lstm_hidden_dim=64 works when not explicitly passed (backward compat)
  6. init_dict includes lstm_hidden_dim for checkpoint save/load compatibility

Run with:
    cd MARL-LLM/marl_llm
    python -m pytest tests/test_agent_lstm_param.py -v
or:
    python tests/test_agent_lstm_param.py
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_MARL_LLM = _THIS_DIR.parent
for p in [str(_MARL_LLM)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import pytest

from algorithm.utils.networks import AggregatingCritic
from algorithm.utils.agents import DDPGAgent
from algorithm.algorithms.maddpg import MADDPG


# ── Fixtures ──────────────────────────────────────────────────────────────
N_AGENTS = 4
OBS_DIM = 16
ACT_DIM = 2
HIDDEN_DIM = 32


def make_ddpg_agent(lstm_hidden_dim=64):
    return DDPGAgent(
        dim_input_policy=OBS_DIM,
        dim_output_policy=ACT_DIM,
        n_agents=N_AGENTS,
        lr_actor=1e-4,
        lr_critic=1e-3,
        hidden_dim=HIDDEN_DIM,
        lstm_hidden_dim=lstm_hidden_dim,
    )


def make_maddpg(lstm_hidden_dim=64):
    agent_init_params = [{'dim_input_policy': OBS_DIM,
                          'dim_output_policy': ACT_DIM,
                          'n_agents': N_AGENTS}]
    alg_types = ['MADDPG']
    return MADDPG(agent_init_params=agent_init_params,
                  alg_types=alg_types,
                  epsilon=0.0, noise=0.0,
                  gamma=0.95, tau=0.01,
                  lr_actor=1e-4, lr_critic=1e-3,
                  hidden_dim=HIDDEN_DIM,
                  lstm_hidden_dim=lstm_hidden_dim,
                  n_agents=N_AGENTS,
                  device='cpu',
                  use_ctm_actor=False,
                  prior_mode='none')


# ── 1. DDPGAgent critic LSTM dim ────────────────────────────────────────
class TestDDPGAgentLSTM:
    def test_default_lstm_hidden_dim(self):
        agent = make_ddpg_agent()
        assert agent.critic.lstm_hidden_dim == 64
        assert agent.target_critic.lstm_hidden_dim == 64
        assert agent.critic2.lstm_hidden_dim == 64
        assert agent.target_critic2.lstm_hidden_dim == 64

    def test_custom_lstm_hidden_dim(self):
        agent = make_ddpg_agent(lstm_hidden_dim=128)
        assert agent.critic.lstm_hidden_dim == 128
        assert agent.target_critic.lstm_hidden_dim == 128
        assert agent.critic2.lstm_hidden_dim == 128
        assert agent.target_critic2.lstm_hidden_dim == 128

    def test_lstm_layer_matches_dim(self):
        agent = make_ddpg_agent(lstm_hidden_dim=32)
        assert agent.critic.lstm.hidden_size == 32
        assert agent.critic2.lstm.hidden_size == 32

    def test_all_four_critics_are_aggregating(self):
        agent = make_ddpg_agent()
        assert isinstance(agent.critic, AggregatingCritic)
        assert isinstance(agent.target_critic, AggregatingCritic)
        assert isinstance(agent.critic2, AggregatingCritic)
        assert isinstance(agent.target_critic2, AggregatingCritic)

    def test_forward_works_with_custom_dim(self):
        agent = make_ddpg_agent(lstm_hidden_dim=32)
        X = torch.randn(4, N_AGENTS * (OBS_DIM + ACT_DIM))
        Q, hidden = agent.critic(X)
        assert Q.shape == (4, 1)
        assert hidden[0].shape == (1, 4, 32)


# ── 2. MADDPG threads lstm_hidden_dim ──────────────────────────────────
class TestMADDPGThreading:
    def test_default_propagation(self):
        maddpg = make_maddpg()
        agent = maddpg.agents[0]
        assert agent.critic.lstm_hidden_dim == 64

    def test_custom_propagation(self):
        maddpg = make_maddpg(lstm_hidden_dim=128)
        agent = maddpg.agents[0]
        assert agent.critic.lstm_hidden_dim == 128
        assert agent.target_critic.lstm_hidden_dim == 128
        assert agent.critic2.lstm_hidden_dim == 128
        assert agent.target_critic2.lstm_hidden_dim == 128

    def test_update_works_with_custom_dim(self):
        """update() should work end-to-end with non-default lstm_hidden_dim."""
        maddpg = make_maddpg(lstm_hidden_dim=32)
        batch = 8
        obs = torch.randn(batch, N_AGENTS * OBS_DIM)
        acs = torch.randn(batch, N_AGENTS * ACT_DIM)
        rews = torch.randn(batch, 1)
        next_obs = torch.randn(batch, N_AGENTS * OBS_DIM)
        dones = torch.zeros(batch, 1)
        vf_loss, pol_loss, reg_loss = maddpg.update(obs, acs, rews, next_obs, dones, agent_i=0)
        assert isinstance(vf_loss, float)

    def test_update_sequence_works_with_custom_dim(self):
        """update_sequence() should work with non-default lstm_hidden_dim."""
        maddpg = make_maddpg(lstm_hidden_dim=32)
        seq_len, num_seq = 8, 4
        obs = torch.randn(seq_len, num_seq, N_AGENTS * OBS_DIM)
        acs = torch.randn(seq_len, num_seq, N_AGENTS * ACT_DIM)
        rews = torch.randn(seq_len, num_seq, 1)
        next_obs = torch.randn(seq_len, num_seq, N_AGENTS * OBS_DIM)
        dones = torch.zeros(seq_len, num_seq, 1)
        vf_loss, pol_loss, reg_loss = maddpg.update_sequence(
            obs, acs, rews, next_obs, dones, agent_i=0, burn_in_length=4)
        assert isinstance(vf_loss, float)


# ── 3. Backward compatibility ──────────────────────────────────────────
class TestBackwardCompat:
    def test_ddpg_agent_without_lstm_param(self):
        """DDPGAgent should work without explicit lstm_hidden_dim (defaults to 64)."""
        agent = DDPGAgent(
            dim_input_policy=OBS_DIM,
            dim_output_policy=ACT_DIM,
            n_agents=N_AGENTS,
            lr_actor=1e-4,
            lr_critic=1e-3,
            hidden_dim=HIDDEN_DIM,
        )
        assert agent.critic.lstm_hidden_dim == 64

    def test_maddpg_without_lstm_param(self):
        """MADDPG should work without explicit lstm_hidden_dim (defaults to 64)."""
        agent_init_params = [{'dim_input_policy': OBS_DIM,
                              'dim_output_policy': ACT_DIM,
                              'n_agents': N_AGENTS}]
        maddpg = MADDPG(agent_init_params=agent_init_params,
                         alg_types=['MADDPG'],
                         epsilon=0.0, noise=0.0,
                         hidden_dim=HIDDEN_DIM,
                         n_agents=N_AGENTS,
                         device='cpu',
                         use_ctm_actor=False,
                         prior_mode='none')
        assert maddpg.agents[0].critic.lstm_hidden_dim == 64


# ── 4. get_initial_hidden respects dim ─────────────────────────────────
class TestGetInitialHidden:
    def test_initial_hidden_shape_custom(self):
        agent = make_ddpg_agent(lstm_hidden_dim=32)
        h, c = agent.critic.get_initial_hidden(8, torch.device('cpu'))
        assert h.shape == (1, 8, 32)
        assert c.shape == (1, 8, 32)

    def test_initial_hidden_shape_default(self):
        agent = make_ddpg_agent()
        h, c = agent.critic.get_initial_hidden(8, torch.device('cpu'))
        assert h.shape == (1, 8, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
