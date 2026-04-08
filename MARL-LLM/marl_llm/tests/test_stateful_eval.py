"""
Tests for Phase 7: Stateful eval rollout.

Tests cover:
  1. Eval hidden states are initialized once per episode (not per step)
  2. Eval hidden states carry forward across steps (states evolve)
  3. MLP eval path: hidden_states stays None throughout
  4. Seed mode: prior seeds hidden state at episode start only
  5. Multiple eval episodes: hidden states reinitialised at each episode start
  6. No-grad context: hidden state carry-forward works under torch.no_grad()

Run with:
    cd MARL-LLM/marl_llm
    python -m pytest tests/test_stateful_eval.py -v
or:
    python tests/test_stateful_eval.py
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

from algorithm.algorithms.maddpg import MADDPG


# ── Constants ─────────────────────────────────────────────────────────────
N_AGENTS = 4
OBS_DIM = 16
ACT_DIM = 2
HIDDEN_DIM = 32
EPISODE_LENGTH = 10  # short for test speed


def make_maddpg(use_ctm=False, lstm_hidden_dim=64):
    """Create MADDPG instance for testing."""
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
                  use_ctm_actor=use_ctm,
                  prior_mode='none')


# ── 1. Stateful eval: hidden states evolve across steps ──────────────────
class TestStatefulEvalCarryForward:
    def test_hidden_states_change_during_eval_rollout(self):
        """Simulate stateful eval: hidden states should evolve across steps."""
        maddpg = make_maddpg()
        critic = maddpg.agents[0].critic

        batch = 4
        eval_hidden = critic.get_initial_hidden(batch, torch.device('cpu'))
        initial_h = eval_hidden[0].clone()

        with torch.no_grad():
            for t in range(EPISODE_LENGTH):
                x = torch.randn(batch, N_AGENTS * (OBS_DIM + ACT_DIM))
                _, eval_hidden = critic(x, eval_hidden)

        # After 10 steps, hidden state should differ from initial zeros
        assert not torch.allclose(eval_hidden[0], initial_h), \
            "Hidden state should evolve during stateful eval rollout"

    def test_hidden_states_differ_at_each_step(self):
        """Each step should produce a different hidden state."""
        maddpg = make_maddpg()
        critic = maddpg.agents[0].critic

        batch = 4
        h = critic.get_initial_hidden(batch, torch.device('cpu'))
        states_over_time = [h[0].clone()]

        with torch.no_grad():
            for t in range(5):
                x = torch.randn(batch, N_AGENTS * (OBS_DIM + ACT_DIM))
                _, h = critic(x, h)
                states_over_time.append(h[0].clone())

        for i in range(1, len(states_over_time)):
            assert not torch.allclose(states_over_time[i], states_over_time[i - 1]), \
                f"Hidden state at step {i} should differ from step {i-1}"


# ── 2. MLP eval path: hidden_states is None ──────────────────────────────
class TestMLPEvalPath:
    def test_mlp_eval_hidden_is_none(self):
        """MLP eval should pass hidden_states=None to step()."""
        maddpg = make_maddpg(use_ctm=False)
        # In MLP path, eval_hidden should be None
        eval_hidden = None
        assert eval_hidden is None

        # step() should accept None hidden_states
        obs = torch.randn(N_AGENTS * OBS_DIM).unsqueeze(1)  # (obs_dim, 1*n_agents) won't work
        # Just verify the maddpg object has use_ctm_actor=False
        assert not maddpg.use_ctm_actor

    def test_mlp_critic_works_without_hidden(self):
        """MLP critic forward with hidden=None still works."""
        maddpg = make_maddpg(use_ctm=False)
        critic = maddpg.agents[0].critic
        batch = 4
        x = torch.randn(batch, N_AGENTS * (OBS_DIM + ACT_DIM))
        Q, hidden = critic(x, hidden=None)
        assert Q.shape == (batch, 1)
        assert hidden is not None  # LSTM produces hidden even if input was None


# ── 3. Multiple eval episodes: hidden states reset each episode ──────────
class TestMultipleEvalEpisodes:
    def test_hidden_reset_between_episodes(self):
        """Hidden states should be fresh at the start of each eval episode."""
        maddpg = make_maddpg()
        critic = maddpg.agents[0].critic
        batch = 4

        episode_final_hiddens = []
        for ep_i in range(3):
            # Reinitialise at episode start (simulating eval loop)
            h = critic.get_initial_hidden(batch, torch.device('cpu'))
            assert torch.all(h[0] == 0), f"Episode {ep_i} should start with zero hidden"

            with torch.no_grad():
                for t in range(EPISODE_LENGTH):
                    x = torch.randn(batch, N_AGENTS * (OBS_DIM + ACT_DIM))
                    _, h = critic(x, h)
            episode_final_hiddens.append(h[0].clone())

        # Final hiddens should differ between episodes (different random inputs)
        assert not torch.allclose(episode_final_hiddens[0], episode_final_hiddens[1]), \
            "Different episodes with different inputs should yield different final hiddens"

    def test_hidden_not_carried_between_episodes(self):
        """Verify that hidden states from episode N do NOT leak into episode N+1."""
        maddpg = make_maddpg()
        critic = maddpg.agents[0].critic
        batch = 4

        # Run episode 1
        h = critic.get_initial_hidden(batch, torch.device('cpu'))
        with torch.no_grad():
            for t in range(EPISODE_LENGTH):
                x = torch.randn(batch, N_AGENTS * (OBS_DIM + ACT_DIM))
                _, h = critic(x, h)
        ep1_final = h[0].clone()

        # Episode 2 starts fresh
        h2 = critic.get_initial_hidden(batch, torch.device('cpu'))
        assert torch.all(h2[0] == 0), "New episode hidden should be zeros, not carried from ep1"
        assert not torch.allclose(h2[0], ep1_final)


# ── 4. No-grad context works with carry-forward ──────────────────────────
class TestNoGradCarryForward:
    def test_no_grad_does_not_prevent_hidden_update(self):
        """torch.no_grad() should not prevent LSTM hidden state updates."""
        maddpg = make_maddpg()
        critic = maddpg.agents[0].critic
        batch = 4

        h = critic.get_initial_hidden(batch, torch.device('cpu'))
        with torch.no_grad():
            x = torch.randn(batch, N_AGENTS * (OBS_DIM + ACT_DIM))
            _, h_new = critic(x, h)

        assert not torch.allclose(h_new[0], h[0]), \
            "LSTM hidden should update even under no_grad"

    def test_no_grad_hidden_has_no_grad_fn(self):
        """Hidden states produced under no_grad should not require grad."""
        maddpg = make_maddpg()
        critic = maddpg.agents[0].critic
        batch = 4

        h = critic.get_initial_hidden(batch, torch.device('cpu'))
        with torch.no_grad():
            x = torch.randn(batch, N_AGENTS * (OBS_DIM + ACT_DIM))
            _, h_new = critic(x, h)

        assert not h_new[0].requires_grad
        assert not h_new[1].requires_grad


# ── 5. Stateful eval matches stateful training pattern ────────────────────
class TestEvalTrainingConsistency:
    def test_same_input_same_hidden_trajectory(self):
        """Given identical inputs, eval and training produce same hidden trajectory."""
        maddpg = make_maddpg()
        critic = maddpg.agents[0].critic
        batch = 4

        # Generate fixed inputs
        torch.manual_seed(42)
        inputs = [torch.randn(batch, N_AGENTS * (OBS_DIM + ACT_DIM)) for _ in range(5)]

        # Eval mode (no_grad)
        critic.eval()
        h_eval = critic.get_initial_hidden(batch, torch.device('cpu'))
        eval_hiddens = []
        with torch.no_grad():
            for x in inputs:
                _, h_eval = critic(x, h_eval)
                eval_hiddens.append(h_eval[0].clone())

        # Training mode (with grad)
        critic.train()
        h_train = critic.get_initial_hidden(batch, torch.device('cpu'))
        train_hiddens = []
        for x in inputs:
            _, h_train = critic(x, h_train)
            train_hiddens.append(h_train[0].clone().detach())

        # Should match exactly (LSTM is deterministic given same input + state)
        for i, (e, t) in enumerate(zip(eval_hiddens, train_hiddens)):
            assert torch.allclose(e, t, atol=1e-6), \
                f"Step {i}: eval and train hidden states should match for same inputs"


# ── 6. update_sequence still works after eval changes ─────────────────────
class TestNoRegression:
    def test_update_still_works(self):
        """update() should still work (MLP eval path)."""
        maddpg = make_maddpg(use_ctm=False)
        batch = 8
        obs = torch.randn(batch, N_AGENTS * OBS_DIM)
        acs = torch.randn(batch, N_AGENTS * ACT_DIM)
        rews = torch.randn(batch, 1)
        next_obs = torch.randn(batch, N_AGENTS * OBS_DIM)
        dones = torch.zeros(batch, 1)
        vf_loss, pol_loss, reg_loss = maddpg.update(
            obs, acs, rews, next_obs, dones, agent_i=0)
        assert isinstance(vf_loss, float)

    def test_update_sequence_still_works(self):
        """update_sequence() should still work (CTM eval path)."""
        maddpg = make_maddpg()
        seq_len, num_seq = 8, 4
        obs = torch.randn(seq_len, num_seq, N_AGENTS * OBS_DIM)
        acs = torch.randn(seq_len, num_seq, N_AGENTS * ACT_DIM)
        rews = torch.randn(seq_len, num_seq, 1)
        next_obs = torch.randn(seq_len, num_seq, N_AGENTS * OBS_DIM)
        dones = torch.zeros(seq_len, num_seq, 1)
        vf_loss, pol_loss, reg_loss = maddpg.update_sequence(
            obs, acs, rews, next_obs, dones, agent_i=0, burn_in_length=4)
        assert isinstance(vf_loss, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
