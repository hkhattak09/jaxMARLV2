"""
Tests for AggregatingCritic with LSTM (recurrent critic).

Tests cover:
  1. Construction — LSTM layer exists, correct dimensions
  2. Forward pass — shape correctness, hidden state returned
  3. Hidden state flow — passing hidden between calls updates state
  4. Fresh hidden — None hidden initializes zeros
  5. Gradient flow — gradients reach encoder, LSTM, and head
  6. Sequence processing — feeding timesteps sequentially produces different hidden states
  7. Backward compatibility — still works with the same input format

Run with:
    cd MARL-LLM/marl_llm
    python -m pytest tests/test_recurrent_critic.py -v
or:
    python tests/test_recurrent_critic.py
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

from algorithm.utils.networks import AggregatingCritic, MLPNetwork


# ── Fixtures ───────────────────────────────────────────────────────────────
N_AGENTS = 24
OBS_DIM = 192
ACT_DIM = 2
HIDDEN_DIM = 128
EMBED_DIM = 64
LSTM_HIDDEN_DIM = 64
BATCH_SIZE = 8


def make_critic():
    return AggregatingCritic(N_AGENTS, OBS_DIM, ACT_DIM,
                              hidden_dim=HIDDEN_DIM, embed_dim=EMBED_DIM,
                              lstm_hidden_dim=LSTM_HIDDEN_DIM)


def make_input(batch_size=BATCH_SIZE):
    """Create fake joint input: (batch, n_agents*obs_dim + n_agents*act_dim)."""
    total_dim = N_AGENTS * OBS_DIM + N_AGENTS * ACT_DIM
    return torch.randn(batch_size, total_dim)


# ── 1. Construction ────────────────────────────────────────────────────────
class TestConstruction:
    def test_has_lstm(self):
        critic = make_critic()
        assert hasattr(critic, 'lstm')
        assert isinstance(critic.lstm, torch.nn.LSTM)

    def test_lstm_dimensions(self):
        critic = make_critic()
        assert critic.lstm.input_size == EMBED_DIM
        assert critic.lstm.hidden_size == LSTM_HIDDEN_DIM

    def test_head_input_matches_lstm_output(self):
        critic = make_critic()
        # Head's first linear layer should accept lstm_hidden_dim
        first_layer = critic.head[0]
        assert first_layer.in_features == LSTM_HIDDEN_DIM


# ── 2. Forward pass ───────────────────────────────────────────────────────
class TestForward:
    def test_output_shapes(self):
        critic = make_critic()
        X = make_input()
        Q, hidden = critic(X)
        assert Q.shape == (BATCH_SIZE, 1)
        h, c = hidden
        assert h.shape == (1, BATCH_SIZE, LSTM_HIDDEN_DIM)
        assert c.shape == (1, BATCH_SIZE, LSTM_HIDDEN_DIM)

    def test_none_hidden_works(self):
        critic = make_critic()
        X = make_input()
        Q, hidden = critic(X, hidden=None)
        assert Q.shape == (BATCH_SIZE, 1)

    def test_explicit_hidden_works(self):
        critic = make_critic()
        X = make_input()
        h0 = torch.zeros(1, BATCH_SIZE, LSTM_HIDDEN_DIM)
        c0 = torch.zeros(1, BATCH_SIZE, LSTM_HIDDEN_DIM)
        Q, hidden = critic(X, hidden=(h0, c0))
        assert Q.shape == (BATCH_SIZE, 1)


# ── 3. Hidden state flow ──────────────────────────────────────────────────
class TestHiddenStateFlow:
    def test_hidden_changes_across_steps(self):
        """Two different inputs should produce different hidden states."""
        critic = make_critic()
        X1 = make_input()
        X2 = make_input() + 1.0  # different input

        _, hidden1 = critic(X1, hidden=None)
        _, hidden2 = critic(X2, hidden=None)

        # Hidden states should differ (different inputs)
        assert not torch.allclose(hidden1[0], hidden2[0])

    def test_sequential_hidden_propagation(self):
        """Passing hidden from step 1 to step 2 changes the output vs fresh hidden."""
        critic = make_critic()
        X = make_input()

        # Step 1: fresh
        Q1, hidden1 = critic(X, hidden=None)
        # Step 2: carry hidden forward
        Q2, hidden2 = critic(X, hidden=hidden1)
        # Step 2 fresh: no carry
        Q2_fresh, _ = critic(X, hidden=None)

        # Q2 (with carried hidden) should differ from Q2_fresh (fresh hidden)
        assert not torch.allclose(Q2, Q2_fresh)

    def test_get_initial_hidden(self):
        critic = make_critic()
        h, c = critic.get_initial_hidden(BATCH_SIZE, torch.device('cpu'))
        assert h.shape == (1, BATCH_SIZE, LSTM_HIDDEN_DIM)
        assert torch.all(h == 0)
        assert torch.all(c == 0)


# ── 4. Gradient flow ──────────────────────────────────────────────────────
class TestGradients:
    def test_gradients_flow_through_all_components(self):
        critic = make_critic()
        X = make_input()
        Q, _ = critic(X)
        loss = Q.mean()
        loss.backward()

        # Check encoder gets gradients
        enc_grad = critic.encoder[0].weight.grad
        assert enc_grad is not None and enc_grad.abs().sum() > 0

        # Check LSTM gets gradients
        lstm_grad = critic.lstm.weight_ih_l0.grad
        assert lstm_grad is not None and lstm_grad.abs().sum() > 0

        # Check head gets gradients
        head_grad = critic.head[0].weight.grad
        assert head_grad is not None and head_grad.abs().sum() > 0


# ── 5. MLPNetwork unchanged ───────────────────────────────────────────────
class TestMLPUnchanged:
    def test_mlp_still_works(self):
        """MLPNetwork should be completely unaffected by AggregatingCritic changes."""
        mlp = MLPNetwork(OBS_DIM, ACT_DIM, hidden_dim=180, constrain_out=True)
        X = torch.randn(BATCH_SIZE, OBS_DIM)
        out = mlp(X)
        assert out.shape == (BATCH_SIZE, ACT_DIM)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
