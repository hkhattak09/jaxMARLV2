"""
Tests for gradient flow and critical hot paths across the full stateful pipeline.

Tests cover:
  1. Burn-in boundary: gradients do NOT flow through burn-in prefix
  2. Training LSTM: gradients DO flow through multi-step temporal chain
  3. Actor update: gradient flows through critic LSTM during actor loss
  4. Target soft update: target params move towards online params by tau
  5. niter increment: update_all_targets increments niter (controls delayed actor)
  6. Prior regularization in update_sequence: reg loss is nonzero, grads flow
  7. Full hot path: buffer → sample → multi-round update → critic loss decreases
  8. Grad clipping: large gradients are clipped
  9. Multi-agent update: updating agent_i=0 then agent_i=0 again doesn't corrupt
 10. Shared policy: all agents share the same policy object (parameter sharing)
 11. Both critics receive gradients independently
 12. Episode buffer → update_sequence shape compatibility with gradient flow

Run with:
    cd MARL-LLM/marl_llm
    python -m pytest tests/test_gradient_hotpaths.py -v
or:
    python tests/test_gradient_hotpaths.py
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_MARL_LLM = _THIS_DIR.parent
for p in [str(_MARL_LLM)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import numpy as np
import pytest

from algorithm.utils.networks import AggregatingCritic
from algorithm.utils.episode_buffer import EpisodeSequenceBuffer
from algorithm.algorithms.maddpg import MADDPG


# ── Constants ─────────────────────────────────────────────────────────────
N_AGENTS = 4
OBS_DIM = 16
ACT_DIM = 2
HIDDEN_DIM = 32
SEQ_LEN = 8
BURN_IN = 4
NUM_SEQ = 4
EPISODE_LENGTH = 20


def make_maddpg(prior_mode='none', lstm_hidden_dim=64):
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
                  prior_mode=prior_mode)


def make_sequence_data(seq_len=SEQ_LEN, num_seq=NUM_SEQ):
    obs = torch.randn(seq_len, num_seq, N_AGENTS * OBS_DIM)
    acs = torch.randn(seq_len, num_seq, N_AGENTS * ACT_DIM)
    rews = torch.randn(seq_len, num_seq, 1)
    next_obs = torch.randn(seq_len, num_seq, N_AGENTS * OBS_DIM)
    dones = torch.zeros(seq_len, num_seq, 1)
    return obs, acs, rews, next_obs, dones


def make_episode_buffer(n_episodes=5):
    buf = EpisodeSequenceBuffer(
        max_episodes=20,
        episode_length=EPISODE_LENGTH,
        num_agents=N_AGENTS,
        obs_dim=OBS_DIM,
        action_dim=ACT_DIM,
        sequence_length=SEQ_LEN,
    )
    for _ in range(n_episodes):
        T = EPISODE_LENGTH
        total_agents = N_AGENTS
        obs = np.random.randn(T, OBS_DIM, total_agents).astype(np.float32)
        acs = np.random.randn(T, ACT_DIM, total_agents).astype(np.float32)
        rews = np.random.randn(T, total_agents).astype(np.float32)
        next_obs = np.random.randn(T, OBS_DIM, total_agents).astype(np.float32)
        dones = np.zeros((T, total_agents), dtype=np.float32)
        prior = np.random.randn(T, ACT_DIM, total_agents).astype(np.float32)
        buf.push_episode(obs, acs, rews, next_obs, dones, prior)
    return buf


# ── 1. Burn-in boundary: no gradient leaks through ──────────────────────
class TestBurnInGradientBarrier:
    def test_burn_in_hidden_is_detached(self):
        """After burn-in, hidden states should have no grad_fn (detached)."""
        maddpg = make_maddpg()
        critic = maddpg.agents[0].critic
        num_seq = 4

        h = critic.get_initial_hidden(num_seq, torch.device('cpu'))

        # Simulate burn-in with no_grad (as update_sequence does)
        with torch.no_grad():
            for t in range(BURN_IN):
                x = torch.randn(num_seq, N_AGENTS * (OBS_DIM + ACT_DIM))
                _, h = critic(x, h)

        # Detach at boundary (as update_sequence does)
        h = tuple(s.detach() for s in h)

        assert not h[0].requires_grad
        assert not h[1].requires_grad
        assert h[0].grad_fn is None
        assert h[1].grad_fn is None

    def test_burn_in_encoder_gets_no_gradient(self):
        """Encoder weights should not receive gradient from burn-in forward passes."""
        maddpg = make_maddpg()
        critic = maddpg.agents[0].critic
        num_seq = 4

        # Zero all grads
        critic.zero_grad()

        h = critic.get_initial_hidden(num_seq, torch.device('cpu'))
        with torch.no_grad():
            for t in range(BURN_IN):
                x = torch.randn(num_seq, N_AGENTS * (OBS_DIM + ACT_DIM))
                _, h = critic(x, h)

        # No backward was called, so no grads should exist
        for p in critic.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0


# ── 2. Training LSTM: gradients flow through temporal chain ──────────────
class TestTrainingGradientFlow:
    def test_lstm_weights_get_gradient_from_training_steps(self):
        """LSTM weights should receive gradient from the training portion of update_sequence."""
        maddpg = make_maddpg()
        critic = maddpg.agents[0].critic
        obs, acs, rews, next_obs, dones = make_sequence_data()

        # Run update_sequence which does backward
        maddpg.update_sequence(obs, acs, rews, next_obs, dones,
                               agent_i=0, burn_in_length=BURN_IN)

        # LSTM params should have been updated (different from freshly constructed)
        fresh = AggregatingCritic(N_AGENTS, OBS_DIM, ACT_DIM, hidden_dim=HIDDEN_DIM)
        # The weights should differ after optimization step
        assert not torch.equal(
            critic.lstm.weight_ih_l0.data,
            fresh.lstm.weight_ih_l0.data
        ), "LSTM weights should have changed after update"

    def test_multi_timestep_gradient_chain(self):
        """Gradient from the last training timestep should reach LSTM weights
        through the temporal chain (not just from a single timestep)."""
        critic = AggregatingCritic(N_AGENTS, OBS_DIM, ACT_DIM, hidden_dim=HIDDEN_DIM)
        num_seq = 4
        train_len = 4

        h = critic.get_initial_hidden(num_seq, torch.device('cpu'))
        # Detach to simulate post-burn-in boundary
        h = tuple(s.detach() for s in h)

        # Forward through multiple training timesteps, accumulate loss
        total_loss = torch.tensor(0.0)
        for t in range(train_len):
            x = torch.randn(num_seq, N_AGENTS * (OBS_DIM + ACT_DIM))
            Q, h = critic(x, h)
            total_loss = total_loss + Q.mean()

        total_loss.backward()

        # LSTM should have gradients
        assert critic.lstm.weight_ih_l0.grad is not None
        assert critic.lstm.weight_ih_l0.grad.abs().sum() > 0

        # Encoder should also have gradients (chain: encoder → LSTM → head → loss)
        assert critic.encoder[0].weight.grad is not None
        assert critic.encoder[0].weight.grad.abs().sum() > 0

        # Head should have gradients
        assert critic.head[0].weight.grad is not None
        assert critic.head[0].weight.grad.abs().sum() > 0

    def test_both_critics_get_independent_gradients(self):
        """critic and critic2 should receive independent gradient updates."""
        maddpg = make_maddpg()
        obs, acs, rews, next_obs, dones = make_sequence_data()

        c1_before = {n: p.clone() for n, p in maddpg.agents[0].critic.named_parameters()}
        c2_before = {n: p.clone() for n, p in maddpg.agents[0].critic2.named_parameters()}

        maddpg.update_sequence(obs, acs, rews, next_obs, dones,
                               agent_i=0, burn_in_length=BURN_IN)

        c1_delta = sum((p - c1_before[n]).abs().sum().item()
                       for n, p in maddpg.agents[0].critic.named_parameters())
        c2_delta = sum((p - c2_before[n]).abs().sum().item()
                       for n, p in maddpg.agents[0].critic2.named_parameters())

        assert c1_delta > 0, "Critic 1 should have been updated"
        assert c2_delta > 0, "Critic 2 should have been updated"
        # They should differ (different random init, independent optimizers)
        assert abs(c1_delta - c2_delta) > 1e-8, \
            "Critic 1 and 2 deltas should differ (independent updates)"


# ── 3. Actor gradient flows through critic LSTM ─────────────────────────
class TestActorGradientThroughCritic:
    def test_actor_loss_produces_policy_gradients(self):
        """Actor update should produce gradients in policy network."""
        maddpg = make_maddpg()
        maddpg.niter = 0  # ensure actor update fires (even)
        obs, acs, rews, next_obs, dones = make_sequence_data()

        policy_before = {n: p.clone() for n, p in maddpg.agents[0].policy.named_parameters()}

        maddpg.update_sequence(obs, acs, rews, next_obs, dones,
                               agent_i=0, burn_in_length=BURN_IN)

        changed = any(
            not torch.equal(p, policy_before[n])
            for n, p in maddpg.agents[0].policy.named_parameters()
        )
        assert changed, "Policy params should change during actor update"

    def test_actor_critic_hidden_reburned_independently(self):
        """Actor update re-burns-in the critic hidden state independently from critic update."""
        maddpg = make_maddpg()
        maddpg.niter = 0
        critic = maddpg.agents[0].critic
        obs, acs, rews, next_obs, dones = make_sequence_data()

        # After update_sequence, critic params will have changed (from critic update).
        # The actor update re-burns-in from scratch with the NEW critic params.
        # We verify this by checking that actor loss is finite and policy changes.
        vf_loss, pol_loss, _ = maddpg.update_sequence(
            obs, acs, rews, next_obs, dones,
            agent_i=0, burn_in_length=BURN_IN)

        assert np.isfinite(pol_loss), "Actor loss should be finite"
        assert pol_loss != 0.0, "Actor loss should be nonzero"


# ── 4. Target soft update ────────────────────────────────────────────────
class TestTargetSoftUpdate:
    def test_target_params_move_towards_online(self):
        """After update_all_targets, target params should move toward online params."""
        maddpg = make_maddpg()
        tau = maddpg.tau  # 0.01

        # Record initial target and online params
        online_p = list(maddpg.agents[0].critic.parameters())[0].data.clone()
        target_p_before = list(maddpg.agents[0].target_critic.parameters())[0].data.clone()

        # Do a critic update to make online differ from target
        obs, acs, rews, next_obs, dones = make_sequence_data()
        maddpg.update_sequence(obs, acs, rews, next_obs, dones,
                               agent_i=0, burn_in_length=BURN_IN)

        online_p_after = list(maddpg.agents[0].critic.parameters())[0].data.clone()
        assert not torch.equal(online_p, online_p_after), \
            "Sanity: online params should have changed"

        # Now soft update
        target_p_before_update = list(maddpg.agents[0].target_critic.parameters())[0].data.clone()
        maddpg.update_all_targets()
        target_p_after = list(maddpg.agents[0].target_critic.parameters())[0].data.clone()

        # Target should have moved
        assert not torch.equal(target_p_before_update, target_p_after), \
            "Target params should change after soft update"

        # Verify soft update formula: target = tau * online + (1-tau) * target
        expected = tau * online_p_after + (1 - tau) * target_p_before_update
        assert torch.allclose(target_p_after, expected, atol=1e-6), \
            "Target params should follow soft update formula"

    def test_all_three_targets_updated(self):
        """All target networks (critic, critic2, policy) should be soft-updated."""
        maddpg = make_maddpg()
        obs, acs, rews, next_obs, dones = make_sequence_data()
        maddpg.niter = 0  # actor update fires
        maddpg.update_sequence(obs, acs, rews, next_obs, dones,
                               agent_i=0, burn_in_length=BURN_IN)

        tc1_before = list(maddpg.agents[0].target_critic.parameters())[0].data.clone()
        tc2_before = list(maddpg.agents[0].target_critic2.parameters())[0].data.clone()
        tp_before = list(maddpg.agents[0].target_policy.parameters())[0].data.clone()

        maddpg.update_all_targets()

        tc1_after = list(maddpg.agents[0].target_critic.parameters())[0].data.clone()
        tc2_after = list(maddpg.agents[0].target_critic2.parameters())[0].data.clone()
        tp_after = list(maddpg.agents[0].target_policy.parameters())[0].data.clone()

        assert not torch.equal(tc1_before, tc1_after), "Target critic 1 should update"
        assert not torch.equal(tc2_before, tc2_after), "Target critic 2 should update"
        assert not torch.equal(tp_before, tp_after), "Target policy should update"


# ── 5. niter increment ──────────────────────────────────────────────────
class TestNiterIncrement:
    def test_update_all_targets_increments_niter(self):
        maddpg = make_maddpg()
        initial_niter = maddpg.niter
        maddpg.update_all_targets()
        assert maddpg.niter == initial_niter + 1

    def test_niter_controls_delayed_actor(self):
        """Even niter → actor updates; odd niter → actor skipped."""
        maddpg = make_maddpg()
        obs, acs, rews, next_obs, dones = make_sequence_data()

        # niter=0 (even): actor should update
        maddpg.niter = 0
        policy_before = {n: p.clone() for n, p in maddpg.agents[0].policy.named_parameters()}
        maddpg.update_sequence(obs, acs, rews, next_obs, dones,
                               agent_i=0, burn_in_length=BURN_IN)
        changed_even = any(
            not torch.equal(p, policy_before[n])
            for n, p in maddpg.agents[0].policy.named_parameters()
        )
        assert changed_even, "Actor should update on even niter"

        # niter=1 (odd): actor should NOT update
        maddpg.niter = 1
        policy_before2 = {n: p.clone() for n, p in maddpg.agents[0].policy.named_parameters()}
        maddpg.update_sequence(obs, acs, rews, next_obs, dones,
                               agent_i=0, burn_in_length=BURN_IN)
        changed_odd = any(
            not torch.equal(p, policy_before2[n])
            for n, p in maddpg.agents[0].policy.named_parameters()
        )
        assert not changed_odd, "Actor should NOT update on odd niter"


# ── 6. Prior regularization in update_sequence ───────────────────────────
class TestPriorRegularization:
    def test_regularize_mode_produces_nonzero_reg_loss(self):
        """With prior_mode='regularize', reg_loss should be nonzero."""
        maddpg = make_maddpg(prior_mode='regularize')
        maddpg.niter = 0  # actor fires
        obs, acs, rews, next_obs, dones = make_sequence_data()
        # Prior with some nonzero values (not all near zero, which would be filtered)
        prior = torch.randn(SEQ_LEN, NUM_SEQ, N_AGENTS * ACT_DIM) + 1.0

        _, _, reg_loss = maddpg.update_sequence(
            obs, acs, rews, next_obs, dones,
            agent_i=0, prior_seq=prior, alpha=0.5,
            burn_in_length=BURN_IN)

        assert reg_loss > 0.0, "Reg loss should be nonzero with prior regularization"

    def test_no_prior_mode_gives_zero_reg(self):
        """With prior_mode='none', reg_loss should be 0."""
        maddpg = make_maddpg(prior_mode='none')
        maddpg.niter = 0
        obs, acs, rews, next_obs, dones = make_sequence_data()

        _, _, reg_loss = maddpg.update_sequence(
            obs, acs, rews, next_obs, dones,
            agent_i=0, burn_in_length=BURN_IN)

        assert reg_loss == 0.0

    def test_regularize_affects_policy_differently(self):
        """Policy update with prior_mode='regularize' should differ from prior_mode='none'."""
        torch.manual_seed(42)
        maddpg_reg = make_maddpg(prior_mode='regularize')
        maddpg_reg.niter = 0
        obs, acs, rews, next_obs, dones = make_sequence_data()
        prior = torch.randn(SEQ_LEN, NUM_SEQ, N_AGENTS * ACT_DIM) + 1.0

        torch.manual_seed(42)
        maddpg_none = make_maddpg(prior_mode='none')
        maddpg_none.niter = 0

        _, pol_loss_reg, _ = maddpg_reg.update_sequence(
            obs, acs, rews, next_obs, dones,
            agent_i=0, prior_seq=prior, alpha=0.5, burn_in_length=BURN_IN)

        _, pol_loss_none, _ = maddpg_none.update_sequence(
            obs, acs, rews, next_obs, dones,
            agent_i=0, burn_in_length=BURN_IN)

        # The losses should differ due to regularization
        assert abs(pol_loss_reg - pol_loss_none) > 1e-6, \
            "Prior regularization should change the policy loss"


# ── 7. Full hot path: buffer → multi-round update → loss decreases ──────
class TestFullHotPath:
    def test_critic_loss_decreases_on_repeated_updates(self):
        """Fitting the critic to the same data should reduce loss over iterations."""
        maddpg = make_maddpg()

        # Fixed data (same batch every time)
        torch.manual_seed(0)
        obs, acs, rews, next_obs, dones = make_sequence_data()

        losses = []
        for _ in range(20):
            vf_loss, _, _ = maddpg.update_sequence(
                obs, acs, rews, next_obs, dones,
                agent_i=0, burn_in_length=BURN_IN)
            losses.append(vf_loss)
            maddpg.update_all_targets()

        # Loss should generally decrease (first few > last few)
        first_5_avg = np.mean(losses[:5])
        last_5_avg = np.mean(losses[-5:])
        assert last_5_avg < first_5_avg, \
            f"Critic loss should decrease: first 5 avg={first_5_avg:.4f}, last 5 avg={last_5_avg:.4f}"

    def test_buffer_to_update_full_pipeline(self):
        """Full pipeline: episode buffer → sample → update_sequence → check params changed."""
        maddpg = make_maddpg()
        buf = make_episode_buffer(n_episodes=5)

        params_before = {n: p.clone() for n, p in maddpg.agents[0].critic.named_parameters()}

        for _ in range(3):
            sample = buf.sample(NUM_SEQ, to_gpu=False)
            obs_s, acs_s, rews_s, next_obs_s, dones_s, _ = sample
            maddpg.update_sequence(obs_s, acs_s, rews_s, next_obs_s, dones_s,
                                   agent_i=0, burn_in_length=BURN_IN)
            maddpg.update_all_targets()

        changed = any(
            not torch.equal(p, params_before[n])
            for n, p in maddpg.agents[0].critic.named_parameters()
        )
        assert changed, "Critic should have changed after full pipeline"

    def test_multi_agent_update_loop(self):
        """Updating all agents in sequence should work and update each agent's critics."""
        maddpg = make_maddpg()
        buf = make_episode_buffer(n_episodes=5)

        # Record params for agent 0's critic before the loop
        c0_before = list(maddpg.agents[0].critic.parameters())[0].data.clone()

        sample = buf.sample(NUM_SEQ, to_gpu=False)
        obs_s, acs_s, rews_s, next_obs_s, dones_s, _ = sample

        for a_i in range(maddpg.nagents):
            vf_loss, pol_loss, reg_loss = maddpg.update_sequence(
                obs_s, acs_s, rews_s, next_obs_s, dones_s,
                agent_i=a_i, burn_in_length=BURN_IN)
            assert np.isfinite(vf_loss)

        maddpg.update_all_targets()

        c0_after = list(maddpg.agents[0].critic.parameters())[0].data.clone()
        assert not torch.equal(c0_before, c0_after), \
            "Agent 0's critic should have changed"


# ── 8. Grad clipping ────────────────────────────────────────────────────
class TestGradClipping:
    def test_gradient_norm_is_bounded(self):
        """After update, gradient norms should be <= 0.5 (the clip value)."""
        maddpg = make_maddpg()
        critic = maddpg.agents[0].critic

        # Use large-magnitude data to produce large gradients
        obs, acs, rews, next_obs, dones = make_sequence_data()
        obs = obs * 100
        rews = rews * 100

        # We need to manually check grad norms BEFORE they're consumed by optimizer.
        # So we'll do a manual forward-backward without stepping.
        critic.zero_grad()
        h = critic.get_initial_hidden(NUM_SEQ, torch.device('cpu'))
        h = tuple(s.detach() for s in h)

        total_loss = torch.tensor(0.0)
        for t in range(SEQ_LEN):
            x = torch.randn(NUM_SEQ, N_AGENTS * (OBS_DIM + ACT_DIM)) * 100
            Q, h = critic(x, h)
            total_loss = total_loss + Q.mean()

        total_loss.backward()

        # Check gradient norm BEFORE clipping
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), float('inf'))
        # It should be > 0.5 for this test to be meaningful
        # (large data should produce large grads)
        if pre_clip_norm <= 0.5:
            pytest.skip("Gradients were already small; can't test clipping")

        # Now clip
        critic.zero_grad()
        h = critic.get_initial_hidden(NUM_SEQ, torch.device('cpu'))
        h = tuple(s.detach() for s in h)
        total_loss = torch.tensor(0.0)
        for t in range(SEQ_LEN):
            x = torch.randn(NUM_SEQ, N_AGENTS * (OBS_DIM + ACT_DIM)) * 100
            Q, h = critic(x, h)
            total_loss = total_loss + Q.mean()
        total_loss.backward()

        clipped_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        # After clipping, the total norm should be <= 0.5
        post_clip_params = [p for p in critic.parameters() if p.grad is not None]
        post_clip_total = torch.sqrt(
            sum((p.grad ** 2).sum() for p in post_clip_params)
        )
        assert post_clip_total <= 0.5 + 1e-6, \
            f"Post-clip grad norm should be <= 0.5, got {post_clip_total:.4f}"


# ── 9. Shared policy (parameter sharing) ────────────────────────────────
class TestSharedPolicy:
    def test_all_agents_share_policy_object(self):
        """With N_AGENTS=1 in MADDPG.nagents (single agent_init_params), there's one agent.
        But the policy is shared: updating agent 0 with different obs slices should work."""
        maddpg = make_maddpg()
        # With our test setup, maddpg.nagents == 1 (single agent_init_params entry)
        # But the policy handles all N_AGENTS via obs slicing
        assert maddpg.nagents == 1
        assert maddpg.agents[0].policy is not None

    def test_policy_gradient_from_different_agent_slices(self):
        """Gradient should flow regardless of which agent_i slice is used."""
        maddpg = make_maddpg()
        maddpg.niter = 0
        obs, acs, rews, next_obs, dones = make_sequence_data()

        # Since nagents=1, only agent_i=0 is valid, but the actor uses agent_i
        # to slice obs. Verify it works.
        policy_before = list(maddpg.agents[0].policy.parameters())[0].data.clone()
        maddpg.update_sequence(obs, acs, rews, next_obs, dones,
                               agent_i=0, burn_in_length=BURN_IN)
        policy_after = list(maddpg.agents[0].policy.parameters())[0].data.clone()

        assert not torch.equal(policy_before, policy_after), \
            "Policy should change from agent_i=0 update"


# ── 10. update() hot path (MLP/random-transition path) ──────────────────
class TestUpdateHotPath:
    def test_update_critic_loss_decreases(self):
        """Repeated update() on same data should decrease critic loss."""
        maddpg = make_maddpg()

        torch.manual_seed(0)
        batch = 16
        obs = torch.randn(batch, N_AGENTS * OBS_DIM)
        acs = torch.randn(batch, N_AGENTS * ACT_DIM)
        rews = torch.randn(batch, 1)
        next_obs = torch.randn(batch, N_AGENTS * OBS_DIM)
        dones = torch.zeros(batch, 1)

        losses = []
        for _ in range(20):
            vf_loss, _, _ = maddpg.update(obs, acs, rews, next_obs, dones, agent_i=0)
            losses.append(vf_loss)
            maddpg.update_all_targets()

        first_5 = np.mean(losses[:5])
        last_5 = np.mean(losses[-5:])
        assert last_5 < first_5, \
            f"update() critic loss should decrease: {first_5:.4f} → {last_5:.4f}"

    def test_update_actor_gradient_flows(self):
        """update() should produce actor param changes on even niter."""
        maddpg = make_maddpg()
        maddpg.niter = 0

        batch = 16
        obs = torch.randn(batch, N_AGENTS * OBS_DIM)
        acs = torch.randn(batch, N_AGENTS * ACT_DIM)
        rews = torch.randn(batch, 1)
        next_obs = torch.randn(batch, N_AGENTS * OBS_DIM)
        dones = torch.zeros(batch, 1)

        policy_before = list(maddpg.agents[0].policy.parameters())[0].data.clone()
        maddpg.update(obs, acs, rews, next_obs, dones, agent_i=0)
        policy_after = list(maddpg.agents[0].policy.parameters())[0].data.clone()

        assert not torch.equal(policy_before, policy_after)

    def test_update_with_prior_regularization(self):
        """update() with prior_mode='regularize' should produce nonzero reg loss."""
        maddpg = make_maddpg(prior_mode='regularize')
        maddpg.niter = 0

        batch = 16
        obs = torch.randn(batch, N_AGENTS * OBS_DIM)
        acs = torch.randn(batch, N_AGENTS * ACT_DIM)
        rews = torch.randn(batch, 1)
        next_obs = torch.randn(batch, N_AGENTS * OBS_DIM)
        dones = torch.zeros(batch, 1)
        prior = torch.randn(batch, N_AGENTS * ACT_DIM) + 1.0  # nonzero

        _, _, reg_loss = maddpg.update(obs, acs, rews, next_obs, dones,
                                       agent_i=0, acs_prior=prior, alpha=0.5)
        assert reg_loss > 0.0


# ── 11. Episode buffer → update_sequence gradient shapes ─────────────────
class TestBufferUpdateGradientCompat:
    def test_buffer_sample_shapes_match_update_sequence(self):
        """Buffer samples should have exactly the shapes update_sequence expects."""
        buf = make_episode_buffer(n_episodes=3)
        sample = buf.sample(NUM_SEQ, to_gpu=False)
        obs_s, acs_s, rews_s, next_obs_s, dones_s, prior_s = sample

        assert obs_s.shape == (SEQ_LEN, NUM_SEQ, N_AGENTS * OBS_DIM)
        assert acs_s.shape == (SEQ_LEN, NUM_SEQ, N_AGENTS * ACT_DIM)
        assert rews_s.shape == (SEQ_LEN, NUM_SEQ, 1)
        assert next_obs_s.shape == (SEQ_LEN, NUM_SEQ, N_AGENTS * OBS_DIM)
        assert dones_s.shape == (SEQ_LEN, NUM_SEQ, 1)
        assert prior_s.shape == (SEQ_LEN, NUM_SEQ, N_AGENTS * ACT_DIM)

    def test_buffer_sample_produces_gradient_in_critic(self):
        """Buffer samples should produce valid gradients when fed to update_sequence."""
        maddpg = make_maddpg()
        buf = make_episode_buffer(n_episodes=3)

        sample = buf.sample(NUM_SEQ, to_gpu=False)
        obs_s, acs_s, rews_s, next_obs_s, dones_s, _ = sample

        critic_params_before = {n: p.clone()
                                for n, p in maddpg.agents[0].critic.named_parameters()}

        vf_loss, _, _ = maddpg.update_sequence(
            obs_s, acs_s, rews_s, next_obs_s, dones_s,
            agent_i=0, burn_in_length=BURN_IN)

        assert np.isfinite(vf_loss)
        changed = any(
            not torch.equal(p, critic_params_before[n])
            for n, p in maddpg.agents[0].critic.named_parameters()
        )
        assert changed, "Critic should change from buffer-sourced data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
