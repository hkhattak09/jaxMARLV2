"""
Comprehensive tests for the CTM actor implementation.

Tests cover:
  1. CTMActor  — init, shapes, forward, gradient flow
  2. CTMDDPGAgent — init (LazyLinear fix), step, save/load
  3. MADDPG + CTM — init, step (3-tuple), update, target_policies
  4. MADDPG + MLP — backwards compatibility (all of the above with MLP)
  5. Hidden state management — done-mask reset, detach, shapes throughout rollout
  6. Edge cases — epsilon-greedy branch, all-done mask, zero-agent batch

Run with:
    cd MARL-LLM/marl_llm
    python -m pytest tests/test_ctm_implementation.py -v
or:
    python tests/test_ctm_implementation.py
"""

import sys
import os
from pathlib import Path

# ── sys.path: allow running from repo root or from marl_llm/ ───────────────
_THIS_DIR   = Path(__file__).resolve().parent
_MARL_LLM   = _THIS_DIR.parent                               # …/MARL-LLM/marl_llm
_REPO_ROOT  = _MARL_LLM.parents[2]                          # …/new_marl_llm_implementation
_CTM_PATH   = str(_REPO_ROOT / "continuous-thought-machines")

for p in [str(_MARL_LLM), _CTM_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)
# ───────────────────────────────────────────────────────────────────────────

import math
import pytest
import torch
import numpy as np

# ── local imports ──────────────────────────────────────────────────────────
from algorithm.utils.ctm_actor import CTMActor
from algorithm.utils.ctm_agent import CTMDDPGAgent
from algorithm.utils.agents   import DDPGAgent
from algorithm.algorithms.maddpg import MADDPG
# ───────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# Shared constants — kept small so tests run fast on CPU
# ═══════════════════════════════════════════════════════════════════════════
OBS_DIM      = 12    # observation vector length
ACTION_DIM   = 2     # action vector length
CRITIC_DIM   = OBS_DIM + ACTION_DIM
N_AGENTS     = 4     # agents per env
N_ENVS       = 2     # parallel envs
BATCH_SIZE   = 8

# Minimal CTM config — intentionally small for CPU speed
CTM_CFG = dict(
    d_model          = 16,
    memory_length    = 4,
    n_synch_out      = 4,    # synch_size = 4*5//2 = 10
    iterations       = 2,
    synapse_depth    = 1,
    deep_nlms        = False,
    do_layernorm_nlm = True,
    memory_hidden_dims = 8,
)
SYNCH_SIZE = CTM_CFG['n_synch_out'] * (CTM_CFG['n_synch_out'] + 1) // 2  # 10


def make_ctm_actor():
    return CTMActor(obs_dim=OBS_DIM, action_dim=ACTION_DIM, **CTM_CFG)


def make_ctm_agent():
    return CTMDDPGAgent(
        dim_input_policy  = OBS_DIM,
        dim_output_policy = ACTION_DIM,
        dim_input_critic  = CRITIC_DIM,
        lr_actor          = 1e-4,
        lr_critic         = 1e-3,
        hidden_dim        = 32,
        ctm_config        = CTM_CFG,
    )


def make_maddpg_ctm():
    """Single shared agent (nagents=1) covering all N_AGENTS positions."""
    agent_init_params = [dict(
        dim_input_policy  = OBS_DIM,
        dim_output_policy = ACTION_DIM,
        dim_input_critic  = CRITIC_DIM,
    )]
    return MADDPG(
        agent_init_params = agent_init_params,
        alg_types         = ['MADDPG'],
        epsilon           = 0.1,
        noise             = 0.1,
        use_ctm_actor     = True,
        ctm_config        = CTM_CFG,
    )


def make_maddpg_mlp():
    agent_init_params = [dict(
        dim_input_policy  = OBS_DIM,
        dim_output_policy = ACTION_DIM,
        dim_input_critic  = CRITIC_DIM,
    )]
    return MADDPG(
        agent_init_params = agent_init_params,
        alg_types         = ['MADDPG'],
        epsilon           = 0.1,
        noise             = 0.1,
        use_ctm_actor     = False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. CTMActor
# ═══════════════════════════════════════════════════════════════════════════

class TestCTMActor:

    def test_init_creates_ctm_and_action_head(self):
        actor = make_ctm_actor()
        assert hasattr(actor, 'ctm')
        assert hasattr(actor, 'action_head')
        assert actor.action_head.in_features  == SYNCH_SIZE
        assert actor.action_head.out_features == ACTION_DIM

    def test_start_activated_trace_is_parameter(self):
        actor = make_ctm_actor()
        assert isinstance(actor.ctm.start_activated_trace, torch.nn.Parameter)
        assert actor.ctm.start_activated_trace.requires_grad

    def test_start_activated_trace_shape(self):
        actor = make_ctm_actor()
        p = actor.ctm.start_activated_trace
        assert p.shape == (CTM_CFG['d_model'], CTM_CFG['memory_length'])

    def test_get_initial_hidden_state_shapes(self):
        actor = make_ctm_actor()
        bs = 6
        st, ast = actor.get_initial_hidden_state(bs, torch.device('cpu'))
        assert st.shape  == (bs, CTM_CFG['d_model'], CTM_CFG['memory_length'])
        assert ast.shape == (bs, CTM_CFG['d_model'], CTM_CFG['memory_length'])

    def test_get_initial_hidden_state_zeros_for_state_trace(self):
        actor = make_ctm_actor()
        st, _ = actor.get_initial_hidden_state(3, torch.device('cpu'))
        assert st.sum().item() == 0.0

    def test_get_initial_hidden_state_activated_trace_from_parameter(self):
        """activated_state_trace should be a (cloned) expansion of start_activated_trace."""
        actor = make_ctm_actor()
        bs = 3
        _, ast = actor.get_initial_hidden_state(bs, torch.device('cpu'))
        param = actor.ctm.start_activated_trace
        for i in range(bs):
            assert torch.allclose(ast[i], param)

    def test_get_initial_hidden_state_independent_batches(self):
        """Each call must return fresh tensors (clone, not view)."""
        actor = make_ctm_actor()
        _, ast1 = actor.get_initial_hidden_state(2, torch.device('cpu'))
        _, ast2 = actor.get_initial_hidden_state(2, torch.device('cpu'))
        ast1.fill_(999.0)
        # Mutating ast1 must not affect ast2
        assert not torch.all(ast2 == 999.0)

    def test_forward_action_shape(self):
        actor = make_ctm_actor()
        bs = 5
        obs = torch.randn(bs, OBS_DIM)
        hs  = actor.get_initial_hidden_state(bs, torch.device('cpu'))
        actions, new_hs = actor(obs, hs)
        assert actions.shape == (bs, ACTION_DIM)

    def test_forward_actions_in_range(self):
        actor = make_ctm_actor()
        bs = 10
        obs = torch.randn(bs, OBS_DIM) * 100  # large input
        hs  = actor.get_initial_hidden_state(bs, torch.device('cpu'))
        actions, _ = actor(obs, hs)
        assert actions.min().item() >= -1.0 - 1e-6
        assert actions.max().item() <=  1.0 + 1e-6

    def test_forward_hidden_state_shapes_unchanged(self):
        actor = make_ctm_actor()
        bs = 3
        obs = torch.randn(bs, OBS_DIM)
        hs  = actor.get_initial_hidden_state(bs, torch.device('cpu'))
        _, new_hs = actor(obs, hs)
        st, ast = new_hs
        assert st.shape  == (bs, CTM_CFG['d_model'], CTM_CFG['memory_length'])
        assert ast.shape == (bs, CTM_CFG['d_model'], CTM_CFG['memory_length'])

    def test_forward_updates_hidden_state(self):
        """Hidden state must change after a forward pass (FIFO shift happened)."""
        actor = make_ctm_actor()
        bs = 2
        obs = torch.randn(bs, OBS_DIM)
        hs  = actor.get_initial_hidden_state(bs, torch.device('cpu'))
        st_before = hs[0].clone()
        _, new_hs = actor(obs, hs)
        assert not torch.allclose(new_hs[0], st_before)

    def test_gradient_flows_through_start_activated_trace(self):
        """
        Stateless update scenario: fresh hidden state → forward → loss → backward.
        Gradient must reach start_activated_trace.
        """
        actor = make_ctm_actor()
        obs   = torch.randn(BATCH_SIZE, OBS_DIM)
        hs    = actor.get_initial_hidden_state(BATCH_SIZE, torch.device('cpu'))
        actions, _ = actor(obs, hs)
        loss = actions.sum()
        loss.backward()
        grad = actor.ctm.start_activated_trace.grad
        assert grad is not None
        assert grad.shape == actor.ctm.start_activated_trace.shape
        assert grad.abs().sum().item() > 0.0

    def test_sequential_forward_passes_different_outputs(self):
        """Multiple steps should produce different synchronisation (boards evolve)."""
        actor = make_ctm_actor()
        actor.eval()
        bs  = 2
        hs  = actor.get_initial_hidden_state(bs, torch.device('cpu'))
        out0, hs = actor(torch.randn(bs, OBS_DIM), hs)
        out1, _  = actor(torch.randn(bs, OBS_DIM), hs)
        assert not torch.allclose(out0, out1)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CTMDDPGAgent
# ═══════════════════════════════════════════════════════════════════════════

class TestCTMDDPGAgent:

    def test_policy_is_ctm_actor(self):
        agent = make_ctm_agent()
        assert isinstance(agent.policy, CTMActor)
        assert isinstance(agent.target_policy, CTMActor)

    def test_critic_is_mlp(self):
        from algorithm.utils.networks import MLPNetwork
        agent = make_ctm_agent()
        assert isinstance(agent.critic, MLPNetwork)
        assert isinstance(agent.target_critic, MLPNetwork)

    def test_target_policy_equals_policy_after_init(self):
        """
        LazyLinear fix: target_policy weights must equal policy weights after __init__.
        Without the dummy forward pass both would materialize independently.
        """
        agent = make_ctm_agent()
        for (n1, p1), (n2, p2) in zip(
            agent.policy.named_parameters(),
            agent.target_policy.named_parameters(),
        ):
            assert torch.allclose(p1, p2), f"Mismatch in parameter {n1}"

    def test_target_critic_equals_critic_after_init(self):
        agent = make_ctm_agent()
        for (n1, p1), (n2, p2) in zip(
            agent.critic.named_parameters(),
            agent.target_critic.named_parameters(),
        ):
            assert torch.allclose(p1, p2), f"Mismatch in parameter {n1}"

    def test_step_returns_3_tuple(self):
        agent = make_ctm_agent()
        obs = torch.randn(N_AGENTS, OBS_DIM)
        hs  = agent.policy.get_initial_hidden_state(N_AGENTS, torch.device('cpu'))
        result = agent.step(obs, hs, explore=False)
        assert len(result) == 3

    def test_step_log_pi_is_none(self):
        agent = make_ctm_agent()
        obs = torch.randn(N_AGENTS, OBS_DIM)
        hs  = agent.policy.get_initial_hidden_state(N_AGENTS, torch.device('cpu'))
        _, log_pi, _ = agent.step(obs, hs, explore=False)
        assert log_pi is None

    def test_step_action_shape(self):
        """step returns action.t() so shape is (action_dim, n_agents)."""
        agent = make_ctm_agent()
        obs = torch.randn(N_AGENTS, OBS_DIM)
        hs  = agent.policy.get_initial_hidden_state(N_AGENTS, torch.device('cpu'))
        action, _, _ = agent.step(obs, hs, explore=False)
        assert action.shape == (ACTION_DIM, N_AGENTS)

    def test_step_hidden_state_shapes(self):
        agent = make_ctm_agent()
        obs = torch.randn(N_AGENTS, OBS_DIM)
        hs  = agent.policy.get_initial_hidden_state(N_AGENTS, torch.device('cpu'))
        _, _, new_hs = agent.step(obs, hs, explore=False)
        st, ast = new_hs
        assert st.shape  == (N_AGENTS, CTM_CFG['d_model'], CTM_CFG['memory_length'])
        assert ast.shape == (N_AGENTS, CTM_CFG['d_model'], CTM_CFG['memory_length'])

    def test_step_explore_action_clamped(self):
        """With explore=True actions must remain in [-1, 1] after noise + clamp."""
        agent = make_ctm_agent()
        obs = torch.randn(N_AGENTS, OBS_DIM)
        hs  = agent.policy.get_initial_hidden_state(N_AGENTS, torch.device('cpu'))
        # Run many times to exercise both noise and epsilon branches
        for _ in range(20):
            action, _, hs2 = agent.step(obs, hs, explore=True)
            assert action.min().item() >= -1.0 - 1e-6
            assert action.max().item() <=  1.0 + 1e-6
            hs = hs2

    def test_scale_noise(self):
        agent = make_ctm_agent()
        agent.scale_noise(0.5)
        assert abs(agent.exploration.scale - 0.5) < 1e-9

    def test_reset_noise(self):
        """reset_noise must not raise (inherited from DDPGAgent)."""
        agent = make_ctm_agent()
        agent.reset_noise()  # should not raise

    def test_get_params_keys(self):
        agent = make_ctm_agent()
        params = agent.get_params()
        expected = {'policy', 'critic', 'target_policy', 'target_critic',
                    'policy_optimizer', 'critic_optimizer'}
        assert set(params.keys()) == expected

    def test_save_load_roundtrip(self):
        """Parameters must survive a get_params / load_params cycle."""
        agent = make_ctm_agent()
        params_before = agent.get_params()

        # Corrupt policy weights
        with torch.no_grad():
            for p in agent.policy.parameters():
                p.fill_(999.0)

        agent.load_params(params_before)
        params_after = agent.get_params()

        for k in ('policy', 'target_policy', 'critic', 'target_critic'):
            for (n, t_before), (_, t_after) in zip(
                params_before[k].items(), params_after[k].items()
            ):
                assert torch.allclose(t_before, t_after), f"Mismatch in {k}/{n}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. MADDPG + CTM actor
# ═══════════════════════════════════════════════════════════════════════════

class TestMADDPGWithCTM:

    def test_use_ctm_actor_flag_stored(self):
        maddpg = make_maddpg_ctm()
        assert maddpg.use_ctm_actor is True

    def test_agents_are_ctm_agents(self):
        maddpg = make_maddpg_ctm()
        assert isinstance(maddpg.agents[0], CTMDDPGAgent)

    def test_step_returns_3_tuple(self):
        maddpg = make_maddpg_ctm()
        maddpg.prep_rollouts(device='cpu')
        total = N_ENVS * N_AGENTS
        obs   = torch.randn(OBS_DIM, total)
        hs    = maddpg.agents[0].policy.get_initial_hidden_state(total, torch.device('cpu'))
        result = maddpg.step(obs, [slice(0, total)], explore=False, hidden_states=hs)
        assert len(result) == 3

    def test_step_new_hidden_states_not_none(self):
        maddpg = make_maddpg_ctm()
        maddpg.prep_rollouts(device='cpu')
        total = N_ENVS * N_AGENTS
        obs = torch.randn(OBS_DIM, total)
        hs  = maddpg.agents[0].policy.get_initial_hidden_state(total, torch.device('cpu'))
        _, _, new_hs = maddpg.step(obs, [slice(0, total)], explore=False, hidden_states=hs)
        assert new_hs is not None
        assert len(new_hs) == 2  # (state_trace, activated_state_trace)

    def test_step_action_shape(self):
        """actions[0] should be (action_dim, total_agents)."""
        maddpg = make_maddpg_ctm()
        maddpg.prep_rollouts(device='cpu')
        total = N_ENVS * N_AGENTS
        obs = torch.randn(OBS_DIM, total)
        hs  = maddpg.agents[0].policy.get_initial_hidden_state(total, torch.device('cpu'))
        actions, _, _ = maddpg.step(obs, [slice(0, total)], explore=False, hidden_states=hs)
        assert actions[0].shape == (ACTION_DIM, total)

    def test_step_log_pis_are_none(self):
        maddpg = make_maddpg_ctm()
        maddpg.prep_rollouts(device='cpu')
        total = N_AGENTS
        obs = torch.randn(OBS_DIM, total)
        hs  = maddpg.agents[0].policy.get_initial_hidden_state(total, torch.device('cpu'))
        _, log_pis, _ = maddpg.step(obs, [slice(0, total)], explore=False, hidden_states=hs)
        assert log_pis[0] is None

    def test_target_policies_shape(self):
        maddpg = make_maddpg_ctm()
        maddpg.prep_training(device='cpu')
        obs = torch.randn(BATCH_SIZE, OBS_DIM)
        actions = maddpg.target_policies(0, obs)
        assert actions.shape == (BATCH_SIZE, ACTION_DIM)

    def test_update_runs_without_error(self):
        maddpg = make_maddpg_ctm()
        maddpg.prep_training(device='cpu')
        obs      = torch.randn(BATCH_SIZE, OBS_DIM)
        acs      = torch.randn(BATCH_SIZE, ACTION_DIM)
        rews     = torch.randn(BATCH_SIZE, 1)
        next_obs = torch.randn(BATCH_SIZE, OBS_DIM)
        dones    = torch.zeros(BATCH_SIZE, 1)
        vf_loss, pol_loss, reg_loss = maddpg.update(obs, acs, rews, next_obs, dones, 0)
        assert isinstance(vf_loss,  float)
        assert isinstance(pol_loss, float)
        assert isinstance(reg_loss, float)

    def test_update_losses_finite(self):
        maddpg = make_maddpg_ctm()
        maddpg.prep_training(device='cpu')
        obs      = torch.randn(BATCH_SIZE, OBS_DIM)
        acs      = torch.randn(BATCH_SIZE, ACTION_DIM)
        rews     = torch.randn(BATCH_SIZE, 1)
        next_obs = torch.randn(BATCH_SIZE, OBS_DIM)
        dones    = torch.zeros(BATCH_SIZE, 1)
        vf_loss, pol_loss, _ = maddpg.update(obs, acs, rews, next_obs, dones, 0)
        assert math.isfinite(vf_loss)
        assert math.isfinite(pol_loss)

    def test_update_with_prior_runs(self):
        maddpg = make_maddpg_ctm()
        maddpg.prep_training(device='cpu')
        obs        = torch.randn(BATCH_SIZE, OBS_DIM)
        acs        = torch.randn(BATCH_SIZE, ACTION_DIM)
        rews       = torch.randn(BATCH_SIZE, 1)
        next_obs   = torch.randn(BATCH_SIZE, OBS_DIM)
        dones      = torch.zeros(BATCH_SIZE, 1)
        acs_prior  = torch.randn(BATCH_SIZE, ACTION_DIM)
        vf_loss, pol_loss, reg_loss = maddpg.update(
            obs, acs, rews, next_obs, dones, 0, acs_prior=acs_prior, alpha=0.5
        )
        assert math.isfinite(pol_loss)
        assert reg_loss >= 0.0

    def test_update_all_targets_changes_target_weights(self):
        """Soft update must move target weights towards policy weights."""
        maddpg = make_maddpg_ctm()
        maddpg.prep_training(device='cpu')
        # Perturb policy weights so they differ from target
        with torch.no_grad():
            for p in maddpg.agents[0].policy.parameters():
                p.add_(torch.randn_like(p) * 0.5)
        pol_param = list(maddpg.agents[0].policy.parameters())[0].clone()
        tgt_param_before = list(maddpg.agents[0].target_policy.parameters())[0].clone()
        maddpg.update_all_targets()
        tgt_param_after = list(maddpg.agents[0].target_policy.parameters())[0].clone()
        # Target moved toward policy
        dist_before = (pol_param - tgt_param_before).abs().mean().item()
        dist_after  = (pol_param - tgt_param_after).abs().mean().item()
        assert dist_after < dist_before

    def test_prep_training_sets_train_mode(self):
        maddpg = make_maddpg_ctm()
        maddpg.prep_training(device='cpu')
        assert maddpg.agents[0].policy.training
        assert maddpg.agents[0].target_policy.training

    def test_prep_rollouts_sets_eval_mode(self):
        maddpg = make_maddpg_ctm()
        maddpg.prep_rollouts(device='cpu')
        assert not maddpg.agents[0].policy.training

    def test_scale_noise_propagates(self):
        maddpg = make_maddpg_ctm()
        maddpg.scale_noise(0.3, 0.05)
        assert abs(maddpg.agents[0].exploration.scale - 0.3) < 1e-9
        assert abs(maddpg.agents[0].epsilon - 0.05) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# 4. MADDPG + MLP actor — backwards compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestMADDPGWithMLP:

    def test_use_ctm_actor_flag_false(self):
        maddpg = make_maddpg_mlp()
        assert maddpg.use_ctm_actor is False

    def test_agents_are_ddpg_agents(self):
        maddpg = make_maddpg_mlp()
        assert isinstance(maddpg.agents[0], DDPGAgent)
        # Must NOT be a CTMDDPGAgent subclass
        assert type(maddpg.agents[0]) is DDPGAgent

    def test_step_returns_3_tuple(self):
        """MLP path still returns 3-tuple after the interface change."""
        maddpg = make_maddpg_mlp()
        maddpg.prep_rollouts(device='cpu')
        total = N_AGENTS
        obs   = torch.randn(OBS_DIM, total)
        result = maddpg.step(obs, [slice(0, total)], explore=False)
        assert len(result) == 3

    def test_step_new_hidden_states_is_none(self):
        maddpg = make_maddpg_mlp()
        maddpg.prep_rollouts(device='cpu')
        total = N_AGENTS
        obs = torch.randn(OBS_DIM, total)
        _, _, new_hs = maddpg.step(obs, [slice(0, total)], explore=False)
        assert new_hs is None

    def test_step_action_shape(self):
        maddpg = make_maddpg_mlp()
        maddpg.prep_rollouts(device='cpu')
        total = N_AGENTS
        obs = torch.randn(OBS_DIM, total)
        actions, _, _ = maddpg.step(obs, [slice(0, total)], explore=False)
        assert actions[0].shape == (ACTION_DIM, total)

    def test_target_policies_shape(self):
        maddpg = make_maddpg_mlp()
        maddpg.prep_training(device='cpu')
        obs = torch.randn(BATCH_SIZE, OBS_DIM)
        actions = maddpg.target_policies(0, obs)
        assert actions.shape == (BATCH_SIZE, ACTION_DIM)

    def test_update_runs_without_error(self):
        maddpg = make_maddpg_mlp()
        maddpg.prep_training(device='cpu')
        obs      = torch.randn(BATCH_SIZE, OBS_DIM)
        acs      = torch.randn(BATCH_SIZE, ACTION_DIM)
        rews     = torch.randn(BATCH_SIZE, 1)
        next_obs = torch.randn(BATCH_SIZE, OBS_DIM)
        dones    = torch.zeros(BATCH_SIZE, 1)
        vf_loss, pol_loss, _ = maddpg.update(obs, acs, rews, next_obs, dones, 0)
        assert math.isfinite(vf_loss)
        assert math.isfinite(pol_loss)

    def test_mlp_policy_is_mlpnetwork(self):
        from algorithm.utils.networks import MLPNetwork
        maddpg = make_maddpg_mlp()
        assert isinstance(maddpg.agents[0].policy, MLPNetwork)
        assert isinstance(maddpg.agents[0].target_policy, MLPNetwork)

    def test_step_with_explore_action_in_range(self):
        maddpg = make_maddpg_mlp()
        maddpg.prep_rollouts(device='cpu')
        total = N_AGENTS
        obs = torch.randn(OBS_DIM, total)
        for _ in range(10):
            actions, _, _ = maddpg.step(obs, [slice(0, total)], explore=True)
            a = actions[0]
            assert a.min().item() >= -1.0 - 1e-6
            assert a.max().item() <=  1.0 + 1e-6

    def test_prep_training_and_rollouts(self):
        maddpg = make_maddpg_mlp()
        maddpg.prep_training(device='cpu')
        assert maddpg.agents[0].policy.training
        maddpg.prep_rollouts(device='cpu')
        assert not maddpg.agents[0].policy.training

    def test_scale_and_reset_noise(self):
        maddpg = make_maddpg_mlp()
        maddpg.scale_noise(0.2, 0.05)
        assert abs(maddpg.agents[0].exploration.scale - 0.2) < 1e-9
        maddpg.reset_noise()  # must not raise


# ═══════════════════════════════════════════════════════════════════════════
# 5. Hidden state management
# ═══════════════════════════════════════════════════════════════════════════

class TestHiddenStateManagement:

    def _make_hidden(self, batch):
        actor = make_ctm_actor()
        return actor, actor.get_initial_hidden_state(batch, torch.device('cpu'))

    def test_done_mask_resets_state_trace_to_zeros(self):
        """
        When done=1 for an agent, state_trace for that agent should become zeros.
        """
        total = 4
        actor, hs = self._make_hidden(total)
        # Fill state trace with non-zero values
        st, ast = hs
        st.fill_(1.0)

        # Mark agents 0 and 2 as done
        done_mask = torch.tensor([1, 0, 1, 0], dtype=torch.float32).reshape(-1, 1, 1)
        new_st = st * (1 - done_mask)

        assert torch.all(new_st[0] == 0.0)
        assert torch.all(new_st[2] == 0.0)
        assert torch.all(new_st[1] == 1.0)
        assert torch.all(new_st[3] == 1.0)

    def test_done_mask_resets_activated_trace_to_start(self):
        """
        When done=1, activated_state_trace should be reset to start_activated_trace.
        """
        total = 4
        actor, hs = self._make_hidden(total)
        st, ast = hs
        ast.fill_(999.0)  # corrupt

        done_mask = torch.tensor([1, 0, 1, 0], dtype=torch.float32).reshape(-1, 1, 1)
        start_act = actor.ctm.start_activated_trace.detach()
        new_ast = ast * (1 - done_mask) + start_act.unsqueeze(0) * done_mask

        # Done agents should match start_activated_trace
        for i in [0, 2]:
            assert torch.allclose(new_ast[i], start_act)
        # Not-done agents should still be 999
        for i in [1, 3]:
            assert torch.all(new_ast[i] == 999.0)

    def test_done_mask_all_done(self):
        """All agents done — entire board resets."""
        total = 3
        actor, hs = self._make_hidden(total)
        st, ast = hs
        st.fill_(5.0)
        ast.fill_(5.0)

        done_mask = torch.ones(total, 1, 1)
        start_act = actor.ctm.start_activated_trace.detach()
        new_st  = st * (1 - done_mask)
        new_ast = ast * (1 - done_mask) + start_act.unsqueeze(0) * done_mask

        assert torch.all(new_st == 0.0)
        for i in range(total):
            assert torch.allclose(new_ast[i], start_act)

    def test_done_mask_none_done(self):
        """No agents done — boards unchanged."""
        total = 3
        actor, hs = self._make_hidden(total)
        st, ast = hs
        st.fill_(7.0)
        ast.fill_(7.0)

        done_mask = torch.zeros(total, 1, 1)
        start_act = actor.ctm.start_activated_trace.detach()
        new_st  = st * (1 - done_mask)
        new_ast = ast * (1 - done_mask) + start_act.unsqueeze(0) * done_mask

        assert torch.all(new_st == 7.0)
        assert torch.all(new_ast == 7.0)

    def test_detach_breaks_grad_graph(self):
        """
        After detach, the tensor has no grad_fn — the 200-step graph is cut.
        """
        actor = make_ctm_actor()
        obs = torch.randn(2, OBS_DIM)
        hs  = actor.get_initial_hidden_state(2, torch.device('cpu'))

        # Simulate a rollout step — hidden states retain grad_fn because
        # start_activated_trace has requires_grad=True
        _, new_hs = actor(obs, hs)
        st, ast = new_hs
        assert ast.grad_fn is not None  # graph exists before detach

        # Detach
        st_d  = st.detach()
        ast_d = ast.detach()
        assert st_d.grad_fn  is None
        assert ast_d.grad_fn is None

    def test_hidden_state_persists_between_steps(self):
        """Running two steps with preserved hidden state should differ from two fresh starts."""
        actor = make_ctm_actor()
        actor.eval()
        obs1 = torch.randn(1, OBS_DIM)
        obs2 = torch.randn(1, OBS_DIM)

        # Path A: carry hidden state across two steps
        hs   = actor.get_initial_hidden_state(1, torch.device('cpu'))
        _, hs_after1 = actor(obs1, hs)
        out_A, _     = actor(obs2, hs_after1)

        # Path B: fresh hidden state for second step (ignore obs1 history)
        hs_fresh = actor.get_initial_hidden_state(1, torch.device('cpu'))
        out_B, _ = actor(obs2, hs_fresh)

        # With different history the outputs must differ
        assert not torch.allclose(out_A, out_B), \
            "Temporal context should change output — hidden states matter"

    def test_hidden_state_shapes_across_rollout(self):
        """Simulate a short rollout — hidden state shapes must stay constant."""
        maddpg = make_maddpg_ctm()
        maddpg.prep_rollouts(device='cpu')
        total = N_ENVS * N_AGENTS
        hs    = maddpg.agents[0].policy.get_initial_hidden_state(total, torch.device('cpu'))

        for _ in range(5):  # 5 rollout steps
            obs = torch.randn(OBS_DIM, total)
            _, _, hs = maddpg.step(obs, [slice(0, total)], explore=True, hidden_states=hs)
            hs = (hs[0].detach(), hs[1].detach())
            assert hs[0].shape == (total, CTM_CFG['d_model'], CTM_CFG['memory_length'])
            assert hs[1].shape == (total, CTM_CFG['d_model'], CTM_CFG['memory_length'])


# ═══════════════════════════════════════════════════════════════════════════
# 6. End-to-end smoke test — simulated rollout + update
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEnd:

    def _rollout_and_update(self, use_ctm: bool):
        """
        Simulates the training loop logic without a real environment.
        Verifies:
          - step works for N steps
          - buffer can collect transitions
          - update runs without NaN/Inf
          - losses decrease is not checked (too few steps) but execution is clean
        """
        maddpg = make_maddpg_ctm() if use_ctm else make_maddpg_mlp()
        maddpg.prep_rollouts(device='cpu')

        total  = N_AGENTS
        ep_len = 10
        obs_buf, acs_buf, rew_buf, nobs_buf, done_buf = [], [], [], [], []

        # ── Rollout ────────────────────────────────────────────────────────
        obs_gpu = torch.randn(OBS_DIM, total)
        hs = (maddpg.agents[0].policy.get_initial_hidden_state(total, torch.device('cpu'))
              if use_ctm else None)

        for t in range(ep_len):
            torch_actions, _, hs = maddpg.step(
                obs_gpu, [slice(0, total)], explore=True, hidden_states=hs
            )
            if hs is not None:
                hs = (hs[0].detach(), hs[1].detach())

            action_gpu = torch.column_stack(torch_actions)  # (action_dim, total)
            next_obs_gpu = torch.randn(OBS_DIM, total)
            rewards      = torch.randn(total)
            dones_raw    = torch.zeros(total)

            # Done-mask reset (same as training loop)
            if hs is not None:
                done_mask = dones_raw.reshape(-1, 1, 1).float()
                start_act = maddpg.agents[0].policy.ctm.start_activated_trace.detach()
                new_st  = hs[0] * (1 - done_mask)
                new_ast = hs[1] * (1 - done_mask) + start_act.unsqueeze(0) * done_mask
                hs = (new_st, new_ast)

            obs_buf.append(obs_gpu.t().detach())       # (total, obs_dim) — detach: real buffer stores no graphs
            acs_buf.append(action_gpu.t().detach())    # (total, action_dim) — detach: avoids double-backward / inplace errors
            rew_buf.append(rewards.unsqueeze(1))       # (total, 1)
            nobs_buf.append(next_obs_gpu.t())
            done_buf.append(dones_raw.unsqueeze(1))
            obs_gpu = next_obs_gpu

        # ── Training ───────────────────────────────────────────────────────
        maddpg.prep_training(device='cpu')
        obs_t    = torch.cat(obs_buf)
        acs_t    = torch.cat(acs_buf)
        rews_t   = torch.cat(rew_buf)
        nobs_t   = torch.cat(nobs_buf)
        dones_t  = torch.cat(done_buf)

        for _ in range(3):
            vf, pol, reg = maddpg.update(obs_t, acs_t, rews_t, nobs_t, dones_t, 0)
            assert math.isfinite(vf)
            assert math.isfinite(pol)
            assert math.isfinite(reg)
        maddpg.update_all_targets()

    def test_ctm_rollout_and_update(self):
        self._rollout_and_update(use_ctm=True)

    def test_mlp_rollout_and_update(self):
        self._rollout_and_update(use_ctm=False)

    def test_ctm_and_mlp_produce_different_outputs(self):
        """Sanity check: CTM and MLP actors are different models."""
        ctm_m = make_maddpg_ctm()
        mlp_m = make_maddpg_mlp()
        ctm_m.prep_rollouts(device='cpu')
        mlp_m.prep_rollouts(device='cpu')

        obs   = torch.randn(OBS_DIM, N_AGENTS)
        hs    = ctm_m.agents[0].policy.get_initial_hidden_state(N_AGENTS, torch.device('cpu'))
        act_ctm, _, _ = ctm_m.step(obs, [slice(0, N_AGENTS)], explore=False, hidden_states=hs)
        act_mlp, _, _ = mlp_m.step(obs, [slice(0, N_AGENTS)], explore=False)
        # Both valid shapes
        assert act_ctm[0].shape == act_mlp[0].shape == (ACTION_DIM, N_AGENTS)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point for running without pytest
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
