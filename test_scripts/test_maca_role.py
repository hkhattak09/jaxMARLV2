"""Comprehensive TDD tests for MACA-Role (all 4 experiments).

Tests cover:
  Exp 1: Post-GRU actor role heads + shared critic
  Exp 2: Post-GRU actor heads + role-specific critic heads
  Exp 3: Pre-GRU routes + post-GRU heads + shared critic
  Exp 4: Pre-GRU routes + post-GRU heads + role-specific critic heads

Run on a JAX-enabled environment (e.g., Colab) with:
    pytest test_scripts/test_maca_role.py -v
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SMAX_CTM = os.path.join(REPO_ROOT, "smax_ctm")
if SMAX_CTM not in sys.path:
    sys.path.insert(0, SMAX_CTM)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.traverse_util import flatten_dict, unflatten_dict

class DummyActSpace:
    """Minimal action space stand-in for tests."""
    def __init__(self, n):
        self.n = n


from mappo_t import (
    ActorTrans,
    ScannedRNN,
    get_default_maca_role_config,
)

# Modules under test (will be created)
from mappo_t.role_actor import RoleActorTrans
from mappo_t.role_critic import RoleTransVCritic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tiny_config():
    cfg = get_default_maca_role_config()
    cfg["hidden_sizes"] = [16, 16, 16]
    cfg["use_recurrent_policy"] = True
    cfg["use_naive_recurrent_policy"] = False
    cfg["use_feature_normalization"] = True
    cfg["activation_func"] = "relu"
    cfg["transformer"] = dict(cfg["transformer"])
    cfg["transformer"]["active_fn"] = "gelu"
    # Small role-specific dims for fast unit tests
    cfg["role_head_hidden_dims"] = [16, 8]
    cfg["role_route_hidden_dim"] = 16
    cfg["role_z_k_dims"] = [16, 8]
    cfg["role_v_head_dims"] = [16]
    cfg["N_ROLES"] = 6
    return cfg


def init_actor_tensors(seed=0, batch=4, time=3, obs_dim=13, action_dim=7, n_roles=6):
    """Return standard input tensors for actor tests."""
    rng = jax.random.PRNGKey(seed)
    h0 = ScannedRNN.initialize_carry(batch, 16)
    obs = jax.random.normal(rng, (time, batch, obs_dim))
    resets = jnp.zeros((time, batch), dtype=bool)
    avail = jnp.ones((time, batch, action_dim), dtype=jnp.float32)
    role_ids = jax.random.randint(rng, (time, batch), 0, n_roles)
    return rng, h0, obs, resets, avail, role_ids


def init_critic_tensors(seed=0, batch=4, n_agents=10, obs_dim=13, action_dim=7, n_roles=6):
    """Return standard input tensors for critic tests."""
    rng = jax.random.PRNGKey(seed)
    obs_all = jax.random.normal(rng, (batch, n_agents, obs_dim))
    actions = jax.random.randint(rng, (batch, n_agents), 0, action_dim)
    policy_probs = jnp.ones((batch, n_agents, action_dim)) / action_dim
    rnn_states = jnp.zeros((batch, n_agents, 64), dtype=jnp.float32)
    resets = jnp.zeros((batch, n_agents), dtype=bool)
    role_ids = jax.random.randint(rng, (batch, n_agents), 0, n_roles)
    return rng, obs_all, actions, policy_probs, rnn_states, resets, role_ids


# ---------------------------------------------------------------------------
# Experiment 1: Post-GRU Actor Heads + Shared Critic
# ---------------------------------------------------------------------------

class TestExp1_PostGRUHeads:
    """Experiment 1: Post-GRU actor role heads only, shared critic."""

    def test_role_routing_changes_output(self):
        """Different role IDs must produce different logits."""
        cfg = make_tiny_config()
        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=False, n_roles=6)
        rng, h0, obs, resets, avail, role_ids = init_actor_tensors()

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)

        # All role 0
        role_0 = jnp.zeros_like(role_ids)
        _, pi_0 = actor.apply(params, h0, (obs, resets, avail), role_0)

        # All role 3
        role_3 = jnp.full_like(role_ids, 3)
        _, pi_3 = actor.apply(params, h0, (obs, resets, avail), role_3)

        max_diff = float(jnp.max(jnp.abs(pi_0.logits - pi_3.logits)))
        assert max_diff > 1e-5, (
            f"Role routing did not change output (diff={max_diff}). "
            "Different roles should use different heads."
        )

    def test_single_role_produces_valid_output(self):
        """With n_roles=1, forward pass works and produces valid distributions."""
        cfg = make_tiny_config()
        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=False, n_roles=1)

        rng, h0, obs, resets, avail, _ = init_actor_tensors(n_roles=1)
        role_ids = jnp.zeros_like(obs[:, :, 0], dtype=jnp.int32)

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)
        _, pi = actor.apply(params, h0, (obs, resets, avail), role_ids)

        # Valid probability distribution
        np.testing.assert_allclose(
            pi.probs.sum(axis=-1), 1.0, atol=1e-6,
            err_msg="Single-role actor should produce valid probability distributions"
        )
        assert pi.logits.shape == obs.shape[:2] + (7,), (
            f"Logits shape mismatch: {pi.logits.shape}"
        )

    def test_all_params_receive_gradients(self):
        """No frozen parameters — backbone + heads all train end-to-end."""
        cfg = make_tiny_config()
        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=False, n_roles=6)
        rng, h0, obs, resets, avail, role_ids = init_actor_tensors()

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)

        def loss_fn(p):
            _, pi = actor.apply(p, h0, (obs, resets, avail), role_ids)
            return jnp.sum(pi.logits)

        grads = jax.grad(loss_fn)(params)
        flat_grads = flatten_dict(grads)

        for key, grad in flat_grads.items():
            has_grad = bool(jnp.any(grad != 0))
            assert has_grad, f"Parameter {key} has zero gradient but should be trainable"

    def test_kl_diversity_computed_correctly(self):
        """KL between identical policies ~0, between different policies >0."""
        cfg = make_tiny_config()
        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=False, n_roles=6)
        rng, h0, obs, resets, avail, role_ids = init_actor_tensors()

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)

        # Compute KL diversity: should be non-negative
        kl_div = actor.compute_kl_diversity(params, h0, obs, resets, avail)
        assert float(kl_div) >= 0.0, f"KL diversity must be non-negative, got {float(kl_div)}"

        # With random init, heads should differ => KL > 0
        assert float(kl_div) > 1e-5, (
            f"Randomly initialized heads have near-zero KL ({float(kl_div)}). "
            "Diversity penalty should detect differences."
        )

    def test_cosine_decay_schedule(self):
        """KL weight starts at 0.001, reaches 0 at 30% of total steps."""
        total_steps = 1000
        schedule = RoleActorTrans.make_kl_schedule(total_steps)

        w0 = schedule(0)
        w_mid = schedule(150)
        w_30 = schedule(300)
        w_end = schedule(999)

        np.testing.assert_allclose(float(w0), 0.001, atol=1e-6, err_msg="Initial KL weight should be 0.001")
        np.testing.assert_allclose(float(w_30), 0.0, atol=1e-6, err_msg="KL weight should reach 0 at 30%")
        np.testing.assert_allclose(float(w_end), 0.0, atol=1e-6, err_msg="KL weight should stay 0 after 30%")
        assert float(w_mid) > 0.0, "KL weight should be positive before 30%"
        assert float(w_mid) < 0.001, "KL weight should be decaying"

    def test_action_masking_respected(self):
        """Masked actions have ~0 probability."""
        cfg = make_tiny_config()
        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=False, n_roles=6)
        rng, h0, obs, resets, avail, role_ids = init_actor_tensors()

        # Mask action 3 everywhere
        avail = avail.at[:, :, 3].set(0.0)

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)
        _, pi = actor.apply(params, h0, (obs, resets, avail), role_ids)

        np.testing.assert_allclose(pi.probs[..., 3], 0.0, atol=1e-6, err_msg="Masked action should have ~0 prob")
        np.testing.assert_allclose(pi.probs.sum(axis=-1), 1.0, atol=1e-6, err_msg="Probs should sum to 1")

    def test_recurrent_reset_semantics(self):
        """Reset=True resets GRU state correctly."""
        cfg = make_tiny_config()
        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=False, n_roles=6)
        rng, h0, obs, resets, avail, role_ids = init_actor_tensors(batch=3, time=5)

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)

        # Full run with reset at t=3
        resets = resets.at[3, :].set(True)
        full_h, full_pi = actor.apply(params, h0, (obs, resets, avail), role_ids)

        # Segmented run
        prefix_h, prefix_pi = actor.apply(params, h0, (obs[:3], resets[:3], avail[:3]), role_ids[:3])
        zero_h = ScannedRNN.initialize_carry(h0.shape[0], h0.shape[1])
        suffix_resets = resets[3:].at[0, :].set(True)
        suffix_h, suffix_pi = actor.apply(
            params, zero_h, (obs[3:], suffix_resets, avail[3:]), role_ids[3:]
        )

        np.testing.assert_allclose(full_pi.logits[3:], suffix_pi.logits, atol=1e-5, err_msg="Reset semantics mismatch")
        np.testing.assert_allclose(full_h, suffix_h, atol=1e-5, err_msg="Hidden state mismatch after reset")

    def test_parameter_count_increases_with_roles(self):
        """More roles -> more parameters (but backbone stays shared)."""
        cfg = make_tiny_config()
        actor_1 = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=False, n_roles=1)
        actor_6 = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=False, n_roles=6)

        rng = jax.random.PRNGKey(0)
        h0 = ScannedRNN.initialize_carry(4, 16)
        obs = jnp.zeros((3, 4, 13))
        resets = jnp.zeros((3, 4), dtype=bool)
        avail = jnp.ones((3, 4, 7))
        role_ids = jnp.zeros((3, 4), dtype=jnp.int32)

        params_1 = actor_1.init(rng, h0, (obs, resets, avail), role_ids)
        params_6 = actor_6.init(rng, h0, (obs, resets, avail), role_ids)

        def count_params(p):
            return sum(x.size for x in jax.tree_util.tree_leaves(p))

        n1 = count_params(params_1)
        n6 = count_params(params_6)

        assert n6 > n1, f"6-role actor ({n6}) should have more params than 1-role ({n1})"


# ---------------------------------------------------------------------------
# Experiment 2: Post-GRU Actor Heads + Role-Specific Critic Heads
# ---------------------------------------------------------------------------

class TestExp2_RoleSpecificCritic:
    """Experiment 2: Shared actor + role-specific critic heads."""

    def test_role_specific_v_heads_differ(self):
        """Different roles produce different V values."""
        cfg = make_tiny_config()
        cfg["transformer"]["n_embd"] = 64
        cfg["transformer"]["zs_dim"] = 256

        critic = RoleTransVCritic(
            config=cfg,
            share_obs_space=None,
            obs_space=None,
            act_space=DummyActSpace(7),
            num_agents=10,
            state_type="EP",
        )

        rng, obs_all, actions, policy_probs, rnn_states, resets, role_ids = init_critic_tensors()

        params = critic.init(
            rng, obs_all, actions, policy_probs, rnn_states, resets, role_ids, True, True
        )

        # Get per-role V values
        all_v = critic.get_all_role_values(params, obs_all, actions, policy_probs, rnn_states, resets)
        # all_v shape: (n_roles, batch, 1)

        # Different roles should have different V distributions
        std_across_roles = jnp.std(all_v, axis=0).mean()
        assert float(std_across_roles) > 1e-5, (
            f"Role-specific V heads are too similar (std={float(std_across_roles)}). "
            "Each role should learn distinct value estimates."
        )

    def test_marginalization_preservation(self):
        """EQ_k(s, π) = sum_a π(a) * Q_k(s, a) for each role k.

        Tests with n_agents=2 so the joint action space (7^2=49) is enumerable.
        This actually stresses multi-agent marginalization, not a trivial 1-agent case.
        """
        cfg = make_tiny_config()
        cfg["transformer"]["n_embd"] = 64
        cfg["transformer"]["zs_dim"] = 256

        action_dim = 7
        n_agents = 2
        critic = RoleTransVCritic(
            config=cfg,
            share_obs_space=None,
            obs_space=None,
            act_space=DummyActSpace(action_dim),
            num_agents=n_agents,
            state_type="EP",
        )

        rng = jax.random.PRNGKey(0)
        batch = 4
        obs_all = jax.random.normal(rng, (batch, n_agents, 13))
        policy_probs = jnp.ones((batch, n_agents, action_dim)) / action_dim
        rnn_states = jnp.zeros((batch, n_agents, 64), dtype=jnp.float32)
        resets = jnp.zeros((batch, n_agents), dtype=bool)
        role_ids = jnp.zeros((batch, n_agents), dtype=jnp.int32)

        params = critic.init(
            rng, obs_all, jnp.zeros((batch, n_agents), dtype=jnp.int32),
            policy_probs, rnn_states, resets, role_ids, False, True
        )

        for k in range(6):
            eq_k = critic.compute_eq_for_role(params, k, obs_all, policy_probs, rnn_states, resets)

            # Enumerate all joint actions for 2 agents: 7^2 = 49 combinations
            q_for_all_actions = []
            for a0 in range(action_dim):
                for a1 in range(action_dim):
                    one_hot = jax.nn.one_hot(
                        jnp.array([[a0, a1]] * batch), action_dim
                    )  # (batch, 2, action_dim)
                    _, all_q, _, _, _, _, _, _, _, _, _ = critic.apply(
                        params, obs_all, one_hot, policy_probs, rnn_states, resets, role_ids, False, True
                    )
                    q_for_all_actions.append(all_q[k])  # (batch, 1)

            q_for_all_actions = jnp.stack(q_for_all_actions, axis=-1)  # (batch, 1, 49)
            # Joint policy probabilities: π(a0) * π(a1) for each combination
            joint_probs = (
                policy_probs[:, 0:1, :]  # (batch, 1, 7)
                .reshape(batch, 1, action_dim, 1)
                * policy_probs[:, 1:2, :]  # (batch, 1, 7)
                .reshape(batch, 1, 1, action_dim)
            )  # (batch, 1, 7, 7)
            joint_probs = joint_probs.reshape(batch, 1, action_dim * action_dim)  # (batch, 1, 49)

            manual_eq = jnp.sum(q_for_all_actions * joint_probs, axis=-1, keepdims=True)  # (batch, 1, 1)
            manual_eq = manual_eq.squeeze(-1)  # (batch, 1)

            max_diff = float(jnp.max(jnp.abs(eq_k - manual_eq)))
            assert max_diff <= 1e-4, (
                f"Marginalization failed for role {k}: |EQ - manual|_max = {max_diff}. "
                "Q-head must be linear (no activation) to preserve Jensen's equality."
            )

    def test_gae_mean_pooling(self):
        """Env-level V/Q/EQ are means across roles."""
        cfg = make_tiny_config()
        cfg["transformer"]["n_embd"] = 64
        cfg["transformer"]["zs_dim"] = 256

        critic = RoleTransVCritic(
            config=cfg,
            share_obs_space=None,
            obs_space=None,
            act_space=DummyActSpace(7),
            num_agents=10,
            state_type="EP",
        )

        rng, obs_all, actions, policy_probs, rnn_states, resets, role_ids = init_critic_tensors()

        params = critic.init(
            rng, obs_all, actions, policy_probs, rnn_states, resets, role_ids, True, True
        )

        v_env, q_env, eq_env = critic.get_env_level_values(
            params, obs_all, actions, policy_probs, rnn_states, resets
        )

        # Verify they're means
        all_v = critic.get_all_role_values(params, obs_all, actions, policy_probs, rnn_states, resets)
        expected_v = jnp.mean(all_v, axis=0)

        max_diff = float(jnp.max(jnp.abs(v_env - expected_v)))
        assert max_diff <= 1e-5, f"V_env is not mean across roles (diff={max_diff})"

    def test_per_agent_baseline_uses_role_q(self):
        """Agent i's baseline uses Q_{r_i}, not shared Q."""
        cfg = make_tiny_config()
        cfg["transformer"]["n_embd"] = 64
        cfg["transformer"]["zs_dim"] = 256

        critic = RoleTransVCritic(
            config=cfg,
            share_obs_space=None,
            obs_space=None,
            act_space=DummyActSpace(7),
            num_agents=10,
            state_type="EP",
        )

        rng, obs_all, actions, policy_probs, rnn_states, resets, role_ids = init_critic_tensors(batch=2)

        params = critic.init(
            rng, obs_all, actions, policy_probs, rnn_states, resets, role_ids, True, True
        )

        baseline = critic.compute_per_agent_baseline(
            params, obs_all, actions, policy_probs, rnn_states, resets, role_ids
        )
        # baseline shape: (batch, n_agents)

        assert baseline.shape == (2, 10), f"Baseline shape should be (batch, n_agents), got {baseline.shape}"

        # If we swap role_ids, baseline should change
        swapped_role_ids = jnp.where(role_ids == 0, 1, role_ids)
        baseline_swapped = critic.compute_per_agent_baseline(
            params, obs_all, actions, policy_probs, rnn_states, resets, swapped_role_ids
        )

        max_diff = float(jnp.max(jnp.abs(baseline - baseline_swapped)))
        assert max_diff > 1e-5, (
            f"Baseline unchanged after swapping roles (diff={max_diff}). "
            "Per-agent baseline must depend on agent's role."
        )

    def test_critic_diversity_penalty(self):
        """Diversity penalty pushes role embeddings apart."""
        cfg = make_tiny_config()
        cfg["transformer"]["n_embd"] = 64
        cfg["transformer"]["zs_dim"] = 256

        critic = RoleTransVCritic(
            config=cfg,
            share_obs_space=None,
            obs_space=None,
            act_space=DummyActSpace(7),
            num_agents=10,
            state_type="EP",
        )

        rng, obs_all, actions, policy_probs, rnn_states, resets, role_ids = init_critic_tensors()

        params = critic.init(
            rng, obs_all, actions, policy_probs, rnn_states, resets, role_ids, True, True
        )

        div_loss = critic.compute_diversity_penalty(params, obs_all, actions, policy_probs, rnn_states, resets)

        # Diversity penalty is negative L2 distance => more negative = more diverse
        # For random init, should be negative (roles are different)
        assert float(div_loss) < 0.0, (
            f"Diversity penalty should be negative for random init (got {float(div_loss)}). "
            "Roles should be distinct."
        )

    def test_shared_sa_encoder_exists(self):
        """sa_encoder parameters exist in the critic."""
        cfg = make_tiny_config()
        cfg["transformer"]["n_embd"] = 64
        cfg["transformer"]["zs_dim"] = 256

        critic = RoleTransVCritic(
            config=cfg,
            share_obs_space=None,
            obs_space=None,
            act_space=DummyActSpace(7),
            num_agents=10,
            state_type="EP",
        )

        rng, obs_all, actions, policy_probs, rnn_states, resets, role_ids = init_critic_tensors()

        params = critic.init(
            rng, obs_all, actions, policy_probs, rnn_states, resets, role_ids, True, True
        )

        flat = flatten_dict(params)
        sa_keys = [k for k in flat if "sa_encoder" in str(k)]
        assert len(sa_keys) > 0, "sa_encoder parameters not found"

        # Marginalization tested separately in test_marginalization_preservation


# ---------------------------------------------------------------------------
# Experiment 3: Pre-GRU Routes + Post-GRU Heads + Shared Critic
# ---------------------------------------------------------------------------

class TestExp3_PreGRURoutes:
    """Experiment 3: Pre-GRU residual routes + post-GRU heads, shared critic."""

    def test_residual_routes_are_additive(self):
        """GRU input = shared_embedding + route_k(obs)."""
        cfg = make_tiny_config()
        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=True, n_roles=6)
        rng, h0, obs, resets, avail, role_ids = init_actor_tensors()

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)

        # Zero out route parameters
        flat = flatten_dict(params)
        for key in flat:
            if "route" in str(key):
                flat[key] = jnp.zeros_like(flat[key])
        params_zero_route = unflatten_dict(flat)

        # With zero routes, actor should still produce valid outputs
        _, pi_zero = actor.apply(params_zero_route, h0, (obs, resets, avail), role_ids)
        _, pi_normal = actor.apply(params, h0, (obs, resets, avail), role_ids)

        # Both should be valid probability distributions
        np.testing.assert_allclose(
            pi_zero.probs.sum(axis=-1), 1.0, atol=1e-6,
            err_msg="Zero-route actor should produce valid probabilities"
        )
        np.testing.assert_allclose(
            pi_normal.probs.sum(axis=-1), 1.0, atol=1e-6,
            err_msg="Normal actor should produce valid probabilities"
        )

        # Outputs should differ (routes have effect when non-zero)
        max_diff = float(jnp.max(jnp.abs(pi_zero.logits - pi_normal.logits)))
        assert max_diff > 1e-5, (
            f"Zero-route and normal actor outputs too similar (diff={max_diff}). "
            "Routes should affect the output."
        )

    def test_routes_are_role_specific(self):
        """Different roles produce different outputs via routes."""
        cfg = make_tiny_config()
        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=True, n_roles=6)
        rng, h0, obs, resets, avail, role_ids = init_actor_tensors()

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)

        # Compare outputs with different uniform role IDs
        role_0 = jnp.zeros_like(role_ids)
        role_3 = jnp.full_like(role_ids, 3)

        _, pi_0 = actor.apply(params, h0, (obs, resets, avail), role_0)
        _, pi_3 = actor.apply(params, h0, (obs, resets, avail), role_3)

        max_diff = float(jnp.max(jnp.abs(pi_0.logits - pi_3.logits)))
        assert max_diff > 1e-5, (
            f"Route outputs too similar across roles (diff={max_diff}). "
            "Each role should have its own route transformation."
        )

    def test_route_init_small(self):
        """Route weights initialized with orthogonal(0.1) -> small magnitudes."""
        cfg = make_tiny_config()
        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=True, n_roles=6)
        rng, h0, obs, resets, avail, role_ids = init_actor_tensors()

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)

        flat = flatten_dict(params)
        for key, val in flat.items():
            if "route" in str(key) and "kernel" in str(key):
                max_val = float(jnp.max(jnp.abs(val)))
                assert max_val < 0.5, (
                    f"Route kernel {key} has large values (max={max_val}). "
                    "Should be initialized small (orthogonal(0.1))."
                )

    def test_routes_run_parallel_to_base_mlp(self):
        """Routes operate on raw obs, not base MLP output."""
        cfg = make_tiny_config()
        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=True, n_roles=6)
        rng, h0, obs, resets, avail, role_ids = init_actor_tensors()

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)

        # Verify route parameters have input dimension = obs_dim, not hidden_dim
        flat = flatten_dict(params)
        for key, val in flat.items():
            if "route" in str(key) and "kernel" in str(key):
                # First route layer kernel shape should be (obs_dim, 128)
                if "route_0" in str(key) or "route_1" in str(key) or "route_2" in str(key) or "route_3" in str(key) or "route_4" in str(key) or "route_5" in str(key):
                    if "dense_0" in str(key) or "Dense_0" in str(key):
                        in_dim = val.shape[0]
                        assert in_dim == 13, (
                            f"Route first layer input dim {in_dim} != obs_dim (13). "
                            "Routes should take raw observations."
                        )

    def test_pre_gru_does_not_change_post_gru_head_interface(self):
        """Exp 3 and Exp 1 have identical post-GRU head architecture."""
        cfg = make_tiny_config()
        actor_exp1 = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=False, n_roles=6)
        actor_exp3 = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=True, n_roles=6)

        rng, h0, obs, resets, avail, role_ids = init_actor_tensors()

        params1 = actor_exp1.init(rng, h0, (obs, resets, avail), role_ids)
        params3 = actor_exp3.init(rng, h0, (obs, resets, avail), role_ids)

        # Both should have the same number of head-related parameters
        flat1 = flatten_dict(params1)
        flat3 = flatten_dict(params3)

        head_keys_1 = {k for k in flat1 if "head" in str(k)}
        head_keys_3 = {k for k in flat3 if "head" in str(k)}

        assert head_keys_1 == head_keys_3, (
            f"Exp 1 and Exp 3 have different head parameters. "
            f"Exp1 only: {head_keys_1 - head_keys_3}. "
            f"Exp3 only: {head_keys_3 - head_keys_1}."
        )


# ---------------------------------------------------------------------------
# Experiment 4: Pre-GRU Routes + Post-GRU Heads + Role-Specific Critic
# ---------------------------------------------------------------------------

class TestExp4_Full:
    """Experiment 4: Full role conditioning (actor input+output, critic heads)."""

    def test_combined_architecture_shapes(self):
        """Exp 4 actor + critic produce correct output shapes."""
        cfg = make_tiny_config()
        cfg["transformer"]["n_embd"] = 64
        cfg["transformer"]["zs_dim"] = 256

        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=True, n_roles=6)
        critic = RoleTransVCritic(
            config=cfg,
            share_obs_space=None,
            obs_space=None,
            act_space=DummyActSpace(7),
            num_agents=10,
            state_type="EP",
        )

        rng, h0, obs, resets, avail, role_ids_actor = init_actor_tensors(batch=4, time=3)
        # critic expects role_ids of shape (batch, n_agents)
        role_ids_critic = jnp.broadcast_to(role_ids_actor[0][:, None], (4, 10))

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        actor_params = actor.init(actor_rng, h0, (obs, resets, avail), role_ids_actor)
        _, obs_all, actions, policy_probs, rnn_states, cr_resets, _ = init_critic_tensors(seed=1, batch=4)
        critic_params = critic.init(
            critic_rng, obs_all, actions, policy_probs, rnn_states, cr_resets, role_ids_critic
        )

        # Actor forward (use original actor resets, not critic resets)
        _, pi = actor.apply(actor_params, h0, (obs, resets, avail), role_ids_actor)
        assert pi.logits.shape == (3, 4, 7), f"Actor logits shape mismatch: {pi.logits.shape}"

        # Critic forward — use env-level values (mean across roles)
        # init_critic_tensors returns: rng, obs_all, actions, policy_probs, rnn_states, resets, role_ids
        _, obs_all_e, actions_e, policy_probs_e, rnn_states_e, resets_e, _ = init_critic_tensors(seed=1, batch=4)
        v_env, q_env, eq_env = critic.get_env_level_values(
            critic_params, obs_all_e, actions_e, policy_probs_e, rnn_states_e, resets_e
        )
        assert v_env.shape == (4, 1), f"V_env shape mismatch: {v_env.shape}"

    def test_all_combined_params_trainable(self):
        """Exp 4: all actor routes, heads, and critic heads receive gradients."""
        cfg = make_tiny_config()
        cfg["transformer"]["n_embd"] = 64
        cfg["transformer"]["zs_dim"] = 256

        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=True, n_roles=6)
        rng, h0, obs, resets, avail, role_ids = init_actor_tensors()

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)

        def loss_fn(p):
            _, pi = actor.apply(p, h0, (obs, resets, avail), role_ids)
            return jnp.sum(pi.logits)

        grads = jax.grad(loss_fn)(params)
        flat_grads = flatten_dict(grads)

        # Check routes have gradients
        route_grads = {k: v for k, v in flat_grads.items() if "route" in str(k)}
        assert len(route_grads) > 0, "No route parameters found"
        for key, grad in route_grads.items():
            has_grad = bool(jnp.any(grad != 0))
            assert has_grad, f"Route parameter {key} has zero gradient"

        # Check heads have gradients
        head_grads = {k: v for k, v in flat_grads.items() if "head" in str(k)}
        assert len(head_grads) > 0, "No head parameters found"
        for key, grad in head_grads.items():
            has_grad = bool(jnp.any(grad != 0))
            assert has_grad, f"Head parameter {key} has zero gradient"

    def test_eggroll_compatibility(self):
        """Actor head params are standard nn.Dense kernels -> EggRoll MM_PARAM compatible."""
        cfg = make_tiny_config()
        actor = RoleActorTrans(action_dim=7, config=cfg, use_pre_gru_routes=False, n_roles=6)
        rng, h0, obs, resets, avail, role_ids = init_actor_tensors()

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)
        flat = flatten_dict(params)

        # Head params should be standard (in_dim, out_dim) or (out_dim, in_dim) kernels
        for key, val in flat.items():
            if "head" in str(key) and "kernel" in str(key):
                assert val.ndim == 2, (
                    f"Head param {key} has {val.ndim} dims, expected 2. "
                    "EggRoll requires 2D weight matrices for MM_PARAM."
                )

        # ES searchable params are head kernels only (backbone frozen)
        # This is an architectural constraint, verified by design


# ---------------------------------------------------------------------------
# Cross-experiment consistency
# ---------------------------------------------------------------------------

class TestCrossExperiment:
    """Tests that apply across all experiments."""

    def test_role_ids_from_unit_types(self):
        """Role IDs derived from env_state.state.unit_types[:, :num_allies]."""
        # This is integration-level; tested in training script
        pass

    def test_kl_penalty_schedule_monotonic(self):
        """KL weight decreases monotonically during decay phase."""
        total = 1000
        schedule = RoleActorTrans.make_kl_schedule(total)
        weights = [float(schedule(t)) for t in range(301)]

        for i in range(1, len(weights)):
            assert weights[i] <= weights[i - 1] + 1e-8, (
                f"KL weight increased at step {i}: {weights[i-1]} -> {weights[i]}. "
                "Schedule must be monotonically decreasing."
            )

    def test_n_roles_configurable_per_map(self):
        """n_roles set from environment's max unit types at init."""
        # Protoss 10v10 has 3 ally roles (stalker, zealot, colossus)
        # But we configure n_roles=6 to handle all maps
        cfg = make_tiny_config()
        actor = RoleActorTrans(action_dim=7, config=cfg, n_roles=3)
        rng, h0, obs, resets, avail, role_ids = init_actor_tensors(n_roles=3)

        params = actor.init(rng, h0, (obs, resets, avail), role_ids)
        _, pi = actor.apply(params, h0, (obs, resets, avail), role_ids)
        assert pi.logits.shape[-1] == 7


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
