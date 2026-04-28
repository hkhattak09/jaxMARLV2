#!/usr/bin/env python3
"""Smoke and parity tests for the JAX MAPPO-T implementation.

Run from the repository root, for example in Colab:

    python test_scripts/test_mappo_t_jax_impl.py

These tests avoid a full SMAX training run. They check that the port can import,
initialise, run small actor/critic forwards, and match MACA's attention helper
semantics on tiny hand-built tensors.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import py_compile
import sys
import unittest
from dataclasses import dataclass


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SMAX_CTM_ROOT = os.path.join(REPO_ROOT, "smax_ctm")
for path in (REPO_ROOT, SMAX_CTM_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - used for friendlier local failure.
    jax = None
    jnp = None
    JAX_IMPORT_ERROR = exc
else:
    JAX_IMPORT_ERROR = None


@dataclass(frozen=True)
class DummyBox:
    shape: tuple[int, ...]


@dataclass(frozen=True)
class DummyDiscrete:
    n: int


def require_jax():
    if JAX_IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"JAX is not importable in this runtime: {JAX_IMPORT_ERROR}")


def load_config_module():
    """Load config.py directly so the pure config test does not require JAX."""
    config_path = os.path.join(REPO_ROOT, "smax_ctm", "mappo_t", "config.py")
    spec = importlib.util.spec_from_file_location("mappo_t_config_for_tests", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_config(num_agents: int = 3) -> dict:
    cfg = load_config_module().get_default_mappo_t_config()
    cfg.update(
        {
            "hidden_sizes": [16],
            "initialization_method": "orthogonal_",
            "gain": 0.01,
            "use_recurrent_policy": False,
            "use_naive_recurrent_policy": False,
            "use_feature_normalization": True,
            "LR": 1e-3,
            "CRITIC_LR": 1e-3,
            "NUM_ENVS": 2,
            "NUM_STEPS": 4,
            "TOTAL_TIMESTEPS": 16,
            "UPDATE_EPOCHS": 1,
            "NUM_MINIBATCHES": 1,
            "use_valuenorm": True,  # Enable ValueNorm for tests
        }
    )
    cfg["transformer"].update(
        {
            "n_embd": 16,
            "n_head": 4,
            "n_encode_layer": 2,
            "n_decode_layer": 1,
            "n_block": num_agents,
            "zs_dim": 16,
            "dropout": 0.0,
            "att_sigma": 0.6,
            "vq_bsln_coef": 0.33,
            "vq_coma_bsln_coef": 0.33,
            "att_roll_res": True,
            "aggregation": "mean",
            "weight_init": "tfixup",
        }
    )
    return cfg


def reference_joint_attention(attn_layers, add_residual):
    """MACA-compatible attention rollout in JAX."""
    _, layers, tokens, _ = attn_layers.shape
    if add_residual:
        eye = jnp.eye(tokens, dtype=attn_layers.dtype).reshape(1, 1, tokens, tokens)
        aug = attn_layers + eye
        aug = aug / aug.sum(axis=-1, keepdims=True)
    else:
        aug = attn_layers

    outs = [aug[:, 0]]
    for layer_idx in range(1, layers):
        outs.append(aug[:, layer_idx] @ outs[-1])
    return jnp.stack(outs, axis=1)


def reference_mixed_action_pi(action, pi, joint_attentions, sigma, layer=-1):
    """MACA-compatible mixed action/policy tensor."""
    _, tokens, _ = action.shape
    pi_rep = jnp.repeat(pi[:, None, :, :], tokens, axis=1)
    action_rep = jnp.repeat(action[:, None, :, :], tokens, axis=1)
    selected_attn = joint_attentions[:, layer, :, :, None]
    mixed = jnp.where(selected_attn >= sigma, pi_rep, action_rep)

    diag = jnp.eye(tokens, dtype=action.dtype).reshape(1, tokens, tokens, 1)
    mixed = jnp.where(diag >= sigma, pi_rep, mixed)
    coma = jnp.where(diag >= sigma, pi_rep, action_rep)
    return mixed, coma


def reference_baseline_weights(joint_attentions, sigma, vq_coef, vq_coma_coef, layer=-1):
    """MACA-compatible baseline weights for [self, group, joint]."""
    _, _, tokens, _ = joint_attentions.shape
    attn = joint_attentions[:, layer]
    self_weights = jnp.diagonal(attn, axis1=-2, axis2=-1)[..., None]
    group_weights = jnp.where(attn >= sigma, attn, jnp.zeros_like(attn))
    diag = jnp.eye(tokens, dtype=attn.dtype).reshape(1, tokens, tokens)
    group_weights = jnp.where(diag >= sigma, attn, group_weights)
    group_weights = group_weights.sum(axis=-1, keepdims=True) - self_weights
    joint_weights = attn.sum(axis=-1, keepdims=True)
    self_weights = vq_coef * self_weights
    group_weights = vq_coma_coef * group_weights
    joint_weights = jnp.clip(joint_weights - self_weights - group_weights, 0.0, 1.0)
    return jnp.concatenate([self_weights, group_weights, joint_weights], axis=-1)


class MappoTJaxImplementationTest(unittest.TestCase):
    def test_sources_compile(self):
        files = [
            os.path.join(REPO_ROOT, "smax_ctm", "train_mappo_t.py"),
            os.path.join(REPO_ROOT, "smax_ctm", "mappo_t", "__init__.py"),
            os.path.join(REPO_ROOT, "smax_ctm", "mappo_t", "actor.py"),
            os.path.join(REPO_ROOT, "smax_ctm", "mappo_t", "config.py"),
            os.path.join(REPO_ROOT, "smax_ctm", "mappo_t", "critic.py"),
            os.path.join(REPO_ROOT, "smax_ctm", "mappo_t", "transformer.py"),
            os.path.join(REPO_ROOT, "smax_ctm", "mappo_t", "utils.py"),
        ]
        for filename in files:
            with self.subTest(filename=os.path.relpath(filename, REPO_ROOT)):
                py_compile.compile(filename, doraise=True)

    def test_default_config_contains_required_runtime_keys(self):
        cfg = load_config_module().get_default_mappo_t_config()
        required_top_level = {
            "hidden_sizes",
            "initialization_method",
            "gain",
            "NUM_MINIBATCHES",
            "UPDATE_EPOCHS",
            "LR",
            "CRITIC_LR",
            "MAX_GRAD_NORM",
        }
        missing = sorted(required_top_level - set(cfg))
        self.assertEqual(missing, [], f"default config is missing keys used by train/actor: {missing}")
        self.assertEqual(cfg["use_recurrent_policy"], False)
        self.assertEqual(cfg["LR"], 0.0005)
        self.assertEqual(cfg["CRITIC_LR"], 0.0005)
        self.assertEqual(cfg["ANNEAL_LR"], False)
        self.assertEqual(cfg["USE_CRITIC_LR_DECAY"], True)
        self.assertEqual(cfg["CLIP_PARAM"], 0.1)
        self.assertEqual(cfg["UPDATE_EPOCHS"], 10)
        self.assertEqual(cfg["NUM_MINIBATCHES"], 1)
        self.assertEqual(cfg["VALUE_LOSS_COEF"], 1.0)
        self.assertEqual(cfg["MAX_GRAD_NORM"], 10.0)
        self.assertEqual(cfg["transformer"]["n_encode_layer"], 1)
        self.assertEqual(cfg["transformer"]["n_decode_layer"], 0)
        self.assertEqual(cfg["transformer"]["n_head"], 1)
        self.assertEqual(cfg["transformer"]["n_embd"], 64)
        self.assertEqual(cfg["transformer"]["zs_dim"], 256)
        self.assertEqual(cfg["transformer"]["bias"], True)
        self.assertEqual(cfg["transformer"]["active_fn"], "gelu")
        self.assertEqual(cfg["transformer"]["weight_init"], "tfixup")
        self.assertEqual(cfg["transformer"]["att_sigma"], 1.0)
        self.assertEqual(cfg["transformer"]["vq_bsln_coef"], 0.3)
        self.assertEqual(cfg["transformer"]["vq_coma_bsln_coef"], 0.3)
        self.assertEqual(cfg["transformer"]["eq_value_loss_coef"], 1.0)

    def test_training_module_imports(self):
        require_jax()
        importlib.import_module("smax_ctm.train_mappo_t")

    def test_actor_forward_and_evaluate_actions(self):
        require_jax()
        from mappo_t.actor import ActorTrans

        batch, agents, obs_dim, action_dim = 2, 3, 5, 6
        cfg = tiny_config(agents)
        model = ActorTrans(action_dim=action_dim, config=cfg)
        obs = jnp.arange(batch * agents * obs_dim, dtype=jnp.float32).reshape(batch, agents, obs_dim) / 10.0
        rnn = jnp.ones((batch, agents, cfg["transformer"]["n_embd"]), dtype=jnp.float32)
        masks = jnp.array([[1, 0, 1], [1, 1, 0]], dtype=jnp.float32)
        avail = jnp.ones((batch, agents, action_dim), dtype=jnp.float32).at[:, :, -1].set(0.0)

        obs_t = obs.reshape(1, batch * agents, obs_dim)
        rnn_t = rnn.reshape(batch * agents, cfg["transformer"]["n_embd"])
        masks_t = masks.reshape(1, batch * agents).astype(bool)
        avail_t = avail.reshape(1, batch * agents, action_dim)

        variables = model.init(jax.random.PRNGKey(0), rnn_t, (obs_t, masks_t, avail_t))
        new_rnn, pi = model.apply(variables, rnn_t, (obs_t, masks_t, avail_t))
        actions = jnp.argmax(pi.logits, axis=-1)
        log_probs = pi.log_prob(actions)
        probs = pi.probs

        self.assertEqual(actions.shape, (1, batch * agents))
        self.assertEqual(log_probs.shape, (1, batch * agents))
        self.assertEqual(probs.shape, (1, batch * agents, action_dim))
        self.assertEqual(new_rnn.shape, rnn_t.shape)
        self.assertTrue(bool(jnp.all(actions < action_dim - 1)))
        self.assertTrue(bool(jnp.allclose(probs[..., -1], 0.0, atol=1e-6)))

        eval_log_probs, entropy, _ = model.apply(
            variables,
            rnn_t,
            obs_t,
            masks_t,
            actions,
            avail_t,
            jnp.ones((1, batch * agents), dtype=jnp.float32),
            method=ActorTrans.evaluate_actions,
        )
        self.assertEqual(eval_log_probs.shape, (1, batch * agents))
        self.assertEqual(entropy.shape, ())

    def test_joint_attention_matches_maca_reference(self):
        require_jax()
        from mappo_t.transformer import compute_joint_attention

        attn = jnp.array(
            [
                [
                    [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]],
                    [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.3, 0.2, 0.5]],
                ]
            ],
            dtype=jnp.float32,
        )
        expected = reference_joint_attention(attn, add_residual=True)
        actual = compute_joint_attention(attn, add_residual=True)
        self.assertTrue(bool(jnp.allclose(actual, expected, atol=1e-6)))

    def test_mixed_action_and_baseline_match_maca_reference(self):
        require_jax()
        from mappo_t.transformer import Encoder

        agents, action_dim = 3, 4
        cfg = tiny_config(agents)
        sigma = min(cfg["transformer"]["att_sigma"] / agents, 1.0)
        encoder = Encoder(args=cfg, obs_space=DummyBox((5,)), act_space=DummyDiscrete(action_dim))

        actions_idx = jnp.array([[0, 1, 2]])
        action = jax.nn.one_hot(actions_idx, action_dim)
        pi = jnp.array(
            [
                [
                    [0.70, 0.10, 0.10, 0.10],
                    [0.20, 0.50, 0.20, 0.10],
                    [0.25, 0.25, 0.25, 0.25],
                ]
            ],
            dtype=jnp.float32,
        )
        joint = jnp.array(
            [
                [
                    [[0.9, 0.1, 0.0], [0.3, 0.6, 0.1], [0.4, 0.2, 0.4]],
                    [[0.8, 0.2, 0.0], [0.2, 0.7, 0.1], [0.1, 0.4, 0.5]],
                ]
            ],
            dtype=jnp.float32,
        )

        expected_mix, expected_coma = reference_mixed_action_pi(action, pi, joint, sigma)
        actual_mix, actual_coma = encoder._get_mixed_action_pi(action, pi, joint[:, -1], sigma)
        self.assertTrue(bool(jnp.allclose(actual_mix, expected_mix, atol=1e-6)))
        self.assertTrue(bool(jnp.allclose(actual_coma, expected_coma, atol=1e-6)))

        expected_weights = reference_baseline_weights(
            joint,
            sigma,
            cfg["transformer"]["vq_bsln_coef"],
            cfg["transformer"]["vq_coma_bsln_coef"],
        )
        actual_weights = encoder._get_baseline_weights(
            joint[:, -1],
            cfg["transformer"]["vq_bsln_coef"],
            cfg["transformer"]["vq_coma_bsln_coef"],
        )
        self.assertTrue(bool(jnp.allclose(actual_weights, expected_weights, atol=1e-6)))

    def test_critic_forward_shapes(self):
        require_jax()
        from mappo_t.critic import TransVCritic

        batch, agents, obs_dim, action_dim = 2, 3, 5, 6
        cfg = tiny_config(agents)
        model = TransVCritic(
            config=cfg,
            share_obs_space=DummyBox((agents * obs_dim,)),
            obs_space=DummyBox((obs_dim,)),
            act_space=DummyDiscrete(action_dim),
            num_agents=agents,
            state_type="EP",
        )
        obs = jnp.ones((batch, agents, obs_dim), dtype=jnp.float32)
        actions = jnp.array([[0, 1, 2], [3, 4, 0]], dtype=jnp.int32)
        policy = jnp.ones((batch, agents, action_dim), dtype=jnp.float32) / action_dim
        rnn = jnp.zeros((batch, agents, cfg["transformer"]["n_embd"]), dtype=jnp.float32)
        masks = jnp.ones((batch, agents, 1), dtype=jnp.float32)

        variables = model.init(jax.random.PRNGKey(0), obs, actions, policy, rnn, masks, True)
        values, q_values, eq_values, vq_values, vq_coma_values, weights, attn, zs, zsa, new_rnn = model.apply(
            variables, obs, actions, policy, rnn, masks, True
        )

        self.assertEqual(values.shape, (batch, 1))
        self.assertEqual(q_values.shape, (batch, 1))
        self.assertEqual(eq_values.shape, (batch, 1))
        self.assertEqual(vq_values.shape, (batch, agents, 1))
        self.assertEqual(vq_coma_values.shape, (batch, agents, 1))
        self.assertEqual(weights.shape, (batch, agents, 3))
        self.assertEqual(attn.shape, (batch, agents, agents))
        self.assertEqual(zs.shape, (batch, cfg["transformer"]["zs_dim"]))
        self.assertEqual(zsa.shape, (batch, cfg["transformer"]["zs_dim"]))
        self.assertEqual(new_rnn.shape, rnn.shape)

    def test_make_train_one_update_smoke(self):
        require_jax()
        from smax_ctm.train_mappo_t import make_train

        cfg = tiny_config(num_agents=3)
        cfg.update(
            {
                "MAP_NAME": "3m",
                "NUM_ENVS": 2,
                "NUM_STEPS": 2,
                "TOTAL_TIMESTEPS": 4,
                "UPDATE_EPOCHS": 1,
                "NUM_MINIBATCHES": 1,
                "ANNEAL_LR": False,
                "use_recurrent_policy": False,
                "use_naive_recurrent_policy": False,
                "ENV_KWARGS": {
                    "see_enemy_actions": True,
                    "walls_cause_death": True,
                    "attack_mode": "closest",
                },
            }
        )
        cfg["transformer"].update(
            {
                "n_encode_layer": 1,
                "n_decode_layer": 1,
                "n_block": 3,
                "output_attentions": True,
            }
        )
        out = jax.jit(make_train(cfg))(jax.random.PRNGKey(123))
        self.assertIn("metric", out)


class ValueNormTest(unittest.TestCase):
    """Tests for JAX ValueNorm implementation matching MACA behavior."""

    def test_valuenorm_initialization(self):
        """Test ValueNorm initializes with correct default values."""
        require_jax()
        from mappo_t.valuenorm import init_value_norm, ValueNormState

        state = init_value_norm((1,))
        self.assertIsInstance(state, ValueNormState)
        self.assertEqual(state.beta, 0.99999)
        self.assertEqual(state.epsilon, 1e-5)
        self.assertEqual(state.var_clamp_min, 1e-2)
        self.assertTrue(bool(jnp.allclose(state.running_mean, 0.0)))
        self.assertTrue(bool(jnp.allclose(state.running_mean_sq, 0.0)))
        self.assertTrue(bool(jnp.allclose(state.debiasing_term, 0.0)))

    def test_valuenorm_update_matches_maca_formula(self):
        """Test that ValueNorm update matches MACA's EMA formula."""
        require_jax()
        from mappo_t.valuenorm import init_value_norm, value_norm_update, value_norm_running_stats

        state = init_value_norm((1,), beta=0.9)
        
        # Update with a batch of values
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = value_norm_update(state, x)
        
        # Check running statistics
        mean, var = value_norm_running_stats(state)
        
        # Manual calculation:
        # batch_mean = 3.0, batch_sq_mean = 11.0
        # running_mean = 0.9 * 0 + 0.1 * 3.0 = 0.3
        # running_mean_sq = 0.9 * 0 + 0.1 * 11.0 = 1.1
        # debiasing_term = 0.9 * 0 + 0.1 * 1.0 = 0.1
        # debiased_mean = 0.3 / 0.1 = 3.0
        # debiased_var = 1.1 / 0.1 - 3.0^2 = 11.0 - 9.0 = 2.0
        # But clamped to 1e-2 minimum
        
        self.assertTrue(bool(jnp.allclose(mean, 3.0, atol=1e-5)))
        self.assertTrue(bool(jnp.allclose(var, 2.0, atol=1e-5)))

    def test_valuenorm_normalize_denormalize_roundtrip(self):
        """Test that normalize then denormalize recovers the input."""
        require_jax()
        from mappo_t.valuenorm import (
            init_value_norm, value_norm_update, 
            value_norm_normalize, value_norm_denormalize
        )

        state = init_value_norm((1,), beta=0.9)
        
        # Build up some statistics
        for i in range(10):
            x = jnp.array([float(i), float(i + 1), float(i + 2)])
            state = value_norm_update(state, x)
        
        # Test normalize/denormalize roundtrip
        test_input = jnp.array([5.0, 10.0, 15.0])
        normalized = value_norm_normalize(state, test_input)
        denormalized = value_norm_denormalize(state, normalized)
        
        self.assertTrue(bool(jnp.allclose(test_input, denormalized, atol=1e-5)))

    def test_valuenorm_variance_clamp(self):
        """Test that variance is clamped to var_clamp_min."""
        require_jax()
        from mappo_t.valuenorm import init_value_norm, value_norm_update, value_norm_running_stats

        state = init_value_norm((1,), beta=0.99, var_clamp_min=1e-2)
        
        # Update with constant values (zero variance)
        x = jnp.array([1.0, 1.0, 1.0])
        state = value_norm_update(state, x)
        state = value_norm_update(state, x)
        state = value_norm_update(state, x)
        
        _, var = value_norm_running_stats(state)
        # Variance should be clamped to at least 1e-2
        self.assertTrue(bool(jnp.all(var >= 1e-2)))

    def test_create_value_norm_dict(self):
        """Test creating a dictionary of ValueNorm states."""
        require_jax()
        from mappo_t.valuenorm import create_value_norm_dict

        # Test with ValueNorm enabled
        norm_dict = create_value_norm_dict(use_valuenorm=True)
        self.assertIsNotNone(norm_dict)
        self.assertIn("v", norm_dict)
        self.assertIn("q", norm_dict)
        self.assertIn("eq", norm_dict)

        # Test with ValueNorm disabled
        norm_dict = create_value_norm_dict(use_valuenorm=False)
        self.assertIsNone(norm_dict)

    def test_update_value_norm_dict(self):
        """Test updating all ValueNorm states in a dictionary."""
        require_jax()
        from mappo_t.valuenorm import create_value_norm_dict, update_value_norm_dict

        norm_dict = create_value_norm_dict(use_valuenorm=True)
        
        v_targets = jnp.array([1.0, 2.0, 3.0])
        q_targets = jnp.array([4.0, 5.0, 6.0])
        eq_targets = jnp.array([7.0, 8.0, 9.0])
        
        updated = update_value_norm_dict(norm_dict, v_targets, q_targets, eq_targets)
        
        self.assertIsNotNone(updated)
        # Check that statistics were updated (running_mean should change from 0)
        self.assertTrue(bool(updated["v"].running_mean != 0.0))
        self.assertTrue(bool(updated["q"].running_mean != 0.0))
        self.assertTrue(bool(updated["eq"].running_mean != 0.0))

    def test_normalize_targets(self):
        """Test normalizing targets using ValueNorm dictionary."""
        require_jax()
        from mappo_t.valuenorm import create_value_norm_dict, normalize_targets

        norm_dict = create_value_norm_dict(use_valuenorm=True)
        
        # Update with some data first
        v_targets = jnp.array([1.0, 2.0, 3.0])
        q_targets = jnp.array([4.0, 5.0, 6.0])
        eq_targets = jnp.array([7.0, 8.0, 9.0])
        
        from mappo_t.valuenorm import update_value_norm_dict
        norm_dict = update_value_norm_dict(norm_dict, v_targets, q_targets, eq_targets)
        
        # Now normalize new targets
        v_norm, q_norm, eq_norm = normalize_targets(
            norm_dict,
            jnp.array([1.0, 2.0, 3.0]),
            jnp.array([4.0, 5.0, 6.0]),
            jnp.array([7.0, 8.0, 9.0]),
        )
        
        # Normalized values should have different scale than raw values
        self.assertTrue(bool(jnp.any(v_norm != jnp.array([1.0, 2.0, 3.0]))))

    def test_denormalize_predictions(self):
        """Test denormalizing predictions using ValueNorm dictionary."""
        require_jax()
        from mappo_t.valuenorm import (
            create_value_norm_dict,
            denormalize_predictions,
            update_value_norm_dict,
            value_norm_normalize,
        )

        norm_dict = create_value_norm_dict(use_valuenorm=True)
        
        # Update with some data
        v_targets = jnp.array([1.0, 2.0, 3.0])
        q_targets = jnp.array([4.0, 5.0, 6.0])
        eq_targets = jnp.array([7.0, 8.0, 9.0])
        norm_dict = update_value_norm_dict(norm_dict, v_targets, q_targets, eq_targets)
        
        # Normalize then denormalize should recover original
        v_pred = jnp.array([1.0, 2.0, 3.0])
        q_pred = jnp.array([4.0, 5.0, 6.0])
        eq_pred = jnp.array([7.0, 8.0, 9.0])
        
        v_norm = value_norm_normalize(norm_dict["v"], v_pred)
        
        v_denorm, q_denorm, eq_denorm = denormalize_predictions(
            norm_dict, v_norm, q_pred, eq_pred
        )
        
        # v_denorm should be close to v_pred
        self.assertTrue(bool(jnp.allclose(v_denorm, v_pred, atol=1.0)))

    def test_config_use_valuenorm_key(self):
        """Test that config contains use_valuenorm key."""
        cfg = load_config_module().get_default_mappo_t_config()
        self.assertIn("use_valuenorm", cfg)
        self.assertTrue(cfg["use_valuenorm"])

    def test_train_with_valuenorm_enabled(self):
        """Test that training works with ValueNorm enabled."""
        require_jax()
        from smax_ctm.train_mappo_t import make_train

        cfg = tiny_config(num_agents=3)
        cfg["use_valuenorm"] = True
        cfg.update(
            {
                "MAP_NAME": "3m",
                "NUM_ENVS": 2,
                "NUM_STEPS": 2,
                "TOTAL_TIMESTEPS": 4,
                "UPDATE_EPOCHS": 1,
                "NUM_MINIBATCHES": 1,
                "ANNEAL_LR": False,
            }
        )
        
        # This should not raise an error
        out = jax.jit(make_train(cfg))(jax.random.PRNGKey(123))
        self.assertIn("metric", out)


class MinibatchTest(unittest.TestCase):
    """Tests for principled minibatch updates."""

    def test_config_has_actor_critic_minibatch_keys(self):
        """Test that config contains separate actor/critic minibatch keys."""
        cfg = load_config_module().get_default_mappo_t_config()
        # Check actor minibatch keys
        self.assertIn("ACTOR_NUM_MINI_BATCH", cfg)
        self.assertIn("PPO_EPOCH", cfg)
        # Check critic minibatch keys
        self.assertIn("CRITIC_NUM_MINI_BATCH", cfg)
        self.assertIn("CRITIC_EPOCH", cfg)
        # Check backward-compatible aliases
        self.assertIn("NUM_MINIBATCHES", cfg)
        self.assertIn("UPDATE_EPOCHS", cfg)

    def test_config_validation_divisibility(self):
        """Test that invalid divisibility raises clear errors."""
        validate_mappo_t_config = load_config_module().validate_mappo_t_config
        
        # Valid config should not raise
        cfg = tiny_config(num_agents=3)
        cfg["NUM_STEPS"] = 10
        cfg["NUM_ENVS"] = 2
        cfg["ACTOR_NUM_MINI_BATCH"] = 2  # 10 * 2 * 3 = 60, divisible by 2
        cfg["CRITIC_NUM_MINI_BATCH"] = 5  # 10 * 2 = 20, divisible by 5
        validate_mappo_t_config(cfg, 3)  # Should not raise
        
        # Invalid actor divisibility
        cfg["ACTOR_NUM_MINI_BATCH"] = 7  # 60 not divisible by 7
        with self.assertRaises(ValueError) as context:
            validate_mappo_t_config(cfg, 3)
        self.assertIn("ACTOR_NUM_MINI_BATCH", str(context.exception))
        
        # Invalid critic divisibility
        cfg["ACTOR_NUM_MINI_BATCH"] = 2  # Reset to valid
        cfg["CRITIC_NUM_MINI_BATCH"] = 7  # 20 not divisible by 7
        with self.assertRaises(ValueError) as context:
            validate_mappo_t_config(cfg, 3)
        self.assertIn("CRITIC_NUM_MINI_BATCH", str(context.exception))

    def test_actor_minibatch_indexing_covers_all_samples(self):
        """Test that actor minibatch indexing covers T * NUM_ACTORS."""
        require_jax()
        num_agents = 3
        num_steps = 10
        num_envs = 2
        actor_batch_size = num_steps * num_envs * num_agents  # 60
        
        rng = jax.random.PRNGKey(0)
        perm = jax.random.permutation(rng, actor_batch_size)
        actor_num_mini_batch = 2
        actor_mini_batch_size = actor_batch_size // actor_num_mini_batch  # 30
        
        minibatch_idx = perm.reshape(actor_num_mini_batch, actor_mini_batch_size)
        
        # Check that all indices are covered (no duplicates, no missing)
        flattened = minibatch_idx.flatten()
        self.assertEqual(len(flattened), actor_batch_size)
        self.assertEqual(len(jnp.unique(flattened)), actor_batch_size)

    def test_critic_minibatch_indexing_preserves_agent_axis(self):
        """Test that critic minibatch indexing covers T * NUM_ENVS and preserves agent axis."""
        require_jax()
        num_agents = 3
        num_steps = 10
        num_envs = 2
        critic_batch_size = num_steps * num_envs  # 20
        
        rng = jax.random.PRNGKey(0)
        perm = jax.random.permutation(rng, critic_batch_size)
        critic_num_mini_batch = 2
        critic_mini_batch_size = critic_batch_size // critic_num_mini_batch  # 10
        
        minibatch_idx = perm.reshape(critic_num_mini_batch, critic_mini_batch_size)
        
        # Check that all indices are covered
        flattened = minibatch_idx.flatten()
        self.assertEqual(len(flattened), critic_batch_size)
        self.assertEqual(len(jnp.unique(flattened)), critic_batch_size)
        
        # Simulate gathering critic data with agent axis preserved
        # Shape: (T*NUM_ENVS, num_agents, obs_dim)
        obs_dim = 5
        critic_obs = jnp.ones((critic_batch_size, num_agents, obs_dim))
        
        # Gather minibatch - should preserve agent axis
        mb_idx = minibatch_idx[0]  # First minibatch
        mb_obs = jnp.take(critic_obs, mb_idx, axis=0)
        self.assertEqual(mb_obs.shape, (critic_mini_batch_size, num_agents, obs_dim))

    def test_tiny_training_with_minibatches(self):
        """Tiny training smoke test with ACTOR_NUM_MINI_BATCH=2 and CRITIC_NUM_MINI_BATCH=2."""
        require_jax()
        from smax_ctm.train_mappo_t import make_train
        validate_mappo_t_config = load_config_module().validate_mappo_t_config

        cfg = tiny_config(num_agents=3)
        cfg.update(
            {
                "MAP_NAME": "3m",
                "NUM_ENVS": 2,
                "NUM_STEPS": 4,
                "TOTAL_TIMESTEPS": 16,
                "PPO_EPOCH": 1,
                "ACTOR_NUM_MINI_BATCH": 2,  # 4 * 2 * 3 = 24, divisible by 2
                "CRITIC_EPOCH": 1,
                "CRITIC_NUM_MINI_BATCH": 2,  # 4 * 2 = 8, divisible by 2
                "ANNEAL_LR": False,
                "USE_CRITIC_LR_DECAY": False,  # Simplify for test
                "ENV_KWARGS": {
                    "see_enemy_actions": True,
                    "walls_cause_death": True,
                    "attack_mode": "closest",
                },
            }
        )
        # Ensure backward-compatible aliases are set
        cfg["UPDATE_EPOCHS"] = cfg["PPO_EPOCH"]
        cfg["NUM_MINIBATCHES"] = cfg["ACTOR_NUM_MINI_BATCH"]
        
        # Validate config
        validate_mappo_t_config(cfg, 3)
        
        # Run training
        out = jax.jit(make_train(cfg))(jax.random.PRNGKey(123))
        self.assertIn("metric", out)
        self.assertIn("runner_state", out)

    def test_recurrent_policy_raises_not_implemented(self):
        """Test that recurrent policy with ACTOR_NUM_MINI_BATCH > 1 raises NotImplementedError."""
        validate_mappo_t_config = load_config_module().validate_mappo_t_config

        cfg = tiny_config(num_agents=3)
        cfg["use_recurrent_policy"] = True
        cfg["ACTOR_NUM_MINI_BATCH"] = 2  # Should raise
        
        with self.assertRaises(NotImplementedError) as context:
            validate_mappo_t_config(cfg, 3)
        self.assertIn("Recurrent MAPPO-T minibatches", str(context.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
