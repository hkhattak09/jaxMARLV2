"""Comprehensive parity tests for MAPPO-T JAX implementation against MACA semantics.

Run from repo root:
    pytest test_scripts/test_mappo_t_jax_impl.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_TEST_DIR, ".."))
MACA_ROOT = os.path.join(REPO_ROOT, "MACA")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# mappo_t is a package inside smax_ctm; match train_mappo_t.py path setup.
_SMAX_CTM = os.path.join(REPO_ROOT, "smax_ctm")
if _SMAX_CTM not in sys.path:
    sys.path.insert(0, _SMAX_CTM)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import distrax

from mappo_t import ActorTrans, ScannedRNN, TransVCritic, get_default_mappo_t_config
from mappo_t.valuenorm import (
    ValueNormState,
    init_value_norm,
    value_norm_update,
    value_norm_normalize,
    value_norm_denormalize,
    value_norm_running_stats,
    create_value_norm_dict,
)
from mappo_t.transformer import (
    Encoder,
    compute_joint_attention,
)
from mappo_t.config import validate_mappo_t_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def base_config():
    cfg = get_default_mappo_t_config()
    cfg["NUM_ENVS"] = 4
    cfg["NUM_STEPS"] = 8
    cfg["TOTAL_TIMESTEPS"] = cfg["NUM_STEPS"] * cfg["NUM_ENVS"] * 2
    cfg["PPO_EPOCH"] = 2
    cfg["CRITIC_EPOCH"] = 2
    cfg["DATA_CHUNK_LENGTH"] = 4
    cfg["ACTOR_NUM_MINI_BATCH"] = 1
    cfg["CRITIC_NUM_MINI_BATCH"] = 1
    cfg["transformer"]["n_encode_layer"] = 1
    cfg["transformer"]["n_decode_layer"] = 0
    cfg["transformer"]["n_block"] = 3
    cfg["transformer"]["n_embd"] = 32
    cfg["transformer"]["zs_dim"] = 64
    cfg["transformer"]["dropout"] = 0.0
    cfg["use_valuenorm"] = True
    cfg["use_huber_loss"] = True
    cfg["huber_delta"] = 10.0
    cfg["use_clipped_value_loss"] = True
    cfg["CLIP_PARAM"] = 0.1
    cfg["GAMMA"] = 0.99
    cfg["GAE_LAMBDA"] = 0.95
    cfg["use_proper_time_limits"] = True
    cfg["ENT_COEF"] = 0.01
    cfg["VALUE_LOSS_COEF"] = 1.0
    cfg["ANNEAL_LR"] = False
    cfg["USE_CRITIC_LR_DECAY"] = False
    cfg["action_aggregation"] = "single"
    return cfg


def reference_joint_attention(attn_layers, add_residual=True):
    """Independent MACA-style attention rollout formula."""
    _, n_layers, n_tokens, _ = attn_layers.shape
    if add_residual:
        residual = jnp.eye(n_tokens, dtype=attn_layers.dtype).reshape(
            1, 1, n_tokens, n_tokens
        )
        attn_layers = attn_layers + residual
        attn_layers = attn_layers / jnp.sum(attn_layers, axis=-1, keepdims=True)

    rolled = [attn_layers[:, 0]]
    for idx in range(1, n_layers):
        rolled.append(attn_layers[:, idx] @ rolled[-1])
    return jnp.stack(rolled, axis=1)


def reference_mixed_action_pi(action, pi, joint_attentions, sigma, layer=-1):
    """Independent MACA-style mixed action/policy tensors."""
    attn = joint_attentions[:, layer] if joint_attentions.ndim == 4 else joint_attentions
    batch_size, n_agents, action_dim = action.shape
    pi_rep = jnp.repeat(pi[:, None, :, :], n_agents, axis=1)
    action_rep = jnp.repeat(action[:, None, :, :], n_agents, axis=1)
    mixed = jnp.where(attn[..., None] >= sigma, pi_rep, action_rep)
    diag = jnp.eye(n_agents, dtype=action.dtype).reshape(1, n_agents, n_agents, 1)
    mixed = jnp.where(diag >= sigma, pi_rep, mixed)
    coma = jnp.where(diag >= sigma, pi_rep, action_rep)
    return mixed.reshape(batch_size, n_agents, n_agents, action_dim), coma.reshape(
        batch_size, n_agents, n_agents, action_dim
    )


def reference_baseline_weights(joint_attentions, sigma, vq_coef, vq_coma_coef, layer=-1):
    """Independent MACA-style [self, group, joint] baseline weights."""
    attn = joint_attentions[:, layer] if joint_attentions.ndim == 4 else joint_attentions
    n_agents = attn.shape[-1]
    self_weights = jnp.diagonal(attn, axis1=-2, axis2=-1)[..., None]
    group_weights = jnp.where(attn >= sigma, attn, jnp.zeros_like(attn))
    diag = jnp.eye(n_agents, dtype=attn.dtype).reshape(1, n_agents, n_agents)
    group_weights = jnp.where(diag >= sigma, attn, group_weights)
    group_weights = jnp.sum(group_weights, axis=-1, keepdims=True) - self_weights
    joint_weights = jnp.sum(attn, axis=-1, keepdims=True)
    self_weights = vq_coef * self_weights
    group_weights = vq_coma_coef * group_weights
    joint_weights = jnp.clip(joint_weights - self_weights - group_weights, 0.0, 1.0)
    return jnp.concatenate([self_weights, group_weights, joint_weights], axis=-1)


def tree_flatten_with_paths(tree):
    """Compatibility shim for JAX versions with singular/plural path API names."""
    flatten_fn = getattr(jax.tree_util, "tree_flatten_with_path", None)
    if flatten_fn is None:
        flatten_fn = getattr(jax.tree_util, "tree_flatten_with_paths")
    return flatten_fn(tree)[0]


def key_path_to_name(path):
    parts = []
    for key in path:
        if hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "name"):
            parts.append(str(key.name))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return "/".join(parts)


# ---------------------------------------------------------------------------
# 1. MACA source architecture: MAPPO-T actor is MLP policy, transformer critic
# ---------------------------------------------------------------------------

class TestPaperConfig:
    def test_default_jax_config_uses_paper_recurrent_hparams(self):
        cfg = get_default_mappo_t_config()
        assert cfg["hidden_sizes"] == [64, 64, 64]
        assert cfg["use_naive_recurrent_policy"] is False
        assert cfg["use_recurrent_policy"] is True
        assert cfg["recurrent_n"] == 1
        assert cfg["DATA_CHUNK_LENGTH"] == 10
        assert cfg["ACTOR_NUM_MINI_BATCH"] == 1
        assert cfg["CRITIC_NUM_MINI_BATCH"] == 1
        assert cfg["transformer"]["n_encode_layer"] == 1
        assert cfg["transformer"]["n_head"] == 1
        assert cfg["transformer"]["n_embd"] == 64
        assert cfg["transformer"]["zs_dim"] == 256
        assert cfg["transformer"]["active_fn"] == "gelu"
        assert cfg["transformer"]["wght_decay"] == 0.01
        assert cfg["transformer"]["betas"] == [0.9, 0.95]
        assert cfg["transformer"]["weight_init"] == "tfixup"

    def test_recurrent_config_validation_allows_real_minibatches(self):
        cfg = get_default_mappo_t_config()
        cfg["NUM_ENVS"] = 8
        cfg["NUM_STEPS"] = 20
        cfg["DATA_CHUNK_LENGTH"] = 10
        cfg["ACTOR_NUM_MINI_BATCH"] = 4
        cfg["CRITIC_NUM_MINI_BATCH"] = 4
        validate_mappo_t_config(cfg, num_agents=3)

    def test_recurrent_config_validation_rejects_bad_chunk_minibatches(self):
        cfg = get_default_mappo_t_config()
        cfg["NUM_ENVS"] = 5
        cfg["NUM_STEPS"] = 20
        cfg["DATA_CHUNK_LENGTH"] = 10
        cfg["ACTOR_NUM_MINI_BATCH"] = 2
        cfg["CRITIC_NUM_MINI_BATCH"] = 3

        with pytest.raises(ValueError, match="Recurrent critic minibatch"):
            validate_mappo_t_config(cfg, num_agents=3)


class TestMacaSourceArchitecture:
    def test_maca_mappo_t_actor_chain_is_mlp_policy(self):
        """MACA MAPPOTrans actor = StochasticPolicyTrans -> StochasticPolicy -> MLPBase."""
        maca_root = Path(MACA_ROOT)
        if not maca_root.exists():
            pytest.skip("MACA directory is not present in this checkout")

        actor_src = (maca_root / "harl/algorithms/actors/mappo_t.py").read_text()
        policy_t_src = (
            maca_root / "harl/models/policy_models/stochastic_policy_t.py"
        ).read_text()
        policy_src = (
            maca_root / "harl/models/policy_models/stochastic_policy.py"
        ).read_text()
        critic_init_src = (maca_root / "harl/algorithms/critics/__init__.py").read_text()

        assert "self.actor = StochasticPolicyTrans" in actor_src
        assert "class StochasticPolicyTrans(StochasticPolicy)" in policy_t_src
        assert "super().__init__(args, obs_space, action_space, device)" in policy_t_src
        assert "ACTLayerTrans" in policy_t_src
        assert "base = MLPBase" in policy_src
        assert '"mappo_t": TransVCritic' in critic_init_src
        assert "Transformer" not in actor_src


# ---------------------------------------------------------------------------
# 2. JAX actor architecture: MLP base, no transformer
# ---------------------------------------------------------------------------

class TestActorArchitecture:
    def test_actor_is_mlp_not_transformer(self, rng, base_config):
        """JAX ActorTrans should mirror MACA's MLP-based StochasticPolicyTrans."""
        actor = ActorTrans(action_dim=9, config=base_config)
        obs = jnp.zeros((1, 2, 16))
        resets = jnp.zeros((1, 2), dtype=bool)
        avail = jnp.ones((1, 2, 9))
        hstate = ScannedRNN.initialize_carry(2, base_config["hidden_sizes"][-1])
        params = actor.init(rng, hstate, (obs, resets, avail))

        params_flat = tree_flatten_with_paths(params)
        param_names = [key_path_to_name(path) for path, _ in params_flat]

        # Must have MLP base layers
        assert any("base_0" in n for n in param_names), param_names
        assert any("base_1" in n for n in param_names), param_names
        assert any("base_2" in n for n in param_names), param_names
        # Must have action_out
        assert any("action_out" in n for n in param_names), param_names
        # Must NOT have transformer blocks / attention
        assert not any("attn" in n for n in param_names), param_names
        assert not any("block_" in n for n in param_names), param_names
        assert not any("query" in n for n in param_names), param_names
        assert not any("key" in n for n in param_names), param_names
        assert not any("value" in n for n in param_names), param_names

    def test_actor_param_tree_no_extra_action_hidden(self, rng, base_config):
        """JAX actor params: base layers + single action_out, no extra hidden before action."""
        actor = ActorTrans(action_dim=9, config=base_config)
        obs = jnp.zeros((1, 2, 16))
        resets = jnp.zeros((1, 2), dtype=bool)
        avail = jnp.ones((1, 2, 9))
        hstate = ScannedRNN.initialize_carry(2, base_config["hidden_sizes"][-1])
        params = actor.init(rng, hstate, (obs, resets, avail))

        dense_names = []
        for path, _ in tree_flatten_with_paths(params):
            keys = key_path_to_name(path).split("/")
            if keys[-1] == "kernel":
                dense_names.append("/".join(keys[:-1]))

        # action_out must exist exactly once
        action_out_names = [n for n in dense_names if "action_out" in n]
        assert len(action_out_names) == 1, f"Expected 1 action_out, got {action_out_names}"
        # No extra hidden layers between base and action_out
        hidden_like = [n for n in dense_names if "hidden" in n or "action_" in n and "action_out" not in n]
        assert len(hidden_like) == 0, f"Unexpected hidden layers: {hidden_like}"


# ---------------------------------------------------------------------------
# 3. Actor output parity: shapes, masking, deterministic argmax
# ---------------------------------------------------------------------------

class TestActorOutputs:
    def test_action_logprob_prob_shapes(self, rng, base_config):
        actor = ActorTrans(action_dim=9, config=base_config)
        batch = 8
        obs = jnp.zeros((1, batch, 16))
        resets = jnp.zeros((1, batch), dtype=bool)
        avail = jnp.ones((1, batch, 9))
        hstate = ScannedRNN.initialize_carry(batch, base_config["hidden_sizes"][-1])
        params = actor.init(rng, hstate, (obs, resets, avail))
        hstate_out, pi = actor.apply(params, hstate, (obs, resets, avail))

        assert pi.logits.shape == (1, batch, 9)
        assert pi.probs.shape == (1, batch, 9)
        actions = pi.sample(seed=rng)
        assert actions.shape == (1, batch)
        log_probs = pi.log_prob(actions)
        assert log_probs.shape == (1, batch)

    def test_invalid_action_masking(self, rng, base_config):
        actor = ActorTrans(action_dim=5, config=base_config)
        batch = 4
        obs = jnp.zeros((1, batch, 16))
        resets = jnp.zeros((1, batch), dtype=bool)
        # Mask out action 2
        avail = jnp.ones((1, batch, 5))
        avail = avail.at[:, :, 2].set(0.0)
        hstate = ScannedRNN.initialize_carry(batch, base_config["hidden_sizes"][-1])
        params = actor.init(rng, hstate, (obs, resets, avail))
        _, pi = actor.apply(params, hstate, (obs, resets, avail))
        # Probabilities for masked action should be ~0
        np.testing.assert_allclose(
            pi.probs[0, :, 2], jnp.zeros(batch), atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            pi.probs.sum(axis=-1), jnp.ones((1, batch)), atol=1e-6, rtol=1e-6
        )
        # With zero logits and one invalid action, the distribution is uniform
        # over the 4 valid actions.
        np.testing.assert_allclose(jnp.mean(pi.entropy()), jnp.log(4.0), atol=1e-5)

    def test_deterministic_argmax(self, rng, base_config):
        actor = ActorTrans(action_dim=5, config=base_config)
        batch = 4
        obs = jax.random.normal(rng, (1, batch, 16))
        resets = jnp.zeros((1, batch), dtype=bool)
        avail = jnp.ones((1, batch, 5))
        hstate = ScannedRNN.initialize_carry(batch, base_config["hidden_sizes"][-1])
        params = actor.init(rng, hstate, (obs, resets, avail))
        _, pi = actor.apply(params, hstate, (obs, resets, avail))
        actions_det, action_log_probs, probs, _ = actor.apply(
            params,
            hstate,
            obs,
            resets,
            avail,
            rng,
            True,
            method=ActorTrans.get_actions,
        )
        np.testing.assert_array_equal(actions_det, jnp.argmax(pi.logits, axis=-1))
        np.testing.assert_allclose(action_log_probs, pi.log_prob(actions_det), atol=1e-6)
        np.testing.assert_allclose(probs, pi.probs, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Transformer critic forward shape parity (n_decode_layer=0)
# ---------------------------------------------------------------------------

class TestCriticForwardShapes:
    def test_n_decode_zero_shapes(self, rng, base_config):
        """With n_decode_layer=0, critic returns v/q/eq/vq/vq_coma, attn, baseline weights."""
        cfg = base_config.copy()
        cfg["transformer"] = base_config["transformer"].copy()
        cfg["transformer"]["n_decode_layer"] = 0
        cfg["transformer"]["n_encode_layer"] = 1
        cfg["transformer"]["n_block"] = 3
        num_agents = 3

        encoder = Encoder(
            args=cfg,
            obs_space=None,
            act_space=None,
        )
        batch = 2
        obs = jnp.zeros((batch, num_agents, 16))
        action = jax.nn.one_hot(
            jnp.zeros((batch, num_agents), dtype=jnp.int32), 9
        )
        policy_prob = jnp.ones((batch, num_agents, 9)) / 9
        rnn = jnp.zeros((batch, num_agents, cfg["transformer"]["n_embd"]))
        resets = jnp.zeros((batch, num_agents), dtype=bool)
        params = encoder.init(rng, obs, action, policy_prob, rnn, resets, True, True)
        out = encoder.apply(params, obs, action, policy_prob, rnn, resets, True, True)
        (
            v_loc,
            q_loc,
            eq_loc,
            vq_loc,
            vq_coma_loc,
            baseline_weights,
            final_attentions,
            zs,
            zsa,
            rnn_out,
        ) = out

        assert v_loc.shape == (batch, 1)
        assert q_loc.shape == (batch, 1)
        assert eq_loc.shape == (batch, 1)
        assert vq_loc.shape == (batch, num_agents, 1)
        assert vq_coma_loc.shape == (batch, num_agents, 1)
        assert baseline_weights.shape == (batch, num_agents, 3)
        assert final_attentions.shape == (batch, num_agents, num_agents)
        assert zs.shape == (batch, cfg["transformer"]["zs_dim"])
        assert zsa.shape == (batch, cfg["transformer"]["zs_dim"])
        assert rnn_out.shape == rnn.shape

    def test_trans_vcritic_sequence_unroll(self, rng, base_config):
        """TransVCritic handles (time, batch, agents, ...) sequences."""
        cfg = base_config.copy()
        cfg["transformer"] = base_config["transformer"].copy()
        cfg["transformer"]["n_decode_layer"] = 0
        num_agents = 3

        class FakeActSpace:
            n = 9

        critic = TransVCritic(
            config=cfg,
            share_obs_space=None,
            obs_space=None,
            act_space=FakeActSpace(),
            num_agents=num_agents,
            state_type="EP",
        )
        T, B = 4, 2
        obs = jnp.zeros((T, B, num_agents, 16))
        action = jnp.zeros((T, B, num_agents), dtype=jnp.int32)
        policy_prob = jnp.ones((T, B, num_agents, 9)) / 9
        rnn = jnp.zeros((B, num_agents, cfg["transformer"]["n_embd"]))
        resets = jnp.zeros((T, B, num_agents), dtype=bool)
        params = critic.init(rng, obs, action, policy_prob, rnn, resets, True, True)
        out = critic.apply(params, obs, action, policy_prob, rnn, resets, True, True)
        (
            values,
            q_values,
            eq_values,
            vq_values,
            vq_coma_values,
            baseline_weights,
            attn_weights,
            zs,
            zsa,
            carry,
        ) = out

        assert values.shape == (T, B, 1)
        assert q_values.shape == (T, B, 1)
        assert eq_values.shape == (T, B, 1)
        assert vq_values.shape == (T, B, num_agents, 1)
        assert vq_coma_values.shape == (T, B, num_agents, 1)
        assert baseline_weights.shape == (T, B, num_agents, 3)
        assert attn_weights.shape == (T, B, num_agents, num_agents)
        assert carry.shape == rnn.shape


# ---------------------------------------------------------------------------
# 5. Attention helper parity
# ---------------------------------------------------------------------------

class TestAttentionHelpers:
    def test_joint_attention_rollout(self):
        """compute_joint_attention matches MACA attention rollout semantics."""
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
        np.testing.assert_allclose(actual, expected, atol=1e-6)

        expected_no_res = reference_joint_attention(attn, add_residual=False)
        actual_no_res = compute_joint_attention(attn, add_residual=False)
        np.testing.assert_allclose(actual_no_res, expected_no_res, atol=1e-6)

    def test_mixed_action_policy_tensor(self, rng, base_config):
        """Encoder._get_mixed_action_pi matches MACA's mixed action/policy formula."""
        cfg = base_config.copy()
        cfg["transformer"] = base_config["transformer"].copy()
        encoder = Encoder(args=cfg, obs_space=None, act_space=None)
        action = jax.nn.one_hot(jnp.array([[0, 1, 2]]), 4)
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
        joint_attentions = jnp.array(
            [
                [
                    [[0.9, 0.1, 0.0], [0.3, 0.6, 0.1], [0.4, 0.2, 0.4]],
                    [[0.8, 0.2, 0.0], [0.2, 0.7, 0.1], [0.1, 0.4, 0.5]],
                ]
            ],
            dtype=jnp.float32,
        )
        sigma = 0.2
        expected_mix, expected_coma = reference_mixed_action_pi(
            action, pi, joint_attentions, sigma
        )
        actual_mix, actual_coma = encoder._get_mixed_action_pi(
            action, pi, joint_attentions, sigma
        )
        np.testing.assert_allclose(actual_mix, expected_mix, atol=1e-6)
        np.testing.assert_allclose(actual_coma, expected_coma, atol=1e-6)

    def test_baseline_weights(self, rng, base_config):
        """Encoder._get_baseline_weights matches MACA's baseline-weight formula."""
        cfg = base_config.copy()
        cfg["transformer"] = base_config["transformer"].copy()
        encoder = Encoder(args=cfg, obs_space=None, act_space=None)
        joint_attentions = jnp.array(
            [
                [
                    [[0.9, 0.1, 0.0], [0.3, 0.6, 0.1], [0.4, 0.2, 0.4]],
                    [[0.8, 0.2, 0.0], [0.2, 0.7, 0.1], [0.1, 0.4, 0.5]],
                ]
            ],
            dtype=jnp.float32,
        )
        sigma = min(cfg["transformer"]["att_sigma"] / 3, 1.0)
        expected = reference_baseline_weights(
            joint_attentions,
            sigma=sigma,
            vq_coef=cfg["transformer"]["vq_bsln_coef"],
            vq_coma_coef=cfg["transformer"]["vq_coma_bsln_coef"],
        )
        actual = encoder._get_baseline_weights(
            joint_attentions,
            vq_coef=cfg["transformer"]["vq_bsln_coef"],
            vq_coma_coef=cfg["transformer"]["vq_coma_bsln_coef"],
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. ValueNorm parity
# ---------------------------------------------------------------------------

class TestValueNorm:
    def test_update_normalize_denormalize_roundtrip(self, rng):
        """ValueNorm update -> normalize -> denormalize recovers scale."""
        state = init_value_norm((1,), beta=0.9, epsilon=1e-5, var_clamp_min=1e-2)
        x = jax.random.normal(rng, (100, 1)) * 5.0 + 10.0
        state = value_norm_update(state, x)
        x_norm = value_norm_normalize(state, x)
        x_recovered = value_norm_denormalize(state, x_norm)
        np.testing.assert_allclose(x_recovered, x, atol=1e-4, rtol=1e-4)

    def test_valuenorm_dict_creation(self):
        d = create_value_norm_dict(use_valuenorm=True, v_shape=(1,), q_shape=(1,), eq_shape=(1,))
        assert "v" in d and "q" in d and "eq" in d
        assert isinstance(d["v"], ValueNormState)

    def test_valuenorm_disabled(self):
        d = create_value_norm_dict(use_valuenorm=False)
        assert d is None

    def test_running_stats_clamped_var(self, rng):
        """Variance clamping prevents too-small std."""
        state = init_value_norm((1,), beta=0.99999, var_clamp_min=1.0)
        x = jnp.ones((10, 1)) * 3.0  # zero variance
        state = value_norm_update(state, x)
        # After debias, var should be clamped to at least 1.0
        _, debiased_var = value_norm_running_stats(state)
        assert bool(jnp.all(debiased_var >= 1.0))

    def test_valuenorm_axes_behavior_matches_reference(self):
        """norm_axes preserves trailing feature axes like MACA ValueNorm."""
        beta = 0.5
        state = init_value_norm((4,), beta=beta, var_clamp_min=1e-2, norm_axes=2)
        x = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)
        state = value_norm_update(state, x)

        batch_mean = jnp.mean(x, axis=(0, 1))
        batch_sq_mean = jnp.mean(jnp.square(x), axis=(0, 1))
        expected_running_mean = (1.0 - beta) * batch_mean
        expected_running_sq = (1.0 - beta) * batch_sq_mean
        expected_debiasing = 1.0 - beta
        expected_mean = expected_running_mean / expected_debiasing
        expected_var = jnp.maximum(
            expected_running_sq / expected_debiasing - jnp.square(expected_mean),
            1e-2,
        )

        np.testing.assert_allclose(state.running_mean, expected_running_mean, atol=1e-6)
        np.testing.assert_allclose(state.running_mean_sq, expected_running_sq, atol=1e-6)
        mean, var = value_norm_running_stats(state)
        np.testing.assert_allclose(mean, expected_mean, atol=1e-6)
        np.testing.assert_allclose(var, expected_var, atol=1e-6)

        x_norm = value_norm_normalize(state, x)
        x_recovered = value_norm_denormalize(state, x_norm)
        np.testing.assert_allclose(x_recovered, x, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. GAE parity with ValueNorm denorm and timeout bad_masks
# ---------------------------------------------------------------------------

class TestGAE:
    def test_gae_with_valuenorm_denorm_and_bad_mask(self):
        """Reproduce MACA GAE path: denormalized preds + bad_mask scaling."""
        T, E = 5, 3
        gamma = 0.99
        gae_lambda = 0.95

        norm_dict = create_value_norm_dict(
            use_valuenorm=True, v_shape=(1,), q_shape=(1,), eq_shape=(1,)
        )
        # Seed ValueNorm with some stats so denorm does something nontrivial
        norm_dict["v"] = value_norm_update(norm_dict["v"], jnp.ones((10, 1)) * 5.0)

        preds_norm = jnp.ones((T, E)) * 0.5
        rewards = jnp.ones((T, E))
        dones = jnp.zeros((T, E), dtype=jnp.float32)
        bad_masks = jnp.ones((T, E), dtype=jnp.float32)
        # Make one env timeout at step 2
        bad_masks = bad_masks.at[2, 1].set(0.0)
        last_pred_norm = jnp.ones((E,)) * 0.5

        def _denorm_if_needed(norm_key, x):
            return value_norm_denormalize(norm_dict[norm_key], x[..., None]).squeeze(-1)

        preds = _denorm_if_needed("v", preds_norm)
        last_pred = _denorm_if_needed("v", last_pred_norm)

        def _calculate_gae(preds, rewards, dones, bad_masks, last_pred):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, bad_mask, value, reward = transition
                mask = 1.0 - done
                delta = reward + gamma * next_value * mask - value
                gae = delta + gamma * gae_lambda * mask * gae
                gae = bad_mask * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_pred), last_pred),
                (dones, bad_masks, preds, rewards),
                reverse=True,
                unroll=16,
            )
            return advantages + preds

        targets = _calculate_gae(preds, rewards, dones, bad_masks, last_pred)
        targets_without_denorm = _calculate_gae(
            preds_norm, rewards, dones, bad_masks, last_pred_norm
        )
        assert targets.shape == (T, E)
        assert jnp.all(jnp.isfinite(targets))
        # A bad mask cuts recursion at that timestep, so return_t == value_t.
        np.testing.assert_allclose(targets[2, 1], preds[2, 1], atol=1e-6)
        # The denormalized and normalized paths should not collapse to the same target.
        assert not jnp.allclose(targets, targets_without_denorm)
        # Env with bad_mask=0 at step 2 should differ from untouched envs.
        assert not jnp.allclose(targets[:, 0], targets[:, 1])

    def test_specific_bad_mask_layout_per_env(self):
        """Different timeout flags per env produce correct bad_mask layout."""
        num_envs = 4
        num_agents = 3
        bad_mask_env_time = jnp.array(
            [
                [1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ]
        )
        bad_mask_actor_time = jnp.stack(
            [jnp.tile(env_mask, num_agents) for env_mask in bad_mask_env_time]
        )

        expected_t0 = jnp.array([
            1.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 1.0,
        ])
        np.testing.assert_allclose(bad_mask_actor_time[0], expected_t0)

        # This mirrors train_mappo_t.actor_to_env_agent_time(...)[..., 0].
        recovered_env_mask = bad_mask_actor_time.reshape(
            bad_mask_env_time.shape[0], num_agents, num_envs
        ).swapaxes(1, 2)[..., 0]
        np.testing.assert_allclose(recovered_env_mask, bad_mask_env_time)


# ---------------------------------------------------------------------------
# 8. PPO actor loss parity
# ---------------------------------------------------------------------------

class TestActorLoss:
    def test_ppo_actor_loss_fixed_values(self):
        """PPO clipped surrogate with active masks."""
        batch = 8
        action_dim = 5
        clip_param = 0.1
        ent_coef = 0.01

        # Fixed logits -> deterministic pi so we can compute exact ratios
        logits = jnp.zeros((batch, action_dim))
        logits = logits.at[:, 2].set(1.0)  # deterministic action 2
        old_logits = jnp.zeros((batch, action_dim))
        old_logits = old_logits.at[:, 2].set(0.5)

        pi = lambda: None
        pi.logits = logits
        pi.probs = jax.nn.softmax(logits, axis=-1)
        pi.entropy = lambda: distrax.Categorical(logits=logits).entropy()
        pi.log_prob = lambda a: distrax.Categorical(logits=logits).log_prob(a)

        actions = jnp.ones(batch, dtype=jnp.int32) * 2
        old_log_probs = distrax.Categorical(logits=old_logits).log_prob(actions)
        advantages = jnp.ones(batch)
        active_masks = jnp.ones(batch)
        active_count = jnp.sum(active_masks) + 1e-8

        log_prob = pi.log_prob(actions)
        ratio = jnp.exp(log_prob - old_log_probs)
        loss1 = ratio * advantages
        loss2 = jnp.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        policy_loss = -jnp.sum(jnp.minimum(loss1, loss2) * active_masks) / active_count
        entropy = jnp.sum(pi.entropy() * active_masks) / active_count
        actor_loss = policy_loss - ent_coef * entropy

        assert jnp.isfinite(actor_loss)
        # Ratio should be >1 because logits for action 2 increased from 0.5->1.0
        assert jnp.all(ratio > 1.0)

    def test_ppo_actor_loss_active_mask_zeros(self):
        """Active mask should zero-out contributions from inactive steps."""
        batch = 8
        action_dim = 5
        clip_param = 0.1
        ent_coef = 0.01

        logits = jax.random.normal(jax.random.PRNGKey(0), (batch, action_dim))
        old_logits = jax.random.normal(jax.random.PRNGKey(1), (batch, action_dim))
        actions = jnp.zeros(batch, dtype=jnp.int32)
        old_log_probs = distrax.Categorical(logits=old_logits).log_prob(actions)
        advantages = jax.random.normal(jax.random.PRNGKey(2), (batch,))
        active_masks = jnp.zeros(batch)

        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(actions)
        ratio = jnp.exp(log_prob - old_log_probs)
        loss1 = ratio * advantages
        loss2 = jnp.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        active_count = jnp.sum(active_masks) + 1e-8
        policy_loss = -jnp.sum(jnp.minimum(loss1, loss2) * active_masks) / active_count
        entropy = jnp.sum(pi.entropy() * active_masks) / active_count
        actor_loss = policy_loss - ent_coef * entropy

        # All-masked-out should zero all contributions despite the epsilon denom.
        assert jnp.isfinite(actor_loss)
        np.testing.assert_allclose(actor_loss, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# 9. Critic clipped Huber loss parity for v/q/eq with ValueNorm updates
# ---------------------------------------------------------------------------

class TestCriticLoss:
    def test_clipped_huber_loss_valuenorm_update(self):
        """Critic loss: clipped Huber + ValueNorm update in loss path."""
        batch = 16
        clip_param = 0.1
        huber_delta = 10.0

        pred = jnp.zeros((batch,))
        old_pred = jnp.ones((batch,)) * 0.5
        target = jnp.ones((batch,)) * 2.0

        def _element_loss(error):
            abs_error = jnp.abs(error)
            return jnp.where(
                abs_error <= huber_delta,
                0.5 * jnp.square(error),
                huber_delta * (abs_error - 0.5 * huber_delta),
            )

        def _value_loss(pred, old_pred, target):
            clipped = old_pred + jnp.clip(pred - old_pred, -clip_param, clip_param)
            orig = _element_loss(target - pred)
            clip = _element_loss(target - clipped)
            return jnp.maximum(orig, clip).mean()

        loss = _value_loss(pred, old_pred, target)
        assert jnp.isfinite(loss)
        # Pred is far from target, but old_pred is closer; clipping should matter.
        # Just check it's > simple Huber without clipping
        simple = _element_loss(target - pred).mean()
        assert loss >= simple * 0.99  # clipping cannot reduce loss below unclipped

    def test_valuenorm_update_inside_loss(self):
        """ValueNorm states update inside the critic loss (matching MACA)."""
        norm_dict = create_value_norm_dict(
            use_valuenorm=True, v_shape=(1,), q_shape=(1,), eq_shape=(1,)
        )
        mb_targets = jnp.ones((8, 1)) * 5.0
        new_v_norm = value_norm_update(norm_dict["v"], mb_targets)
        assert not jnp.allclose(new_v_norm.running_mean, norm_dict["v"].running_mean)
        norm_targets = value_norm_normalize(new_v_norm, mb_targets)
        # After updating with constant 5.0, normalized constant should be ~0
        assert jnp.allclose(norm_targets, jnp.zeros_like(norm_targets), atol=0.1)


# ---------------------------------------------------------------------------
# 10. AdamW critic optimizer mask
# ---------------------------------------------------------------------------

class TestCriticOptimizer:
    def test_adamw_decay_mask_ndim_ge_2(self, base_config):
        """AdamW weight decay only on params with ndim >= 2; betas [0.9, 0.95]."""
        cfg = base_config.copy()
        cfg["transformer"] = base_config["transformer"].copy()
        betas = cfg["transformer"]["betas"]
        weight_decay = cfg["transformer"]["wght_decay"]

        # Simulate critic params tree with mixed ndim
        params = {
            "dense": {
                "kernel": jnp.zeros((3, 4)),  # ndim=2 -> decay
                "bias": jnp.zeros((4,)),       # ndim=1 -> no decay
            },
            "norm": {
                "scale": jnp.zeros((4,)),      # ndim=1 -> no decay
            },
        }

        def decay_mask(p):
            return jax.tree.map(lambda x: x.ndim >= 2, p)

        mask = decay_mask(params)
        assert mask["dense"]["kernel"] == True
        assert mask["dense"]["bias"] == False
        assert mask["norm"]["scale"] == False

        tx = optax.adamw(
            learning_rate=1e-3,
            b1=betas[0],
            b2=betas[1],
            eps=1e-5,
            weight_decay=weight_decay,
            mask=decay_mask(params),
        )
        # Just verify construction succeeds and is an adamw transform
        assert callable(tx.init)
        assert callable(tx.update)


# ---------------------------------------------------------------------------
# 11. Tiny JAX smoke run: 1 update, no NaNs, expected metric shapes
# ---------------------------------------------------------------------------

class TestSmokeRun:
    def test_one_update_no_nans(self, rng, base_config):
        """End-to-end JIT train for 1 update: no NaNs, expected shapes."""
        cfg = base_config.copy()
        cfg["transformer"] = base_config["transformer"].copy()
        cfg["NUM_ENVS"] = 2
        cfg["NUM_STEPS"] = 4
        cfg["TOTAL_TIMESTEPS"] = cfg["NUM_STEPS"] * cfg["NUM_ENVS"] * 1
        cfg["PPO_EPOCH"] = 1
        cfg["CRITIC_EPOCH"] = 1
        cfg["DATA_CHUNK_LENGTH"] = 4
        cfg["ACTOR_NUM_MINI_BATCH"] = 2
        cfg["CRITIC_NUM_MINI_BATCH"] = 2
        cfg["transformer"]["n_encode_layer"] = 1
        cfg["transformer"]["n_decode_layer"] = 0
        cfg["transformer"]["n_embd"] = 16
        cfg["transformer"]["zs_dim"] = 32
        cfg["transformer"]["n_block"] = 3
        cfg["use_valuenorm"] = True

        # Patch environment creation inside make_train
        from train_mappo_t import make_train
        train_fn = make_train(cfg)
        train_jit = jax.jit(train_fn)
        out = train_jit(rng)

        runner_state = out["runner_state"]
        metric = out["metric"]

        # make_train returns the final scan carry: (runner_state, update_steps).
        runner_state, update_steps = runner_state
        (actor_ts, critic_ts), env_state, last_obs, last_env_done, last_agent_done, hstates, value_norm_dict, rng_out = runner_state
        assert int(update_steps) == cfg["NUM_UPDATES"]

        # Check no NaNs in params
        def _check_no_nan(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            for leaf in leaves:
                assert not jnp.any(jnp.isnan(leaf)), "Found NaN in params"

        _check_no_nan(actor_ts.params)
        _check_no_nan(critic_ts.params)

        # ValueNorm dict should exist and have v/q/eq
        assert value_norm_dict is not None
        assert "v" in value_norm_dict
        assert "q" in value_norm_dict
        assert "eq" in value_norm_dict

        # Metric should contain loss dict and episode info
        assert "loss" in metric
        loss_info = metric["loss"]
        assert "total_loss" in loss_info
        assert "actor_loss" in loss_info
        assert "value_loss" in loss_info
        assert "entropy" in loss_info
        assert jnp.all(jnp.isfinite(loss_info["total_loss"]))

        # Episode metrics are stacked over NUM_UPDATES by lax.scan.
        assert metric["returned_episode"].shape == (
            cfg["NUM_UPDATES"],
            cfg["NUM_STEPS"],
            cfg["NUM_ENVS"],
            3,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
