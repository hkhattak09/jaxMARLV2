"""Safety tests for MAPPO-T JAX optimization rewrites.

Run from repo root:
    pytest test_scripts/test_mappo_t_safety_optimizations.py -q

These tests are intentionally focused on semantic equivalence for the recent
performance changes:
  * transformer critic sequence scan
  * vectorized v/q/eq GAE
  * pre-shuffle minibatch slicing
  * ValueNorm update-before-normalize ordering outside autodiff
  * eval moved behind a host callback / separate jit
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
SMAX_CTM = REPO_ROOT / "smax_ctm"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SMAX_CTM) not in sys.path:
    sys.path.insert(0, str(SMAX_CTM))

pytest.importorskip("distrax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("chex")

from jaxmarl.environments.spaces import Discrete
from mappo_t import TransVCritic, get_default_mappo_t_config
from mappo_t.valuenorm import (
    create_value_norm_dict,
    value_norm_normalize,
    value_norm_update,
)


def _small_critic_config():
    cfg = get_default_mappo_t_config()
    cfg["NUM_ENVS"] = 2
    cfg["NUM_STEPS"] = 4
    cfg["TOTAL_TIMESTEPS"] = cfg["NUM_ENVS"] * cfg["NUM_STEPS"]
    cfg["use_recurrent_policy"] = True
    cfg["use_naive_recurrent_policy"] = False
    cfg["use_feature_normalization"] = True
    cfg["transformer"] = cfg["transformer"].copy()
    cfg["transformer"]["n_encode_layer"] = 1
    cfg["transformer"]["n_decode_layer"] = 0
    cfg["transformer"]["n_head"] = 1
    cfg["transformer"]["n_embd"] = 16
    cfg["transformer"]["zs_dim"] = 32
    cfg["transformer"]["dropout"] = 0.0
    cfg["transformer"]["n_block"] = 3
    cfg["hidden_sizes"] = [16, 16, 16]
    return cfg


def _assert_tree_allclose(actual, expected, *, atol=1e-6, rtol=1e-6):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    flat_actual = jax.tree_util.tree_leaves(actual)
    flat_expected = jax.tree_util.tree_leaves(expected)
    assert len(flat_actual) == len(flat_expected)
    for a, e in zip(flat_actual, flat_expected):
        np.testing.assert_allclose(np.asarray(a), np.asarray(e), atol=atol, rtol=rtol)


def _gae_scan(preds, rewards, dones, bad_masks, last_pred, gamma=0.99, gae_lambda=0.95):
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
        unroll=4,
    )
    return advantages + preds


class TestCriticSequenceScan:
    @pytest.mark.parametrize("output_attentions", [True, False])
    def test_sequence_scan_matches_manual_single_step_unroll(self, output_attentions):
        cfg = _small_critic_config()
        num_agents = 3
        action_dim = 5
        obs_dim = 7
        critic = TransVCritic(
            config=cfg,
            share_obs_space=None,
            obs_space=None,
            act_space=Discrete(action_dim),
            num_agents=num_agents,
            state_type="EP",
        )

        rng = jax.random.PRNGKey(7)
        init_rng, obs_rng, action_rng, policy_rng = jax.random.split(rng, 4)
        time, batch = 4, 2
        obs = jax.random.normal(obs_rng, (time, batch, num_agents, obs_dim))
        action = jax.random.randint(
            action_rng, (time, batch, num_agents), minval=0, maxval=action_dim
        )
        policy_logits = jax.random.normal(policy_rng, (time, batch, num_agents, action_dim))
        policy_prob = jax.nn.softmax(policy_logits, axis=-1)
        rnn = jnp.zeros((batch, num_agents, cfg["transformer"]["n_embd"]))
        resets = jnp.array(
            [
                [[False, False, False], [False, False, False]],
                [[False, True, False], [False, False, True]],
                [[False, False, False], [True, False, False]],
                [[False, False, False], [False, False, False]],
            ],
            dtype=bool,
        )

        params = critic.init(
            init_rng, obs, action, policy_prob, rnn, resets, output_attentions, True
        )
        scanned = critic.apply(
            params, obs, action, policy_prob, rnn, resets, output_attentions, True
        )

        carry = rnn
        per_timestep = []
        for t in range(time):
            out = critic.apply(
                params,
                obs[t],
                action[t],
                policy_prob[t],
                carry,
                resets[t],
                output_attentions,
                True,
            )
            per_timestep.append(out[:-1])
            carry = out[-1]

        manual = tuple(
            None if per_timestep[0][i] is None else jnp.stack([step[i] for step in per_timestep])
            for i in range(len(per_timestep[0]))
        ) + (carry,)
        _assert_tree_allclose(scanned, manual)


class TestVectorizedGAE:
    def test_vectorized_v_q_eq_gae_matches_three_independent_scans(self):
        rng = jax.random.PRNGKey(11)
        pred_rng, reward_rng, done_rng, bad_rng, last_rng = jax.random.split(rng, 5)
        time, num_envs = 9, 4
        value_preds = jax.random.normal(pred_rng, (3, time, num_envs))
        rewards = jax.random.normal(reward_rng, (time, num_envs))
        dones = jax.random.bernoulli(done_rng, 0.2, (time, num_envs)).astype(jnp.float32)
        bad_masks = 1.0 - jax.random.bernoulli(bad_rng, 0.1, (time, num_envs)).astype(jnp.float32)
        last_preds = jax.random.normal(last_rng, (3, num_envs))

        separate = jnp.stack(
            [
                _gae_scan(value_preds[i], rewards, dones, bad_masks, last_preds[i])
                for i in range(3)
            ],
            axis=1,
        )
        stacked_preds = value_preds.swapaxes(0, 1)
        vectorized = _gae_scan(stacked_preds, rewards, dones, bad_masks, last_preds)

        np.testing.assert_allclose(vectorized, separate, atol=1e-6, rtol=1e-6)


class TestMinibatchShuffle:
    def test_feedforward_pre_shuffle_matches_take_inside_scan(self):
        sample_count = 24
        num_minibatches = 4
        minibatch_size = sample_count // num_minibatches
        perm = jnp.array([5, 0, 22, 9, 7, 13, 3, 19, 1, 8, 11, 20, 2, 10, 4, 6, 18, 12, 14, 15, 16, 17, 21, 23])
        x = jnp.arange(sample_count * 3).reshape(sample_count, 3)
        y = jnp.arange(sample_count)
        minibatch_idx = perm.reshape(num_minibatches, minibatch_size)

        def old_body(_, mb_idx):
            return None, (jnp.take(x, mb_idx, axis=0), jnp.take(y, mb_idx, axis=0))

        _, old_batches = jax.lax.scan(old_body, None, minibatch_idx)

        shuffled_x = jnp.take(x, perm, axis=0).reshape(num_minibatches, minibatch_size, 3)
        shuffled_y = jnp.take(y, perm, axis=0).reshape(num_minibatches, minibatch_size)

        def new_body(_, i):
            return None, (shuffled_x[i], shuffled_y[i])

        _, new_batches = jax.lax.scan(new_body, None, jnp.arange(num_minibatches))
        _assert_tree_allclose(new_batches, old_batches)

    def test_recurrent_pre_shuffle_matches_take_inside_scan_with_time_swap(self):
        chunks = 12
        chunk_len = 5
        num_minibatches = 3
        minibatch_size = chunks // num_minibatches
        perm = jnp.array([8, 2, 0, 7, 4, 3, 1, 10, 11, 5, 9, 6])
        seq = jnp.arange(chunks * chunk_len * 2).reshape(chunks, chunk_len, 2)
        init_h = jnp.arange(chunks * 3).reshape(chunks, 3)
        minibatch_idx = perm.reshape(num_minibatches, minibatch_size)

        def old_body(_, mb_idx):
            return None, (
                jnp.take(seq, mb_idx, axis=0).swapaxes(0, 1),
                jnp.take(init_h, mb_idx, axis=0),
            )

        _, old_batches = jax.lax.scan(old_body, None, minibatch_idx)

        shuffled_seq = jnp.take(seq, perm, axis=0).reshape(
            num_minibatches, minibatch_size, chunk_len, 2
        )
        shuffled_init_h = jnp.take(init_h, perm, axis=0).reshape(
            num_minibatches, minibatch_size, 3
        )

        def new_body(_, i):
            return None, (shuffled_seq[i].swapaxes(0, 1), shuffled_init_h[i])

        _, new_batches = jax.lax.scan(new_body, None, jnp.arange(num_minibatches))
        _assert_tree_allclose(new_batches, old_batches)


class TestValueNormOrdering:
    def test_update_before_normalize_outside_grad_matches_original_order(self):
        norm_dict = create_value_norm_dict(
            use_valuenorm=True, v_shape=(1,), q_shape=(1,), eq_shape=(1,)
        )
        targets = {
            "v": jnp.linspace(-2.0, 3.0, 8)[..., None],
            "q": jnp.linspace(1.0, 6.0, 8)[..., None],
            "eq": jnp.linspace(-4.0, 2.0, 8)[..., None],
        }
        params = {"w": jnp.array(0.25)}

        def original_loss_fn(p, nd):
            updated = {k: value_norm_update(nd[k], targets[k]) for k in ("v", "q", "eq")}
            normalized = {
                k: value_norm_normalize(updated[k], targets[k]).squeeze(-1)
                for k in ("v", "q", "eq")
            }
            pred = p["w"] * jnp.ones_like(normalized["v"])
            loss = sum(jnp.mean((pred - normalized[k]) ** 2) for k in ("v", "q", "eq"))
            return loss, updated

        def optimized_loss_fn(p, updated):
            normalized = {
                k: value_norm_normalize(updated[k], targets[k]).squeeze(-1)
                for k in ("v", "q", "eq")
            }
            pred = p["w"] * jnp.ones_like(normalized["v"])
            loss = sum(jnp.mean((pred - normalized[k]) ** 2) for k in ("v", "q", "eq"))
            return loss

        (old_loss, old_norm), old_grad = jax.value_and_grad(original_loss_fn, has_aux=True)(
            params, norm_dict
        )
        new_norm = {k: value_norm_update(norm_dict[k], targets[k]) for k in ("v", "q", "eq")}
        new_loss, new_grad = jax.value_and_grad(optimized_loss_fn)(params, new_norm)

        np.testing.assert_allclose(new_loss, old_loss, atol=1e-7, rtol=1e-7)
        _assert_tree_allclose(new_grad, old_grad)
        _assert_tree_allclose(new_norm, old_norm)

        stale_loss = optimized_loss_fn(params, norm_dict)
        assert not np.allclose(np.asarray(stale_loss), np.asarray(old_loss), atol=1e-4)


class TestEvalRefactor:
    def test_eval_is_separate_jit_and_called_through_callback(self):
        train_src = (SMAX_CTM / "train_mappo_t.py").read_text()

        assert "def _run_eval(eval_rng, actor_params):" in train_src
        assert "_run_eval_jit = jax.jit(_run_eval)" in train_src
        assert "jax.experimental.io_callback" in train_src
        assert "_run_eval_jit(er2, p)" in train_src
        assert "actor_train_state.params" in train_src
        assert "rng, eval_rng = jax.random.split(rng)" in train_src
        assert "lambda r: _run_eval(r, actor_train_state)" not in train_src

    def test_eval_function_no_longer_carries_train_state(self):
        import smax_ctm.train_mappo_t as train_mappo_t

        src = inspect.getsource(train_mappo_t.make_train)
        eval_start = src.index("def _run_eval(eval_rng, actor_params):")
        eval_end = src.index("_run_eval_jit = jax.jit(_run_eval)", eval_start)
        eval_src = src[eval_start:eval_end]

        assert "actor_train_state, env_s" not in eval_src
        assert "actor_train_state.params" not in eval_src
        assert "actor_network.apply(\n                    actor_params" in eval_src
