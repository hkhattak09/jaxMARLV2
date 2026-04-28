#!/usr/bin/env python3
"""Smoke tests for MAPPO-T's explicit-GRU actor and backbone checkpoint schema.

Run from the repository root, for example in Colab:

    python test_scripts/test_mappo_t_explicit_gru_actor.py

This script is intentionally lightweight. It does not run environment training.
It checks the new actor path directly, because the explicit GRU is the risky
change that future LoRASA GRU adapters will depend on.
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
from flax.traverse_util import flatten_dict

from mappo_t import ActorTrans, ScannedRNN, get_default_mappo_t_config


def assert_true(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def tree_paths(tree):
    flat = flatten_dict(tree, sep="/")
    return sorted(flat.keys())


def make_tiny_config():
    cfg = get_default_mappo_t_config()
    cfg["hidden_sizes"] = [16, 16, 16]
    cfg["use_recurrent_policy"] = True
    cfg["use_naive_recurrent_policy"] = False
    cfg["use_feature_normalization"] = True
    cfg["activation_func"] = "relu"
    cfg["transformer"] = dict(cfg["transformer"])
    cfg["transformer"]["active_fn"] = "gelu"
    return cfg


def init_actor(seed=0, batch=5, time=4, obs_dim=13, action_dim=7):
    cfg = make_tiny_config()
    actor = ActorTrans(action_dim=action_dim, config=cfg)
    h0 = ScannedRNN.initialize_carry(batch, cfg["hidden_sizes"][-1])
    obs = jax.random.normal(jax.random.PRNGKey(seed + 1), (time, batch, obs_dim))
    resets = jnp.zeros((time, batch), dtype=bool)
    avail = jnp.ones((time, batch, action_dim), dtype=jnp.float32)
    params = actor.init(jax.random.PRNGKey(seed), h0, (obs, resets, avail))
    return actor, params, h0, obs, resets, avail


def test_explicit_gru_param_tree():
    actor, params, *_ = init_actor()
    del actor
    names = tree_paths(params["params"])

    expected = [
        "rnn/gru_cell/input_reset/kernel",
        "rnn/gru_cell/input_update/kernel",
        "rnn/gru_cell/input_candidate/kernel",
        "rnn/gru_cell/recurrent_reset/kernel",
        "rnn/gru_cell/recurrent_update/kernel",
        "rnn/gru_cell/recurrent_candidate/kernel",
    ]
    for name in expected:
        assert_true(name in names, f"Missing explicit GRU parameter: {name}")

    forbidden_fragments = ["GRUCell", "hf/", "hr/", "hz/"]
    for frag in forbidden_fragments:
        assert_true(
            not any(frag in name for name in names),
            f"Found old/opaque GRU parameter fragment {frag}: {names}",
        )

    print("ok: explicit GRU parameter names are present")


def test_forward_shapes_and_action_masking():
    actor, params, h0, obs, resets, avail = init_actor()

    # Mask action 3 everywhere. Its probability should become ~0.
    avail = avail.at[:, :, 3].set(0.0)
    h1, pi = actor.apply(params, h0, (obs, resets, avail))

    assert_true(h1.shape == h0.shape, f"Bad hidden shape: {h1.shape} vs {h0.shape}")
    assert_true(pi.logits.shape == avail.shape, f"Bad logits shape: {pi.logits.shape}")
    np.testing.assert_allclose(pi.probs[..., 3], 0.0, atol=1e-6)
    np.testing.assert_allclose(pi.probs.sum(axis=-1), 1.0, atol=1e-6)

    actions = pi.sample(seed=jax.random.PRNGKey(99))
    log_probs = pi.log_prob(actions)
    assert_true(actions.shape == obs.shape[:2], f"Bad action shape: {actions.shape}")
    assert_true(log_probs.shape == obs.shape[:2], f"Bad log-prob shape: {log_probs.shape}")

    print("ok: forward shapes and action masking work")


def test_reset_semantics_match_segmented_run():
    actor, params, h0, obs, resets, avail = init_actor(batch=3, time=5)

    # Force every actor to reset at t=3. A full scan with reset at t=3 should
    # match running the prefix, then restarting from zero carry on the suffix.
    resets = resets.at[3, :].set(True)
    full_h, full_pi = actor.apply(params, h0, (obs, resets, avail))

    prefix_h, prefix_pi = actor.apply(
        params,
        h0,
        (obs[:3], resets[:3], avail[:3]),
    )
    del prefix_h, prefix_pi

    zero_h = ScannedRNN.initialize_carry(h0.shape[0], h0.shape[1])
    suffix_resets = resets[3:].at[0, :].set(True)
    suffix_h, suffix_pi = actor.apply(
        params,
        zero_h,
        (obs[3:], suffix_resets, avail[3:]),
    )

    np.testing.assert_allclose(full_pi.logits[3:], suffix_pi.logits, atol=1e-5)
    np.testing.assert_allclose(full_h, suffix_h, atol=1e-5)

    print("ok: recurrent reset semantics are stable")


def test_gradients_reach_explicit_gru_kernels():
    actor, params, h0, obs, resets, avail = init_actor()

    def loss_fn(p):
        _, pi = actor.apply(p, h0, (obs, resets, avail))
        return jnp.mean(jnp.square(pi.logits))

    grads = jax.grad(loss_fn)(params)
    flat_grads = flatten_dict(grads["params"], sep="/")
    gru_grad_paths = [
        path
        for path in flat_grads
        if path.startswith("rnn/gru_cell/") and path.endswith("/kernel")
    ]
    assert_true(gru_grad_paths, "No gradients found for explicit GRU kernels")
    grad_norm = optax.global_norm({path: flat_grads[path] for path in gru_grad_paths})
    assert_true(jnp.isfinite(grad_norm), f"Non-finite GRU grad norm: {grad_norm}")
    assert_true(float(grad_norm) > 0.0, "Explicit GRU kernel gradients are all zero")

    print("ok: gradients reach explicit GRU kernels")


def test_checkpoint_source_schema():
    src_path = os.path.join(SMAX_CTM, "train_mappo_t.py")
    with open(src_path, "r") as f:
        src = f.read()

    required = [
        '"model_type": "mappo_t_backbone"',
        '"checkpoint_kind": "periodic"',
        '"checkpoint_kind": "final"',
        '"actor_params":',
        '"critic_params":',
        '"value_norm_dict":',
        '"actor_opt_state":',
        '"critic_opt_state":',
        'os.path.join(run_dir, f"checkpoint_{s_int}.pkl")',
        'os.path.join(run_dir, "checkpoint_final.pkl")',
    ]
    for snippet in required:
        assert_true(snippet in src, f"Missing checkpoint schema snippet: {snippet}")

    forbidden = [
        'os.path.join(_REPO_ROOT, "saved_model")',
        'f"actor_{s_int}.pkl"',
        'f"critic_{s_int}.pkl"',
        'f"valuenorm_{s_int}.pkl"',
        'os.path.join(run_dir, f"step_{s_int}")',
    ]
    for snippet in forbidden:
        assert_true(snippet not in src, f"Found obsolete checkpoint pattern: {snippet}")

    print("ok: checkpoint schema/source layout is the new single-file format")


def main():
    print("Running MAPPO-T explicit-GRU actor smoke tests...")
    test_explicit_gru_param_tree()
    test_forward_shapes_and_action_masking()
    test_reset_semantics_match_segmented_run()
    test_gradients_reach_explicit_gru_kernels()
    test_checkpoint_source_schema()
    print("All explicit-GRU/checkpoint smoke tests passed.")


if __name__ == "__main__":
    main()
