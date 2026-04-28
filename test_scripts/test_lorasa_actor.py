"""Tests for LoRASA actor modules.

These tests validate:
1. Zero-LoRA equivalence with the original ActorTrans.
2. Adapter routing actually changes output.
3. Frozen backbone: non-LoRA gradients are zero and masked optimizer leaves them unchanged.
4. Checkpoint round-trip preserves all required keys.

Run on a JAX-enabled environment (e.g., Colab) with:
    pytest test_scripts/test_lorasa_actor.py -v
"""

from __future__ import annotations

import os
import pickle
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import optax

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smax_ctm.mappo_t.actor import ActorTrans, ScannedRNN
from smax_ctm.mappo_t.lorasa_actor import LoRASAActorTrans
from flax.traverse_util import flatten_dict, unflatten_dict


# Common test config matching a small actor
_TEST_CONFIG = {
    "hidden_sizes": [32, 32],
    "activation_func": "relu",
    "transformer": {"active_fn": "relu"},
    "use_feature_normalization": True,
    "use_naive_recurrent_policy": False,
    "use_recurrent_policy": True,
    "gain": 0.01,
}

_OBS_DIM = 16
_ACTION_DIM = 5
_BATCH = 4
_TIME = 3
_NUM_ADAPTERS = 9
_RANK = 4


def _init_actors():
    """Initialize original and LoRASA actors, overlay pretrained params."""
    actor_old = ActorTrans(action_dim=_ACTION_DIM, config=_TEST_CONFIG)
    actor_new = LoRASAActorTrans(
        action_dim=_ACTION_DIM,
        config=_TEST_CONFIG,
        num_adapter_slots=_NUM_ADAPTERS,
        rank=_RANK,
        init_scale=0.01,
    )

    rng = jax.random.PRNGKey(0)
    rng, r1, r2 = jax.random.split(rng, 3)

    hstate = ScannedRNN.initialize_carry(_BATCH, _TEST_CONFIG["hidden_sizes"][-1])
    obs = jnp.linspace(
        -1.0,
        1.0,
        _TIME * _BATCH * _OBS_DIM,
        dtype=jnp.float32,
    ).reshape((_TIME, _BATCH, _OBS_DIM))
    resets = jnp.zeros((_TIME, _BATCH), dtype=bool)
    avail = jnp.ones((_TIME, _BATCH, _ACTION_DIM), dtype=jnp.float32)

    params_old = actor_old.init(r1, hstate, (obs, resets, avail))

    adapter_ids = jnp.zeros((_TIME, _BATCH), dtype=jnp.int32)
    params_new = actor_new.init(r2, hstate, (obs, resets, avail), adapter_ids)

    # Overlay old params into new backbone
    new_flat = flatten_dict(params_new)
    old_flat = flatten_dict(params_old)
    for key, value in old_flat.items():
        if key not in new_flat:
            raise ValueError(f"Old param path {key} missing in new params")
        new_flat[key] = value
    params_new = unflatten_dict(new_flat)

    return actor_old, params_old, actor_new, params_new, hstate, obs, resets, avail


def test_zero_lora_equivalence_single_step():
    """Test 1a: Single-step forward with zero LoRA matches original exactly."""
    actor_old, params_old, actor_new, params_new, hstate, obs, resets, avail = _init_actors()

    # Single step: time=1
    obs_t = obs[:1]
    resets_t = resets[:1]
    avail_t = avail[:1]
    adapter_ids = jnp.zeros((1, _BATCH), dtype=jnp.int32)

    _, pi_old = actor_old.apply(params_old, hstate, (obs_t, resets_t, avail_t))
    _, pi_new = actor_new.apply(params_new, hstate, (obs_t, resets_t, avail_t), adapter_ids)

    max_diff = float(jnp.max(jnp.abs(pi_old.logits - pi_new.logits)))
    assert max_diff <= 1e-5, f"Single-step logits differ by {max_diff}"


def test_zero_lora_equivalence_recurrent_sequence():
    """Test 1b: Recurrent sequence with zero LoRA matches original exactly."""
    actor_old, params_old, actor_new, params_new, hstate, obs, resets, avail = _init_actors()

    adapter_ids = jnp.zeros((_TIME, _BATCH), dtype=jnp.int32)

    _, pi_old = actor_old.apply(params_old, hstate, (obs, resets, avail))
    _, pi_new = actor_new.apply(params_new, hstate, (obs, resets, avail), adapter_ids)

    max_diff = float(jnp.max(jnp.abs(pi_old.logits - pi_new.logits)))
    assert max_diff <= 1e-5, f"Recurrent logits differ by {max_diff}"


def test_zero_lora_equivalence_with_resets():
    """Test 1c: Equivalence holds when some resets are True."""
    actor_old, params_old, actor_new, params_new, hstate, obs, resets, avail = _init_actors()

    resets_mixed = resets.at[1, 2].set(True)
    adapter_ids = jnp.zeros((_TIME, _BATCH), dtype=jnp.int32)

    _, pi_old = actor_old.apply(params_old, hstate, (obs, resets_mixed, avail))
    _, pi_new = actor_new.apply(params_new, hstate, (obs, resets_mixed, avail), adapter_ids)

    max_diff = float(jnp.max(jnp.abs(pi_old.logits - pi_new.logits)))
    assert max_diff <= 1e-5, f"Reset-mixed logits differ by {max_diff}"


def test_adapter_routing_changes_output():
    """Test 2: Different adapter ids produce different logits when one adapter is nonzero."""
    actor_old, params_old, actor_new, params_new, hstate, obs, resets, avail = _init_actors()

    # Manually set adapter 3's action-head LoRA params to a non-uniform pattern.
    # A constant shift to every action logit is distribution-invariant, and
    # uniform hidden-layer shifts can be removed by LayerNorm.
    flat = flatten_dict(params_new)
    for key in list(flat.keys()):
        if "action_out" in key and "lora_a" in key:
            arr = flat[key]
            feature_pattern = jnp.linspace(-1.0, 1.0, arr.shape[1], dtype=arr.dtype)
            alt_pattern = jnp.where(
                jnp.arange(arr.shape[1]) % 2 == 0,
                jnp.asarray(0.5, dtype=arr.dtype),
                jnp.asarray(-0.5, dtype=arr.dtype),
            )
            arr = arr.at[3, :, 0].set(feature_pattern)
            arr = arr.at[3, :, 1].set(alt_pattern)
            flat[key] = arr
        if "action_out" in key and "lora_b" in key:
            arr = flat[key]
            pattern = jnp.linspace(-1.0, 1.0, _ACTION_DIM, dtype=arr.dtype)
            alt_pattern = jnp.array([1.0, -0.5, 0.25, -1.0, 0.5], dtype=arr.dtype)
            arr = arr.at[3, 0, :].set(pattern)
            arr = arr.at[3, 1, :].set(alt_pattern)
            flat[key] = arr
    params_modified = unflatten_dict(flat)

    obs_t = obs[:1]
    resets_t = resets[:1]
    avail_t = avail[:1]

    adapter_a = jnp.zeros((1, _BATCH), dtype=jnp.int32)
    adapter_b = jnp.full((1, _BATCH), 3, dtype=jnp.int32)

    _, pi_a = actor_new.apply(params_modified, hstate, (obs_t, resets_t, avail_t), adapter_a)
    _, pi_b = actor_new.apply(params_modified, hstate, (obs_t, resets_t, avail_t), adapter_b)

    max_diff = float(jnp.max(jnp.abs(pi_a.logits - pi_b.logits)))
    assert max_diff > 1e-5, f"Adapter routing did not change output (diff={max_diff})"


def test_frozen_backbone_optimizer_update():
    """Test 3: Backbone has zero grad, and optimizer updates only LoRA leaves."""
    actor_old, params_old, actor_new, params_new, hstate, obs, resets, avail = _init_actors()

    adapter_ids = jnp.zeros((_TIME, _BATCH), dtype=jnp.int32)

    def loss_fn(p):
        _, pi = actor_new.apply(p, hstate, (obs, resets, avail), adapter_ids)
        return jnp.sum(pi.logits)

    grads = jax.grad(loss_fn)(params_new)
    flat_grads = flatten_dict(grads)
    lora_has_grad = False
    for key, grad in flat_grads.items():
        if "lora_a" in key or "lora_b" in key:
            lora_has_grad = lora_has_grad or bool(jnp.any(grad != 0))
        else:
            assert jnp.all(grad == 0), (
                f"Backbone param {key} has nonzero grad but should be frozen "
                f"(max abs grad = {float(jnp.max(jnp.abs(grad)))})"
            )
    assert lora_has_grad, "No LoRA parameter received a nonzero gradient"

    mask = unflatten_dict(
        {k: ("lora_a" in k or "lora_b" in k) for k in flatten_dict(params_new)}
    )
    tx = optax.masked(optax.sgd(learning_rate=0.1), mask)
    updates, _ = tx.update(grads, tx.init(params_new), params_new)
    params_updated = optax.apply_updates(params_new, updates)

    flat_before = flatten_dict(params_new)
    flat_after = flatten_dict(params_updated)
    lora_changed = False
    for key, before in flat_before.items():
        after = flat_after[key]
        changed = bool(jnp.any(before != after))
        if "lora_a" in key or "lora_b" in key:
            lora_changed = lora_changed or changed
        else:
            assert not changed, f"Backbone param {key} changed but should be frozen"
    assert lora_changed, "No LoRA parameter changed after a masked optimizer update"


def test_checkpoint_round_trip():
    """Test 4: Save and reload a LoRASA checkpoint; assert required keys present."""
    actor_old, params_old, actor_new, params_new, hstate, obs, resets, avail = _init_actors()

    checkpoint = {
        "model_type": "mappo_t_lorasa",
        "config": _TEST_CONFIG,
        "actor_params": params_new,
        "critic_params": {"dummy": jnp.array([1.0])},
        "value_norm_dict": {"v": {"mean": 0.0, "var": 1.0}},
        "lorasa": {
            "rank": _RANK,
            "num_adapter_slots": _NUM_ADAPTERS,
            "routing": "unit_type_id",
        },
    }

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
        pickle.dump(checkpoint, f)

    try:
        with open(path, "rb") as f:
            loaded = pickle.load(f)

        assert loaded["model_type"] == "mappo_t_lorasa"
        assert "actor_params" in loaded
        assert "critic_params" in loaded
        assert "value_norm_dict" in loaded
        assert "lorasa" in loaded
        assert loaded["lorasa"]["rank"] == _RANK

        # Verify actor params have LoRA leaves
        flat = flatten_dict(loaded["actor_params"])
        lora_keys = [k for k in flat if "lora_a" in k or "lora_b" in k]
        assert len(lora_keys) > 0, "No LoRA keys found in loaded actor params"
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_zero_lora_equivalence_single_step()
    print("PASS: test_zero_lora_equivalence_single_step")

    test_zero_lora_equivalence_recurrent_sequence()
    print("PASS: test_zero_lora_equivalence_recurrent_sequence")

    test_zero_lora_equivalence_with_resets()
    print("PASS: test_zero_lora_equivalence_with_resets")

    test_adapter_routing_changes_output()
    print("PASS: test_adapter_routing_changes_output")

    test_frozen_backbone_optimizer_update()
    print("PASS: test_frozen_backbone_optimizer_update")

    test_checkpoint_round_trip()
    print("PASS: test_checkpoint_round_trip")

    print("\nAll LoRASA actor tests passed!")
