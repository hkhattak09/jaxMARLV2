#!/usr/bin/env python3
"""Single-GPU population-axis Riemannian LoRASA-EGGROLL trainer.

This is the first throughput-oriented trainer for the phase-3 LoRASA work.  It
keeps the sequential trainer as an off-hot-path reference implementation, but
evaluates antithetic ES candidates in vmapped population chunks on one GPU.

One epoch means:

    population evaluation -> antithetic direction weights -> one adapter update

The user-facing rollout scale is expressed as ``episodes_per_candidate``.  The
``num_envs_per_candidate`` argument only controls how many of those episodes are
run in parallel for memory/throughput.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import pickle
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import jax.numpy as jnp
import numpy as np


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


import lorasa_eggroll as re  # noqa: E402
from eval_smax import build_env, validate_actor_params, validate_checkpoint  # noqa: E402
from mappo_t import LoRASAActorTrans, ScannedRNN  # noqa: E402
from mappo_t.utils import batchify, unbatchify  # noqa: E402


def _parse_slots(raw: str) -> Tuple[int, ...]:
    try:
        slots = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid slot list: {raw}") from e
    if not slots:
        raise argparse.ArgumentTypeError("Slot list cannot be empty")
    return slots


def _load_checkpoint(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with path.open("rb") as f:
        ckpt = pickle.load(f)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt).__name__}")
    return ckpt


def _save_checkpoint(ckpt: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(dict(ckpt), f, protocol=pickle.HIGHEST_PROTOCOL)


def _write_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metrics_to_json(metrics) -> List[Dict[str, Any]]:
    return [asdict(metric) for metric in metrics]


def _block_until_ready_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if leaves:
        jax.block_until_ready(leaves)
    return tree


def _summarize_update_metrics(metrics) -> Dict[str, Any]:
    if not metrics:
        return {
            "num_metrics": 0,
            "mean_delta_fro_norm": 0.0,
            "mean_step_fro_norm": 0.0,
            "max_step_fro_norm": 0.0,
            "mean_applied_update_fro_norm": 0.0,
            "max_applied_update_fro_norm": 0.0,
            "mean_singular_shift": 0.0,
            "max_singular_shift": 0.0,
        }

    delta_norms = np.asarray([m.delta_fro_norm for m in metrics], dtype=np.float64)
    step_norms = np.asarray([m.step_fro_norm for m in metrics], dtype=np.float64)
    applied_norms = np.asarray(
        [m.applied_update_fro_norm for m in metrics],
        dtype=np.float64,
    )
    singular_shifts = np.asarray(
        [
            float(
                np.linalg.norm(
                    np.asarray(m.retracted_singular_values, dtype=np.float64)
                    - np.asarray(m.singular_values, dtype=np.float64)
                )
            )
            for m in metrics
        ],
        dtype=np.float64,
    )
    return {
        "num_metrics": int(len(metrics)),
        "mean_delta_fro_norm": float(delta_norms.mean()),
        "mean_step_fro_norm": float(step_norms.mean()),
        "max_step_fro_norm": float(step_norms.max()),
        "mean_applied_update_fro_norm": float(applied_norms.mean()),
        "max_applied_update_fro_norm": float(applied_norms.max()),
        "mean_singular_shift": float(singular_shifts.mean()),
        "max_singular_shift": float(singular_shifts.max()),
    }


def make_eval_stats_fn(
    env,
    actor_network,
    actor_hidden_dim: int,
    eval_steps: int,
    num_envs: int,
    rollout_batches: int,
):
    """Create a deterministic evaluator returning wins, returns, and lengths."""

    num_agents = env.num_agents

    def _actor_apply(actor_params, ac_hstate, ac_in, env_state):
        unit_types_env = env_state.env_state.state.unit_types[:, :num_agents]
        adapter_ids = unit_types_env.T.reshape(num_envs * num_agents)
        adapter_ids = adapter_ids.astype(jnp.int32)
        ac_hstate, pi = actor_network.apply(
            actor_params, ac_hstate, ac_in, adapter_ids[None, :]
        )
        return ac_hstate, pi

    def eval_fn(rng, actor_params):
        def _outer_loop(rng, _):
            rng, reset_rng = jax.random.split(rng)
            reset_keys = jax.random.split(reset_rng, num_envs)
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

            ac_hstate = ScannedRNN.initialize_carry(
                num_envs * num_agents, actor_hidden_dim
            )
            last_done = jnp.zeros((num_envs * num_agents,), dtype=bool)
            already_recorded = jnp.zeros((num_envs,), dtype=bool)
            episode_won = jnp.zeros((num_envs,), dtype=jnp.float32)
            episode_return = jnp.zeros((num_envs,), dtype=jnp.float32)
            episode_length = jnp.zeros((num_envs,), dtype=jnp.float32)

            def _step(carry, _):
                (
                    env_state,
                    obsv,
                    ac_hstate,
                    last_done,
                    already_recorded,
                    episode_won,
                    episode_return,
                    episode_length,
                    rng,
                ) = carry
                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(step_rng, num_envs)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, num_envs * num_agents)
                )
                obs_batch = batchify(obsv, env.agents, num_envs * num_agents)
                ac_in = (
                    obs_batch[None, :],
                    last_done[None, :],
                    avail_actions[None, :],
                )

                ac_hstate, pi = _actor_apply(actor_params, ac_hstate, ac_in, env_state)
                action = jnp.argmax(pi.logits, axis=-1).squeeze(0)

                env_act = unbatchify(
                    action, env.agents, num_envs, num_envs * num_agents
                )
                env_act = {k: v.squeeze(-1) for k, v in env_act.items()}

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_keys, env_state, env_act)
                del reward

                done_env = done["__all__"]
                new_record = done_env & ~already_recorded
                episode_won = jnp.where(
                    new_record, info["returned_won_episode"][:, 0], episode_won
                )
                episode_return = jnp.where(
                    new_record, info["returned_episode_returns"][:, 0], episode_return
                )
                episode_length = jnp.where(
                    new_record,
                    info["returned_episode_lengths"][:, 0].astype(jnp.float32),
                    episode_length,
                )
                already_recorded = already_recorded | new_record
                last_done = jnp.tile(done["__all__"], num_agents)

                return (
                    env_state,
                    obsv,
                    ac_hstate,
                    last_done,
                    already_recorded,
                    episode_won,
                    episode_return,
                    episode_length,
                    rng,
                ), None

            init_carry = (
                env_state,
                obsv,
                ac_hstate,
                last_done,
                already_recorded,
                episode_won,
                episode_return,
                episode_length,
                rng,
            )
            final_carry, _ = jax.lax.scan(_step, init_carry, None, eval_steps)
            (
                _,
                _,
                _,
                _,
                already_recorded,
                episode_won,
                episode_return,
                episode_length,
                rng,
            ) = final_carry

            episode_won = jnp.where(already_recorded, episode_won, 0.0)
            episode_return = jnp.where(already_recorded, episode_return, 0.0)
            episode_length = jnp.where(
                already_recorded, episode_length, float(eval_steps)
            )
            stats = {
                "wins": episode_won,
                "returns": episode_return,
                "lengths": episode_length,
                "recorded": already_recorded.astype(jnp.float32),
            }
            return rng, stats

        _, stats_by_loop = jax.lax.scan(_outer_loop, rng, None, rollout_batches)
        return stats_by_loop

    return jax.jit(eval_fn)


def _build_eval_stats(
    ckpt: Mapping[str, Any],
    checkpoint_path: str,
    map_name: str,
    num_envs: int,
    rollout_batches: int,
    max_steps: Optional[int],
):
    validate_checkpoint(dict(ckpt), checkpoint_path)
    if ckpt["model_type"] != "mappo_t_lorasa":
        raise ValueError(
            "Riemannian LoRASA-EGGROLL requires "
            f"model_type='mappo_t_lorasa', got {ckpt['model_type']!r}"
        )

    config = copy.deepcopy(ckpt["config"])
    env = build_env(map_name, config, max_steps=max_steps)
    eval_steps = (
        max_steps
        if max_steps is not None
        else config.get("ENV_KWARGS", {}).get("max_steps", config.get("NUM_STEPS", 200))
    )

    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    actor_hidden_dim = config["hidden_sizes"][-1]

    if "transformer" not in config:
        config["transformer"] = {}
    config["transformer"]["n_block"] = env.num_agents

    lorasa = ckpt["lorasa"]
    actor_network = LoRASAActorTrans(
        action_dim=action_dim,
        config=config,
        num_adapter_slots=lorasa["num_adapter_slots"],
        rank=lorasa["rank"],
        init_scale=lorasa.get("init_scale", config.get("LORASA_INIT_SCALE", 0.01)),
        ablation_mode=lorasa["ablation_mode"],
    )

    rng = jax.random.PRNGKey(0)
    init_ac_hstate = ScannedRNN.initialize_carry(1, actor_hidden_dim)
    init_ac_in = (
        jnp.zeros((1, 1, obs_dim), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=bool),
        jnp.ones((1, 1, action_dim), dtype=jnp.float32),
    )
    init_adapter_ids = jnp.zeros((1, 1), dtype=jnp.int32)
    init_params = actor_network.init(rng, init_ac_hstate, init_ac_in, init_adapter_ids)
    validate_actor_params(init_params, ckpt["actor_params"], "initial checkpoint")

    eval_fn = make_eval_stats_fn(
        env,
        actor_network,
        actor_hidden_dim,
        eval_steps,
        num_envs,
        rollout_batches,
    )
    return eval_fn, config, eval_steps


def _make_population_in_axes(
    actor_params: Any,
    active_slots: Sequence[int],
    target_rank: int,
    block_patterns: Sequence[str] = re.NO_RECURRENT_BLOCK_PATTERNS,
) -> Any:
    del active_slots, target_rank
    flat = re.flatten_tree(actor_params)
    axis_flat: Dict[Tuple[str, ...], Optional[int]] = {key: None for key in flat}
    selected = re.select_lora_blocks(re.discover_lora_blocks(actor_params), block_patterns)
    for block in selected:
        axis_flat[block.lora_a_key] = 0
        axis_flat[block.lora_b_key] = 0
    return re.unflatten_tree(axis_flat)


def _thread_to_direction_and_sign(thread_id: int) -> Tuple[int, int, str]:
    direction_id = int(thread_id) // 2
    if int(thread_id) % 2 == 0:
        return direction_id, 1, "plus"
    return direction_id, -1, "minus"


def _jax_key_for_direction(
    base_seed: int,
    epoch,
    direction_id,
    block_slot_seed: int,
):
    key = jax.random.PRNGKey(int(base_seed))
    key = jax.random.fold_in(key, jnp.asarray(epoch, dtype=jnp.uint32))
    key = jax.random.fold_in(key, jnp.asarray(direction_id, dtype=jnp.uint32))
    key = jax.random.fold_in(key, jnp.asarray(block_slot_seed, dtype=jnp.uint32))
    return key


def _jax_thin_svd(delta, target_rank: int):
    u, s, vt = jnp.linalg.svd(delta, full_matrices=False)
    return u[..., :target_rank], s[..., :target_rank], vt[..., :target_rank, :]


def _jax_tangent_project(z, u, vt):
    v = jnp.swapaxes(vt, -1, -2)
    z_v_vt = jnp.matmul(jnp.matmul(z, v), vt)
    u_t = jnp.swapaxes(u, -1, -2)
    u_ut_z = jnp.matmul(u, jnp.matmul(u_t, z))
    u_ut_z_v_vt = jnp.matmul(
        u,
        jnp.matmul(jnp.matmul(jnp.matmul(u_t, z), v), vt),
    )
    return z_v_vt + u_ut_z - u_ut_z_v_vt


def _jax_balanced_factors_from_svd(
    u,
    s,
    vt,
    configured_rank: int,
    out_a_shape: Tuple[int, ...],
    out_b_shape: Tuple[int, ...],
    out_dtype,
    singular_floor: float,
):
    sqrt_s = jnp.sqrt(jnp.maximum(s, jnp.asarray(singular_floor, dtype=s.dtype)))
    active_a = u * sqrt_s[..., None, :]
    active_b = sqrt_s[..., :, None] * vt
    a = jnp.zeros(out_a_shape, dtype=out_dtype)
    b = jnp.zeros(out_b_shape, dtype=out_dtype)
    active_rank = active_a.shape[-1]
    del configured_rank
    a = a.at[..., :active_rank].set(active_a.astype(out_dtype))
    b = b.at[..., :active_rank, :].set(active_b.astype(out_dtype))
    return a, b


def _jax_low_rank_ambient(
    direction_id,
    rows: int,
    cols: int,
    noise_rank: int,
    base_seed: int,
    epoch,
    block_slot_seed: int,
    dtype,
):
    key = _jax_key_for_direction(base_seed, epoch, direction_id, block_slot_seed)
    p_key, q_key = jax.random.split(key)
    p = jax.random.normal(p_key, (rows, noise_rank), dtype=dtype)
    q = jax.random.normal(q_key, (noise_rank, cols), dtype=dtype)
    return (p @ q) / math.sqrt(noise_rank)


def _jax_step_from_svd(
    direction_id,
    delta,
    u,
    vt,
    noise_rank: int,
    base_seed: int,
    epoch,
    block_slot_seed: int,
    relative_scale: bool,
):
    rows = int(delta.shape[0])
    cols = int(delta.shape[1])
    z = _jax_low_rank_ambient(
        direction_id,
        rows=rows,
        cols=cols,
        noise_rank=noise_rank,
        base_seed=base_seed,
        epoch=epoch,
        block_slot_seed=block_slot_seed,
        dtype=delta.dtype,
    )
    tangent = _jax_tangent_project(z, u, vt)
    tangent_norm = jnp.sqrt(jnp.sum(jnp.square(tangent)))
    unit_step = jnp.where(
        tangent_norm > 0.0,
        tangent / jnp.maximum(tangent_norm, jnp.asarray(1e-20, dtype=delta.dtype)),
        jnp.zeros_like(tangent),
    )
    if relative_scale:
        delta_norm = jnp.sqrt(jnp.sum(jnp.square(delta)))
        unit_step = unit_step * jnp.maximum(delta_norm, jnp.asarray(1e-8, dtype=delta.dtype))
    step_norm = jnp.sqrt(jnp.sum(jnp.square(unit_step)))
    return tangent, unit_step, tangent_norm, step_norm


def _jax_candidate_factors_for_slot(
    a_slot,
    b_slot,
    direction_ids,
    signs,
    sigma: float,
    target_rank: int,
    noise_rank: int,
    base_seed: int,
    epoch,
    block_slot_seed: int,
    relative_scale: bool,
    singular_floor: float,
):
    batch_size = int(direction_ids.shape[0])
    if float(sigma) == 0.0:
        return (
            jnp.broadcast_to(a_slot, (batch_size,) + tuple(a_slot.shape)),
            jnp.broadcast_to(b_slot, (batch_size,) + tuple(b_slot.shape)),
        )

    work_dtype = jnp.float32
    a_work = a_slot.astype(work_dtype)
    b_work = b_slot.astype(work_dtype)
    delta = a_work @ b_work
    u, _, vt = _jax_thin_svd(delta, target_rank)

    def _one(direction_id, sign):
        _, step, _, _ = _jax_step_from_svd(
            direction_id,
            delta,
            u,
            vt,
            noise_rank=noise_rank,
            base_seed=base_seed,
            epoch=epoch,
            block_slot_seed=block_slot_seed,
            relative_scale=relative_scale,
        )
        return delta + sign.astype(work_dtype) * jnp.asarray(sigma, dtype=work_dtype) * step

    candidate_delta = jax.vmap(_one)(direction_ids, signs)
    u_new, s_new, vt_new = _jax_thin_svd(candidate_delta, target_rank)
    new_a, new_b = _jax_balanced_factors_from_svd(
        u_new,
        s_new,
        vt_new,
        configured_rank=int(a_slot.shape[-1]),
        out_a_shape=(batch_size,) + tuple(a_slot.shape),
        out_b_shape=(batch_size,) + tuple(b_slot.shape),
        out_dtype=a_slot.dtype,
        singular_floor=singular_floor,
    )
    return new_a, new_b


def _make_device_candidate_builder(
    actor_params: Any,
    active_slots: Sequence[int],
    target_rank: int,
    noise_rank: int,
    sigma: float,
    base_seed: int,
    relative_scale: bool,
    singular_floor: float,
    block_patterns: Sequence[str] = re.NO_RECURRENT_BLOCK_PATTERNS,
):
    selected = re.select_lora_blocks(re.discover_lora_blocks(actor_params), block_patterns)
    active_slots = tuple(sorted(int(slot) for slot in active_slots))
    for block in selected:
        if target_rank > block.configured_rank:
            raise ValueError(
                f"target_rank {target_rank} exceeds configured rank "
                f"{block.configured_rank} at {block.path}"
            )
        for slot in active_slots:
            if slot < 0 or slot >= block.num_slots:
                raise IndexError(f"slot {slot} out of range for {block.path}")
    block_slot_seeds = {
        (block.path, slot): re.stable_uint32_seed("device_candidate", block.path, slot)
        for block in selected
        for slot in active_slots
    }

    def _build(params, thread_ids, epoch):
        flat = re.flatten_tree(params)
        batch_flat: Dict[Tuple[str, ...], Any] = dict(flat)
        direction_ids = (thread_ids // 2).astype(jnp.uint32)
        signs = jnp.where(thread_ids % 2 == 0, 1.0, -1.0)
        batch_size = int(thread_ids.shape[0])

        for block in selected:
            base_a = jnp.asarray(flat[block.lora_a_key])
            base_b = jnp.asarray(flat[block.lora_b_key])
            batch_a = jnp.broadcast_to(base_a, (batch_size,) + tuple(base_a.shape))
            batch_b = jnp.broadcast_to(base_b, (batch_size,) + tuple(base_b.shape))

            for slot in active_slots:
                seed = block_slot_seeds[(block.path, slot)]
                new_a, new_b = _jax_candidate_factors_for_slot(
                    base_a[slot],
                    base_b[slot],
                    direction_ids,
                    signs,
                    sigma=sigma,
                    target_rank=target_rank,
                    noise_rank=noise_rank,
                    base_seed=base_seed,
                    epoch=epoch,
                    block_slot_seed=seed,
                    relative_scale=relative_scale,
                    singular_floor=singular_floor,
                )
                batch_a = batch_a.at[:, slot].set(new_a)
                batch_b = batch_b.at[:, slot].set(new_b)

            batch_flat[block.lora_a_key] = batch_a
            batch_flat[block.lora_b_key] = batch_b

        return re.unflatten_tree(batch_flat)

    return jax.jit(_build), selected, active_slots


def _jax_weighted_update_for_slot(
    a_slot,
    b_slot,
    direction_ids,
    direction_weights,
    eta: float,
    target_rank: int,
    noise_rank: int,
    base_seed: int,
    epoch,
    block_slot_seed: int,
    direction_normalizer: int,
    relative_scale: bool,
    singular_floor: float,
):
    work_dtype = jnp.float32
    a_work = a_slot.astype(work_dtype)
    b_work = b_slot.astype(work_dtype)
    delta = a_work @ b_work
    u, s_old, vt = _jax_thin_svd(delta, target_rank)

    def _one(direction_id, weight):
        tangent, step, tangent_norm, step_norm = _jax_step_from_svd(
            direction_id,
            delta,
            u,
            vt,
            noise_rank=noise_rank,
            base_seed=base_seed,
            epoch=epoch,
            block_slot_seed=block_slot_seed,
            relative_scale=relative_scale,
        )
        nonzero = weight != 0.0
        weighted_step = weight.astype(work_dtype) * step
        tangent_norm = jnp.where(nonzero, tangent_norm, 0.0)
        step_norm = jnp.where(nonzero, step_norm, 0.0)
        return weighted_step, tangent_norm, step_norm

    weighted_steps, tangent_norms, step_norms = jax.vmap(_one)(
        direction_ids,
        direction_weights.astype(work_dtype),
    )
    aggregate = jnp.sum(weighted_steps, axis=0) / jnp.asarray(
        max(1, int(direction_normalizer)),
        dtype=work_dtype,
    )
    applied_update = jnp.asarray(eta, dtype=work_dtype) * aggregate
    updated_delta = delta + applied_update
    u_new, s_new, vt_new = _jax_thin_svd(updated_delta, target_rank)
    new_a, new_b = _jax_balanced_factors_from_svd(
        u_new,
        s_new,
        vt_new,
        configured_rank=int(a_slot.shape[-1]),
        out_a_shape=tuple(a_slot.shape),
        out_b_shape=tuple(b_slot.shape),
        out_dtype=a_slot.dtype,
        singular_floor=singular_floor,
    )
    delta_norm = jnp.sqrt(jnp.sum(jnp.square(delta)))
    applied_norm = jnp.sqrt(jnp.sum(jnp.square(applied_update)))
    return (
        new_a,
        new_b,
        delta_norm,
        jnp.sum(tangent_norms),
        jnp.sum(step_norms),
        s_old,
        s_new,
        applied_norm,
    )


def _make_device_update_fn(
    actor_params: Any,
    selected_blocks: Sequence[re.LoRABlock],
    active_slots: Sequence[int],
    target_rank: int,
    noise_rank: int,
    eta: float,
    base_seed: int,
    direction_normalizer: int,
    relative_scale: bool,
    singular_floor: float,
):
    for block in selected_blocks:
        if target_rank > block.configured_rank:
            raise ValueError(
                f"target_rank {target_rank} exceeds configured rank "
                f"{block.configured_rank} at {block.path}"
            )
        for slot in active_slots:
            if slot < 0 or slot >= block.num_slots:
                raise IndexError(f"slot {slot} out of range for {block.path}")
    block_slot_seeds = {
        (block.path, slot): re.stable_uint32_seed("device_candidate", block.path, slot)
        for block in selected_blocks
        for slot in active_slots
    }
    del actor_params

    def _update(params, direction_weights, epoch):
        flat = re.flatten_tree(params)
        updated_flat: Dict[Tuple[str, ...], Any] = dict(flat)
        direction_ids = jnp.arange(direction_weights.shape[0], dtype=jnp.uint32)
        delta_norms = []
        tangent_norms = []
        step_norms = []
        singular_values = []
        retracted_singular_values = []
        applied_norms = []

        for block in selected_blocks:
            lora_a = jnp.asarray(flat[block.lora_a_key])
            lora_b = jnp.asarray(flat[block.lora_b_key])
            new_lora_a = lora_a
            new_lora_b = lora_b

            for slot in active_slots:
                seed = block_slot_seeds[(block.path, slot)]
                (
                    new_a,
                    new_b,
                    delta_norm,
                    tangent_norm,
                    step_norm,
                    s_old,
                    s_new,
                    applied_norm,
                ) = _jax_weighted_update_for_slot(
                    lora_a[slot],
                    lora_b[slot],
                    direction_ids,
                    direction_weights,
                    eta=eta,
                    target_rank=target_rank,
                    noise_rank=noise_rank,
                    base_seed=base_seed,
                    epoch=epoch,
                    block_slot_seed=seed,
                    direction_normalizer=direction_normalizer,
                    relative_scale=relative_scale,
                    singular_floor=singular_floor,
                )
                new_lora_a = new_lora_a.at[slot].set(new_a)
                new_lora_b = new_lora_b.at[slot].set(new_b)
                delta_norms.append(delta_norm)
                tangent_norms.append(tangent_norm)
                step_norms.append(step_norm)
                singular_values.append(s_old)
                retracted_singular_values.append(s_new)
                applied_norms.append(applied_norm)

            updated_flat[block.lora_a_key] = new_lora_a
            updated_flat[block.lora_b_key] = new_lora_b

        metric_tree = {
            "delta_fro_norm": jnp.stack(delta_norms),
            "tangent_fro_norm": jnp.stack(tangent_norms),
            "step_fro_norm": jnp.stack(step_norms),
            "singular_values": jnp.stack(singular_values),
            "retracted_singular_values": jnp.stack(retracted_singular_values),
            "applied_update_fro_norm": jnp.stack(applied_norms),
        }
        return re.unflatten_tree(updated_flat), metric_tree

    return jax.jit(_update)


def _device_metrics_to_list(
    metric_tree: Mapping[str, Any],
    selected_blocks: Sequence[re.LoRABlock],
    active_slots: Sequence[int],
    target_rank: int,
) -> List[re.SlotUpdateMetrics]:
    arrays = jax.device_get(metric_tree)
    metrics: List[re.SlotUpdateMetrics] = []
    idx = 0
    for block in selected_blocks:
        for slot in active_slots:
            metrics.append(
                re.SlotUpdateMetrics(
                    path=block.path,
                    slot=int(slot),
                    target_rank=int(target_rank),
                    configured_rank=int(block.configured_rank),
                    delta_fro_norm=float(arrays["delta_fro_norm"][idx]),
                    tangent_fro_norm=float(arrays["tangent_fro_norm"][idx]),
                    step_fro_norm=float(arrays["step_fro_norm"][idx]),
                    singular_values=tuple(
                        float(x) for x in arrays["singular_values"][idx]
                    ),
                    retracted_singular_values=tuple(
                        float(x) for x in arrays["retracted_singular_values"][idx]
                    ),
                    applied_update_fro_norm=float(
                        arrays["applied_update_fro_norm"][idx]
                    ),
                )
            )
            idx += 1
    return metrics


def _make_candidate_batch_actor_params(
    actor_params: Any,
    thread_ids: Sequence[int],
    epoch: int,
    sigma: float,
    base_seed: int,
    active_slots: Sequence[int],
    target_rank: int,
    noise_rank: int,
    relative_scale: bool,
    singular_floor: float,
    block_patterns: Sequence[str] = re.NO_RECURRENT_BLOCK_PATTERNS,
) -> Tuple[Dict[str, Any], int]:
    """Build a pytree with a population axis only on selected LoRA leaves."""

    source_flat = re.flatten_tree(actor_params)
    batch_flat: Dict[Tuple[str, ...], Any] = dict(source_flat)
    selected = re.select_lora_blocks(re.discover_lora_blocks(actor_params), block_patterns)
    active_set = set(int(slot) for slot in active_slots)
    thread_ids = [int(x) for x in thread_ids]
    metric_count = 0

    for block in selected:
        if target_rank > block.configured_rank:
            raise ValueError(
                f"target_rank {target_rank} exceeds configured rank "
                f"{block.configured_rank} at {block.path}"
            )

        base_a = np.asarray(source_flat[block.lora_a_key])
        base_b = np.asarray(source_flat[block.lora_b_key])
        batch_a = np.repeat(base_a[None, ...], len(thread_ids), axis=0)
        batch_b = np.repeat(base_b[None, ...], len(thread_ids), axis=0)

        for pop_idx, thread_id in enumerate(thread_ids):
            direction_id, sign, _ = _thread_to_direction_and_sign(thread_id)
            for slot in sorted(active_set):
                if slot < 0 or slot >= block.num_slots:
                    raise IndexError(f"slot {slot} out of range for {block.path}")

                seed = re.stable_uint32_seed(
                    base_seed, epoch, direction_id, block.path, slot
                )
                delta, _, step, _, _, _ = re.tangent_step_for_slot(
                    base_a[slot],
                    base_b[slot],
                    target_rank=target_rank,
                    noise_rank=noise_rank,
                    seed=seed,
                    relative_scale=relative_scale,
                )
                if float(sigma) != 0.0:
                    candidate_delta = delta + float(sign) * float(sigma) * step
                    new_a, new_b, _ = re.retract_to_balanced_lora(
                        candidate_delta,
                        target_rank=target_rank,
                        configured_rank=block.configured_rank,
                        a_dtype=base_a.dtype,
                        b_dtype=base_b.dtype,
                        singular_floor=singular_floor,
                    )
                    batch_a[pop_idx, slot] = new_a
                    batch_b[pop_idx, slot] = new_b
                metric_count += 1

        batch_flat[block.lora_a_key] = batch_a
        batch_flat[block.lora_b_key] = batch_b

    return re.unflatten_tree(batch_flat), metric_count


def _summarize_stats(stats: Mapping[str, Any]) -> Dict[str, Any]:
    wins = np.asarray(stats["wins"], dtype=np.float64).reshape(-1)
    returns = np.asarray(stats["returns"], dtype=np.float64).reshape(-1)
    lengths = np.asarray(stats["lengths"], dtype=np.float64).reshape(-1)
    recorded = np.asarray(stats["recorded"], dtype=np.float64).reshape(-1)
    episodes = int(wins.size)
    win_rate = float(wins.mean()) if episodes else 0.0
    std = float(wins.std(ddof=1)) if episodes > 1 else 0.0
    sem = std / math.sqrt(episodes) if episodes > 1 else 0.0
    return {
        "episodes": episodes,
        "mean_wr": win_rate,
        "std": std,
        "sem": sem,
        "mean_return": float(returns.mean()) if episodes else 0.0,
        "mean_ep_len": float(lengths.mean()) if episodes else 0.0,
        "recorded_fraction": float(recorded.mean()) if episodes else 0.0,
    }


def _summarize_population_stats(stats: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    wins = np.asarray(stats["wins"], dtype=np.float64)
    returns = np.asarray(stats["returns"], dtype=np.float64)
    lengths = np.asarray(stats["lengths"], dtype=np.float64)
    recorded = np.asarray(stats["recorded"], dtype=np.float64)

    batch = wins.shape[0]
    wins_flat = wins.reshape(batch, -1)
    returns_flat = returns.reshape(batch, -1)
    lengths_flat = lengths.reshape(batch, -1)
    recorded_flat = recorded.reshape(batch, -1)
    episodes = wins_flat.shape[1]
    std = wins_flat.std(axis=1, ddof=1) if episodes > 1 else np.zeros(batch)
    return {
        "mean_wr": wins_flat.mean(axis=1),
        "std": std,
        "sem": std / math.sqrt(episodes) if episodes > 1 else np.zeros(batch),
        "mean_return": returns_flat.mean(axis=1),
        "mean_ep_len": lengths_flat.mean(axis=1),
        "recorded_fraction": recorded_flat.mean(axis=1),
        "episodes": np.full(batch, episodes, dtype=np.int32),
    }


def _compute_fitness_scores(
    win_rates: np.ndarray,
    mean_returns: np.ndarray,
    mode: str,
    return_tiebreak_weight: float,
) -> np.ndarray:
    if mode == "win_rate":
        return win_rates.astype(np.float64)
    if mode == "win_rate_return_tiebreak":
        return win_rates.astype(np.float64) + float(return_tiebreak_weight) * re.centered_ranks(
            mean_returns
        )
    raise ValueError(f"Unknown fitness mode: {mode}")


def _format_weights(weights: Mapping[int, float], max_items: int = 12) -> str:
    items = sorted((int(k), float(v)) for k, v in weights.items())
    if len(items) <= max_items:
        return ", ".join(f"{k}:{v:+.6f}" for k, v in items)

    abs_sorted = sorted(items, key=lambda kv: abs(kv[1]), reverse=True)
    top = ", ".join(f"{k}:{v:+.6f}" for k, v in abs_sorted[:max_items])
    nonzero = sum(1 for _, v in items if v != 0.0)
    max_abs = max((abs(v) for _, v in items), default=0.0)
    mean_abs = float(np.mean([abs(v) for _, v in items])) if items else 0.0
    return (
        f"nonzero={nonzero}/{len(items)} mean_abs={mean_abs:.6f} "
        f"max_abs={max_abs:.6f} top_abs=[{top}]"
    )


def _make_checkpoint(
    source_ckpt: Mapping[str, Any],
    actor_params: Any,
    epoch: int,
    args: argparse.Namespace,
    latest_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    ckpt = copy.deepcopy(dict(source_ckpt))
    ckpt["actor_params"] = actor_params
    ckpt["checkpoint_kind"] = "lorasa_eggroll_population"
    ckpt["lorasa_eggroll"] = {
        "trainer": "single_gpu_population_axis",
        "epoch": int(epoch),
        "source_checkpoint": str(args.checkpoint),
        "active_slots": list(args.active_slots),
        "target_rank": int(args.target_rank),
        "noise_rank": int(args.noise_rank),
        "sigma": float(args.sigma),
        "eta": float(args.eta),
        "num_directions": int(args.num_directions),
        "population_size": int(2 * args.num_directions),
        "population_batch_size": int(args.population_batch_size),
        "num_envs_per_candidate": int(args.num_envs_per_candidate),
        "episodes_per_candidate": int(args.episodes_per_candidate),
        "fitness_mode": str(args.fitness_mode),
        "candidate_build": str(args.candidate_build),
        "latest_stats": dict(latest_stats),
    }
    return ckpt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-GPU population-axis Riemannian LoRASA-EGGROLL trainer."
    )
    parser.add_argument("--checkpoint", required=True, help="Schedule A LoRASA checkpoint")
    parser.add_argument("--output_dir", default="lorasa_eggroll_pop_runs")
    parser.add_argument("--map_name", default=None)
    parser.add_argument("--max_steps", type=int, default=None)

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_directions", type=int, default=64)
    parser.add_argument("--population_batch_size", type=int, default=8)
    parser.add_argument("--num_envs_per_candidate", type=int, default=16)
    parser.add_argument("--episodes_per_candidate", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--heldout_num_envs", type=int, default=32)
    parser.add_argument("--heldout_episodes", type=int, default=512)

    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--eta", type=float, default=0.0015)
    parser.add_argument("--target_rank", type=int, default=re.DEFAULT_TARGET_RANK)
    parser.add_argument("--noise_rank", type=int, default=re.DEFAULT_NOISE_RANK)
    parser.add_argument("--active_slots", default="2,3,6", type=_parse_slots)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=1000)
    parser.add_argument("--eval_seed_stride", type=int, default=100)
    parser.add_argument(
        "--fitness_mode",
        choices=("win_rate", "win_rate_return_tiebreak"),
        default="win_rate_return_tiebreak",
    )
    parser.add_argument("--return_tiebreak_weight", type=float, default=0.001)
    parser.add_argument(
        "--raw_score_weights",
        action="store_true",
        help="Use raw fitness differences instead of centered-rank utilities",
    )
    parser.add_argument(
        "--no_relative_scale",
        action="store_true",
        help="Do not scale unit tangent directions by adapter Frobenius norm",
    )
    parser.add_argument("--singular_floor", type=float, default=0.0)
    parser.add_argument(
        "--candidate_build",
        choices=("device", "cpu"),
        default="device",
        help=(
            "Build per-chunk candidates on device with JAX, or use the older "
            "host NumPy correctness path."
        ),
    )
    parser.add_argument(
        "--print_candidates",
        action="store_true",
        help="Print every candidate score instead of only epoch summaries",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_epochs < 1:
        raise ValueError("--num_epochs must be >= 1")
    if args.num_directions < 1:
        raise ValueError("--num_directions must be >= 1")
    if args.population_batch_size < 1:
        raise ValueError("--population_batch_size must be >= 1")
    population_size = 2 * args.num_directions
    if population_size % args.population_batch_size != 0:
        raise ValueError(
            "--population_batch_size must divide 2 * --num_directions "
            f"({population_size}), got {args.population_batch_size}"
        )
    if args.num_envs_per_candidate < 1:
        raise ValueError("--num_envs_per_candidate must be >= 1")
    if args.episodes_per_candidate < 1:
        raise ValueError("--episodes_per_candidate must be >= 1")
    if args.episodes_per_candidate % args.num_envs_per_candidate != 0:
        raise ValueError(
            "--episodes_per_candidate must be divisible by "
            "--num_envs_per_candidate"
        )
    if args.heldout_num_envs < 1:
        raise ValueError("--heldout_num_envs must be >= 1")
    if args.heldout_episodes < 1:
        raise ValueError("--heldout_episodes must be >= 1")
    if args.heldout_episodes % args.heldout_num_envs != 0:
        raise ValueError("--heldout_episodes must be divisible by --heldout_num_envs")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _validate_args(args)

    source_ckpt = _load_checkpoint(args.checkpoint)
    map_name = args.map_name or source_ckpt.get("config", {}).get("MAP_NAME", "protoss_10_vs_10")

    rollout_batches = args.episodes_per_candidate // args.num_envs_per_candidate
    heldout_batches = args.heldout_episodes // args.heldout_num_envs
    population_size = 2 * args.num_directions
    train_episodes_per_epoch = population_size * args.episodes_per_candidate

    run_id = datetime.now().strftime("lorasa_eggroll_pop_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    args_json = vars(args).copy()
    args_json["active_slots"] = list(args.active_slots)
    args_json["map_name"] = map_name
    args_json["population_size"] = population_size
    args_json["rollout_batches_per_candidate"] = rollout_batches
    args_json["heldout_batches"] = heldout_batches
    args_json["train_episodes_per_epoch"] = train_episodes_per_epoch
    (run_dir / "args.json").write_text(json.dumps(args_json, indent=2, sort_keys=True) + "\n")

    print(f"Run directory: {run_dir}")
    print(f"Map: {map_name}")
    print(
        "Population: "
        f"directions={args.num_directions} candidates={population_size} "
        f"chunk={args.population_batch_size}"
    )
    print(
        "Training scale: "
        f"episodes_per_candidate={args.episodes_per_candidate} "
        f"num_envs_per_candidate={args.num_envs_per_candidate} "
        f"rollout_batches={rollout_batches} "
        f"train_episodes_per_epoch={train_episodes_per_epoch}"
    )

    print("Building train evaluator...")
    train_eval_fn, _, eval_steps = _build_eval_stats(
        source_ckpt,
        args.checkpoint,
        map_name,
        args.num_envs_per_candidate,
        rollout_batches,
        args.max_steps,
    )
    print(f"Eval steps: {eval_steps}")

    print("Building held-out evaluator...")
    heldout_eval_fn, _, _ = _build_eval_stats(
        source_ckpt,
        args.checkpoint,
        map_name,
        args.heldout_num_envs,
        heldout_batches,
        args.max_steps,
    )

    population_in_axes = _make_population_in_axes(
        source_ckpt["actor_params"],
        active_slots=args.active_slots,
        target_rank=args.target_rank,
    )
    selected_blocks = re.select_lora_blocks(
        re.discover_lora_blocks(source_ckpt["actor_params"]),
        re.NO_RECURRENT_BLOCK_PATTERNS,
    )
    active_slots = tuple(sorted(int(slot) for slot in args.active_slots))

    device_candidate_builder = None
    device_update_fn = None
    if args.candidate_build == "device":
        print("Candidate build: device/JAX batched SVD")
        device_candidate_builder, selected_blocks, active_slots = _make_device_candidate_builder(
            source_ckpt["actor_params"],
            active_slots=args.active_slots,
            target_rank=args.target_rank,
            noise_rank=args.noise_rank,
            sigma=args.sigma,
            base_seed=args.noise_seed,
            relative_scale=not args.no_relative_scale,
            singular_floor=args.singular_floor,
        )
        device_update_fn = _make_device_update_fn(
            source_ckpt["actor_params"],
            selected_blocks=selected_blocks,
            active_slots=active_slots,
            target_rank=args.target_rank,
            noise_rank=args.noise_rank,
            eta=args.eta,
            base_seed=args.noise_seed,
            direction_normalizer=args.num_directions,
            relative_scale=not args.no_relative_scale,
            singular_floor=args.singular_floor,
        )
    else:
        print("Candidate build: cpu/NumPy reference")

    def _batched_eval(rng, batched_actor_params):
        return jax.vmap(
            lambda one_params: train_eval_fn(rng, one_params),
            in_axes=(population_in_axes,),
        )(batched_actor_params)

    batched_eval = jax.jit(_batched_eval)

    actor_params = copy.deepcopy(source_ckpt["actor_params"])
    history_jsonl = run_dir / "history.jsonl"
    candidate_csv_rows: List[Dict[str, Any]] = []
    latest_stats: Dict[str, Any] = {}

    for epoch in range(args.num_epochs):
        print(f"\n=== Population ES epoch {epoch} ===")
        common_eval_seed = args.eval_seed + epoch * args.eval_seed_stride
        rng = jax.random.PRNGKey(int(common_eval_seed))
        raw_win_rates = np.zeros(population_size, dtype=np.float64)
        raw_returns = np.zeros(population_size, dtype=np.float64)
        raw_lengths = np.zeros(population_size, dtype=np.float64)
        raw_recorded = np.zeros(population_size, dtype=np.float64)
        raw_sem = np.zeros(population_size, dtype=np.float64)
        raw_std = np.zeros(population_size, dtype=np.float64)

        epoch_eval_start = time.time()
        for start_idx in range(0, population_size, args.population_batch_size):
            end_idx = start_idx + args.population_batch_size
            thread_ids = list(range(start_idx, end_idx))
            build_start = time.time()
            if args.candidate_build == "device":
                assert device_candidate_builder is not None
                thread_ids_array = jnp.arange(start_idx, end_idx, dtype=jnp.int32)
                batched_params = device_candidate_builder(
                    actor_params,
                    thread_ids_array,
                    jnp.asarray(epoch, dtype=jnp.int32),
                )
                _block_until_ready_tree(batched_params)
                candidate_metric_count = (
                    len(thread_ids) * len(selected_blocks) * len(active_slots)
                )
            else:
                batched_params, candidate_metric_count = _make_candidate_batch_actor_params(
                    actor_params,
                    thread_ids=thread_ids,
                    epoch=epoch,
                    sigma=args.sigma,
                    base_seed=args.noise_seed,
                    active_slots=args.active_slots,
                    target_rank=args.target_rank,
                    noise_rank=args.noise_rank,
                    relative_scale=not args.no_relative_scale,
                    singular_floor=args.singular_floor,
                )
            build_elapsed = time.time() - build_start

            eval_start = time.time()
            chunk_stats = jax.device_get(batched_eval(rng, batched_params))
            eval_elapsed = time.time() - eval_start
            chunk_summary = _summarize_population_stats(chunk_stats)

            raw_win_rates[start_idx:end_idx] = chunk_summary["mean_wr"]
            raw_returns[start_idx:end_idx] = chunk_summary["mean_return"]
            raw_lengths[start_idx:end_idx] = chunk_summary["mean_ep_len"]
            raw_recorded[start_idx:end_idx] = chunk_summary["recorded_fraction"]
            raw_std[start_idx:end_idx] = chunk_summary["std"]
            raw_sem[start_idx:end_idx] = chunk_summary["sem"]

            print(
                f"chunk {start_idx:04d}-{end_idx - 1:04d} "
                f"wr_mean={chunk_summary['mean_wr'].mean():.4f} "
                f"wr_min={chunk_summary['mean_wr'].min():.4f} "
                f"wr_max={chunk_summary['mean_wr'].max():.4f} "
                f"build={build_elapsed:.2f}s eval={eval_elapsed:.2f}s "
                f"adapter_metrics={candidate_metric_count}"
            )

        fitness_scores = _compute_fitness_scores(
            raw_win_rates,
            raw_returns,
            mode=args.fitness_mode,
            return_tiebreak_weight=args.return_tiebreak_weight,
        )
        direction_weights = re.antithetic_direction_weights(
            fitness_scores,
            use_centered_ranks=not args.raw_score_weights,
        )

        for thread_id in range(population_size):
            direction_id, _, sign_label = _thread_to_direction_and_sign(thread_id)
            record = {
                "epoch": epoch,
                "thread_id": thread_id,
                "direction_id": direction_id,
                "sign": sign_label,
                "win_rate": float(raw_win_rates[thread_id]),
                "std": float(raw_std[thread_id]),
                "sem": float(raw_sem[thread_id]),
                "mean_return": float(raw_returns[thread_id]),
                "mean_ep_len": float(raw_lengths[thread_id]),
                "recorded_fraction": float(raw_recorded[thread_id]),
                "fitness_score": float(fitness_scores[thread_id]),
                "episodes": int(args.episodes_per_candidate),
            }
            candidate_csv_rows.append(record)
            if args.print_candidates:
                print(
                    f"thread={thread_id:04d} dir={direction_id:03d} {sign_label:<5} "
                    f"wr={record['win_rate']:.4f} ret={record['mean_return']:.4f} "
                    f"fit={record['fitness_score']:.6f}"
                )

        print(
            "population summary: "
            f"wr_mean={raw_win_rates.mean():.4f} wr_std={raw_win_rates.std():.4f} "
            f"wr_min={raw_win_rates.min():.4f} wr_max={raw_win_rates.max():.4f} "
            f"return_mean={raw_returns.mean():.4f} eval_total={time.time() - epoch_eval_start:.1f}s"
        )
        print(f"direction weights: {_format_weights(direction_weights)}")

        if (
            args.candidate_build == "device"
            and float(args.eta) != 0.0
            and any(float(weight) != 0.0 for weight in direction_weights.values())
        ):
            assert device_update_fn is not None
            weights_array = np.zeros(args.num_directions, dtype=np.float32)
            for direction_id, weight in direction_weights.items():
                weights_array[int(direction_id)] = float(weight)
            actor_params, device_metric_tree = device_update_fn(
                actor_params,
                jnp.asarray(weights_array),
                jnp.asarray(epoch, dtype=jnp.int32),
            )
            _block_until_ready_tree((actor_params, device_metric_tree))
            update_metrics = _device_metrics_to_list(
                device_metric_tree,
                selected_blocks=selected_blocks,
                active_slots=active_slots,
                target_rank=args.target_rank,
            )
        else:
            actor_params, update_metrics = re.apply_weighted_tangent_update(
                actor_params,
                direction_weights=direction_weights,
                eta=args.eta,
                epoch=epoch,
                base_seed=args.noise_seed,
                active_slots=args.active_slots,
                target_rank=args.target_rank,
                noise_rank=args.noise_rank,
                relative_scale=not args.no_relative_scale,
                direction_normalizer=args.num_directions,
                singular_floor=args.singular_floor,
            )
        update_summary = _summarize_update_metrics(update_metrics)
        print(
            "update summary: "
            f"applied_mean={update_summary['mean_applied_update_fro_norm']:.6f} "
            f"applied_max={update_summary['max_applied_update_fro_norm']:.6f} "
            f"raw_step_sum_mean={update_summary['mean_step_fro_norm']:.6f} "
            f"mean_singular_shift={update_summary['mean_singular_shift']:.6f}"
        )

        heldout_eval = None
        if args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0):
            heldout_seed = common_eval_seed + args.eval_seed_stride // 2
            heldout_start = time.time()
            heldout_stats = jax.device_get(
                heldout_eval_fn(jax.random.PRNGKey(int(heldout_seed)), actor_params)
            )
            heldout_eval = _summarize_stats(heldout_stats)
            heldout_eval["elapsed_sec"] = time.time() - heldout_start
            print(
                "heldout: "
                f"wr={heldout_eval['mean_wr']:.4f} sem={heldout_eval['sem']:.4f} "
                f"return={heldout_eval['mean_return']:.4f} "
                f"episodes={heldout_eval['episodes']} "
                f"time={heldout_eval['elapsed_sec']:.1f}s"
            )

        latest_stats = {
            "epoch": epoch,
            "fitness_mode": args.fitness_mode,
            "candidate_build": args.candidate_build,
            "population_summary": {
                "win_rate_mean": float(raw_win_rates.mean()),
                "win_rate_std": float(raw_win_rates.std()),
                "win_rate_min": float(raw_win_rates.min()),
                "win_rate_max": float(raw_win_rates.max()),
                "return_mean": float(raw_returns.mean()),
                "fitness_mean": float(fitness_scores.mean()),
                "fitness_std": float(fitness_scores.std()),
            },
            "direction_weights": {str(k): float(v) for k, v in direction_weights.items()},
            "update_summary": update_summary,
            "heldout_eval": heldout_eval,
            "num_update_metrics": len(update_metrics),
        }
        _write_jsonl(
            history_jsonl,
            {
                **latest_stats,
                "update_metrics": _metrics_to_json(update_metrics),
            },
        )

        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
            ckpt = _make_checkpoint(source_ckpt, actor_params, epoch, args, latest_stats)
            epoch_path = run_dir / f"checkpoint_epoch_{epoch + 1:05d}.pkl"
            latest_path = run_dir / "checkpoint_latest.pkl"
            _save_checkpoint(ckpt, epoch_path)
            _save_checkpoint(ckpt, latest_path)
            print(f"Saved {epoch_path}")

    final_ckpt = _make_checkpoint(
        source_ckpt,
        actor_params,
        args.num_epochs - 1,
        args,
        latest_stats,
    )
    final_path = run_dir / "checkpoint_final.pkl"
    _save_checkpoint(final_ckpt, final_path)
    _write_csv(run_dir / "candidate_scores.csv", candidate_csv_rows)
    print(f"\nFinal checkpoint: {final_path}")
    print(f"History: {history_jsonl}")
    print(f"Candidate scores: {run_dir / 'candidate_scores.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
