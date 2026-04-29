#!/usr/bin/env python3
"""Riemannian LoRASA-EGGROLL prototype helpers.

This module implements the adapter-only geometry needed before wiring the ES
loop into SMAX rollouts:

* discover LoRASA ``lora_a`` / ``lora_b`` adapter blocks in actor params
* select the no-recurrent active slots used by the current phase-3 plan
* generate deterministic low-rank ambient perturbations
* project perturbations to the fixed-rank tangent space
* retract by truncated SVD and write balanced LoRA factors back to checkpoint

It intentionally does not import or depend on the official HyperscaleES repo.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_ACTIVE_SLOTS: Tuple[int, ...] = (2, 3, 6)
DEFAULT_TARGET_RANK = 4
DEFAULT_NOISE_RANK = 4

NO_RECURRENT_BLOCK_PATTERNS: Tuple[str, ...] = (
    "base_0",
    "base_1",
    "base_2",
    "rnn/gru_cell/input_reset",
    "rnn/gru_cell/input_update",
    "rnn/gru_cell/input_candidate",
    "action_out",
)

RECURRENT_BLOCK_SUBSTRINGS: Tuple[str, ...] = (
    "recurrent_reset",
    "recurrent_update",
    "recurrent_candidate",
)


@dataclass(frozen=True)
class LoRABlock:
    """Metadata for one LoRA adapter block."""

    prefix: Tuple[str, ...]
    path: str
    lora_a_key: Tuple[str, ...]
    lora_b_key: Tuple[str, ...]
    num_slots: int
    input_dim: int
    configured_rank: int
    output_dim: int


@dataclass(frozen=True)
class SlotUpdateMetrics:
    """Per block/slot metrics produced by candidate or update construction."""

    path: str
    slot: int
    target_rank: int
    configured_rank: int
    delta_fro_norm: float
    tangent_fro_norm: float
    step_fro_norm: float
    singular_values: Tuple[float, ...]
    retracted_singular_values: Tuple[float, ...]


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    """Load a pickle checkpoint and require a mapping at top level."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with path.open("rb") as f:
        ckpt = pickle.load(f)
    if not isinstance(ckpt, Mapping):
        raise ValueError(f"Checkpoint must be a mapping, got {type(ckpt).__name__}")
    return dict(ckpt)


def save_checkpoint(ckpt: Mapping[str, Any], path: str | Path) -> None:
    """Save a checkpoint pickle, creating parent directories."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(dict(ckpt), f, protocol=pickle.HIGHEST_PROTOCOL)


def extract_actor_params(obj: Mapping[str, Any]) -> Any:
    """Return actor params from a checkpoint or return the object itself."""

    if "actor_params" in obj:
        return obj["actor_params"]
    return obj


def path_to_str(path: Sequence[Any]) -> str:
    """Convert a tree path to the slash-separated style used in diagnostics."""

    return "/".join(str(part) for part in path)


def _normalize_flat_key(key: Any) -> Tuple[str, ...]:
    if isinstance(key, tuple):
        return tuple(str(part) for part in key)
    if isinstance(key, str):
        return tuple(key.split("/"))
    return (str(key),)


def flatten_tree(tree: Any, prefix: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], Any]:
    """Recursively flatten mapping/list/tuple pytrees to tuple-key leaves."""

    flat: Dict[Tuple[str, ...], Any] = {}
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            flat.update(flatten_tree(value, prefix + (str(key),)))
    elif isinstance(tree, (list, tuple)):
        for idx, value in enumerate(tree):
            flat.update(flatten_tree(value, prefix + (str(idx),)))
    else:
        flat[prefix] = tree
    return flat


def unflatten_tree(flat: Mapping[Tuple[str, ...], Any]) -> Dict[str, Any]:
    """Rebuild a nested dict from tuple-key leaves."""

    root: Dict[str, Any] = {}
    for raw_key, value in flat.items():
        key = _normalize_flat_key(raw_key)
        if not key:
            raise ValueError("Cannot unflatten an empty key")
        cursor: MutableMapping[str, Any] = root
        for part in key[:-1]:
            if part not in cursor:
                cursor[part] = {}
            child = cursor[part]
            if not isinstance(child, MutableMapping):
                raise ValueError(f"Path collision while unflattening at {key}")
            cursor = child
        cursor[key[-1]] = value
    return root


def discover_lora_blocks(actor_params: Any) -> List[LoRABlock]:
    """Discover well-formed LoRA blocks under an actor params tree."""

    flat = flatten_tree(actor_params)
    a_keys: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    b_keys: Dict[Tuple[str, ...], Tuple[str, ...]] = {}

    for key in flat:
        if not key:
            continue
        if key[-1] == "lora_a":
            a_keys[key[:-1]] = key
        elif key[-1] == "lora_b":
            b_keys[key[:-1]] = key

    blocks: List[LoRABlock] = []
    for prefix in sorted(set(a_keys) & set(b_keys)):
        a = np.asarray(flat[a_keys[prefix]])
        b = np.asarray(flat[b_keys[prefix]])

        if a.ndim != 3 or b.ndim != 3:
            warnings.warn(
                f"Skipping {path_to_str(prefix)}: expected 3D lora arrays, "
                f"got lora_a.ndim={a.ndim}, lora_b.ndim={b.ndim}"
            )
            continue

        num_slots_a, input_dim, rank_a = a.shape
        num_slots_b, rank_b, output_dim = b.shape
        if num_slots_a != num_slots_b:
            warnings.warn(
                f"Skipping {path_to_str(prefix)}: slot mismatch "
                f"{num_slots_a} != {num_slots_b}"
            )
            continue
        if rank_a != rank_b:
            warnings.warn(
                f"Skipping {path_to_str(prefix)}: rank mismatch "
                f"{rank_a} != {rank_b}"
            )
            continue

        blocks.append(
            LoRABlock(
                prefix=prefix,
                path=path_to_str(prefix),
                lora_a_key=a_keys[prefix],
                lora_b_key=b_keys[prefix],
                num_slots=int(num_slots_a),
                input_dim=int(input_dim),
                configured_rank=int(rank_a),
                output_dim=int(output_dim),
            )
        )

    for prefix in sorted(set(a_keys) - set(b_keys)):
        warnings.warn(f"Orphan lora_a without lora_b at {path_to_str(prefix)}")
    for prefix in sorted(set(b_keys) - set(a_keys)):
        warnings.warn(f"Orphan lora_b without lora_a at {path_to_str(prefix)}")

    return blocks


def is_no_recurrent_block(path: str) -> bool:
    """Return True if a block is in the current no-recurrent phase-3 target."""

    if any(part in path for part in RECURRENT_BLOCK_SUBSTRINGS):
        return False
    return any(path == pat or path.endswith("/" + pat) for pat in NO_RECURRENT_BLOCK_PATTERNS)


def select_lora_blocks(
    blocks: Sequence[LoRABlock],
    block_patterns: Sequence[str] = NO_RECURRENT_BLOCK_PATTERNS,
) -> List[LoRABlock]:
    """Select blocks matching the configured phase-3 block patterns."""

    selected: List[LoRABlock] = []
    for block in blocks:
        if any(part in block.path for part in RECURRENT_BLOCK_SUBSTRINGS):
            continue
        if any(block.path == pat or block.path.endswith("/" + pat) for pat in block_patterns):
            selected.append(block)
    return selected


def stable_uint32_seed(*parts: Any) -> int:
    """Build a deterministic 32-bit seed from hashable parts."""

    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "little") % (2**32)


def centered_ranks(scores: Sequence[float]) -> np.ndarray:
    """Convert scores to centered rank utilities in roughly [-0.5, 0.5].

    Tied scores receive the average rank for their tie group so antithetic
    candidates with equal fitness produce zero directional pressure.
    """

    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape {x.shape}")
    if x.size == 0:
        return x
    if x.size == 1:
        return np.zeros_like(x)

    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]

    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end

    return ranks / (x.size - 1) - 0.5


def antithetic_direction_weights(
    scores: Sequence[float],
    use_centered_ranks: bool = True,
) -> Dict[int, float]:
    """Collapse [dir0+, dir0-, dir1+, dir1-, ...] scores to direction weights."""

    utilities = centered_ranks(scores) if use_centered_ranks else np.asarray(scores, dtype=np.float64)
    if utilities.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape {utilities.shape}")
    if utilities.size % 2 != 0:
        raise ValueError("antithetic scores must have even length")
    return {
        direction_id: float(utilities[2 * direction_id] - utilities[2 * direction_id + 1])
        for direction_id in range(utilities.size // 2)
    }


def thin_svd(delta: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return top-rank SVD coordinates for a matrix."""

    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    max_rank = min(delta.shape)
    if rank > max_rank:
        raise ValueError(f"rank {rank} exceeds max matrix rank {max_rank} for {delta.shape}")
    u, s, vt = np.linalg.svd(delta.astype(np.float64), full_matrices=False)
    return u[:, :rank], s[:rank], vt[:rank, :]


def balanced_factors_from_svd(
    u: np.ndarray,
    s: np.ndarray,
    vt: np.ndarray,
    configured_rank: int,
    a_dtype: np.dtype,
    b_dtype: np.dtype,
    singular_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Write ``U S Vt`` as balanced LoRA factors with original rank width."""

    active_rank = int(s.shape[0])
    if configured_rank < active_rank:
        raise ValueError(
            f"configured_rank {configured_rank} is smaller than active rank {active_rank}"
        )

    s_safe = np.maximum(s.astype(np.float64), singular_floor)
    sqrt_s = np.sqrt(s_safe)

    a = np.zeros((u.shape[0], configured_rank), dtype=np.float64)
    b = np.zeros((configured_rank, vt.shape[1]), dtype=np.float64)
    a[:, :active_rank] = u[:, :active_rank] * sqrt_s[None, :]
    b[:active_rank, :] = sqrt_s[:, None] * vt[:active_rank, :]
    return a.astype(a_dtype, copy=False), b.astype(b_dtype, copy=False)


def retract_to_balanced_lora(
    delta: np.ndarray,
    target_rank: int,
    configured_rank: int,
    a_dtype: np.dtype,
    b_dtype: np.dtype,
    singular_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retract a matrix by rank-r SVD projection and return balanced factors."""

    u, s, vt = thin_svd(delta, target_rank)
    a, b = balanced_factors_from_svd(
        u,
        s,
        vt,
        configured_rank=configured_rank,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        singular_floor=singular_floor,
    )
    return a, b, s


def tangent_project(z: np.ndarray, u: np.ndarray, vt: np.ndarray) -> np.ndarray:
    """Project ambient matrix ``z`` onto the tangent space at ``U S Vt``."""

    v = vt.T
    z_v_vt = (z @ v) @ v.T
    u_ut_z = u @ (u.T @ z)
    u_ut_z_v_vt = u @ ((u.T @ z @ v) @ v.T)
    return z_v_vt + u_ut_z - u_ut_z_v_vt


def sample_low_rank_ambient(
    shape: Tuple[int, int],
    noise_rank: int,
    seed: int,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Sample ``P @ Q / sqrt(noise_rank)`` with deterministic NumPy RNG."""

    if noise_rank <= 0:
        raise ValueError(f"noise_rank must be positive, got {noise_rank}")
    rows, cols = shape
    rng = np.random.default_rng(int(seed))
    p = rng.standard_normal((rows, noise_rank), dtype=np.float64)
    q = rng.standard_normal((noise_rank, cols), dtype=np.float64)
    z = (p @ q) / math.sqrt(noise_rank)
    return z.astype(dtype, copy=False)


def _direction_seed(
    base_seed: int,
    epoch: int,
    direction_id: int,
    block_path: str,
    slot: int,
) -> int:
    return stable_uint32_seed(base_seed, epoch, direction_id, block_path, slot)


def tangent_step_for_slot(
    a_slot: np.ndarray,
    b_slot: np.ndarray,
    target_rank: int,
    noise_rank: int,
    seed: int,
    relative_scale: bool = True,
    min_scale: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a normalized tangent search step for one adapter slot.

    Returns ``(delta, tangent, step, u, s, vt)`` where ``step`` is what should be
    multiplied by ``sigma`` or aggregated by ES weights.
    """

    a64 = np.asarray(a_slot, dtype=np.float64)
    b64 = np.asarray(b_slot, dtype=np.float64)
    delta = a64 @ b64
    u, s, vt = thin_svd(delta, target_rank)
    z = sample_low_rank_ambient(delta.shape, noise_rank, seed, dtype=np.float64)
    tangent = tangent_project(z, u, vt)

    tangent_norm = float(np.linalg.norm(tangent, "fro"))
    if tangent_norm <= 0:
        step = np.zeros_like(tangent)
    else:
        step = tangent / tangent_norm
        if relative_scale:
            delta_norm = float(np.linalg.norm(delta, "fro"))
            step = step * max(delta_norm, min_scale)

    return delta, tangent, step, u, s, vt


def _replace_block_arrays(
    flat: MutableMapping[Tuple[str, ...], Any],
    block: LoRABlock,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
) -> None:
    flat[block.lora_a_key] = lora_a
    flat[block.lora_b_key] = lora_b


def make_candidate_actor_params(
    actor_params: Any,
    direction_id: int,
    sign: int,
    sigma: float,
    epoch: int = 0,
    base_seed: int = 0,
    active_slots: Sequence[int] = DEFAULT_ACTIVE_SLOTS,
    target_rank: int = DEFAULT_TARGET_RANK,
    noise_rank: int = DEFAULT_NOISE_RANK,
    block_patterns: Sequence[str] = NO_RECURRENT_BLOCK_PATTERNS,
    relative_scale: bool = True,
    singular_floor: float = 0.0,
) -> Tuple[Dict[str, Any], List[SlotUpdateMetrics]]:
    """Materialize one antithetic candidate actor params tree.

    This is the correctness-first path. It copies the actor params and writes
    retracted active adapter slots for one ES population member.
    """

    if sign not in (-1, 1):
        raise ValueError(f"sign must be -1 or 1, got {sign}")

    flat = flatten_tree(copy.deepcopy(actor_params))
    selected = select_lora_blocks(discover_lora_blocks(actor_params), block_patterns)
    active_set = set(int(slot) for slot in active_slots)
    metrics: List[SlotUpdateMetrics] = []

    for block in selected:
        if target_rank > block.configured_rank:
            raise ValueError(
                f"target_rank {target_rank} exceeds configured rank "
                f"{block.configured_rank} at {block.path}"
            )

        lora_a = np.asarray(flat[block.lora_a_key]).copy()
        lora_b = np.asarray(flat[block.lora_b_key]).copy()

        for slot in sorted(active_set):
            if slot < 0 or slot >= block.num_slots:
                raise IndexError(f"slot {slot} out of range for {block.path}")

            seed = _direction_seed(base_seed, epoch, direction_id, block.path, slot)
            delta, tangent, step, _, s, _ = tangent_step_for_slot(
                lora_a[slot],
                lora_b[slot],
                target_rank=target_rank,
                noise_rank=noise_rank,
                seed=seed,
                relative_scale=relative_scale,
            )
            if float(sigma) == 0.0:
                s_new = s
            else:
                candidate_delta = delta + float(sign) * float(sigma) * step
                new_a, new_b, s_new = retract_to_balanced_lora(
                    candidate_delta,
                    target_rank=target_rank,
                    configured_rank=block.configured_rank,
                    a_dtype=lora_a.dtype,
                    b_dtype=lora_b.dtype,
                    singular_floor=singular_floor,
                )
                lora_a[slot] = new_a
                lora_b[slot] = new_b

            metrics.append(
                SlotUpdateMetrics(
                    path=block.path,
                    slot=slot,
                    target_rank=target_rank,
                    configured_rank=block.configured_rank,
                    delta_fro_norm=float(np.linalg.norm(delta, "fro")),
                    tangent_fro_norm=float(np.linalg.norm(tangent, "fro")),
                    step_fro_norm=float(np.linalg.norm(step, "fro")),
                    singular_values=tuple(float(x) for x in s),
                    retracted_singular_values=tuple(float(x) for x in s_new),
                )
            )

        _replace_block_arrays(flat, block, lora_a, lora_b)

    return unflatten_tree(flat), metrics


def apply_weighted_tangent_update(
    actor_params: Any,
    direction_weights: Mapping[int, float],
    eta: float,
    epoch: int = 0,
    base_seed: int = 0,
    active_slots: Sequence[int] = DEFAULT_ACTIVE_SLOTS,
    target_rank: int = DEFAULT_TARGET_RANK,
    noise_rank: int = DEFAULT_NOISE_RANK,
    block_patterns: Sequence[str] = NO_RECURRENT_BLOCK_PATTERNS,
    relative_scale: bool = True,
    average_directions: bool = True,
    direction_normalizer: Optional[int] = None,
    singular_floor: float = 0.0,
) -> Tuple[Dict[str, Any], List[SlotUpdateMetrics]]:
    """Apply an ES-weighted tangent update and retract active adapters."""

    flat = flatten_tree(copy.deepcopy(actor_params))
    selected = select_lora_blocks(discover_lora_blocks(actor_params), block_patterns)
    active_set = set(int(slot) for slot in active_slots)
    clean_weights = {
        int(direction_id): float(weight)
        for direction_id, weight in direction_weights.items()
        if float(weight) != 0.0
    }
    direction_count = max(1, len(clean_weights))
    metrics: List[SlotUpdateMetrics] = []

    for block in selected:
        if target_rank > block.configured_rank:
            raise ValueError(
                f"target_rank {target_rank} exceeds configured rank "
                f"{block.configured_rank} at {block.path}"
            )

        lora_a = np.asarray(flat[block.lora_a_key]).copy()
        lora_b = np.asarray(flat[block.lora_b_key]).copy()

        for slot in sorted(active_set):
            if slot < 0 or slot >= block.num_slots:
                raise IndexError(f"slot {slot} out of range for {block.path}")

            delta = np.asarray(lora_a[slot], dtype=np.float64) @ np.asarray(
                lora_b[slot], dtype=np.float64
            )
            _, s_old, _ = thin_svd(delta, target_rank)
            aggregate = np.zeros_like(delta, dtype=np.float64)
            tangent_norm_accum = 0.0
            step_norm_accum = 0.0

            if float(eta) == 0.0 or not clean_weights:
                s_new = s_old
            else:
                for direction_id, weight in clean_weights.items():
                    seed = _direction_seed(base_seed, epoch, direction_id, block.path, slot)
                    _, tangent, step, _, _, _ = tangent_step_for_slot(
                        lora_a[slot],
                        lora_b[slot],
                        target_rank=target_rank,
                        noise_rank=noise_rank,
                        seed=seed,
                        relative_scale=relative_scale,
                    )
                    aggregate += weight * step
                    tangent_norm_accum += float(np.linalg.norm(tangent, "fro"))
                    step_norm_accum += float(np.linalg.norm(step, "fro"))

                if average_directions:
                    normalizer = direction_count if direction_normalizer is None else int(
                        direction_normalizer
                    )
                    aggregate /= max(1, normalizer)

                updated_delta = delta + float(eta) * aggregate
                new_a, new_b, s_new = retract_to_balanced_lora(
                    updated_delta,
                    target_rank=target_rank,
                    configured_rank=block.configured_rank,
                    a_dtype=lora_a.dtype,
                    b_dtype=lora_b.dtype,
                    singular_floor=singular_floor,
                )
                lora_a[slot] = new_a
                lora_b[slot] = new_b

            metrics.append(
                SlotUpdateMetrics(
                    path=block.path,
                    slot=slot,
                    target_rank=target_rank,
                    configured_rank=block.configured_rank,
                    delta_fro_norm=float(np.linalg.norm(delta, "fro")),
                    tangent_fro_norm=tangent_norm_accum,
                    step_fro_norm=step_norm_accum,
                    singular_values=tuple(float(x) for x in s_old),
                    retracted_singular_values=tuple(float(x) for x in s_new),
                )
            )

        _replace_block_arrays(flat, block, lora_a, lora_b)

    return unflatten_tree(flat), metrics


def summarize_actor_adapters(
    actor_params: Any,
    active_slots: Sequence[int] = DEFAULT_ACTIVE_SLOTS,
    target_rank: int = DEFAULT_TARGET_RANK,
) -> Dict[str, Any]:
    """Return a JSON-friendly summary of discovered and selected adapters."""

    blocks = discover_lora_blocks(actor_params)
    selected = select_lora_blocks(blocks)
    active_set = set(int(slot) for slot in active_slots)

    selected_rows: List[Dict[str, Any]] = []
    flat = flatten_tree(actor_params)
    for block in selected:
        a = np.asarray(flat[block.lora_a_key])
        b = np.asarray(flat[block.lora_b_key])
        for slot in sorted(active_set):
            if slot < 0 or slot >= block.num_slots:
                continue
            delta = a[slot].astype(np.float64) @ b[slot].astype(np.float64)
            s = np.linalg.svd(delta, compute_uv=False)
            selected_rows.append(
                {
                    "path": block.path,
                    "slot": slot,
                    "shape": [block.input_dim, block.output_dim],
                    "configured_rank": block.configured_rank,
                    "target_rank": target_rank,
                    "fro_norm": float(np.linalg.norm(delta, "fro")),
                    "top_singular_values": [float(x) for x in s[:target_rank]],
                    "numerical_rank_rel_1e-6": int(
                        np.sum(s > (1e-6 * s[0])) if s.size and s[0] > 0 else 0
                    ),
                }
            )

    return {
        "num_discovered_blocks": len(blocks),
        "num_selected_blocks": len(selected),
        "active_slots": sorted(active_set),
        "selected_block_paths": [block.path for block in selected],
        "recurrent_block_paths": [
            block.path for block in blocks if any(p in block.path for p in RECURRENT_BLOCK_SUBSTRINGS)
        ],
        "selected_slots": selected_rows,
    }


def _leaf_arrays_equal(left: Any, right: Any) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right))


def _numerical_rank(s: np.ndarray, rel_tol: float = 1e-6) -> int:
    if s.size == 0 or s[0] <= 0:
        return 0
    return int(np.sum(s > (rel_tol * s[0])))


def validate_actor_update_against_reference(
    reference_actor_params: Any,
    updated_actor_params: Any,
    active_slots: Sequence[int] = DEFAULT_ACTIVE_SLOTS,
    target_rank: int = DEFAULT_TARGET_RANK,
    block_patterns: Sequence[str] = NO_RECURRENT_BLOCK_PATTERNS,
    require_active_change: bool = False,
    rank_rel_tol: float = 1e-6,
) -> Dict[str, Any]:
    """Validate that only selected active LoRA slots changed after an update."""

    active_set = set(int(slot) for slot in active_slots)
    ref_flat = flatten_tree(reference_actor_params)
    upd_flat = flatten_tree(updated_actor_params)
    violations: List[str] = []

    ref_keys = set(ref_flat)
    upd_keys = set(upd_flat)
    for key in sorted(ref_keys - upd_keys):
        violations.append(f"missing updated leaf: {path_to_str(key)}")
    for key in sorted(upd_keys - ref_keys):
        violations.append(f"extra updated leaf: {path_to_str(key)}")

    ref_blocks_by_path = {block.path: block for block in discover_lora_blocks(reference_actor_params)}
    upd_blocks_by_path = {block.path: block for block in discover_lora_blocks(updated_actor_params)}
    if set(ref_blocks_by_path) != set(upd_blocks_by_path):
        violations.append(
            "LoRA block path mismatch: "
            f"reference={sorted(ref_blocks_by_path)} updated={sorted(upd_blocks_by_path)}"
        )

    selected_paths = {
        block.path for block in select_lora_blocks(ref_blocks_by_path.values(), block_patterns)
    }
    lora_leaf_to_block: Dict[Tuple[str, ...], Tuple[LoRABlock, str]] = {}
    for block in ref_blocks_by_path.values():
        lora_leaf_to_block[block.lora_a_key] = (block, "lora_a")
        lora_leaf_to_block[block.lora_b_key] = (block, "lora_b")

    unchanged_non_active_leaves = 0
    changed_non_active_leaves = 0
    active_slot_pairs_changed = 0
    active_slot_pairs_unchanged = 0
    active_rank_violations = 0

    for key in sorted(ref_keys & upd_keys):
        ref_value = ref_flat[key]
        upd_value = upd_flat[key]
        lora_info = lora_leaf_to_block.get(key)
        if lora_info is None:
            if _leaf_arrays_equal(ref_value, upd_value):
                unchanged_non_active_leaves += 1
            else:
                changed_non_active_leaves += 1
                violations.append(f"non-LoRA actor leaf changed: {path_to_str(key)}")
            continue

        block, leaf_name = lora_info
        ref_arr = np.asarray(ref_value)
        upd_arr = np.asarray(upd_value)
        if ref_arr.shape != upd_arr.shape:
            violations.append(
                f"{leaf_name} shape changed at {block.path}: {ref_arr.shape} -> {upd_arr.shape}"
            )
            continue

        for slot in range(block.num_slots):
            is_allowed_active = block.path in selected_paths and slot in active_set
            slot_equal = np.array_equal(ref_arr[slot], upd_arr[slot])
            if is_allowed_active:
                continue
            if slot_equal:
                unchanged_non_active_leaves += 1
            else:
                changed_non_active_leaves += 1
                violations.append(f"frozen {leaf_name} slot {slot} changed at {block.path}")

    for block in select_lora_blocks(ref_blocks_by_path.values(), block_patterns):
        upd_block = upd_blocks_by_path.get(block.path)
        if upd_block is None:
            continue
        ref_a = np.asarray(ref_flat[block.lora_a_key])
        ref_b = np.asarray(ref_flat[block.lora_b_key])
        upd_a = np.asarray(upd_flat[upd_block.lora_a_key])
        upd_b = np.asarray(upd_flat[upd_block.lora_b_key])
        for slot in sorted(active_set):
            if slot < 0 or slot >= block.num_slots:
                violations.append(f"active slot {slot} out of range for {block.path}")
                continue

            a_equal = np.array_equal(ref_a[slot], upd_a[slot])
            b_equal = np.array_equal(ref_b[slot], upd_b[slot])
            if a_equal and b_equal:
                active_slot_pairs_unchanged += 1
                if require_active_change:
                    violations.append(f"active slot {slot} did not change at {block.path}")
            else:
                active_slot_pairs_changed += 1

            delta = upd_a[slot].astype(np.float64) @ upd_b[slot].astype(np.float64)
            singular_values = np.linalg.svd(delta, compute_uv=False)
            rank = _numerical_rank(singular_values, rel_tol=rank_rel_tol)
            if rank != target_rank:
                active_rank_violations += 1
                violations.append(
                    f"active slot {slot} rank {rank} != {target_rank} at {block.path}"
                )

    return {
        "passed": not violations,
        "num_violations": len(violations),
        "violations": violations,
        "active_slots": sorted(active_set),
        "selected_block_paths": sorted(selected_paths),
        "active_slot_pairs_changed": active_slot_pairs_changed,
        "active_slot_pairs_unchanged": active_slot_pairs_unchanged,
        "changed_non_active_leaves": changed_non_active_leaves,
        "unchanged_non_active_leaves": unchanged_non_active_leaves,
        "active_rank_violations": active_rank_violations,
        "target_rank": int(target_rank),
        "require_active_change": bool(require_active_change),
    }


def _fake_actor_params(seed: int = 123) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    num_slots = 9
    configured_rank = 8
    hidden = 16
    obs_dim = 11
    action_dim = 7

    def lora(in_dim: int, out_dim: int) -> Dict[str, np.ndarray]:
        return {
            "kernel": rng.standard_normal((in_dim, out_dim)).astype(np.float32),
            "bias": np.zeros(out_dim, dtype=np.float32),
            "lora_a": rng.standard_normal((num_slots, in_dim, configured_rank)).astype(np.float32),
            "lora_b": (0.05 * rng.standard_normal((num_slots, configured_rank, out_dim))).astype(
                np.float32
            ),
        }

    return {
        "params": {
            "base_0": lora(obs_dim, hidden),
            "base_1": lora(hidden, hidden),
            "base_2": lora(hidden, hidden),
            "rnn": {
                "gru_cell": {
                    "input_reset": lora(hidden, hidden),
                    "input_update": lora(hidden, hidden),
                    "input_candidate": lora(hidden, hidden),
                    "recurrent_reset": lora(hidden, hidden),
                }
            },
            "action_out": lora(hidden, action_dim),
        }
    }


def _delta_ranks_for_selected(
    actor_params: Any,
    active_slots: Sequence[int],
    target_rank: int,
) -> List[Tuple[str, int, int]]:
    flat = flatten_tree(actor_params)
    out: List[Tuple[str, int, int]] = []
    for block in select_lora_blocks(discover_lora_blocks(actor_params)):
        a = np.asarray(flat[block.lora_a_key])
        b = np.asarray(flat[block.lora_b_key])
        for slot in active_slots:
            s = np.linalg.svd(
                a[slot].astype(np.float64) @ b[slot].astype(np.float64),
                compute_uv=False,
            )
            rank = int(np.sum(s > (1e-6 * s[0])) if s.size and s[0] > 0 else 0)
            out.append((block.path, int(slot), min(rank, target_rank)))
    return out


def _assert_tree_lora_slots_equal(
    before: Any,
    after: Any,
    inactive_slots: Iterable[int],
) -> None:
    flat_before = flatten_tree(before)
    flat_after = flatten_tree(after)
    for block in discover_lora_blocks(before):
        a0 = np.asarray(flat_before[block.lora_a_key])
        b0 = np.asarray(flat_before[block.lora_b_key])
        a1 = np.asarray(flat_after[block.lora_a_key])
        b1 = np.asarray(flat_after[block.lora_b_key])
        for slot in inactive_slots:
            if slot < block.num_slots:
                if not np.array_equal(a0[slot], a1[slot]):
                    raise AssertionError(f"inactive lora_a slot {slot} changed at {block.path}")
                if not np.array_equal(b0[slot], b1[slot]):
                    raise AssertionError(f"inactive lora_b slot {slot} changed at {block.path}")


def _assert_all_lora_equal(before: Any, after: Any) -> None:
    flat_before = flatten_tree(before)
    flat_after = flatten_tree(after)
    for block in discover_lora_blocks(before):
        a0 = np.asarray(flat_before[block.lora_a_key])
        b0 = np.asarray(flat_before[block.lora_b_key])
        a1 = np.asarray(flat_after[block.lora_a_key])
        b1 = np.asarray(flat_after[block.lora_b_key])
        if not np.array_equal(a0, a1):
            raise AssertionError(f"lora_a changed at {block.path}")
        if not np.array_equal(b0, b1):
            raise AssertionError(f"lora_b changed at {block.path}")


def self_test() -> None:
    """Run dependency-light invariants for the Riemannian helper layer."""

    actor_params = _fake_actor_params()
    active_slots = DEFAULT_ACTIVE_SLOTS
    inactive_slots = [0, 1, 4, 5, 7, 8]

    blocks = discover_lora_blocks(actor_params)
    selected = select_lora_blocks(blocks)
    if len(blocks) != 8:
        raise AssertionError(f"expected 8 fake LoRA blocks, found {len(blocks)}")
    if len(selected) != 7:
        raise AssertionError(f"expected 7 selected no-recurrent blocks, found {len(selected)}")

    zero_candidate, zero_metrics = make_candidate_actor_params(
        actor_params,
        direction_id=0,
        sign=1,
        sigma=0.0,
        epoch=0,
        base_seed=0,
    )
    if not zero_metrics:
        raise AssertionError("candidate metrics were empty")
    _assert_all_lora_equal(actor_params, zero_candidate)
    _assert_tree_lora_slots_equal(actor_params, zero_candidate, inactive_slots)

    updated_zero, update_metrics = apply_weighted_tangent_update(
        actor_params,
        direction_weights={0: 1.0, 1: -0.5},
        eta=0.0,
        epoch=0,
        base_seed=0,
    )
    if not update_metrics:
        raise AssertionError("update metrics were empty")
    _assert_all_lora_equal(actor_params, updated_zero)
    _assert_tree_lora_slots_equal(actor_params, updated_zero, inactive_slots)

    plus_params, _ = make_candidate_actor_params(
        actor_params,
        direction_id=3,
        sign=1,
        sigma=0.01,
        epoch=5,
        base_seed=7,
    )
    minus_params, _ = make_candidate_actor_params(
        actor_params,
        direction_id=3,
        sign=-1,
        sigma=0.01,
        epoch=5,
        base_seed=7,
    )
    _assert_tree_lora_slots_equal(actor_params, plus_params, inactive_slots)
    _assert_tree_lora_slots_equal(actor_params, minus_params, inactive_slots)

    plus_validation = validate_actor_update_against_reference(
        actor_params,
        plus_params,
        active_slots=active_slots,
        target_rank=DEFAULT_TARGET_RANK,
        require_active_change=True,
    )
    if not plus_validation["passed"]:
        raise AssertionError(f"plus candidate validation failed: {plus_validation}")

    ranks = _delta_ranks_for_selected(plus_params, active_slots, DEFAULT_TARGET_RANK)
    for path, slot, rank in ranks:
        if rank > DEFAULT_TARGET_RANK:
            raise AssertionError(f"{path} slot {slot} rank {rank} exceeds target")

    weights = antithetic_direction_weights([1.0, 0.0, -2.0, 3.0])
    if set(weights) != {0, 1}:
        raise AssertionError(f"unexpected antithetic weight keys: {weights}")
    tied_weights = antithetic_direction_weights([1.0, 1.0])
    if tied_weights != {0: 0.0}:
        raise AssertionError(f"ties should produce zero direction weight: {tied_weights}")


def _parse_slots(raw: str) -> Tuple[int, ...]:
    try:
        slots = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid slot list: {raw}") from e
    if not slots:
        raise argparse.ArgumentTypeError("Slot list cannot be empty")
    return slots


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Riemannian LoRASA-EGGROLL checkpoint/math helpers."
    )
    parser.add_argument("--self_test", action="store_true", help="Run internal self-test")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to inspect")
    parser.add_argument(
        "--reference_checkpoint",
        default=None,
        help="Optional source checkpoint for before/after update validation",
    )
    parser.add_argument("--summary_json", default=None, help="Optional path for adapter summary JSON")
    parser.add_argument(
        "--validation_json",
        default=None,
        help="Optional path for before/after validation JSON",
    )
    parser.add_argument(
        "--active_slots",
        default="2,3,6",
        type=_parse_slots,
        help="Comma-separated active adapter slots",
    )
    parser.add_argument("--target_rank", type=int, default=DEFAULT_TARGET_RANK)
    parser.add_argument(
        "--require_active_change",
        action="store_true",
        help="Fail validation if any selected active adapter slot is unchanged",
    )
    parser.add_argument("--rank_rel_tol", type=float, default=1e-6)

    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        print("Riemannian LoRASA-EGGROLL helper self-test passed.")
        return 0

    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --self_test is used")

    ckpt = load_checkpoint(args.checkpoint)
    if args.reference_checkpoint is not None:
        ref_ckpt = load_checkpoint(args.reference_checkpoint)
        validation = validate_actor_update_against_reference(
            extract_actor_params(ref_ckpt),
            extract_actor_params(ckpt),
            active_slots=args.active_slots,
            target_rank=args.target_rank,
            require_active_change=args.require_active_change,
            rank_rel_tol=args.rank_rel_tol,
        )
        text = json.dumps(validation, indent=2)
        print(text)
        if args.validation_json:
            out_path = Path(args.validation_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n")
        return 0 if validation["passed"] else 1

    summary = summarize_actor_adapters(
        extract_actor_params(ckpt),
        active_slots=args.active_slots,
        target_rank=args.target_rank,
    )
    text = json.dumps(summary, indent=2)
    print(text)
    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
