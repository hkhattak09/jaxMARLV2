"""Post-hoc LoRASA adapter compression tool.

Loads a trained no-recurrent LoRASA checkpoint, SVD-compresses selected LoRA
adapter matrices according to schedules A/B/C, and writes new checkpoint
variants.  Does not train, evaluate, instantiate SMAX, or modify model code.

Usage:
    python smax_ctm/tools/compress_lorasa_adapters.py \
        --checkpoint /path/to/no_recurrent_checkpoint_final.pkl \
        --output_dir compressed_lorasa \
        --active_slots 2,3,6

    python smax_ctm/tools/compress_lorasa_adapters.py --self_test
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_THIS_FILE = os.path.abspath(__file__)
_TOOL_DIR = os.path.dirname(_THIS_FILE)
_CANDIDATE_ROOTS = (
    os.path.abspath(os.path.join(_TOOL_DIR, "..", "..")),
    os.path.abspath(os.path.join(_TOOL_DIR, "..")),
    os.getcwd(),
)
for _root in _CANDIDATE_ROOTS:
    _smax_ctm_dir = os.path.join(_root, "smax_ctm")
    for _path in (_root, _smax_ctm_dir):
        if os.path.isdir(_path) and _path not in sys.path:
            sys.path.insert(0, _path)


try:
    from flax.traverse_util import flatten_dict, unflatten_dict

    _HAS_FLAX = True
except ImportError:
    _HAS_FLAX = False


try:
    import jax

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


RECURRENT_BLOCK_SUBSTRINGS = (
    "recurrent_reset",
    "recurrent_update",
    "recurrent_candidate",
)

SCHEDULE_A_RULES: Optional[Dict[str, int]] = None  # all blocks -> rank 4

SCHEDULE_B_RULES: Dict[str, int] = {
    "base_0": 4,
    "base_1": 4,
    "base_2": 4,
    "rnn/gru_cell/input_reset": 2,
    "rnn/gru_cell/input_update": 2,
    "rnn/gru_cell/input_candidate": 2,
    "action_out": 4,
}

SCHEDULE_C_RULES: Dict[str, int] = {
    "base_0": 4,
    "base_1": 4,
    "base_2": 4,
    "rnn/gru_cell/input_candidate": 4,
    "rnn/gru_cell/input_reset": 2,
    "rnn/gru_cell/input_update": 2,
    "action_out": 4,
}

ALL_SCHEDULES: Dict[str, Optional[Dict[str, int]]] = {
    "A": SCHEDULE_A_RULES,
    "B": SCHEDULE_B_RULES,
    "C": SCHEDULE_C_RULES,
}

DEFAULT_RANK_ALL = 4


def _load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a checkpoint from a pickle file.

    Handles JAX Array serialization if JAX is available.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, "rb") as f:
        try:
            data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint at {path}: {e}"
            ) from e
    if _HAS_JAX:
        data = _convert_jax_to_numpy(data)
    return data


def _convert_jax_to_numpy(tree: Any) -> Any:
    """Recursively convert jax.Arrays to numpy arrays."""
    if _HAS_JAX:
        import jax.numpy as jnp

        if isinstance(tree, jnp.ndarray):
            return np.asarray(tree)
    if isinstance(tree, dict):
        return {k: _convert_jax_to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        converted = [_convert_jax_to_numpy(item) for item in tree]
        return type(tree)(converted)
    return tree


def _save_pickle(obj: Any, path: str) -> None:
    """Save object to pickle file, creating parent directories."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _normalize_flat_key(key) -> Tuple[str, ...]:
    """Normalize a flat dict key to a tuple of strings.

    Flax flatten_dict with sep='/' returns string keys like 'a/b/c'.
    Custom flatten returns tuple keys like ('a', 'b', 'c').
    This function normalizes both to tuple form.
    """
    if isinstance(key, str):
        return tuple(key.split("/"))
    if isinstance(key, tuple):
        result: Tuple[str, ...] = ()
        for part in key:
            if isinstance(part, str):
                result += (part,)
            else:
                result += (str(part),)
        return result
    return (str(key),)


def _denormalize_flat_keys(
    flat: Dict[Tuple[str, ...], Any], original_keys
) -> Dict:
    """Convert tuple keys back to the original key format for unflattening.

    original_keys is the set of keys from the original flattened dict
    (before normalization), used to determine the target key format.
    """
    if not original_keys:
        return {k: v for k, v in flat.items()}

    sample_key = next(iter(original_keys))
    if isinstance(sample_key, tuple):
        return {k: v for k, v in flat.items()}
    if isinstance(sample_key, str):
        return {"/".join(k): v for k, v in flat.items()}
    return {k: v for k, v in flat.items()}


def _flatten_params(params: Dict) -> Tuple[Dict[Tuple[str, ...], Any], Any]:
    """Flatten a nested param dict to {(path, ...): value}.

    Returns (normalized_flat_dict, original_keys) where original_keys
    is the key set before normalization (needed for unflattening).
    """
    if _HAS_FLAX:
        raw = flatten_dict(params, keep_empty_nodes=False, sep="/")
    else:
        raw = _custom_flatten(params)

    original_keys = set(raw.keys())
    normalized = {}
    for k, v in raw.items():
        normalized[_normalize_flat_key(k)] = v
    return normalized, original_keys


def _custom_flatten(
    d: Dict, prefix: Tuple[str, ...] = ()
) -> Dict[Tuple[str, ...], Any]:
    """Custom recursive flattening fallback."""
    out: Dict[Tuple[str, ...], Any] = {}
    for k, v in d.items():
        key = prefix + (k,)
        if isinstance(v, dict) and v:
            out.update(_custom_flatten(v, key))
        else:
            out[key] = v
    return out


def _unflatten_params(
    flat: Dict[Tuple[str, ...], Any], original_keys: Any
) -> Dict:
    """Reconstruct a nested dict from flattened keys.

    original_keys is used to determine the correct key format for
    the unflatten function (string vs tuple).
    """
    denorm = _denormalize_flat_keys(flat, original_keys)

    if _HAS_FLAX:
        return unflatten_dict(denorm, sep="/")
    return _custom_unflatten(flat)


def _custom_unflatten(flat: Dict[Tuple[str, ...], Any]) -> Dict:
    """Custom recursive unflattening fallback."""
    result: Dict = {}
    for key_tuple, value in flat.items():
        d = result
        for part in key_tuple[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[key_tuple[-1]] = value
    return result


def _key_to_readable(key: Tuple[str, ...]) -> str:
    """Convert a key tuple to a readable path string."""
    return "/".join(str(k) for k in key)


def _discover_lora_blocks(
    flat_params: Dict[Tuple[str, ...], Any],
) -> List[Dict[str, Any]]:
    """Discover LoRA blocks by pairing lora_a/lora_b keys.

    Returns a list of dicts with keys:
        prefix: tuple of key components shared by the pair
        readable_path: human-readable path string
        lora_a_key: full key tuple for lora_a
        lora_b_key: full key tuple for lora_b
        lora_a: numpy array
        lora_b: numpy array
        num_slots: int
        configured_rank: int
    """
    a_keys: Dict[Tuple[str, ...], Tuple[str, ...]] = {}
    b_keys: Dict[Tuple[str, ...], Tuple[str, ...]] = {}

    for key in flat_params:
        if key[-1] == "lora_a":
            prefix = key[:-1]
            a_keys[prefix] = key
        elif key[-1] == "lora_b":
            prefix = key[:-1]
            b_keys[prefix] = key

    blocks = []
    for prefix in sorted(set(a_keys.keys()) & set(b_keys.keys())):
        a_arr = np.asarray(flat_params[a_keys[prefix]])
        b_arr = np.asarray(flat_params[b_keys[prefix]])

        if a_arr.ndim != 3 or b_arr.ndim != 3:
            warnings.warn(
                f"LoRA block {_key_to_readable(prefix)} has unexpected ndim: "
                f"lora_a.ndim={a_arr.ndim}, lora_b.ndim={b_arr.ndim}. Skipping."
            )
            continue

        num_slots_a, input_dim, rank_a = a_arr.shape
        num_slots_b, rank_b, output_dim = b_arr.shape

        if num_slots_a != num_slots_b:
            warnings.warn(
                f"LoRA block {_key_to_readable(prefix)} slot count mismatch: "
                f"lora_a slots={num_slots_a}, lora_b slots={num_slots_b}. Skipping."
            )
            continue

        if rank_a != rank_b:
            warnings.warn(
                f"LoRA block {_key_to_readable(prefix)} rank mismatch: "
                f"lora_a rank={rank_a}, lora_b rank={rank_b}. Skipping."
            )
            continue

        blocks.append({
            "prefix": prefix,
            "readable_path": _key_to_readable(prefix),
            "lora_a_key": a_keys[prefix],
            "lora_b_key": b_keys[prefix],
            "lora_a": a_arr,
            "lora_b": b_arr,
            "num_slots": num_slots_a,
            "configured_rank": rank_a,
        })

    orphan_a = set(a_keys.keys()) - set(b_keys.keys())
    orphan_b = set(b_keys.keys()) - set(a_keys.keys())
    for prefix in orphan_a:
        warnings.warn(
            f"Orphan lora_a without lora_b at {_key_to_readable(prefix)}"
        )
    for prefix in orphan_b:
        warnings.warn(
            f"Orphan lora_b without lora_a at {_key_to_readable(prefix)}"
        )

    return blocks


def _check_recurrent_blocks(
    blocks: List[Dict[str, Any]],
) -> List[str]:
    """Warn if recurrent LoRA blocks are present. Returns warning strings."""
    found: List[str] = []
    for block in blocks:
        path = block["readable_path"]
        for substr in RECURRENT_BLOCK_SUBSTRINGS:
            if substr in path:
                msg = (
                    f"WARNING: Recurrent LoRA block detected at {path}. "
                    f"This checkpoint may not be a no-recurrent checkpoint."
                )
                warnings.warn(msg)
                found.append(msg)
    return found


def _match_schedule(
    block: Dict[str, Any],
    schedule_name: str,
    schedule_rules: Optional[Dict[str, int]],
) -> Optional[int]:
    """Determine the target rank for a block under a schedule.

    Returns None if the block does not match any schedule rule
    (meaning it should be left unchanged with a warning).
    """
    path = block["readable_path"]

    if schedule_rules is None:
        return DEFAULT_RANK_ALL

    for pattern, target_rank in schedule_rules.items():
        if path == pattern or path.endswith("/" + pattern):
            return target_rank

    return None


def _compute_effective_rank(s: np.ndarray, eps: float = 1e-12) -> float:
    """Compute effective rank from singular values.

    effective_rank = exp(-sum(p * log(p + eps))) where p = s / s.sum()
    """
    total = s.sum()
    if total < eps:
        return 0.0
    p = s / total
    log_p = np.log(p + eps)
    entropy = -np.sum(p * log_p)
    return float(np.exp(entropy))


def _compress_slot(
    A: np.ndarray,
    B: np.ndarray,
    target_rank: int,
    configured_rank: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """SVD-compress a single slot's LoRA pair.

    Args:
        A: (input_dim, configured_rank)
        B: (configured_rank, output_dim)
        target_rank: desired compression rank
        configured_rank: original rank dimension size

    Returns:
        (new_A, new_B, metrics_dict)
        new_A and new_B have the same shapes as input A, B.
    """
    A_f64 = A.astype(np.float64)
    B_f64 = B.astype(np.float64)

    Delta = A_f64 @ B_f64
    original_fro = float(np.linalg.norm(Delta, "fro"))

    is_zero = np.allclose(Delta, 0.0, atol=1e-15)

    if is_zero:
        new_A = np.zeros_like(A)
        new_B = np.zeros_like(B)
        metrics = {
            "configured_rank": configured_rank,
            "target_rank": target_rank,
            "effective_written_rank": 0,
            "original_fro_norm": 0.0,
            "compressed_fro_norm": 0.0,
            "relative_fro_error": 0.0,
            "energy_retained": 1.0,
            "original_effective_rank": 0.0,
            "compressed_effective_rank": 0.0,
        }
        return new_A, new_B, metrics

    U, s, Vt = np.linalg.svd(Delta, full_matrices=False)

    r_eff = min(target_rank, len(s), configured_rank)

    sqrt_s = np.sqrt(s[:r_eff])
    A_comp = U[:, :r_eff] * sqrt_s[None, :]
    B_comp = sqrt_s[:, None] * Vt[:r_eff, :]

    new_A = np.zeros_like(A_f64)
    new_B = np.zeros_like(B_f64)
    new_A[:, :r_eff] = A_comp
    new_B[:r_eff, :] = B_comp

    new_A = new_A.astype(A.dtype)
    new_B = new_B.astype(B.dtype)

    Delta_comp = new_A.astype(np.float64) @ new_B.astype(np.float64)
    compressed_fro = float(np.linalg.norm(Delta_comp, "fro"))

    error_norm = float(np.linalg.norm(Delta - Delta_comp, "fro"))
    relative_fro_error = error_norm / original_fro if original_fro > 0 else 0.0

    energy_total = float(np.sum(s ** 2))
    energy_retained = float(np.sum(s[:r_eff] ** 2)) / energy_total if energy_total > 0 else 1.0

    original_effective_rank = _compute_effective_rank(s)
    compressed_effective_rank = _compute_effective_rank(s[:r_eff])

    metrics = {
        "configured_rank": configured_rank,
        "target_rank": target_rank,
        "effective_written_rank": r_eff,
        "original_fro_norm": original_fro,
        "compressed_fro_norm": compressed_fro,
        "relative_fro_error": relative_fro_error,
        "energy_retained": energy_retained,
        "original_effective_rank": original_effective_rank,
        "compressed_effective_rank": compressed_effective_rank,
    }

    return new_A, new_B, metrics


def compress_checkpoint(
    checkpoint: Dict[str, Any],
    schedule_name: str,
    schedule_rules: Optional[Dict[str, int]],
    active_slots: List[int],
    zero_inactive: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[str], List[str]]:
    """Compress a checkpoint according to a schedule.

    Args:
        checkpoint: full checkpoint dict (must contain 'actor_params')
        schedule_name: 'A', 'B', or 'C'
        schedule_rules: schedule-specific rules dict or None for all->4
        active_slots: list of slot indices to compress
        zero_inactive: if True, zero out inactive slot LoRA params

    Returns:
        (new_checkpoint, metadata, report_metrics, warnings_list, errors_list)
    """
    ckpt = copy.deepcopy(checkpoint)

    if "actor_params" not in ckpt:
        err = "Checkpoint does not contain 'actor_params' key."
        return ckpt, {}, [], [err]

    actor_params = ckpt["actor_params"]
    flat, original_keys = _flatten_params(actor_params)

    blocks = _discover_lora_blocks(flat)

    recurrent_warnings = _check_recurrent_blocks(blocks)

    all_warnings: List[str] = list(recurrent_warnings)
    all_errors: List[str] = []
    compressed_blocks: List[str] = []
    report_metrics: List[Dict[str, Any]] = []
    rules_applied: Dict[str, int] = {}

    all_slots: Optional[set] = None
    if blocks:
        all_slots = set(range(blocks[0]["num_slots"]))
    active_set = set(active_slots)
    inactive_set = all_slots - active_set if all_slots is not None else set()

    for block in blocks:
        path = block["readable_path"]
        prefix = block["prefix"]
        target_rank = _match_schedule(block, schedule_name, schedule_rules)

        if target_rank is None:
            msg = (
                f"LoRA block '{path}' does not match any rule in "
                f"schedule {schedule_name}. Leaving unchanged."
            )
            warnings.warn(msg)
            all_warnings.append(msg)
            continue

        rules_applied[path] = target_rank

        lora_a = block["lora_a"].copy()
        lora_b = block["lora_b"].copy()
        configured_rank = block["configured_rank"]

        for slot in active_slots:
            if slot >= block["num_slots"]:
                msg = (
                    f"Slot {slot} out of range for block '{path}' "
                    f"(num_slots={block['num_slots']}). Skipping slot."
                )
                all_warnings.append(msg)
                continue

            A_slot = lora_a[slot]
            B_slot = lora_b[slot]

            new_A, new_B, metrics = _compress_slot(
                A_slot, B_slot, target_rank, configured_rank
            )

            lora_a[slot] = new_A
            lora_b[slot] = new_B

            metrics["path"] = path
            metrics["slot"] = slot
            report_metrics.append(metrics)

        if zero_inactive:
            for slot in inactive_set:
                if slot >= block["num_slots"]:
                    continue
                lora_a[slot] = np.zeros_like(lora_a[slot])
                lora_b[slot] = np.zeros_like(lora_b[slot])

        flat[block["lora_a_key"]] = lora_a
        flat[block["lora_b_key"]] = lora_b
        compressed_blocks.append(path)

    new_actor_params = _unflatten_params(flat, original_keys)
    ckpt["actor_params"] = new_actor_params

    metadata = {
        "source_checkpoint": checkpoint.get("lorasa_compression", {}).get(
            "source_checkpoint", ""
        ),
        "schedule": schedule_name,
        "active_slots": sorted(active_slots),
        "kept_original_shapes": True,
        "zero_inactive_slots": zero_inactive,
        "rules": rules_applied,
        "compressed_blocks": compressed_blocks,
        "warnings": all_warnings,
        "errors": all_errors,
    }

    existing = ckpt.get("lorasa_compression")
    if existing and isinstance(existing, dict):
        existing.update(metadata)
        ckpt["lorasa_compression"] = existing
    else:
        ckpt["lorasa_compression"] = metadata

    return ckpt, metadata, report_metrics, all_warnings, all_errors


def _report_json(
    metrics: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    indent: int = 2,
) -> str:
    """Produce a JSON report string."""
    obj = {
        "metadata": metadata,
        "metrics": metrics,
    }
    return json.dumps(obj, indent=indent, default=str)


def _report_md(
    metrics: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> str:
    """Produce a Markdown report string."""
    lines: List[str] = []
    lines.append(f"# Compression Report — Schedule {metadata.get('schedule', '?')}")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    for k, v in metadata.items():
        if k in ("warnings", "errors", "rules", "compressed_blocks"):
            lines.append(f"- **{k}**: {v}")
        else:
            lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    if not metrics:
        lines.append("No blocks were compressed.")
        return "\n".join(lines)

    lines.append("## Per-Block/Slot Metrics")
    lines.append("")
    lines.append(
        "| path | slot | cfg_rank | tgt_rank | eff_rank | orig_fro | comp_fro | "
        "rel_fro_err | energy_ret | orig_eff_rank | comp_eff_rank |"
    )
    lines.append(
        "|------|------|---------|---------|---------|----------|---------| "
        "----------- | --------- | ------------- | ------------- |"
    )
    for m in metrics:
        lines.append(
            f"| {m.get('path','')} "
            f"| {m.get('slot','')} "
            f"| {m.get('configured_rank','')} "
            f"| {m.get('target_rank','')} "
            f"| {m.get('effective_written_rank','')} "
            f"| {m.get('original_fro_norm',0):.4f} "
            f"| {m.get('compressed_fro_norm',0):.4f} "
            f"| {m.get('relative_fro_error',0):.6f} "
            f"| {m.get('energy_retained',0):.6f} "
            f"| {m.get('original_effective_rank',0):.2f} "
            f"| {m.get('compressed_effective_rank',0):.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _summary_md(
    results: Dict[str, Dict[str, Any]],
) -> str:
    """Produce a top-level summary Markdown."""
    lines: List[str] = []
    lines.append("# LoRASA Adapter Compression Summary")
    lines.append("")
    for sched, res in sorted(results.items()):
        meta = res.get("metadata", {})
        n_compressed = len(meta.get("compressed_blocks", []))
        n_warnings = len(meta.get("warnings", []))
        n_errors = len(meta.get("errors", []))
        metrics = res.get("metrics", [])
        avg_energy = (
            sum(m.get("energy_retained", 0) for m in metrics) / len(metrics)
            if metrics
            else 0.0
        )
        avg_error = (
            sum(m.get("relative_fro_error", 0) for m in metrics) / len(metrics)
            if metrics
            else 0.0
        )
        lines.append(f"## Schedule {sched}")
        lines.append(f"- Blocks compressed: {n_compressed}")
        lines.append(f"- Warnings: {n_warnings}, Errors: {n_errors}")
        lines.append(f"- Avg energy retained: {avg_energy:.6f}")
        lines.append(f"- Avg relative Frobenius error: {avg_error:.6f}")
        lines.append("")
    return "\n".join(lines)


def run_compression(
    checkpoint_path: str,
    output_dir: str,
    active_slots: List[int],
    schedules: List[str],
    zero_inactive: bool,
    save_actor_params_only: bool,
    json_indent: int,
) -> Dict[str, Dict[str, Any]]:
    """Main compression pipeline.

    Returns dict mapping schedule name -> {metadata, metrics, checkpoint, actor_params}.
    """
    source_checkpoint = os.path.abspath(checkpoint_path)
    ckpt = _load_checkpoint(checkpoint_path)

    os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}

    for sched_name in schedules:
        rules = ALL_SCHEDULES.get(sched_name)
        if rules is None and sched_name != "A":
            warnings.warn(f"Unknown schedule '{sched_name}', skipping.")
            continue

        new_ckpt, metadata, metrics, warn_list, err_list = compress_checkpoint(
            ckpt, sched_name, rules, active_slots, zero_inactive
        )
        metadata["source_checkpoint"] = source_checkpoint
        if isinstance(new_ckpt.get("lorasa_compression"), dict):
            new_ckpt["lorasa_compression"]["source_checkpoint"] = source_checkpoint

        sched_dir = os.path.join(output_dir, f"schedule_{sched_name}")
        os.makedirs(sched_dir, exist_ok=True)

        ckpt_path = os.path.join(
            sched_dir, f"checkpoint_final_compressed_{sched_name}.pkl"
        )
        _save_pickle(new_ckpt, ckpt_path)

        report_json_path = os.path.join(sched_dir, "compression_report.json")
        with open(report_json_path, "w") as f:
            f.write(_report_json(metrics, metadata, indent=json_indent))

        report_md_path = os.path.join(sched_dir, "compression_report.md")
        with open(report_md_path, "w") as f:
            f.write(_report_md(metrics, metadata))

        actor_params = None
        if save_actor_params_only:
            actor_params = new_ckpt.get("actor_params")
            ap_path = os.path.join(
                sched_dir, f"actor_params_compressed_{sched_name}.pkl"
            )
            _save_pickle(actor_params, ap_path)

        results[sched_name] = {
            "metadata": metadata,
            "metrics": metrics,
            "checkpoint": new_ckpt,
            "actor_params": actor_params,
        }

    summary_path = os.path.join(output_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write(_summary_md(results))

    return results


def _build_fake_checkpoint() -> Dict[str, Any]:
    """Build a fake checkpoint for self-test."""
    num_slots = 7
    rank = 8
    hidden = 64
    action_dim = 10
    input_dim = 32

    np.random.seed(42)

    actor_params = {
        "params": {
            "feature_norm": {
                "scale": np.ones(input_dim, dtype=np.float32),
                "bias": np.zeros(input_dim, dtype=np.float32),
            },
            "base_0": {
                "kernel": np.random.randn(input_dim, hidden).astype(np.float32),
                "bias": np.zeros(hidden, dtype=np.float32),
                "lora_a": np.random.randn(num_slots, input_dim, rank).astype(
                    np.float32
                ),
                "lora_b": (np.random.randn(num_slots, rank, hidden) * 0.01).astype(
                    np.float32
                ),
            },
            "base_norm_0": {
                "scale": np.ones(hidden, dtype=np.float32),
                "bias": np.zeros(hidden, dtype=np.float32),
            },
            "base_1": {
                "kernel": np.random.randn(hidden, hidden).astype(np.float32),
                "bias": np.zeros(hidden, dtype=np.float32),
                "lora_a": np.random.randn(num_slots, hidden, rank).astype(
                    np.float32
                ),
                "lora_b": (np.random.randn(num_slots, rank, hidden) * 0.01).astype(
                    np.float32
                ),
            },
            "base_norm_1": {
                "scale": np.ones(hidden, dtype=np.float32),
                "bias": np.zeros(hidden, dtype=np.float32),
            },
            "base_2": {
                "kernel": np.random.randn(hidden, hidden).astype(np.float32),
                "bias": np.zeros(hidden, dtype=np.float32),
                "lora_a": np.random.randn(num_slots, hidden, rank).astype(
                    np.float32
                ),
                "lora_b": (np.random.randn(num_slots, rank, hidden) * 0.01).astype(
                    np.float32
                ),
            },
            "base_norm_2": {
                "scale": np.ones(hidden, dtype=np.float32),
                "bias": np.zeros(hidden, dtype=np.float32),
            },
            "rnn": {
                "gru_cell": {
                    "input_reset": {
                        "kernel": np.random.randn(hidden, hidden).astype(np.float32),
                        "bias": np.zeros(hidden, dtype=np.float32),
                        "lora_a": np.random.randn(num_slots, hidden, rank).astype(
                            np.float32
                        ),
                        "lora_b": (
                            np.random.randn(num_slots, rank, hidden) * 0.01
                        ).astype(np.float32),
                    },
                    "input_update": {
                        "kernel": np.random.randn(hidden, hidden).astype(np.float32),
                        "bias": np.zeros(hidden, dtype=np.float32),
                        "lora_a": np.random.randn(num_slots, hidden, rank).astype(
                            np.float32
                        ),
                        "lora_b": (
                            np.random.randn(num_slots, rank, hidden) * 0.01
                        ).astype(np.float32),
                    },
                    "input_candidate": {
                        "kernel": np.random.randn(hidden, hidden).astype(np.float32),
                        "bias": np.zeros(hidden, dtype=np.float32),
                        "lora_a": np.random.randn(num_slots, hidden, rank).astype(
                            np.float32
                        ),
                        "lora_b": (
                            np.random.randn(num_slots, rank, hidden) * 0.01
                        ).astype(np.float32),
                    },
                },
                "rnn_norm": {
                    "scale": np.ones(hidden, dtype=np.float32),
                    "bias": np.zeros(hidden, dtype=np.float32),
                },
            },
            "action_out": {
                "kernel": np.random.randn(hidden, action_dim).astype(np.float32),
                "bias": np.zeros(action_dim, dtype=np.float32),
                "lora_a": np.random.randn(num_slots, hidden, rank).astype(
                    np.float32
                ),
                "lora_b": (
                    np.random.randn(num_slots, rank, action_dim) * 0.01
                ).astype(np.float32),
            },
        }
    }

    checkpoint = {
        "actor_params": actor_params,
        "critic_params": {"some_key": "preserved"},
        "optim_state": {"step": 1000},
    }
    return checkpoint


def _self_test() -> bool:
    """Run self-test. Returns True on success."""
    print("=== LoRASA Adapter Compression Self-Test ===")
    print()

    ckpt = _build_fake_checkpoint()
    active_slots = [2, 3, 6]
    num_slots = 5

    passed = True
    errors: List[str] = []

    for sched_name in ["A", "B", "C"]:
        rules = ALL_SCHEDULES[sched_name]
        new_ckpt, metadata, metrics, warn_list, err_list = compress_checkpoint(
            ckpt, sched_name, rules, active_slots, zero_inactive=False
        )

        print(f"--- Schedule {sched_name} ---")
        print(f"  Compressed blocks: {metadata.get('compressed_blocks', [])}")
        print(f"  Warnings: {len(warn_list)}")
        print(f"  Errors: {len(err_list)}")

        if err_list:
            errors.extend(err_list)
            passed = False

        flat_new, _ = _flatten_params(new_ckpt["actor_params"])
        flat_orig, _ = _flatten_params(ckpt["actor_params"])

        blocks_new = _discover_lora_blocks(flat_new)
        blocks_orig = _discover_lora_blocks(flat_orig)

        block_map_new = {b["readable_path"]: b for b in blocks_new}
        block_map_orig = {b["readable_path"]: b for b in blocks_orig}

        for block in blocks_new:
            path = block["readable_path"]

            if path not in block_map_orig:
                continue

            orig_block = block_map_orig[path]

            new_a = block["lora_a"]
            new_b = block["lora_b"]
            orig_a = orig_block["lora_a"]
            orig_b = orig_block["lora_b"]

            if new_a.shape != orig_a.shape:
                msg = (
                    f"[{sched_name}] Shape mismatch for lora_a at {path}: "
                    f"original={orig_a.shape}, new={new_a.shape}"
                )
                errors.append(msg)
                passed = False

            if new_b.shape != orig_b.shape:
                msg = (
                    f"[{sched_name}] Shape mismatch for lora_b at {path}: "
                    f"original={orig_b.shape}, new={new_b.shape}"
                )
                errors.append(msg)
                passed = False

            target_rank = _match_schedule(block, sched_name, rules)
            if target_rank is None:
                for slot in active_slots:
                    if slot < block["num_slots"]:
                        if not np.array_equal(new_a[slot], orig_a[slot]):
                            msg = (
                                f"[{sched_name}] Unmatched block {path} slot {slot} "
                                f"lora_a was modified but shouldn't be"
                            )
                            errors.append(msg)
                            passed = False
                continue

            for slot in active_slots:
                if slot >= block["num_slots"]:
                    continue

                A_slot = new_a[slot]
                B_slot = new_b[slot]
                r_eff = 0
                for r in range(A_slot.shape[1]):
                    if np.any(A_slot[:, r] != 0) or np.any(B_slot[r, :] != 0):
                        r_eff = r + 1

                if r_eff > target_rank:
                    msg = (
                        f"[{sched_name}] Effective rank {r_eff} > target {target_rank} "
                        f"at {path} slot {slot}"
                    )
                    errors.append(msg)
                    passed = False

                print(
                    f"  {path} slot {slot}: eff_rank={r_eff}, "
                    f"target={target_rank}, shapes OK"
                )

            for slot in range(block["num_slots"]):
                if slot not in active_slots:
                    if not np.array_equal(new_a[slot], orig_a[slot]):
                        msg = (
                            f"[{sched_name}] Inactive slot {slot} lora_a at {path} "
                            f"was modified but shouldn't be"
                        )
                        errors.append(msg)
                        passed = False
                    if not np.array_equal(new_b[slot], orig_b[slot]):
                        msg = (
                            f"[{sched_name}] Inactive slot {slot} lora_b at {path} "
                            f"was modified but shouldn't be"
                        )
                        errors.append(msg)
                        passed = False

        has_lorasa_meta = "lorasa_compression" in new_ckpt
        if not has_lorasa_meta:
            msg = f"[{sched_name}] Missing lorasa_compression metadata"
            errors.append(msg)
            passed = False
        else:
            meta_sched = new_ckpt["lorasa_compression"].get("schedule")
            if meta_sched != sched_name:
                msg = f"[{sched_name}] Metadata schedule mismatch: {meta_sched}"
                errors.append(msg)
                passed = False

        has_energy = all("energy_retained" in m for m in metrics)
        has_fro_error = all("relative_fro_error" in m for m in metrics)
        if not has_energy:
            msg = f"[{sched_name}] Missing energy_retained in metrics"
            errors.append(msg)
            passed = False
        if not has_fro_error:
            msg = f"[{sched_name}] Missing relative_fro_error in metrics"
            errors.append(msg)
            passed = False

    for key in ["critic_params", "optim_state"]:
        if ckpt.get(key) != _build_fake_checkpoint().get(key):
            if key not in ckpt or ckpt[key] is None:
                continue

    print()
    print("--- Schedule-specific checks ---")

    sched_a_ckpt, _, _, _, _ = compress_checkpoint(
        ckpt, "A", SCHEDULE_A_RULES, active_slots, zero_inactive=False
    )
    flat_a, _ = _flatten_params(sched_a_ckpt["actor_params"])
    blocks_a = _discover_lora_blocks(flat_a)
    for b in blocks_a:
        for slot in active_slots:
            if slot >= b["num_slots"]:
                continue
            A_s = b["lora_a"][slot]
            r_eff = 0
            for r in range(A_s.shape[1]):
                if np.any(A_s[:, r] != 0):
                    r_eff = r + 1
            if r_eff > 4:
                msg = f"[A] Rank {r_eff} > 4 at {b['readable_path']} slot {slot}"
                errors.append(msg)
                passed = False
            else:
                print(f"  [A] {b['readable_path']} slot {slot}: rank <= 4 OK")

    sched_b_ckpt, _, _, _, _ = compress_checkpoint(
        ckpt, "B", SCHEDULE_B_RULES, active_slots, zero_inactive=False
    )
    flat_b, _ = _flatten_params(sched_b_ckpt["actor_params"])
    blocks_b = _discover_lora_blocks(flat_b)
    for b in blocks_b:
        path = b["readable_path"]
        if "input_reset" in path or "input_update" in path or "input_candidate" in path:
            for slot in active_slots:
                if slot >= b["num_slots"]:
                    continue
                A_s = b["lora_a"][slot]
                r_eff = 0
                for r in range(A_s.shape[1]):
                    if np.any(A_s[:, r] != 0):
                        r_eff = r + 1
                if r_eff > 2:
                    msg = (
                        f"[B] Rank {r_eff} > 2 at {b['readable_path']} slot {slot} "
                        f"(expected rank 2 for input_reset/input_update/input_candidate)"
                    )
                    errors.append(msg)
                    passed = False
                else:
                    print(
                        f"  [B] {b['readable_path']} slot {slot}: "
                        f"rank <= 2 OK"
                    )

    sched_c_ckpt, _, _, _, _ = compress_checkpoint(
        ckpt, "C", SCHEDULE_C_RULES, active_slots, zero_inactive=False
    )
    flat_c, _ = _flatten_params(sched_c_ckpt["actor_params"])
    blocks_c = _discover_lora_blocks(flat_c)
    for b in blocks_c:
        path = b["readable_path"]
        if "input_candidate" in path:
            for slot in active_slots:
                if slot >= b["num_slots"]:
                    continue
                A_s = b["lora_a"][slot]
                r_eff = 0
                for r in range(A_s.shape[1]):
                    if np.any(A_s[:, r] != 0):
                        r_eff = r + 1
                if r_eff > 4:
                    msg = (
                        f"[C] Rank {r_eff} > 4 at {b['readable_path']} slot {slot} "
                        f"(expected rank 4 for input_candidate)"
                    )
                    errors.append(msg)
                    passed = False
                else:
                    print(
                        f"  [C] {b['readable_path']} slot {slot}: "
                        f"rank <= 4 OK"
                    )
        if "input_reset" in path or "input_update" in path:
            for slot in active_slots:
                if slot >= b["num_slots"]:
                    continue
                A_s = b["lora_a"][slot]
                r_eff = 0
                for r in range(A_s.shape[1]):
                    if np.any(A_s[:, r] != 0):
                        r_eff = r + 1
                if r_eff > 2:
                    msg = (
                        f"[C] Rank {r_eff} > 2 at {b['readable_path']} slot {slot} "
                        f"(expected rank 2 for input_reset/input_update)"
                    )
                    errors.append(msg)
                    passed = False
                else:
                    print(
                        f"  [C] {b['readable_path']} slot {slot}: "
                        f"rank <= 2 OK"
                    )

    print()
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
    print()
    if passed:
        print("=== SELF-TEST PASSED ===")
    else:
        print("=== SELF-TEST FAILED ===")
    return passed


def _parse_active_slots(s: str) -> List[int]:
    """Parse comma-separated active slots string."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_schedules(s: str) -> List[str]:
    """Parse comma-separated schedules string."""
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc LoRASA adapter compression tool"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to no_recurrent LoRASA checkpoint .pkl file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="compressed_lorasa",
        help="Output directory for compressed checkpoints and reports",
    )
    parser.add_argument(
        "--active_slots",
        type=str,
        default="2,3,6",
        help="Comma-separated list of active slot indices (default: 2,3,6)",
    )
    parser.add_argument(
        "--schedules",
        type=str,
        default="A,B,C",
        help="Comma-separated list of schedules to apply (default: A,B,C)",
    )
    parser.add_argument(
        "--zero_inactive_slots",
        action="store_true",
        default=False,
        help="Zero out LoRA params for inactive slots",
    )
    parser.add_argument(
        "--save_actor_params_only",
        action="store_true",
        default=False,
        help="Also save actor_params only as separate .pkl files",
    )
    parser.add_argument(
        "--json_indent",
        type=int,
        default=2,
        help="JSON indentation level for reports (default: 2)",
    )
    parser.add_argument(
        "--self_test",
        action="store_true",
        default=False,
        help="Run self-test with fake checkpoint",
    )

    args = parser.parse_args()

    if args.self_test:
        success = _self_test()
        sys.exit(0 if success else 1)

    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --self_test is used")

    active_slots = _parse_active_slots(args.active_slots)
    schedules = _parse_schedules(args.schedules)

    try:
        results = run_compression(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            active_slots=active_slots,
            schedules=schedules,
            zero_inactive=args.zero_inactive_slots,
            save_actor_params_only=args.save_actor_params_only,
            json_indent=args.json_indent,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Compression complete. Output in: {args.output_dir}")
    for sched_name, res in sorted(results.items()):
        n_blocks = len(res["metadata"].get("compressed_blocks", []))
        n_warn = len(res["metadata"].get("warnings", []))
        print(f"  Schedule {sched_name}: {n_blocks} blocks compressed, {n_warn} warnings")


if __name__ == "__main__":
    main()
