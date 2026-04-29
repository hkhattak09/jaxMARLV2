#!/usr/bin/env python3
"""Dependency-light spectral diagnostic for trained LoRASA actor adapters.

Only uses Python stdlib + numpy.  No JAX / Flax / pandas / repo training code.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import pickle
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: str | Path) -> Any:
    """Load a checkpoint pickle, catching JAX-missing errors gracefully."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        # Typical JAX-specific unpickle failures
        msg = (
            "Could not unpickle checkpoint without JAX. "
            "This script is NumPy-only; rerun in an environment with JAX installed "
            "or resave actor_params as NumPy arrays."
        )
        print(msg, file=sys.stderr)
        raise RuntimeError(msg) from e


def extract_actor_params(obj: Any) -> tuple[Any, dict[str, Any]]:
    """Return (actor_params, metadata_dict)."""
    metadata: dict[str, Any] = {}
    if isinstance(obj, Mapping) and "actor_params" in obj:
        # Full checkpoint dict
        actor_params = obj["actor_params"]
        for key in ("model_type", "checkpoint_kind", "step", "update", "actor_step"):
            if key in obj:
                metadata[key] = obj[key]
        if "config" in obj and isinstance(obj["config"], Mapping):
            cfg = obj["config"]
            if "MAP_NAME" in cfg:
                metadata["map_name"] = cfg["MAP_NAME"]
            if "lorasa" in cfg and isinstance(cfg["lorasa"], Mapping):
                lorasa_cfg = cfg["lorasa"]
                for subkey in ("rank", "num_adapter_slots"):
                    if subkey in lorasa_cfg:
                        metadata[f"lorasa_{subkey}"] = lorasa_cfg[subkey]
        return actor_params, metadata
    # Assume raw actor params tree
    return obj, metadata


def flatten_tree(tree: Any, path: tuple = ()) -> dict[tuple[str, ...], Any]:
    """Recursively flatten a nested structure of dicts / FrozenDict-like / list / tuple.

    Returns a mapping from tuple-paths to leaf values.
    """
    flat: dict[tuple[str, ...], Any] = {}
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            flat.update(flatten_tree(v, path + (str(k),)))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            flat.update(flatten_tree(v, path + (str(i),)))
    else:
        flat[path] = tree
    return flat


def path_to_str(path: tuple[str, ...]) -> str:
    return "/".join(path)


def discover_lora_pairs(flat: dict[tuple[str, ...], Any]) -> list[dict[str, Any]]:
    """Dynamically discover LoRA A/B/kernel triplets by shared prefix.

    Returns a list of dicts with keys:
        prefix_tuple, prefix_str, lora_a_key, lora_b_key, kernel_key (or None)
    """
    # Collect keys ending in lora_a, lora_b, kernel
    lora_a_keys: list[tuple[str, ...]] = []
    lora_b_keys: list[tuple[str, ...]] = []
    kernel_keys: list[tuple[str, ...]] = []

    for path in flat:
        if not path:
            continue
        if path[-1] == "lora_a":
            lora_a_keys.append(path)
        elif path[-1] == "lora_b":
            lora_b_keys.append(path)
        elif path[-1] == "kernel":
            kernel_keys.append(path)

    # Build set of prefixes that have both A and B
    a_prefixes: dict[tuple[str, ...], tuple[str, ...]] = {p[:-1]: p for p in lora_a_keys}
    b_prefixes: dict[tuple[str, ...], tuple[str, ...]] = {p[:-1]: p for p in lora_b_keys}
    common_prefixes = sorted(set(a_prefixes.keys()) & set(b_prefixes.keys()))

    kernel_prefix_map: dict[tuple[str, ...], tuple[str, ...]] = {}
    for k in kernel_keys:
        prefix = k[:-1]
        # The kernel may be directly under the same prefix as A/B,
        # or one level deeper (e.g., params/dense/kernel vs params/dense/lora_a).
        # We try exact prefix match first.
        kernel_prefix_map[prefix] = k

    results: list[dict[str, Any]] = []
    for prefix in common_prefixes:
        a_key = a_prefixes[prefix]
        b_key = b_prefixes[prefix]
        kernel_key = kernel_prefix_map.get(prefix)
        results.append(
            {
                "prefix_tuple": prefix,
                "prefix_str": path_to_str(prefix),
                "lora_a_key": a_key,
                "lora_b_key": b_key,
                "kernel_key": kernel_key,
            }
        )
    return results


def to_numpy_array(x: Any) -> np.ndarray:
    """Convert a leaf (possibly JAX array, np.ndarray, list) to np.ndarray."""
    if isinstance(x, np.ndarray):
        return x
    # Handle JAX arrays via their duck-typed __array__ if present
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return np.array(x)


def frobenius_norm(arr: np.ndarray) -> float:
    return float(np.linalg.norm(arr, "fro"))


def compute_slot_metrics(
    A: np.ndarray,
    B: np.ndarray,
    kernel: np.ndarray | None,
    abs_tol: float,
    rel_tols: list[float],
) -> dict[str, Any]:
    """Compute all spectral and norm metrics for a single slot."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    input_dim, configured_rank = A.shape
    rank_b, output_dim = B.shape

    # Basic validation (should already be checked by caller)
    if configured_rank != rank_b:
        raise ValueError(
            f"Shape mismatch: A rank {configured_rank} vs B rank {rank_b}"
        )

    Delta = A @ B
    s = np.linalg.svd(Delta, compute_uv=False)

    max_sv = float(s[0]) if s.size > 0 else 0.0
    min_sv = float(s[-1]) if s.size > 0 else 0.0

    # min nonzero sv
    nonzero_mask = s > abs_tol
    if np.any(nonzero_mask):
        min_nonzero_sv = float(np.min(s[nonzero_mask]))
    else:
        min_nonzero_sv = None

    # condition number
    if min_nonzero_sv is not None and min_nonzero_sv > 0:
        condition_number = max_sv / min_nonzero_sv
    else:
        condition_number = None

    # effective rank
    s_sum = float(np.sum(s))
    if s_sum > 0:
        p = s / s_sum
        eps = np.finfo(float).eps
        entropy = -np.sum(p * np.log(p + eps))
        effective_rank = math.exp(float(entropy))
    else:
        effective_rank = 0.0

    # stable rank
    delta_fro = frobenius_norm(Delta)
    if max_sv > 0:
        stable_rank = (delta_fro**2) / (max_sv**2)
    else:
        stable_rank = 0.0

    # numerical ranks
    numerical_ranks: dict[str, Any] = {}
    numerical_ranks[f"numerical_rank_abs_{abs_tol}"] = int(np.sum(s > abs_tol))
    for rt in rel_tols:
        numerical_ranks[f"numerical_rank_rel_{rt}"] = int(np.sum(s > (rt * max_sv)))

    # norms
    a_fro = frobenius_norm(A)
    b_fro = frobenius_norm(B)
    adapter_fro_norm = delta_fro

    if kernel is not None:
        backbone_fro_norm = frobenius_norm(kernel)
        adapter_backbone_ratio = (
            adapter_fro_norm / backbone_fro_norm if backbone_fro_norm > 0 else None
        )
    else:
        backbone_fro_norm = None
        adapter_backbone_ratio = None

    if b_fro > 0:
        factor_balance_ratio = a_fro / b_fro
    else:
        factor_balance_ratio = None

    # Edge-case overrides
    if s_sum == 0:
        effective_rank = 0.0
        stable_rank = 0.0
        condition_number = None
        min_nonzero_sv = None

    metrics: dict[str, Any] = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "configured_rank": configured_rank,
        "singular_values": s.tolist(),
        "max_singular_value": max_sv,
        "min_singular_value": min_sv,
        "min_nonzero_singular_value": min_nonzero_sv,
        "condition_number": condition_number,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "adapter_fro_norm": adapter_fro_norm,
        "backbone_fro_norm": backbone_fro_norm,
        "adapter_backbone_ratio": adapter_backbone_ratio,
        "lora_a_fro_norm": a_fro,
        "lora_b_fro_norm": b_fro,
        "factor_balance_ratio": factor_balance_ratio,
    }
    metrics.update(numerical_ranks)
    return metrics


def flag_metrics(metrics: dict[str, Any], abs_tol: float) -> dict[str, bool]:
    """Compute boolean flags from metrics."""
    configured_rank = metrics["configured_rank"]
    effective_rank = metrics["effective_rank"]
    min_sv = metrics["min_singular_value"]
    condition_number = metrics["condition_number"]
    adapter_backbone_ratio = metrics["adapter_backbone_ratio"]
    factor_balance_ratio = metrics["factor_balance_ratio"]
    adapter_fro_norm = metrics["adapter_fro_norm"]

    numerical_rank_rel_1e_3 = metrics.get("numerical_rank_rel_0.001")

    flags: dict[str, bool] = {
        "effective_rank_lt_half_configured": effective_rank < 0.5 * configured_rank,
        "numerical_rank_rel_1e_3_lt_configured": (
            numerical_rank_rel_1e_3 is not None
            and numerical_rank_rel_1e_3 < configured_rank
        ),
        "tiny_min_sv": min_sv < abs_tol,
        "ill_conditioned": (
            condition_number is not None and condition_number > 1e6
        ),
        "tiny_adapter_ratio": (
            adapter_backbone_ratio is not None
            and adapter_backbone_ratio < 1e-4
        ),
        "large_adapter_ratio": (
            adapter_backbone_ratio is not None
            and adapter_backbone_ratio > 1.0
        ),
        "factor_imbalance": (
            factor_balance_ratio is not None
            and (factor_balance_ratio < 1e-2 or factor_balance_ratio > 1e2)
        ),
        "zero_or_near_zero_adapter": adapter_fro_norm < abs_tol,
    }
    return flags


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    # Flatten singular_values and flags into JSON strings
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["singular_values"] = json.dumps(out_row["singular_values"])
            # flags dict -> JSON string
            if "flags" in out_row and isinstance(out_row["flags"], dict):
                out_row["flags"] = json.dumps(out_row["flags"])
            writer.writerow(out_row)


def write_json(data: dict[str, Any], output_path: Path, indent: int | None) -> None:
    with open(output_path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def write_npz(
    all_singular_values: dict[str, np.ndarray], output_path: Path
) -> None:
    if not all_singular_values:
        # Write empty npz
        np.savez(output_path)
        return
    np.savez(output_path, **all_singular_values)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def build_summary(
    metadata: dict[str, Any],
    errors: list[str],
    rows: list[dict[str, Any]],
    checkpoint_path: str,
) -> str:
    buf = io.StringIO()

    buf.write("# LoRASA Adapter Diagnostic Summary\n\n")
    buf.write(f"**Checkpoint:** `{checkpoint_path}`\n\n")

    # Metadata
    buf.write("## Metadata\n\n")
    if metadata:
        for k, v in metadata.items():
            buf.write(f"- **{k}**: {v}\n")
    else:
        buf.write("_No metadata extracted._\n")
    buf.write("\n")

    # Errors / warnings
    buf.write("## Errors / Warnings\n\n")
    if errors:
        for e in errors:
            buf.write(f"- ⚠️ {e}\n")
    else:
        buf.write("_No errors or warnings._\n")
    buf.write("\n")

    # High-level counts
    num_blocks = len({r["path"] for r in rows})
    num_rows = len(rows)
    ranks_seen = sorted({r["configured_rank"] for r in rows})
    buf.write("## Overview\n\n")
    buf.write(f"- **LoRA blocks discovered:** {num_blocks}\n")
    buf.write(f"- **Adapter rows analyzed:** {num_rows}\n")
    buf.write(f"- **Configured ranks seen:** {ranks_seen}\n")
    buf.write("\n")

    # Flag counts
    buf.write("## Flag Counts\n\n")
    if rows:
        flag_names = sorted(rows[0]["flags"].keys())
        counts: dict[str, int] = {fn: 0 for fn in flag_names}
        for r in rows:
            for fn, val in r["flags"].items():
                if val:
                    counts[fn] += 1
        for fn in flag_names:
            buf.write(f"- `{fn}`: {counts[fn]} / {num_rows}\n")
    buf.write("\n")

    # Worst 10 by effective_rank / configured_rank
    buf.write("## Worst 10 by Effective-Rank / Configured-Rank Ratio\n\n")
    sorted_eff = sorted(
        rows, key=lambda r: r["effective_rank"] / max(r["configured_rank"], 1)
    )[:10]
    buf.write("| path | slot | configured_rank | effective_rank | ratio |\n")
    buf.write("|------|------|-----------------|----------------|-------|\n")
    for r in sorted_eff:
        ratio = r["effective_rank"] / max(r["configured_rank"], 1)
        buf.write(
            f"| {r['path']} | {r['slot']} | {r['configured_rank']} | "
            f"{r['effective_rank']:.4f} | {ratio:.4f} |\n"
        )
    buf.write("\n")

    # Worst 10 by condition_number
    buf.write("## Worst 10 by Condition Number\n\n")
    rows_with_cn = [r for r in rows if r["condition_number"] is not None]
    sorted_cn = sorted(rows_with_cn, key=lambda r: -r["condition_number"])[:10]
    buf.write("| path | slot | condition_number | max_sv | min_nonzero_sv |\n")
    buf.write("|------|------|------------------|--------|----------------|\n")
    for r in sorted_cn:
        buf.write(
            f"| {r['path']} | {r['slot']} | {r['condition_number']:.4e} | "
            f"{r['max_singular_value']:.4e} | {r['min_nonzero_singular_value']:.4e} |\n"
        )
    buf.write("\n")

    # Smallest adapter/backbone ratios
    buf.write("## Smallest Adapter / Backbone Ratios\n\n")
    rows_with_ratio = [r for r in rows if r["adapter_backbone_ratio"] is not None]
    sorted_small = sorted(rows_with_ratio, key=lambda r: r["adapter_backbone_ratio"])[:10]
    buf.write("| path | slot | adapter_fro_norm | backbone_fro_norm | ratio |\n")
    buf.write("|------|------|------------------|-------------------|-------|\n")
    for r in sorted_small:
        buf.write(
            f"| {r['path']} | {r['slot']} | {r['adapter_fro_norm']:.4e} | "
            f"{r['backbone_fro_norm']:.4e} | {r['adapter_backbone_ratio']:.4e} |\n"
        )
    buf.write("\n")

    # Largest adapter/backbone ratios
    buf.write("## Largest Adapter / Backbone Ratios\n\n")
    sorted_large = sorted(
        rows_with_ratio, key=lambda r: -r["adapter_backbone_ratio"]
    )[:10]
    buf.write("| path | slot | adapter_fro_norm | backbone_fro_norm | ratio |\n")
    buf.write("|------|------|------------------|-------------------|-------|\n")
    for r in sorted_large:
        buf.write(
            f"| {r['path']} | {r['slot']} | {r['adapter_fro_norm']:.4e} | "
            f"{r['backbone_fro_norm']:.4e} | {r['adapter_backbone_ratio']:.4e} |\n"
        )
    buf.write("\n")

    # Per-block aggregate stats
    buf.write("## Per-Block Aggregate Stats\n\n")
    block_groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        block_groups.setdefault(r["path"], []).append(r)
    buf.write(
        "| block | rows | mean_eff_rank | mean_cond_num | mean_adapter_norm | mean_ratio |\n"
    )
    buf.write(
        "|-------|------|---------------|---------------|-------------------|------------|\n"
    )
    for block in sorted(block_groups):
        grp = block_groups[block]
        effs = [r["effective_rank"] for r in grp]
        cns = [r["condition_number"] for r in grp if r["condition_number"] is not None]
        ad_norms = [r["adapter_fro_norm"] for r in grp]
        ratios = [
            r["adapter_backbone_ratio"]
            for r in grp
            if r["adapter_backbone_ratio"] is not None
        ]
        buf.write(
            f"| {block} | {len(grp)} | {sum(effs)/len(effs):.4f} | "
            f"{sum(cns)/max(len(cns),1):.4e} | {sum(ad_norms)/len(ad_norms):.4e} | "
            f"{sum(ratios)/max(len(ratios),1):.4e} |\n"
        )
    buf.write("\n")

    # Per-slot aggregate stats
    buf.write("## Per-Slot Aggregate Stats\n\n")
    slot_groups: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        slot_groups.setdefault(r["slot"], []).append(r)
    buf.write(
        "| slot | rows | mean_eff_rank | mean_cond_num | mean_adapter_norm | mean_ratio |\n"
    )
    buf.write(
        "|------|------|---------------|---------------|-------------------|------------|\n"
    )
    for slot in sorted(slot_groups):
        grp = slot_groups[slot]
        effs = [r["effective_rank"] for r in grp]
        cns = [r["condition_number"] for r in grp if r["condition_number"] is not None]
        ad_norms = [r["adapter_fro_norm"] for r in grp]
        ratios = [
            r["adapter_backbone_ratio"]
            for r in grp
            if r["adapter_backbone_ratio"] is not None
        ]
        buf.write(
            f"| {slot} | {len(grp)} | {sum(effs)/len(effs):.4f} | "
            f"{sum(cns)/max(len(cns),1):.4e} | {sum(ad_norms)/len(ad_norms):.4e} | "
            f"{sum(ratios)/max(len(ratios),1):.4e} |\n"
        )
    buf.write("\n")

    # Recommendation
    buf.write("## Final Recommendation\n\n")
    rec = _recommendation(rows)
    buf.write(f"{rec}\n")

    return buf.getvalue()


def _recommendation(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No adapters were analyzed."

    total = len(rows)
    collapsed = sum(1 for r in rows if r["flags"]["effective_rank_lt_half_configured"])
    tiny = sum(1 for r in rows if r["flags"]["tiny_adapter_ratio"])
    ill = sum(1 for r in rows if r["flags"]["ill_conditioned"])

    parts: list[str] = []
    if collapsed / total > 0.3:
        parts.append(
            "Many adapters appear rank-collapsed. "
            "Consider lower active rank, singular-value flooring, "
            "or rank-expanded residual ES before strict fixed-rank Riemannian ES."
        )
    if ill / total > 0.2:
        parts.append(
            "Several adapters are ill-conditioned. "
            "Consider singular-value flooring during SVD retraction and stronger held-out validation."
        )
    if tiny / total > 0.3:
        parts.append(
            "Many adapters are tiny relative to the backbone. "
            "Use norm-scaled sigma and conservative ES step sizes."
        )
    if not parts:
        parts.append(
            "Most adapters look numerically rank-healthy. "
            "Fixed-rank Riemannian LoRASA-EGGROLL appears plausible."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> None:
    """Run a tiny synthetic test without any real checkpoint."""
    print("Running self-test...")

    num_slots = 3
    rank = 4
    in_dim = 8
    out_dim = 6

    # Healthy full-rank adapter
    A_healthy = np.random.randn(num_slots, in_dim, rank)
    B_healthy = np.random.randn(num_slots, rank, out_dim)

    # Rank-collapsed adapter (last singular value near zero)
    A_collapsed = np.random.randn(num_slots, in_dim, rank)
    B_collapsed = np.random.randn(num_slots, rank, out_dim)
    # Force low-rank by zeroing a column of B
    B_collapsed[:, -1, :] = 0.0

    # Missing-kernel adapter
    A_missing = np.random.randn(num_slots, in_dim, rank)
    B_missing = np.random.randn(num_slots, rank, out_dim)

    tree = {
        "params": {
            "mlp": {
                "dense_0": {
                    "lora_a": A_healthy,
                    "lora_b": B_healthy,
                    "kernel": np.random.randn(in_dim, out_dim),
                },
                "dense_1": {
                    "lora_a": A_collapsed,
                    "lora_b": B_collapsed,
                    "kernel": np.random.randn(in_dim, out_dim),
                },
            },
            "action_out": {
                "lora_a": A_missing,
                "lora_b": B_missing,
                # no kernel
            },
        }
    }

    flat = flatten_tree(tree)
    pairs = discover_lora_pairs(flat)
    assert len(pairs) == 3, f"Expected 3 pairs, got {len(pairs)}"

    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    all_svs: dict[str, np.ndarray] = {}
    abs_tol = 1e-8
    rel_tols = [1e-3, 1e-4]

    for pair in pairs:
        prefix_str = pair["prefix_str"]
        a_arr = to_numpy_array(flat[pair["lora_a_key"]])
        b_arr = to_numpy_array(flat[pair["lora_b_key"]])
        k_arr = (
            to_numpy_array(flat[pair["kernel_key"]])
            if pair["kernel_key"] is not None
            else None
        )

        if a_arr.ndim != 3 or b_arr.ndim != 3:
            errors.append(f"Unexpected shape at {prefix_str}")
            continue
        ns, in_d, r = a_arr.shape
        _, r2, out_d = b_arr.shape
        if r != r2:
            errors.append(f"Rank mismatch at {prefix_str}: {r} vs {r2}")
            continue
        if k_arr is not None and k_arr.shape != (in_d, out_d):
            errors.append(
                f"Kernel shape mismatch at {prefix_str}: {k_arr.shape} vs ({in_d},{out_d})"
            )
            # continue anyway; kernel just becomes null for metrics
            k_arr = None

        for slot in range(ns):
            metrics = compute_slot_metrics(
                a_arr[slot], b_arr[slot], k_arr, abs_tol, rel_tols
            )
            flags = flag_metrics(metrics, abs_tol)
            row = {
                "path": prefix_str,
                "slot": slot,
                "num_slots": ns,
                "input_dim": metrics["input_dim"],
                "output_dim": metrics["output_dim"],
                "configured_rank": metrics["configured_rank"],
                "singular_values": metrics["singular_values"],
                "max_singular_value": metrics["max_singular_value"],
                "min_singular_value": metrics["min_singular_value"],
                "min_nonzero_singular_value": metrics["min_nonzero_singular_value"],
                "condition_number": metrics["condition_number"],
                "effective_rank": metrics["effective_rank"],
                "stable_rank": metrics["stable_rank"],
                "adapter_fro_norm": metrics["adapter_fro_norm"],
                "backbone_fro_norm": metrics["backbone_fro_norm"],
                "adapter_backbone_ratio": metrics["adapter_backbone_ratio"],
                "lora_a_fro_norm": metrics["lora_a_fro_norm"],
                "lora_b_fro_norm": metrics["lora_b_fro_norm"],
                "factor_balance_ratio": metrics["factor_balance_ratio"],
                "flags": flags,
            }
            for k, v in metrics.items():
                if k.startswith("numerical_rank"):
                    row[k] = v
            rows.append(row)
            safe_key = f"{prefix_str.replace('/', '__')}__slot_{slot}"
            all_svs[safe_key] = np.array(metrics["singular_values"])

    assert len(rows) == 3 * num_slots

    # Sanity checks on metrics
    healthy_rows = [r for r in rows if "dense_0" in r["path"]]
    collapsed_rows = [r for r in rows if "dense_1" in r["path"]]
    missing_rows = [r for r in rows if "action_out" in r["path"]]

    for hr in healthy_rows:
        assert hr["backbone_fro_norm"] is not None
        assert hr["adapter_backbone_ratio"] is not None
        assert not hr["flags"]["zero_or_near_zero_adapter"]

    for cr in collapsed_rows:
        assert cr["flags"]["effective_rank_lt_half_configured"] or cr["flags"]["numerical_rank_rel_1e_3_lt_configured"]

    for mr in missing_rows:
        assert mr["backbone_fro_norm"] is None
        assert mr["adapter_backbone_ratio"] is None

    print("Self-test passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spectral diagnostic for LoRASA actor adapters"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint pickle")
    parser.add_argument("--output_dir", type=str, default="diagnostics/lorasa_spectrum")
    parser.add_argument("--abs_tol", type=float, default=1e-8)
    parser.add_argument(
        "--rel_tols",
        type=str,
        default="1e-3,1e-4",
        help="Comma-separated relative tolerances for numerical rank",
    )
    parser.add_argument("--json_indent", type=int, default=2)
    parser.add_argument("--self_test", action="store_true", help="Run internal self-test")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --self_test is used")

    rel_tols = [float(x.strip()) for x in args.rel_tols.split(",") if x.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    ckpt = load_checkpoint(args.checkpoint)
    actor_params, metadata = extract_actor_params(ckpt)

    # Flatten
    flat = flatten_tree(actor_params)

    # Discover
    pairs = discover_lora_pairs(flat)

    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    all_svs: dict[str, np.ndarray] = {}

    for pair in pairs:
        prefix_str = pair["prefix_str"]
        a_arr = to_numpy_array(flat[pair["lora_a_key"]])
        b_arr = to_numpy_array(flat[pair["lora_b_key"]])
        k_arr = (
            to_numpy_array(flat[pair["kernel_key"]])
            if pair["kernel_key"] is not None
            else None
        )

        if a_arr.ndim != 3 or b_arr.ndim != 3:
            errors.append(
                f"Unexpected shape at {prefix_str}: A {a_arr.shape}, B {b_arr.shape}"
            )
            continue
        num_slots, in_dim, rank = a_arr.shape
        _, rank2, out_dim = b_arr.shape
        if rank != rank2:
            errors.append(
                f"Rank mismatch at {prefix_str}: {rank} vs {rank2}"
            )
            continue
        if k_arr is not None and k_arr.shape != (in_dim, out_dim):
            errors.append(
                f"Kernel shape mismatch at {prefix_str}: {k_arr.shape} vs ({in_dim},{out_dim})"
            )
            k_arr = None  # treat as missing

        for slot in range(num_slots):
            try:
                metrics = compute_slot_metrics(
                    a_arr[slot], b_arr[slot], k_arr, args.abs_tol, rel_tols
                )
            except Exception as exc:
                errors.append(f"Metrics failure at {prefix_str} slot {slot}: {exc}")
                continue
            flags = flag_metrics(metrics, args.abs_tol)
            row: dict[str, Any] = {
                "path": prefix_str,
                "slot": slot,
                "num_slots": num_slots,
                "input_dim": metrics["input_dim"],
                "output_dim": metrics["output_dim"],
                "configured_rank": metrics["configured_rank"],
                "singular_values": metrics["singular_values"],
                "max_singular_value": metrics["max_singular_value"],
                "min_singular_value": metrics["min_singular_value"],
                "min_nonzero_singular_value": metrics["min_nonzero_singular_value"],
                "condition_number": metrics["condition_number"],
                "effective_rank": metrics["effective_rank"],
                "stable_rank": metrics["stable_rank"],
                "adapter_fro_norm": metrics["adapter_fro_norm"],
                "backbone_fro_norm": metrics["backbone_fro_norm"],
                "adapter_backbone_ratio": metrics["adapter_backbone_ratio"],
                "lora_a_fro_norm": metrics["lora_a_fro_norm"],
                "lora_b_fro_norm": metrics["lora_b_fro_norm"],
                "factor_balance_ratio": metrics["factor_balance_ratio"],
                "flags": flags,
            }
            for k, v in metrics.items():
                if k.startswith("numerical_rank"):
                    row[k] = v
            rows.append(row)
            safe_key = f"{prefix_str.replace('/', '__')}__slot_{slot}"
            all_svs[safe_key] = np.array(metrics["singular_values"])

    # Expand flags into individual columns for CSV convenience
    csv_rows: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        flags = d.pop("flags")
        d["flags_json"] = json.dumps(flags)
        for fn, fv in flags.items():
            d[fn] = fv
        csv_rows.append(d)

    # Write outputs
    write_csv(csv_rows, output_dir / "per_adapter.csv")
    write_json(
        {
            "metadata": metadata,
            "errors": errors,
            "rows": rows,
        },
        output_dir / "per_adapter.json",
        args.json_indent,
    )
    write_npz(all_svs, output_dir / "singular_values.npz")
    summary_md = build_summary(metadata, errors, rows, args.checkpoint)
    with open(output_dir / "summary.md", "w") as f:
        f.write(summary_md)

    print(f"Wrote diagnostics to {output_dir}")
    print(f"  Blocks: {len({r['path'] for r in rows})}, Rows: {len(rows)}")
    if errors:
        print(f"  Errors: {len(errors)}")


if __name__ == "__main__":
    main()
