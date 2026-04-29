#!/usr/bin/env python3
"""Correctness-first Riemannian LoRASA-EGGROLL prototype.

This is intentionally not the final hyperscale implementation.  It evaluates
one antithetic population sequentially with the deterministic SMAX evaluator,
then applies the adapter-only Riemannian tangent update from
``smax_ctm/lorasa_eggroll.py``.

The goal of this script is to validate geometry + checkpoint compatibility +
fitness plumbing before adding a JAX population-axis implementation.
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
from eval_smax import (  # noqa: E402
    build_env,
    make_eval_fn,
    validate_actor_params,
    validate_checkpoint,
)
from mappo_t import LoRASAActorTrans, ScannedRNN  # noqa: E402


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


def _build_eval(
    ckpt: Mapping[str, Any],
    checkpoint_path: str,
    map_name: str,
    num_envs: int,
    num_loops: int,
    max_steps: Optional[int],
):
    """Build actor network and deterministic evaluation function."""

    validate_checkpoint(dict(ckpt), checkpoint_path)
    if ckpt["model_type"] != "mappo_t_lorasa":
        raise ValueError(
            f"Riemannian LoRASA-EGGROLL requires model_type='mappo_t_lorasa', "
            f"got {ckpt['model_type']!r}"
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

    eval_fn = make_eval_fn(
        env,
        actor_network,
        actor_hidden_dim,
        eval_steps,
        num_envs,
        num_loops,
        is_lorasa=True,
    )
    return eval_fn, config, eval_steps


def _evaluate(eval_fn, actor_params: Any, seed: int) -> Dict[str, Any]:
    """Run deterministic evaluator and return JSON-friendly win statistics."""

    rng = jax.random.PRNGKey(int(seed))
    start = time.time()
    wins = np.asarray(jax.device_get(eval_fn(rng, actor_params)), dtype=np.float64)
    elapsed = time.time() - start

    flat = wins.reshape(-1)
    episodes = int(flat.size)
    mean_wr = float(flat.mean()) if episodes else 0.0
    std = float(flat.std(ddof=1)) if episodes > 1 else 0.0
    sem = std / math.sqrt(episodes) if episodes > 1 else 0.0
    return {
        "episodes": episodes,
        "mean_wr": mean_wr,
        "std": std,
        "sem": sem,
        "elapsed_sec": elapsed,
        "loop_win_rates": [float(x) for x in wins.mean(axis=1)],
    }


def _metrics_to_json(metrics) -> List[Dict[str, Any]]:
    return [asdict(metric) for metric in metrics]


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


def _make_checkpoint(
    source_ckpt: Mapping[str, Any],
    actor_params: Any,
    epoch: int,
    args: argparse.Namespace,
    latest_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    ckpt = copy.deepcopy(dict(source_ckpt))
    ckpt["actor_params"] = actor_params
    ckpt["checkpoint_kind"] = "lorasa_eggroll"
    ckpt["lorasa_eggroll"] = {
        "epoch": int(epoch),
        "source_checkpoint": str(args.checkpoint),
        "active_slots": list(args.active_slots),
        "target_rank": int(args.target_rank),
        "noise_rank": int(args.noise_rank),
        "sigma": float(args.sigma),
        "eta": float(args.eta),
        "num_directions": int(args.num_directions),
        "fitness": "deterministic_win_rate",
        "latest_stats": dict(latest_stats),
    }
    return ckpt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correctness-first adapter-only Riemannian LoRASA-EGGROLL."
    )
    parser.add_argument("--checkpoint", required=True, help="Schedule A LoRASA checkpoint")
    parser.add_argument("--output_dir", default="lorasa_eggroll_runs")
    parser.add_argument("--map_name", default=None)
    parser.add_argument("--max_steps", type=int, default=None)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_directions", type=int, default=2)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_loops", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--eta", type=float, default=0.02)
    parser.add_argument("--target_rank", type=int, default=re.DEFAULT_TARGET_RANK)
    parser.add_argument("--noise_rank", type=int, default=re.DEFAULT_NOISE_RANK)
    parser.add_argument("--active_slots", default="2,3,6", type=_parse_slots)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=1000)
    parser.add_argument("--eval_seed_stride", type=int, default=100)
    parser.add_argument(
        "--raw_score_weights",
        action="store_true",
        help="Use raw score differences instead of centered-rank utilities",
    )
    parser.add_argument(
        "--no_relative_scale",
        action="store_true",
        help="Do not scale unit tangent directions by adapter Frobenius norm",
    )
    parser.add_argument("--singular_floor", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.num_epochs < 1:
        raise ValueError("--num_epochs must be >= 1")
    if args.num_directions < 1:
        raise ValueError("--num_directions must be >= 1")
    if args.num_envs < 1 or args.num_loops < 1:
        raise ValueError("--num_envs and --num_loops must be >= 1")

    source_ckpt = _load_checkpoint(args.checkpoint)
    map_name = args.map_name or source_ckpt.get("config", {}).get("MAP_NAME", "protoss_10_vs_10")

    run_id = datetime.now().strftime("lorasa_eggroll_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    args_path = run_dir / "args.json"
    args_json = vars(args).copy()
    args_json["active_slots"] = list(args.active_slots)
    args_json["map_name"] = map_name
    args_path.write_text(json.dumps(args_json, indent=2, sort_keys=True) + "\n")

    print(f"Run directory: {run_dir}")
    print(f"Map: {map_name}")
    print("Building deterministic evaluator...")
    eval_fn, _, eval_steps = _build_eval(
        source_ckpt,
        args.checkpoint,
        map_name,
        args.num_envs,
        args.num_loops,
        args.max_steps,
    )
    print(f"Eval steps: {eval_steps}")

    actor_params = copy.deepcopy(source_ckpt["actor_params"])
    history_jsonl = run_dir / "history.jsonl"
    candidate_csv_rows: List[Dict[str, Any]] = []
    latest_stats: Dict[str, Any] = {}

    for epoch in range(args.num_epochs):
        print(f"\n=== ES epoch {epoch} ===")
        common_eval_seed = args.eval_seed + epoch * args.eval_seed_stride
        raw_scores: List[float] = []
        candidate_records: List[Dict[str, Any]] = []

        for direction_id in range(args.num_directions):
            for sign_label, sign in (("plus", 1), ("minus", -1)):
                candidate_params, candidate_metrics = re.make_candidate_actor_params(
                    actor_params,
                    direction_id=direction_id,
                    sign=sign,
                    sigma=args.sigma,
                    epoch=epoch,
                    base_seed=args.noise_seed,
                    active_slots=args.active_slots,
                    target_rank=args.target_rank,
                    noise_rank=args.noise_rank,
                    relative_scale=not args.no_relative_scale,
                    singular_floor=args.singular_floor,
                )
                summary = _evaluate(eval_fn, candidate_params, seed=common_eval_seed)
                score = float(summary["mean_wr"])
                raw_scores.append(score)

                record = {
                    "epoch": epoch,
                    "direction_id": direction_id,
                    "sign": sign_label,
                    "score": score,
                    "episodes": summary["episodes"],
                    "std": summary["std"],
                    "sem": summary["sem"],
                    "elapsed_sec": summary["elapsed_sec"],
                    "num_adapter_metrics": len(candidate_metrics),
                }
                candidate_records.append(record)
                candidate_csv_rows.append(record)
                print(
                    f"dir={direction_id:03d} {sign_label:<5} "
                    f"wr={score:.4f} sem={summary['sem']:.4f} "
                    f"time={summary['elapsed_sec']:.1f}s"
                )

        direction_weights = re.antithetic_direction_weights(
            raw_scores,
            use_centered_ranks=not args.raw_score_weights,
        )
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
            singular_floor=args.singular_floor,
        )

        post_update_eval = None
        if args.eval_every > 0 and ((epoch + 1) % args.eval_every == 0):
            post_seed = common_eval_seed + args.eval_seed_stride // 2
            post_update_eval = _evaluate(eval_fn, actor_params, seed=post_seed)
            print(
                f"post-update wr={post_update_eval['mean_wr']:.4f} "
                f"sem={post_update_eval['sem']:.4f}"
            )

        latest_stats = {
            "epoch": epoch,
            "raw_scores": raw_scores,
            "direction_weights": {str(k): v for k, v in direction_weights.items()},
            "candidate_records": candidate_records,
            "post_update_eval": post_update_eval,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
