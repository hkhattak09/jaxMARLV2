"""Stage 2.1 disambiguation runner.

Launches cells D and E from the Stage 2.1 plan on SMAX 3m, three seeds each.
Cells A/B/C are assumed to already be logged from the Stage 2 run — this
script deliberately does NOT re-run them.

Cells:
  D — INC_ENABLED=False, CTM_ITER_DROPOUT=0.25
      ("stochastic iteration loop" control — no teammate pooling at all,
       but dropout is applied to activated_state_trace between iters.)

  E — INC_ENABLED=True,  INC_CONSENSUS_DROPOUT=0.25,
      INC_FORCE_ZERO_CONSENSUS=True
      ("zeroed pool, same dropout pattern as cell C" control — identical
       RNG consumption and dropout mask site as cell C, but teammate
       information is stripped to zeros before dropout is applied.)

All other hyperparameters match the Stage 2 INC run: iter=3, mean pooling,
3M total timesteps, 3m map.

Results are written to analysis_results_inc/stage2_1/ as pickled runner-state
plus a small JSON summary per (cell, seed). The Stage 2.1 analysis step
(learning-curve + final WR bar chart) reads from this directory.
"""

import argparse
import json
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SMAX_CTM_DIR = os.path.join(_REPO_ROOT, "smax_ctm")
if _SMAX_CTM_DIR not in sys.path:
    sys.path.insert(0, _SMAX_CTM_DIR)

from smax_ctm.train_mappo_ctm import make_train  # noqa: E402


# --- Base config ------------------------------------------------------------
# Matches the Stage 2 INC run: iter=3, mean pooling, 3m map, 3M timesteps.
# Only the disambiguation flags differ between cells D and E.

BASE_CONFIG = {
    "LR": 0.002,
    "NUM_ENVS": 128,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": int(3e6),
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "SCALE_CLIP_EPS": False,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.25,
    "ACTIVATION": "relu",
    "OBS_WITH_AGENT_ID": True,
    "CTM_D_MODEL": 128,
    "CTM_D_INPUT": 64,
    "CTM_ITERATIONS": 3,
    "CTM_N_SYNCH_OUT": 32,
    "CTM_MEMORY_LENGTH": 5,
    "CTM_DEEP_NLMS": True,
    "CTM_NLM_HIDDEN_DIM": 2,
    "CTM_DO_LAYERNORM_NLM": False,
    "CTM_USE_SYNC": True,
    "CTM_NEURON_SELECT": "first-last",
    "CTM_ACTOR_HEAD_DIM": 64,
    "NUM_CONSENSUS_ITERATIONS": 0,
    # Stage 2 INC defaults:
    "INC_ENABLED": True,
    "INC_POOLING": "mean",
    "INC_CONSENSUS_DROPOUT": 0.0,
    "INC_DEBUG_SHAPES": False,
    "INC_USE_ALIVE_MASK_FROM_DONES": True,
    # Stage 2.1 controls (overridden per-cell below):
    "CTM_ITER_DROPOUT": 0.0,
    "INC_FORCE_ZERO_CONSENSUS": False,
    "ENV_NAME": "HeuristicEnemySMAX",
    "MAP_NAME": "3m",
    "SEED": 0,  # overwritten per seed
    "ENV_KWARGS": {
        "see_enemy_actions": True,
        "walls_cause_death": True,
        "attack_mode": "closest",
    },
    "ANNEAL_LR": True,
}


# --- Cell definitions -------------------------------------------------------
# Seeds are deliberately NOT the seeds from the Stage 2 run (0/1/2) to avoid
# biased overlap. Pick 103/104/105.
CELL_SEEDS = [103, 104, 105]

CELLS = {
    "D_iter_dropout_no_inc": {
        "INC_ENABLED": False,
        "INC_CONSENSUS_DROPOUT": 0.0,
        "CTM_ITER_DROPOUT": 0.25,
        "INC_FORCE_ZERO_CONSENSUS": False,
    },
    "E_zero_consensus_same_dropout": {
        "INC_ENABLED": True,
        "INC_CONSENSUS_DROPOUT": 0.25,
        "CTM_ITER_DROPOUT": 0.0,
        "INC_FORCE_ZERO_CONSENSUS": True,
    },
}


def _summary_from_metric(metric) -> dict:
    """Extract a tiny JSON-serialisable summary from the training metric tree.

    Keeps only a per-update win-rate curve and final-WR mean so the on-disk
    footprint is cheap to load in the analysis step. Full traces are also
    pickled separately for anyone who needs them.
    """
    out = {}
    try:
        # SMAX log wrapper records "returned_won_episode" per env step;
        # we reduce over steps/envs/agents to a per-update mean.
        wr_key = None
        for k in metric.keys():
            if "won" in k.lower() or "win" in k.lower():
                wr_key = k
                break
        if wr_key is not None:
            wr = metric[wr_key]
            # Shape may be (num_updates, num_steps, num_envs, num_agents).
            while wr.ndim > 1:
                wr = wr.mean(axis=-1)
            out["win_rate_curve"] = [float(x) for x in wr.tolist()]
            if len(out["win_rate_curve"]) > 0:
                out["final_win_rate"] = float(
                    sum(out["win_rate_curve"][-20:]) / max(1, len(out["win_rate_curve"][-20:]))
                )
    except Exception as exc:  # noqa: BLE001
        out["summary_error"] = repr(exc)
    return out


def run_cell(cell_name: str, cell_overrides: dict, seed: int, out_dir: str) -> dict:
    config = dict(BASE_CONFIG)
    config.update(cell_overrides)
    config["SEED"] = seed

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{cell_name}__seed{seed}"
    print(f"[stage2_1] launching {tag}")
    print(f"[stage2_1]   overrides: {cell_overrides}")

    rng = jax.random.PRNGKey(seed)
    train_jit = jax.jit(make_train(config))

    start = time.time()
    out = train_jit(rng)
    # Block on host to get real wall-clock.
    jax.block_until_ready(out["metric"])
    elapsed = time.time() - start
    print(f"[stage2_1]   {tag} finished in {elapsed / 60.0:.1f} min")

    metric = jax.tree.map(lambda x: jnp.asarray(x), out["metric"])
    summary = _summary_from_metric(metric)
    summary["cell"] = cell_name
    summary["seed"] = seed
    summary["wall_clock_seconds"] = elapsed
    summary["overrides"] = cell_overrides

    summary_path = os.path.join(out_dir, f"{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    metric_path = os.path.join(out_dir, f"{tag}_metric.pkl")
    with open(metric_path, "wb") as f:
        pickle.dump(jax.tree.map(lambda x: x.copy(), metric), f)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 2.1 disambiguation runner.")
    parser.add_argument(
        "--cells",
        nargs="+",
        default=list(CELLS.keys()),
        help="Which cells to run (subset of the Stage 2.1 plan's new cells).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=CELL_SEEDS,
        help="Override seed list (default: 103 104 105, non-overlapping with Stage 2).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_REPO_ROOT, "analysis_results_inc", "stage2_1"),
        help="Where to write per-run summaries and pickled metrics.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_summaries = []
    for cell_name in args.cells:
        if cell_name not in CELLS:
            raise KeyError(f"Unknown cell {cell_name}. Known: {list(CELLS)}")
        for seed in args.seeds:
            summary = run_cell(cell_name, CELLS[cell_name], seed, args.out_dir)
            all_summaries.append(summary)

    roll_up_path = os.path.join(args.out_dir, "stage2_1_runs.json")
    with open(roll_up_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[stage2_1] wrote rollup to {roll_up_path}")


if __name__ == "__main__":
    main()
