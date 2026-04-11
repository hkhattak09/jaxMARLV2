"""Stage 5 SAAL runner.

Runs the Stage 5 SAAL validation on SMAX 3m:
- Sanity run: seed 200
- Main comparison: seeds 201, 202, 203

The script writes per-run metrics and JSON summaries to:
  analysis_results_inc/stage5/

Execution is expected on Colab. Local use is for static checks only.
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
    "CTM_ITERATIONS": 1,
    "CTM_N_SYNCH_OUT": 32,
    "CTM_MEMORY_LENGTH": 5,
    "CTM_DEEP_NLMS": True,
    "CTM_NLM_HIDDEN_DIM": 2,
    "CTM_DO_LAYERNORM_NLM": False,
    "CTM_USE_SYNC": True,
    "CTM_NEURON_SELECT": "first-last",
    "CTM_ACTOR_HEAD_DIM": 64,
    "NUM_CONSENSUS_ITERATIONS": 0,
    "INC_ENABLED": False,
    "INC_POOLING": "mean",
    "INC_CONSENSUS_DROPOUT": 0.0,
    "INC_DEBUG_SHAPES": False,
    "INC_USE_ALIVE_MASK_FROM_DONES": True,
    "CTM_ITER_DROPOUT": 0.0,
    "INC_FORCE_ZERO_CONSENSUS": False,
    "ALIGN_ENABLED": True,
    "ALIGN_ALPHA": 0.05,
    "ALIGN_BETA": 0.025,
    "ENV_NAME": "HeuristicEnemySMAX",
    "MAP_NAME": "3m",
    "SEED": 0,
    "ENV_KWARGS": {
        "see_enemy_actions": True,
        "walls_cause_death": True,
        "attack_mode": "closest",
    },
    "ANNEAL_LR": True,
}


def _summary_from_metric(metric) -> dict:
    out = {}
    loss = metric.get("loss")
    if loss is not None:
        for key in ("pair_cos_ff", "pair_cos_nff", "pair_cos_all", "L_align", "ff_frac"):
            if key in loss:
                series = loss[key]
                out[f"{key}_final20"] = float(jnp.mean(series[-20:]))

    wr_key = None
    for key in metric.keys():
        low = key.lower()
        if "won" in low or "win" in low:
            wr_key = key
            break

    if wr_key is not None:
        wr = metric[wr_key]
        while wr.ndim > 1:
            wr = wr.mean(axis=-1)
        wr_curve = [float(x) for x in wr.tolist()]
        out["win_rate_curve"] = wr_curve
        if wr_curve:
            out["final_win_rate"] = float(sum(wr_curve[-20:]) / max(1, len(wr_curve[-20:])))
            out["frac_win_rate_ge_0_8"] = float(sum(1.0 for v in wr_curve if v >= 0.8) / len(wr_curve))
            rolling_20 = [
                sum(wr_curve[max(0, i - 20):i]) / min(i, 20)
                for i in range(1, len(wr_curve) + 1)
            ]
            hits = [i for i, v in enumerate(rolling_20) if v >= 0.8]
            out["first_update_rolling20_wr_ge_0_8"] = hits[0] if hits else None

    return out


def run_one(seed: int, out_dir: str) -> dict:
    config = dict(BASE_CONFIG)
    config["SEED"] = int(seed)

    os.makedirs(out_dir, exist_ok=True)
    tag = f"SAAL_a05_b025__seed{seed}"
    print(f"[stage5] launching {tag}")

    rng = jax.random.PRNGKey(seed)
    train_jit = jax.jit(make_train(config))

    start = time.time()
    out = train_jit(rng)
    jax.block_until_ready(out["metric"])
    elapsed = time.time() - start
    print(f"[stage5]   {tag} finished in {elapsed / 60.0:.1f} min")

    metric = jax.tree.map(lambda x: jnp.asarray(x), out["metric"])
    summary = _summary_from_metric(metric)
    summary.update(
        {
            "cell": "SAAL_a05_b025",
            "seed": seed,
            "wall_clock_seconds": elapsed,
            "align_alpha": config["ALIGN_ALPHA"],
            "align_beta": config["ALIGN_BETA"],
        }
    )

    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, f"{tag}_metric.pkl"), "wb") as f:
        pickle.dump(jax.tree.map(lambda x: x.copy(), metric), f)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 5 SAAL runner")
    parser.add_argument(
        "--mode",
        choices=["sanity", "main", "all"],
        default="all",
        help="sanity=seed200, main=201/202/203, all=both",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_REPO_ROOT, "analysis_results_inc", "stage5"),
        help="Output directory for summaries and metrics.",
    )
    args = parser.parse_args()

    if args.mode == "sanity":
        seeds = [200]
    elif args.mode == "main":
        seeds = [201, 202, 203]
    else:
        seeds = [200, 201, 202, 203]

    os.makedirs(args.out_dir, exist_ok=True)
    all_summaries = []
    for seed in seeds:
        all_summaries.append(run_one(seed, args.out_dir))

    with open(os.path.join(args.out_dir, "stage5_runs.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"[stage5] wrote rollup to {os.path.join(args.out_dir, 'stage5_runs.json')}")


if __name__ == "__main__":
    main()
