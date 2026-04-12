"""Baseline re-establishment runner (no SAAL).

Runs two matched cells on SMAX 3m with identical seeds and budget:
  1) CTM vanilla, iter=3
  2) GRU baseline (same PPO/env budget)

Execution order: for each seed, CTM runs first then GRU, then a per-seed
comparison plot is saved to the current working directory.

Design intent:
- Same seeds across both cells.
- Same training budget and environment settings.
- No SAAL in CTM cell (ALIGN_ENABLED=False, alpha=beta=0).
- One launcher with terminal tables only (no JSON/PKL logs).

Execution is expected on Colab. Local use is for static checks only.
"""

import argparse
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SMAX_CTM_DIR = os.path.join(_REPO_ROOT, "smax_ctm")
if _SMAX_CTM_DIR not in sys.path:
    sys.path.insert(0, _SMAX_CTM_DIR)

from smax_ctm.train_mappo_ctm import make_train as make_train_ctm  # noqa: E402
from smax_ctm.train_mappo_gru import make_train as make_train_gru  # noqa: E402


BASE_CONFIG = {
    "LR": 0.002,
    "NUM_ENVS": 256,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": int(10e6),
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
    "CTM_MEMORY_LENGTH": 10,
    "CTM_DEEP_NLMS": True,
    "CTM_NLM_HIDDEN_DIM": 8,
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
    "ALIGN_ENABLED": False,
    "ALIGN_ALPHA": 0.0,
    "ALIGN_BETA": 0.0,
    "ENV_NAME": "HeuristicEnemySMAX",
    "MAP_NAME": "smacv2_10_units",
    "SEED": 0,
    "ENV_KWARGS": {
        "see_enemy_actions": True,
        "walls_cause_death": True,
        "attack_mode": "closest",
    },
    "ANNEAL_LR": True,
}

# Ordered: CTM first, then GRU.
CELLS = [
    (
        "ctm_iter3_vanilla",
        {
            "label": "CTM 3 iter",
            "trainer": "ctm",
            "overrides": {
                "CTM_ITERATIONS": 3,
                "INC_ENABLED": False,
                "ALIGN_ENABLED": False,
                "ALIGN_ALPHA": 0.0,
                "ALIGN_BETA": 0.0,
            },
        },
    ),
    (
        "gru_same_budget",
        {
            "label": "GRU",
            "trainer": "gru",
            "overrides": {},
        },
    ),
]
DEFAULT_SEEDS = [201, 202, 203]
CELLS_DICT = {name: defn for name, defn in CELLS}

THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 9)]


def _count_params(tree) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    return int(sum(np.asarray(leaf).size for leaf in leaves))


def _final20_mean(curve: np.ndarray) -> float:
    if curve.size == 0:
        return float("nan")
    return float(np.mean(curve[-20:]))


def _safe_curve_mean(curve: np.ndarray) -> float:
    if curve.size == 0:
        return float("nan")
    return float(np.mean(curve))


def _find_first_reach(curve: np.ndarray, thr: float):
    idx = np.where(curve >= thr)[0]
    if idx.size == 0:
        return "never"
    return str(int(idx[0]))


def _frac_reach(curve: np.ndarray, thr: float) -> float:
    if curve.size == 0:
        return float("nan")
    return float(np.mean(curve >= thr))


def _curve_slope(curve: np.ndarray, start_frac: float, end_frac: float) -> float:
    if curve.size < 2:
        return float("nan")
    start = int((curve.size - 1) * start_frac)
    end = int((curve.size - 1) * end_frac)
    if end <= start:
        return float("nan")
    return float((curve[end] - curve[start]) / (end - start))


def _extract_episode_curve(metric: dict, value_key: str) -> np.ndarray:
    if "returned_episode" not in metric or value_key not in metric:
        return np.array([], dtype=np.float64)

    mask = np.asarray(metric["returned_episode"])
    value = np.asarray(metric[value_key])

    # Expected shape after scan: (updates, steps, envs, agents).
    if mask.ndim == 4:
        mask = mask[..., 0]
    if value.ndim == 4:
        value = value[..., 0]

    if mask.ndim != 3 or value.ndim != 3:
        return np.array([], dtype=np.float64)

    mask = mask.astype(np.float64)
    value = value.astype(np.float64)
    num = np.sum(value * mask, axis=(1, 2))
    den = np.sum(mask, axis=(1, 2))
    curve = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return curve


def _extract_loss_curve(metric: dict, key: str) -> np.ndarray:
    loss = metric.get("loss")
    if not isinstance(loss, dict) or key not in loss:
        return np.array([], dtype=np.float64)
    arr = np.asarray(loss[key], dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _fmt_float(v: float, digits: int = 3) -> str:
    if np.isnan(v):
        return "n/a"
    return f"{v:.{digits}f}"


def _format_table(headers, rows) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(parts):
        return " | ".join(p.ljust(widths[i]) for i, p in enumerate(parts))

    sep = "-+-".join("-" * w for w in widths)
    lines = [_line(headers), sep]
    lines.extend(_line(r) for r in rows)
    return "\n".join(lines)


def _summarize_run(out: dict, trainer: str, seed: int, elapsed_sec: float) -> dict:
    metric = jax.tree.map(lambda x: jnp.asarray(x), out["metric"])

    runner_state = out["runner_state"][0]
    train_states = runner_state[0]
    actor_params = train_states[0].params
    critic_params = train_states[1].params

    actor_param_count = _count_params(actor_params)
    critic_param_count = _count_params(critic_params)
    total_param_count = actor_param_count + critic_param_count

    win_curve = _extract_episode_curve(metric, "returned_won_episode")
    ret_curve = _extract_episode_curve(metric, "returned_episode_returns")
    ent_curve = _extract_loss_curve(metric, "entropy")
    kl_curve = _extract_loss_curve(metric, "approx_kl")
    tot_loss_curve = _extract_loss_curve(metric, "total_loss")
    actor_grad_curve = _extract_loss_curve(metric, "actor_grad_norm")
    critic_grad_curve = _extract_loss_curve(metric, "critic_grad_norm")

    thr_hits = {f"hit@{t:.1f}": _find_first_reach(win_curve, t) for t in THRESHOLDS}
    thr_freq = {f"freq@{t:.1f}": _frac_reach(win_curve, t) for t in THRESHOLDS}

    return {
        "seed": seed,
        "trainer": trainer,
        "wall_min": elapsed_sec / 60.0,
        "actor_params": actor_param_count,
        "critic_params": critic_param_count,
        "total_params": total_param_count,
        "final_wr": _final20_mean(win_curve),
        "peak_wr": float(np.max(win_curve)) if win_curve.size else float("nan"),
        "auc_wr": _safe_curve_mean(win_curve),
        "slope_e": _curve_slope(win_curve, 0.00, 0.33),
        "slope_m": _curve_slope(win_curve, 0.33, 0.66),
        "slope_l": _curve_slope(win_curve, 0.66, 1.00),
        "final_ret": _final20_mean(ret_curve),
        "peak_ret": float(np.max(ret_curve)) if ret_curve.size else float("nan"),
        "ent_f20": _final20_mean(ent_curve),
        "kl_f20": _final20_mean(kl_curve),
        "totloss_f20": _final20_mean(tot_loss_curve),
        "agrad_f20": _final20_mean(actor_grad_curve),
        "cgrad_f20": _final20_mean(critic_grad_curve),
        # Raw curves stored for per-seed plotting.
        "win_curve": win_curve,
        "ret_curve": ret_curve,
        **thr_hits,
        **thr_freq,
    }


def _print_config_table(config_label: str, rows: list):
    if not rows:
        return

    param_actor = rows[0]["actor_params"]
    param_critic = rows[0]["critic_params"]
    param_total = rows[0]["total_params"]

    print("\n" + "=" * 120)
    print(f"{config_label}")
    print(
        "Params: "
        f"actor={param_actor:,}, critic={param_critic:,}, total={param_total:,}"
    )

    headers = [
        "Seed", "FinalWR", "PeakWR", "AUC_WR", "SlopeE", "SlopeM", "SlopeL",
        "FinalRet", "PeakRet", "Ent(f20)", "KL(f20)", "TotLoss(f20)",
        "GradA(f20)", "GradC(f20)", "WallMin",
    ]
    for t in THRESHOLDS:
        headers.append(f"Hit@{t:.1f}")
    for t in THRESHOLDS:
        headers.append(f"Freq@{t:.1f}")

    table_rows = []
    for r in rows:
        row = [
            str(r["seed"]),
            _fmt_float(r["final_wr"]),
            _fmt_float(r["peak_wr"]),
            _fmt_float(r["auc_wr"]),
            _fmt_float(r["slope_e"], digits=4),
            _fmt_float(r["slope_m"], digits=4),
            _fmt_float(r["slope_l"], digits=4),
            _fmt_float(r["final_ret"]),
            _fmt_float(r["peak_ret"]),
            _fmt_float(r["ent_f20"]),
            _fmt_float(r["kl_f20"], digits=4),
            _fmt_float(r["totloss_f20"], digits=4),
            _fmt_float(r["agrad_f20"], digits=4),
            _fmt_float(r["cgrad_f20"], digits=4),
            _fmt_float(r["wall_min"], digits=2),
        ]
        for t in THRESHOLDS:
            row.append(r[f"hit@{t:.1f}"])
        for t in THRESHOLDS:
            row.append(_fmt_float(r[f"freq@{t:.1f}"], digits=3))
        table_rows.append(row)

    print(_format_table(headers, table_rows))


def _print_overall_summary(by_cell: dict):
    print("\n" + "=" * 120)
    print("Overall means (across seeds)")
    headers = ["Config", "Mean FinalWR", "Std FinalWR", "Mean FinalRet", "Mean Ent(f20)", "Mean WallMin"]
    rows = []
    for label, runs in by_cell.items():
        final_wrs = np.array([r["final_wr"] for r in runs], dtype=np.float64)
        final_rets = np.array([r["final_ret"] for r in runs], dtype=np.float64)
        ents = np.array([r["ent_f20"] for r in runs], dtype=np.float64)
        wall = np.array([r["wall_min"] for r in runs], dtype=np.float64)
        rows.append([
            label,
            _fmt_float(float(np.nanmean(final_wrs))),
            _fmt_float(float(np.nanstd(final_wrs))),
            _fmt_float(float(np.nanmean(final_rets))),
            _fmt_float(float(np.nanmean(ents))),
            _fmt_float(float(np.nanmean(wall)), digits=2),
        ])
    print(_format_table(headers, rows))


def _print_gpu_memory(label: str):
    """Print actual GPU memory usage and a breakdown after training completes."""
    GiB = 1 << 30
    MiB = 1 << 20

    print("\n" + "=" * 80)
    print(f"GPU memory report  [{label}]")
    print("=" * 80)

    # --- JAX device memory stats ---
    for i, dev in enumerate(jax.devices()):
        stats = dev.memory_stats()
        if stats is None:
            print(f"  device[{i}] ({dev.device_kind}): memory_stats() not available")
            continue

        in_use     = stats.get("bytes_in_use",      stats.get("bytes_used",       0))
        peak       = stats.get("peak_bytes_in_use",  stats.get("peak_bytes_used",   0))
        limit      = stats.get("bytes_limit",        stats.get("memory_limit",      0))
        reserved   = stats.get("bytes_reservable",   stats.get("bytes_reserved",    0))
        fragmented = stats.get("fragmentation",      None)

        print(f"\n  [JAX] device[{i}]  {dev.device_kind}  platform={dev.platform}")
        print(f"    In-use (live buffers) : {in_use  / GiB:.3f} GiB  ({in_use  / MiB:.1f} MiB)")
        print(f"    Peak since process    : {peak    / GiB:.3f} GiB  ({peak    / MiB:.1f} MiB)")
        if limit:
            print(f"    Device limit          : {limit  / GiB:.3f} GiB  ({limit  / MiB:.1f} MiB)")
            print(f"    Free (limit - in-use) : {(limit - in_use) / GiB:.3f} GiB")
        if reserved:
            print(f"    Reservable pool       : {reserved / GiB:.3f} GiB")
        if fragmented is not None:
            print(f"    Fragmentation         : {fragmented:.4f}")

        # Print remaining keys for completeness.
        known = {"bytes_in_use", "bytes_used", "peak_bytes_in_use", "peak_bytes_used",
                 "bytes_limit", "memory_limit", "bytes_reservable", "bytes_reserved",
                 "fragmentation"}
        other = {k: v for k, v in stats.items() if k not in known}
        if other:
            print("    Additional stats:")
            for k, v in sorted(other.items()):
                if isinstance(v, int) and v > MiB:
                    print(f"      {k}: {v / GiB:.3f} GiB  ({v / MiB:.1f} MiB)")
                else:
                    print(f"      {k}: {v}")

    # --- nvidia-smi breakdown (actual VRAM, outside JAX's view) ---
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            print("\n  [nvidia-smi] per-GPU VRAM (MiB):")
            print(f"    {'idx':>3}  {'name':<30}  {'total':>8}  {'used':>8}  {'free':>8}  {'gpu%':>5}")
            print("    " + "-" * 65)
            for line in smi.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    idx, name, total, used, free, util = parts[:6]
                    print(f"    {idx:>3}  {name:<30}  {total:>8}  {used:>8}  {free:>8}  {util:>4}%")
        else:
            print("\n  [nvidia-smi] not available or returned error")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("\n  [nvidia-smi] not found on this system")

    print("=" * 80 + "\n")


def _make_train_fn(trainer: str, config: dict):
    if trainer == "ctm":
        return make_train_ctm(config)
    if trainer == "gru":
        return make_train_gru(config)
    raise ValueError(f"Unknown trainer={trainer}.")


def _plot_seed(seed: int, ctm_summary: dict, gru_summary: dict):
    """Save a 2-panel plot (Win Rate | Return) with CTM and GRU curves."""
    ctm_wr = ctm_summary["win_curve"]
    ctm_ret = ctm_summary["ret_curve"]
    gru_wr = gru_summary["win_curve"]
    gru_ret = gru_summary["ret_curve"]

    step_size = BASE_CONFIG["NUM_ENVS"] * BASE_CONFIG["NUM_STEPS"]

    def _steps(curve):
        return np.arange(1, len(curve) + 1) * step_size

    fig, (ax_wr, ax_ret) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"CTM iter=3 vs GRU  |  seed={seed}", fontsize=13)

    # --- Win Rate panel ---
    if ctm_wr.size:
        ax_wr.plot(_steps(ctm_wr), ctm_wr, label="CTM 3 iter", color="royalblue")
    if gru_wr.size:
        ax_wr.plot(_steps(gru_wr), gru_wr, label="GRU", color="darkorange")
    ax_wr.set_xlabel("Environment steps")
    ax_wr.set_ylabel("Win Rate")
    ax_wr.set_title("Win Rate vs Steps")
    ax_wr.legend()
    ax_wr.grid(True, alpha=0.3)

    # --- Return panel ---
    if ctm_ret.size:
        ax_ret.plot(_steps(ctm_ret), ctm_ret, label="CTM 3 iter", color="royalblue")
    if gru_ret.size:
        ax_ret.plot(_steps(gru_ret), gru_ret, label="GRU", color="darkorange")
    ax_ret.set_xlabel("Environment steps")
    ax_ret.set_ylabel("Episode Return")
    ax_ret.set_title("Return vs Steps")
    ax_ret.legend()
    ax_ret.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = os.path.join(os.getcwd(), f"baseline_seed{seed}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[rebaseline] plot saved → {out_path}")


def run_one(cell_name: str, cell_def: dict, seed: int, report_memory: bool = False) -> dict:
    trainer = cell_def["trainer"]
    config = dict(BASE_CONFIG)
    config.update(cell_def["overrides"])
    config["SEED"] = int(seed)

    tag = f"{cell_name}__seed{seed}"
    print(f"[rebaseline] launching {tag}")

    rng = jax.random.PRNGKey(seed)
    train_jit = jax.jit(_make_train_fn(trainer, config))

    start = time.time()
    out = train_jit(rng)
    jax.block_until_ready(out["metric"])
    elapsed = time.time() - start
    print(f"[rebaseline]   {tag} finished in {elapsed / 60.0:.1f} min")

    if report_memory:
        _print_gpu_memory(f"{cell_def['label']}  seed={seed}  envs={config['NUM_ENVS']}")

    summary = _summarize_run(out=out, trainer=trainer, seed=seed, elapsed_sec=elapsed)
    summary["cell"] = cell_name
    summary["label"] = cell_def["label"]
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run matched CTM iter=3 / GRU baseline cells.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Shared seed list.",
    )
    args = parser.parse_args()

    summaries = []
    by_cell = {cell_def["label"]: [] for _, cell_def in CELLS}

    for seed_idx, seed in enumerate(args.seeds):
        seed_summaries = {}

        # Run CTM first, then GRU for this seed.
        # Report GPU memory after the very first CTM run (JIT compiled + buffers live).
        for cell_name, cell_def in CELLS:
            report_mem = (cell_def["trainer"] == "ctm" and seed_idx == 0)
            s = run_one(cell_name, cell_def, seed, report_memory=report_mem)
            summaries.append(s)
            by_cell[cell_def["label"]].append(s)
            seed_summaries[cell_name] = s

        # Per-seed plot immediately after both runs complete.
        ctm_s = seed_summaries["ctm_iter3_vanilla"]
        gru_s = seed_summaries["gru_same_budget"]
        _plot_seed(seed, ctm_s, gru_s)

    # Final per-config tables.
    for _, cell_def in CELLS:
        label = cell_def["label"]
        if label in by_cell and by_cell[label]:
            _print_config_table(label, by_cell[label])

    _print_overall_summary(by_cell)


if __name__ == "__main__":
    main()
