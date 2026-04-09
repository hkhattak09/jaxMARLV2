import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_event_delta_bars(event_stats: Dict[str, Any], figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    events = event_stats["events"]

    labels = []
    deltas = []
    pvals = []
    for name, entry in events.items():
        if entry.get("status") != "ok":
            continue
        labels.append(name)
        deltas.append(float(entry["delta"]))
        pvals.append(float(entry["perm_p_one_sided"]))

    if not labels:
        raise ValueError("No valid event entries to plot in event_stats")

    x = np.arange(len(labels))
    colors = ["#2a9d8f" if d >= 0 else "#e76f51" for d in deltas]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(x, deltas, color=colors, alpha=0.9)
    plt.axhline(0.0, color="black", linewidth=1)
    for i, (bar, p) in enumerate(zip(bars, pvals)):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.01 if y >= 0 else -0.01
        plt.text(i, y + offset, f"p={p:.3g}", ha="center", va=va, fontsize=9)

    plt.xticks(x, labels)
    plt.ylabel("During - Outside Sync Corr")
    plt.title("Event-Conditional Sync Correlation Delta")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "event_conditional.png"), dpi=150)
    plt.close()


def plot_event_lag_profiles(lag_profiles: Dict[str, Any], figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    events = lag_profiles["events"]
    valid_events = [(name, v) for name, v in events.items() if v.get("status") == "ok"]
    if not valid_events:
        raise ValueError("No valid lag profiles to plot")

    nrows = len(valid_events)
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 3.0 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, (name, payload) in zip(axes, valid_events):
        profile = payload["profile"]
        lags = np.asarray(profile["lags"], dtype=np.int32)
        mean = np.asarray(profile["mean"], dtype=np.float64)
        sem = np.asarray(profile["sem"], dtype=np.float64)

        ax.plot(lags, mean, color="#264653", linewidth=2, label=name)
        finite = np.isfinite(mean) & np.isfinite(sem)
        if np.any(finite):
            ax.fill_between(lags[finite], mean[finite] - sem[finite], mean[finite] + sem[finite], alpha=0.2)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_ylabel("Sync Corr")
        ax.set_title(
            f"{name} | lead-lag delta={payload['lead_minus_lag_delta']:.4f} "
            f"(null mean={payload['null_lead_minus_lag_mean']:.4f}, "
            f"p={payload['lead_minus_lag_perm_p_one_sided']:.3g})"
        )
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Lag relative to event timestep")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "event_lag_profiles.png"), dpi=150)
    plt.close(fig)


def plot_sync_vs_outcome(outcomes: Dict[str, Any], figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    rows = outcomes["episodes"]
    x = np.asarray([r["sync_mean"] for r in rows], dtype=np.float64)
    y = np.asarray([r["episode_return_mean"] for r in rows], dtype=np.float64)
    won = np.asarray([r["episode_won"] for r in rows], dtype=bool)
    if x.size == 0:
        raise ValueError("No rows in outcomes to plot")

    plt.figure(figsize=(6.5, 5.0))
    plt.scatter(x[~won], y[~won], label="lost", alpha=0.8, color="#e76f51")
    plt.scatter(x[won], y[won], label="won", alpha=0.8, color="#2a9d8f")
    plt.xlabel("Episode Mean Sync Corr")
    plt.ylabel("Episode Mean Return")
    plt.title("Sync Correlation vs Episode Outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "sync_vs_outcome.png"), dpi=150)
    plt.close()


def plot_win_loss_sync_trajectories(metrics: Dict[str, Any], outcomes: Dict[str, Any], figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    by_idx = {int(r["episode_index"]): bool(r["episode_won"]) for r in outcomes["episodes"]}

    won_ts = []
    lost_ts = []
    max_t = 0
    for ep in metrics["episodes"]:
        idx = int(ep["episode_index"])
        if idx not in by_idx:
            raise KeyError(f"Episode {idx} in metrics missing from outcomes")
        ts = np.asarray(ep["sync_pair_mean_ts"], dtype=np.float64)
        max_t = max(max_t, ts.shape[0])
        if by_idx[idx]:
            won_ts.append(ts)
        else:
            lost_ts.append(ts)

    if not won_ts or not lost_ts:
        raise ValueError("Need both won and lost episodes to plot conditional trajectories")

    def _pad_stack(seq):
        arr = np.full((len(seq), max_t), np.nan, dtype=np.float64)
        for i, ts in enumerate(seq):
            arr[i, : ts.shape[0]] = ts
        return arr

    won_arr = _pad_stack(won_ts)
    lost_arr = _pad_stack(lost_ts)

    won_mean = np.nanmean(won_arr, axis=0)
    lost_mean = np.nanmean(lost_arr, axis=0)
    t = np.arange(max_t)

    plt.figure(figsize=(9, 4.5))
    plt.plot(t, won_mean, label="won", linewidth=2, color="#2a9d8f")
    plt.plot(t, lost_mean, label="lost", linewidth=2, color="#e76f51")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Pairwise Sync Corr")
    plt.title("Won vs Lost Sync Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "won_vs_lost_sync.png"), dpi=150)
    plt.close()
