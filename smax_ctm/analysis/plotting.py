import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def _to_np(x: Any) -> np.ndarray:
    return np.asarray(x)


def _draw_event_lines(ax: Any, event_masks: Dict[str, Any]) -> None:
    required = ("focus_fire", "grouping", "enemy_kill")
    missing = [k for k in required if k not in event_masks]
    if missing:
        raise KeyError(f"event_masks missing required keys: {missing}")

    style = {
        "focus_fire": {"color": "#f4a261", "label": "focus_fire"},
        "grouping": {"color": "#2a9d8f", "label": "grouping"},
        "enemy_kill": {"color": "#e76f51", "label": "enemy_kill"},
    }
    added_labels = set()
    for event_name, cfg in style.items():
        mask = _to_np(event_masks[event_name]).astype(bool)
        idxs = np.where(mask)[0]
        for t in idxs:
            label = cfg["label"] if cfg["label"] not in added_labels else None
            ax.axvline(int(t), color=cfg["color"], linestyle="--", linewidth=0.9, alpha=0.45, label=label)
            if label is not None:
                added_labels.add(label)


def plot_sync_timeseries(metrics: Dict[str, Any], figures_dir: str, max_episodes: int = 3) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    for ep in metrics["episodes"][:max_episodes]:
        sync_mean = _to_np(ep["sync_pair_mean_ts"])
        obs_mean = _to_np(ep["obs_pair_mean_ts"])
        event_masks = ep.get("event_masks", None)
        if event_masks is None:
            raise KeyError(
                f"Episode {ep['episode_index']} missing event_masks. "
                "Recompute metrics with the updated pipeline."
            )
        t = np.arange(sync_mean.shape[0])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, sync_mean, label="sync pairwise corr", linewidth=2)
        ax.plot(t, obs_mean, label="obs pairwise corr", linewidth=1.5, alpha=0.8)
        _draw_event_lines(ax, event_masks)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Correlation")
        ax.set_title(f"Episode {ep['episode_index']} | Sync vs Obs Correlation")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"sync_timeseries_ep{ep['episode_index']}.png"), dpi=150)
        plt.close(fig)


def plot_pairwise_heatmap(metrics: Dict[str, Any], figures_dir: str, episode_index: int = 0) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    episodes = metrics["episodes"]
    selected = None
    for ep in episodes:
        if int(ep["episode_index"]) == int(episode_index):
            selected = ep
            break
    if selected is None:
        raise ValueError(f"Could not find episode_index={episode_index} in computed metrics")

    arr = _to_np(selected["sync_corr_ts"]).T
    pair_labels = [f"{i}-{j}" for i, j in selected["pair_indices"]]

    plt.figure(figsize=(10, 3 + 0.5 * len(pair_labels)))
    plt.imshow(arr, aspect="auto", interpolation="nearest")
    plt.yticks(np.arange(len(pair_labels)), pair_labels)
    plt.xlabel("Timestep")
    plt.ylabel("Agent pair")
    plt.title(f"Episode {selected['episode_index']} | Pairwise Sync Correlation")
    plt.colorbar(label="Pearson corr")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"sync_heatmap_ep{selected['episode_index']}.png"), dpi=150)
    plt.close()
