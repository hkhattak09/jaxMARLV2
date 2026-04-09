import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def _to_np(x: Any) -> np.ndarray:
    return np.asarray(x)


def plot_sync_timeseries(metrics: Dict[str, Any], figures_dir: str, max_episodes: int = 3) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    for ep in metrics["episodes"][:max_episodes]:
        sync_mean = _to_np(ep["sync_pair_mean_ts"])
        obs_mean = _to_np(ep["obs_pair_mean_ts"])
        t = np.arange(sync_mean.shape[0])

        plt.figure(figsize=(10, 4))
        plt.plot(t, sync_mean, label="sync pairwise corr", linewidth=2)
        plt.plot(t, obs_mean, label="obs pairwise corr", linewidth=1.5, alpha=0.8)
        plt.xlabel("Timestep")
        plt.ylabel("Correlation")
        plt.title(f"Episode {ep['episode_index']} | Sync vs Obs Correlation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"sync_timeseries_ep{ep['episode_index']}.png"), dpi=150)
        plt.close()


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
