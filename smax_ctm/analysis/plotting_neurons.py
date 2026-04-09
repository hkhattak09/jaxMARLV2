import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _select_episode(collection: Dict[str, Any], episode_index: int) -> Dict[str, Any]:
    for ep in collection["episodes"]:
        if int(ep["episode_index"]) == int(episode_index):
            return ep
    raise ValueError(f"Could not find episode_index={episode_index} in collection")


def _extract_agent_neuron_time(
    steps: List[Dict[str, Any]],
    agent_idx: int,
    memory_reduction: str,
) -> np.ndarray:
    columns = []
    for step in steps:
        activated = np.asarray(step["activated_trace"], dtype=np.float64)
        if activated.ndim != 3:
            raise ValueError(
                f"Expected activated_trace shape (num_agents, d_model, memory_length), got {activated.shape}"
            )
        if not (0 <= agent_idx < activated.shape[0]):
            raise IndexError(f"agent_idx={agent_idx} out of bounds for num_agents={activated.shape[0]}")

        trace = activated[agent_idx]  # (d_model, memory_length)
        if memory_reduction == "last":
            vec = trace[:, -1]
        elif memory_reduction == "mean":
            vec = np.mean(trace, axis=-1)
        else:
            raise ValueError(
                f"Unsupported memory_reduction='{memory_reduction}'. Choose one of ['last', 'mean']."
            )
        columns.append(vec)

    if not columns:
        raise ValueError("Episode has zero steps; cannot plot neuron heatmap")

    # Return neurons x timesteps matrix.
    return np.stack(columns, axis=1)


def plot_neuron_activation_heatmap(
    collection: Dict[str, Any],
    figures_dir: str,
    episode_index: int = 0,
    memory_reduction: str = "last",
) -> None:
    os.makedirs(figures_dir, exist_ok=True)

    episode = _select_episode(collection, episode_index)
    steps = episode["steps"]
    if not steps:
        raise ValueError(f"Episode {episode_index} has no steps")

    first = np.asarray(steps[0]["activated_trace"], dtype=np.float64)
    if first.ndim != 3:
        raise ValueError(
            f"Expected activated_trace shape (num_agents, d_model, memory_length), got {first.shape}"
        )

    num_agents, d_model, _ = first.shape
    per_agent = [_extract_agent_neuron_time(steps, i, memory_reduction) for i in range(num_agents)]

    vmin = float(min(np.nanmin(m) for m in per_agent))
    vmax = float(max(np.nanmax(m) for m in per_agent))

    fig, axes = plt.subplots(1, num_agents, figsize=(4.8 * num_agents, 6.0), sharey=True)
    if num_agents == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        arr = per_agent[i]
        im = ax.imshow(arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"Agent {i} | d_model={d_model}")
        ax.set_xlabel("Timestep")
        if i == 0:
            ax.set_ylabel("Neuron index")

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Activated state")
    fig.suptitle(
        f"Episode {episode_index} | Neuron Activation Heatmap ({memory_reduction} over memory axis)",
        y=1.02,
    )
    fig.tight_layout()
    out_name = f"neuron_activation_heatmap_ep{episode_index}.png"
    fig.savefig(os.path.join(figures_dir, out_name), dpi=160, bbox_inches="tight")
    plt.close(fig)
