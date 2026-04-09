import itertools
from typing import Any, Dict, List, Tuple

import numpy as np


def _pearson_or_nan(x: np.ndarray, y: np.ndarray, context: str) -> Tuple[float, bool]:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"{context}: expected 1D vectors, got {x.shape} and {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{context}: vector length mismatch {x.shape[0]} vs {y.shape[0]}")

    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return np.nan, True

    return float(np.corrcoef(x, y)[0, 1]), False


def _event_conditional_mean(values: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if values.shape[0] != mask.shape[0]:
        raise ValueError(f"Event/value shape mismatch: values={values.shape}, mask={mask.shape}")

    event_values = values[mask]
    non_event_values = values[~mask]

    return {
        "during": float(np.nanmean(event_values)) if event_values.size else np.nan,
        "outside": float(np.nanmean(non_event_values)) if non_event_values.size else np.nan,
        "delta": (
            float(np.nanmean(event_values) - np.nanmean(non_event_values))
            if event_values.size and non_event_values.size
            else np.nan
        ),
    }


def compute_pairwise_metrics(collection: Dict[str, Any]) -> Dict[str, Any]:
    episodes: List[Dict[str, Any]] = collection["episodes"]
    if not episodes:
        raise ValueError("No episodes present in collection")

    num_agents = collection["metadata"]["num_agents"]
    if num_agents < 2:
        raise ValueError(f"Need >=2 agents for cross-agent metrics, got {num_agents}")

    pair_indices = list(itertools.combinations(range(num_agents), 2))
    episode_metrics = []
    global_sync_corr = []
    global_obs_corr = []
    undefined_corr_count = 0

    for ep in episodes:
        steps = ep["steps"]
        sync_corr_ts = []
        obs_corr_ts = []
        focus_fire = []
        grouping = []
        enemy_kill = []

        for step in steps:
            synch = np.asarray(step["synch"], dtype=np.float64)
            obs = np.asarray(step["obs"], dtype=np.float64)
            if synch.shape[0] != num_agents:
                raise ValueError(
                    "Synch batch mismatch inside collection: "
                    f"expected {num_agents}, got {synch.shape[0]}"
                )
            if obs.shape[0] != num_agents:
                raise ValueError(
                    "Obs batch mismatch inside collection: "
                    f"expected {num_agents}, got {obs.shape[0]}"
                )

            step_sync_corr = []
            step_obs_corr = []
            for i, j in pair_indices:
                sync_corr, sync_undefined = _pearson_or_nan(
                    synch[i], synch[j], context=f"episode={ep['episode_index']} step={step['t']} sync"
                )
                obs_corr, obs_undefined = _pearson_or_nan(
                    obs[i], obs[j], context=f"episode={ep['episode_index']} step={step['t']} obs"
                )
                undefined_corr_count += int(sync_undefined) + int(obs_undefined)
                step_sync_corr.append(sync_corr)
                step_obs_corr.append(obs_corr)

            sync_corr_ts.append(step_sync_corr)
            obs_corr_ts.append(step_obs_corr)
            focus_fire.append(bool(step["events"]["focus_fire"]))
            grouping.append(bool(step["events"]["grouping"]))
            enemy_kill.append(bool(step["events"]["enemy_kill"]))

        sync_corr_ts = np.asarray(sync_corr_ts, dtype=np.float64)
        obs_corr_ts = np.asarray(obs_corr_ts, dtype=np.float64)
        focus_fire = np.asarray(focus_fire, dtype=bool)
        grouping = np.asarray(grouping, dtype=bool)
        enemy_kill = np.asarray(enemy_kill, dtype=bool)

        sync_pair_mean = np.nanmean(sync_corr_ts, axis=1)
        obs_pair_mean = np.nanmean(obs_corr_ts, axis=1)
        global_sync_corr.extend(sync_pair_mean.tolist())
        global_obs_corr.extend(obs_pair_mean.tolist())

        episode_metrics.append(
            {
                "episode_index": ep["episode_index"],
                "timesteps": len(steps),
                "pair_indices": pair_indices,
                "sync_corr_ts": sync_corr_ts,
                "obs_corr_ts": obs_corr_ts,
                "sync_pair_mean_ts": sync_pair_mean,
                "obs_pair_mean_ts": obs_pair_mean,
                "event_conditional": {
                    "focus_fire": _event_conditional_mean(sync_pair_mean, focus_fire),
                    "grouping": _event_conditional_mean(sync_pair_mean, grouping),
                    "enemy_kill": _event_conditional_mean(sync_pair_mean, enemy_kill),
                },
                "event_counts": {
                    "focus_fire": int(np.sum(focus_fire)),
                    "grouping": int(np.sum(grouping)),
                    "enemy_kill": int(np.sum(enemy_kill)),
                },
            }
        )

    global_sync_corr = np.asarray(global_sync_corr, dtype=np.float64)
    global_obs_corr = np.asarray(global_obs_corr, dtype=np.float64)

    return {
        "metadata": {
            "num_agents": num_agents,
            "pair_indices": pair_indices,
            "num_episodes": len(episodes),
            "undefined_corr_count": int(undefined_corr_count),
            "undefined_corr_warning": (
                "Some correlations are undefined (constant vectors). They are stored as NaN."
                if undefined_corr_count > 0
                else "none"
            ),
        },
        "global": {
            "sync_corr_mean": float(np.nanmean(global_sync_corr)),
            "sync_corr_std": float(np.nanstd(global_sync_corr)),
            "obs_corr_mean": float(np.nanmean(global_obs_corr)),
            "obs_corr_std": float(np.nanstd(global_obs_corr)),
            "sync_minus_obs_mean": float(np.nanmean(global_sync_corr - global_obs_corr)),
        },
        "episodes": episode_metrics,
    }
