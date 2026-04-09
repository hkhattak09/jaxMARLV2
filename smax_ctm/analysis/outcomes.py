from typing import Any, Dict, List, Optional

import numpy as np


def _pearson_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"Expected rank-1 vectors, got {x.shape} and {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Length mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.size < 2:
        return np.nan
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _episode_return_map(collection: Dict[str, Any]) -> Dict[int, float]:
    mapping: Dict[int, float] = {}
    for ep in collection["episodes"]:
        idx = int(ep["episode_index"])
        if idx in mapping:
            raise ValueError(f"Duplicate episode index in collection: {idx}")
        mapping[idx] = float(ep["episode_return_mean"])
    return mapping


def _episode_win_map(collection: Dict[str, Any]) -> Dict[int, Optional[bool]]:
    mapping: Dict[int, Optional[bool]] = {}
    for ep in collection["episodes"]:
        idx = int(ep["episode_index"])
        won = ep.get("episode_won", None)
        if won is None:
            mapping[idx] = None
        else:
            mapping[idx] = bool(won)
    return mapping


def compute_outcome_diagnostics(
    metrics: Dict[str, Any],
    collection: Dict[str, Any],
    win_return_threshold: float = 0.0,
    strict: bool = True,
) -> Dict[str, Any]:
    ret_by_ep = _episode_return_map(collection)
    win_by_ep = _episode_win_map(collection)

    rows: List[Dict[str, Any]] = []
    for ep in metrics["episodes"]:
        idx = int(ep["episode_index"])
        if idx not in ret_by_ep:
            raise KeyError(
                f"Episode index {idx} exists in metrics but not in collection. "
                "Ensure both files are from the same run."
            )

        sync_mean = float(np.nanmean(np.asarray(ep["sync_pair_mean_ts"], dtype=np.float64)))
        obs_mean = float(np.nanmean(np.asarray(ep["obs_pair_mean_ts"], dtype=np.float64)))
        episode_return = float(ret_by_ep[idx])

        won_flag = win_by_ep[idx]
        if won_flag is None:
            won_flag = bool(episode_return > win_return_threshold)
            win_source = "return_threshold"
        else:
            win_source = "env_info"

        rows.append(
            {
                "episode_index": idx,
                "sync_mean": sync_mean,
                "obs_mean": obs_mean,
                "episode_return_mean": episode_return,
                "episode_won": bool(won_flag),
                "win_source": win_source,
            }
        )

    if not rows:
        raise ValueError("No episode rows available for outcome diagnostics")

    sync_vals = np.asarray([r["sync_mean"] for r in rows], dtype=np.float64)
    obs_vals = np.asarray([r["obs_mean"] for r in rows], dtype=np.float64)
    ret_vals = np.asarray([r["episode_return_mean"] for r in rows], dtype=np.float64)
    win_vals = np.asarray([r["episode_won"] for r in rows], dtype=bool)

    if strict and np.all(win_vals == win_vals[0]):
        raise ValueError(
            "All episodes have the same win label; won-vs-lost trajectory analysis is undefined. "
            "Collect a more diverse episode set or use a harder map."
        )

    won_sync = sync_vals[win_vals]
    lost_sync = sync_vals[~win_vals]

    result = {
        "metadata": {
            "num_episodes": int(len(rows)),
            "win_return_threshold": float(win_return_threshold),
            "strict": bool(strict),
        },
        "episodes": rows,
        "global": {
            "sync_return_corr": _pearson_or_nan(sync_vals, ret_vals),
            "obs_return_corr": _pearson_or_nan(obs_vals, ret_vals),
            "won_sync_mean": float(np.mean(won_sync)) if won_sync.size else np.nan,
            "lost_sync_mean": float(np.mean(lost_sync)) if lost_sync.size else np.nan,
            "won_minus_lost_sync": (
                float(np.mean(won_sync) - np.mean(lost_sync))
                if won_sync.size and lost_sync.size
                else np.nan
            ),
            "num_won": int(np.sum(win_vals)),
            "num_lost": int(np.sum(~win_vals)),
        },
    }
    return result
