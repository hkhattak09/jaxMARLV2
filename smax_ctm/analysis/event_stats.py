from math import erfc, sqrt
from typing import Any, Dict, List, Tuple

import numpy as np


EVENT_NAMES = ("focus_fire", "grouping", "enemy_kill")


def _validate_1d(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {arr.shape}")
    return arr


def _drop_nan_pairs(during: np.ndarray, outside: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    during = _validate_1d("during", during)
    outside = _validate_1d("outside", outside)
    during = during[np.isfinite(during)]
    outside = outside[np.isfinite(outside)]
    return during, outside


def _normal_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2.0))


def _welch_t_stats(during: np.ndarray, outside: np.ndarray) -> Tuple[float, float]:
    n1 = during.size
    n0 = outside.size
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan

    m1 = float(np.mean(during))
    m0 = float(np.mean(outside))
    v1 = float(np.var(during, ddof=1))
    v0 = float(np.var(outside, ddof=1))
    denom = np.sqrt(v1 / n1 + v0 / n0)
    if denom == 0.0:
        return np.nan, np.nan

    t_stat = (m1 - m0) / denom
    # We use a normal approximation to keep dependencies minimal.
    p_one_sided = 1.0 - _normal_cdf(t_stat)
    return float(t_stat), float(p_one_sided)


def _permutation_p_value(
    during: np.ndarray,
    outside: np.ndarray,
    num_permutations: int,
    rng: np.random.Generator,
) -> float:
    if num_permutations <= 0:
        raise ValueError(f"num_permutations must be > 0, got {num_permutations}")

    observed = float(np.mean(during) - np.mean(outside))
    pooled = np.concatenate([during, outside], axis=0)
    n_during = during.size
    ge_count = 0

    for _ in range(num_permutations):
        perm = rng.permutation(pooled)
        perm_during = perm[:n_during]
        perm_outside = perm[n_during:]
        perm_diff = float(np.mean(perm_during) - np.mean(perm_outside))
        if perm_diff >= observed:
            ge_count += 1

    # Add-one smoothing for an unbiased finite-sample estimate.
    return float((ge_count + 1) / (num_permutations + 1))


def _event_samples(metrics: Dict[str, Any], event_name: str) -> Tuple[np.ndarray, np.ndarray]:
    during: List[np.ndarray] = []
    outside: List[np.ndarray] = []

    for ep in metrics["episodes"]:
        sync = np.asarray(ep["sync_pair_mean_ts"], dtype=np.float64)
        masks = ep.get("event_masks", None)
        if masks is None or event_name not in masks:
            raise KeyError(
                f"Episode {ep['episode_index']} missing event mask '{event_name}'. "
                "Recompute metrics with the updated pipeline."
            )

        mask = np.asarray(masks[event_name], dtype=bool)
        if sync.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Episode {ep['episode_index']} has mismatched sync/mask lengths: "
                f"{sync.shape[0]} vs {mask.shape[0]}"
            )

        during.append(sync[mask])
        outside.append(sync[~mask])

    return np.concatenate(during), np.concatenate(outside)


def compute_event_statistics(
    metrics: Dict[str, Any],
    num_permutations: int = 5000,
    seed: int = 0,
    strict: bool = True,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    event_results: Dict[str, Dict[str, Any]] = {}

    for event_name in EVENT_NAMES:
        during_raw, outside_raw = _event_samples(metrics, event_name)
        during, outside = _drop_nan_pairs(during_raw, outside_raw)

        if during.size == 0 or outside.size == 0:
            msg = (
                f"Event '{event_name}' has insufficient samples after NaN filtering: "
                f"during={during.size}, outside={outside.size}."
            )
            if strict:
                raise ValueError(msg)
            event_results[event_name] = {
                "status": "insufficient_data",
                "message": msg,
                "during_mean": np.nan,
                "outside_mean": np.nan,
                "delta": np.nan,
                "t_stat": np.nan,
                "t_p_one_sided": np.nan,
                "perm_p_one_sided": np.nan,
                "num_during": int(during.size),
                "num_outside": int(outside.size),
            }
            continue

        t_stat, t_p_one_sided = _welch_t_stats(during, outside)
        perm_p_one_sided = _permutation_p_value(during, outside, num_permutations, rng)

        event_results[event_name] = {
            "status": "ok",
            "during_mean": float(np.mean(during)),
            "outside_mean": float(np.mean(outside)),
            "delta": float(np.mean(during) - np.mean(outside)),
            "t_stat": t_stat,
            "t_p_one_sided": t_p_one_sided,
            "perm_p_one_sided": perm_p_one_sided,
            "num_during": int(during.size),
            "num_outside": int(outside.size),
        }

    return {
        "metadata": {
            "num_episodes": int(metrics["metadata"]["num_episodes"]),
            "num_permutations": int(num_permutations),
            "seed": int(seed),
            "strict": bool(strict),
            "event_names": list(EVENT_NAMES),
        },
        "events": event_results,
    }
