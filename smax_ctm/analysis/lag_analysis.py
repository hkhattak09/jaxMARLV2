from typing import Any, Dict, List

import numpy as np

from smax_ctm.analysis.event_stats import EVENT_NAMES


def _validate_lag_args(max_lag: int, lead_window: int, num_permutations: int) -> None:
    if max_lag <= 0:
        raise ValueError(f"max_lag must be > 0, got {max_lag}")
    if lead_window <= 0:
        raise ValueError(f"lead_window must be > 0, got {lead_window}")
    if lead_window >= max_lag:
        raise ValueError(
            f"lead_window ({lead_window}) must be smaller than max_lag ({max_lag})"
        )
    if num_permutations <= 0:
        raise ValueError(f"num_permutations must be > 0, got {num_permutations}")


def _event_lag_values(sync: np.ndarray, event_mask: np.ndarray, max_lag: int) -> Dict[int, List[float]]:
    values = {lag: [] for lag in range(-max_lag, max_lag + 1)}
    event_idxs = np.where(event_mask)[0]

    for t in event_idxs:
        for lag in range(-max_lag, max_lag + 1):
            idx = t + lag
            if 0 <= idx < sync.shape[0]:
                val = float(sync[idx])
                if np.isfinite(val):
                    values[lag].append(val)

    return values


def _merge_lag_values(dst: Dict[int, List[float]], src: Dict[int, List[float]]) -> None:
    for lag in dst:
        dst[lag].extend(src[lag])


def _lag_profile_from_values(values: Dict[int, List[float]]) -> Dict[str, Any]:
    lags = np.array(sorted(values.keys()), dtype=np.int32)
    means = []
    sems = []
    counts = []
    for lag in lags:
        arr = np.asarray(values[int(lag)], dtype=np.float64)
        counts.append(int(arr.size))
        if arr.size == 0:
            means.append(np.nan)
            sems.append(np.nan)
        else:
            means.append(float(np.mean(arr)))
            if arr.size == 1:
                sems.append(np.nan)
            else:
                sems.append(float(np.std(arr, ddof=1) / np.sqrt(arr.size)))

    return {
        "lags": lags,
        "mean": np.asarray(means, dtype=np.float64),
        "sem": np.asarray(sems, dtype=np.float64),
        "count": np.asarray(counts, dtype=np.int32),
    }


def _lead_minus_lag_delta(profile: Dict[str, Any], lead_window: int) -> float:
    lags = np.asarray(profile["lags"], dtype=np.int32)
    means = np.asarray(profile["mean"], dtype=np.float64)

    lead_mask = (lags >= -lead_window) & (lags <= -1)
    lag_mask = (lags >= 1) & (lags <= lead_window)

    lead_vals = means[lead_mask]
    lag_vals = means[lag_mask]
    if lead_vals.size == 0 or lag_vals.size == 0:
        return np.nan
    if not np.any(np.isfinite(lead_vals)) or not np.any(np.isfinite(lag_vals)):
        return np.nan

    return float(np.nanmean(lead_vals) - np.nanmean(lag_vals))


def _perm_delta(
    episodes: List[Dict[str, Any]],
    event_name: str,
    observed_delta: float,
    max_lag: int,
    lead_window: int,
    num_permutations: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    # Null model: keep event count per episode, randomize event timesteps.
    perm_deltas = []
    for _ in range(num_permutations):
        merged = {lag: [] for lag in range(-max_lag, max_lag + 1)}
        for ep in episodes:
            sync = np.asarray(ep["sync_pair_mean_ts"], dtype=np.float64)
            mask = np.asarray(ep["event_masks"][event_name], dtype=bool)
            if sync.shape[0] != mask.shape[0]:
                raise ValueError(
                    f"Episode {ep['episode_index']} has mismatched sync/mask lengths: "
                    f"{sync.shape[0]} vs {mask.shape[0]}"
                )
            k = int(np.sum(mask))
            if k == 0:
                continue
            perm_idx = rng.choice(sync.shape[0], size=k, replace=False)
            perm_mask = np.zeros_like(mask)
            perm_mask[perm_idx] = True
            vals = _event_lag_values(sync, perm_mask, max_lag)
            _merge_lag_values(merged, vals)

        profile = _lag_profile_from_values(merged)
        perm_deltas.append(_lead_minus_lag_delta(profile, lead_window))

    perm_deltas = np.asarray(perm_deltas, dtype=np.float64)
    perm_deltas = perm_deltas[np.isfinite(perm_deltas)]
    if perm_deltas.size == 0:
        return {
            "mean": np.nan,
            "p_one_sided": np.nan,
            "num_valid": 0,
        }

    return {
        "mean": float(np.mean(perm_deltas)),
        "p_one_sided": float((np.sum(perm_deltas >= observed_delta) + 1) / (perm_deltas.size + 1)),
        "num_valid": int(perm_deltas.size),
    }


def compute_event_lag_profiles(
    metrics: Dict[str, Any],
    max_lag: int = 12,
    lead_window: int = 3,
    num_permutations: int = 2000,
    seed: int = 0,
    strict: bool = True,
) -> Dict[str, Any]:
    _validate_lag_args(max_lag, lead_window, num_permutations)
    rng = np.random.default_rng(seed)

    episodes = metrics["episodes"]
    results: Dict[str, Any] = {}

    for event_name in EVENT_NAMES:
        merged = {lag: [] for lag in range(-max_lag, max_lag + 1)}
        event_count = 0
        for ep in episodes:
            sync = np.asarray(ep["sync_pair_mean_ts"], dtype=np.float64)
            if "event_masks" not in ep or event_name not in ep["event_masks"]:
                raise KeyError(
                    f"Episode {ep['episode_index']} missing event mask '{event_name}'. "
                    "Recompute metrics with the updated pipeline."
                )
            mask = np.asarray(ep["event_masks"][event_name], dtype=bool)
            if sync.shape[0] != mask.shape[0]:
                raise ValueError(
                    f"Episode {ep['episode_index']} has mismatched sync/mask lengths: "
                    f"{sync.shape[0]} vs {mask.shape[0]}"
                )

            event_count += int(np.sum(mask))
            vals = _event_lag_values(sync, mask, max_lag)
            _merge_lag_values(merged, vals)

        if event_count == 0:
            msg = f"No events found for '{event_name}', cannot compute lag profile."
            if strict:
                raise ValueError(msg)
            results[event_name] = {"status": "insufficient_data", "message": msg}
            continue

        profile = _lag_profile_from_values(merged)
        lead_minus_lag = _lead_minus_lag_delta(profile, lead_window)
        null_stats = _perm_delta(
            episodes,
            event_name=event_name,
            observed_delta=lead_minus_lag,
            max_lag=max_lag,
            lead_window=lead_window,
            num_permutations=num_permutations,
            rng=rng,
        )

        results[event_name] = {
            "status": "ok",
            "event_count": int(event_count),
            "profile": profile,
            "lead_minus_lag_delta": lead_minus_lag,
            "null_lead_minus_lag_mean": null_stats["mean"],
            "lead_minus_lag_perm_p_one_sided": null_stats["p_one_sided"],
            "lead_minus_lag_perm_num_valid": null_stats["num_valid"],
        }

    return {
        "metadata": {
            "max_lag": int(max_lag),
            "lead_window": int(lead_window),
            "num_permutations": int(num_permutations),
            "seed": int(seed),
            "strict": bool(strict),
            "event_names": list(EVENT_NAMES),
        },
        "events": results,
    }
