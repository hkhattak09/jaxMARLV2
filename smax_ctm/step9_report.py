import numpy as np
import jax


def _to_np(x):
    return np.asarray(jax.device_get(x))


def _window_mean(values, frac=0.1):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    n = max(1, int(round(vals.size * frac)))
    return float(np.mean(vals[:n])), float(np.mean(vals[-n:]))


def _per_update_episode_means(metric):
    if (
        "returned_episode" not in metric
        or "returned_episode_returns" not in metric
        or "returned_won_episode" not in metric
    ):
        return None, None

    mask = _to_np(metric["returned_episode"])
    rets = _to_np(metric["returned_episode_returns"])
    wins = _to_np(metric["returned_won_episode"])

    # Expected shape after training scan: (updates, steps, envs, agents)
    if mask.ndim < 4:
        return None, None

    mask = mask[..., 0].astype(bool)
    rets = rets[..., 0]
    wins = wins[..., 0]

    num_updates = mask.shape[0]
    ret_series = np.full((num_updates,), np.nan, dtype=np.float64)
    win_series = np.full((num_updates,), np.nan, dtype=np.float64)

    for u in range(num_updates):
        m = mask[u]
        if np.any(m):
            ret_series[u] = float(np.mean(rets[u][m]))
            win_series[u] = float(np.mean(wins[u][m]))

    return ret_series, win_series


def _status(name, passed, detail):
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}: {detail}")


def print_step9_summary(metric):
    print("\n=== Step 9 Console Summary ===")

    loss = metric.get("loss", {})
    total_loss = _to_np(loss.get("total_loss", np.array([np.nan])))
    entropy = _to_np(loss.get("entropy", np.array([np.nan])))
    actor_grad_norm = _to_np(loss.get("actor_grad_norm", np.array([np.nan])))
    critic_grad_norm = _to_np(loss.get("critic_grad_norm", np.array([np.nan])))

    any_nan = (
        np.isnan(total_loss).any()
        or np.isnan(entropy).any()
        or np.isnan(actor_grad_norm).any()
        or np.isnan(critic_grad_norm).any()
    )
    _status("No NaN in loss diagnostics", not any_nan, f"nan_present={any_nan}")

    ent_start, ent_end = _window_mean(entropy)
    entropy_ok = np.isfinite(ent_end) and ent_end > 1e-3
    _status(
        "Entropy not collapsed",
        entropy_ok,
        f"start={ent_start:.4f}, end={ent_end:.4f}",
    )

    agn_start, agn_end = _window_mean(actor_grad_norm)
    cgn_start, cgn_end = _window_mean(critic_grad_norm)
    grad_ok = np.isfinite(agn_end) and np.isfinite(cgn_end)
    _status(
        "Gradient norms finite",
        grad_ok,
        (
            f"actor(start/end)={agn_start:.4f}/{agn_end:.4f}, "
            f"critic(start/end)={cgn_start:.4f}/{cgn_end:.4f}"
        ),
    )

    ret_series, win_series = _per_update_episode_means(metric)
    if ret_series is None or win_series is None:
        print("[WARN] Episode trend checks: required metric keys missing; skipped")
        return

    ret_start, ret_end = _window_mean(ret_series)
    win_start, win_end = _window_mean(win_series)

    ret_trend_ok = np.isfinite(ret_start) and np.isfinite(ret_end) and (ret_end >= ret_start)
    win_trend_ok = np.isfinite(win_start) and np.isfinite(win_end) and (win_end >= win_start)

    _status(
        "Return trend non-decreasing",
        ret_trend_ok,
        f"start={ret_start:.4f}, end={ret_end:.4f}",
    )
    _status(
        "Win-rate trend non-decreasing",
        win_trend_ok,
        f"start={win_start:.4f}, end={win_end:.4f}",
    )

    final_ret = ret_series[np.isfinite(ret_series)]
    final_win = win_series[np.isfinite(win_series)]
    if final_ret.size > 0 and final_win.size > 0:
        print(
            "Final observed episode stats: "
            f"return={final_ret[-1]:.4f}, win_rate={final_win[-1]:.4f}"
        )
