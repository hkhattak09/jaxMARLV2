import os
import pickle
from typing import Any, Dict


def load_pickle(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file does not exist: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def prepare_output_dirs(base_dir: str) -> Dict[str, str]:
    if not base_dir:
        raise ValueError("base_dir must be a non-empty path")

    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    return {
        "base_dir": base_dir,
        "figures_dir": figures_dir,
        "episode_traces_path": os.path.join(base_dir, "episode_traces.pkl"),
        "sync_metrics_path": os.path.join(base_dir, "sync_metrics.pkl"),
        "event_stats_path": os.path.join(base_dir, "event_stats.pkl"),
        "lag_profiles_path": os.path.join(base_dir, "event_lag_profiles.pkl"),
        "outcomes_path": os.path.join(base_dir, "outcome_diagnostics.pkl"),
    }
