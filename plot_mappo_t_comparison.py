"""Plot MAPPO-T training comparison across backbone, LoRASA, and non-LoRASA runs.

Can be imported in a Jupyter notebook cell or run as a script.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt


def _read_progress_csv(path: str) -> Dict[str, np.ndarray]:
    """Read a progress CSV and return a dict of column-name -> np.ndarray."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    columns = {}
    for key in rows[0].keys():
        columns[key] = np.array([float(row[key]) for row in rows])
    return columns


def _moving_average(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Compute a simple causal moving average."""
    if len(data) == 0:
        return data
    window = min(window, len(data))
    cumsum = np.cumsum(np.insert(data, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / window


def _apply_ma_to_columns(data: Dict[str, np.ndarray], window: int = 10) -> Dict[str, np.ndarray]:
    """Apply moving average to every column, aligning step to the window end."""
    ma_data = {}
    n = len(data["step"])
    effective_window = min(window, n)
    offset = effective_window - 1
    for key, arr in data.items():
        if key == "step":
            ma_data[key] = arr[offset:]
        else:
            ma_data[key] = _moving_average(arr, window=window)
    return ma_data


def plot_comparison(backbone_csv: str, losara_csv: str, no_losara_csv: str, ma_window: int = 10):
    """Plot backbone, LoRASA continuation, and non-LoRASA continuation.

    Displays a 1x2 figure: left = raw curves, right = moving-average curves.
    The moving averages for LoRASA and No-LoRASA are computed on the **full**
    CSVs (including the backbone portion) so the curves are continuous at the
    transition.  Only the continuation tail is plotted in the distinct colours.

    Args:
        backbone_csv: Path to the backbone training progress.csv.
        losara_csv: Path to the LoRASA fine-tuning progress.csv (includes
            backbone rows prepended).
        no_losara_csv: Path to the non-LoRASA fine-tuning progress.csv
            (includes backbone rows prepended).
        ma_window: Window size (in data points) for the moving-average panel.
    """
    backbone = _read_progress_csv(backbone_csv)
    losara = _read_progress_csv(losara_csv)
    no_losara = _read_progress_csv(no_losara_csv)

    backbone_last_step = float(backbone["step"][-1])

    # --- Raw masks ---
    losara_raw_mask = losara["step"] > backbone_last_step
    no_losara_raw_mask = no_losara["step"] > backbone_last_step

    # --- Moving averages on FULL datasets (continuous) ---
    backbone_ma = _apply_ma_to_columns(backbone, window=ma_window)
    losara_ma = _apply_ma_to_columns(losara, window=ma_window)
    no_losara_ma = _apply_ma_to_columns(no_losara, window=ma_window)

    losara_ma_mask = losara_ma["step"] > backbone_last_step
    no_losara_ma_mask = no_losara_ma["step"] > backbone_last_step

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---------- Left: Raw ----------
    ax = axes[0]
    ax.plot(
        backbone["step"], backbone["win_rate"],
        color="tab:blue", linewidth=1.5, label="Backbone",
    )
    ax.fill_between(
        backbone["step"],
        backbone["win_rate"] - backbone["win_rate_std"],
        backbone["win_rate"] + backbone["win_rate_std"],
        alpha=0.2, color="tab:blue",
    )

    if np.any(losara_raw_mask):
        ax.plot(
            losara["step"][losara_raw_mask], losara["win_rate"][losara_raw_mask],
            color="tab:orange", linewidth=1.5, label="LoRASA",
        )
        ax.fill_between(
            losara["step"][losara_raw_mask],
            losara["win_rate"][losara_raw_mask] - losara["win_rate_std"][losara_raw_mask],
            losara["win_rate"][losara_raw_mask] + losara["win_rate_std"][losara_raw_mask],
            alpha=0.2, color="tab:orange",
        )

    if np.any(no_losara_raw_mask):
        ax.plot(
            no_losara["step"][no_losara_raw_mask], no_losara["win_rate"][no_losara_raw_mask],
            color="tab:green", linewidth=1.5, label="No LoRASA",
        )
        ax.fill_between(
            no_losara["step"][no_losara_raw_mask],
            no_losara["win_rate"][no_losara_raw_mask] - no_losara["win_rate_std"][no_losara_raw_mask],
            no_losara["win_rate"][no_losara_raw_mask] + no_losara["win_rate_std"][no_losara_raw_mask],
            alpha=0.2, color="tab:green",
        )

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Win Rate")
    ax.set_title("MAPPO-T Comparison (Raw)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)

    # ---------- Right: Moving Average ----------
    ax = axes[1]
    ax.plot(
        backbone_ma["step"], backbone_ma["win_rate"],
        color="tab:blue", linewidth=1.5, label="Backbone",
    )
    ax.fill_between(
        backbone_ma["step"],
        backbone_ma["win_rate"] - backbone_ma["win_rate_std"],
        backbone_ma["win_rate"] + backbone_ma["win_rate_std"],
        alpha=0.2, color="tab:blue",
    )

    if np.any(losara_ma_mask):
        ax.plot(
            losara_ma["step"][losara_ma_mask], losara_ma["win_rate"][losara_ma_mask],
            color="tab:orange", linewidth=1.5, label="LoRASA",
        )
        ax.fill_between(
            losara_ma["step"][losara_ma_mask],
            losara_ma["win_rate"][losara_ma_mask] - losara_ma["win_rate_std"][losara_ma_mask],
            losara_ma["win_rate"][losara_ma_mask] + losara_ma["win_rate_std"][losara_ma_mask],
            alpha=0.2, color="tab:orange",
        )

    if np.any(no_losara_ma_mask):
        ax.plot(
            no_losara_ma["step"][no_losara_ma_mask], no_losara_ma["win_rate"][no_losara_ma_mask],
            color="tab:green", linewidth=1.5, label="No LoRASA",
        )
        ax.fill_between(
            no_losara_ma["step"][no_losara_ma_mask],
            no_losara_ma["win_rate"][no_losara_ma_mask] - no_losara_ma["win_rate_std"][no_losara_ma_mask],
            no_losara_ma["win_rate"][no_losara_ma_mask] + no_losara_ma["win_rate_std"][no_losara_ma_mask],
            alpha=0.2, color="tab:green",
        )

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Win Rate")
    ax.set_title(f"MAPPO-T Comparison (MA-{ma_window})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)

    fig.tight_layout()

    out_path = "comparison_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {os.path.abspath(out_path)}")

    plt.show()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MAPPO-T backbone + continuation runs.")
    parser.add_argument("--backbone_csv", required=True)
    parser.add_argument("--losara_csv", required=True)
    parser.add_argument("--no_losara_csv", required=True)
    parser.add_argument("--ma_window", type=int, default=10)
    args = parser.parse_args()

    plot_comparison(args.backbone_csv, args.losara_csv, args.no_losara_csv, ma_window=args.ma_window)
