import argparse
import os
import sys

# Keep repo-root import behavior consistent with existing training/eval scripts.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from smax_ctm.analysis.checkpoint import extract_ctm_and_head_params, load_ctm_checkpoint
from smax_ctm.analysis.collector import collect_episodes
from smax_ctm.analysis.io_utils import prepare_output_dirs, save_pickle
from smax_ctm.analysis.metrics import compute_pairwise_metrics
from smax_ctm.analysis.plotting import plot_pairwise_heatmap, plot_sync_timeseries


def _resolve_relative_to_script(path: str) -> str:
    if not path:
        raise ValueError("Path argument must be non-empty")
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(_SCRIPT_DIR, path))


def _resolve_checkpoint_path(path: str) -> str:
    # For Colab usage from /content, resolve relative paths from script location.
    resolved = _resolve_relative_to_script(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            "Checkpoint path does not exist after script-relative resolution. "
            f"provided='{path}' resolved='{resolved}'"
        )
    return resolved


def _resolve_output_dir(path: str) -> str:
    return _resolve_relative_to_script(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect CTM diagnostics and compute cross-agent synchronisation metrics."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_REPO_ROOT, "model", "smax_mappo_ctm_actor.pkl"),
        help="Path to CTM actor checkpoint (.pkl). Relative paths are resolved from smax_ctm/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "analysis_results"),
        help="Directory for saved traces, metrics, and figures. Relative paths are resolved from smax_ctm/.",
    )
    parser.add_argument("--num-episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Base PRNG seed.")
    parser.add_argument("--map-name", type=str, default=None, help="Override map name from checkpoint config.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max rollout steps per episode.")
    parser.add_argument(
        "--grouping-radius",
        type=float,
        default=2.5,
        help="Mean ally pair-distance threshold for grouping event.",
    )
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of greedy mode().")
    parser.add_argument("--no-plots", action="store_true", help="Skip writing PNG plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    output_dir = _resolve_output_dir(args.output_dir)

    checkpoint = load_ctm_checkpoint(checkpoint_path)
    ctm_params, head_params = extract_ctm_and_head_params(checkpoint.actor_params)
    out_paths = prepare_output_dirs(output_dir)

    print(f"Resolved checkpoint path: {checkpoint_path}")
    print(f"Resolved output directory: {output_dir}")
    print(f"Loaded checkpoint: {checkpoint.path}")
    print(
        "Collecting episodes "
        f"(num_episodes={args.num_episodes}, seed={args.seed}, map={args.map_name or checkpoint.config.get('MAP_NAME')})"
    )
    collection = collect_episodes(
        config=checkpoint.config,
        ctm_params=ctm_params,
        head_params=head_params,
        num_episodes=args.num_episodes,
        seed=args.seed,
        map_name=args.map_name,
        max_steps=args.max_steps,
        stochastic=args.stochastic,
        grouping_radius=args.grouping_radius,
    )
    save_pickle(collection, out_paths["episode_traces_path"])

    print("Computing pairwise synchronisation metrics...")
    metrics = compute_pairwise_metrics(collection)
    save_pickle(metrics, out_paths["sync_metrics_path"])

    if metrics["metadata"]["undefined_corr_count"] > 0:
        print(
            "WARNING: "
            f"{metrics['metadata']['undefined_corr_count']} correlation values were undefined and stored as NaN."
        )

    print(
        "Global means: "
        f"sync={metrics['global']['sync_corr_mean']:.4f}, "
        f"obs={metrics['global']['obs_corr_mean']:.4f}, "
        f"sync-obs={metrics['global']['sync_minus_obs_mean']:.4f}"
    )

    if not args.no_plots:
        print("Writing quick-look figures...")
        plot_sync_timeseries(metrics, out_paths["figures_dir"], max_episodes=min(3, args.num_episodes))
        plot_pairwise_heatmap(metrics, out_paths["figures_dir"], episode_index=0)

    print(f"Saved traces: {out_paths['episode_traces_path']}")
    print(f"Saved metrics: {out_paths['sync_metrics_path']}")
    if not args.no_plots:
        print(f"Saved figures in: {out_paths['figures_dir']}")


if __name__ == "__main__":
    main()
