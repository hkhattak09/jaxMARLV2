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
from smax_ctm.analysis.event_stats import compute_event_statistics
from smax_ctm.analysis.io_utils import prepare_output_dirs, save_pickle
from smax_ctm.analysis.lag_analysis import compute_event_lag_profiles
from smax_ctm.analysis.metrics import compute_pairwise_metrics
from smax_ctm.analysis.outcomes import compute_outcome_diagnostics
from smax_ctm.analysis.plotting_neurons import plot_neuron_activation_heatmap
from smax_ctm.analysis.plotting import plot_pairwise_heatmap, plot_sync_timeseries
from smax_ctm.analysis.plotting_stage4 import (
    plot_event_delta_bars,
    plot_event_lag_profiles,
    plot_sync_vs_outcome,
    plot_win_loss_sync_trajectories,
)


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
        default=None,
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
    parser.add_argument("--max-lag", type=int, default=12, help="Max lag for event-aligned analysis.")
    parser.add_argument(
        "--lead-window",
        type=int,
        default=3,
        help="Lead/lag window size for predictive signal summary.",
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=5000,
        help="Permutation count for stage-4 significance tests.",
    )
    parser.add_argument(
        "--win-return-threshold",
        type=float,
        default=0.0,
        help="Fallback win classifier threshold on episode_return_mean.",
    )
    parser.add_argument(
        "--non-strict-stage4",
        action="store_true",
        help="Continue stage-4 analyses with warnings when an event has insufficient samples.",
    )
    parser.add_argument(
        "--non-strict-outcomes",
        action="store_true",
        help=(
            "Continue outcome diagnostics with a warning when all episodes share the same win label "
            "(e.g. trained model wins every episode). Won-vs-lost comparison will be NaN; "
            "sync-return correlation is still reported."
        ),
    )
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of greedy mode().")
    parser.add_argument("--no-plots", action="store_true", help="Skip writing PNG plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)

    checkpoint = load_ctm_checkpoint(checkpoint_path)
    ctm_params, head_params = extract_ctm_and_head_params(checkpoint.actor_params)
    use_sync = bool(checkpoint.config.get("CTM_USE_SYNC", True))

    if args.output_dir is None:
        default_dir = "analysis_results" if use_sync else "analysis_results_nosync"
        output_dir = _resolve_output_dir(os.path.join(_REPO_ROOT, default_dir))
    else:
        output_dir = _resolve_output_dir(args.output_dir)

    out_paths = prepare_output_dirs(output_dir)

    print(f"Resolved checkpoint path: {checkpoint_path}")
    print(f"Resolved output directory: {output_dir}")
    print(f"Loaded checkpoint: {checkpoint.path}")
    print(f"Mode: {'sync' if use_sync else 'no-sync ablation'}")
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
    metrics.setdefault("metadata", {})["use_sync"] = use_sync
    save_pickle(metrics, out_paths["sync_metrics_path"])

    strict_stage4 = not args.non_strict_stage4
    strict_outcomes = not args.non_strict_outcomes
    print("Computing stage-4 event statistics...")
    event_stats = compute_event_statistics(
        metrics,
        num_permutations=args.num_permutations,
        seed=args.seed,
        strict=strict_stage4,
    )
    event_stats.setdefault("metadata", {})["use_sync"] = use_sync
    save_pickle(event_stats, out_paths["event_stats_path"])

    print("Computing stage-4 lag profiles...")
    lag_profiles = compute_event_lag_profiles(
        metrics,
        max_lag=args.max_lag,
        lead_window=args.lead_window,
        num_permutations=args.num_permutations,
        seed=args.seed,
        strict=strict_stage4,
    )
    lag_profiles.setdefault("metadata", {})["use_sync"] = use_sync
    save_pickle(lag_profiles, out_paths["lag_profiles_path"])

    print("Computing outcome diagnostics...")
    outcomes = compute_outcome_diagnostics(
        metrics,
        collection,
        win_return_threshold=args.win_return_threshold,
        strict=strict_outcomes,
    )
    outcomes.setdefault("metadata", {})["use_sync"] = use_sync
    save_pickle(outcomes, out_paths["outcomes_path"])

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
    print("Stage-4 event deltas (during - outside, permutation p):")
    for event_name, entry in event_stats["events"].items():
        if entry.get("status") != "ok":
            print(f"  {event_name}: {entry.get('status')} | {entry.get('message')}")
            continue
        print(
            f"  {event_name}: delta={entry['delta']:.4f}, "
            f"perm_p={entry['perm_p_one_sided']:.4g}, "
            f"n_during={entry['num_during']}, n_outside={entry['num_outside']}"
        )
    print("Stage-4 lag summaries (lead-lag delta, permutation p):")
    for event_name, entry in lag_profiles["events"].items():
        if entry.get("status") != "ok":
            print(f"  {event_name}: {entry.get('status')} | {entry.get('message')}")
            continue
        print(
            f"  {event_name}: lead-lag={entry['lead_minus_lag_delta']:.4f}, "
            f"perm_p={entry['lead_minus_lag_perm_p_one_sided']:.4g}, "
            f"events={entry['event_count']}"
        )
    print(
        "Outcome summary: "
        f"sync-return corr={outcomes['global']['sync_return_corr']:.4f}, "
        f"won-minus-lost sync={outcomes['global']['won_minus_lost_sync']:.4f}, "
        f"won={outcomes['global']['num_won']}, lost={outcomes['global']['num_lost']}"
    )

    if not args.no_plots:
        print("Writing quick-look figures...")
        plot_sync_timeseries(metrics, out_paths["figures_dir"], max_episodes=min(3, args.num_episodes))
        plot_pairwise_heatmap(metrics, out_paths["figures_dir"], episode_index=0)
        try:
            plot_event_delta_bars(event_stats, out_paths["figures_dir"])
        except ValueError as e:
            if strict_stage4:
                raise
            print(f"WARNING: skipped event delta plot in non-strict mode: {e}")
        try:
            plot_event_lag_profiles(lag_profiles, out_paths["figures_dir"])
        except ValueError as e:
            if strict_stage4:
                raise
            print(f"WARNING: skipped lag profile plot in non-strict mode: {e}")
        plot_sync_vs_outcome(outcomes, out_paths["figures_dir"])
        try:
            plot_win_loss_sync_trajectories(metrics, outcomes, out_paths["figures_dir"])
        except ValueError as e:
            if strict_outcomes:
                raise
            print(f"WARNING: skipped won/lost trajectory plot: {e}")
        try:
            plot_neuron_activation_heatmap(
                collection,
                out_paths["figures_dir"],
                episode_index=0,
                memory_reduction="last",
            )
        except (ValueError, KeyError, IndexError) as e:
            if strict_stage4:
                raise
            print(f"WARNING: skipped neuron activation heatmap in non-strict mode: {e}")

    print(f"Saved traces: {out_paths['episode_traces_path']}")
    print(f"Saved metrics: {out_paths['sync_metrics_path']}")
    print(f"Saved event stats: {out_paths['event_stats_path']}")
    print(f"Saved lag profiles: {out_paths['lag_profiles_path']}")
    print(f"Saved outcome diagnostics: {out_paths['outcomes_path']}")
    if not args.no_plots:
        print(f"Saved figures in: {out_paths['figures_dir']}")


if __name__ == "__main__":
    main()
