"""Evaluation script for trained MADDPG model.

Runs evaluation episodes per shape (no augmentations) for one or more weight files
and saves the last episode of each shape as a GIF.

Usage:
    python /path/to/eval_shapes.py

Configuration:
    Edit WEIGHTS_PATHS below to point to your model weights.
"""

# Configure JAX GPU memory BEFORE any imports
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.15'

import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================
WEIGHTS_PATHS = [
    "/content/cpp_model.pt",
]
EPISODES_PER_SHAPE = 10  # cfg.eval_episodes is 3, using 10 for thorough eval
OUTPUT_DIR = "./eval_results"
# Other values from cfg: cfg.episode_length, cfg.seed
# ============================================================================


# ── sys.path setup (must be before local imports) ─────────────────────────
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
_MARL_LLM_PATH = os.path.join(_REPO_ROOT, "MARL-LLM", "marl_llm")
_JAXMARL_PATH = os.path.join(_REPO_ROOT, "JaxMARL")
_CUS_GYM_PATH = os.path.join(_REPO_ROOT, "MARL-LLM", "cus_gym")
for p in [_MARL_LLM_PATH, _JAXMARL_PATH, _CUS_GYM_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)
# ───────────────────────────────────────────────────────────────────────────

import torch
import numpy as np
import random

from jaxmarl.environments.mpe.assembly import AssemblyEnv
from gym.wrappers.customized_envs.jax_assembly_wrapper_gpu import JaxAssemblyAdapterGPU
from algorithm.algorithms import MADDPG
from train.eval_render import save_eval_gif
from cfg.assembly_cfg import gpsargs as cfg


def _sanitize_model_name(weights_path: Path) -> str:
    """Create a folder-safe model name from weight file stem."""
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in weights_path.stem)
    return sanitized or "model"


def _build_comparison_table(model_summaries):
    """Build a readable metrics-vs-models text table."""
    headers = ["Metric"] + [summary["model_name"] for summary in model_summaries]
    metric_rows = [
        ("Reward", "overall_reward", ".4f"),
        ("Sensing Coverage", "overall_coverage", ".3f"),
        ("Dist Uniformity", "overall_dist_uniformity", ".3f"),
        ("Voronoi Uniformity", "overall_voronoi_uniformity", ".3f"),
        ("Neighbor Dist", "overall_neighbor_dist", ".4f"),
        ("R-Avoid Violations", "overall_r_avoid_violations", ".1f"),
        ("Spring Collisions", "overall_spring_collisions", ".1f"),
    ]

    rows = []
    for label, key, fmt in metric_rows:
        row = [label] + [format(summary[key], fmt) for summary in model_summaries]
        rows.append(row)

    col_widths = [len(h) for h in headers]
    for row in rows:
        col_widths = [max(w, len(cell)) for w, cell in zip(col_widths, row)]

    def format_row(cells):
        return " | ".join(cell.ljust(width) for cell, width in zip(cells, col_widths))

    separator = "-+-".join("-" * width for width in col_widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def _evaluate_single_model(weights_path, model_output_dir, env, num_shapes, start_stop_num):
    """Evaluate one weight file and save per-shape GIFs/results in model_output_dir."""
    print("\n" + "="*100)
    print(f"EVALUATING MODEL: {weights_path.name}")
    print("="*100)

    # Set random seeds for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU available, running on CPU")
    
    # Load trained model
    print(f"\nLoading model from: {weights_path}")
    maddpg = MADDPG.init_from_save(str(weights_path))
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    maddpg.prep_rollouts(device=device)  # Eval mode, deterministic policy
    print(f"Saving model outputs to: {model_output_dir.resolve()}")

    # Results storage
    all_results = []

    print("\n" + "="*100)
    print("EVALUATION PHASE STARTING")
    print("="*100)
    print(f"Running {EPISODES_PER_SHAPE} evaluation episodes per shape (no offsets, no rotation)")
    print(f"Number of shapes to evaluate: {num_shapes}")
    print("="*100 + "\n")
    
    # Evaluate each shape
    for shape_idx in range(num_shapes):
        print("\n" + "="*80)
        print(f"Evaluating Shape {shape_idx}")
        print("="*80)
        
        shape_rewards = []
        shape_coverages = []
        shape_dist_uniformities = []
        shape_voronoi_uniformities = []
        shape_neighbor_dists = []
        shape_r_avoid_violations = []
        shape_spring_collisions = []

        for ep_idx in range(EPISODES_PER_SHAPE):
            is_last_episode = (ep_idx == EPISODES_PER_SHAPE - 1)
            state_history = [] if is_last_episode else None

            # Reset with specific shape (no augmentation)
            obs = env.reset_eval(shape_idx)
            ep_reward = 0.0
            ep_spring_collisions = 0.0
            prev_prior_eval = None  # one-step-lag prior for seed mode

            with torch.no_grad():
                for step in range(cfg.episode_length):
                    # Stateless rollout: fresh hidden state every step.
                    # Seed mode: one-step-lag prior seeds CTM (zeros at t=0).
                    if maddpg.use_ctm_actor:
                        if maddpg.prior_mode == 'seed' and prev_prior_eval is not None:
                            eval_hidden = maddpg.agents[0].policy.get_prior_seeded_hidden_state(
                                obs.t(), prev_prior_eval.t()
                            )
                        else:
                            eval_hidden = maddpg.agents[0].policy.get_initial_hidden_state(
                                env.n_a, torch_device
                            )
                    else:
                        eval_hidden = None
                    actions, _, _ = maddpg.step(obs, start_stop_num, explore=False, hidden_states=eval_hidden)
                    actions_stacked = torch.column_stack(actions)  # (2, n_a)

                    # Step environment
                    obs, rewards, dones, _, prior_gpu = env.step(actions_stacked.t().detach())
                    prev_prior_eval = prior_gpu.detach()

                    # Accumulate reward (rewards is (1, n_a) GPU tensor)
                    ep_reward += rewards.mean().item()
                    ep_spring_collisions += env.springboard_collision_count()

                    # Collect state for GIF (last episode only)
                    if is_last_episode:
                        state_history.append(env._states)

            # Record metrics
            avg_reward = ep_reward / cfg.episode_length
            coverage = env.sensing_coverage()
            dist_uniformity = env.distribution_uniformity()
            voronoi_uniformity = env.voronoi_based_uniformity()
            neighbor_dist = env.mean_neighbor_distance()
            r_avoid_violations = env.r_avoid_violation_count()

            shape_rewards.append(avg_reward)
            shape_coverages.append(coverage)
            shape_dist_uniformities.append(dist_uniformity)
            shape_voronoi_uniformities.append(voronoi_uniformity)
            shape_neighbor_dists.append(neighbor_dist)
            shape_r_avoid_violations.append(r_avoid_violations)
            shape_spring_collisions.append(ep_spring_collisions)

            print(f"  Episode {ep_idx + 1}/{EPISODES_PER_SHAPE}: "
                  f"Reward={avg_reward:.4f}, Coverage={coverage:.3f}, "
                  f"Dist Uniformity={dist_uniformity:.3f}, Voronoi={voronoi_uniformity:.3f}, "
                  f"Neighbor Dist={neighbor_dist:.4f}, R-Avoid Violations={r_avoid_violations:.0f}, "
                  f"Spring Collisions={ep_spring_collisions:.0f}")

            # Save GIF for last episode
            if is_last_episode:
                gif_path = model_output_dir / f"shape_{shape_idx:02d}.gif"
                save_eval_gif(state_history, gif_path, fps=12, frame_skip=2,
                              size_a=env.size_a, d_sen=env.d_sen, r_avoid=env.r_avoid)

        # Compute shape statistics
        mean_reward = np.mean(shape_rewards)
        mean_coverage = np.mean(shape_coverages)
        mean_dist_uniformity = np.mean(shape_dist_uniformities)
        mean_voronoi = np.mean(shape_voronoi_uniformities)
        mean_neighbor_dist = np.mean(shape_neighbor_dists)
        mean_r_avoid_violations = np.mean(shape_r_avoid_violations)
        mean_spring_collisions = np.mean(shape_spring_collisions)

        shape_result = {
            'shape_index': shape_idx,
            'mean_reward': mean_reward,
            'mean_coverage': mean_coverage,
            'mean_dist_uniformity': mean_dist_uniformity,
            'mean_voronoi_uniformity': mean_voronoi,
            'mean_neighbor_dist': mean_neighbor_dist,
            'mean_r_avoid_violations': mean_r_avoid_violations,
            'mean_spring_collisions': mean_spring_collisions,
            'all_rewards': shape_rewards,
            'all_coverages': shape_coverages,
            'all_dist_uniformities': shape_dist_uniformities,
            'all_voronoi_uniformities': shape_voronoi_uniformities,
            'all_neighbor_dists': shape_neighbor_dists,
            'all_r_avoid_violations': shape_r_avoid_violations,
            'all_spring_collisions': shape_spring_collisions,
        }
        all_results.append(shape_result)

        print(f"\n  --- Shape {shape_idx} Average (over {EPISODES_PER_SHAPE} episodes) ---")
        print(f"  Reward:              {mean_reward:.4f}")
        print(f"  Coverage:            {mean_coverage:.3f}")
        print(f"  Dist Uniformity:     {mean_dist_uniformity:.3f}")
        print(f"  Voronoi:             {mean_voronoi:.3f}")
        print(f"  Neighbor Dist:       {mean_neighbor_dist:.4f}")
        print(f"  R-Avoid Violations:  {mean_r_avoid_violations:.1f}")
        print(f"  Spring Collisions:   {mean_spring_collisions:.1f}")
    
    # Print overall summary
    print("\n" + "="*100)
    print("EVALUATION COMPLETE")
    print("="*100)
    
    overall_rewards = [r['mean_reward'] for r in all_results]
    overall_coverages = [r['mean_coverage'] for r in all_results]
    overall_dist_uniformities = [r['mean_dist_uniformity'] for r in all_results]
    overall_voronoi = [r['mean_voronoi_uniformity'] for r in all_results]
    overall_neighbor_dists = [r['mean_neighbor_dist'] for r in all_results]
    overall_r_avoid_violations = [r['mean_r_avoid_violations'] for r in all_results]
    overall_spring_collisions = [r['mean_spring_collisions'] for r in all_results]

    print(f"\n--- Overall Average (across {num_shapes} shapes) ---")
    overall_reward = float(np.mean(overall_rewards))
    overall_coverage = float(np.mean(overall_coverages))
    overall_dist_uniformity = float(np.mean(overall_dist_uniformities))
    overall_voronoi_uniformity = float(np.mean(overall_voronoi))
    overall_neighbor_dist = float(np.mean(overall_neighbor_dists))
    overall_r_avoid_violations = float(np.mean(overall_r_avoid_violations))
    overall_spring_collision = float(np.mean(overall_spring_collisions))

    print(f"  Reward:              {overall_reward:.4f}")
    print(f"  Coverage:            {overall_coverage:.3f}")
    print(f"  Dist Uniformity:     {overall_dist_uniformity:.3f}")
    print(f"  Voronoi:             {overall_voronoi_uniformity:.3f}")
    print(f"  Neighbor Dist:       {overall_neighbor_dist:.4f}")
    print(f"  R-Avoid Violations:  {overall_r_avoid_violations:.1f}")
    print(f"  Spring Collisions:   {overall_spring_collision:.1f}")

    print(f"\nGIFs saved to: {model_output_dir.resolve()}/")
    for shape_idx in range(num_shapes):
        print(f"  - shape_{shape_idx:02d}.gif")

    # Save results to file
    import pickle
    results_path = model_output_dir / "eval_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'weights_path': str(weights_path),
            'episodes_per_shape': EPISODES_PER_SHAPE,
            'episode_length': cfg.episode_length,
            'seed': cfg.seed,
            'num_shapes': num_shapes,
            'overall_reward': overall_reward,
            'overall_coverage': overall_coverage,
            'overall_dist_uniformity': overall_dist_uniformity,
            'overall_voronoi_uniformity': overall_voronoi_uniformity,
            'overall_neighbor_dist': overall_neighbor_dist,
            'overall_r_avoid_violations': overall_r_avoid_violations,
            'overall_spring_collisions': overall_spring_collision,
            'results': all_results,
        }, f)
    print(f"\nResults saved to: {results_path}")

    return {
        'weights_path': str(weights_path),
        'model_name': model_output_dir.name,
        'overall_reward': overall_reward,
        'overall_coverage': overall_coverage,
        'overall_dist_uniformity': overall_dist_uniformity,
        'overall_voronoi_uniformity': overall_voronoi_uniformity,
        'overall_neighbor_dist': overall_neighbor_dist,
        'overall_r_avoid_violations': overall_r_avoid_violations,
        'overall_spring_collisions': overall_spring_collision,
        'results_path': str(results_path),
    }


def run_eval():
    """Run evaluation for each weight file and print a cross-model comparison table."""
    if not WEIGHTS_PATHS:
        print("ERROR: WEIGHTS_PATHS is empty. Please provide at least one .pt file path.")
        sys.exit(1)

    # Validate all weights paths before starting long evaluations
    weights_paths = [Path(p) for p in WEIGHTS_PATHS]
    missing_paths = [p for p in weights_paths if not p.exists()]
    if missing_paths:
        print("ERROR: The following weights files were not found:")
        for missing in missing_paths:
            print(f"  - {missing}")
        print("Please update WEIGHTS_PATHS at the top of this script.")
        sys.exit(1)

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base output directory: {output_dir.resolve()}")

    # Initialize environment once and reuse for all models
    print("\nInitializing environment...")
    jax_env = AssemblyEnv(
        results_file=cfg.results_file,
        n_a=cfg.n_a,
        topo_nei_max=cfg.topo_nei_max,
        grid_obs_fraction=cfg.grid_obs_fraction,
        d_sen=cfg.d_sen,
        r_avoid=cfg.r_avoid,
    )
    env = JaxAssemblyAdapterGPU(
        jax_env,
        n_envs=1,
        seed=cfg.seed,
        alpha=0.1,
    )

    num_shapes = env.num_shapes
    print(f"Number of shapes: {num_shapes}")
    print(f"Number of agents: {env.n_a}")

    start_stop_num = [slice(0, env.n_a)]
    used_model_dir_names = set()
    model_summaries = []

    for weights_path in weights_paths:
        model_dir_name = _sanitize_model_name(weights_path)
        if model_dir_name in used_model_dir_names:
            suffix = 2
            while f"{model_dir_name}_{suffix}" in used_model_dir_names:
                suffix += 1
            model_dir_name = f"{model_dir_name}_{suffix}"

        used_model_dir_names.add(model_dir_name)
        model_output_dir = output_dir / model_dir_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        summary = _evaluate_single_model(
            weights_path=weights_path,
            model_output_dir=model_output_dir,
            env=env,
            num_shapes=num_shapes,
            start_stop_num=start_stop_num,
        )
        model_summaries.append(summary)

    print("\n" + "="*100)
    print("FINAL MODEL COMPARISON (AVERAGE ACROSS SHAPES)")
    print("="*100)
    comparison_table = _build_comparison_table(model_summaries)
    print(comparison_table)

    comparison_table_path = output_dir / "comparison_table.txt"
    comparison_table_path.write_text(comparison_table + "\n", encoding="utf-8")
    print(f"\nComparison table saved to: {comparison_table_path}")


if __name__ == '__main__':
    run_eval()
