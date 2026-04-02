"""Evaluation script for trained MADDPG model.

Runs 10 episodes per shape (no augmentations) and saves the last episode 
of each shape as a GIF.

Usage:
    cd MARL-LLM/marl_llm
    python eval/eval_shapes.py
"""

import os
import sys

# JAX memory limit must be set before importing JAX
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.15'

import torch
import numpy as np
import random
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxmarl.environments.mpe.assembly import AssemblyEnv
from cus_gym.gym.wrappers.customized_envs.jax_assembly_wrapper_gpu import JaxAssemblyAdapterGPU
from algorithm.algorithms import MADDPG
from train.eval_render import save_eval_gif
from cfg.assembly_cfg import gpsargs as cfg

# ============================================================================
# CONFIGURATION - Set your weights path here
# ============================================================================
WEIGHTS_PATH = "./models/assembly/YOUR_RUN_NAME/model.pt"  # <-- UPDATE THIS

# Evaluation parameters
EPISODES_PER_SHAPE = 10
EPISODE_LENGTH = 200
SEED = 42
OUTPUT_DIR = "./eval_results"
# ============================================================================


def run_eval():
    """Run evaluation: 10 episodes per shape, save last episode GIF for each shape."""
    
    # Validate weights path
    weights_path = Path(WEIGHTS_PATH)
    if not weights_path.exists():
        print(f"ERROR: Weights file not found at: {weights_path}")
        print("Please update WEIGHTS_PATH at the top of this script.")
        sys.exit(1)
    
    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU available, running on CPU")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")
    
    # Initialize environment
    print("\nInitializing environment...")
    jax_env = AssemblyEnv(
        results_file=cfg.results_file,
        n_a=cfg.n_a,
    )
    env = JaxAssemblyAdapterGPU(
        jax_env,
        n_envs=1,
        seed=SEED,
        alpha=0.1,
    )
    
    num_shapes = env.num_shapes
    print(f"Number of shapes: {num_shapes}")
    print(f"Number of agents: {env.n_a}")
    
    # Load trained model
    print(f"\nLoading model from: {weights_path}")
    maddpg = MADDPG.init_from_save(str(weights_path))
    maddpg.prep_rollouts(device='cpu')  # Eval mode, deterministic policy
    
    start_stop_num = [slice(0, env.n_a)]
    
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
        
        for ep_idx in range(EPISODES_PER_SHAPE):
            is_last_episode = (ep_idx == EPISODES_PER_SHAPE - 1)
            state_history = [] if is_last_episode else None
            
            # Reset with specific shape (no augmentation)
            obs = env.reset_eval(shape_idx)
            ep_reward = 0.0
            
            with torch.no_grad():
                for step in range(EPISODE_LENGTH):
                    # Get deterministic action from policy
                    torch_obs = obs.cpu() if obs.is_cuda else obs
                    torch_obs = torch_obs.requires_grad_(False)
                    
                    actions, _ = maddpg.step(torch_obs, start_stop_num, explore=False)
                    actions_stacked = torch.column_stack(actions)  # (2, n_a)
                    
                    # Step environment
                    obs, rewards, dones, _, _ = env.step(actions_stacked.t().detach())
                    
                    # Accumulate reward (rewards is (1, n_a) GPU tensor)
                    ep_reward += rewards.mean().item()
                    
                    # Collect state for GIF (last episode only)
                    if is_last_episode:
                        state_history.append(env._states)
            
            # Record metrics
            avg_reward = ep_reward / EPISODE_LENGTH
            coverage = env.coverage_rate()
            dist_uniformity = env.distribution_uniformity()
            voronoi_uniformity = env.voronoi_based_uniformity()
            
            shape_rewards.append(avg_reward)
            shape_coverages.append(coverage)
            shape_dist_uniformities.append(dist_uniformity)
            shape_voronoi_uniformities.append(voronoi_uniformity)
            
            print(f"  Episode {ep_idx + 1}/{EPISODES_PER_SHAPE}: "
                  f"Reward={avg_reward:.4f}, Coverage={coverage:.3f}, "
                  f"Dist Uniformity={dist_uniformity:.3f}, Voronoi={voronoi_uniformity:.3f}")
            
            # Save GIF for last episode
            if is_last_episode:
                gif_path = output_dir / f"shape_{shape_idx:02d}.gif"
                save_eval_gif(state_history, gif_path, fps=12, frame_skip=2)
        
        # Compute shape statistics
        mean_reward = np.mean(shape_rewards)
        mean_coverage = np.mean(shape_coverages)
        mean_dist_uniformity = np.mean(shape_dist_uniformities)
        mean_voronoi = np.mean(shape_voronoi_uniformities)
        
        shape_result = {
            'shape_index': shape_idx,
            'mean_reward': mean_reward,
            'mean_coverage': mean_coverage,
            'mean_dist_uniformity': mean_dist_uniformity,
            'mean_voronoi_uniformity': mean_voronoi,
            'all_rewards': shape_rewards,
            'all_coverages': shape_coverages,
            'all_dist_uniformities': shape_dist_uniformities,
            'all_voronoi_uniformities': shape_voronoi_uniformities,
        }
        all_results.append(shape_result)
        
        print(f"\n  --- Shape {shape_idx} Average (over {EPISODES_PER_SHAPE} episodes) ---")
        print(f"  Reward:            {mean_reward:.4f}")
        print(f"  Coverage:          {mean_coverage:.3f}")
        print(f"  Dist Uniformity:   {mean_dist_uniformity:.3f}")
        print(f"  Voronoi:           {mean_voronoi:.3f}")
    
    # Print overall summary
    print("\n" + "="*100)
    print("EVALUATION COMPLETE")
    print("="*100)
    
    overall_rewards = [r['mean_reward'] for r in all_results]
    overall_coverages = [r['mean_coverage'] for r in all_results]
    overall_dist_uniformities = [r['mean_dist_uniformity'] for r in all_results]
    overall_voronoi = [r['mean_voronoi_uniformity'] for r in all_results]
    
    print(f"\n--- Overall Average (across {num_shapes} shapes) ---")
    print(f"  Reward:            {np.mean(overall_rewards):.4f}")
    print(f"  Coverage:          {np.mean(overall_coverages):.3f}")
    print(f"  Dist Uniformity:   {np.mean(overall_dist_uniformities):.3f}")
    print(f"  Voronoi:           {np.mean(overall_voronoi):.3f}")
    
    print(f"\nGIFs saved to: {output_dir.resolve()}/")
    for shape_idx in range(num_shapes):
        print(f"  - shape_{shape_idx:02d}.gif")
    
    # Save results to file
    import pickle
    results_path = output_dir / "eval_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'weights_path': str(weights_path),
            'episodes_per_shape': EPISODES_PER_SHAPE,
            'episode_length': EPISODE_LENGTH,
            'seed': SEED,
            'num_shapes': num_shapes,
            'results': all_results,
        }, f)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    run_eval()
