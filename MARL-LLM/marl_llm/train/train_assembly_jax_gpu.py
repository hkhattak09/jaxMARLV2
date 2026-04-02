"""GPU-optimized training script for MADDPG with JAX environment using DLPack.

All data stays on GPU during rollout. Uses DLPack for zero-copy tensor sharing
between JAX and PyTorch on the same GPU device.

Key differences from train_assembly_jax.py:
  1. Policy networks stay on GPU during rollout (no prep_rollouts CPU switch)
  2. Environment returns PyTorch GPU tensors (via DLPack)
  3. No NumPy conversions during rollout loop
  4. Replay buffer handles GPU tensors (must copy to CPU for storage)

Expected speedup: 15-25% on rollout phase vs CPU version.

Requirements:
  - Single GPU with both JAX and PyTorch
  - JAX 0.7.2+ with CUDA support
  - PyTorch with CUDA enabled
  - CUDA device 0 (can be modified)
"""

# Configure JAX GPU memory BEFORE any imports
# JAX only runs environments (~20-100MB), PyTorch runs networks (needs most GPU memory)
# Limit JAX to 15% of 14GB T4 GPU (~2.1GB), leaving ~11.9GB for PyTorch
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.15'
print(f"[JAX Memory] Limited to 15% of GPU memory (~2.1GB on T4)")

import matplotlib
matplotlib.use('Agg')  # headless backend — must be set before any other matplotlib import
import matplotlib.pyplot as plt
import torch
import time
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime
from pathlib import Path

# ── sys.path setup (must be before local imports) ─────────────────────────
import sys
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
_MARL_LLM_PATH = os.path.join(_REPO_ROOT, "MARL-LLM", "marl_llm")
_JAXMARL_PATH = os.path.join(_REPO_ROOT, "JaxMARL")
_CUS_GYM_PATH = os.path.join(_REPO_ROOT, "MARL-LLM", "cus_gym")
for p in [_MARL_LLM_PATH, _JAXMARL_PATH, _CUS_GYM_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)
# ───────────────────────────────────────────────────────────────────────────

from cfg.assembly_cfg import gpsargs as args
from algorithm.utils import ReplayBufferAgent
from algorithm.algorithms import MADDPG
from jaxmarl.environments.mpe.assembly import AssemblyEnv
from gym.wrappers.customized_envs.jax_assembly_wrapper_gpu import JaxAssemblyAdapterGPU
from train.eval_render import save_eval_gif


def run_final_eval(maddpg, env, cfg, logger, run_dir):
    """Run final evaluation on each shape after training completes.
    
    For each shape (no random rotation/offset):
      - Runs 3 episodes
      - Averages metrics across the 3 episodes
      - Saves a GIF of the last episode
      
    Results are logged to TensorBoard under 'final_eval/shape_{i}'.
    """
    print("\n" + "="*80)
    print("FINAL EVALUATION - Testing on each shape")
    print("="*80)
    
    maddpg.prep_rollouts(device="gpu")
    start_stop_num = [slice(0, env.n_a)]
    num_shapes = env.num_shapes
    episodes_per_shape = 3
    
    # Store results for summary
    all_results = []
    
    with torch.no_grad():
        for shape_idx in range(num_shapes):
            shape_reward = 0.0
            shape_coverage = 0.0
            shape_uniformity = 0.0
            shape_voronoi = 0.0
            
            for ep_i in range(episodes_per_shape):
                # Reset with specific shape, no rotation/offset
                obs_gpu = env.reset_eval(shape_idx)
                ep_reward = 0.0
                is_last_ep = (ep_i == episodes_per_shape - 1)
                state_history = [] if is_last_ep else None
                
                for _ in range(cfg.episode_length):
                    torch_agent_actions, _ = maddpg.step(obs_gpu, start_stop_num, explore=False)
                    agent_actions_gpu = torch.column_stack(torch_agent_actions)
                    obs_gpu, rewards_gpu, _, _, _ = env.step(agent_actions_gpu.t().detach())
                    ep_reward += rewards_gpu.cpu().mean().item()
                    
                    if is_last_ep:
                        state_history.append(env._states)
                
                shape_reward += ep_reward / cfg.episode_length
                shape_coverage += env.coverage_rate()
                shape_uniformity += env.distribution_uniformity()
                shape_voronoi += env.voronoi_based_uniformity()
                
                # Save GIF of last episode for this shape
                if is_last_ep:
                    gif_path = run_dir / "final_eval" / f"shape_{shape_idx}.gif"
                    save_eval_gif(state_history, gif_path)
            
            # Average over episodes
            mean_reward = shape_reward / episodes_per_shape
            mean_coverage = shape_coverage / episodes_per_shape
            mean_uniformity = shape_uniformity / episodes_per_shape
            mean_voronoi = shape_voronoi / episodes_per_shape
            
            all_results.append({
                'shape': shape_idx,
                'reward': mean_reward,
                'coverage': mean_coverage,
                'uniformity': mean_uniformity,
                'voronoi': mean_voronoi,
            })
            
            print(f"Shape {shape_idx}: reward={mean_reward:.4f} | coverage={mean_coverage:.4f} | "
                  f"uniformity={mean_uniformity:.4f} | voronoi={mean_voronoi:.4f}")
            
            # Log to TensorBoard
            logger.add_scalars(
                f"final_eval/shape_{shape_idx}",
                {
                    "reward": mean_reward,
                    "coverage": mean_coverage,
                    "uniformity": mean_uniformity,
                    "voronoi": mean_voronoi,
                },
                0,  # Single data point
            )
    
    # Print summary
    print("\n" + "-"*80)
    print("FINAL EVAL SUMMARY (averaged across all shapes):")
    avg_reward = np.mean([r['reward'] for r in all_results])
    avg_coverage = np.mean([r['coverage'] for r in all_results])
    avg_uniformity = np.mean([r['uniformity'] for r in all_results])
    avg_voronoi = np.mean([r['voronoi'] for r in all_results])
    print(f"  Reward: {avg_reward:.4f} | Coverage: {avg_coverage:.4f} | "
          f"Uniformity: {avg_uniformity:.4f} | Voronoi: {avg_voronoi:.4f}")
    print("="*80 + "\n")
    
    # Log overall averages
    logger.add_scalars(
        "final_eval/overall",
        {
            "reward": avg_reward,
            "coverage": avg_coverage,
            "uniformity": avg_uniformity,
            "voronoi": avg_voronoi,
        },
        0,
    )
    
    # Restore training mode (in case user wants to continue)
    maddpg.prep_training(device="gpu")
    
    return all_results


def run_eval(maddpg, env, cfg, ep_index, logger):
    """Run evaluation episodes with no exploration noise or gradient tracking.

    Switches networks to eval mode for the duration, then restores training mode.
    Logs per-episode reward and environment quality metrics to TensorBoard.
    On the last eval episode, collects JAX states on GPU and saves a GIF in one
    bulk CPU transfer after the episode finishes.
    """
    maddpg.prep_rollouts(device="gpu")
    start_stop_num = [slice(0, env.n_a)]

    total_reward = 0.0
    total_coverage = 0.0
    total_uniformity = 0.0

    with torch.no_grad():
        for ep_i in range(cfg.eval_episodes):
            obs_gpu = env.reset()
            ep_reward = 0.0
            is_last_ep = (ep_i == cfg.eval_episodes - 1)
            state_history = [] if is_last_ep else None

            for _ in range(cfg.episode_length):
                torch_agent_actions, _ = maddpg.step(obs_gpu, start_stop_num, explore=False)
                agent_actions_gpu = torch.column_stack(torch_agent_actions)  # (2, N*n_a)
                # Transpose for env, detach for DLPack
                obs_gpu, rewards_gpu, _, _, _ = env.step(agent_actions_gpu.t().detach())
                ep_reward += rewards_gpu.cpu().mean().item()

                # Collect state snapshot on GPU — no transfer yet
                if is_last_ep:
                    state_history.append(env._states)

            total_reward    += ep_reward / cfg.episode_length
            total_coverage  += env.coverage_rate()
            total_uniformity += env.distribution_uniformity()

            # Bulk GPU→CPU transfer + GIF rendering after the episode is done
            if is_last_ep:
                gif_path = Path(cfg.gif_dir) / f"eval_{ep_index}.gif"
                save_eval_gif(state_history, gif_path)

    mean_reward     = total_reward    / cfg.eval_episodes
    mean_coverage   = total_coverage  / cfg.eval_episodes
    mean_uniformity = total_uniformity / cfg.eval_episodes

    print(
        f"[EVAL] ep {ep_index} | reward: {mean_reward:.4f} | "
        f"coverage: {mean_coverage:.4f} | uniformity: {mean_uniformity:.4f}"
    )
    logger.add_scalars(
        "eval",
        {
            "reward":      mean_reward,
            "coverage":    mean_coverage,
            "uniformity":  mean_uniformity,
        },
        ep_index,
    )

    # Restore training mode so the main loop can continue updating
    maddpg.prep_training(device="gpu")


def run(cfg):
    """Main training function for MADDPG with JAX environment (GPU-optimized)."""

    ## ======================================= Setup Logging =======================================
    model_dir = "./" / Path("./models") / cfg.env_name
    curr_run  = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir   = model_dir / curr_run
    log_dir   = run_dir / "logs"
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    ## ======================================= Initialize Environment =======================================
    # Verify GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-optimized version requires CUDA-enabled PyTorch")
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # ── Build JAX environment with GPU adapter ────────────────────────────
    jax_env = AssemblyEnv(
        results_file=cfg.results_file,
        n_a=cfg.n_a,
    )
    # GPU adapter returns PyTorch CUDA tensors via DLPack
    env = JaxAssemblyAdapterGPU(
        jax_env,
        n_envs=cfg.n_rollout_threads,
        seed=cfg.seed,
        alpha=1.0,
    )
    # ──────────────────────────────────────────────────────────────────────

    start_stop_num = [slice(0, env.num_agents)]

    # Initialize MADDPG algorithm
    adversary_alg = None
    maddpg = MADDPG.init_from_env(
        env,
        agent_alg=cfg.agent_alg,
        adversary_alg=adversary_alg,
        tau=cfg.tau,
        lr_actor=cfg.lr_actor,
        lr_critic=cfg.lr_critic,
        hidden_dim=cfg.hidden_dim,
        device="gpu",  # Force GPU for both rollout and training
        epsilon=cfg.epsilon,
        noise=cfg.noise_scale,
        name=cfg.env_name,
    )

    # Replay buffer (stores on CPU, samples to GPU for training)
    agent_buffer = [
        ReplayBufferAgent(
            cfg.buffer_length,
            env.num_agents,
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            start_stop_index=start_stop_num[0],
        )
    ]

    torch_agent_actions = []

    # Initialize all networks on GPU in training mode
    maddpg.prep_training(device="gpu")

    # Metric history for 10-episode rolling statistics
    metric_history = {
        "coverage": [],
        "uniformity": [],
        "voronoi": [],
        "reward_mean": [],
        "reward_std": [],
        "reward_uniformity": [],
        "vf_loss": [],
        "pol_loss": [],
        "reg_loss": [],
        "rollout_time": [],
        "policy_time": [],
        "env_time": [],
        "training_time": [],
    }

    ## ======================================= Training Loop =======================================
    print("Training Starts (GPU-optimized mode)...")
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):
        episode_reward_mean_bar = 0
        episode_reward_std_bar  = 0

        obs_gpu = env.reset()  # torch.cuda.FloatTensor (obs_dim, N*n_a)
        start_stop_num = [slice(0, env.n_a)]
        
        # Set networks to eval mode for rollout (keep on GPU)
        maddpg.prep_rollouts(device="gpu")
        maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
        maddpg.reset_noise()

        ########################### Episode Rollout ###########################
        # Accumulate on GPU, single bulk transfer at episode end
        obs_list = []
        actions_list = []
        rewards_list = []
        next_obs_list = []
        dones_list = []
        prior_list = []

        start_time_1 = time.time()
        policy_time = 0.0
        env_time = 0.0
        
        for et_index in range(cfg.episode_length):
            if ep_index % 500 == 0:
                env.render()

            # obs_gpu is already a torch.cuda tensor (via DLPack)
            t0 = time.time()
            torch_agent_actions, _ = maddpg.step(obs_gpu, start_stop_num, explore=True)
            agent_actions_gpu = torch.column_stack(torch_agent_actions)  # (2, N*n_a) GPU
            policy_time += time.time() - t0

            # DDPGAgent.step returns action.t() so agent_actions_gpu is (2, N*n_a).
            # env.step expects (N*n_a, 2); buffer.push expects (2, N*n_a) via [:, index].T.
            # detach() required for DLPack export (can't export tensors with gradients)
            t0 = time.time()
            next_obs_gpu, rewards_gpu, dones_gpu, _, agent_actions_prior_gpu = env.step(agent_actions_gpu.t().detach())
            env_time += time.time() - t0

            # Accumulate on GPU (no CPU transfer yet)
            obs_list.append(obs_gpu)
            actions_list.append(agent_actions_gpu.detach())
            rewards_list.append(rewards_gpu)
            next_obs_list.append(next_obs_gpu)
            dones_list.append(dones_gpu)
            prior_list.append(agent_actions_prior_gpu)
            
            obs_gpu = next_obs_gpu  # Stay on GPU for next step

        # Single bulk GPU→CPU transfer at episode end
        t0 = time.time()
        obs_batch = torch.stack(obs_list).cpu().numpy()           # (T, obs_dim, N*n_a)
        actions_batch = torch.stack(actions_list).cpu().numpy()   # (T, 2, N*n_a)
        rewards_batch = torch.stack(rewards_list).cpu().numpy()   # (T, N*n_a)
        next_obs_batch = torch.stack(next_obs_list).cpu().numpy() # (T, obs_dim, N*n_a)
        dones_batch = torch.stack(dones_list).cpu().numpy()       # (T, N*n_a)
        prior_batch = torch.stack(prior_list).cpu().numpy()       # (T, 2, N*n_a)
        transfer_time = time.time() - t0

        # Push all transitions to buffer
        for t in range(cfg.episode_length):
            agent_buffer[0].push(
                obs_batch[t], actions_batch[t], rewards_batch[t], 
                next_obs_batch[t], dones_batch[t],
                start_stop_num[0], prior_batch[t],
            )

        episode_reward_mean_bar = rewards_batch.mean() * cfg.episode_length
        episode_reward_std_bar = rewards_batch.std()  # Don't scale std - it's reported directly

        end_time_1 = time.time()

        ########################### Training Phase ###########################
        start_time_2 = time.time()
        # Switch networks to training mode (keep on GPU)
        maddpg.prep_training(device="gpu")
        
        total_vf_loss = 0.0
        total_pol_loss = 0.0
        total_reg_loss = 0.0
        update_count = 0

        for _ in range(20):
            for a_i in range(maddpg.nagents):
                if len(agent_buffer[a_i]) >= cfg.batch_size:
                    sample = agent_buffer[a_i].sample(
                        cfg.batch_size,
                        to_gpu=True,  # Sample from CPU buffer to GPU
                        is_prior=True,
                    )
                    obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, acs_prior_sample, _ = sample

                    vf_loss, pol_loss, reg_loss = maddpg.update(
                        obs_sample, acs_sample, rews_sample, next_obs_sample,
                        dones_sample, a_i, acs_prior_sample, env.alpha, logger=logger,
                    )
                    total_vf_loss += vf_loss
                    total_pol_loss += pol_loss
                    total_reg_loss += reg_loss
                    update_count += 1

            maddpg.update_all_targets()

        maddpg.noise = max(0.5, maddpg.noise - cfg.noise_scale / cfg.n_episodes)
        avg_vf_loss = total_vf_loss / max(update_count, 1)
        avg_pol_loss = total_pol_loss / max(update_count, 1)
        avg_reg_loss = total_reg_loss / max(update_count, 1)

        # Update alpha for prior action regularization
        env.alpha = 0.1
        end_time_2 = time.time()

        ########################### Logging and Checkpointing ###########################
        # Compute end-of-episode metrics
        coverage = env.coverage_rate()
        uniformity = env.distribution_uniformity()
        voronoi_uniformity = env.voronoi_based_uniformity()
        avg_reward = episode_reward_mean_bar / cfg.episode_length
        reward_uniformity = 1.0 / (1.0 + episode_reward_std_bar / (abs(episode_reward_mean_bar) + 1e-8))
        
        # Track metrics for rolling statistics
        metric_history["coverage"].append(coverage)
        metric_history["uniformity"].append(uniformity)
        metric_history["voronoi"].append(voronoi_uniformity)
        metric_history["reward_mean"].append(avg_reward)
        metric_history["reward_std"].append(episode_reward_std_bar)
        metric_history["reward_uniformity"].append(reward_uniformity)
        metric_history["vf_loss"].append(avg_vf_loss)
        metric_history["pol_loss"].append(avg_pol_loss)
        metric_history["reg_loss"].append(avg_reg_loss)
        metric_history["rollout_time"].append(end_time_1 - start_time_1)
        metric_history["policy_time"].append(policy_time)
        metric_history["env_time"].append(env_time)
        metric_history["training_time"].append(end_time_2 - start_time_2)
        # Keep only last 10 episodes
        for k in metric_history:
            if len(metric_history[k]) > 10:
                metric_history[k] = metric_history[k][-10:]
        
        if ep_index % 10 == 0:
            # Compute mean and std over last 10 episodes (or fewer if just starting)
            cov_mean, cov_std = np.mean(metric_history["coverage"]), np.std(metric_history["coverage"])
            uni_mean, uni_std = np.mean(metric_history["uniformity"]), np.std(metric_history["uniformity"])
            vor_mean, vor_std = np.mean(metric_history["voronoi"]), np.std(metric_history["voronoi"])
            
            # Compute 10-episode averages for rewards, losses, and timing
            avg_reward_10 = np.mean(metric_history["reward_mean"])
            avg_reward_std_10 = np.mean(metric_history["reward_std"])
            avg_reward_uniformity_10 = np.mean(metric_history["reward_uniformity"])
            avg_vf_loss_10 = np.mean(metric_history["vf_loss"])
            avg_pol_loss_10 = np.mean(metric_history["pol_loss"])
            avg_reg_loss_10 = np.mean(metric_history["reg_loss"])
            avg_rollout_time_10 = np.mean(metric_history["rollout_time"])
            avg_policy_time_10 = np.mean(metric_history["policy_time"])
            avg_env_time_10 = np.mean(metric_history["env_time"])
            avg_training_time_10 = np.mean(metric_history["training_time"])
            
            sep = "=" * 100
            print(f"\n{sep}")
            print(f"Episode {ep_index:5d}/{cfg.n_episodes:5d} | Agents: {env.n_a}")
            print(sep)
            print(f"REWARDS (last 10 eps):  Mean: {avg_reward_10:7.4f} | Std: {avg_reward_std_10:7.4f} | Uniformity: {avg_reward_uniformity_10:6.3f}")
            print(f"ENVIRONMENT METRICS (last 10 eps):")
            print(f"  - Coverage: {cov_mean:.3f}(std:{cov_std:.3f}) | Dist Uniformity: {uni_mean:.3f}(std:{uni_std:.3f}) | Voronoi Uniformity: {vor_mean:.3f}(std:{vor_std:.3f})")
            print(f"LOSSES (last 10 eps):   VF: {avg_vf_loss_10:7.4f} | Policy: {avg_pol_loss_10:7.4f} | Reg: {avg_reg_loss_10:7.4f}")
            print(f"TIMING (last 10 eps):   Rollout: {avg_rollout_time_10:6.2f} | Policy Exec: {avg_policy_time_10:6.2f} | Env Step: {avg_env_time_10:6.2f} | Training: {avg_training_time_10:6.2f}")
            print(f"{sep}\n")

        if ep_index % cfg.save_interval == 0:
            logger.add_scalars(
                "agent/data",
                {
                    "episode_reward": avg_reward,
                    "coverage": coverage,
                    "nn_uniformity": uniformity,
                    "voronoi_uniformity": voronoi_uniformity,
                    "vf_loss": avg_vf_loss,
                    "pol_loss": avg_pol_loss,
                    "noise": maddpg.noise,
                },
                ep_index,
            )

        if ep_index % (4 * cfg.save_interval) < cfg.n_rollout_threads:
            os.makedirs(run_dir / "incremental", exist_ok=True)
            maddpg.save(run_dir / "incremental" / ("model_ep%i.pt" % (ep_index + 1)))
            # Restore all networks to GPU after save (save moves them to CPU)
            maddpg.prep_training(device="gpu")

        if ep_index > 0 and ep_index % cfg.eval_interval < cfg.n_rollout_threads:
            run_eval(maddpg, env, cfg, ep_index, logger)

    # ========================== Final Evaluation ==========================
    # Run evaluation on each shape (no rotation/offset), 3 episodes each
    run_final_eval(maddpg, env, cfg, logger, run_dir)

    maddpg.save(run_dir / "model.pt")

    logger.export_scalars_to_json(str(log_dir / "summary.json"))
    logger.close()
    plt.close("all")


if __name__ == "__main__":
    run(args)
