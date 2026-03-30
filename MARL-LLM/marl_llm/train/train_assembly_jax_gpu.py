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

    # Set all networks to train mode and ensure they are on GPU
    maddpg.prep_training(device="gpu")

    ## ======================================= Training Loop =======================================
    print("Training Starts (GPU-optimized mode)...")
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):
        episode_reward_mean_bar = 0
        episode_reward_std_bar  = 0

        obs_gpu = env.reset()  # torch.cuda.FloatTensor (obs_dim, N*n_a)
        start_stop_num = [slice(0, env.n_a)]
        
        # Keep networks on GPU for rollout (no prep_rollouts CPU switch)
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
            torch.cuda.synchronize()  # Ensure timing is accurate
            policy_time += time.time() - t0

            # DDPGAgent.step returns action.t() so agent_actions_gpu is (2, N*n_a).
            # env.step expects (N*n_a, 2); buffer.push expects (2, N*n_a) via [:, index].T.
            # detach() required for DLPack export (can't export tensors with gradients)
            t0 = time.time()
            next_obs_gpu, rewards_gpu, dones_gpu, _, agent_actions_prior_gpu = env.step(agent_actions_gpu.t().detach())
            torch.cuda.synchronize()
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
        
        if ep_index % 10 == 0:
            print(f"  [TIMING] policy: {policy_time:.2f}s | env: {env_time:.2f}s | transfer: {transfer_time:.2f}s")

        # Push all transitions to buffer
        for t in range(cfg.episode_length):
            agent_buffer[0].push(
                obs_batch[t], actions_batch[t], rewards_batch[t], 
                next_obs_batch[t], dones_batch[t],
                start_stop_num[0], prior_batch[t],
            )

        episode_reward_mean_bar = rewards_batch.mean() * cfg.episode_length
        episode_reward_std_bar = rewards_batch.std() * cfg.episode_length

        end_time_1 = time.time()

        ########################### Training Phase ###########################
        start_time_2 = time.time()
        # Networks already on GPU, no prep_training needed
        
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
        
        if ep_index % 10 == 0:
            sep = "=" * 100
            print(f"\n{sep}")
            print(f"Episode {ep_index:5d}/{cfg.n_episodes:5d} | Agents: {env.n_a}")
            print(sep)
            print(f"REWARDS:            Mean: {avg_reward:7.4f} | Std: {episode_reward_std_bar:7.4f} | Uniformity: {reward_uniformity:6.3f}")
            print(f"ENVIRONMENT METRICS:")
            print(f"  - Coverage Rate: {coverage:6.3f} | Dist Uniformity (NN): {uniformity:6.3f} | Voronoi Uniformity: {voronoi_uniformity:7.3f}")
            print(f"LOSSES:             VF: {avg_vf_loss:7.4f} | Policy: {avg_pol_loss:7.4f} | Reg: {avg_reg_loss:7.4f}")
            print(f"TIMING (sec):       Rollout: {end_time_1 - start_time_1:6.2f} | Policy Exec: {policy_time:6.2f} | Env Step: {env_time:6.2f} | Training: {end_time_2 - start_time_2:6.2f}")
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

        if ep_index > 0 and ep_index % cfg.eval_interval < cfg.n_rollout_threads:
            run_eval(maddpg, env, cfg, ep_index, logger)

    maddpg.save(run_dir / "model.pt")

    logger.export_scalars_to_json(str(log_dir / "summary.json"))
    logger.close()
    plt.close("all")


if __name__ == "__main__":
    run(args)
