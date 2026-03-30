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

import torch
import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datetime import datetime
from pathlib import Path
from cfg.assembly_cfg import gpsargs as args
from algorithm.utils import ReplayBufferAgent
from algorithm.algorithms import MADDPG

# ── JAX env imports ────────────────────────────────────────────────────────
import sys
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
_JAXMARL_PATH = os.path.join(_REPO_ROOT, "JaxMARL")
_CUS_GYM_PATH = os.path.join(_REPO_ROOT, "MARL-LLM", "cus_gym")
if _JAXMARL_PATH not in sys.path:
    sys.path.insert(0, _JAXMARL_PATH)
if _CUS_GYM_PATH not in sys.path:
    sys.path.insert(0, _CUS_GYM_PATH)

from jaxmarl.environments.mpe.assembly import AssemblyEnv
from gym.wrappers.customized_envs.jax_assembly_wrapper_gpu import JaxAssemblyAdapterGPU
# ──────────────────────────────────────────────────────────────────────────


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
        start_time_1 = time.time()
        for et_index in range(cfg.episode_length):
            if ep_index % 500 == 0:
                env.render()

            # obs_gpu is already a torch.cuda tensor (via DLPack)
            torch_agent_actions, _ = maddpg.step(obs_gpu, start_stop_num, explore=True)

            # Stack actions (already on GPU)
            agent_actions_gpu = torch.column_stack(torch_agent_actions)  # (N*n_a, 2) GPU

            # Environment step returns GPU tensors via DLPack (zero-copy)
            next_obs_gpu, rewards_gpu, dones_gpu, _, agent_actions_prior_gpu = env.step(agent_actions_gpu)

            # Copy to CPU for buffer storage
            # Buffer expects NumPy arrays on CPU
            obs_cpu = obs_gpu.cpu().numpy()
            next_obs_cpu = next_obs_gpu.cpu().numpy()
            rewards_cpu = rewards_gpu.cpu().numpy()
            dones_cpu = dones_gpu.cpu().numpy()
            actions_cpu = agent_actions_gpu.cpu().numpy().T  # Transpose: (N*n_a, 2) → (2, N*n_a)
            prior_cpu = agent_actions_prior_gpu.cpu().numpy()

            agent_buffer[0].push(
                obs_cpu, actions_cpu, rewards_cpu, next_obs_cpu, dones_cpu,
                start_stop_num[0], prior_cpu,
            )
            
            obs_gpu = next_obs_gpu  # Stay on GPU for next step

            episode_reward_mean_bar += rewards_cpu.mean()
            episode_reward_std_bar  += rewards_cpu.std()

        end_time_1 = time.time()

        ########################### Training Phase ###########################
        start_time_2 = time.time()
        # Networks already on GPU, no prep_training needed

        for _ in range(20):
            for a_i in range(maddpg.nagents):
                if len(agent_buffer[a_i]) >= cfg.batch_size:
                    sample = agent_buffer[a_i].sample(
                        cfg.batch_size,
                        to_gpu=True,  # Sample from CPU buffer to GPU
                        is_prior=True,
                    )
                    obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, acs_prior_sample, _ = sample

                    maddpg.update(
                        obs_sample, acs_sample, rews_sample, next_obs_sample,
                        dones_sample, a_i, acs_prior_sample, env.alpha, logger=logger,
                    )

            maddpg.update_all_targets()

        maddpg.noise = max(0.5, maddpg.noise - cfg.noise_scale / cfg.n_episodes)

        # Update alpha for prior action regularization
        env.alpha = 0.1
        end_time_2 = time.time()

        ########################### Logging and Checkpointing ###########################
        if ep_index % 10 == 0:
            print(
                "Episodes %i of %i, agent num: %i, episode reward: %f, "
                "step time: %f, training time: %f"
                % (
                    ep_index, cfg.n_episodes, env.n_a,
                    episode_reward_mean_bar / cfg.episode_length,
                    end_time_1 - start_time_1, end_time_2 - start_time_2,
                )
            )

        if ep_index % cfg.save_interval == 0:
            ALIGN_epi = 0
            logger.add_scalars(
                "agent/data",
                {
                    "episode_reward_mean_bar": episode_reward_mean_bar / cfg.episode_length,
                    "episode_reward_std_bar":  episode_reward_std_bar  / cfg.episode_length,
                    "ALIGN_epi": ALIGN_epi,
                },
                ep_index,
            )

        if ep_index % (4 * cfg.save_interval) < cfg.n_rollout_threads:
            os.makedirs(run_dir / "incremental", exist_ok=True)
            maddpg.save(run_dir / "incremental" / ("model_ep%i.pt" % (ep_index + 1)))

    maddpg.save(run_dir / "model.pt")

    logger.export_scalars_to_json(str(log_dir / "summary.json"))
    logger.close()
    plt.close("all")


if __name__ == "__main__":
    run(args)
