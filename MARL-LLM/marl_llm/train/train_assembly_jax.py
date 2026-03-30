"""Training script for MADDPG on AssemblySwarm using the JAX environment.

Drop-in replacement for train_assembly.py. Changes from the original:

  1. Env construction: JAX AssemblyEnv + JaxAssemblyAdapter instead of
     C++ AssemblySwarmEnv + AssemblySwarmWrapper  (~3 lines).
  2. Action shape fix: maddpg.step returns (n_a, 2) but the replay buffer
     expects (2, n_a) layout — add a .T before buffer push  (1 line).
  3. Alpha update: env.alpha instead of env.env.alpha  (1 line).

Everything else — MADDPG, buffer, LLM regularisation, logging — is untouched.

Memory note for n_rollout_threads > 1:
  Each parallel env adds n_a agents to the flat buffer.  With N envs the
  buffer occupies  buffer_length × N × n_a × obs_dim × 4 bytes.  For
  N=10, n_a=30, obs_dim=192 that is ~4.6 GB.  Reduce buffer_length or N
  if GPU/CPU memory is limited.
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
_REPO_ROOT = str(Path(__file__).resolve().parents[3])          # …/new_marl_llm_implementation
_JAXMARL_PATH = os.path.join(_REPO_ROOT, "JaxMARL")
_CUS_GYM_PATH = os.path.join(_REPO_ROOT, "MARL-LLM", "cus_gym")
if _JAXMARL_PATH not in sys.path:
    sys.path.insert(0, _JAXMARL_PATH)
if _CUS_GYM_PATH not in sys.path:
    sys.path.insert(0, _CUS_GYM_PATH)

from jaxmarl.environments.mpe.assembly import AssemblyEnv
from gym.wrappers import JaxAssemblyAdapter
# ──────────────────────────────────────────────────────────────────────────


def run(cfg):
    """Main training function for MADDPG with JAX environment."""

    ## ======================================= Setup Logging =======================================
    model_dir = "./" / Path("./models") / cfg.env_name
    curr_run  = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir   = model_dir / curr_run
    log_dir   = run_dir / "logs"
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    ## ======================================= Initialize Environment =======================================
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if cfg.device == "cpu":
        torch.set_num_threads(cfg.n_training_threads)
    elif cfg.device == "gpu":
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    # ── Build JAX environment ──────────────────────────────────────────────
    jax_env = AssemblyEnv(
        results_file=cfg.results_file,
        n_a=cfg.n_a,
    )
    # n_rollout_threads controls how many envs run in parallel via vmap
    env = JaxAssemblyAdapter(
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
        device=cfg.device,
        epsilon=cfg.epsilon,
        noise=cfg.noise_scale,
        name=cfg.env_name,
    )

    # Replay buffer — num_agents scales with n_rollout_threads
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
    print("Training Starts...")
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):
        episode_reward_mean_bar = 0
        episode_reward_std_bar  = 0

        obs = env.reset()                                  # (obs_dim, N*n_a)
        start_stop_num = [slice(0, env.n_a)]
        maddpg.prep_rollouts(device="cpu")

        maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
        maddpg.reset_noise()

        ########################### Episode Rollout ###########################
        start_time_1 = time.time()
        for et_index in range(cfg.episode_length):
            if ep_index % 500 == 0:
                env.render()

            torch_obs = torch.Tensor(obs).requires_grad_(False)
            torch_agent_actions, _ = maddpg.step(torch_obs, start_stop_num, explore=True)

            # maddpg.step returns a list of (N*n_a, 2) tensors (one per agent group).
            # np.column_stack gives (N*n_a, 2) — format our adapter accepts directly.
            agent_actions = np.column_stack(
                [ac.data.numpy() for ac in torch_agent_actions]
            )  # (N*n_a, 2)

            next_obs, rewards, dones, _, agent_actions_prior = env.step(agent_actions)

            # Buffer expects (act_dim, N*n_a) layout for the [:, index].T slicing
            # inside ReplayBufferAgent.push — transpose the (N*n_a, 2) actions.
            agent_actions_buf = agent_actions.T  # (2, N*n_a)

            agent_buffer[0].push(
                obs, agent_actions_buf, rewards, next_obs, dones,
                start_stop_num[0], agent_actions_prior,
            )
            obs = next_obs

            episode_reward_mean_bar += np.mean(rewards)
            episode_reward_std_bar  += np.std(rewards)

        end_time_1 = time.time()

        ########################### Training Phase ###########################
        start_time_2 = time.time()
        maddpg.prep_training(device=cfg.device)

        for _ in range(20):
            for a_i in range(maddpg.nagents):
                if len(agent_buffer[a_i]) >= cfg.batch_size:
                    sample = agent_buffer[a_i].sample(
                        cfg.batch_size,
                        to_gpu=True if cfg.device == "gpu" else False,
                        is_prior=True,
                    )
                    obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, acs_prior_sample, _ = sample

                    maddpg.update(
                        obs_sample, acs_sample, rews_sample, next_obs_sample,
                        dones_sample, a_i, acs_prior_sample, env.alpha, logger=logger,
                    )

            maddpg.update_all_targets()

        maddpg.prep_rollouts(device="cpu")

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

    maddpg.prep_training(device=cfg.device)
    maddpg.save(run_dir / "model.pt")

    logger.export_scalars_to_json(str(log_dir / "summary.json"))
    logger.close()
    plt.close("all")


if __name__ == "__main__":
    run(args)
