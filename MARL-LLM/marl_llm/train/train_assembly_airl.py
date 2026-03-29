import torch
import time
import os
import random
import numpy as np
import gym
from gym.wrappers import AssemblySwarmWrapper
from cfg.assembly_cfg import gpsargs as args
from pathlib import Path
from tensorboardX import SummaryWriter
from datetime import datetime
from algorithm.utils import ReplayBufferAgent, ReplayBufferExpert
from algorithm.algorithms import AIRL, MADDPG

def run(cfg):
    """
    Main training function for AIRL (Adversarial Inverse Reinforcement Learning) with MADDPG.
    This function implements the complete AIRL training pipeline:
    1. Loads expert demonstrations
    2. Trains a discriminator to distinguish expert vs policy actions
    3. Uses discriminator rewards to train MADDPG agents
    4. Alternates between discriminator and policy updates
    
    Args:
        cfg: Configuration object containing training parameters.
    """
    ## ======================================= Setup Logging =======================================
    # Create timestamped directories for model saving and logging
    model_dir = './' / Path('./models') / cfg.env_name 
    curr_run = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = model_dir / curr_run   
    log_dir = run_dir / 'logs'    
    os.makedirs(log_dir)    
    logger = SummaryWriter(str(log_dir))  # TensorBoard logging

    ## ======================================= Initialize =======================================
    # Set random seeds for reproducible training
    torch.manual_seed(cfg.seed)  
    np.random.seed(cfg.seed) 
    random.seed(cfg.seed)  
    
    # Configure PyTorch based on device
    if cfg.device == 'cpu':
        torch.set_num_threads(cfg.n_training_threads) 
    elif cfg.device == 'gpu':
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    # Initialize assembly swarm environment
    scenario_name = 'AssemblySwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = AssemblySwarmWrapper(base_env, args)
    start_stop_num = [slice(0, env.num_agents)]  # Multi-agent indexing

    ## ======================================= Setup Buffers =======================================
    # Load expert demonstration data
    model_dir = './' / Path('./eval') / 'expert_data' 
    curr_run = 'your_run_name'  # Replace with actual expert data run name, e.g., '2025-01-19-11-48-15'
    expert_dir = os.path.join(model_dir, curr_run, 'expert_data.npz') 
    
    # Initialize expert buffer and load demonstration data
    expert_buffer = ReplayBufferExpert(
        cfg.buffer_length, env.num_agents, 
        state_dim=env.observation_space.shape[0], 
        action_dim=env.action_space.shape[0], 
        start_stop_index=start_stop_num[0]
    )
    expert_buffer.load(expert_dir)
    expert_buffer.total_length = expert_buffer.obs_buffs.shape[0]
    
    # Initialize agent experience buffer for policy learning
    agent_buffer = [ReplayBufferAgent(
        cfg.buffer_length, env.num_agents, 
        state_dim=env.observation_space.shape[0], 
        action_dim=env.action_space.shape[0], 
        start_stop_index=start_stop_num[0]
    )]    
    
    ## ======================================= Initialize Algorithms =======================================
    # Initialize MADDPG
    adversary_alg = None
    maddpg = MADDPG.init_from_env(
        env, agent_alg=cfg.agent_alg, adversary_alg=adversary_alg, 
        tau=cfg.tau, lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic, 
        hidden_dim=cfg.hidden_dim, device=cfg.device, 
        epsilon=cfg.epsilon, noise=cfg.noise_scale, name=cfg.env_name
    )
    
    # Option to load pre-trained MADDPG model
    # last_run = '2024-10-12-19-25-13'
    # last_run_dir = model_dir / last_run / 'model.pt'
    # maddpg = MADDPG.init_from_save(last_run_dir)

    # Initialize AIRL discriminator
    airl = AIRL(
        state_dim=env.observation_space.shape[0], 
        action_dim=env.action_space.shape[0], 
        hidden_dim=cfg.hidden_dim, hidden_num=cfg.hidden_num, 
        lr_discriminator=cfg.lr_discriminator, 
        expert_buffer=expert_buffer, device=cfg.device
    )

    torch_agent_actions = []  # Store agent actions for training

    ## ======================================= Training Loop =======================================
    print('AIRL training starts...')
    
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):
        # Initialize episode metrics
        episode_reward_mean_bar = 0
        episode_reward_std_bar = 0
        obs = env.reset()  
        start_stop_num = [slice(0, env.n_a)]    
        
        # Prepare MADDPG for data collection
        maddpg.prep_rollouts(device='cpu') 
        maddpg.scale_noise(maddpg.noise, maddpg.epsilon)  # Apply exploration noise
        maddpg.reset_noise()

        # Apply learning rate decay to discriminator if enabled
        if cfg.disc_use_linear_lr_decay:
            airl.lr_decay(ep_index, cfg.n_episodes)
        
        ########################### Episode Rollout ###########################
        start_time_1 = time.time()
        
        for et_index in range(cfg.episode_length):
            # Render environment periodically for visualization
            if ep_index % 500 == 0:
                env.render()

            # Get actions from MADDPG policy with exploration
            torch_obs = torch.Tensor(obs).requires_grad_(False)  
            torch_agent_actions, torch_log_pis = maddpg.step(torch_obs, start_stop_num, explore=True) 
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])
            log_pis = np.column_stack([log_pi.data.numpy() for log_pi in torch_log_pis])

            # Execute actions in environment
            next_obs, rewards, dones, _, _ = env.step(agent_actions)

            # Store transition in agent buffer (includes log probabilities for AIRL)
            agent_buffer[0].push(obs, agent_actions, rewards, next_obs, dones, 
                               start_stop_num[0], log_pi_orig=log_pis)
            obs = next_obs  

            # Accumulate episode rewards for logging
            episode_reward_mean_bar += np.mean(rewards)  # Sparse environment rewards
            episode_reward_std_bar += np.std(rewards)    # Reward variance

        end_time_1 = time.time()
        
        ########################### Training Updates ###########################
        start_time_2 = time.time()
        maddpg.prep_training(device=cfg.device)
        
        # Train discriminator every 3 episodes
        if ep_index % 3 == 0:
            for _ in range(20):  # Multiple discriminator updates per training step
                for a_i in range(maddpg.nagents):
                    # Sample agent experiences
                    sample = agent_buffer[a_i].sample(cfg.batch_size, 
                                                    to_gpu=True if cfg.device == 'gpu' else False, 
                                                    is_log_pi=True)
                    obs_sample, acs_sample, _, next_obs_sample, dones_sample, _, log_pis_sample = sample

                    # Update AIRL discriminator to distinguish expert vs policy actions
                    airl.update(obs_sample, acs_sample, log_pis_sample, 
                              next_obs_sample, dones_sample, logger=logger)

        # Train MADDPG policy using discriminator rewards
        for _ in range(20):  # Multiple policy updates per training step
            for a_i in range(maddpg.nagents):
                if len(agent_buffer[a_i]) >= cfg.batch_size:
                    # Sample agent experiences
                    sample = agent_buffer[a_i].sample(cfg.batch_size, 
                                                    to_gpu=True if cfg.device == 'gpu' else False, 
                                                    is_log_pi=True)  
                    obs_sample, acs_sample, _, next_obs_sample, dones_sample, _, log_pis_sample = sample

                    # Calculate rewards using AIRL discriminator (replaces environment rewards)
                    rew_sample = airl.discriminator.calculate_reward(obs_sample, acs_sample, 
                                                                   log_pis_sample, next_obs_sample, dones_sample)

                    # Update MADDPG policy with discriminator rewards
                    maddpg.update(obs_sample, acs_sample, rew_sample, next_obs_sample, 
                                dones_sample, a_i, logger=logger)
            
            # Update target networks for stable learning
            maddpg.update_all_targets()
            
        # Switch back to rollout mode
        maddpg.prep_rollouts(device='cpu')  

        # Decay exploration noise over training
        maddpg.noise = max(0.4, maddpg.noise - cfg.noise_scale/cfg.n_episodes)
        # Optional: decay epsilon-greedy exploration
        # maddpg.epsilon = max(0.1, maddpg.epsilon - cfg.epsilon/cfg.n_episodes)
        
        end_time_2 = time.time()

        ########################### Logging and Saving ###########################
        # Print training progress
        if ep_index % 10 == 0:
            print("Episode %i/%i | Agents: %i | Reward (sparse): %.4f | Rollout time: %.2fs | Training time: %.2fs" % 
                  (ep_index, cfg.n_episodes, env.n_a, episode_reward_mean_bar/cfg.episode_length, 
                   end_time_1 - start_time_1, end_time_2 - start_time_2))
            
        # Log metrics to TensorBoard
        if ep_index % cfg.save_interval == 0:
            ALIGN_epi = 0  # Placeholder for alignment metric
            logger.add_scalars('agent/data', {
                'episode_reward_mean_bar': episode_reward_mean_bar/cfg.episode_length, 
                'episode_reward_std_bar': episode_reward_std_bar/cfg.episode_length, 
                'ALIGN_epi': ALIGN_epi
            }, ep_index)

        # Save incremental model checkpoints
        if ep_index % (4 * cfg.save_interval) < cfg.n_rollout_threads:   
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_index + 1)))

    ## ======================================= Final Saving =======================================
    # Save final trained models
    maddpg.save(run_dir / 'model.pt')          # Save MADDPG policy
    airl.save(run_dir / 'discriminator.pt')    # Save AIRL discriminator
    
    # Export training logs and close logger
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    
    print("AIRL training completed successfully!")
    print(f"Models saved to: {run_dir}")
    
if __name__ == '__main__':
    run(args)