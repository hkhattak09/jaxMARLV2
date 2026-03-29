import torch
import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datetime import datetime
import gym
from gym.wrappers import AssemblySwarmWrapper
from cfg.assembly_cfg import gpsargs as args
from pathlib import Path
from algorithm.utils import ReplayBufferAgent
from algorithm.algorithms import MADDPG


def run(cfg):
    """
    Main training function for MADDPG.
    
    Args:
        cfg: Configuration object containing training hyperparameters and settings
    """
    
    ## ======================================= Setup Logging =======================================
    # Create directories for model saving and logging
    model_dir = './' / Path('./models') / cfg.env_name 
    curr_run = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = model_dir / curr_run   
    log_dir = run_dir / 'logs'    
    os.makedirs(log_dir)    
    logger = SummaryWriter(str(log_dir))  

    ## ======================================= Initialize Environment =======================================
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.seed)  
    np.random.seed(cfg.seed) 
    random.seed(cfg.seed)  
    
    # Configure device-specific settings
    if cfg.device == 'cpu':
        torch.set_num_threads(cfg.n_training_threads) 
    elif cfg.device == 'gpu':
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    # Initialize environment
    scenario_name = 'AssemblySwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = AssemblySwarmWrapper(base_env, args)
    start_stop_num = [slice(0, env.num_agents)]  # Agent indexing for multi-agent setup

    # Initialize MADDPG algorithm
    adversary_alg = None  # No adversarial agents in this setup
    maddpg = MADDPG.init_from_env(env, agent_alg=cfg.agent_alg, adversary_alg=adversary_alg, 
                                  tau=cfg.tau, lr_actor=cfg.lr_actor, lr_critic=cfg.lr_critic, 
                                  hidden_dim=cfg.hidden_dim, device=cfg.device, epsilon=cfg.epsilon, 
                                  noise=cfg.noise_scale, name=cfg.env_name)
    
    # Option to load from previous checkpoint (currently commented out)
    # last_run = '2025-01-08-19-57-46'
    # last_run_dir = model_dir / last_run / 'model.pt'
    # maddpg = MADDPG.init_from_save(last_run_dir)

    # Initialize replay buffer for experience storage
    agent_buffer = [ReplayBufferAgent(cfg.buffer_length, env.num_agents, 
                                     state_dim=env.observation_space.shape[0], 
                                     action_dim=env.action_space.shape[0], 
                                     start_stop_index=start_stop_num[0])]  

    torch_agent_actions = []  # Container for agent actions

    ## ======================================= Training Loop =======================================
    print('Training Starts...')
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):
        # Initialize episode metrics
        episode_reward_mean_bar = 0
        episode_reward_std_bar = 0
        
        # Reset environment and prepare for rollout
        obs = env.reset()  
        start_stop_num = [slice(0, env.n_a)]    
        maddpg.prep_rollouts(device='cpu')  # Set networks to CPU for environment interaction
      
        # Configure exploration noise
        maddpg.scale_noise(maddpg.noise, maddpg.epsilon)
        maddpg.reset_noise()
        
        ########################### Episode Rollout ###########################
        start_time_1 = time.time()
        for et_index in range(cfg.episode_length):
            # Render environment periodically for visualization
            if ep_index % 500 == 0:
                env.render()

            # Get actions from MADDPG agents
            torch_obs = torch.Tensor(obs).requires_grad_(False)  
            torch_agent_actions, _ = maddpg.step(torch_obs, start_stop_num, explore=True) 
            agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])

            # Execute actions in environment
            next_obs, rewards, dones, _, agent_actions_prior = env.step(agent_actions)

            # Store experience in replay buffer
            agent_buffer[0].push(obs, agent_actions, rewards, next_obs, dones, 
                               start_stop_num[0], agent_actions_prior)
            obs = next_obs  

            # Accumulate reward statistics
            episode_reward_mean_bar += np.mean(rewards)  # Mean reward across agents
            episode_reward_std_bar += np.std(rewards)    # Reward variance across agents

        end_time_1 = time.time()
        
        ########################### Training Phase ###########################
        start_time_2 = time.time()
        maddpg.prep_training(device=cfg.device)  # Move networks to training device
        
        # Perform multiple training updates per episode
        for _ in range(20):      
            for a_i in range(maddpg.nagents):
                # Check if buffer has enough samples for training
                if len(agent_buffer[a_i]) >= cfg.batch_size:
                    # Sample batch from replay buffer
                    sample = agent_buffer[a_i].sample(cfg.batch_size, 
                                                    to_gpu=True if cfg.device == 'gpu' else False, 
                                                    is_prior=True if cfg.training_method == 'llm_rl' else False)  
                    obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, acs_prior_sample, _ = sample
                    
                    # Update MADDPG networks
                    maddpg.update(obs_sample, acs_sample, rews_sample, next_obs_sample, dones_sample, 
                                a_i, acs_prior_sample, env.alpha, logger=logger)
            
            # Update target networks
            maddpg.update_all_targets()
            
        maddpg.prep_rollouts(device='cpu')   # Return to CPU for next rollout

        # Decay exploration parameters
        maddpg.noise = max(0.5, maddpg.noise - cfg.noise_scale/cfg.n_episodes)
        # maddpg.epsilon = max(0.1, maddpg.epsilon - cfg.epsilon/cfg.n_episodes)  # Optional epsilon decay

        # Update regularization coefficient for prior actions
        env.env.alpha = 0.1
        end_time_2 = time.time()

        ########################### Logging and Checkpointing ###########################
        # Print training progress
        if ep_index % 10 == 0:
            print("Episodes %i of %i, agent num: %i, episode reward: %f, step time: %f, training time: %f" % 
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

    # Save final trained model
    maddpg.prep_training(device=cfg.device)
    maddpg.save(run_dir / 'model.pt')
      
    # Export training logs and cleanup
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    plt.close('all')


if __name__ == '__main__':
    """Entry point for training script."""
    run(args)