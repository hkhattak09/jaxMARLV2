import torch
import time
import os
import numpy as np
import random
from datetime import datetime
import gym
from gym.wrappers import AssemblySwarmWrapper
from cfg.assembly_cfg import gpsargs as args
from pathlib import Path
from algorithm.utils import ReplayBufferExpert

def run(cfg):
    """
    This function runs multiple episodes using a rule-based strategy to generate
    expert trajectories for imitation learning or inverse reinforcement learning.
    
    Args:
        cfg: Configuration object containing experiment parameters.
    """
    ## ======================================= Initialize =======================================
    # Set random seeds for reproducible expert data collection
    torch.manual_seed(cfg.seed)  
    np.random.seed(cfg.seed) 
    random.seed(cfg.seed)  
    
    # Configure PyTorch threading based on device
    if cfg.device == 'cpu':
        torch.set_num_threads(cfg.n_training_threads) 
    elif cfg.device == 'gpu':
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed) 

    # Create timestamped directory for expert data storage
    model_dir = './' / Path('./eval/expert_data')
    curr_run = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = model_dir / curr_run      
    os.makedirs(run_dir)

    # Configure video recording if enabled
    if args.video:
        args.video_path = str(run_dir) + '/video.mp4'

    # Initialize assembly swarm environment
    scenario_name = 'AssemblySwarm-v0'
    base_env = gym.make(scenario_name).unwrapped
    env = AssemblySwarmWrapper(base_env, args)
    start_stop_num = slice(0, env.num_agents)  # Agent indexing for multi-agent setup

    # Initialize expert replay buffer for storing demonstration data
    expert_buffer = ReplayBufferExpert(
        cfg.buffer_length, 
        env.num_agents, 
        state_dim=env.observation_space.shape[0], 
        action_dim=env.action_space.shape[0], 
        start_stop_index=start_stop_num
    )

    ## ======================================= Expert Data Collection =======================================
    print('Expert data collection starts...')
    
    for ep_index in range(0, cfg.n_episodes, cfg.n_rollout_threads):
        # Initialize episode variables
        episode_reward = 0
        obs = env.reset() 
        agent_actions = np.zeros((2, env.n_a))  # Initialize action array for multi-agent system
        start_stop_num = slice(0, env.n_a)      # Update agent slice for current episode
        
        ########################### Execute Episode ###########################
        start_time_1 = time.time()
        
        for step in range(cfg.episode_length):
            # Render environment periodically for visualization
            if ep_index % 50 == 0:
                env.render()

            # Execute step with rule-based expert policy
            next_obs, rewards, dones, _, agent_actions = env.step(agent_actions)
            
            # Store expert transition in replay buffer
            expert_buffer.push(obs, agent_actions, next_obs, dones, start_stop_num) 
            
            # Update state and accumulate rewards
            obs = next_obs    
            episode_reward += np.mean(rewards) 

        # Calculate final performance metrics for the episode
        coverage_rate = env.coverage_rate()           # Percentage of target shape covered
        uniformity_degree = env.distribution_uniformity()  # Uniformity of agent distribution
        
        print('Coverage rate: {:.4f}, Distribution uniformity: {:.4f}'.format(
            coverage_rate, uniformity_degree))

        end_time_1 = time.time()

        ########################### Episode Summary ###########################
        print("Episode %i/%i | Avg reward: %.4f | Buffer: %i/%i | Runtime: %.2fs" % (
            ep_index + 1, cfg.n_episodes, 
            episode_reward / cfg.episode_length, 
            expert_buffer.filled_i, expert_buffer.total_length, 
            end_time_1 - start_time_1))
    
    # Save collected expert demonstration data
    print(f"Saving expert data to: {run_dir}")
    expert_buffer.save(run_dir)
    print("Expert data collection completed successfully!")

if __name__ == '__main__':
    # Configure expert data collection parameters
    args.agent_strategy = 'rule'      # Use rule-based expert strategy
    args.n_episodes = 500             # Total episodes to collect (increased from 300)
    args.buffer_length = int(1e5)     # Buffer capacity (total capacity = 50 episodes × 30k steps = 1.5M transitions)
    args.is_collected = True          # Enable expert data collection
    
    # Start expert data collection
    run(args)