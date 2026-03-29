"""
Multi-Agent Replay Buffer Module

This module implements a replay buffer for storing and sampling experience data
in multi-agent reinforcement learning environments with parallel rollouts.
"""

import numpy as np
from torch import Tensor
import torch


class ReplayBufferAgent(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts.
    
    Stores experience tuples (observation, action, reward, next_observation, done)
    for multiple agents and supports efficient sampling for training.
    """
    
    def __init__(self, max_steps, num_agents, start_stop_index, state_dim, action_dim):
        """
        Initialize the replay buffer with specified dimensions.
        
        Args:
            max_steps (int): Maximum number of timesteps to store in buffer
            num_agents (int): Number of agents in the environment
            start_stop_index (slice): Index range for agent selection
            state_dim (int): Dimension of observation space
            action_dim (int): Dimension of action space
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        
        # Initialize buffer arrays for different data types
        self.obs_buffs = []
        self.ac_buffs = []
        self.ac_prior_buffs = []        # Prior actions (for guided learning)
        self.log_pi_buffs = []          # Log probabilities
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        
        # Calculate total buffer capacity (steps × agents)
        self.total_length = self.max_steps * self.num_agents

        # Allocate memory for all experience components
        self.obs_buffs = np.zeros((self.total_length, state_dim)) 
        self.ac_buffs = np.zeros((self.total_length, action_dim))
        self.ac_prior_buffs = np.zeros((self.total_length, action_dim))
        self.log_pi_buffs = np.zeros((self.total_length, 1))
        self.rew_buffs = np.zeros((self.total_length, 1))
        self.next_obs_buffs = np.zeros((self.total_length, state_dim))
        self.done_buffs = np.zeros((self.total_length, 1))

        # Buffer management indices
        self.filled_i = 0  # Index of first empty location (total filled when full)
        self.curr_i = 0    # Current write index (circular buffer)

        # Agent indexing for multi-agent data selection
        self.agent_index = start_stop_index

    def __len__(self):
        """Return the number of stored experiences."""
        return self.filled_i
                                            
    def push(self, observations_orig, actions_orig, rewards_orig, next_observations_orig, 
             dones_orig, index, actions_prior_orig=None, log_pi_orig=None):
        """
        Add new experience data to the buffer.
        
        Args:
            observations_orig (np.array): Current observations for all agents
            actions_orig (np.array): Actions taken by all agents
            rewards_orig (np.array): Rewards received by all agents
            next_observations_orig (np.array): Next observations for all agents
            dones_orig (np.array): Done flags for all agents
            index (slice): Agent indices to store
            actions_prior_orig (np.array, optional): Prior/expert actions
            log_pi_orig (np.array, optional): Log probabilities of actions
        """
        # Extract agent-specific data using index slice
        start = index.start
        stop = index.stop
        span = range(start, stop)
        data_length = len(span)
        
        # Transpose and select agent data
        observations = observations_orig[:, index].T   
        actions = actions_orig[:, index].T
        rewards = rewards_orig[:, index].T                  
        next_observations = next_observations_orig[:, index].T
        dones = dones_orig[:, index].T
        
        # Process optional data if provided
        if actions_prior_orig is not None:
            actions_prior = actions_prior_orig[:, index].T
        if log_pi_orig is not None:
            log_pis = log_pi_orig[:, index].T 
     
        # Handle circular buffer overflow
        if self.curr_i + data_length > self.total_length:   
            rollover = data_length - (self.total_length - self.curr_i)
            self.curr_i -= rollover

        # Store experience data in buffer arrays
        self.obs_buffs[self.curr_i:self.curr_i + data_length, :] = observations             
        self.ac_buffs[self.curr_i:self.curr_i + data_length, :] = actions
        self.rew_buffs[self.curr_i:self.curr_i + data_length, :] = rewards
        self.next_obs_buffs[self.curr_i:self.curr_i + data_length, :] = next_observations     
        self.done_buffs[self.curr_i:self.curr_i + data_length, :] = dones
        
        # Store optional data if available
        if actions_prior_orig is not None:
            self.ac_prior_buffs[self.curr_i:self.curr_i + data_length, :] = actions_prior
        if log_pi_orig is not None:
            self.log_pi_buffs[self.curr_i:self.curr_i + data_length, :] = log_pis    

        # Update buffer indices
        self.curr_i += data_length

        # Track total filled capacity
        if self.filled_i < self.total_length:
            self.filled_i += data_length
            
        # Reset current index when buffer is full (circular buffer)
        if self.curr_i == self.total_length: 
            self.curr_i = 0  

    def sample(self, N, to_gpu=False, is_prior=False, is_log_pi=False):
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            N (int): Number of experiences to sample
            to_gpu (bool): Whether to move tensors to GPU
            is_prior (bool): Whether to include prior actions in sample
            is_log_pi (bool): Whether to include log probabilities in sample
            
        Returns:
            tuple: Batch of (observations, actions, rewards, next_observations, 
                          dones, prior_actions, log_probabilities)
        """
        # Define sampling range 
        begin_index_range = 3e5 
        begin_index = np.random.randint(0, begin_index_range)
        
        # Randomly sample indices from valid range
        inds = np.random.choice(
            np.arange(begin_index, 
                     self.total_length - begin_index_range + begin_index, 
                     dtype=np.int32), 
            size=N, replace=False
        )

        # Extract sampled experiences
        obs_inds = self.obs_buffs[inds, :]
        act_inds = self.ac_buffs[inds, :]
        rew_inds = self.rew_buffs[inds, :]
        next_obs_inds = self.next_obs_buffs[inds, :]
        done_inds = self.done_buffs[inds, :]
        
        # Extract optional data if requested
        if is_prior:
            act_prior_inds = self.ac_prior_buffs[inds, :]
        if is_log_pi:
            log_pis_inds = self.log_pi_buffs[inds, :]
        
        # Convert to tensors with appropriate device placement
        if to_gpu:
            cast = lambda x: Tensor(x).requires_grad_(False).cuda()
        else:
            cast = lambda x: Tensor(x).requires_grad_(False)

        return (cast(obs_inds), cast(act_inds), cast(rew_inds), cast(next_obs_inds), 
                cast(done_inds), cast(act_prior_inds) if is_prior else None, 
                cast(log_pis_inds) if is_log_pi else None)

    def get_average_rewards(self, N):
        """
        Calculate average rewards over the last N experiences.
        
        Args:
            N (int): Number of recent experiences to average over
            
        Returns:
            list: Average rewards for each agent
        """
        # Handle different buffer fill states
        if self.filled_i == self.max_steps:
            # Buffer is full, use circular indexing
            inds = np.arange(self.curr_i - N, self.curr_i)
        else:
            # Buffer not full, use simple indexing
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
            
        # Calculate mean rewards for each agent
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]