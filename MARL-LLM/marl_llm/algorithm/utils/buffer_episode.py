import numpy as np
from torch import Tensor

class ReplayBufferEpisode(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts.
    
    Stores episodic trajectories for multiple agents and supports 
    random sampling for training.
    
    Args:
        max_size (int): Maximum number of episodes to store
    """
    
    def __init__(self, max_size):
        """Initialize the replay buffer with specified capacity."""
        self.max_size = max_size
        self.data = []  # Storage for episode trajectories
        self.ptr = 0    # Circular buffer pointer
    
    def append(self, state, act, rew, state_act, n_a, epi_len):
        """
        Add a new episode to the buffer.
        
        Args:
            state (list): List of state tensors for each timestep
            act (list): List of action tensors for each timestep  
            rew (list): List of reward tensors for each timestep
            state_act (list): List of state-action tensors for each timestep
            n_a (int): Number of agents
            epi_len (int): Episode length (number of timesteps)
        """
        # Process each agent's trajectory separately
        for agent_i in range(n_a):
            state_list = []
            act_list = []
            rew_list = []
            state_act_list = []
            
            # Extract agent-specific data for each timestep
            for time_i in range(epi_len):
                state_list.append(state[time_i][:, agent_i])
                act_list.append(act[time_i][:, agent_i])
                rew_list.append(rew[time_i][:, agent_i])
                state_act_list.append(state_act[time_i][:, agent_i])
            
            # Store or overwrite trajectory data
            if self.full():
                self.data[self.ptr] = [state_list, act_list, rew_list, state_act_list]
            else:
                self.data.append([state_list, act_list, rew_list, state_act_list])
            
            # Update circular buffer pointer
            self.ptr = (self.ptr + 1) % self.max_size
    
    def sample(self, sample_size, to_gpu):
        """
        Sample random episodes from the buffer.
        
        Args:
            sample_size (int): Number of episodes to sample
            to_gpu (bool): Whether to move tensors to GPU
            
        Returns:
            list: Mini-batch of sampled trajectories as tensors
        """
        # Randomly select episode indices
        idxes = np.random.choice(len(self.data), sample_size)
        mini_batch = []
        
        # Set tensor casting function based on GPU usage
        if to_gpu:
            cast = lambda x: Tensor(x).requires_grad_(False).cuda()
        else:
            cast = lambda x: Tensor(x).requires_grad_(False)
        
        # Convert sampled data to tensors
        for idx in idxes:
            traj_data = [[cast(arr) for arr in row] for row in self.data[idx]]
            mini_batch.append(traj_data)
        
        return mini_batch
    
    def __len__(self):
        """Return the current number of stored episodes."""
        return len(self.data)
    
    def full(self):
        """Check if the buffer has reached maximum capacity."""
        return len(self.data) == self.max_size