import numpy as np
from torch import Tensor
import os

class ReplayBufferExpert(object):
    """
    Replay Buffer for storing expert demonstrations in multi-agent RL.
    
    Stores transitions (s, a, s', done) from expert trajectories with
    circular buffer implementation for memory efficiency.
    """
    
    def __init__(self, max_steps, num_agents, start_stop_index, state_dim, action_dim):
        """
        Initialize the expert replay buffer.
        
        Args:
            max_steps (int): Maximum number of timesteps to store in buffer
            num_agents (int): Number of agents in environment
            start_stop_index: Agent index range for data slicing
            state_dim (int): Dimension of state/observation space
            action_dim (int): Dimension of action space
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.total_length = self.max_steps * self.num_agents

        # Initialize buffer arrays
        self.obs_buffs = np.zeros((self.total_length, state_dim)) 
        self.ac_buffs = np.zeros((self.total_length, action_dim))
        self.next_obs_buffs = np.zeros((self.total_length, state_dim))
        self.done_buffs = np.zeros((self.total_length, 1))

        self.filled_i = 0  # Index of first empty location (last index when full)
        self.curr_i = 0    # Current write index (overwrites oldest data when full)

        self.agent_index = start_stop_index

    def __len__(self):
        """Return the number of stored transitions."""
        return self.filled_i
                                            
    def push(self, observations_original, actions_original, next_observations_original, dones_original, index):
        """
        Add new transitions to the buffer.
        
        Args:
            observations_original: Original observation data
            actions_original: Original action data  
            next_observations_original: Original next observation data
            dones_original: Original done flags
            index: Slice index for agent data extraction
        """
        # Extract agent-specific data range
        start = index.start
        stop = index.stop
        span = range(start, stop)
        data_length = len(span)
        
        # Transpose and slice data for specified agents
        observations = observations_original[:, index].T   
        actions = actions_original[:, index].T                 
        next_observations = next_observations_original[:, index].T
        dones = dones_original[:, index].T          
     
        # Handle circular buffer wraparound
        if self.curr_i + data_length > self.total_length:   
            rollover = data_length - (self.total_length - self.curr_i)
            self.curr_i -= rollover

        # Store transitions in buffer
        self.obs_buffs[self.curr_i:self.curr_i + data_length, :] = observations             
        self.ac_buffs[self.curr_i:self.curr_i + data_length, :] = actions
        self.next_obs_buffs[self.curr_i:self.curr_i + data_length, :] = next_observations     
        self.done_buffs[self.curr_i:self.curr_i + data_length, :] = dones         

        # Update buffer indices
        self.curr_i += data_length

        if self.filled_i < self.total_length:
            self.filled_i += data_length         
        if self.curr_i == self.total_length: 
            self.curr_i = 0  

    def sample(self, N, to_gpu=False, norm_rews=True, agent_index=0):
        """
        Sample random transitions from the buffer.
        
        Args:
            N (int): Number of transitions to sample
            to_gpu (bool): Whether to move tensors to GPU
            norm_rews (bool): Whether to normalize rewards (unused)
            agent_index (int): Specific agent index (unused)
            
        Returns:
            tuple: (observations, actions, next_observations, dones) as tensors
        """
        # Initialize sample arrays
        obs_inds = np.zeros((N, self.obs_buffs.shape[1]))
        act_inds = np.zeros((N, self.ac_buffs.shape[1]))
        next_obs_inds = np.zeros((N, self.next_obs_buffs.shape[1]))
        done_inds = np.zeros((N, 1))

        # Sample from middle portion of buffer to avoid recent/old data bias
        begin_index = np.random.randint(0, 3 * self.total_length // 4)
        inds = np.random.choice(
            np.arange(begin_index, self.total_length // 4 + begin_index, dtype=np.int32), 
            size=N, 
            replace=False
        )

        # Extract sampled data
        obs_inds = self.obs_buffs[inds, :]
        act_inds = self.ac_buffs[inds, :]
        next_obs_inds = self.next_obs_buffs[inds, :]
        done_inds = self.done_buffs[inds, :]
        
        # Convert to tensors
        if to_gpu:
            cast = lambda x: Tensor(x).requires_grad_(False).cuda()
        else:
            cast = lambda x: Tensor(x).requires_grad_(False)

        return (cast(obs_inds), cast(act_inds), cast(next_obs_inds), cast(done_inds))
    
    def save(self, file_dir):
        """
        Save buffer data to disk.
        
        Args:
            file_dir (str): Directory path to save the data
        """
        file_name = os.path.join(file_dir, 'expert_data.npz')
        np.savez(
            file=file_name,
            obs_buffs=self.obs_buffs,
            ac_buffs=self.ac_buffs,
            next_obs_buffs=self.next_obs_buffs,
            done_buffs=self.done_buffs
        )
    
    def load(self, file_name):
        """
        Load buffer data from disk.
        
        Args:
            file_name (str): Path to the saved data file
        """
        data = np.load(file_name)
        self.obs_buffs = data['obs_buffs']
        self.ac_buffs = data['ac_buffs']
        self.next_obs_buffs = data['next_obs_buffs']
        self.done_buffs = data['done_buffs']