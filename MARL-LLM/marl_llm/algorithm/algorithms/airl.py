import torch
from torch.optim import Adam
from algorithm.utils import Discriminator
import torch.nn.functional as F
import numpy as np

def check(input):
    """
    Convert numpy array to torch tensor if necessary.
    
    Args:
        input: Input data (numpy array or torch tensor)
        
    Returns:
        torch.Tensor: Converted tensor
    """
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """
    Decreases the learning rate linearly over training epochs.
    
    Args:
        optimizer: PyTorch optimizer to update
        epoch (int): Current epoch number
        total_num_epochs (int): Total number of training epochs
        initial_lr (float): Initial learning rate
    """
    # Linear decay with 60% reduction over total epochs
    lr = initial_lr - (initial_lr * (0.6 * epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AIRL(object):
    """
    AIRL learns a reward function by training a discriminator to distinguish
    between expert demonstrations and policy-generated trajectories.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim, hidden_num, lr_discriminator, expert_buffer, batch_size=512, gamma=0.95, device='cpu'):
        """
        Initialize the AIRL discriminator and optimizer.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden layer dimension for discriminator
            hidden_num (int): Number of hidden layers
            lr_discriminator (float): Learning rate for discriminator
            expert_buffer: Buffer containing expert demonstrations
            batch_size (int): Batch size for training
            gamma (float): Discount factor
            device (str): Device to run on ('cpu' or 'gpu')
        """
        # Convert device string to proper format
        device = 'cuda' if device == 'gpu' else 'cpu'
        
        # Initialize discriminator network
        self.discriminator = Discriminator(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            hidden_dim=hidden_dim,
            hidden_num=hidden_num
        ).to(device)

        self.device = device
        self.lr_discriminator = lr_discriminator
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=self.lr_discriminator)
        self.expert_buffer = expert_buffer
        self.batch_size = batch_size
        self.n_iter = 0

    def update(self, states, actions, log_pis, next_states, dones=0, logger=None):
        """
        Update the discriminator using policy and expert trajectories.
        
        Args:
            states: Current states from policy rollout
            actions: Actions taken by policy
            log_pis: Log probabilities of policy actions
            next_states: Next states from policy rollout
            dones: Episode termination flags (default: 0)
            logger: Optional tensorboard logger
        """
        # Sample expert demonstrations (6x larger batch for stability)
        states_exp, actions_exp, next_states_exp, dones_exp = self.expert_buffer.sample(
            6 * self.batch_size, 
            to_gpu=True if self.device == 'cuda' else False
        )

        # Calculate log probabilities for expert actions
        # Using uniform distribution assumption for expert actions
        with torch.no_grad():
            log_pis_exp = torch.full(
                (states_exp.size(0), 1), 
                -actions_exp.size(1) * np.log(1), 
                device=states_exp.device
            )

        # Get discriminator outputs (logits in (-inf, inf) range)
        logits_pi = self.discriminator(states, actions, log_pis, next_states, dones)
        logits_exp = self.discriminator(states_exp, actions_exp, log_pis_exp, next_states_exp, dones_exp)

        # AIRL discriminator loss: maximize log(D_exp) + log(1 - D_pi)
        # Policy samples should be classified as fake (logits < 0)
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        # Expert samples should be classified as real (logits > 0)
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_discriminator = loss_pi + loss_exp

        # Update discriminator parameters
        self.discriminator_optimizer.zero_grad()
        loss_discriminator.backward()
        self.discriminator_optimizer.step()

        # Calculate classification accuracies for monitoring
        with torch.no_grad():
            acc_pi = (logits_pi < 0).float().mean().item()    # Policy classified as fake
            acc_exp = (logits_exp > 0).float().mean().item()  # Expert classified as real
            
        # Log training metrics
        if logger is not None:
            logger.add_scalars('agent0/losses', {
                'loss_discriminator': loss_discriminator, 
                'accuracy_pi': acc_pi, 
                'accuracy_exp': acc_exp
            }, self.n_iter)
        self.n_iter += 1

    def lr_decay(self, episode, episodes):
        """
        Apply linear learning rate decay to discriminator optimizer.
        
        Args:
            episode (int): Current episode number
            episodes (int): Total number of episodes
        """
        update_linear_schedule(self.discriminator_optimizer, episode, episodes, self.lr_discriminator)

    def save(self, filename):
        """
        Save the discriminator's state dictionary to file.
        
        Args:
            filename (str): Path to save the model
        """
        torch.save(self.discriminator.state_dict(), filename)

    def load(self, filename):
        """
        Load the discriminator's state dictionary from file.
        
        Args:
            filename (str): Path to load the model from
        """
        self.discriminator.load_state_dict(torch.load(filename))