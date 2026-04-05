from torch import Tensor
import torch
import numpy as np
from torch.optim import Adam
from .networks import MLPNetwork, AggregatingCritic
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import GaussianNoise

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, dim_input_policy, dim_output_policy, n_agents,
                 lr_actor, lr_critic, hidden_dim=64,
                 discrete_action=False, device='cpu', epsilon=0.1, noise=0.1):
        """
        Inputs:
            dim_input_policy (int): per-agent observation dim
            dim_output_policy (int): per-agent action dim
            n_agents (int): total number of agents (used by AggregatingCritic)
        """
        self.policy = MLPNetwork(dim_input_policy, dim_output_policy,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.target_policy = MLPNetwork(dim_input_policy, dim_output_policy,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)

        self.critic = AggregatingCritic(n_agents, dim_input_policy, dim_output_policy, hidden_dim=180)
        self.target_critic = AggregatingCritic(n_agents, dim_input_policy, dim_output_policy, hidden_dim=180)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr_critic)

        self.epsilon = epsilon
        self.noise = noise
        if not discrete_action:
            self.exploration = GaussianNoise(dim_output_policy, noise)
        else:
            self.exploration = 0.3
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations.
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        log_pi = torch.full((action.shape[0], 1), -action.shape[1] * np.log(1), device=action.device)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:
            if explore:
                dev = action.device
                if np.random.rand() < self.epsilon:
                    action = Tensor(np.random.uniform(-1, 1, size=action.shape)).to(dev).requires_grad_(False)
                    log_pi = torch.full((action.shape[0], 1), -action.shape[1] * np.log(2), device=dev)
                else:
                    action_noise = Tensor(self.exploration.noise(action.shape[0])).to(dev).requires_grad_(False)
                    log_pi = Tensor(self.exploration.log_prob(action_noise)).to(dev).unsqueeze(-1).requires_grad_(False)
                    action += action_noise
                    action = action.clamp(-1, 1)

        return action.t(), log_pi.t()

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
