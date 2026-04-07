import torch
import numpy as np
from torch import Tensor
from torch.optim import Adam

from .agents import DDPGAgent
from .networks import AggregatingCritic
from .misc import hard_update
from .noise import GaussianNoise
from .ctm_actor import CTMActor


class CTMDDPGAgent(DDPGAgent):
    """
    DDPGAgent subclass that replaces the MLP actor with a CTMActor.
    Critic and target critic use AggregatingCritic (permutation-equivariant
    centralised critic shared across MLP and CTM actor types).

    Does not call DDPGAgent.__init__ to avoid creating unwanted MLP policy networks.
    All attributes expected by DDPGAgent's inherited methods (scale_noise, reset_noise)
    are set explicitly here.
    """

    def __init__(self, dim_input_policy, dim_output_policy, n_agents,
                 lr_actor, lr_critic, hidden_dim=64,
                 discrete_action=False, device='cpu', epsilon=0.1, noise=0.1,
                 ctm_config=None):
        # Intentionally skip DDPGAgent.__init__ — it would create MLP policy networks
        ctm_config = ctm_config or {}

        # CTM actor (single shared network, same weights for all agents)
        self.policy = CTMActor(
            obs_dim=dim_input_policy,
            action_dim=dim_output_policy,
            **ctm_config,
        )
        self.target_policy = CTMActor(
            obs_dim=dim_input_policy,
            action_dim=dim_output_policy,
            **ctm_config,
        )

        self.critic = AggregatingCritic(n_agents, dim_input_policy, dim_output_policy, hidden_dim=180)
        self.target_critic = AggregatingCritic(n_agents, dim_input_policy, dim_output_policy, hidden_dim=180)
        self.critic2 = AggregatingCritic(n_agents, dim_input_policy, dim_output_policy, hidden_dim=180)
        self.target_critic2 = AggregatingCritic(n_agents, dim_input_policy, dim_output_policy, hidden_dim=180)

        # Materialize nn.LazyLinear layers in CTMActor before hard_update.
        # LazyLinear weights don't exist until the first forward pass — copying
        # UninitializedParameters would leave both networks uninitialized and they
        # would materialize with different random seeds on their first forward call.
        with torch.no_grad():
            _dummy = torch.zeros(1, dim_input_policy)
            _hs = self.policy.get_initial_hidden_state(1, torch.device('cpu'))
            self.policy(_dummy, _hs)
            _hs_t = self.target_policy.get_initial_hidden_state(1, torch.device('cpu'))
            self.target_policy(_dummy, _hs_t)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        hard_update(self.target_critic2, self.critic2)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=lr_critic)

        # Attributes used by inherited scale_noise / reset_noise
        self.epsilon = epsilon
        self.noise = noise
        self.discrete_action = False  # CTM actor is always continuous
        self.exploration = GaussianNoise(dim_output_policy, noise)

    def step(self, obs, hidden_states, explore=False):
        """
        Forward pass through CTM actor with optional exploration noise.

        Args:
            obs: (n_agents, obs_dim) — transposed slice from column-major env output
            hidden_states: (state_trace, activated_state_trace),
                           each (n_agents, d_model, memory_length)
            explore: whether to add exploration noise
        Returns:
            action.t(): (action_dim, n_agents) — matches DDPGAgent.step return convention
            None: log_pi placeholder (discarded in training loop with _)
            new_hidden_states: updated (state_trace, activated_state_trace) tuple
        """
        action, new_hidden_states = self.policy(obs, hidden_states)

        if explore:
            dev = action.device
            if np.random.rand() < self.epsilon:
                action = Tensor(np.random.uniform(-1, 1, size=action.shape)).to(dev).requires_grad_(False)
            else:
                action_noise = Tensor(self.exploration.noise(action.shape[0])).to(dev).requires_grad_(False)
                action = (action + action_noise).clamp(-1, 1)

        return action.t(), None, new_hidden_states

    def get_params(self):
        return {
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_policy': self.target_policy.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.critic2.load_state_dict(params['critic2'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.target_critic2.load_state_dict(params['target_critic2'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        self.critic2_optimizer.load_state_dict(params['critic2_optimizer'])
