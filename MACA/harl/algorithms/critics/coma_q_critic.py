"""V Critic."""
import torch
import torch.nn as nn
from harl.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)
from harl.utils.envs_tools import check
from harl.models.value_function_models.coma_q_net import ComaQNet
from harl.algorithms.critics.v_critic import VCritic


class ComaQCritic(VCritic):
    """V Critic.
    Critic that learns a Q-function.
    """

    def __init__(
            self,
            args,
            share_obs_space,
            act_space,
            num_agents,
            state_type,
            device=torch.device("cpu"),
        ):
        super().__init__(args, share_obs_space, act_space, num_agents, state_type, device)

        self.action_space = act_space
        self.num_agents = num_agents
        self.critic = ComaQNet(
            args,
            self.share_obs_space,
            self.action_space,
            num_agents,
            self.device,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def get_values(self, cent_obs, action, policy_prob, rnn_states_critic, masks):
        """Get value function predictions.
        Args:
            cent_obs: (np.ndarray) centralized input to the critic.
            rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        q_values, rnn_states_critic = self.critic(
            cent_obs,
            action,
            rnn_states_critic,
            masks[:, None].repeat(repeats=self.num_agents, axis=1),
        )
        action = check(action).to(**self.tpdv).long()
        policy_prob = check(policy_prob).to(**self.tpdv)
        q_taken = q_values.gather(dim=2, index=action)
        vq_taken = (policy_prob * q_values).sum(dim=2, keepdim=True)
        return q_taken, vq_taken, rnn_states_critic

    def update(self, sample, value_normalizer=None):
        """Update critic network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
            critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        """
        (
            share_obs_batch,
            rnn_states_critic_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            actions_batch,
            old_policy_probs_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values, _, _ = self.get_values(
            share_obs_batch, actions_batch, old_policy_probs_batch, rnn_states_critic_batch, masks_batch
        )

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, value_normalizer=value_normalizer
        )

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm
