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
from harl.models.value_function_models.discrete_q_net import DiscreteQNet
from harl.models.value_function_models.mixers import MIXER_REGISTRY
from harl.algorithms.critics.v_critic import VCritic


class ValueDecompQCritic(VCritic):
    """V Critic.
    Critic that learns a Q-function.
    """

    def __init__(
            self,
            args,
            share_obs_space,
            obs_space,
            act_space,
            num_agents,
            state_type,
            device=torch.device("cpu"),
        ):
        super().__init__(args, share_obs_space, act_space, num_agents, state_type, device)

        self.action_space = act_space
        self.num_agents = num_agents

        self.share_param = args["share_param"]
        if self.share_param:
            self.ind_critic = nn.ModuleList()
            critic = DiscreteQNet(
                args,
                obs_space,
                act_space,
                self.device,
            )
            self.ind_critic.append(critic)
            for agent_id in range(1, num_agents):
                self.ind_critic.append(self.ind_critic[0])
        else:
            self.ind_critic = nn.ModuleList()
            for agent_id in range(num_agents):
                critic = DiscreteQNet(
                    args,
                    obs_space,
                    act_space,
                    self.device,
                )
                self.ind_critic.append(critic)

        mixer_type = args["valuedecomp"]["mixer"]
        self.mixer = MIXER_REGISTRY[mixer_type](
            args=args, cent_obs_space=share_obs_space, num_agents=num_agents, device=self.device,
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.ind_critic.parameters()) + list(self.mixer.parameters()),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def get_values(self, cent_obs, obs, action, policy_prob, rnn_states_critic, masks):
        """Get value function predictions.
        Args:
            cent_obs: (np.ndarray) centralized input to the critic.
            rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        ind_q_value_collector = []
        rnn_state_ind_critic_collector = []
        for agent_id in range(self.num_agents):
            ind_q_value, rnn_state_ind_critic = self.ind_critic[agent_id](
                obs[:, agent_id],
                rnn_states_critic[:, agent_id],
                masks,
            )
            ind_q_value_collector.append(ind_q_value)
            rnn_state_ind_critic_collector.append(rnn_state_ind_critic)
        ind_q_values = torch.stack(ind_q_value_collector, dim=1)
        rnn_states_critic = torch.stack(rnn_state_ind_critic_collector, dim=1)

        action = check(action).to(**self.tpdv).long()
        policy_prob = check(policy_prob).to(**self.tpdv)
        ind_q_taken = ind_q_values.gather(dim=2, index=action)
        ind_v_values = (policy_prob * ind_q_values).sum(dim=2, keepdim=True)

        jnt_q_values = self.mixer(agent_qs=ind_q_taken, states=cent_obs)
        return jnt_q_values, ind_q_taken, ind_v_values, rnn_states_critic

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
            obs_batch,
            actions_batch,
            old_policy_probs_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values, _, _, _ = self.get_values(
            share_obs_batch,
            obs_batch,
            actions_batch,
            old_policy_probs_batch,
            rnn_states_critic_batch,
            masks_batch,
        )

        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, value_normalizer=value_normalizer
        )

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                list(self.ind_critic.parameters()) + list(self.mixer.parameters()),
                self.max_grad_norm,
            )
        else:
            critic_grad_norm = get_grad_norm(
                list(self.ind_critic.parameters()) + list(self.mixer.parameters())
            )

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm

    def prep_training(self):
        """Prepare for training."""
        self.ind_critic.train()
        self.mixer.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.ind_critic.eval()
        self.mixer.eval()

    def save(self, save_dir, id=""):
        """Save the model parameters."""
        torch.save(self.ind_critic.state_dict(), f"{str(save_dir)}/ind_critic{id}.pt")
        torch.save(self.mixer.state_dict(), f"{str(save_dir)}/mixer{id}.pt")

    def restore(self, save_dir, id=""):
        """Restore the model parameters."""
        self.ind_critic.load_state_dict(torch.load(f"{str(save_dir)}/ind_critic{id}.pt"))
        self.mixer.load_state_dict(torch.load(f"{str(save_dir)}/mixer{id}.pt"))

    def get_num_params(self):
        return sum(p.numel() for p in list(self.ind_critic.parameters()) + list(self.mixer.parameters()))