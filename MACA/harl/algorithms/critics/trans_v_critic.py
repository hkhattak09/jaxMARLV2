"""Transformer V Critic."""
import torch
import torch.nn as nn
from harl.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
    update_cosine_schedule,
)
from harl.utils.envs_tools import check
from harl.models.base.transformer import Encoder as Transformer
from harl.algorithms.critics.v_critic import VCritic

class TransVCritic(VCritic):
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
        super().__init__(
            args,
            share_obs_space,
            act_space,
            num_agents,
            state_type,
            device
        )
        # how many steps to warm up for. ~= max_iters/300 per nanoGPT
        self.warmup_epochs = args["transformer"]["warmup_epochs"]
        # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        self.min_lr = 0.1 * self.critic_lr
        self.q_value_loss_coef = args["transformer"]["q_value_loss_coef"]
        self.eq_value_loss_coef = args["transformer"]["eq_value_loss_coef"]
        self.output_attentions = args["transformer"].get("output_attentions", True)

        self.next_s_pred_loss_coef = args["transformer"]["next_s_pred_loss_coef"]
        if self.next_s_pred_loss_coef:
            assert self.use_recurrent_policy or self.use_naive_recurrent_policy, \
                "Next state prediction requires RNN net in current implementation," \
                " since data generator for MLP net returns IID samples."
        self.num_agents = num_agents
        args["transformer"]["n_block"] = num_agents
        self.critic = Transformer(args, obs_space, act_space, device)
        self.critic_optimizer = self.critic.configure_optimizers(self.critic_lr, device.type)

    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_cosine_schedule(
            self.critic_optimizer,
            episode,
            episodes,
            self.warmup_epochs,
            self.critic_lr,
            self.min_lr,
        )

    def get_values(self, obs, action, policy_prob, rnn_states_critic, masks):
        """Get value function predictions.
        Args:
            obs: (np.ndarray) centralized input to the critic.
            action: (np.ndarray) actions taken by the agent.
            rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        (
            values,
            q_values,
            eq_values,
            vq_values,
            vq_coma_values,
            bsln_weights,
            attn_weights,
            self.zs,
            self.zsa,
            rnn_states_critic,
        ) = self.critic(
            obs,
            action,
            policy_prob,
            rnn_states_critic,
            masks[:, None].repeat(repeats=self.num_agents, axis=1),
            self.output_attentions,
        )
        return (
            values,
            q_values,
            eq_values,
            vq_values,
            vq_coma_values,
            bsln_weights,
            attn_weights,
            rnn_states_critic,
        )

    # TODO may implement gradient_accumulation_steps as per nanoGPT
    def update(self, sample, value_normalizer=dict()):
        """Update critic network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
            value_normalizer: (Dict) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
            critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        """
        (
            episode_chunk_length,
            share_obs_batch,
            rnn_states_critic_batch,
            value_preds_batch,
            q_value_preds_batch,
            eq_value_preds_batch,
            return_batch,
            q_return_batch,
            eq_return_batch,
            vq_return_batch,
            vq_coma_return_batch,
            masks_batch,
            obs_batch,
            rnn_states_batch,
            actions_batch,
            active_masks_batch,
            old_policy_probs_batch,
            available_actions_batch,
            rewards_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        q_value_preds_batch = check(q_value_preds_batch).to(**self.tpdv)
        eq_value_preds_batch = check(eq_value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        q_return_batch = check(q_return_batch).to(**self.tpdv)
        eq_return_batch = check(eq_return_batch).to(**self.tpdv)

        values, q_values, eq_values, _, _, _, _, _ = self.get_values(
            obs_batch, actions_batch, old_policy_probs_batch, rnn_states_critic_batch, masks_batch
        )

        value_loss = torch.tensor(0.0, device=self.device)
        if self.value_loss_coef > 0:
            value_loss = self.cal_value_loss(
                values, value_preds_batch, return_batch, value_normalizer=value_normalizer["v"]
            )
        q_value_loss = torch.tensor(0.0, device=self.device)
        if self.q_value_loss_coef > 0:
            q_value_loss = self.cal_value_loss(
                q_values, q_value_preds_batch, q_return_batch, value_normalizer=value_normalizer["q"]
            )
        eq_value_loss = torch.tensor(0.0, device=self.device)
        if self.eq_value_loss_coef > 0:
            eq_value_loss = self.cal_value_loss(
                eq_values, eq_value_preds_batch, eq_return_batch, value_normalizer=value_normalizer["eq"]
            )

        self.critic_optimizer.zero_grad()

        critic_loss = 0
        critic_loss += value_loss * self.value_loss_coef
        critic_loss += q_value_loss * self.q_value_loss_coef
        critic_loss += eq_value_loss * self.eq_value_loss_coef

        next_s_pred_loss = torch.tensor(0.0, device=self.device)
        if self.next_s_pred_loss_coef > 0:
            zsa = self.zsa.view(episode_chunk_length, -1, *self.zsa.shape[1:])[:-1]
            zs = self.zs.view(episode_chunk_length, -1, *self.zs.shape[1:])[1:]
            if self.use_huber_loss:
                next_s_pred_loss = huber_loss(zsa - zs.detach(), self.huber_delta).mean()
            else:
                next_s_pred_loss = mse_loss(zsa - zs.detach()).mean()
            critic_loss += next_s_pred_loss * self.next_s_pred_loss_coef

        # TODO may add reward prediction loss.
        critic_loss.backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, q_value_loss, eq_value_loss, next_s_pred_loss, critic_grad_norm

    def train(self, critic_buffer, value_normalizer=dict()):
        """Perform a training update using minibatch GD.
        Args:
            critic_buffer: (OnPolicyCriticBufferEP or OnPolicyCriticBufferFP) buffer containing training data related to critic.
            value_normalizer: (Dict) normalize the rewards, denormalize critic outputs.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        train_info = {}

        train_info["value_loss"] = 0
        train_info["q_value_loss"] = 0
        train_info["eq_value_loss"] = 0
        train_info["next_s_pred_loss"] = 0
        train_info["critic_grad_norm"] = 0

        for _ in range(self.critic_epoch):
            if self.use_recurrent_policy:
                data_generator = critic_buffer.recurrent_generator_critic(
                    self.critic_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = critic_buffer.naive_recurrent_generator_critic(
                    self.critic_num_mini_batch
                )
            else:
                data_generator = critic_buffer.feed_forward_generator_critic(
                    self.critic_num_mini_batch
                )

            for sample in data_generator:
                (
                    value_loss,
                    q_value_loss,
                    eq_value_loss,
                    next_s_pred_loss,
                    critic_grad_norm,
                ) = self.update(
                    sample, value_normalizer=value_normalizer
                )

                train_info["value_loss"] += value_loss.item()
                train_info["q_value_loss"] += q_value_loss.item()
                train_info["eq_value_loss"] += eq_value_loss.item()
                train_info["next_s_pred_loss"] += next_s_pred_loss.item()
                train_info["critic_grad_norm"] += critic_grad_norm.item()

        num_updates = self.critic_epoch * self.critic_num_mini_batch

        for k, _ in train_info.items():
            train_info[k] /= num_updates

        return train_info

    def save(self, save_dir):
        super().save(save_dir)
        torch.save(self.critic_optimizer.state_dict(), f"{str(save_dir)}/critic_optimizer.pt")

    def restore(self, save_dir):
        super().restore(save_dir)
        self.critic_optimizer.load_state_dict(torch.load(f"{str(save_dir)}/critic_optimizer.pt"))