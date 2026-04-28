"""MAPPO algorithm."""
import numpy as np
import torch
from harl.models.policy_models.stochastic_policy_t import StochasticPolicyTrans
from harl.algorithms.actors.mappo import MAPPO


class MAPPOTrans(MAPPO):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize MAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super().__init__(args, obs_space, act_space, device)
        # create actor network
        self.actor = StochasticPolicyTrans(args, self.obs_space, self.act_space, self.device)
        # create actor optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def get_actions(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False
    ):
        """Compute actions for the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor has RNN layer, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, action_log_probs, policy_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, action_log_probs, policy_probs, rnn_states_actor

    def act(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False
    ):
        """Compute actions using the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update for non-parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        # advantages already normalized
        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, actor_grad, imp_weights = self.update(sample)

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm.item()
                train_info["ratio"] += imp_weights.mean().item()
                self.grad_ms.update(actor_grad)

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        train_info["actor_grad_var"] = self.grad_ms.get_variance()
        self.grad_ms.reset()

        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        """Perform a training update for parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (list[OnPolicyActorBuffer]) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            num_agents: (int) number of agents.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        # advantages already normalized
        advantages_list = []
        for agent_id in range(num_agents):
            advantages_list.append(advantages[:, :, agent_id])

        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id],
                        self.actor_num_mini_batch,
                        self.data_chunk_length,
                    )
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                data_generators.append(data_generator)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(8)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(8):
                        batches[i].append(sample[i])
                for i in range(7):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[7][0] is None:
                    batches[7] = None
                else:
                    batches[7] = np.concatenate(batches[7], axis=0)
                policy_loss, dist_entropy, actor_grad_norm, actor_grad, imp_weights = self.update(
                    tuple(batches)
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm.item()
                train_info["ratio"] += imp_weights.mean().item()
                self.grad_ms.update(actor_grad)

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        train_info["actor_grad_var"] = self.grad_ms.get_variance()
        self.grad_ms.reset()

        return train_info
