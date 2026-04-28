"""On-policy buffer for critic that uses Environment-Provided (EP) state."""
import torch
import numpy as np
from harl.utils.envs_tools import (
    get_shape_from_obs_space,
    get_shape_from_act_space,
    get_dim_from_act_space,
)
from harl.utils.trans_tools import _flatten, _sa_cast, _sa_cast_ma
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP


class OnPolicyCriticBufferEPComa(OnPolicyCriticBufferEP):
    """On-policy buffer for critic that uses Environment-Provided (EP) state."""

    def __init__(self, args, share_obs_space, act_space, num_agents):
        """Initialize on-policy critic buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
        """
        super().__init__(args, share_obs_space, num_agents)

        # Buffer for rnn states of critic
        self.rnn_states_critic = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # Use value_preds and returns to store Q values and Q returns
        # Buffer for Q predictions made by this critic
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32
        )

        # Buffer for returns calculated at each timestep
        self.returns = np.zeros_like(self.value_preds)

        # Buffer for mixed V and Q predictions made by this critic.
        # Used for training the actor but not the critic.
        self.vq_value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32
        )

        act_dim = get_dim_from_act_space(act_space)
        act_shape = get_shape_from_act_space(act_space)
        # Buffer for actions of each actor.
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32
        )

        # Buffer for action log probs of each actor.
        self.policy_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_dim), dtype=np.float32
        )

    def insert(
        self,
        share_obs,
        rnn_states_critic,
        value_preds,
        vq_value_preds,
        rewards,
        masks,
        bad_masks,
        actions,
        policy_probs,
    ):
        """Insert data into buffer."""
        self.actions[self.step] = actions.copy()
        self.policy_probs[self.step] = policy_probs.copy()
        self.vq_value_preds[self.step] = vq_value_preds.copy()

        super().insert(share_obs, rnn_states_critic, value_preds, rewards, masks, bad_masks)

    def compute_returns(self, next_value, value_normalizer=None):
        """Compute returns either as discounted sum of rewards, or using GAE.
        Args:
            next_value: (np.ndarray) value predictions for the step after the last episode step.
            value_normalizer: (ValueNorm) If not None, ValueNorm value normalizer instance.
        """
        rewards = self.rewards[..., None, :]
        masks = self.masks[..., None, :]
        bad_masks = self.bad_masks[..., None, :]

        if (
            self.use_proper_time_limits
        ):  # consider the difference between truncation and termination
            if self.use_gae:  # use GAE
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * masks[step + 1] * gae
                        )
                        gae = bad_masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * masks[step + 1] * gae
                        )
                        gae = bad_masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:  # do not use GAE
                self.returns[-1] = next_value
                for step in reversed(range(rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * masks[step + 1]
                            + rewards[step]
                        ) * bad_masks[step + 1] + (
                            1 - bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * masks[step + 1]
                            + rewards[step]
                        ) * bad_masks[step + 1] + (
                            1 - bad_masks[step + 1]
                        ) * self.value_preds[
                            step
                        ]
        else:  # do not consider the difference between truncation and termination, i.e. all done episodes are terminated
            if self.use_gae:  # use GAE
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * masks[step + 1] * gae
                        )
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * masks[step + 1] * gae
                        )
                        self.returns[step] = gae + self.value_preds[step]
            else:  # do not use GAE
                self.returns[-1] = next_value
                for step in reversed(range(rewards.shape[0])):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * masks[step + 1]
                        + rewards[step]
                    )

    def feed_forward_generator_critic(
        self, critic_num_mini_batch=None, mini_batch_size=None
    ):
        """Training data generator for critic that uses MLP network.
        Args:
            critic_num_mini_batch: (int) Number of mini batches for critic.
            mini_batch_size: (int) Size of mini batch for critic.
        """

        # get episode_length, n_rollout_threads, mini_batch_size
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            assert batch_size >= critic_num_mini_batch, (
                f"The number of processes ({n_rollout_threads}) "
                f"* number of steps ({episode_length}) = {n_rollout_threads * episode_length} "
                f"is required to be greater than or equal to the number of critic mini batches ({critic_num_mini_batch})."
            )
            mini_batch_size = batch_size // critic_num_mini_batch

        # shuffle indices
        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]

        # Combine the first two dimensions (episode_length and n_rollout_threads) to form batch.
        # Take share_obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *share_obs_shape) --> (episode_length, n_rollout_threads, *share_obs_shape)
        # --> (episode_length * n_rollout_threads, *share_obs_shape)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[2:]
        )  # actually not used, just for consistency
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        policy_probs = self.policy_probs.reshape(-1, *self.policy_probs.shape[2:])

        for indices in sampler:
            # share_obs shape:
            # (episode_length * n_rollout_threads, *share_obs_shape) --> (mini_batch_size, *share_obs_shape)
            share_obs_batch = share_obs[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            actions_batch = actions[indices]
            old_policy_probs_batch = policy_probs[indices]

            yield (
                share_obs_batch,
                rnn_states_critic_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                actions_batch,
                old_policy_probs_batch,
            )

    def naive_recurrent_generator_critic(self, critic_num_mini_batch):
        """Training data generator for critic that uses RNN network.
        This generator does not split the trajectories into chunks,
        and therefore maybe less efficient than the recurrent_generator_critic in training.
        Args:
            critic_num_mini_batch: (int) Number of mini batches for critic.
        """

        # get n_rollout_threads and num_envs_per_batch
        n_rollout_threads = self.rewards.shape[1]
        assert n_rollout_threads >= critic_num_mini_batch, (
            f"The number of processes ({n_rollout_threads}) "
            f"has to be greater than or equal to the number of "
            f"mini batches ({critic_num_mini_batch})."
        )
        num_envs_per_batch = n_rollout_threads // critic_num_mini_batch

        # shuffle indices
        perm = torch.randperm(n_rollout_threads).numpy()

        T, N = self.episode_length, num_envs_per_batch

        for batch_id in range(critic_num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[start_id : start_id + num_envs_per_batch]
            share_obs_batch = _flatten(T, N, self.share_obs[:-1, ids])
            value_preds_batch = _flatten(T, N, self.value_preds[:-1, ids])
            return_batch = _flatten(T, N, self.returns[:-1, ids])
            masks_batch = _flatten(T, N, self.masks[:-1, ids])
            rnn_states_critic_batch = self.rnn_states_critic[0, ids]
            actions_batch = _flatten(T, N, self.actions[:, ids])
            old_policy_probs_batch = _flatten(T, N, self.policy_probs[:, ids])

            yield (
                share_obs_batch,
                rnn_states_critic_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                actions_batch,
                old_policy_probs_batch,
            )

    def recurrent_generator_critic(self, critic_num_mini_batch, data_chunk_length):
        """Training data generator for critic that uses RNN network.
        This generator splits the trajectories into chunks of length data_chunk_length,
        and therefore maybe more efficient than the naive_recurrent_generator_actor in training.
        Args:
            critic_num_mini_batch: (int) Number of mini batches for critic.
            data_chunk_length: (int) Length of data chunks.
        """

        # get episode_length, n_rollout_threads, and mini_batch_size
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // critic_num_mini_batch

        assert (
            episode_length % data_chunk_length == 0
        ), f"episode length ({episode_length}) must be a multiple of data chunk length ({data_chunk_length})."
        assert data_chunks >= 2, "need larger batch size"

        # shuffle indices
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]

        # The following data operations first transpose the first two dimensions of the data (episode_length, n_rollout_threads)
        # to (n_rollout_threads, episode_length), then reshape the data to (n_rollout_threads * episode_length, *dim).
        # Take share_obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *share_obs_shape) --> (episode_length, n_rollout_threads, *share_obs_shape)
        # --> (n_rollout_threads, episode_length, *share_obs_shape) --> (n_rollout_threads * episode_length, *share_obs_shape)
        if len(self.share_obs.shape) > 3:
            share_obs = (
                self.share_obs[:-1]
                .transpose(1, 0, 2, 3, 4)
                .reshape(-1, *self.share_obs.shape[2:])
            )
        else:
            share_obs = _sa_cast(self.share_obs[:-1])
        value_preds = _sa_cast_ma(self.value_preds[:-1])
        returns = _sa_cast_ma(self.returns[:-1])
        masks = _sa_cast(self.masks[:-1])
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 0, 2, 3, 4)
            .reshape(-1, *self.rnn_states_critic.shape[2:])
        )
        actions = _sa_cast_ma(self.actions)
        policy_probs = _sa_cast_ma(self.policy_probs)

        # generate mini-batches
        for indices in sampler:
            share_obs_batch = []
            rnn_states_critic_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            actions_batch = []
            old_policy_probs_batch = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                rnn_states_critic_batch.append(
                    rnn_states_critic[ind]
                )  # only the beginning rnn states are needed
                actions_batch.append(actions[ind : ind + data_chunk_length])
                old_policy_probs_batch.append(policy_probs[ind : ind + data_chunk_length])

            L, N = data_chunk_length, mini_batch_size
            # These are all ndarrays of size (data_chunk_length, mini_batch_size, *dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            # rnn_states_critic_batch is a (mini_batch_size, *dim) ndarray
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[2:]
            )
            actions_batch = np.stack(actions_batch, axis=1)
            old_policy_probs_batch = np.stack(old_policy_probs_batch, axis=1)

            # Flatten the (data_chunk_length, mini_batch_size, *dim) ndarrays to (data_chunk_length * mini_batch_size, *dim)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            actions_batch = _flatten(L, N, actions_batch)
            old_policy_probs_batch = _flatten(L, N, old_policy_probs_batch)

            yield (
                share_obs_batch,
                rnn_states_critic_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                actions_batch,
                old_policy_probs_batch,
            )
