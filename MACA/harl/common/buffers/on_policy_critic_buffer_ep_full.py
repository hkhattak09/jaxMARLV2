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


class OnPolicyCriticBufferEPFull(OnPolicyCriticBufferEP):
    """On-policy buffer for critic that uses Environment-Provided (EP) state."""

    def __init__(self, args, share_obs_space, obs_space, act_space, num_agents):
        """Initialize on-policy critic buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
        """
        super().__init__(args, share_obs_space, num_agents)

        obs_shape = get_shape_from_obs_space(obs_space)
        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]

        # Buffer for observations of each actor
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape),
            dtype=np.float32,
        )

        # Buffer for rnn states of each actor
        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

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

        # Buffer for Q predictions made by this critic
        self.q_value_preds = np.zeros_like(self.value_preds)

        # Buffer for expected Q predictions made by this critic
        self.eq_value_preds = np.zeros_like(self.value_preds)

        # Buffer for mixed V and Q predictions made by this critic
        # Used for training the actor but not the critic.
        self.vq_value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32
        )

        # Buffer for Coma mixed V and Q predictions made by this critic
        # Used for training the actor but not the critic.
        self.vq_coma_value_preds = np.zeros_like(self.vq_value_preds)

        # Buffer for Q returns calculated at each timestep.
        # Use different Q return targets to avoid learning trivial Q function.
        self.q_returns = np.zeros_like(self.returns)

        # Buffer for expected Q returns calculated at each timestep
        self.eq_returns = np.zeros_like(self.returns)

        # Buffer for mixed V and Q returns calculated at each timestep
        # Used for training the actor but not the critic.
        self.vq_returns = np.zeros_like(self.vq_value_preds)

        # Buffer for Coma mixed V and Q returns calculated at each timestep
        # Used for training the actor but not the critic.
        self.vq_coma_returns = np.zeros_like(self.vq_coma_value_preds)

        self.return_preds = {
            "v": self.value_preds,
            "q": self.q_value_preds,
            "eq": self.eq_value_preds,
            "vq": self.vq_value_preds,
            "vq_coma": self.vq_coma_value_preds,
        }
        self.return_targs = {
            "v": self.returns,
            "q": self.q_returns,
            "eq": self.eq_returns,
            "vq": self.vq_returns,
            "vq_coma": self.vq_coma_returns,
        }

        # Buffer for attention weights of each actor.
        self.bsln_weights = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 3), dtype=np.float32
        )

        # Buffer for attention rollout weights.
        self.attn_weights = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, num_agents), dtype=np.float32
        )

        # Buffer for available actions of each actor.
        act_dim = get_dim_from_act_space(act_space)
        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (self.episode_length + 1, self.n_rollout_threads, num_agents, act_dim),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)
        # Buffer for actions of each actor.
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32
        )

        # Buffer for action log probs of each actor.
        self.policy_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_dim), dtype=np.float32
        )

        # Buffer for active masks of each actor. Active masks denotes whether the agent is alive.
        self.active_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )

    def insert(
        self,
        share_obs,
        rnn_states_critic,
        value_preds,
        q_value_preds,
        eq_value_preds,
        vq_value_preds,
        vq_coma_value_preds,
        bsln_weights,
        attn_weights,
        rewards,
        masks,
        bad_masks,
        obs,
        rnn_states,
        actions,
        policy_probs,
        active_masks=None,
        available_actions=None,
    ):
        """Insert data into buffer."""
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.actions[self.step] = actions.copy()
        self.policy_probs[self.step] = policy_probs.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()
        self.q_value_preds[self.step] = q_value_preds.copy()
        self.eq_value_preds[self.step] = eq_value_preds.copy()
        self.vq_value_preds[self.step] = vq_value_preds.copy()
        self.vq_coma_value_preds[self.step] = vq_coma_value_preds.copy()
        self.bsln_weights[self.step] = bsln_weights.copy()
        self.attn_weights[self.step] = attn_weights.copy()

        super().insert(share_obs, rnn_states_critic, value_preds, rewards, masks, bad_masks)

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        super().after_update()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def compute_returns(self, next_value, return_type="v", value_normalizer=None):
        """Compute returns either as discounted sum of rewards, or using GAE.
        Args:
            next_value: (np.ndarray) value predictions for the step after the last episode step.
            return_type: (str) Return prediction type, one of ['v', 'q', 'eq'].
            value_normalizer: (ValueNorm) If not None, ValueNorm value normalizer instance.
        """
        return_preds = self.return_preds[return_type]
        return_targs = self.return_targs[return_type]
        rewards = self.rewards
        masks = self.masks
        bad_masks = self.bad_masks
        if return_type in ["vq", "vq_coma"]:
            rewards = rewards[..., None, :]
            masks = masks[..., None, :]
            bad_masks = bad_masks[..., None, :]

        if (
            self.use_proper_time_limits
        ):  # consider the difference between truncation and termination
            if self.use_gae:  # use GAE
                return_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(return_preds[step + 1])
                            * masks[step + 1]
                            - value_normalizer.denormalize(return_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * masks[step + 1] * gae
                        )
                        gae = bad_masks[step + 1] * gae
                        return_targs[step] = gae + value_normalizer.denormalize(
                            return_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            rewards[step]
                            + self.gamma
                            * return_preds[step + 1]
                            * masks[step + 1]
                            - return_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * masks[step + 1] * gae
                        )
                        gae = bad_masks[step + 1] * gae
                        return_targs[step] = gae + return_preds[step]
            else:  # do not use GAE
                return_targs[-1] = next_value
                for step in reversed(range(rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        return_targs[step] = (
                            return_targs[step + 1] * self.gamma * masks[step + 1]
                            + rewards[step]
                        ) * bad_masks[step + 1] + (
                            1 - bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            return_preds[step]
                        )
                    else:  # do not use ValueNorm
                        return_targs[step] = (
                            return_targs[step + 1] * self.gamma * masks[step + 1]
                            + rewards[step]
                        ) * bad_masks[step + 1] + (
                            1 - bad_masks[step + 1]
                        ) * return_preds[
                            step
                        ]
        else:  # do not consider the difference between truncation and termination, i.e. all done episodes are terminated
            if self.use_gae:  # use GAE
                return_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.shape[0])):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(return_preds[step + 1])
                            * masks[step + 1]
                            - value_normalizer.denormalize(return_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * masks[step + 1] * gae
                        )
                        return_targs[step] = gae + value_normalizer.denormalize(
                            return_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            rewards[step]
                            + self.gamma
                            * return_preds[step + 1]
                            * masks[step + 1]
                            - return_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * masks[step + 1] * gae
                        )
                        return_targs[step] = gae + return_preds[step]
            else:  # do not use GAE
                return_targs[-1] = next_value
                for step in reversed(range(rewards.shape[0])):
                    return_targs[step] = (
                        return_targs[step + 1] * self.gamma * masks[step + 1]
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
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        q_value_preds = self.q_value_preds[:-1].reshape(-1, 1)
        eq_value_preds = self.eq_value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        q_returns = self.q_returns[:-1].reshape(-1, 1)
        eq_returns = self.eq_returns[:-1].reshape(-1, 1)
        vq_returns = self.vq_returns[:-1].reshape(-1, *self.vq_returns.shape[2:])
        vq_coma_returns = self.vq_coma_returns[:-1].reshape(-1, *self.vq_coma_returns.shape[2:])
        rewards = self.rewards.reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])  # actually not used, just for consistency
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(
                -1, *self.available_actions.shape[2:]
            )
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        policy_probs = self.policy_probs.reshape(-1, *self.policy_probs.shape[2:])

        for indices in sampler:
            # share_obs shape:
            # (episode_length * n_rollout_threads, *share_obs_shape) --> (mini_batch_size, *share_obs_shape)
            share_obs_batch = share_obs[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            value_preds_batch = value_preds[indices]
            q_value_preds_batch = q_value_preds[indices]
            eq_value_preds_batch = eq_value_preds[indices]
            return_batch = returns[indices]
            q_return_batch = q_returns[indices]
            eq_return_batch = eq_returns[indices]
            vq_return_batch = vq_returns[indices]
            vq_coma_return_batch = vq_coma_returns[indices]
            rewards_batch = rewards[indices]
            masks_batch = masks[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            active_masks_batch = active_masks[indices]
            old_policy_probs_batch = policy_probs[indices]

            yield (
                min(mini_batch_size, self.episode_length),  # dummy
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
            q_value_preds_batch = _flatten(T, N, self.q_value_preds[:-1, ids])
            eq_value_preds_batch = _flatten(T, N, self.eq_value_preds[:-1, ids])
            return_batch = _flatten(T, N, self.returns[:-1, ids])
            q_return_batch = _flatten(T, N, self.q_returns[:-1, ids])
            eq_return_batch = _flatten(T, N, self.eq_returns[:-1, ids])
            vq_return_batch = _flatten(T, N, self.vq_returns[:-1, ids])
            vq_coma_return_batch = _flatten(T, N, self.vq_coma_returns[:-1, ids])
            rewards_batch = _flatten(T, N, self.rewards[:, ids])
            masks_batch = _flatten(T, N, self.masks[:-1, ids])
            rnn_states_critic_batch = self.rnn_states_critic[0, ids]
            obs_batch = _flatten(T, N, self.obs[:-1, ids])
            actions_batch = _flatten(T, N, self.actions[:, ids])
            active_masks_batch = _flatten(T, N, self.active_masks[:-1, ids])
            old_policy_probs_batch = _flatten(T, N, self.policy_probs[:, ids])
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, self.available_actions[:-1, ids])
            else:
                available_actions_batch = None
            rnn_states_batch = self.rnn_states[0, ids]

            yield (
                self.episode_length,
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
        value_preds = _sa_cast(self.value_preds[:-1])
        q_value_preds = _sa_cast(self.q_value_preds[:-1])
        eq_value_preds = _sa_cast(self.eq_value_preds[:-1])
        returns = _sa_cast(self.returns[:-1])
        q_returns = _sa_cast(self.q_returns[:-1])
        eq_returns = _sa_cast(self.eq_returns[:-1])
        vq_returns = _sa_cast_ma(self.vq_returns[:-1])
        vq_coma_returns = _sa_cast_ma(self.vq_coma_returns[:-1])
        rewards = _sa_cast(self.rewards)
        masks = _sa_cast(self.masks[:-1])
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 0, 2, 3, 4)
            .reshape(-1, *self.rnn_states_critic.shape[2:])
        )
        if len(self.obs.shape) > 4:
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4, 5).reshape(-1, *self.obs.shape[2:])
        else:
            obs = _sa_cast_ma(self.obs[:-1])
        actions = _sa_cast_ma(self.actions)
        policy_probs = _sa_cast_ma(self.policy_probs)
        active_masks = _sa_cast_ma(self.active_masks[:-1])
        rnn_states = (
            self.rnn_states[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.rnn_states.shape[2:])
        )
        if self.available_actions is not None:
            available_actions = _sa_cast_ma(self.available_actions[:-1])

        # generate mini-batches
        for indices in sampler:
            share_obs_batch = []
            rnn_states_critic_batch = []
            value_preds_batch = []
            q_value_preds_batch = []
            eq_value_preds_batch = []
            return_batch = []
            q_return_batch = []
            eq_return_batch = []
            vq_return_batch = []
            vq_coma_return_batch = []
            rewards_batch = []
            masks_batch = []
            obs_batch = []
            rnn_states_batch = []
            actions_batch = []
            available_actions_batch = []
            active_masks_batch = []
            old_policy_probs_batch = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                q_value_preds_batch.append(q_value_preds[ind : ind + data_chunk_length])
                eq_value_preds_batch.append(eq_value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                q_return_batch.append(q_returns[ind : ind + data_chunk_length])
                eq_return_batch.append(eq_returns[ind : ind + data_chunk_length])
                vq_return_batch.append(vq_returns[ind : ind + data_chunk_length])
                vq_coma_return_batch.append(vq_coma_returns[ind : ind + data_chunk_length])
                rewards_batch.append(rewards[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                rnn_states_critic_batch.append(
                    rnn_states_critic[ind]
                )  # only the beginning rnn states are needed
                obs_batch.append(obs[ind : ind + data_chunk_length])
                actions_batch.append(actions[ind : ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_policy_probs_batch.append(policy_probs[ind : ind + data_chunk_length])
                rnn_states_batch.append(rnn_states[ind])  # only the beginning rnn states are needed

            L, N = data_chunk_length, mini_batch_size
            # These are all ndarrays of size (data_chunk_length, mini_batch_size, *dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            q_value_preds_batch = np.stack(q_value_preds_batch, axis=1)
            eq_value_preds_batch = np.stack(eq_value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            q_return_batch = np.stack(q_return_batch, axis=1)
            eq_return_batch = np.stack(eq_return_batch, axis=1)
            vq_return_batch = np.stack(vq_return_batch, axis=1)
            vq_coma_return_batch = np.stack(vq_coma_return_batch, axis=1)
            rewards_batch = np.stack(rewards_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            # rnn_states_critic_batch is a (mini_batch_size, *dim) ndarray
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[2:]
            )
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_policy_probs_batch = np.stack(old_policy_probs_batch, axis=1)
            # rnn_states_batch is a (mini_batch_size, *dim) ndarray
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])

            # Flatten the (data_chunk_length, mini_batch_size, *dim) ndarrays to (data_chunk_length * mini_batch_size, *dim)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            q_value_preds_batch = _flatten(L, N, q_value_preds_batch)
            eq_value_preds_batch = _flatten(L, N, eq_value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            q_return_batch = _flatten(L, N, q_return_batch)
            eq_return_batch = _flatten(L, N, eq_return_batch)
            vq_return_batch = _flatten(L, N, vq_return_batch)
            vq_coma_return_batch = _flatten(L, N, vq_coma_return_batch)
            rewards_batch = _flatten(L, N, rewards_batch)
            masks_batch = _flatten(L, N, masks_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_policy_probs_batch = _flatten(L, N, old_policy_probs_batch)

            yield (
                data_chunk_length,
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
            )
