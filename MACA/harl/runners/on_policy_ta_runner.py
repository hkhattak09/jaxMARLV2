"""Runner for on-policy MA algorithms."""
import numpy as np
import torch
import os
from harl.common.valuenorm import ValueNorm
from harl.common.buffers import CRITIC_BUFFER_REGISTRY, OnPolicyActorBuffer
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics import CRITIC_REGISTRY
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config, get_task_name
from harl.envs import LOGGER_REGISTRY
from harl.runners.on_policy_ma_runner import OnPolicyMARunner


class OnPolicyTARunner(OnPolicyMARunner):
    """Runner for on-policy Transformer-based MA algorithms."""

    def __init__(self, args, algo_args, env_args, sacred_run, console_logger):
        super().__init__(args, algo_args, env_args, sacred_run, console_logger)
        assert self.state_type == "EP", "OnPolicyTARunner only supports EP state type."

    def init_train(self, sacred_run, console_logger):
        if self.algo_args["render"]["use_render"]:
            return

        # train, not render
        self.actor_buffer = []
        for agent_id in range(self.num_agents):
            ac_bu = OnPolicyActorBuffer(
                {**self.algo_args["train"], **self.algo_args["model"]},
                self.envs.observation_space[agent_id],
                self.envs.action_space[agent_id],
            )
            self.actor_buffer.append(ac_bu)

        share_observation_space = self.envs.share_observation_space[0]
        observation_space = self.envs.observation_space[0]
        action_space = self.envs.action_space[0]
        self.critic = CRITIC_REGISTRY[self.args["algo"]](
            {**self.algo_args["train"], **self.algo_args["model"], **self.algo_args["algo"]},
            share_observation_space,
            observation_space,
            action_space,
            self.num_agents,
            self.state_type,
            device=self.device,
        )
        console_logger.info(f"critic number of parameters: {self.critic.get_num_params()}")
        self.critic_buffer = CRITIC_BUFFER_REGISTRY[f"{self.state_type}_full"](
            {**self.algo_args["train"], **self.algo_args["model"], **self.algo_args["algo"]},
            share_observation_space,
            observation_space,
            action_space,
            self.num_agents,
        )

        # ValueNorm is updated with return data. Not used for vq pred for now.
        self.value_types = ["v", "q", "eq"]
        if self.algo_args["train"].get("use_valuenorm", True):
            self.value_normalizer = {
                "v": ValueNorm(1, device=self.device),
                "q": ValueNorm(1, device=self.device),
                "eq": ValueNorm(1, device=self.device),
            }
        else:
            self.value_normalizer = dict()

        self.logger = LOGGER_REGISTRY[self.args["env"]](
            self.args, self.algo_args, self.env_args, self.num_agents, sacred_run, console_logger,
        )

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["render"]["use_render"] is True:
            self.render()
            return
        self.logger.console_logger.info("start running")
        self.warmup()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        self.logger.init()  # logger callback at the beginning of training

        for episode in range(episodes):
            if self.algo_args["train"][
                "use_linear_lr_decay"
            ]:  # linear decay of learning rate
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                self.critic.lr_decay(episode, episodes)
            if self.algo_args["train"].get("use_critic_lr_decay", False):
                self.critic.lr_decay(episode, episodes)

            self.logger.episode_init(
                episode
            )  # logger callback at the beginning of each episode

            self.prep_rollout()  # change to eval mode
            for step in range(self.algo_args["train"]["episode_length"]):
                # Sample actions from actors and values from critics
                (
                    values,
                    q_values,
                    eq_values,
                    vq_values,
                    vq_coma_values,
                    bsln_weights,
                    attn_weights,
                    actions,
                    action_log_probs,
                    policy_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
                # actions: (n_threads, n_agents, action_dim)
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                data = (
                    share_obs,
                    obs,
                    actions,
                    action_log_probs,
                    available_actions,
                    rewards,
                    dones,
                    infos,
                    values,
                    rnn_states,
                    rnn_states_critic,
                    None,
                    None,
                    None,
                )

                self.logger.per_step(data)  # logger callback at each step

                data = (
                    share_obs,
                    obs,
                    actions,
                    action_log_probs,
                    policy_probs,   # unneeded by logger
                    available_actions,
                    rewards,
                    dones,
                    infos,
                    values,
                    q_values,
                    eq_values,
                    vq_values,
                    vq_coma_values,
                    bsln_weights,
                    attn_weights,
                    rnn_states,
                    rnn_states_critic,
                    None,
                    None,
                    None,
                )
                self.insert(data)  # insert data into buffer

            # compute return and update network
            self.compute()
            self.prep_training()  # change to train mode

            actor_train_infos, critic_train_info = self.train()

            # log information
            if (
                episode % self.algo_args["train"]["log_interval"] == 0
                or episode == episodes - 1
            ):
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )

            # eval
            if (
                episode % self.algo_args["train"]["eval_interval"] == 0
                or episode == episodes - 1
            ):
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()

            # save model
            if (
                episode % self.algo_args["train"]["save_interval"] == 0
                or episode == episodes - 1
            ):
                self.save(episode=episode)

            self.after_update()
        self.logger.console_logger.info("Finished Training")

    def warmup(self):
        """Warm up the replay buffer."""
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()

        self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
        self.critic_buffer.obs[0] = obs.copy()
        if self.critic_buffer.available_actions is not None:
            self.critic_buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        """Collect actions and values from actors and critics.
        Args:
            step: step in the episode.
        Returns:
            values, actions, action_log_probs, rnn_states, rnn_states_critic
        """
        # collect actions, action_log_probs, rnn_states from n actors
        action_collector = []
        action_log_prob_collector = []
        policy_prob_collector = []
        rnn_state_collector = []
        for agent_id in range(self.num_agents):
            action, action_log_prob, policy_prob, rnn_state = self.actor[agent_id].get_actions(
                self.actor_buffer[agent_id].obs[step],
                self.actor_buffer[agent_id].rnn_states[step],
                self.actor_buffer[agent_id].masks[step],
                self.actor_buffer[agent_id].available_actions[step]
                if self.actor_buffer[agent_id].available_actions is not None
                else None,
            )
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            policy_prob_collector.append(_t2n(policy_prob))
            rnn_state_collector.append(_t2n(rnn_state))
        # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        policy_probs = np.array(policy_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)

        # collect values, rnn_states_critic from 1 critic
        (
            value,
            q_value,
            eq_value,
            vq_value,
            vq_coma_value,
            bsln_weight,
            attn_weight,
            rnn_state_critic,
        ) = self.critic.get_values(
            self.critic_buffer.obs[step],
            actions,
            policy_probs,
            self.critic_buffer.rnn_states_critic[step],
            self.critic_buffer.masks[step],
        )
        # (n_threads, dim)
        values = _t2n(value)
        q_values = _t2n(q_value)
        eq_values = _t2n(eq_value)
        vq_values = _t2n(vq_value)
        vq_coma_values = _t2n(vq_coma_value)
        bsln_weights = _t2n(bsln_weight)
        attn_weights = _t2n(attn_weight)
        rnn_states_critic = _t2n(rnn_state_critic)

        return (
            values,
            q_values,
            eq_values,
            vq_values,
            vq_coma_values,
            bsln_weights,
            attn_weights,
            actions,
            action_log_probs,
            policy_probs,
            rnn_states,
            rnn_states_critic,
        )

    def insert(self, data):
        """Insert data into buffer."""
        (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_threads, n_agents, obs_dim)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
            policy_probs,  # (n_threads, n_agents, n_action)
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_number)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            q_values,   # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            eq_values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            vq_values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            vq_coma_values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            bsln_weights,  # (n_threads, n_agents, 3)
            attn_weights,  # (n_threads, n_agents, n_agents)
            rnn_states,  # (n_threads, n_agents, dim)
            rnn_states_critic,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            next_share_obs,  # (n_threads, n_agents, next_share_obs_dim)
            next_obs,  # (n_threads, n_agents, next_obs_dim)
            next_available_actions,  # None or (n_agents, n_threads, next_action_number)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        rnn_states[
            dones_env == True
        ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # If env is done, then reset rnn_state_critic to all zero
        rnn_states_critic[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # masks use 0 to mask out threads that just finish.
        # this is used for denoting at which point should rnn state be reset
        masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # active_masks use 0 to mask out agents that have died
        active_masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # bad_masks use 0 to denote truncation and 1 to denote termination
        bad_masks = np.array(
            [
                [0.0]
                if "bad_transition" in info[0].keys()
                and info[0]["bad_transition"] == True
                else [1.0]
                for info in infos
            ]
        )

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

        self.critic_buffer.insert(
            share_obs[:, 0],
            rnn_states_critic,
            values,
            q_values,
            eq_values,
            vq_values,
            vq_coma_values,
            bsln_weights,
            attn_weights,
            rewards[:, 0],
            masks[:, 0],
            bad_masks,
            obs,
            rnn_states,
            actions,
            policy_probs,
            active_masks,
            available_actions if available_actions[0] is not None else None,
        )

    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages.
        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        (
            next_value,
            next_q_value,
            next_eq_value,
            next_vq_value,
            next_vq_coma_value,
            _,
            _,
            _,
        ) = self.critic.get_values(
            self.critic_buffer.obs[-1],
            self.critic_buffer.actions[-1],
            self.critic_buffer.policy_probs[-1],
            self.critic_buffer.rnn_states_critic[-1],
            self.critic_buffer.masks[-1],
        )
        next_values = {
            "v": _t2n(next_value),
            "q": _t2n(next_q_value),
            "eq": _t2n(next_eq_value),
            "vq": _t2n(next_vq_value),
            "vq_coma": _t2n(next_vq_coma_value),
        }
        for k in self.value_types:
            self.critic_buffer.compute_returns(next_values[k], k, self.value_normalizer.get(k, None))

    def train(self):
        """Training procedure for MAPPO."""
        actor_train_infos = []

        # compute advantages
        if len(self.value_normalizer):
            values = self.value_normalizer["v"].denormalize(self.critic_buffer.value_preds[:-1])
            q_values = self.value_normalizer["q"].denormalize(self.critic_buffer.q_value_preds[:-1])
            eq_values = self.value_normalizer["eq"].denormalize(self.critic_buffer.eq_value_preds[:-1])
        else:
            values = self.critic_buffer.value_preds[:-1]
            q_values = self.critic_buffer.q_value_preds[:-1]
            eq_values = self.critic_buffer.eq_value_preds[:-1]
        eq_values = eq_values[..., None, :]
        vq_values = self.critic_buffer.vq_value_preds[:-1]
        vq_coma_values = self.critic_buffer.vq_coma_value_preds[:-1]
        bsln_weights = self.critic_buffer.bsln_weights
        baselines = (
            bsln_weights[..., (0,)] * vq_coma_values
            + bsln_weights[..., (1,)] * vq_values
            + bsln_weights[..., (2,)] * eq_values
        )

        returns = self.critic_buffer.eq_returns[:-1]
        returns = returns[..., None, :]
        advantages = returns - baselines    # (episode_len, n_threads, n_agents, 1)

        # normalize advantages
        active_masks_collector = [
            self.actor_buffer[i].active_masks for i in range(self.num_agents)
        ]
        active_masks_array = np.stack(active_masks_collector, axis=2)
        advantages_copy = advantages.copy()
        advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # update actors
        if self.share_param:
            actor_train_info = self.actor[0].share_param_train(
                self.actor_buffer, advantages.copy(), self.num_agents, self.state_type
            )
            for _ in torch.randperm(self.num_agents):
                actor_train_infos.append(actor_train_info)
        else:
            for agent_id in range(self.num_agents):
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id],
                    advantages[:, :, agent_id].copy(), self.state_type
                )
                actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info

    def save(self, episode=None, timestep=None):
        """Save model parameters."""
        if timestep is None:
            assert episode is not None
            timestep = (
                episode
                * self.algo_args["train"]["episode_length"]
                * self.algo_args["train"]["n_rollout_threads"]
            )
        save_path = os.path.join(self.save_dir, str(timestep))
        os.makedirs(save_path, exist_ok=True)
        self.logger.console_logger.info(f"Saving models to {save_path}")
        for agent_id in range(self.num_agents):
            self.actor[agent_id].save(save_path, agent_id)
        self.critic.save(save_path)
        for k in self.value_normalizer:
            torch.save(
                self.value_normalizer[k].state_dict(),
                f"{str(save_path)}/{k}_value_normalizer.pt",
            )

    def restore(self, timestep=None):
        """Restore model parameters."""
        load_dir = self.algo_args["train"]["model_dir"]
        if not load_dir or load_dir != self.save_dir:
            return
        if not os.path.isdir(load_dir):
            self.logger.console_logger.info(
                f"The model checkpoint dir {load_dir} does not exist."            )
            return

        # Load model from the max timestep or closest to specifid timestep.
        timestep_to_load = 0
        timesteps = []

        # Go through all files in load_dir
        for name in os.listdir(load_dir):
            full_name = os.path.join(load_dir, name)
            # Check if they are dirs whose names are numbers
            if not os.path.isdir(full_name) or not name.isdigit():
                continue
            # Check if models in this dir are empty
            exist_empty_model = False
            for model in os.listdir(full_name):
                load_path = os.path.join(full_name, model)
                if os.path.getsize(load_path) == 0:
                    exist_empty_model = True
                    break
            if not exist_empty_model:
                timesteps.append(int(name))

        if timestep is None:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - timestep))

        load_path = os.path.join(load_dir, str(timestep_to_load))
        self.logger.console_logger.info(f"Loading model from {load_path}")
        for agent_id in range(self.num_agents):
            self.actor[agent_id].restore(load_path, agent_id)
        if not self.algo_args["render"]["use_render"]:
            self.critic.restore(load_path)
            for k in self.value_normalizer:
                self.value_normalizer[k].load_state_dict(
                    torch.load(f"{str(load_path)}/{k}_value_normalizer.pt")
                )
