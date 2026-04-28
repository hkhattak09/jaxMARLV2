"""Base runner for on-policy algorithms."""

import time
import numpy as np
import torch
import setproctitle
import os
from pprint import pformat
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


class OnPolicyBaseRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args, sacred_run, console_logger):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.critic_hidden_x = algo_args["model"]["critic_hidden_x"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]
        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        if not self.algo_args["render"]["use_render"]:  # train, not render
            self.run_dir, self.save_dir = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                args["hms_time"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            config_json = save_config(args, algo_args, env_args, self.run_dir)
            config_formatted = pformat(config_json, indent=4, width=1)
            console_logger.info(f"Experiment Parameters:\n\n{config_formatted}\n")
        # set the title of the process
        self.task_name = get_task_name(args["env"], env_args)
        setproctitle.setproctitle(
            "-".join([
                args["env"],
                self.task_name,
                args["algo"],
                args["exp_name"],
                f'seed{algo_args["seed"]["seed"]}_time{args["hms_time"]}',
            ])
        )

        # set the config of env
        self.eval_deterministic = algo_args["eval"].get("eval_deterministic", True)
        if self.algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        console_logger.info(f"share_observation_space: {self.envs.share_observation_space}")
        console_logger.info(f"observation_space: {self.envs.observation_space}")
        console_logger.info(f"action_space: {self.envs.action_space}")

        # actor
        if self.share_param:
            self.actor = []
            agent = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            self.actor.append(agent)
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id]
                    == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])    # TODO add agent identity to produce diff behavior
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                self.actor.append(agent)

        self.init_train(sacred_run, console_logger)

        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()

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
        self.critic = CRITIC_REGISTRY[self.args["algo"]](
            {**self.algo_args["train"], **self.algo_args["model"], **self.algo_args["algo"]},
            share_observation_space,
            self.envs.action_space,
            self.num_agents,
            self.state_type,
            device=self.device,
        )
        console_logger.info(f"critic number of parameters: {self.critic.get_num_params()}")
        self.critic_buffer = CRITIC_BUFFER_REGISTRY[self.state_type](
            {**self.algo_args["train"], **self.algo_args["model"], **self.algo_args["algo"]},
            share_observation_space,
            self.num_agents,
        )

        if self.algo_args["train"].get("use_valuenorm", True):
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

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

            self.logger.episode_init(
                episode
            )  # logger callback at the beginning of each episode

            self.prep_rollout()  # change to eval mode
            for step in range(self.algo_args["train"]["episode_length"]):
                # Sample actions from actors and values from critics
                (
                    values,
                    actions,
                    action_log_probs,
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
        if self.state_type == "EP":
            self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = share_obs.copy()

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
        rnn_state_collector = []
        for agent_id in range(self.num_agents):
            action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
                self.actor_buffer[agent_id].obs[step],
                self.actor_buffer[agent_id].rnn_states[step],
                self.actor_buffer[agent_id].masks[step],
                self.actor_buffer[agent_id].available_actions[step]
                if self.actor_buffer[agent_id].available_actions is not None
                else None,
            )
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
        # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)

        # collect values, rnn_states_critic from 1 critic
        if self.state_type == "EP":
            value, rnn_state_critic = self.critic.get_values(
                self.critic_buffer.share_obs[step],
                self.critic_buffer.rnn_states_critic[step],
                self.critic_buffer.masks[step],
            )
            # (n_threads, dim)
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)
        elif self.state_type == "FP":
            value, rnn_state_critic = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[step]),
                np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                np.concatenate(self.critic_buffer.masks[step]),
            )  # concatenate (n_threads, n_agents, dim) into (n_threads * n_agents, dim)
            # split (n_threads * n_agents, dim) into (n_threads, n_agents, dim)
            values = np.array(
                np.split(_t2n(value), self.algo_args["train"]["n_rollout_threads"])
            )
            rnn_states_critic = np.array(
                np.split(
                    _t2n(rnn_state_critic), self.algo_args["train"]["n_rollout_threads"]
                )
            )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        """Insert data into buffer."""
        (
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            obs,  # (n_threads, n_agents, obs_dim)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_number)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
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
        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size * self.critic_hidden_x),
                dtype=np.float32,
            )
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size * self.critic_hidden_x,
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
        if self.state_type == "EP":
            bad_masks = np.array(
                [
                    [0.0]
                    if "bad_transition" in info[0].keys()
                    and info[0]["bad_transition"] == True
                    else [1.0]
                    for info in infos
                ]
            )
        elif self.state_type == "FP":
            bad_masks = np.array(
                [
                    [
                        [0.0]
                        if "bad_transition" in info[agent_id].keys()
                        and info[agent_id]["bad_transition"] == True
                        else [1.0]
                        for agent_id in range(self.num_agents)
                    ]
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

        if self.state_type == "EP":
            self.critic_buffer.insert(
                share_obs[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                share_obs, rnn_states_critic, values, rewards, masks, bad_masks
            )

    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages.
        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        if self.state_type == "EP":
            next_value, _ = self.critic.get_values(
                self.critic_buffer.share_obs[-1],
                self.critic_buffer.rnn_states_critic[-1],
                self.critic_buffer.masks[-1],
            )
            next_value = _t2n(next_value)
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[-1]),
                np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                np.concatenate(self.critic_buffer.masks[-1]),
            )
            next_value = np.array(
                np.split(_t2n(next_value), self.algo_args["train"]["n_rollout_threads"])
            )
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)

    def train(self):
        """Train the model."""
        raise NotImplementedError

    def after_update(self):
        """Do the necessary data operations after an update.
        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()

    @torch.no_grad()
    def eval(self):
        """Evaluate the model."""
        self.logger.eval_init()  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.algo_args["eval"]["n_eval_rollout_threads"],
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None
                    else None,
                    deterministic=self.eval_deterministic,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(
                eval_data
            )  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[
                eval_dones_env == True
            ] = np.zeros(  # if env is done, then reset rnn_state to all zero
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(
                        eval_i
                    )  # logger callback when an episode is done

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(
                    eval_episode
                )  # logger callback at the end of evaluation
                break
        self.logger.console_logger.info("Finished Evaluation")

    @torch.no_grad()
    def render(self):
        """Render the model."""
        print("start rendering")
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = (
                    np.expand_dims(np.array(eval_available_actions), axis=0)
                    if eval_available_actions is not None
                    else None
                )
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions is not None
                            else None,
                            deterministic=self.eval_deterministic,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = (
                        np.expand_dims(np.array(eval_available_actions), axis=0)
                        if eval_available_actions is not None
                        else None
                    )
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions[0] is not None
                            else None,
                            deterministic=self.eval_deterministic,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions)
                    rewards += eval_rewards[0][0][0]
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        if "smac" in self.args["env"]:  # replay for smac, no rendering
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def prep_rollout(self):
        """Prepare for rollout."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self):
        """Prepare for training."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
        self.critic.prep_training()

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
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                f"{str(save_path)}/value_normalizer.pt",
            )

    def restore(self, timestep=None):
        """Restore model parameters."""
        load_dir = self.algo_args["train"]["model_dir"]
        if not load_dir:
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
            if self.value_normalizer is not None:
                self.value_normalizer.load_state_dict(
                    torch.load(f"{str(load_path)}/value_normalizer.pt")
                )

    def close(self):
        """Close environment."""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
