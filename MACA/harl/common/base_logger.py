"""Base logger."""

import time
import os
import numpy as np
from collections import defaultdict
from logging import getLogger, StreamHandler, Formatter


class BaseLogger:
    """Base logger class.
    Used for logging information in the on-policy training pipeline.
    """

    def __init__(
            self, args, algo_args, env_args, num_agents, sacred_run, console_logger,
        ):
        """Initialize the logger."""
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.task_name = self.get_task_name()
        self.num_agents = num_agents
        self.max_train_steps = self.algo_args["train"]["num_env_steps"]
        self.max_train_episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        self.console_logger = console_logger
        self.use_sacred = self.algo_args["logger"].get("use_sacred", True)
        self.use_wandb = self.algo_args["logger"].get("use_wandb", False)
        self.use_tb = self.algo_args["logger"].get("use_tb", False)
        if self.use_sacred:
            self.setup_sacred(sacred_run)
        if self.use_wandb:
            self.setup_wandb()
        if self.use_tb:
            self.setup_tb()

        self.stats = defaultdict(lambda: [])

    def get_task_name(self):
        """Get the task name."""
        raise NotImplementedError

    def init(self):
        """Initialize the logger."""
        self.console_logger.info(
            "Env: {}, Task: {}, Algo: {}, Exp: {}.\tStart training for {} episodes = {} steps.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.max_train_episodes,
                self.max_train_steps,
            )
        )
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []
        self.train_episode_lens = []
        self.train_one_episode_len = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.int64
        )
        self.train_epis_stats = defaultdict(lambda: 0)
        self.train_cumu_stats = defaultdict(lambda: 0)
        self.eval_epis_stats = defaultdict(lambda: 0)
        self.eval_cumu_stats = defaultdict(lambda: 0)
        self.train_num_episode = 0
        self.train_num_episode_log = 0
        self.last_eval_T = -self.algo_args["train"]["eval_interval"] - 1
        self.start_time = time.time()
        self.last_time = self.start_time

    def episode_init(self, episode):
        """Initialize the logger for each episode."""
        self.episode = episode
        # self.train_epis_stats = defaultdict(lambda: 0)

    def per_step(self, data):
        """Process data per step."""
        (
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
            next_share_obs,
            next_obs,
            next_available_actions,
        ) = data
        self.train_infos = infos
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env
        self.train_one_episode_len += 1
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                self.done_episodes_rewards.append(self.train_episode_rewards[t])
                self.train_episode_rewards[t] = 0
                self.train_episode_lens.append(self.train_one_episode_len[t].copy())
                self.train_one_episode_len[t] = 0
                self.train_num_episode += 1
                # self.update_epis_infos(self.train_epis_stats, self.train_infos, t)

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer,
        curr_timestep=None,
    ):
        """Log information for each episode."""
        if curr_timestep is not None:
            self.curr_timestep = curr_timestep
        else:
            self.curr_timestep = (
                self.episode
                * self.algo_args["train"]["episode_length"]
                * self.algo_args["train"]["n_rollout_threads"]
            )
        log_prefix = ""
        self.log_stat(f"{log_prefix}episode", self.episode, self.curr_timestep)

        step_rewards = critic_buffer.get_rewards()
        critic_train_info["step_rewards_mean"] = np.mean(step_rewards)
        critic_train_info["step_rewards_std"] = np.std(step_rewards)
        self.log_train(actor_train_infos, critic_train_info)

        if len(self.done_episodes_rewards) > 0:
            self.console_logger.info(
                f"Some episodes done, mean episode reward: {np.mean(self.done_episodes_rewards)}"
            )
            train_env_infos = {
                "episode_rewards_mean": self.done_episodes_rewards,
                "episode_rewards_max": [np.max(self.done_episodes_rewards)],
                "episode_rewards_std": [np.std(self.done_episodes_rewards)],
                "episode_lengths_mean": self.train_episode_lens,
                "episode_lengths_max": [np.max(self.train_episode_lens)],
            }
            self.log_env(train_env_infos, log_prefix)
            self.update_cumu_infos(self.train_cumu_stats, self.train_infos)
            self.log_env_infos(
                self.train_epis_stats,
                self.train_cumu_stats,
                self.train_num_episode - self.train_num_episode_log,
                log_prefix,
            )
            self.done_episodes_rewards = []
            self.train_episode_lens = []
            self.train_num_episode_log = self.train_num_episode
        self.print_recent_stats()

    def eval_init(self):
        """Initialize the logger for evaluation."""
        self.console_logger.info(
            "Episode: {}/{}, timestep: {}/{}.".format(
                self.episode,
                self.max_train_episodes,
                self.curr_timestep,
                self.max_train_steps,
            )
        )
        self.console_logger.info(
            "Estimated time left: {}, Time passed: {}, FPS: {}.".format(
                time_left(self.last_time, self.last_eval_T, self.curr_timestep, self.max_train_steps),
                time_str(time.time() - self.start_time),
                int(self.curr_timestep / (time.time() - self.start_time)),
            )
        )
        self.last_time = time.time()
        self.last_eval_T = self.curr_timestep

        log_prefix = "eval_"
        self.log_stat(f"{log_prefix}episode", self.episode, self.curr_timestep)
        self.log_stat(f"{log_prefix}timestep", self.curr_timestep, self.curr_timestep)

        self.eval_episode_rewards = np.zeros(
            self.algo_args["eval"]["n_eval_rollout_threads"]
        )
        self.eval_done_episodes_rewards = []
        self.eval_episode_lens = []
        self.eval_one_episode_len = np.zeros(
            self.algo_args["eval"]["n_eval_rollout_threads"], dtype=np.int64
        )
        self.eval_num_episode = 0
        # self.eval_epis_stats = defaultdict(lambda: 0)

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        self.eval_infos = eval_infos
        eval_dones_env = np.all(eval_dones, axis=1)
        eval_reward_env = np.mean(eval_rewards, axis=1).flatten()
        self.eval_episode_rewards += eval_reward_env
        self.eval_one_episode_len += 1
        for t in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            if eval_dones_env[t]:
                self.eval_done_episodes_rewards.append(self.eval_episode_rewards[t])
                self.eval_episode_rewards[t] = 0
                self.eval_episode_lens.append(self.eval_one_episode_len[t].copy())
                self.eval_one_episode_len[t] = 0
                self.eval_num_episode += 1
                # self.update_epis_infos(self.eval_epis_stats, self.eval_infos, t)

    def eval_thread_done(self, tid):
        """Log evaluation information when a thread is done."""
        pass

    def eval_log(self, eval_episode):
        """Log evaluation information."""
        log_prefix = "eval_"
        if len(self.eval_done_episodes_rewards) > 0:
            eval_env_infos = {
                "episode_rewards_mean": self.eval_done_episodes_rewards,
                "episode_rewards_max": [np.max(self.eval_done_episodes_rewards)],
                "episode_rewards_std": [np.std(self.eval_done_episodes_rewards)],
                "episode_lengths_mean": self.eval_episode_lens,
                "episode_lengths_max": [np.max(self.eval_episode_lens)],
            }
            self.log_env(eval_env_infos, log_prefix)
            self.update_cumu_infos(self.eval_cumu_stats, self.eval_infos)
            self.log_env_infos(
                self.eval_epis_stats,
                self.eval_cumu_stats,
                self.eval_num_episode,
                log_prefix,
            )
        self.print_recent_stats()
        self.console_logger.info("Finished Evaluation")

    def log_train(self, actor_train_infos, critic_train_info):
        """Log training information."""
        # log actor
        for agent_id in range(self.num_agents):
            for k, v in actor_train_infos[agent_id].items():
                self.log_stat(f"agent{agent_id}/{k}", v, self.curr_timestep)
        # log critic
        for k, v in critic_train_info.items():
            self.log_stat(f"critic/{k}", v, self.curr_timestep)

    def log_env(self, env_infos, prefix=""):
        """Log environment information."""
        for k, v in env_infos.items():
            if len(v) > 0:
                self.log_stat(f"{prefix}{k}", np.mean(v), self.curr_timestep)

    def update_epis_infos(self, epis_stats, infos, tid):
        """Update episodic environment information, i.e. info that is reset every episode.
        infos type: list, shape: (n_threads, n_agents)
        """
        key_set = set.union(*[set(d[0]) for d in infos]) - INFO_IGNORE - INFO_CUMULATIVE
        if not key_set:
            return
        for k in key_set:
            epis_stats[k] += infos[tid][0][k]

    def update_cumu_infos(self, cumu_stats, infos):
        """Update cumulative environment information, i.e. info that is accumulated over episodes."""
        key_set = set.union(*[set(d[0]) for d in infos]).intersection(INFO_CUMULATIVE)
        if not key_set:
            return
        cumu_stats.update({
            k: sum(d[0].get(k, 0) for d in infos) - cumu_stats[k]
            for k in key_set
        })

    def log_env_infos(self, epis_stats, cumu_stats, n_episode, prefix=""):
        for k, v in epis_stats.items() | cumu_stats.items():
            self.log_stat(f"{prefix}{k}", v/n_episode, self.curr_timestep)
        epis_stats.clear()

    def setup_tb(self):
        from tensorboard_logger import configure, log_value
        tb_logs_dir = os.path.join(
            self.algo_args["logger"]["log_dir"], "tb_logs",
            self.args["env"], self.task_name, self.args["algo"],
            self.args["exp_name"],
        )
        configure(tb_logs_dir)
        self.tb_logger = log_value

    def setup_wandb(self):
        import wandb
        wandb_logs_dir = os.path.join(
            self.algo_args["logger"]["log_dir"], "wandb_logs",
            self.args["env"], self.task_name, self.args["algo"],
            self.args["exp_name"],
        )
        os.makedirs(wandb_logs_dir, exist_ok=True)
        wandb.init(
            project=self.algo_args["logger"]["wandb_project"],
            entity=self.algo_args["logger"]["wandb_entity"],
            dir=wandb_logs_dir,
            config={
                "main_args": self.args,
                "algo_args": self.algo_args,
                "env_args": self.env_args,
            },
            mode=self.algo_args["logger"]["wandb_mode"],
        )
        wandb.run.name = (
            "_".join([
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                f'seed{self.algo_args["seed"]["seed"]}',
            ])
        )
        self.wandb_logger = wandb.log

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_wandb:
            self.wandb_logger({key: value,
                               f"{key}_step": t})

        if self.use_sacred:
            if key in self.sacred_info:
                self.sacred_info[f"{key}_T"].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info[f"{key}_T"] = [t]
                self.sacred_info[key] = [value]

            self._run_obj.log_scalar(key, value, t)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(np.mean([x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


INFO_IGNORE = {
    "original_obs",
    "original_state",
    "original_avail_actions",
}

INFO_CUMULATIVE = {
    "battles_won",
    "battles_game",
    "battles_draw",
    "restarts",
}

# set up a custom logger
def get_logger():
    logger = getLogger()
    logger.handlers = []
    ch = StreamHandler()
    formatter = Formatter("[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("DEBUG")

    return logger

def time_left(start_time, t_start, t_current, t_max):
    if t_current >= t_max:
        return "-"
    time_elapsed = time.time() - start_time
    t_current = max(1, t_current)
    time_left = time_elapsed * (t_max - t_current) / (t_current - t_start)
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    return time_str(time_left)

def time_str(s):
    """
    Convert seconds to a nicer string showing days, hours, minutes and seconds
    """
    days, remainder = divmod(s, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""
    if days > 0:
        string += "{:d} days, ".format(int(days))
    if hours > 0:
        string += "{:d} hours, ".format(int(hours))
    if minutes > 0:
        string += "{:d} minutes, ".format(int(minutes))
    string += "{:d} seconds".format(int(seconds))
    return string
