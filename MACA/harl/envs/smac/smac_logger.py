from functools import reduce
import numpy as np
from harl.common.base_logger import BaseLogger


class SMACLogger(BaseLogger):
    def __init__(
            self, args, algo_args, env_args, num_agents,
            sacred_run, console_logger,
        ):
        super(SMACLogger, self).__init__(
            args, algo_args, env_args, num_agents,
            sacred_run, console_logger,
        )
        self.win_key = "won"

    def get_task_name(self):
        return self.env_args["map_name"]

    def init(self):
        super().init()
        self.last_battles_game = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.float32
        )
        self.last_battles_won = np.zeros(
            self.algo_args["train"]["n_rollout_threads"], dtype=np.float32
        )

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer,
        curr_timestep=None,
    ):
        if curr_timestep is not None:
            self.curr_timestep = curr_timestep
        else:
            self.curr_timestep = (
                self.episode
                * self.algo_args["train"]["episode_length"]
                * self.algo_args["train"]["n_rollout_threads"]
            )

        battles_won = []
        battles_game = []
        incre_battles_won = []
        incre_battles_game = []
        is_smacv1 = False

        for i, info in enumerate(self.train_infos):
            if "battles_won" in info[0].keys():
                is_smacv1 = True
                battles_won.append(info[0]["battles_won"])
                incre_battles_won.append(
                    info[0]["battles_won"] - self.last_battles_won[i]
                )
            if "battles_game" in info[0].keys():
                battles_game.append(info[0]["battles_game"])
                incre_battles_game.append(
                    info[0]["battles_game"] - self.last_battles_game[i]
                )

        if is_smacv1:
            incre_win_rate = (
                np.sum(incre_battles_won) / np.sum(incre_battles_game)
                if np.sum(incre_battles_game) > 0
                else 0.0
            )
            self.log_stat("incre_win_rate", incre_win_rate, self.curr_timestep)
            self.log_stat("incre_game", np.sum(incre_battles_game), self.curr_timestep)

            self.last_battles_game = battles_game
            self.last_battles_won = battles_won

        for agent_id in range(self.num_agents):
            actor_train_infos[agent_id]["dead_ratio"] = 1 - actor_buffer[
                agent_id
            ].active_masks.sum() / (
                self.num_agents
                * reduce(
                    lambda x, y: x * y, list(actor_buffer[agent_id].active_masks.shape)
                )
            )

        super().episode_log(
            actor_train_infos, critic_train_info, actor_buffer, critic_buffer,
            curr_timestep,
        )

    def eval_init(self):
        super().eval_init()
        self.eval_battles_won = 0

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        if self.eval_infos[tid][0][self.win_key] == True:
            self.eval_battles_won += 1

    def eval_log(self, eval_episode):
        eval_win_rate = self.eval_battles_won / eval_episode
        eval_env_infos = {
            "eval_win_rate": [eval_win_rate],
        }
        self.log_env(eval_env_infos)

        super().eval_log(eval_episode)
