from harl.envs.smac.smac_logger import SMACLogger


class SMACv2Logger(SMACLogger):
    def __init__(
            self, args, algo_args, env_args, num_agents,
            sacred_run, console_logger,
        ):
        super(SMACv2Logger, self).__init__(
            args, algo_args, env_args, num_agents,
            sacred_run, console_logger,
        )
        self.win_key = "battle_won"
