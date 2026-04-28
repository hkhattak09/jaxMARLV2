"""Tools for loading and updating configs."""
import time
import os
import json
import yaml
import torch as th
from uu import Error
import datetime


def get_defaults_yaml_args(algo, env):
    """Load config file for user-specified algo and env.
    Args:
        algo: (str) Algorithm name.
        env: (str) Environment name.
    Returns:
        algo_args: (dict) Algorithm config.
        env_args: (dict) Environment config.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "configs", "envs_cfgs", f"{env}.yaml")

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)
    return algo_args, env_args


def update_args(unparsed_dict, *args):
    """Update loaded config with unparsed command-line arguments.
    Args:
        unparsed_dict: (dict) Unparsed command-line arguments.
        *args: (list[dict]) argument dicts to be updated.
    """

    def update_dict(dict1, dict2):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(dict1, dict2[k])
            else:
                if k in dict1:
                    dict2[k] = dict1[k]

    for args_dict in args:
        update_dict(unparsed_dict, args_dict)


def get_task_name(env, env_args):
    """Get task name."""
    if env == "smac":
        task = env_args["map_name"]
    elif env == "smacv2":
        task = env_args["map_name"]
    elif env == "mamujoco":
        task = f"{env_args['scenario']}-{env_args['agent_conf']}"
    elif env == "pettingzoo_mpe":
        if env_args["continuous_actions"]:
            task = f"{env_args['scenario']}-continuous"
        else:
            task = f"{env_args['scenario']}-discrete"
    elif env == "gym":
        task = env_args["scenario"]
    return task


def init_dir(env, env_args, algo, exp_name, seed, hms_time, logger_path):
    """Init directory for saving results."""
    task = get_task_name(env, env_args)
    exp_path = [
        env,
        task,
        algo,
        f"{exp_name}_seed{seed}_time{hms_time}"
    ]

    run_path = os.path.join(logger_path, "slurm", *exp_path)
    models_path = os.path.join(logger_path, "models", *exp_path)
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    return run_path, models_path


def is_json_serializable(value):
    """Check if v is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except Error:
        return False


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def save_config(args, algo_args, env_args, run_dir):
    """Save the configuration of the program."""
    config = {"main_args": args, "algo_args": algo_args, "env_args": env_args}
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(",", ": "), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as out:
        out.write(output)
    return config_json


def args_sanity_check(config, main_args, console_logger):
    # set CUDA flags. Use cuda whenever possible!
    if config["device"]["cuda"] and not th.cuda.is_available():
        config["device"]["cuda"] = False
        console_logger.warning(
            "CUDA flag cuda was switched OFF automatically because no CUDA devices are available!"
        )

    # set eval_episodes to be a multiple of n_eval_rollout_threads
    if config["eval"]["eval_episodes"] < config["eval"]["n_eval_rollout_threads"]:
        config["eval"]["eval_episodes"] = config["eval"]["n_eval_rollout_threads"]
    else:
        config["eval"]["eval_episodes"] = (
            config["eval"]["eval_episodes"] // config["eval"]["n_eval_rollout_threads"]
        ) * config["eval"]["n_eval_rollout_threads"]

    # set eval_interval to be a multiple of log_interval
    if config["train"]["eval_interval"] < config["train"]["log_interval"]:
        config["train"]["eval_interval"] = config["train"]["log_interval"]
    else:
        config["train"]["eval_interval"] = (
            config["train"]["eval_interval"] // config["train"]["log_interval"]
        ) * config["train"]["log_interval"]

    # set save_interval to be 10*eval_interval if not specified
    if config["train"].get("save_interval", 0) == 0:
        config["train"]["save_interval"] = 10 * config["train"]["eval_interval"]

    # set current time if not specified
    if not main_args["hms_time"]:
        main_args["hms_time"] = datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S-%f")
