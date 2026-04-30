"""HAPPO Configuration.

Defaults follow the MACA HAPPO hyperparameter settings.
"""


def get_default_happo_config():
    """Return the default HAPPO configuration.

    This config combines model, training, and algorithm hyperparameters
    for the HAPPO implementation using SMAX environment.

    Returns:
        dict: Complete configuration dictionary.
    """
    return {
        # === Environment ===
        "ENV_NAME": "HeuristicEnemySMAX",
        "MAP_NAME": "protoss_10_vs_10",
        "NUM_ENVS": 128,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": int(3e6),
        "SAVE_INTERVAL": 1000000,

        # === Model Architecture ===
        "hidden_sizes": [128, 128],
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "activation_func": "relu",
        "use_feature_normalization": True,
        "initialization_method": "orthogonal_",
        "gain": 0.01,
        "recurrent_n": 1,
        "use_naive_recurrent_policy": False,
        "use_recurrent_policy": True,

        # === Training ===
        "LR": 0.0005,
        "CRITIC_LR": 0.0005,
        "ANNEAL_LR": False,
        "CLIP_PARAM": 0.2,
        "SCALE_CLIP_EPS": False,
        "PPO_EPOCH": 5,
        "UPDATE_EPOCHS": 5,
        "ACTOR_NUM_MINI_BATCH": 1,
        "NUM_MINIBATCHES": 1,
        "CRITIC_EPOCH": 5,
        "CRITIC_NUM_MINI_BATCH": 1,

        # === Loss Coefficients ===
        "VALUE_LOSS_COEF": 1.0,
        "ENT_COEF": 0.01,
        "MAX_GRAD_NORM": 10.0,
        "use_max_grad_norm": True,
        "use_clipped_value_loss": True,
        "use_huber_loss": True,
        "huber_delta": 10.0,
        "use_policy_active_masks": True,
        "action_aggregation": "prod",
        "share_param": False,
        "fixed_order": False,
        "use_proper_time_limits": True,

        # === GAE ===
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "use_gae": True,

        # === Value Normalization (placeholder for future extension) ===
        "use_valuenorm": True,

        # === Optimizer ===
        "opti_eps": 1e-5,
        "weight_decay": 0.0,

        # === Observation ===
        "OBS_WITH_AGENT_ID": True,
        "ENV_KWARGS": {
            "see_enemy_actions": True,
            "walls_cause_death": True,
            "attack_mode": "closest",
            "max_steps": 200,
            "smacv2_unit_stats": True,
            "smacv2_position_parity": True,
            "reward_mode": "smacv2",
            "movement_mode": "smacv2",
        },

        # === Seed ===
        "SEED": 42,
    }


def validate_happo_config(config, num_agents):
    """Validate HAPPO configuration for minibatch divisibility.

    Args:
        config: HAPPO configuration dictionary.
        num_agents: Number of agents in the environment.

    Returns:
        dict: Validated configuration dictionary.

    Raises:
        ValueError: If minibatch configuration is invalid.
    """
    # Ensure backward-compatible aliases
    if "PPO_EPOCH" not in config and "UPDATE_EPOCHS" in config:
        config["PPO_EPOCH"] = config["UPDATE_EPOCHS"]
    if "UPDATE_EPOCHS" not in config and "PPO_EPOCH" in config:
        config["UPDATE_EPOCHS"] = config["PPO_EPOCH"]
    if "ACTOR_NUM_MINI_BATCH" not in config and "NUM_MINIBATCHES" in config:
        config["ACTOR_NUM_MINI_BATCH"] = config["NUM_MINIBATCHES"]
    if "NUM_MINIBATCHES" not in config and "ACTOR_NUM_MINI_BATCH" in config:
        config["NUM_MINIBATCHES"] = config["ACTOR_NUM_MINI_BATCH"]

    actor_num_mini_batch = config.get("ACTOR_NUM_MINI_BATCH", 1)
    critic_num_mini_batch = config.get("CRITIC_NUM_MINI_BATCH", 1)

    actor_batch_size = config["NUM_STEPS"] * config["NUM_ENVS"] * num_agents
    if actor_batch_size % actor_num_mini_batch != 0:
        raise ValueError(
            f"Actor batch size {actor_batch_size} must be divisible by "
            f"ACTOR_NUM_MINI_BATCH ({actor_num_mini_batch})"
        )

    critic_batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    if critic_batch_size % critic_num_mini_batch != 0:
        raise ValueError(
            f"Critic batch size {critic_batch_size} must be divisible by "
            f"CRITIC_NUM_MINI_BATCH ({critic_num_mini_batch})"
        )

    return config
