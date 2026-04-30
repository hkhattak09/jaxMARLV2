"""MAPPO Configuration.

The defaults follow the MACA MAPPO paper hyperparameter tables.
"""


def get_default_mappo_config():
    """Return the default MAPPO configuration.

    This config combines model, training, and algorithm hyperparameters
    for the MAPPO implementation using the SMAX environment.

    Returns:
        dict: Complete configuration dictionary.
    """
    return {
        # === Environment ===
        "ENV_NAME": "HeuristicEnemySMAX",
        "MAP_NAME": "protoss_10_vs_10",
        "NUM_ENVS": 128,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": int(4e7),

        # === Model Architecture ===
        "hidden_sizes": [128, 128],
        "activation_func": "relu",
        "initialization_method": "orthogonal_",
        "gain": 0.01,
        "recurrent_n": 1,
        "use_naive_recurrent_policy": False,
        "use_recurrent_policy": True,
        "use_feature_normalization": True,

        # === Training ===
        "lr": 0.0005,
        "critic_lr": 0.0005,
        "ANNEAL_LR": False,
        "use_linear_lr_decay": False,
        "ppo_epoch": 5,
        "critic_epoch": 5,
        "clip_param": 0.2,
        "SCALE_CLIP_EPS": False,
        "scale_clip_eps": False,
        "actor_num_mini_batch": 1,
        "critic_num_mini_batch": 1,
        "entropy_coef": 0.01,
        "value_loss_coef": 1,
        "use_max_grad_norm": True,
        "max_grad_norm": 10.0,
        "use_clipped_value_loss": True,
        "use_huber_loss": True,
        "huber_delta": 10.0,
        "use_policy_active_masks": True,
        "action_aggregation": "prod",
        "share_param": True,
        "fixed_order": True,
        "use_proper_time_limits": True,

        # === Value Normalization ===
        "use_valuenorm": True,

        # === GAE ===
        "gamma": 0.99,
        "gae_lambda": 0.95,

        # === Optimizer ===
        "opti_eps": 1e-5,
        "weight_decay": 0.0,

        # === Observation ===
        "OBS_WITH_AGENT_ID": True,
        "obs_with_agent_id": True,

        # === Environment kwargs ===
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
