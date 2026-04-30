"""IPPO Configuration.

The defaults follow the MACA IPPO hyperparameter configuration.
"""


def get_default_ippo_config():
    """Return the default IPPO configuration.

    Returns:
        dict: Complete configuration dictionary.
    """
    return {
        # === Environment ===
        "ENV_NAME": "HeuristicEnemySMAX",
        "MAP_NAME": "protoss_10_vs_10",
        "NUM_ENVS": 20,
        "NUM_STEPS": 200,
        "TOTAL_TIMESTEPS": int(4e7),
        "SAVE_INTERVAL": 1000000,

        # === Model Architecture ===
        "hidden_sizes": [64, 64, 64],
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "activation_func": "relu",
        "initialization_method": "orthogonal_",
        "gain": 0.01,
        "recurrent_n": 1,
        "use_naive_recurrent_policy": False,
        "use_recurrent_policy": True,
        "use_feature_normalization": True,

        # === Training ===
        "LR": 0.0005,
        "CRITIC_LR": 0.0005,
        "ANNEAL_LR": False,
        "CLIP_PARAM": 0.1,
        "CLIP_EPS": 0.1,
        "SCALE_CLIP_EPS": False,
        "PPO_EPOCH": 10,
        "UPDATE_EPOCHS": 10,
        "ACTOR_NUM_MINI_BATCH": 1,
        "NUM_MINIBATCHES": 1,
        "CRITIC_EPOCH": 10,
        "CRITIC_NUM_MINI_BATCH": 1,
        "DATA_CHUNK_LENGTH": 10,

        # === Value Normalization ===
        "use_valuenorm": True,

        # === Loss Coefficients ===
        "VALUE_LOSS_COEF": 1.0,
        "ENT_COEF": 0.01,
        "MAX_GRAD_NORM": 10.0,

        # === GAE ===
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,

        # === Optimizer ===
        "opti_eps": 1e-5,
        "weight_decay": 0.0,
        "use_max_grad_norm": True,
        "use_clipped_value_loss": True,
        "use_huber_loss": True,
        "HUBER_DELTA": 10.0,
        "use_policy_active_masks": True,
        "action_aggregation": "prod",
        "share_param": True,
        "fixed_order": True,
        "use_proper_time_limits": True,

        # === Observation ===
        "OBS_WITH_AGENT_ID": True,
        "LOCAL_OBS_WITH_AGENT_ID": True,
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
