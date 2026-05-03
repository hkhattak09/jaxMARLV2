"""MACA-Role configuration.

Completely standalone config for MACA-Role experiments.
Does NOT import from config.py so hyperparameters can be changed independently.
"""

from __future__ import annotations


# Explicit experiment definitions
MACA_ROLE_EXPERIMENTS = {
    1: {
        "name": "post_gru_heads_shared_critic",
        "description": "Post-GRU actor role heads + shared critic (baseline)",
        "use_pre_gru_routes": False,
        "use_role_critic": False,
    },
    2: {
        "name": "post_gru_heads_role_critic",
        "description": "Post-GRU actor role heads + role-specific critic heads",
        "use_pre_gru_routes": False,
        "use_role_critic": True,
    },
    3: {
        "name": "pre_gru_routes_shared_critic",
        "description": "Pre-GRU residual routes + post-GRU heads + shared critic",
        "use_pre_gru_routes": True,
        "use_role_critic": False,
    },
    4: {
        "name": "pre_gru_routes_role_critic",
        "description": "Pre-GRU residual routes + post-GRU heads + role-specific critic heads",
        "use_pre_gru_routes": True,
        "use_role_critic": True,
    },
}


def get_default_maca_role_config():
    """Return default MACA-Role configuration."""
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
        "activation_func": "relu",
        "initialization_method": "orthogonal_",
        "gain": 0.01,
        "recurrent_n": 1,
        "use_naive_recurrent_policy": False,
        "use_recurrent_policy": True,
        "use_feature_normalization": True,

        # === Transformer Config ===
        "transformer": {
            "n_embd": 64,
            "n_head": 1,
            "n_encode_layer": 1,
            "n_decode_layer": 0,
            "n_block": None,
            "bias": True,
            "dropout": 0.0,
            "active_fn": "gelu",
            "weight_init": "tfixup",
            "zs_dim": 256,
            "vq_bsln_coef": 0.3,
            "vq_coma_bsln_coef": 0.3,
            "att_sigma": 1.0,
            "att_roll_res": False,
            "aggregation": "mean",
            "output_attentions": True,
            "warmup_epochs": 10,
            "wght_decay": 0.01,
            "betas": [0.9, 0.95],
            "min_lr": 0.00005,
            "q_value_loss_coef": 0.5,
            "eq_value_loss_coef": 1.0,
            "next_s_pred_loss_coef": 0.0,
            "is_causal": False,
        },

        # === Training ===
        "LR": 0.0005,
        "CRITIC_LR": 0.0005,
        "ANNEAL_LR": False,
        "USE_CRITIC_LR_DECAY": True,
        "CLIP_PARAM": 0.1,
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
        "Q_VALUE_LOSS_COEF": 0.5,
        "EQ_VALUE_LOSS_COEF": 1.0,
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
        "huber_delta": 10.0,
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

        # === Role Architecture (MACA-Role specific) ===
        "ROLE_EXPERIMENT": 1,
        "N_ROLES": 6,
        "USE_PRE_GRU_ROUTES": False,
        "USE_ROLE_CRITIC": False,

        # === Actor Role-Specific Dimensions ===
        "role_route_hidden_dim": 128,          # Pre-GRU route intermediate dim (Exp 3/4)
        "role_head_hidden_dims": [64, 32],     # Post-GRU head MLP layers

        # === Critic Role-Specific Dimensions ===
        "role_z_k_dims": [128, 64],            # Per-role z_k projection: Dense(128)→ReLU→Dense(64)
        "role_v_head_dims": [64, 64],         # V-head MLP layers: Dense(128)→ReLU→Dense(64)→ReLU→Dense(1)

        # === KL Diversity (actor) ===
        "USE_KL_DIVERSITY": True,
        "KL_DIVERSITY_WEIGHT": 0.001,
        "KL_DECAY_FRACTION": 0.3,

        # === Critic Diversity (role-specific heads only) ===
        "USE_CRITIC_DIVERSITY": True,
        "CRITIC_DIVERSITY_COEF": 1e-4,
    }


def validate_maca_role_config(config, num_agents):
    """Validate MACA-Role config, including role-specific invariants."""
    # Ensure backward-compatible aliases
    if "PPO_EPOCH" not in config and "UPDATE_EPOCHS" in config:
        config["PPO_EPOCH"] = config["UPDATE_EPOCHS"]
    if "UPDATE_EPOCHS" not in config and "PPO_EPOCH" in config:
        config["UPDATE_EPOCHS"] = config["PPO_EPOCH"]
    if "ACTOR_NUM_MINI_BATCH" not in config and "NUM_MINIBATCHES" in config:
        config["ACTOR_NUM_MINI_BATCH"] = config["NUM_MINIBATCHES"]
    if "NUM_MINIBATCHES" not in config and "ACTOR_NUM_MINI_BATCH" in config:
        config["NUM_MINIBATCHES"] = config["ACTOR_NUM_MINI_BATCH"]

    num_steps = config["NUM_STEPS"]
    num_envs = config["NUM_ENVS"]

    use_naive_recurrent = config.get("use_naive_recurrent_policy", False)
    use_recurrent = config.get("use_recurrent_policy", False)
    if use_naive_recurrent:
        raise NotImplementedError(
            "use_naive_recurrent_policy=True is not implemented. "
            "Use use_recurrent_policy=True instead."
        )
    if use_recurrent and config.get("recurrent_n", 1) != 1:
        raise NotImplementedError("recurrent_n=1 is the only supported value.")

    actor_num_mini_batch = config.get("ACTOR_NUM_MINI_BATCH", config.get("NUM_MINIBATCHES", 1))
    if actor_num_mini_batch <= 0:
        raise ValueError("ACTOR_NUM_MINI_BATCH must be positive.")
    critic_num_mini_batch = config.get("CRITIC_NUM_MINI_BATCH", 1)
    if critic_num_mini_batch <= 0:
        raise ValueError("CRITIC_NUM_MINI_BATCH must be positive.")

    if use_recurrent:
        data_chunk_length = config.get("DATA_CHUNK_LENGTH", num_steps)
        if data_chunk_length <= 0:
            raise ValueError("DATA_CHUNK_LENGTH must be positive for recurrent minibatches.")
        if num_steps % data_chunk_length != 0:
            raise ValueError(
                f"NUM_STEPS({num_steps}) must be divisible by DATA_CHUNK_LENGTH({data_chunk_length})."
            )
        actor_chunks = num_envs * num_agents * (num_steps // data_chunk_length)
        if actor_chunks % actor_num_mini_batch != 0:
            raise ValueError(
                f"Recurrent actor minibatch invalid: {actor_chunks} must be divisible by "
                f"ACTOR_NUM_MINI_BATCH({actor_num_mini_batch})"
            )
        critic_chunks = num_envs * (num_steps // data_chunk_length)
        if critic_chunks % critic_num_mini_batch != 0:
            raise ValueError(
                f"Recurrent critic minibatch invalid: {critic_chunks} must be divisible by "
                f"CRITIC_NUM_MINI_BATCH({critic_num_mini_batch})"
            )
    else:
        actor_batch_size = num_steps * num_envs * num_agents
        if actor_batch_size % actor_num_mini_batch != 0:
            raise ValueError(
                f"Actor minibatch invalid: {actor_batch_size} must be divisible by "
                f"ACTOR_NUM_MINI_BATCH({actor_num_mini_batch})"
            )
        critic_batch_size = num_steps * num_envs
        if critic_batch_size % critic_num_mini_batch != 0:
            raise ValueError(
                f"Critic minibatch invalid: {critic_batch_size} must be divisible by "
                f"CRITIC_NUM_MINI_BATCH({critic_num_mini_batch})"
            )

    # MACA-Role specific validation
    exp = config.get("ROLE_EXPERIMENT", 1)
    if exp not in MACA_ROLE_EXPERIMENTS:
        raise ValueError(
            f"ROLE_EXPERIMENT must be one of {list(MACA_ROLE_EXPERIMENTS.keys())}, got {exp}"
        )

    exp_cfg = MACA_ROLE_EXPERIMENTS[exp]
    config["USE_ROLE_CRITIC"] = exp_cfg["use_role_critic"]
    config["USE_PRE_GRU_ROUTES"] = exp_cfg["use_pre_gru_routes"]

    if config.get("USE_CRITIC_DIVERSITY", False) and not config["USE_ROLE_CRITIC"]:
        raise ValueError(
            "USE_CRITIC_DIVERSITY=True requires a role-specific critic "
            f"(ROLE_EXPERIMENT=2 or 4, got {exp})"
        )

    return config
