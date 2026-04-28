"""MAPPO-T Configuration.

The defaults follow the MAPPO-T/MACA paper hyperparameter tables.  The code
still supports overriding these values for larger JAX runs, including using
multiple recurrent mini-batches.
"""

def get_default_mappo_t_config():
    """Return the default MAPPO-T configuration.
    
    This config combines model, training, and algorithm hyperparameters
    for the MAPPO-T implementation using SMAX environment.
    
    Returns:
        dict: Complete configuration dictionary.
    """
    return {
        # === Environment ===
        "ENV_NAME": "HeuristicEnemySMAX",
        "MAP_NAME": "protoss_10_vs_10",
        "NUM_ENVS": 20,
        "NUM_STEPS": 200,
        "TOTAL_TIMESTEPS": int(1e7),
        "SAVE_INTERVAL": 1000000,   # Timesteps between checkpoint saves
        
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
            "n_embd": 64,            # Embedding dimension
            "n_head": 1,              # Number of attention heads
            "n_encode_layer": 1,       # Number of encoder blocks (n_block = num_agents)
            "n_decode_layer": 0,       # MACA MAPPO-T default uses no decoder blocks
            "n_block": None,           # Will be set to num_agents
            "bias": True,
            "dropout": 0.0,
            "active_fn": "gelu",
            "weight_init": "tfixup",  # Options: default, tfixup, nanogpt
            "zs_dim": 256,             # State representation dimension
            "vq_bsln_coef": 0.3,      # VQ baseline weight
            "vq_coma_bsln_coef": 0.3, # VQ-COMA baseline weight
            "att_sigma": 1.0,          # Attention sigma for mixing
            "att_roll_res": False,      # Attention rollout residual
            "aggregation": "mean",      # Attention aggregation method
            "output_attentions": True,  # Output attention weights
            "warmup_epochs": 10,       # LR warmup epochs
            "wght_decay": 0.01,
            "betas": [0.9, 0.95],
            "min_lr": 0.00005,         # Minimum learning rate
            "q_value_loss_coef": 0.5,  # Q value loss coefficient
            "eq_value_loss_coef": 1.0, # EQ value loss coefficient
            "next_s_pred_loss_coef": 0.0, # Next state prediction loss (disabled)
            "is_causal": False,        # Causal attention (for decoder)
        },
        
        # === Training ===
        "LR": 0.0005,
        "CRITIC_LR": 0.0005,
        "ANNEAL_LR": False,
        "USE_CRITIC_LR_DECAY": True,
        "CLIP_PARAM": 0.1,        # PPO clip parameter
        "SCALE_CLIP_EPS": False,   # Scale clip epsilon by num_agents
        "PPO_EPOCH": 10,           # Actor PPO epochs
        "UPDATE_EPOCHS": 10,       # Backward-compatible alias for PPO_EPOCH
        "ACTOR_NUM_MINI_BATCH": 1, # Paper actor minibatch count; can be increased
        "NUM_MINIBATCHES": 1,      # Backward-compatible alias for ACTOR_NUM_MINI_BATCH
        "CRITIC_EPOCH": 10,        # Critic update epochs
        "CRITIC_NUM_MINI_BATCH": 1,# Paper critic minibatch count; can be increased
        "DATA_CHUNK_LENGTH": 10,    # For recurrent generator
        
        # === Value Normalization ===
        "use_valuenorm": True,      # Use ValueNorm for v/q/eq critics (MACA default: True)
        
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
        # MACA's SMAC wrapper exposes agent IDs in local observations by default
        # (obs_agent_id=True). MAPPO-T feeds local obs to both actor and
        # transformer critic, so this must affect local obs, not only the
        # centralized world_state used by older GRU baselines.
        "OBS_WITH_AGENT_ID": True,
        "LOCAL_OBS_WITH_AGENT_ID": True,
        "ENV_KWARGS": {
            "see_enemy_actions": True,
            "walls_cause_death": True,
            "attack_mode": "closest",
            # Match MACA's SMACv2 paper profile, where episode_length=200 is
            # also the battle horizon. JaxMARL SMAX defaults to 100.
            "max_steps": 200,
            # Opt-in SMACv2 parity flags for race scenarios
            "smacv2_unit_stats": True,
            "smacv2_position_parity": True,
            "reward_mode": "smacv2",
            "movement_mode": "smacv2",
        },
        
        # === Seed ===
        "SEED": 42,
    }


def get_transformer_config(num_agents):
    """Get transformer config with num_agents set as n_block.
    
    Args:
        num_agents: Number of agents in the environment.
        
    Returns:
        dict: Transformer configuration with n_block set.
    """
    config = get_default_mappo_t_config()
    config["transformer"]["n_block"] = num_agents
    return config


def validate_mappo_t_config(config, num_agents):
    """Validate MAPPO-T configuration for minibatch divisibility.
    
    Args:
        config: MAPPO-T configuration dictionary.
        num_agents: Number of agents in the environment.
        
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
    
    num_steps = config["NUM_STEPS"]
    num_envs = config["NUM_ENVS"]
    
    use_naive_recurrent = config.get("use_naive_recurrent_policy", False)
    use_recurrent = config.get("use_recurrent_policy", False)
    if use_naive_recurrent:
        raise NotImplementedError(
            "use_naive_recurrent_policy=True is not implemented in this JAX MAPPO-T trainer. "
            "Use the paper setting use_recurrent_policy=True instead."
        )
    if use_recurrent and config.get("recurrent_n", 1) != 1:
        raise NotImplementedError(
            "This JAX MAPPO-T trainer currently supports recurrent_n=1, matching the paper."
        )

    actor_num_mini_batch = config.get("ACTOR_NUM_MINI_BATCH", config.get("NUM_MINIBATCHES", 1))
    if actor_num_mini_batch <= 0:
        raise ValueError("ACTOR_NUM_MINI_BATCH must be positive.")
    critic_num_mini_batch = config.get("CRITIC_NUM_MINI_BATCH", 1)
    if critic_num_mini_batch <= 0:
        raise ValueError("CRITIC_NUM_MINI_BATCH must be positive.")

    if use_recurrent:
        data_chunk_length = config.get("DATA_CHUNK_LENGTH", num_steps)
        if data_chunk_length <= 0:
            raise ValueError("DATA_CHUNK_LENGTH must be positive for recurrent MAPPO-T minibatches.")
        if num_steps % data_chunk_length != 0:
            raise ValueError(
                f"NUM_STEPS({num_steps}) must be divisible by DATA_CHUNK_LENGTH({data_chunk_length}) "
                "for recurrent MAPPO-T minibatches."
            )
        actor_chunks = num_envs * num_agents * (num_steps // data_chunk_length)
        if actor_chunks % actor_num_mini_batch != 0:
            raise ValueError(
                f"Recurrent actor minibatch config invalid: NUM_ENVS({num_envs}) * "
                f"num_agents({num_agents}) * NUM_STEPS/DATA_CHUNK_LENGTH"
                f"({num_steps // data_chunk_length}) = {actor_chunks} must be divisible by "
                f"ACTOR_NUM_MINI_BATCH({actor_num_mini_batch})"
            )
        critic_chunks = num_envs * (num_steps // data_chunk_length)
        if critic_chunks % critic_num_mini_batch != 0:
            raise ValueError(
                f"Recurrent critic minibatch config invalid: NUM_ENVS({num_envs}) * "
                f"NUM_STEPS/DATA_CHUNK_LENGTH({num_steps // data_chunk_length}) = "
                f"{critic_chunks} must be divisible by CRITIC_NUM_MINI_BATCH({critic_num_mini_batch})"
            )
    else:
        # Actor minibatch validation: flatten over (time, env, agent)
        actor_batch_size = num_steps * num_envs * num_agents
        if actor_batch_size % actor_num_mini_batch != 0:
            raise ValueError(
                f"Actor minibatch config invalid: NUM_STEPS({num_steps}) * "
                f"NUM_ENVS({num_envs}) * num_agents({num_agents}) = {actor_batch_size} "
                f"must be divisible by ACTOR_NUM_MINI_BATCH({actor_num_mini_batch})"
            )
        
        # Critic minibatch validation: flatten over (time, env) only, preserve agent axis
        critic_batch_size = num_steps * num_envs
        if critic_batch_size % critic_num_mini_batch != 0:
            raise ValueError(
                f"Critic minibatch config invalid: NUM_STEPS({num_steps}) * "
                f"NUM_ENVS({num_envs}) = {critic_batch_size} "
                f"must be divisible by CRITIC_NUM_MINI_BATCH({critic_num_mini_batch})"
            )
    
    return config
