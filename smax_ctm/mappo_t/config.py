"""MAPPO-T Configuration.

This module contains the default configuration for MAPPO-T,
ported from the MACA framework's YAML configs.
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
        "MAP_NAME": "3m",
        "NUM_ENVS": 20,
        "NUM_STEPS": 200,
        "TOTAL_TIMESTEPS": int(1e7),
        
        # === Model Architecture ===
        "hidden_sizes": [128, 128],
        "activation_func": "relu",
        "initialization_method": "orthogonal_",
        "gain": 0.01,
        "recurrent_n": 1,
        "use_naive_recurrent_policy": False,
        "use_recurrent_policy": False,
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
        "ACTOR_NUM_MINI_BATCH": 1, # Actor minibatch count
        "NUM_MINIBATCHES": 1,      # Backward-compatible alias for ACTOR_NUM_MINI_BATCH
        "CRITIC_EPOCH": 10,        # Critic update epochs
        "CRITIC_NUM_MINI_BATCH": 1,# Critic minibatch count
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
        "OBS_WITH_AGENT_ID": True,
        "ENV_KWARGS": {
            "see_enemy_actions": True,
            "walls_cause_death": True,
            "attack_mode": "closest",
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
    
    # Actor minibatch validation: flatten over (time, env, agent)
    actor_batch_size = num_steps * num_envs * num_agents
    actor_num_mini_batch = config.get("ACTOR_NUM_MINI_BATCH", config.get("NUM_MINIBATCHES", 1))
    if actor_batch_size % actor_num_mini_batch != 0:
        raise ValueError(
            f"Actor minibatch config invalid: NUM_STEPS({num_steps}) * "
            f"NUM_ENVS({num_envs}) * num_agents({num_agents}) = {actor_batch_size} "
            f"must be divisible by ACTOR_NUM_MINI_BATCH({actor_num_mini_batch})"
        )
    
    # Critic minibatch validation: flatten over (time, env) only, preserve agent axis
    critic_batch_size = num_steps * num_envs
    critic_num_mini_batch = config.get("CRITIC_NUM_MINI_BATCH", 1)
    if critic_batch_size % critic_num_mini_batch != 0:
        raise ValueError(
            f"Critic minibatch config invalid: NUM_STEPS({num_steps}) * "
            f"NUM_ENVS({num_envs}) = {critic_batch_size} "
            f"must be divisible by CRITIC_NUM_MINI_BATCH({critic_num_mini_batch})"
        )
    
    # Check recurrent policy constraints
    use_recurrent = (
        config.get("use_recurrent_policy", False)
        or config.get("use_naive_recurrent_policy", False)
    )
    if use_recurrent:
        raise NotImplementedError(
            "Recurrent MAPPO-T minibatches require sequence/chunk handling and are "
            "not implemented in this JAX trainer yet. Set use_recurrent_policy=False "
            "and use_naive_recurrent_policy=False."
        )
    
    return config
