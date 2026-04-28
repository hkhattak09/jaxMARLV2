import argparse
import json
from typing import Dict


SUPPORTED_ADAPTER_MODES = (
    "none",
    "role_lora",
    "global_lora",
    "agent_lora",
    "role_maca_official",
)


def default_config():
    return {
        "LR": 0.002,
        "NUM_ENVS": 128,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": int(3e6),
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "SCALE_CLIP_EPS": False,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,
        "ACTIVATION": "relu",
        "OBS_WITH_AGENT_ID": True,
        "ENV_NAME": "HeuristicEnemySMAX",
        "MAP_NAME": "3m",
        "SEED": 42,
        "RUN_NAME": "rosa_mappo",
        "ADAPTER_MODE": "none",
        "NUM_UNIT_TYPES": 6,
        "ROLE_ID_SOURCE": "env_state_unit_type",
        "USE_ROLE_LORA": False,
        "ROLE_LORA_RANK": 4,
        "ROLE_LORA_SCALE": 1.0,
        "ROLE_LORA_A_INIT_STD": 0.01,
        "ROLE_MACA_BLEND_ALPHA": 0.15,
        "ROLE_MACA_BLEND_SCHEDULE": "constant",
        "ROLE_MACA_BLEND_WARMUP_STEPS": 300000,
        "ROLE_MACA_BLEND_RAMP_STEPS": 700000,
        "ROLE_MACA_BLEND_NORMALIZE": False,
        "ROLE_MACA_EMBED_DIM": 64,
        "ROLE_MACA_Z_DIM": 256,
        "ROLE_MACA_ATT_SIGMA": 1.0,
        "ROLE_MACA_SELF_COEF": 0.3,
        "ROLE_MACA_GROUP_COEF": 0.3,
        "ROLE_MACA_VALUE_LOSS_COEF": 1.0,
        "ROLE_MACA_Q_LOSS_COEF": 0.5,
        "ROLE_MACA_EQ_LOSS_COEF": 1.0,
        "LOG_ROLE_DIAGNOSTICS": True,
        "LOG_ROLE_DIAGNOSTIC_TABLE": False,
        "ENV_KWARGS": {
            "see_enemy_actions": True,
            "walls_cause_death": True,
            "attack_mode": "closest",
        },
        "ANNEAL_LR": True,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="ROSA/Role-LoRA MAPPO experiment runner for SMAX."
    )
    parser.add_argument("--map_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--adapter_mode", type=str, choices=SUPPORTED_ADAPTER_MODES, default=None)
    parser.add_argument("--role_lora_rank", type=int, default=None)
    parser.add_argument("--role_lora_scale", type=float, default=None)
    parser.add_argument("--role_maca_blend_alpha", type=float, default=None)
    parser.add_argument(
        "--role_maca_blend_schedule",
        type=str,
        choices=("constant", "linear_warmup"),
        default=None,
    )
    parser.add_argument("--role_maca_blend_warmup_steps", type=int, default=None)
    parser.add_argument("--role_maca_blend_ramp_steps", type=int, default=None)
    parser.add_argument(
        "--role_maca_raw_blend",
        action="store_true",
        help="Blend raw GAE and MACA advantages instead of standardizing each component first.",
    )
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--num_minibatches", type=int, default=None)
    parser.add_argument("--update_epochs", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def apply_cli_overrides(config: Dict, args):
    if args.map_name is not None:
        config["MAP_NAME"] = args.map_name
    if args.seed is not None:
        config["SEED"] = args.seed
    if args.total_timesteps is not None:
        config["TOTAL_TIMESTEPS"] = args.total_timesteps
    if args.role_lora_rank is not None:
        config["ROLE_LORA_RANK"] = args.role_lora_rank
    if args.role_lora_scale is not None:
        config["ROLE_LORA_SCALE"] = args.role_lora_scale
    if args.role_maca_blend_alpha is not None:
        config["ROLE_MACA_BLEND_ALPHA"] = args.role_maca_blend_alpha
    if args.role_maca_blend_schedule is not None:
        config["ROLE_MACA_BLEND_SCHEDULE"] = args.role_maca_blend_schedule
    if args.role_maca_blend_warmup_steps is not None:
        config["ROLE_MACA_BLEND_WARMUP_STEPS"] = args.role_maca_blend_warmup_steps
    if args.role_maca_blend_ramp_steps is not None:
        config["ROLE_MACA_BLEND_RAMP_STEPS"] = args.role_maca_blend_ramp_steps
    if args.role_maca_raw_blend:
        config["ROLE_MACA_BLEND_NORMALIZE"] = False
    if args.num_envs is not None:
        config["NUM_ENVS"] = args.num_envs
    if args.num_steps is not None:
        config["NUM_STEPS"] = args.num_steps
    if args.num_minibatches is not None:
        config["NUM_MINIBATCHES"] = args.num_minibatches
    if args.update_epochs is not None:
        config["UPDATE_EPOCHS"] = args.update_epochs
    if args.run_name is not None:
        config["RUN_NAME"] = args.run_name
    if args.adapter_mode is not None:
        config["ADAPTER_MODE"] = args.adapter_mode

    adapter_mode = config["ADAPTER_MODE"]
    if adapter_mode not in SUPPORTED_ADAPTER_MODES:
        raise ValueError(
            f"Unsupported adapter_mode={adapter_mode!r}. "
            f"Supported modes: {', '.join(SUPPORTED_ADAPTER_MODES)}"
        )
    if config["ROLE_MACA_BLEND_SCHEDULE"] not in {"constant", "linear_warmup"}:
        raise ValueError(
            f"Unsupported ROLE_MACA_BLEND_SCHEDULE={config['ROLE_MACA_BLEND_SCHEDULE']!r}. "
            "Supported schedules: constant, linear_warmup"
        )
    if config["ROLE_MACA_BLEND_WARMUP_STEPS"] < 0:
        raise ValueError("ROLE_MACA_BLEND_WARMUP_STEPS must be non-negative.")
    if config["ROLE_MACA_BLEND_RAMP_STEPS"] <= 0:
        raise ValueError("ROLE_MACA_BLEND_RAMP_STEPS must be positive.")
    for coef_key in ("ROLE_MACA_SELF_COEF", "ROLE_MACA_GROUP_COEF"):
        if not 0.0 <= config[coef_key] <= 1.0:
            raise ValueError(f"{coef_key} must be in [0, 1].")
    config["USE_ROLE_LORA"] = adapter_mode in {
        "role_lora",
        "global_lora",
        "agent_lora",
        "role_maca_official",
    }
    config["USE_ROLE_MACA"] = adapter_mode == "role_maca_official"
    return config


def print_resolved_config(config: Dict):
    role_lora = {
        key: value
        for key, value in sorted(config.items())
        if key.startswith("ROLE_") or key in ("ADAPTER_MODE", "USE_ROLE_LORA", "USE_ROLE_MACA")
    }
    print("Resolved config:")
    print(json.dumps(config, indent=2, sort_keys=True))
    print(f"adapter_mode: {config['ADAPTER_MODE']}")
    print(f"map_name: {config['MAP_NAME']}")
    print(f"seed: {config['SEED']}")
    print(f"run_name: {config['RUN_NAME']}")
    print("role/adapters config values:")
    print(json.dumps(role_lora, indent=2, sort_keys=True))
