"""Tiny smoke tests for MAPPO-T unified checkpoint warm-starting.

Verifies that train_mappo_t.py can load the repo's single-file checkpoint format
and use it to warm-start a minimal continued MAPPO-T run.

Example:
    python test_scripts/test_mappo_t_resume_load.py
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "smax_ctm"))

from mappo_t import ActorTrans, TransVCritic, get_default_mappo_t_config
from mappo_t.actor import ScannedRNN
from mappo_t.valuenorm import create_value_norm_dict
from train_mappo_t import SMAXWorldStateWrapper, _load_pretrained_state, make_train


def _build_tiny_config():
    config = get_default_mappo_t_config()
    config["MAP_NAME"] = "protoss_5_vs_5"
    config["NUM_ENVS"] = 1
    config["NUM_STEPS"] = 2
    config["DATA_CHUNK_LENGTH"] = 2
    config["TOTAL_TIMESTEPS"] = 2
    config["PPO_EPOCH"] = 1
    config["CRITIC_EPOCH"] = 1
    config["ACTOR_NUM_MINI_BATCH"] = 1
    config["CRITIC_NUM_MINI_BATCH"] = 1
    config["SAVE_INTERVAL"] = 1000000
    config["USE_EVAL"] = False
    config["ANNEAL_LR"] = False
    config["USE_CRITIC_LR_DECAY"] = False
    config["use_valuenorm"] = True
    return config


def _create_unified_checkpoint(config):
    from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario

    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config.get("ENV_KWARGS", {}))
    env = SMAXWorldStateWrapper(
        env,
        obs_with_agent_id=config["OBS_WITH_AGENT_ID"],
        local_obs_with_agent_id=config.get(
            "LOCAL_OBS_WITH_AGENT_ID", config["OBS_WITH_AGENT_ID"]
        ),
    )

    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    actor_hidden_dim = config["hidden_sizes"][-1]
    critic_hidden_dim = config["transformer"]["n_embd"]

    actor = ActorTrans(action_dim=action_dim, config=config)
    critic = TransVCritic(
        config=config,
        share_obs_space=None,
        obs_space=env.observation_space(env.agents[0]),
        act_space=env.action_space(env.agents[0]),
        num_agents=env.num_agents,
        state_type="EP",
    )

    rng = jax.random.PRNGKey(123)
    rng, actor_rng, critic_rng = jax.random.split(rng, 3)

    actor_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], actor_hidden_dim)
    actor_x = (
        jnp.zeros((1, config["NUM_ENVS"], obs_dim), dtype=jnp.float32),
        jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
        jnp.ones((1, config["NUM_ENVS"], action_dim), dtype=jnp.float32),
    )
    actor_params = actor.init(actor_rng, actor_hstate, actor_x)

    critic_hstate = jnp.zeros(
        (config["NUM_ENVS"], env.num_agents, critic_hidden_dim), dtype=jnp.float32
    )
    critic_params = critic.init(
        critic_rng,
        jnp.zeros((config["NUM_ENVS"], env.num_agents, obs_dim), dtype=jnp.float32),
        jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=jnp.int32),
        jnp.ones((config["NUM_ENVS"], env.num_agents, action_dim), dtype=jnp.float32)
        / action_dim,
        critic_hstate,
        jnp.zeros((config["NUM_ENVS"], env.num_agents), dtype=bool),
        True,
        True,
    )

    value_norm_dict = create_value_norm_dict(
        use_valuenorm=True,
        v_shape=(1,),
        q_shape=(1,),
        eq_shape=(1,),
    )

    return {
        "model_type": "mappo_t_backbone",
        "format_version": 1,
        "checkpoint_kind": "test",
        "step": 0,
        "update": 0,
        "config": config,
        "actor_params": actor_params,
        "critic_params": critic_params,
        "value_norm_dict": value_norm_dict,
    }


def test_load_unified_checkpoint():
    config = _build_tiny_config()
    checkpoint = _create_unified_checkpoint(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pkl")
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

        config["PRETRAINED_CHECKPOINT_PATH"] = path
        actor_params, critic_params, value_norm_dict = _load_pretrained_state(config)

    assert set(flatten_dict(actor_params)) == set(flatten_dict(checkpoint["actor_params"]))
    assert set(flatten_dict(critic_params)) == set(flatten_dict(checkpoint["critic_params"]))
    assert set(value_norm_dict.keys()) == set(checkpoint["value_norm_dict"].keys())
    print("PASS: unified checkpoint load")


def test_missing_valuenorm_fails_loudly():
    config = _build_tiny_config()
    checkpoint = _create_unified_checkpoint(config)
    checkpoint.pop("value_norm_dict")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pkl")
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

        config["PRETRAINED_CHECKPOINT_PATH"] = path
        try:
            _load_pretrained_state(config)
        except ValueError as exc:
            assert "value_norm_dict" in str(exc)
        else:
            raise AssertionError("Missing value_norm_dict did not fail loudly")

    print("PASS: missing ValueNorm fails loudly")


def test_warm_start_train_smoke():
    config = _build_tiny_config()
    checkpoint = _create_unified_checkpoint(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pkl")
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

        config["PRETRAINED_CHECKPOINT_PATH"] = path
        actor_params, critic_params, value_norm_dict = _load_pretrained_state(config)
        train_fn = make_train(config)
        train_jit = jax.jit(train_fn)
        out = train_jit(
            jax.random.PRNGKey(config["SEED"]),
            actor_params,
            critic_params,
            value_norm_dict,
        )
        jax.block_until_ready(out)

    assert "runner_state" in out
    assert "metric" in out
    assert "loss" in out["metric"]
    print("PASS: warm-start train smoke")


if __name__ == "__main__":
    test_load_unified_checkpoint()
    test_missing_valuenorm_fails_loudly()
    test_warm_start_train_smoke()
    print("\nAll MAPPO-T resume-load tests passed!")
