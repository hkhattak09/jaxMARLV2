"""
MAPPO GRU Baseline for SMAX
Colab-ready, dependency-light version (no Hydra/wandb).
"""
import os
import sys
import pickle
import argparse
import json
# Inject repo root into sys.path so 'jaxmarl' is always found regardless of CWD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
import functools
from flax.training.train_state import TrainState
import distrax
from functools import partial
import time

# You may need to adapt imports based on where this is running relative to JaxMARL
from jaxmarl.wrappers.baselines import SMAXLogWrapper, JaxMARLWrapper
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

SUPPORTED_ADAPTER_MODES = (
    "none",
    "role_lora",
    "global_lora",
    "agent_lora",
    "role_maca_lite_jnt",
    "role_maca_lite_ind",
    "role_maca_lite_role",
    "role_maca_lite",
)

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
    config["USE_ROLE_LORA"] = adapter_mode in {
        "role_lora",
        "global_lora",
        "agent_lora",
        "role_maca_lite_jnt",
        "role_maca_lite_ind",
        "role_maca_lite_role",
        "role_maca_lite",
    }
    config["USE_ROLE_MACA"] = adapter_mode in {
        "role_maca_lite_jnt",
        "role_maca_lite_ind",
        "role_maca_lite_role",
        "role_maca_lite",
    }
    config["ROLE_MACA_WEIGHT_JNT"] = 0.0
    config["ROLE_MACA_WEIGHT_IND"] = 0.0
    config["ROLE_MACA_WEIGHT_ROLE"] = 0.0
    if adapter_mode == "role_maca_lite_jnt":
        config["ROLE_MACA_WEIGHT_JNT"] = 1.0
    elif adapter_mode == "role_maca_lite_ind":
        config["ROLE_MACA_WEIGHT_IND"] = 1.0
    elif adapter_mode == "role_maca_lite_role":
        config["ROLE_MACA_WEIGHT_ROLE"] = 1.0
    elif adapter_mode == "role_maca_lite":
        config["ROLE_MACA_WEIGHT_JNT"] = 1.0 / 3.0
        config["ROLE_MACA_WEIGHT_IND"] = 1.0 / 3.0
        config["ROLE_MACA_WEIGHT_ROLE"] = 1.0 / 3.0
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

class SMAXWorldStateWrapper(JaxMARLWrapper):
    """Provides a 'world_state' observation for the centralised critic."""
    def __init__(self, env: HeuristicEnemySMAX, obs_with_agent_id=True):
        super().__init__(env)
        self.obs_with_agent_id = obs_with_agent_id
        if not self.obs_with_agent_id:
            self._world_state_size = self._env.state_size
            self.world_state_fn = self.ws_just_env_state
        else:
            self._world_state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self.ws_with_agent_id

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state_fn(obs, state)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def ws_just_env_state(self, obs, state):
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        return world_state

    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, state):
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)

    def world_state_size(self):
        return self._world_state_size 


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        if len(x) == 5:
            obs, dones, avail_actions, role_id, adapter_id = x
        elif len(x) == 4:
            obs, dones, avail_actions, role_id = x
            adapter_id = role_id
        else:
            obs, dones, avail_actions = x
            role_id = jnp.zeros(obs.shape[:-1], dtype=jnp.int32)
            adapter_id = role_id
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        base_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        lora_delta = jnp.zeros_like(base_logits)
        if self.config["USE_ROLE_LORA"]:
            rank = self.config["ROLE_LORA_RANK"]
            lora_a = self.param(
                "role_lora_A",
                nn.initializers.normal(self.config["ROLE_LORA_A_INIT_STD"]),
                (self.config["LORA_NUM_ADAPTERS"], rank, self.config["GRU_HIDDEN_DIM"]),
            )
            lora_b = self.param(
                "role_lora_B",
                nn.initializers.zeros,
                (self.config["LORA_NUM_ADAPTERS"], self.action_dim, rank),
            )
            safe_adapter_id = jnp.clip(
                adapter_id.astype(jnp.int32),
                0,
                self.config["LORA_NUM_ADAPTERS"] - 1,
            )
            role_a = lora_a[safe_adapter_id]
            role_b = lora_b[safe_adapter_id]
            lora_hidden = jnp.einsum("...rh,...h->...r", role_a, actor_mean)
            lora_delta = jnp.einsum("...ar,...r->...a", role_b, lora_hidden)
        actor_mean = base_logits + self.config["ROLE_LORA_SCALE"] * lora_delta
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        aux = {
            "base_logits": base_logits,
            "lora_delta": lora_delta,
            "action_logits": action_logits,
            "unmasked_logits": actor_mean,
        }
        return hidden, pi, aux


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)


class QCritic(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, q_input):
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(q_input)
        embedding = nn.relu(embedding)
        hidden = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        hidden = nn.relu(hidden)
        q_value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(hidden)
        return jnp.squeeze(q_value, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    role_id: jnp.ndarray
    adapter_id: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def extract_role_id(obs_batch: jnp.ndarray, env_state, env, config: Dict):
    if config["ROLE_ID_SOURCE"] == "own_obs_unit_type":
        own_type_bits = obs_batch[:, -config["NUM_UNIT_TYPES"]:]
        return jnp.argmax(own_type_bits, axis=-1).astype(jnp.int32)
    if config["ROLE_ID_SOURCE"] == "env_state_unit_type":
        unit_types = env_state.env_state.state.unit_types[:, : env.num_agents]
        return unit_types.T.reshape(-1).astype(jnp.int32)
    else:
        raise ValueError(
            f"Unsupported ROLE_ID_SOURCE={config['ROLE_ID_SOURCE']!r}; "
            "Stage 1 supports 'env_state_unit_type' and 'own_obs_unit_type'."
        )

def adapter_id_from_mode(role_id: jnp.ndarray, agent_id: jnp.ndarray, config: Dict):
    adapter_mode = config["ADAPTER_MODE"]
    if adapter_mode == "global_lora":
        return jnp.zeros_like(role_id, dtype=jnp.int32)
    if adapter_mode == "agent_lora":
        return agent_id.astype(jnp.int32)
    return role_id.astype(jnp.int32)

def role_mean_metric(values: jnp.ndarray, role_id: jnp.ndarray, num_roles: int):
    flat_values = values.reshape(-1)
    flat_roles = role_id.reshape(-1)
    counts = jnp.bincount(flat_roles, length=num_roles)
    sums = jnp.bincount(flat_roles, weights=flat_values, length=num_roles)
    return sums / jnp.maximum(counts, 1)

def role_std_metric(values: jnp.ndarray, role_id: jnp.ndarray, num_roles: int):
    mean = role_mean_metric(values, role_id, num_roles)
    mean_sq = role_mean_metric(jnp.square(values), role_id, num_roles)
    return jnp.sqrt(jnp.maximum(mean_sq - jnp.square(mean), 0.0))

def role_lora_param_norms(actor_params, config: Dict):
    if not config["USE_ROLE_LORA"]:
        return jnp.zeros((config["LORA_NUM_ADAPTERS"],))
    params = actor_params["params"]
    lora_a = params["role_lora_A"]
    lora_b = params["role_lora_B"]
    return jnp.sqrt(jnp.sum(jnp.square(lora_a), axis=(1, 2)) + jnp.sum(jnp.square(lora_b), axis=(1, 2)))

def joint_action_features(action_probs: jnp.ndarray, num_agents: int, num_envs: int):
    num_steps = action_probs.shape[0]
    action_dim = action_probs.shape[-1]
    by_agent_env = action_probs.reshape(num_steps, num_agents, num_envs, action_dim)
    by_env = by_agent_env.transpose(0, 2, 1, 3).reshape(
        num_steps,
        num_envs,
        num_agents * action_dim,
    )
    broadcast = jnp.broadcast_to(
        by_env[:, None, :, :],
        (num_steps, num_agents, num_envs, num_agents * action_dim),
    )
    return broadcast.reshape(num_steps, num_agents * num_envs, num_agents * action_dim)

def individual_counterfactual_features(
    taken_probs: jnp.ndarray,
    policy_probs: jnp.ndarray,
    num_agents: int,
    num_envs: int,
):
    taken_joint = joint_action_features(taken_probs, num_agents, num_envs)
    num_steps = taken_probs.shape[0]
    action_dim = taken_probs.shape[-1]
    agent_id = jnp.repeat(jnp.arange(num_agents, dtype=jnp.int32), num_envs)
    agent_mask = jax.nn.one_hot(agent_id, num_agents)
    segment_mask = jnp.repeat(agent_mask, action_dim, axis=-1)
    replacement = (agent_mask[None, :, :, None] * policy_probs[:, :, None, :]).reshape(
        num_steps,
        num_agents * num_envs,
        num_agents * action_dim,
    )
    return taken_joint * (1.0 - segment_mask[None, :, :]) + replacement

def same_role_counterfactual_features(
    taken_probs: jnp.ndarray,
    policy_probs: jnp.ndarray,
    role_id: jnp.ndarray,
    num_agents: int,
    num_envs: int,
):
    num_steps = taken_probs.shape[0]
    action_dim = taken_probs.shape[-1]
    taken_by_agent_env = taken_probs.reshape(num_steps, num_agents, num_envs, action_dim)
    policy_by_agent_env = policy_probs.reshape(num_steps, num_agents, num_envs, action_dim)
    role_by_agent_env = role_id.reshape(num_steps, num_agents, num_envs)

    query_role = role_by_agent_env[:, :, :, None]
    member_role = role_by_agent_env.transpose(0, 2, 1)[:, None, :, :]
    same_role = query_role == member_role

    taken_member = taken_by_agent_env.transpose(0, 2, 1, 3)[:, None, :, :, :]
    policy_member = policy_by_agent_env.transpose(0, 2, 1, 3)[:, None, :, :, :]
    counterfactual = jnp.where(same_role[..., None], policy_member, taken_member)
    counterfactual = counterfactual.reshape(
        num_steps,
        num_agents,
        num_envs,
        num_agents * action_dim,
    )
    return counterfactual.reshape(num_steps, num_agents * num_envs, num_agents * action_dim)

def build_q_inputs(
    world_state: jnp.ndarray,
    joint_features: jnp.ndarray,
    role_id: jnp.ndarray,
    num_roles: int,
):
    role_onehot = jax.nn.one_hot(role_id.astype(jnp.int32), num_roles)
    return jnp.concatenate((world_state, joint_features, role_onehot), axis=-1)

def make_train(config):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    if obs_dim < config["NUM_UNIT_TYPES"]:
        raise ValueError(
            f"Cannot extract role IDs: obs_dim={obs_dim}, "
            f"NUM_UNIT_TYPES={config['NUM_UNIT_TYPES']}. Expected own unit-type bits "
            "at the end of the SMAX local observation."
        )
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_AGENTS"] = env.num_agents
    config["ACTION_DIM"] = env.action_space(env.agents[0]).n
    if config["ADAPTER_MODE"] == "global_lora":
        config["LORA_NUM_ADAPTERS"] = 1
    elif config["ADAPTER_MODE"] == "agent_lora":
        config["LORA_NUM_ADAPTERS"] = env.num_agents
    else:
        config["LORA_NUM_ADAPTERS"] = config["NUM_UNIT_TYPES"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    if config["NUM_ACTORS"] * config["NUM_STEPS"] % config["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            "Invalid minibatch setup: NUM_ACTORS * NUM_STEPS must be divisible by "
            f"NUM_MINIBATCHES. Got NUM_ACTORS={config['NUM_ACTORS']}, "
            f"NUM_STEPS={config['NUM_STEPS']}, NUM_MINIBATCHES={config['NUM_MINIBATCHES']}."
        )
    if config["NUM_UPDATES"] <= 0:
        raise ValueError(
            "Invalid training horizon: TOTAL_TIMESTEPS must be at least NUM_ENVS * NUM_STEPS. "
            f"Got TOTAL_TIMESTEPS={config['TOTAL_TIMESTEPS']}, NUM_ENVS={config['NUM_ENVS']}, "
            f"NUM_STEPS={config['NUM_STEPS']}."
        )
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)
    config["Q_INPUT_DIM"] = (
        env.world_state_size()
        + config["NUM_AGENTS"] * config["ACTION_DIM"]
        + config["NUM_UNIT_TYPES"]
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        if config["USE_ROLE_MACA"]:
            q_network = QCritic(config=config)
            rng, _rng_actor, _rng_critic, _rng_q = jax.random.split(rng, 4)
        else:
            rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
            jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.int32),
            jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.int32),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        
        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.world_state_size(),)),  
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
            critic_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
        else:
            actor_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
            critic_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
            
        actor_train_state = TrainState.create(apply_fn=actor_network.apply, params=actor_network_params, tx=actor_tx)
        critic_train_state = TrainState.create(apply_fn=critic_network.apply, params=critic_network_params, tx=critic_tx)
        if config["USE_ROLE_MACA"]:
            q_network_params = q_network.init(
                _rng_q,
                jnp.zeros((1, config["NUM_ENVS"], config["Q_INPUT_DIM"])),
            )
            if config["ANNEAL_LR"]:
                q_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
            else:
                q_tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
            q_train_state = TrainState.create(apply_fn=q_network.apply, params=q_network_params, tx=q_tx)
            train_states = (actor_train_state, critic_train_state, q_train_state)
        else:
            train_states = (actor_train_state, critic_train_state)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state
            
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions, env.agents, config["NUM_ACTORS"]))
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                role_id = extract_role_id(obs_batch, env_state, env, config)
                agent_id = jnp.repeat(jnp.arange(env.num_agents, dtype=jnp.int32), config["NUM_ENVS"])
                adapter_id = adapter_id_from_mode(role_id, agent_id, config)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                    role_id[np.newaxis, :],
                    adapter_id[np.newaxis, :],
                )
                
                ac_hstate, pi, _ = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                world_state = last_obs["world_state"].swapaxes(0,1)  
                world_state = world_state.reshape((config["NUM_ACTORS"],-1))
                cr_in = (world_state[None, :], last_done[np.newaxis, :])
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                    avail_actions,
                    role_id,
                    adapter_id,
                )
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
            
            train_states, env_state, last_obs, last_done, hstates, rng = runner_state
            
            last_world_state = last_obs["world_state"].swapaxes(0,1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
            cr_in = (last_world_state[None, :], last_done[np.newaxis, :])
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.global_done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _empty_maca_info():
                zeros_by_role = jnp.zeros((config["NUM_UNIT_TYPES"],))
                return {
                    "advantages": advantages,
                    "q_taken_inputs": jnp.zeros(
                        (
                            config["NUM_STEPS"],
                            config["NUM_ACTORS"],
                            config["Q_INPUT_DIM"],
                        )
                    ),
                    "q_taken_mean_by_role": zeros_by_role,
                    "baseline_joint_by_role": zeros_by_role,
                    "baseline_individual_by_role": zeros_by_role,
                    "baseline_role_by_role": zeros_by_role,
                    "adv_maca_mean_by_role": zeros_by_role,
                    "adv_maca_std_by_role": zeros_by_role,
                    "component_gap_joint_by_role": zeros_by_role,
                    "component_gap_individual_by_role": zeros_by_role,
                    "component_gap_role_by_role": zeros_by_role,
                }

            def _calculate_role_maca_info():
                _, _, actor_aux = actor_network.apply(
                    train_states[0].params,
                    initial_hstates[0],
                    (
                        traj_batch.obs,
                        traj_batch.done,
                        traj_batch.avail_actions,
                        traj_batch.role_id,
                        traj_batch.adapter_id,
                    ),
                )
                policy_probs = jax.lax.stop_gradient(
                    jax.nn.softmax(actor_aux["action_logits"], axis=-1)
                )
                taken_probs = jax.nn.one_hot(traj_batch.action, config["ACTION_DIM"])

                taken_joint = joint_action_features(
                    taken_probs,
                    config["NUM_AGENTS"],
                    config["NUM_ENVS"],
                )
                joint_cf = joint_action_features(
                    policy_probs,
                    config["NUM_AGENTS"],
                    config["NUM_ENVS"],
                )
                individual_cf = individual_counterfactual_features(
                    taken_probs,
                    policy_probs,
                    config["NUM_AGENTS"],
                    config["NUM_ENVS"],
                )
                role_cf = same_role_counterfactual_features(
                    taken_probs,
                    policy_probs,
                    traj_batch.role_id,
                    config["NUM_AGENTS"],
                    config["NUM_ENVS"],
                )

                q_taken_inputs = build_q_inputs(
                    traj_batch.world_state,
                    taken_joint,
                    traj_batch.role_id,
                    config["NUM_UNIT_TYPES"],
                )
                q_joint_inputs = build_q_inputs(
                    traj_batch.world_state,
                    joint_cf,
                    traj_batch.role_id,
                    config["NUM_UNIT_TYPES"],
                )
                q_individual_inputs = build_q_inputs(
                    traj_batch.world_state,
                    individual_cf,
                    traj_batch.role_id,
                    config["NUM_UNIT_TYPES"],
                )
                q_role_inputs = build_q_inputs(
                    traj_batch.world_state,
                    role_cf,
                    traj_batch.role_id,
                    config["NUM_UNIT_TYPES"],
                )

                q_params = train_states[2].params
                q_taken = q_network.apply(q_params, q_taken_inputs)
                b_joint = q_network.apply(q_params, q_joint_inputs)
                b_individual = q_network.apply(q_params, q_individual_inputs)
                b_role = q_network.apply(q_params, q_role_inputs)
                baseline = (
                    config["ROLE_MACA_WEIGHT_JNT"] * b_joint
                    + config["ROLE_MACA_WEIGHT_IND"] * b_individual
                    + config["ROLE_MACA_WEIGHT_ROLE"] * b_role
                )
                maca_advantages = jax.lax.stop_gradient(q_taken - baseline)
                return {
                    "advantages": maca_advantages,
                    "q_taken_inputs": q_taken_inputs,
                    "q_taken_mean_by_role": role_mean_metric(
                        q_taken,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "baseline_joint_by_role": role_mean_metric(
                        b_joint,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "baseline_individual_by_role": role_mean_metric(
                        b_individual,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "baseline_role_by_role": role_mean_metric(
                        b_role,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "adv_maca_mean_by_role": role_mean_metric(
                        maca_advantages,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "adv_maca_std_by_role": role_std_metric(
                        maca_advantages,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "component_gap_joint_by_role": role_mean_metric(
                        q_taken - b_joint,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "component_gap_individual_by_role": role_mean_metric(
                        q_taken - b_individual,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                    "component_gap_role_by_role": role_mean_metric(
                        q_taken - b_role,
                        traj_batch.role_id,
                        config["NUM_UNIT_TYPES"],
                    ),
                }

            maca_info = _calculate_role_maca_info() if config["USE_ROLE_MACA"] else _empty_maca_info()
            actor_advantages = maca_info["advantages"] if config["USE_ROLE_MACA"] else advantages

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    if config["USE_ROLE_MACA"]:
                        actor_train_state, critic_train_state, q_train_state = train_states
                        (
                            ac_init_hstate,
                            cr_init_hstate,
                            traj_batch,
                            advantages,
                            targets,
                            q_taken_inputs,
                        ) = batch_info
                    else:
                        actor_train_state, critic_train_state = train_states
                        ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        _, pi, actor_aux = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (
                                traj_batch.obs,
                                traj_batch.done,
                                traj_batch.avail_actions,
                                traj_batch.role_id,
                                traj_batch.adapter_id,
                            ),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        role_adv_mean = role_mean_metric(gae, traj_batch.role_id, config["NUM_UNIT_TYPES"])
                        role_adv_std = role_std_metric(gae, traj_batch.role_id, config["NUM_UNIT_TYPES"])
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        ppo_loss_per_sample = -jnp.minimum(loss_actor1, loss_actor2)
                        entropy_per_sample = pi.entropy()
                        entropy = entropy_per_sample.mean()
                        
                        approx_kl_per_sample = (ratio - 1) - logratio
                        clip_frac_per_sample = (jnp.abs(ratio - 1) > config["CLIP_EPS"]).astype(jnp.float32)
                        approx_kl = approx_kl_per_sample.mean()
                        clip_frac = clip_frac_per_sample.mean()
                        role_entropy = role_mean_metric(
                            entropy_per_sample,
                            traj_batch.role_id,
                            config["NUM_UNIT_TYPES"],
                        )
                        role_approx_kl = role_mean_metric(
                            approx_kl_per_sample,
                            traj_batch.role_id,
                            config["NUM_UNIT_TYPES"],
                        )
                        role_clip_frac = role_mean_metric(
                            clip_frac_per_sample,
                            traj_batch.role_id,
                            config["NUM_UNIT_TYPES"],
                        )
                        role_ppo_loss = role_mean_metric(
                            ppo_loss_per_sample,
                            traj_batch.role_id,
                            config["NUM_UNIT_TYPES"],
                        )
                        lora_delta_norm_by_role = role_mean_metric(
                            jnp.linalg.norm(actor_aux["lora_delta"], axis=-1),
                            traj_batch.role_id,
                            config["NUM_UNIT_TYPES"],
                        )
                        policy_loss = ppo_loss_per_sample.mean()
                        loss_actor = policy_loss - config["ENT_COEF"] * entropy
                        return loss_actor, {
                            "policy_loss": policy_loss,
                            "entropy": entropy,
                            "approx_kl": approx_kl,
                            "clip_frac": clip_frac,
                            "lora_delta_norm_by_role": lora_delta_norm_by_role,
                            "role_adv_mean": role_adv_mean,
                            "role_adv_std": role_adv_std,
                            "role_entropy": role_entropy,
                            "role_approx_kl": role_approx_kl,
                            "role_clip_frac": role_clip_frac,
                            "role_ppo_loss": role_ppo_loss,
                        }
                    
                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        _, value = critic_network.apply(critic_params, init_hstate.squeeze(), (traj_batch.world_state,  traj_batch.done)) 
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    def _q_loss_fn(q_params, q_taken_inputs, targets):
                        q_taken = q_network.apply(q_params, q_taken_inputs)
                        q_loss = 0.5 * jnp.square(q_taken - targets).mean()
                        return q_loss, {
                            "q_loss": q_loss,
                        }

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(actor_train_state.params, ac_init_hstate, traj_batch, advantages)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(critic_train_state.params, cr_init_hstate, traj_batch, targets)
                    if config["USE_ROLE_MACA"]:
                        q_grad_fn = jax.value_and_grad(_q_loss_fn, has_aux=True)
                        q_loss, q_grads = q_grad_fn(q_train_state.params, q_taken_inputs, targets)
                    else:
                        q_loss = (jnp.array(0.0), {"q_loss": jnp.array(0.0)})
                    
                    actor_grad_norm = optax.global_norm(actor_grads)
                    critic_grad_norm = optax.global_norm(critic_grads)
                    q_grad_norm = optax.global_norm(q_grads) if config["USE_ROLE_MACA"] else jnp.array(0.0)
                    lora_grad_norm_by_role = role_lora_param_norms(actor_grads, config)
                    lora_param_norm_by_role = role_lora_param_norms(actor_train_state.params, config)

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                    if config["USE_ROLE_MACA"]:
                        q_train_state = q_train_state.apply_gradients(grads=q_grads)
                        next_train_states = (actor_train_state, critic_train_state, q_train_state)
                    else:
                        next_train_states = (actor_train_state, critic_train_state)

                    total_loss = actor_loss[0] + critic_loss[0] + q_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "q_loss": q_loss[1]["q_loss"],
                        "entropy": actor_loss[1]["entropy"],
                        "approx_kl": actor_loss[1]["approx_kl"],
                        "clip_frac": actor_loss[1]["clip_frac"],
                        "actor_grad_norm": actor_grad_norm,
                        "critic_grad_norm": critic_grad_norm,
                        "q_grad_norm": q_grad_norm,
                        "lora_delta_norm_by_role": actor_loss[1]["lora_delta_norm_by_role"],
                        "role_adv_mean": actor_loss[1]["role_adv_mean"],
                        "role_adv_std": actor_loss[1]["role_adv_std"],
                        "role_entropy": actor_loss[1]["role_entropy"],
                        "role_approx_kl": actor_loss[1]["role_approx_kl"],
                        "role_clip_frac": actor_loss[1]["role_clip_frac"],
                        "role_ppo_loss": actor_loss[1]["role_ppo_loss"],
                        "lora_grad_norm_by_role": lora_grad_norm_by_role,
                        "lora_param_norm_by_role": lora_param_norm_by_role,
                    }
                    return next_train_states, loss_info

                if config["USE_ROLE_MACA"]:
                    train_states, init_hstates, traj_batch, advantages, targets, rng, q_taken_inputs = update_state
                else:
                    train_states, init_hstates, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                init_hstates = jax.tree.map(lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstates)
                
                if config["USE_ROLE_MACA"]:
                    batch = (
                        init_hstates[0],
                        init_hstates[1],
                        traj_batch,
                        advantages.squeeze(),
                        targets.squeeze(),
                        q_taken_inputs,
                    )
                else:
                    batch = (init_hstates[0], init_hstates[1], traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])), 1, 0
                    ), shuffled_batch
                )

                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                if config["USE_ROLE_MACA"]:
                    update_state = (
                        train_states,
                        jax.tree.map(lambda x: x.squeeze(), init_hstates),
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                        q_taken_inputs,
                    )
                else:
                    update_state = (
                        train_states,
                        jax.tree.map(lambda x: x.squeeze(), init_hstates),
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    )
                return update_state, loss_info

            if config["USE_ROLE_MACA"]:
                update_state = (
                    train_states,
                    initial_hstates,
                    traj_batch,
                    actor_advantages,
                    targets,
                    rng,
                    maca_info["q_taken_inputs"],
                )
            else:
                update_state = (train_states, initial_hstates, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            def reduce_loss_metric(x):
                if x.ndim <= 2:
                    return x.mean()
                return x.mean(axis=tuple(range(x.ndim - 1)))
            loss_info = jax.tree.map(reduce_loss_metric, loss_info)
            train_states = update_state[0]
            rng_after_update = update_state[5] if config["USE_ROLE_MACA"] else update_state[-1]

            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)), traj_batch.info
            )
            metric["loss"] = loss_info
            
            # JAX 0.7.x compatible logging: boolean masked indexing not allowed
            # inside jit/scan. Use weighted sum instead.
            mask = metric["returned_episode"][:, :, 0]  # (steps, envs) bool
            ep_count = jnp.sum(mask) + 1e-8
            returns = jnp.sum(metric["returned_episode_returns"][:, :, 0] * mask) / ep_count
            win_rate = jnp.sum(metric["returned_won_episode"][:, :, 0] * mask) / ep_count

            total_loss = loss_info["total_loss"]
            entropy = loss_info["entropy"]
            actor_grad_norm = loss_info["actor_grad_norm"]
            critic_grad_norm = loss_info["critic_grad_norm"]
            q_loss_value = loss_info["q_loss"]
            q_grad_norm = loss_info["q_grad_norm"]
            lora_delta_norm_by_role = loss_info["lora_delta_norm_by_role"]
            lora_grad_norm_by_role = loss_info["lora_grad_norm_by_role"]
            lora_param_norm_by_role = loss_info["lora_param_norm_by_role"]
            role_adv_mean = loss_info["role_adv_mean"]
            role_adv_std = loss_info["role_adv_std"]
            role_entropy = loss_info["role_entropy"]
            role_approx_kl = loss_info["role_approx_kl"]
            role_clip_frac = loss_info["role_clip_frac"]
            role_ppo_loss = loss_info["role_ppo_loss"]
            role_counts = jnp.bincount(
                traj_batch.role_id.reshape(-1),
                length=config["NUM_UNIT_TYPES"],
            )
            role_present_mask = role_counts > 0
            role_bits = traj_batch.obs[:, :, -config["NUM_UNIT_TYPES"]:]
            role_id_min = traj_batch.role_id.min()
            role_id_max = traj_batch.role_id.max()
            role_bit_sums = role_bits.sum(axis=-1)
            role_bit_sum_min = role_bit_sums.min()
            role_bit_sum_max = role_bit_sums.max()
            nonzero_role_bits = role_bit_sums > 0
            obs_role_id = jnp.argmax(role_bits, axis=-1).astype(jnp.int32)
            obs_role_mismatch_count = jnp.sum(
                nonzero_role_bits & (obs_role_id != traj_batch.role_id)
            )
            zero_obs_role_bits = jnp.sum(role_bit_sums == 0)

            def log_callback(
                r, w, s, tl, ent, agn, cgn, ql, qgn, rc, rpm, rbmin, rbmax, mismatch,
                zero_bits, ldn, lgn, lpn, ram, ras, rent, rkl, rcf, rpl,
                qtm, bjm, bim, brm, mam, mas, cgj, cgi, cgr, role_min, role_max,
            ):
                if role_min < 0 or role_max >= config["NUM_UNIT_TYPES"]:
                    raise ValueError(
                        "Role extraction failed: role ids are outside the configured range. "
                        f"role_id_min={int(role_min)}, role_id_max={int(role_max)}, "
                        f"NUM_UNIT_TYPES={config['NUM_UNIT_TYPES']}, "
                        f"ROLE_ID_SOURCE={config['ROLE_ID_SOURCE']!r}"
                    )
                if rbmax > 1.0 or mismatch > 0:
                    raise ValueError(
                        "Role extraction failed: obs unit-type bits disagree with env-state roles. "
                        f"role_bit_sum_min={rbmin}, role_bit_sum_max={rbmax}, "
                        f"obs_role_mismatch_count={int(mismatch)}, "
                        f"zero_obs_role_bits={int(zero_bits)}, "
                        f"NUM_UNIT_TYPES={config['NUM_UNIT_TYPES']}, "
                        f"ROLE_ID_SOURCE={config['ROLE_ID_SOURCE']!r}"
                    )
                role_counts_str = " ".join(
                    f"role_count_{idx}: {int(count)}" for idx, count in enumerate(np.asarray(rc))
                )
                role_mask_str = "[" + " ".join(str(int(x)) for x in np.asarray(rpm)) + "]"
                msg = (
                    f"Step {s:8d} | Return: {r:10.2f} | Win Rate: {w:5.2f} "
                    f"| Loss: {tl:10.4f} | Ent: {ent:8.4f} "
                    f"| GradN(actor/critic): {agn:8.4f}/{cgn:8.4f} "
                    f"| {role_counts_str} | role_present_mask: {role_mask_str} "
                    f"| zero_obs_role_bits: {int(zero_bits)}"
                )
                def fmt(xs):
                    return "[" + " ".join(
                        f"{float(x):.4e}" if present else "NA"
                        for x, present in zip(np.asarray(xs), np.asarray(rpm))
                    ) + "]"
                def fmt_all(xs):
                    return "[" + " ".join(f"{float(x):.4e}" for x in np.asarray(xs)) + "]"

                if config["USE_ROLE_LORA"]:
                    msg += (
                        f" | lora_delta_norm_by_role: {fmt(ldn)}"
                        f" | lora_grad_norm_by_adapter: {fmt_all(lgn)}"
                        f" | lora_param_norm_by_adapter: {fmt_all(lpn)}"
                    )
                if config["LOG_ROLE_DIAGNOSTICS"]:
                    msg += (
                        f" | role_adv_mean: {fmt(ram)}"
                        f" | role_adv_std: {fmt(ras)}"
                        f" | role_entropy: {fmt(rent)}"
                        f" | role_approx_kl: {fmt(rkl)}"
                        f" | role_clip_frac: {fmt(rcf)}"
                        f" | role_ppo_loss: {fmt(rpl)}"
                    )
                if config["USE_ROLE_MACA"]:
                    msg += (
                        f" | q_loss: {float(ql):.4e}"
                        f" | q_grad_norm: {float(qgn):.4e}"
                        f" | q_taken_mean_by_role: {fmt(qtm)}"
                        f" | baseline_joint_by_role: {fmt(bjm)}"
                        f" | baseline_individual_by_role: {fmt(bim)}"
                        f" | baseline_role_by_role: {fmt(brm)}"
                        f" | adv_maca_mean_by_role: {fmt(mam)}"
                        f" | adv_maca_std_by_role: {fmt(mas)}"
                        f" | component_gap_joint_by_role: {fmt(cgj)}"
                        f" | component_gap_individual_by_role: {fmt(cgi)}"
                        f" | component_gap_role_by_role: {fmt(cgr)}"
                    )
                print(msg)
                if config["LOG_ROLE_DIAGNOSTIC_TABLE"]:
                    print("role | count | adv_mean | adv_std | entropy | approx_kl | clip_frac | ppo_loss | lora_grad_norm")
                    lgn_arr = np.asarray(lgn)
                    for idx in range(len(np.asarray(rc))):
                        if not bool(np.asarray(rpm)[idx]):
                            print(f"{idx:4d} | {0:5d} | NA | NA | NA | NA | NA | NA | NA")
                            continue
                        adapter_grad_norm = float(lgn_arr[idx]) if idx < len(lgn_arr) else float("nan")
                        print(
                            f"{idx:4d} | {int(np.asarray(rc)[idx]):5d} "
                            f"| {float(np.asarray(ram)[idx]): .4e} "
                            f"| {float(np.asarray(ras)[idx]): .4e} "
                            f"| {float(np.asarray(rent)[idx]): .4e} "
                            f"| {float(np.asarray(rkl)[idx]): .4e} "
                            f"| {float(np.asarray(rcf)[idx]): .4e} "
                            f"| {float(np.asarray(rpl)[idx]): .4e} "
                            f"| {adapter_grad_norm: .4e}"
                        )

            step_count = update_steps * config["NUM_ENVS"] * config["NUM_STEPS"]
            jax.experimental.io_callback(
                log_callback, None,
                returns, win_rate, step_count,
                total_loss, entropy, actor_grad_norm, critic_grad_norm, q_loss_value, q_grad_norm,
                role_counts, role_present_mask, role_bit_sum_min, role_bit_sum_max,
                obs_role_mismatch_count, zero_obs_role_bits,
                lora_delta_norm_by_role, lora_grad_norm_by_role, lora_param_norm_by_role,
                role_adv_mean, role_adv_std, role_entropy, role_approx_kl, role_clip_frac,
                role_ppo_loss,
                maca_info["q_taken_mean_by_role"],
                maca_info["baseline_joint_by_role"],
                maca_info["baseline_individual_by_role"],
                maca_info["baseline_role_by_role"],
                maca_info["adv_maca_mean_by_role"],
                maca_info["adv_maca_std_by_role"],
                maca_info["component_gap_joint_by_role"],
                maca_info["component_gap_individual_by_role"],
                maca_info["component_gap_role_by_role"],
                role_id_min, role_id_max,
            )
            
            update_steps = update_steps + 1
            runner_state = (
                train_states,
                env_state,
                last_obs,
                last_done,
                hstates,
                rng_after_update,
            )
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_states,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(_update_step, (runner_state, 0), None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metric": metric}

    return train

if __name__ == "__main__":
    config = {
        "LR": 0.002,
        "NUM_ENVS": 128,
        "NUM_STEPS": 128, 
        "TOTAL_TIMESTEPS": int(3e6),  # Train for 3M steps to see convergence
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
        "MAP_NAME": "3m",    # We start with 3m
        "SEED": 42,
        "RUN_NAME": "rosa_mappo",
        "ADAPTER_MODE": "none",
        "NUM_UNIT_TYPES": 6,
        "ROLE_ID_SOURCE": "env_state_unit_type",
        "USE_ROLE_LORA": False,
        "ROLE_LORA_RANK": 4,
        "ROLE_LORA_SCALE": 1.0,
        "ROLE_LORA_A_INIT_STD": 0.01,
        "LOG_ROLE_DIAGNOSTICS": True,
        "LOG_ROLE_DIAGNOSTIC_TABLE": False,
        "ENV_KWARGS": {
            "see_enemy_actions": True,
            "walls_cause_death": True,
            "attack_mode": "closest"
        },
        "ANNEAL_LR": True
    }

    args = parse_args()
    config = apply_cli_overrides(config, args)
    print_resolved_config(config)
    print(f"Starting {config['MAP_NAME']} ROSA-MAPPO experiment ({config['RUN_NAME']})...")
    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train(config))
    
    start_time = time.time()
    out = train_jit(rng)
    end_time = time.time()
    
    print(f"Training completed in {(end_time - start_time) / 60:.1f} minutes.")

    model_dir = os.path.join(_REPO_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    safe_run_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in config["RUN_NAME"])
    model_path = os.path.join(model_dir, f"{safe_run_name}_actor.pkl")

    final_runner_state = out["runner_state"][0]
    final_train_states = final_runner_state[0]
    final_actor_state = final_train_states[0]
    actor_params = jax.device_get(final_actor_state.params)
    checkpoint = {
        "model_type": "gru",
        "config": config,
        "actor_params": actor_params,
    }
    if config["USE_ROLE_MACA"]:
        final_q_state = final_train_states[2]
        checkpoint["q_critic_params"] = jax.device_get(final_q_state.params)
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved ROSA-MAPPO actor checkpoint to {model_path}")
    
    # Can optionally save metrics
    # jnp.save("gru_baseline_metrics.npy", out["metric"])
