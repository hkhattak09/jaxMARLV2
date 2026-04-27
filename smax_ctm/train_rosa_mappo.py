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
from flax import struct
from flax.core import unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
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
    "sequential_polish",
    "role_balanced",
    "role_residual",
    "role_context_residual",
    "global_residual",
    "agent_residual",
    "shuffled_context_residual",
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
    parser.add_argument("--residual_loss_coef", type=float, default=None)
    parser.add_argument("--residual_kl_coef", type=float, default=None)
    parser.add_argument("--context_gate_hidden_dim", type=int, default=None)
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
    if args.residual_loss_coef is not None:
        config["RESIDUAL_LOSS_COEF"] = args.residual_loss_coef
    if args.residual_kl_coef is not None:
        config["RESIDUAL_KL_COEF"] = args.residual_kl_coef
    if args.context_gate_hidden_dim is not None:
        config["CONTEXT_GATE_HIDDEN_DIM"] = args.context_gate_hidden_dim
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
        "sequential_polish",
        "role_balanced",
        "role_residual",
        "role_context_residual",
        "global_residual",
        "agent_residual",
        "shuffled_context_residual",
    }
    config["USE_SEQUENTIAL_ROLE_UPDATES"] = adapter_mode == "sequential_polish"
    config["ROLE_BALANCED_PPO"] = adapter_mode == "role_balanced"
    config["USE_ROLE_RESIDUAL_LOSS"] = adapter_mode in {
        "role_residual",
        "role_context_residual",
        "global_residual",
        "agent_residual",
        "shuffled_context_residual",
    }
    config["USE_CONTEXT_GATE"] = adapter_mode in {
        "role_context_residual",
        "shuffled_context_residual",
    }
    config["CONTEXT_SHUFFLE"] = adapter_mode == "shuffled_context_residual"
    if config["USE_ROLE_RESIDUAL_LOSS"] and not config["RESIDUAL_STOP_BACKBONE"]:
        raise ValueError(
            "Stage 4 residual modes require RESIDUAL_STOP_BACKBONE=True so the auxiliary "
            "loss cannot push role-specific gradients into the shared backbone."
        )
    if adapter_mode == "role_lora":
        config["FREEZE_LORA_IN_SHARED_UPDATE"] = False
    if adapter_mode == "sequential_polish":
        config["FREEZE_LORA_IN_SHARED_UPDATE"] = False
    return config


def print_resolved_config(config: Dict):
    residual_context = {
        key: value
        for key, value in sorted(config.items())
        if "RESIDUAL" in key or "CONTEXT" in key
    }
    print("Resolved config:")
    print(json.dumps(config, indent=2, sort_keys=True))
    print(f"adapter_mode: {config['ADAPTER_MODE']}")
    print(f"map_name: {config['MAP_NAME']}")
    print(f"seed: {config['SEED']}")
    print(f"run_name: {config['RUN_NAME']}")
    print("residual/context config values:")
    print(json.dumps(residual_context, indent=2, sort_keys=True))

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
            role_context = None
        elif len(x) == 6:
            obs, dones, avail_actions, role_id, adapter_id, role_context = x
        elif len(x) == 4:
            obs, dones, avail_actions, role_id = x
            adapter_id = role_id
            role_context = None
        else:
            obs, dones, avail_actions = x
            role_id = jnp.zeros(obs.shape[:-1], dtype=jnp.int32)
            adapter_id = role_id
            role_context = None
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
        context_gate = jnp.ones(base_logits.shape[:-1], dtype=base_logits.dtype)
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
            if self.config["USE_CONTEXT_GATE"]:
                if role_context is None:
                    raise ValueError("USE_CONTEXT_GATE=True requires role_context in ActorRNN input.")
                safe_role_id = jnp.clip(role_id.astype(jnp.int32), 0, self.config["NUM_UNIT_TYPES"] - 1)
                role_onehot = jax.nn.one_hot(safe_role_id, self.config["NUM_UNIT_TYPES"])
                gate_input = jnp.concatenate(
                    (
                        jax.lax.stop_gradient(actor_mean),
                        role_onehot,
                        role_context,
                    ),
                    axis=-1,
                )
                gate_hidden = nn.Dense(
                    self.config["CONTEXT_GATE_HIDDEN_DIM"],
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                    name="context_gate_hidden",
                )(gate_input)
                gate_hidden = nn.relu(gate_hidden)
                gate_logit = nn.Dense(
                    1,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(self.config["RESIDUAL_GATE_INIT_BIAS"]),
                    name="context_gate_out",
                )(gate_hidden)
                context_gate = jnp.squeeze(nn.sigmoid(gate_logit), axis=-1)
        actor_mean = base_logits + self.config["ROLE_LORA_SCALE"] * context_gate[..., None] * lora_delta
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        aux = {
            "base_logits": base_logits,
            "lora_delta": lora_delta,
            "context_gate": context_gate,
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
    role_context: jnp.ndarray


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
    if adapter_mode in ("global_lora", "global_residual"):
        return jnp.zeros_like(role_id, dtype=jnp.int32)
    if adapter_mode in ("agent_lora", "agent_residual"):
        return agent_id.astype(jnp.int32)
    return role_id.astype(jnp.int32)

def build_role_context(role_id: jnp.ndarray, env_num_agents: int, config: Dict):
    if config["CONTEXT_SOURCE"] != "team_role_histogram":
        raise ValueError(
            f"Unsupported CONTEXT_SOURCE={config['CONTEXT_SOURCE']!r}; "
            "Stage 4 supports 'team_role_histogram'."
        )
    role_by_agent_env = role_id.reshape(env_num_agents, config["NUM_ENVS"])
    one_hot = jax.nn.one_hot(role_by_agent_env, config["NUM_UNIT_TYPES"])
    team_counts_env_role = one_hot.sum(axis=0)
    own_role_count = jnp.take_along_axis(
        team_counts_env_role.T,
        role_by_agent_env,
        axis=0,
    )
    team_hist_by_agent_env = jnp.broadcast_to(
        team_counts_env_role[None, :, :],
        (env_num_agents, config["NUM_ENVS"], config["NUM_UNIT_TYPES"]),
    )
    context = jnp.concatenate(
        (
            team_hist_by_agent_env / jnp.maximum(float(env_num_agents), 1.0),
            (own_role_count[..., None] / jnp.maximum(float(env_num_agents), 1.0)),
        ),
        axis=-1,
    ).reshape(env_num_agents * config["NUM_ENVS"], config["NUM_UNIT_TYPES"] + 1)
    if config["CONTEXT_SHUFFLE"]:
        context = jnp.roll(context, shift=1, axis=0)
    return context

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

def role_max_metric(values: jnp.ndarray, role_id: jnp.ndarray, num_roles: int):
    flat_values = values.reshape(-1)
    flat_roles = role_id.reshape(-1)

    def max_for_role(role):
        role_values = jnp.where(flat_roles == role, flat_values, -jnp.inf)
        max_value = jnp.max(role_values)
        return jnp.where(jnp.isfinite(max_value), max_value, 0.0)

    return jax.vmap(max_for_role)(jnp.arange(num_roles))

def role_normalize_advantages(advantages: jnp.ndarray, role_id: jnp.ndarray, config: Dict, norm_mode: str):
    if norm_mode == "global":
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    if norm_mode != "role":
        raise ValueError(f"Unsupported residual advantage norm={norm_mode!r}; use 'role' or 'global'.")
    role_mean = role_mean_metric(advantages, role_id, config["NUM_UNIT_TYPES"])
    role_std = role_std_metric(advantages, role_id, config["NUM_UNIT_TYPES"])
    gathered_mean = role_mean[role_id]
    gathered_std = role_std[role_id]
    return (advantages - gathered_mean) / (gathered_std + 1e-8)

def role_lora_param_norms(actor_params, config: Dict):
    if not config["USE_ROLE_LORA"]:
        return jnp.zeros((config["LORA_NUM_ADAPTERS"],))
    params = actor_params["params"]
    lora_a = params["role_lora_A"]
    lora_b = params["role_lora_B"]
    return jnp.sqrt(jnp.sum(jnp.square(lora_a), axis=(1, 2)) + jnp.sum(jnp.square(lora_b), axis=(1, 2)))

def init_role_lora_adam_state(actor_params, config: Dict, action_dim: int):
    rank = config["ROLE_LORA_RANK"]
    if config["USE_ROLE_LORA"]:
        lora_a = actor_params["params"]["role_lora_A"]
        lora_b = actor_params["params"]["role_lora_B"]
        a_shape = lora_a.shape
        b_shape = lora_b.shape
    else:
        a_shape = (config["LORA_NUM_ADAPTERS"], rank, config["GRU_HIDDEN_DIM"])
        b_shape = (config["LORA_NUM_ADAPTERS"], action_dim, rank)
    return {
        "m_A": jnp.zeros(a_shape),
        "v_A": jnp.zeros(a_shape),
        "m_B": jnp.zeros(b_shape),
        "v_B": jnp.zeros(b_shape),
        "count": jnp.zeros((config["LORA_NUM_ADAPTERS"],), dtype=jnp.float32),
    }

def zero_role_lora_grads(actor_grads, config: Dict):
    if not config["USE_ROLE_LORA"]:
        return actor_grads
    grads = unfreeze(actor_grads)
    grads["params"]["role_lora_A"] = jnp.zeros_like(grads["params"]["role_lora_A"])
    grads["params"]["role_lora_B"] = jnp.zeros_like(grads["params"]["role_lora_B"])
    return grads

def filter_residual_grads(actor_grads, config: Dict):
    flat = flatten_dict(unfreeze(actor_grads), sep="/")
    kept = {}
    rejected = {}
    for path, value in flat.items():
        is_lora = path.endswith("role_lora_A") or path.endswith("role_lora_B")
        is_gate = config["USE_CONTEXT_GATE"] and "context_gate" in path
        keep = is_lora or is_gate
        kept[path] = value if keep else jnp.zeros_like(value)
        rejected[path] = jnp.zeros_like(value) if keep else value
    return unflatten_dict(kept, sep="/"), unflatten_dict(rejected, sep="/")

def stop_non_residual_params(actor_params, config: Dict):
    flat = flatten_dict(unfreeze(actor_params), sep="/")
    stopped = {}
    for path, value in flat.items():
        is_lora = path.endswith("role_lora_A") or path.endswith("role_lora_B")
        is_gate = config["USE_CONTEXT_GATE"] and "context_gate" in path
        stopped[path] = value if (is_lora or is_gate) else jax.lax.stop_gradient(value)
    return unflatten_dict(stopped, sep="/")

def residual_gate_grad_norm(actor_grads, config: Dict):
    if not config["USE_CONTEXT_GATE"]:
        return jnp.array(0.0)
    flat = flatten_dict(unfreeze(actor_grads), sep="/")
    leaves = [value for path, value in flat.items() if "context_gate" in path]
    if not leaves:
        raise ValueError("USE_CONTEXT_GATE=True but no context_gate gradients were found.")
    return optax.global_norm(leaves)

def update_single_role_lora_params_adam(
    actor_params,
    actor_grads,
    adam_state,
    role_id: jnp.ndarray,
    lr: jnp.ndarray,
    config: Dict,
):
    params = unfreeze(actor_params)
    grads = actor_grads["params"]
    old_lora_a = params["params"]["role_lora_A"]
    old_lora_b = params["params"]["role_lora_B"]

    beta1 = config["ROLE_SEQ_ADAM_BETA1"]
    beta2 = config["ROLE_SEQ_ADAM_BETA2"]
    eps = config["ROLE_SEQ_ADAM_EPS"]
    grad_a = grads["role_lora_A"][role_id]
    grad_b = grads["role_lora_B"][role_id]
    step = adam_state["count"][role_id] + 1.0

    m_a = beta1 * adam_state["m_A"][role_id] + (1.0 - beta1) * grad_a
    v_a = beta2 * adam_state["v_A"][role_id] + (1.0 - beta2) * jnp.square(grad_a)
    m_b = beta1 * adam_state["m_B"][role_id] + (1.0 - beta1) * grad_b
    v_b = beta2 * adam_state["v_B"][role_id] + (1.0 - beta2) * jnp.square(grad_b)

    m_a_hat = m_a / (1.0 - jnp.power(beta1, step))
    v_a_hat = v_a / (1.0 - jnp.power(beta2, step))
    m_b_hat = m_b / (1.0 - jnp.power(beta1, step))
    v_b_hat = v_b / (1.0 - jnp.power(beta2, step))

    new_lora_a = old_lora_a.at[role_id].add(-lr * m_a_hat / (jnp.sqrt(v_a_hat) + eps))
    new_lora_b = old_lora_b.at[role_id].add(-lr * m_b_hat / (jnp.sqrt(v_b_hat) + eps))
    params["params"]["role_lora_A"] = new_lora_a
    params["params"]["role_lora_B"] = new_lora_b
    new_adam_state = {
        "m_A": adam_state["m_A"].at[role_id].set(m_a),
        "v_A": adam_state["v_A"].at[role_id].set(v_a),
        "m_B": adam_state["m_B"].at[role_id].set(m_b),
        "v_B": adam_state["v_B"].at[role_id].set(v_b),
        "count": adam_state["count"].at[role_id].set(step),
    }
    return params, new_adam_state

def role_lora_update_norm(old_params, new_params, config: Dict):
    if not config["USE_ROLE_LORA"]:
        return jnp.zeros((config["LORA_NUM_ADAPTERS"],))
    old_lora_a = old_params["params"]["role_lora_A"]
    old_lora_b = old_params["params"]["role_lora_B"]
    new_lora_a = new_params["params"]["role_lora_A"]
    new_lora_b = new_params["params"]["role_lora_B"]
    return jnp.sqrt(
        jnp.sum(jnp.square(new_lora_a - old_lora_a), axis=(1, 2))
        + jnp.sum(jnp.square(new_lora_b - old_lora_b), axis=(1, 2))
    )

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
    if config["ADAPTER_MODE"] in ("global_lora", "global_residual"):
        config["LORA_NUM_ADAPTERS"] = 1
    elif config["ADAPTER_MODE"] in ("agent_lora", "agent_residual"):
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

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
            jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.int32),
            jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.int32),
            jnp.zeros((1, config["NUM_ENVS"], config["NUM_UNIT_TYPES"] + 1)),
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
        role_lora_adam_state = init_role_lora_adam_state(
            actor_network_params,
            config,
            env.action_space(env.agents[0]).n,
        )

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
                train_states, role_lora_adam_state, env_state, last_obs, last_done, hstates, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions, env.agents, config["NUM_ACTORS"]))
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                role_id = extract_role_id(obs_batch, env_state, env, config)
                agent_id = jnp.repeat(jnp.arange(env.num_agents, dtype=jnp.int32), config["NUM_ENVS"])
                adapter_id = adapter_id_from_mode(role_id, agent_id, config)
                role_context = build_role_context(role_id, env.num_agents, config)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                    role_id[np.newaxis, :],
                    adapter_id[np.newaxis, :],
                    role_context[np.newaxis, :],
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
                    role_context,
                )
                runner_state = (train_states, role_lora_adam_state, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
            
            train_states, role_lora_adam_state, env_state, last_obs, last_done, hstates, rng = runner_state
            
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
            _, _, pre_update_actor_aux = actor_network.apply(
                train_states[0].params,
                initial_hstates[0],
                (
                    traj_batch.obs,
                    traj_batch.done,
                    traj_batch.avail_actions,
                    traj_batch.role_id,
                    traj_batch.adapter_id,
                    traj_batch.role_context,
                ),
            )
            pre_update_logit_delta = config["ROLE_LORA_SCALE"] * pre_update_actor_aux["lora_delta"]
            max_abs_logit_diff = jnp.max(jnp.abs(pre_update_logit_delta))
            mean_abs_logit_diff = jnp.mean(jnp.abs(pre_update_logit_delta))

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
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
                                traj_batch.role_context,
                            ),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        role_adv_mean = role_mean_metric(gae, traj_batch.role_id, config["NUM_UNIT_TYPES"])
                        role_adv_std = role_std_metric(gae, traj_batch.role_id, config["NUM_UNIT_TYPES"])
                        if config["ROLE_BALANCED_PPO"]:
                            gae = role_normalize_advantages(
                                gae,
                                traj_batch.role_id,
                                config,
                                config["RESIDUAL_ADV_NORM"],
                            )
                        else:
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        ppo_loss_per_sample = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = ppo_loss_per_sample.mean()
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
                        role_mean_ratio = role_mean_metric(
                            ratio,
                            traj_batch.role_id,
                            config["NUM_UNIT_TYPES"],
                        )
                        role_max_ratio = role_max_metric(
                            ratio,
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
                        actor_loss = loss_actor - config["ENT_COEF"] * entropy
                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                            lora_delta_norm_by_role,
                            role_adv_mean,
                            role_adv_std,
                            role_entropy,
                            role_approx_kl,
                            role_clip_frac,
                            role_mean_ratio,
                            role_max_ratio,
                            role_ppo_loss,
                        )
                    
                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        _, value = critic_network.apply(critic_params, init_hstate.squeeze(), (traj_batch.world_state,  traj_batch.done)) 
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    def _residual_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        residual_params = stop_non_residual_params(actor_params, config)
                        _, pi, actor_aux = actor_network.apply(
                            residual_params,
                            init_hstate.squeeze(),
                            (
                                traj_batch.obs,
                                traj_batch.done,
                                traj_batch.avail_actions,
                                traj_batch.role_id,
                                traj_batch.adapter_id,
                                traj_batch.role_context,
                            ),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        residual_gae = role_normalize_advantages(
                            gae,
                            traj_batch.role_id,
                            config,
                            config["RESIDUAL_ADV_NORM"],
                        )
                        loss_actor1 = ratio * residual_gae
                        loss_actor2 = (
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"])
                            * residual_gae
                        )
                        residual_loss_per_sample = -jnp.minimum(loss_actor1, loss_actor2)
                        residual_loss = residual_loss_per_sample.mean()

                        unavail_actions = 1 - traj_batch.avail_actions
                        full_logits = actor_aux["unmasked_logits"] - (unavail_actions * 1e10)
                        base_logits = jax.lax.stop_gradient(actor_aux["base_logits"]) - (unavail_actions * 1e10)
                        full_log_probs = jax.nn.log_softmax(full_logits, axis=-1)
                        base_log_probs = jax.nn.log_softmax(base_logits, axis=-1)
                        base_probs = jax.lax.stop_gradient(jnp.exp(base_log_probs))
                        kl_to_base_per_sample = jnp.sum(
                            base_probs * (base_log_probs - full_log_probs),
                            axis=-1,
                        )
                        kl_to_base = kl_to_base_per_sample.mean()

                        action_onehot = jax.nn.one_hot(traj_batch.action, full_logits.shape[-1])
                        action_base_log_prob = jnp.sum(base_log_probs * action_onehot, axis=-1)
                        logprob_margin = log_prob - action_base_log_prob
                        adv_alignment_per_sample = residual_gae * logprob_margin
                        clip_frac_per_sample = (jnp.abs(ratio - 1) > config["CLIP_EPS"]).astype(jnp.float32)
                        context_gate = actor_aux["context_gate"]
                        total_residual_loss = (
                            config["RESIDUAL_LOSS_COEF"] * residual_loss
                            + config["RESIDUAL_KL_COEF"] * kl_to_base
                        )
                        return total_residual_loss, {
                            "residual_loss": residual_loss,
                            "residual_kl_to_base": kl_to_base,
                            "residual_loss_by_role": role_mean_metric(
                                residual_loss_per_sample,
                                traj_batch.role_id,
                                config["NUM_UNIT_TYPES"],
                            ),
                            "residual_kl_to_base_by_role": role_mean_metric(
                                kl_to_base_per_sample,
                                traj_batch.role_id,
                                config["NUM_UNIT_TYPES"],
                            ),
                            "residual_adv_alignment_by_role": role_mean_metric(
                                adv_alignment_per_sample,
                                traj_batch.role_id,
                                config["NUM_UNIT_TYPES"],
                            ),
                            "residual_logprob_margin_by_role": role_mean_metric(
                                logprob_margin,
                                traj_batch.role_id,
                                config["NUM_UNIT_TYPES"],
                            ),
                            "residual_clip_frac_by_role": role_mean_metric(
                                clip_frac_per_sample,
                                traj_batch.role_id,
                                config["NUM_UNIT_TYPES"],
                            ),
                            "residual_ratio_mean_by_role": role_mean_metric(
                                ratio,
                                traj_batch.role_id,
                                config["NUM_UNIT_TYPES"],
                            ),
                            "residual_ratio_max_by_role": role_max_metric(
                                ratio,
                                traj_batch.role_id,
                                config["NUM_UNIT_TYPES"],
                            ),
                            "context_gate_mean_by_role": role_mean_metric(
                                context_gate,
                                traj_batch.role_id,
                                config["NUM_UNIT_TYPES"],
                            ),
                            "context_gate_std_by_role": role_std_metric(
                                context_gate,
                                traj_batch.role_id,
                                config["NUM_UNIT_TYPES"],
                            ),
                            "context_gate_min_by_role": -role_max_metric(
                                -context_gate,
                                traj_batch.role_id,
                                config["NUM_UNIT_TYPES"],
                            ),
                            "context_gate_max_by_role": role_max_metric(
                                context_gate,
                                traj_batch.role_id,
                                config["NUM_UNIT_TYPES"],
                            ),
                            "role_context_histogram_mean": traj_batch.role_context[..., :-1].mean(axis=(0, 1)),
                        }

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(actor_train_state.params, ac_init_hstate, traj_batch, advantages)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(critic_train_state.params, cr_init_hstate, traj_batch, targets)
                    residual_grad_fn = jax.value_and_grad(_residual_loss_fn, has_aux=True)
                    residual_loss, residual_grads = residual_grad_fn(
                        actor_train_state.params,
                        ac_init_hstate,
                        traj_batch,
                        advantages,
                    )
                    residual_apply_grads, residual_rejected_grads = filter_residual_grads(residual_grads, config)
                    
                    actor_grad_norm = optax.global_norm(actor_grads)
                    critic_grad_norm = optax.global_norm(critic_grads)
                    lora_grad_norm_by_role = role_lora_param_norms(actor_grads, config)
                    lora_param_norm_by_role = role_lora_param_norms(actor_train_state.params, config)
                    residual_lora_grad_norm_by_role = role_lora_param_norms(residual_apply_grads, config)
                    residual_backbone_grad_norm = optax.global_norm(residual_rejected_grads)
                    residual_gate_norm = residual_gate_grad_norm(residual_apply_grads, config)

                    actor_apply_grads = actor_grads
                    if (
                        config["USE_SEQUENTIAL_ROLE_UPDATES"]
                        and config["FREEZE_LORA_IN_SHARED_UPDATE"]
                    ):
                        actor_apply_grads = zero_role_lora_grads(actor_grads, config)

                    if config["USE_ROLE_RESIDUAL_LOSS"]:
                        # Apply actor and residual gradients in one optimizer step. Calling
                        # apply_gradients twice on the same Adam state can move parameters on
                        # the second call even when residual gradients are exactly zero,
                        # because Adam's momentum state is still active.
                        actor_apply_grads = jax.tree.map(
                            lambda actor_grad, residual_grad: actor_grad + residual_grad,
                            actor_apply_grads,
                            residual_apply_grads,
                        )
                    actor_train_state = actor_train_state.apply_gradients(grads=actor_apply_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    residual_applied_loss = (
                        residual_loss[0] if config["USE_ROLE_RESIDUAL_LOSS"] else jnp.array(0.0)
                    )
                    total_loss = actor_loss[0] + critic_loss[0] + residual_applied_loss
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "residual_total_loss": residual_applied_loss,
                        "residual_loss": residual_loss[1]["residual_loss"],
                        "residual_kl_to_base": residual_loss[1]["residual_kl_to_base"],
                        "entropy": actor_loss[1][1],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                        "actor_grad_norm": actor_grad_norm,
                        "critic_grad_norm": critic_grad_norm,
                        "lora_delta_norm_by_role": actor_loss[1][5],
                        "role_adv_mean": actor_loss[1][6],
                        "role_adv_std": actor_loss[1][7],
                        "role_entropy": actor_loss[1][8],
                        "role_approx_kl": actor_loss[1][9],
                        "role_clip_frac": actor_loss[1][10],
                        "role_mean_ratio": actor_loss[1][11],
                        "role_max_ratio": actor_loss[1][12],
                        "role_ppo_loss": actor_loss[1][13],
                        "lora_grad_norm_by_role": lora_grad_norm_by_role,
                        "lora_param_norm_by_role": lora_param_norm_by_role,
                        "residual_loss_by_role": residual_loss[1]["residual_loss_by_role"],
                        "residual_kl_to_base_by_role": residual_loss[1]["residual_kl_to_base_by_role"],
                        "residual_adv_alignment_by_role": residual_loss[1]["residual_adv_alignment_by_role"],
                        "residual_logprob_margin_by_role": residual_loss[1]["residual_logprob_margin_by_role"],
                        "residual_lora_grad_norm_by_role": residual_lora_grad_norm_by_role,
                        "residual_backbone_grad_norm": residual_backbone_grad_norm,
                        "residual_gate_grad_norm": residual_gate_norm,
                        "residual_clip_frac_by_role": residual_loss[1]["residual_clip_frac_by_role"],
                        "residual_ratio_mean_by_role": residual_loss[1]["residual_ratio_mean_by_role"],
                        "residual_ratio_max_by_role": residual_loss[1]["residual_ratio_max_by_role"],
                        "context_gate_mean_by_role": residual_loss[1]["context_gate_mean_by_role"],
                        "context_gate_std_by_role": residual_loss[1]["context_gate_std_by_role"],
                        "context_gate_min_by_role": residual_loss[1]["context_gate_min_by_role"],
                        "context_gate_max_by_role": residual_loss[1]["context_gate_max_by_role"],
                        "role_context_histogram_mean": residual_loss[1]["role_context_histogram_mean"],
                    }
                    return (actor_train_state, critic_train_state), loss_info

                train_states, init_hstates, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                init_hstates = jax.tree.map(lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)), init_hstates)
                
                batch = (init_hstates[0], init_hstates[1], traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(x, [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:])), 1, 0
                    ), shuffled_batch
                )

                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (train_states, jax.tree.map(lambda x: x.squeeze(), init_hstates), traj_batch, advantages, targets, rng)
                return update_state, loss_info

            update_state = (train_states, initial_hstates, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            def reduce_loss_metric(x):
                if x.ndim <= 2:
                    return x.mean()
                return x.mean(axis=tuple(range(x.ndim - 1)))
            loss_info = jax.tree.map(reduce_loss_metric, loss_info)
            train_states = update_state[0]

            role_counts = jnp.bincount(
                traj_batch.role_id.reshape(-1),
                length=config["NUM_UNIT_TYPES"],
            )

            def _empty_role_seq_info():
                zeros = jnp.zeros((config["NUM_UNIT_TYPES"],))
                return {
                    "role_order": jnp.arange(config["NUM_UNIT_TYPES"]),
                    "role_seq_loss_by_role": zeros,
                    "role_seq_kl_by_role": zeros,
                    "role_seq_clip_frac_by_role": zeros,
                    "role_seq_ratio_mean_by_role": zeros,
                    "role_seq_ratio_max_by_role": zeros,
                    "correction_mean": jnp.array(1.0),
                    "correction_std": jnp.array(0.0),
                    "correction_min": jnp.array(1.0),
                    "correction_max": jnp.array(1.0),
                    "correction_after_by_role": jnp.ones((config["NUM_UNIT_TYPES"],)),
                    "adapter_update_norm_by_role": zeros,
                    "role_seq_skipped_by_role": jnp.ones((config["NUM_UNIT_TYPES"],), dtype=jnp.bool_),
                }

            def _run_sequential_role_updates(actor_train_state, role_lora_adam_state, rng):
                if config["ROLE_ORDERING"] == "random":
                    role_order = jax.random.permutation(rng, config["NUM_UNIT_TYPES"])
                elif config["ROLE_ORDERING"] == "fixed":
                    role_order = jnp.arange(config["NUM_UNIT_TYPES"])
                else:
                    raise ValueError(
                        f"Unsupported ROLE_ORDERING={config['ROLE_ORDERING']!r}; "
                        "use 'random' or 'fixed'."
                    )

                seq_lr_base = config["LR"] * config["ROLE_SEQ_LR_MULT"]
                if config["ANNEAL_LR"]:
                    seq_lr_base = seq_lr_base * (
                        1.0 - update_steps.astype(jnp.float32) / jnp.maximum(config["NUM_UPDATES"], 1)
                    )

                zeros = jnp.zeros((config["NUM_UNIT_TYPES"],))
                init_metrics = {
                    "role_seq_loss_by_role": zeros,
                    "role_seq_kl_by_role": zeros,
                    "role_seq_clip_frac_by_role": zeros,
                    "role_seq_ratio_mean_by_role": zeros,
                    "role_seq_ratio_max_by_role": zeros,
                    "adapter_update_norm_by_role": zeros,
                    "correction_after_by_role": jnp.ones((config["NUM_UNIT_TYPES"],)),
                    "role_seq_skipped_by_role": jnp.ones((config["NUM_UNIT_TYPES"],), dtype=jnp.bool_),
                }

                def _correction_env_to_samples(correction_env):
                    correction_by_agent = jnp.broadcast_to(
                        correction_env[:, None, :],
                        (config["NUM_STEPS"], env.num_agents, config["NUM_ENVS"]),
                    )
                    return correction_by_agent.reshape(config["NUM_STEPS"], config["NUM_ACTORS"])

                def _policy_ratios(actor_params):
                    _, pi, _ = actor_network.apply(
                        actor_params,
                        initial_hstates[0],
                        (
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.avail_actions,
                            traj_batch.role_id,
                            traj_batch.adapter_id,
                            traj_batch.role_context,
                        ),
                    )
                    log_prob = pi.log_prob(traj_batch.action)
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(logratio)
                    return ratio, logratio, pi

                def _one_role_update(carry, current_role):
                    actor_train_state, role_lora_adam_state, correction_env, metrics = carry
                    role_mask = (traj_batch.role_id == current_role).astype(jnp.float32)
                    role_count = role_mask.sum()
                    role_present = role_count > 0

                    def _role_loss_fn(actor_params):
                        ratio, logratio, pi = _policy_ratios(actor_params)
                        if config["ROLE_SEQ_ADV_NORM"] == "role":
                            role_adv_mean = jnp.sum(role_mask * advantages) / jnp.maximum(role_count, 1.0)
                            role_adv_var = (
                                jnp.sum(role_mask * jnp.square(advantages - role_adv_mean))
                                / jnp.maximum(role_count, 1.0)
                            )
                            gae = (advantages - role_adv_mean) / (jnp.sqrt(role_adv_var) + 1e-8)
                        elif config["ROLE_SEQ_ADV_NORM"] == "global":
                            gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        else:
                            raise ValueError(
                                f"Unsupported ROLE_SEQ_ADV_NORM={config['ROLE_SEQ_ADV_NORM']!r}; "
                                "use 'role' or 'global'."
                            )
                        correction_per_sample = _correction_env_to_samples(correction_env)
                        corrected_gae = jax.lax.stop_gradient(correction_per_sample) * gae
                        loss_actor1 = ratio * corrected_gae
                        loss_actor2 = (
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"])
                            * corrected_gae
                        )
                        ppo_loss = -jnp.minimum(loss_actor1, loss_actor2)
                        entropy = pi.entropy()
                        weighted_loss = (
                            jnp.sum(role_mask * (ppo_loss - config["ENT_COEF"] * entropy))
                            / jnp.maximum(role_count, 1.0)
                        )
                        approx_kl_sample = (ratio - 1) - logratio
                        clip_frac_sample = (jnp.abs(ratio - 1) > config["CLIP_EPS"]).astype(jnp.float32)
                        role_kl = jnp.sum(role_mask * approx_kl_sample) / jnp.maximum(role_count, 1.0)
                        role_clip_frac = jnp.sum(role_mask * clip_frac_sample) / jnp.maximum(role_count, 1.0)
                        role_ratio_mean = jnp.sum(role_mask * ratio) / jnp.maximum(role_count, 1.0)
                        role_ratio_max = jnp.max(jnp.where(role_mask > 0, ratio, 0.0))
                        return weighted_loss, (role_kl, role_clip_frac, role_ratio_mean, role_ratio_max)

                    (role_loss, aux), role_grads = jax.value_and_grad(_role_loss_fn, has_aux=True)(
                        actor_train_state.params
                    )
                    old_params = unfreeze(actor_train_state.params)
                    old_adam_state = role_lora_adam_state
                    new_params, new_adam_state = update_single_role_lora_params_adam(
                        actor_train_state.params,
                        role_grads,
                        role_lora_adam_state,
                        current_role,
                        seq_lr_base,
                        config,
                    )
                    selected_params = jax.tree.map(
                        lambda new, old: jnp.where(role_present, new, old),
                        new_params,
                        old_params,
                    )
                    selected_adam_state = jax.tree.map(
                        lambda new, old: jnp.where(role_present, new, old),
                        new_adam_state,
                        old_adam_state,
                    )
                    actor_train_state = actor_train_state.replace(
                        params=selected_params
                    )
                    role_lora_adam_state = selected_adam_state
                    update_norm = role_lora_update_norm(old_params, actor_train_state.params, config)
                    role_kl, role_clip_frac, role_ratio_mean, role_ratio_max = aux

                    post_ratio, post_logratio, _ = _policy_ratios(actor_train_state.params)

                    if config["ROLE_SEQ_CORRECTION_MODE"] == "none":
                        next_correction_env = correction_env
                    elif config["ROLE_SEQ_CORRECTION_MODE"] == "role_mean":
                        clipped_role_mean_ratio = jnp.clip(
                            jnp.sum(role_mask * post_ratio) / jnp.maximum(role_count, 1.0),
                            1.0 - config["CORRECTION_CLIP"],
                            1.0 + config["CORRECTION_CLIP"],
                        )
                        next_correction_env = correction_env * clipped_role_mean_ratio
                    elif config["ROLE_SEQ_CORRECTION_MODE"] == "env_mean":
                        role_mask_ae = role_mask.reshape(
                            config["NUM_STEPS"],
                            env.num_agents,
                            config["NUM_ENVS"],
                        )
                        clipped_logratio = jnp.clip(
                            post_logratio,
                            jnp.log(1.0 - config["CORRECTION_CLIP"]),
                            jnp.log(1.0 + config["CORRECTION_CLIP"]),
                        )
                        clipped_logratio_ae = clipped_logratio.reshape(
                            config["NUM_STEPS"],
                            env.num_agents,
                            config["NUM_ENVS"],
                        )
                        role_count_env = jnp.sum(role_mask_ae, axis=1)
                        role_mean_logratio_env = (
                            jnp.sum(role_mask_ae * clipped_logratio_ae, axis=1)
                            / jnp.maximum(role_count_env, 1.0)
                        )
                        correction_multiplier_env = jnp.exp(role_mean_logratio_env)
                        correction_multiplier_env = jnp.where(
                            role_count_env > 0,
                            correction_multiplier_env,
                            1.0,
                        )
                        next_correction_env = correction_env * correction_multiplier_env
                    else:
                        raise ValueError(
                            f"Unsupported ROLE_SEQ_CORRECTION_MODE={config['ROLE_SEQ_CORRECTION_MODE']!r}; "
                            "use 'none', 'role_mean', or 'env_mean'."
                        )
                    next_correction_env = jnp.clip(
                        next_correction_env,
                        1.0 - config["CORRECTION_CLIP"],
                        1.0 + config["CORRECTION_CLIP"],
                    )
                    next_correction_env = jax.lax.stop_gradient(
                        jnp.where(role_present, next_correction_env, correction_env)
                    )
                    correction_after = jnp.mean(next_correction_env)
                    metrics = {
                        "role_seq_loss_by_role": metrics["role_seq_loss_by_role"].at[current_role].set(role_loss),
                        "role_seq_kl_by_role": metrics["role_seq_kl_by_role"].at[current_role].set(role_kl),
                        "role_seq_clip_frac_by_role": metrics["role_seq_clip_frac_by_role"].at[current_role].set(role_clip_frac),
                        "role_seq_ratio_mean_by_role": metrics["role_seq_ratio_mean_by_role"].at[current_role].set(role_ratio_mean),
                        "role_seq_ratio_max_by_role": metrics["role_seq_ratio_max_by_role"].at[current_role].set(role_ratio_max),
                        "adapter_update_norm_by_role": metrics["adapter_update_norm_by_role"].at[current_role].set(update_norm[current_role]),
                        "correction_after_by_role": metrics["correction_after_by_role"].at[current_role].set(correction_after),
                        "role_seq_skipped_by_role": metrics["role_seq_skipped_by_role"].at[current_role].set(~role_present),
                    }
                    return (actor_train_state, role_lora_adam_state, next_correction_env, metrics), None

                correction_env = jnp.ones((config["NUM_STEPS"], config["NUM_ENVS"]))
                (actor_train_state, role_lora_adam_state, correction_env, metrics), _ = jax.lax.scan(
                    _one_role_update,
                    (actor_train_state, role_lora_adam_state, correction_env, init_metrics),
                    role_order,
                )
                correction_values = correction_env.reshape(-1)
                correction_mean = jnp.mean(correction_values)
                correction_std = jnp.std(correction_values)
                seq_info = {
                    "role_order": role_order,
                    **metrics,
                    "correction_mean": correction_mean,
                    "correction_std": correction_std,
                    "correction_min": jnp.min(correction_values),
                    "correction_max": jnp.max(correction_values),
                }
                return actor_train_state, role_lora_adam_state, seq_info

            actor_seq_state = train_states[0]
            role_seq_info = _empty_role_seq_info()
            rng_after_update = update_state[-1]
            if config["USE_SEQUENTIAL_ROLE_UPDATES"]:
                if not config["USE_ROLE_LORA"]:
                    raise ValueError("USE_SEQUENTIAL_ROLE_UPDATES=True requires USE_ROLE_LORA=True.")
                for _ in range(config["ROLE_SEQ_UPDATE_EPOCHS"]):
                    rng_after_update, role_order_rng = jax.random.split(rng_after_update)
                    actor_seq_state, role_lora_adam_state, role_seq_info = _run_sequential_role_updates(
                        actor_seq_state,
                        role_lora_adam_state,
                        role_order_rng,
                    )
                train_states = (actor_seq_state, train_states[1])

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
            lora_delta_norm_by_role = loss_info["lora_delta_norm_by_role"]
            lora_grad_norm_by_role = loss_info["lora_grad_norm_by_role"]
            lora_param_norm_by_role = loss_info["lora_param_norm_by_role"]
            role_adv_mean = loss_info["role_adv_mean"]
            role_adv_std = loss_info["role_adv_std"]
            role_entropy = loss_info["role_entropy"]
            role_approx_kl = loss_info["role_approx_kl"]
            role_clip_frac = loss_info["role_clip_frac"]
            role_mean_ratio = loss_info["role_mean_ratio"]
            role_max_ratio = loss_info["role_max_ratio"]
            role_ppo_loss = loss_info["role_ppo_loss"]
            residual_total_loss = loss_info["residual_total_loss"]
            residual_loss_value = loss_info["residual_loss"]
            residual_kl_to_base = loss_info["residual_kl_to_base"]
            residual_loss_by_role = loss_info["residual_loss_by_role"]
            residual_kl_to_base_by_role = loss_info["residual_kl_to_base_by_role"]
            residual_adv_alignment_by_role = loss_info["residual_adv_alignment_by_role"]
            residual_logprob_margin_by_role = loss_info["residual_logprob_margin_by_role"]
            residual_lora_grad_norm_by_role = loss_info["residual_lora_grad_norm_by_role"]
            residual_backbone_grad_norm = loss_info["residual_backbone_grad_norm"]
            residual_gate_grad_norm_value = loss_info["residual_gate_grad_norm"]
            residual_clip_frac_by_role = loss_info["residual_clip_frac_by_role"]
            residual_ratio_mean_by_role = loss_info["residual_ratio_mean_by_role"]
            residual_ratio_max_by_role = loss_info["residual_ratio_max_by_role"]
            context_gate_mean_by_role = loss_info["context_gate_mean_by_role"]
            context_gate_std_by_role = loss_info["context_gate_std_by_role"]
            context_gate_min_by_role = loss_info["context_gate_min_by_role"]
            context_gate_max_by_role = loss_info["context_gate_max_by_role"]
            role_context_histogram_mean = loss_info["role_context_histogram_mean"]
            role_order = role_seq_info["role_order"]
            role_seq_loss_by_role = role_seq_info["role_seq_loss_by_role"]
            role_seq_kl_by_role = role_seq_info["role_seq_kl_by_role"]
            role_seq_clip_frac_by_role = role_seq_info["role_seq_clip_frac_by_role"]
            role_seq_ratio_mean_by_role = role_seq_info["role_seq_ratio_mean_by_role"]
            role_seq_ratio_max_by_role = role_seq_info["role_seq_ratio_max_by_role"]
            correction_mean = role_seq_info["correction_mean"]
            correction_std = role_seq_info["correction_std"]
            correction_min = role_seq_info["correction_min"]
            correction_max = role_seq_info["correction_max"]
            adapter_update_norm_by_role = role_seq_info["adapter_update_norm_by_role"]
            role_seq_skipped_by_role = role_seq_info["role_seq_skipped_by_role"]
            role_counts = jnp.bincount(
                traj_batch.role_id.reshape(-1),
                length=config["NUM_UNIT_TYPES"],
            )
            role_present_mask = role_counts > 0
            present_float = role_present_mask.astype(jnp.float32)
            present_count = jnp.maximum(present_float.sum(), 1.0)
            present_min_count = jnp.min(jnp.where(role_present_mask, role_counts, jnp.iinfo(role_counts.dtype).max))
            present_max_count = jnp.max(jnp.where(role_present_mask, role_counts, 0))
            max_role_kl = jnp.max(jnp.where(role_present_mask, role_approx_kl, -jnp.inf))
            max_role_clip_frac = jnp.max(jnp.where(role_present_mask, role_clip_frac, -jnp.inf))
            max_role_adv_std = jnp.max(jnp.where(role_present_mask, role_adv_std, -jnp.inf))
            global_adv_std = jnp.sqrt(jnp.maximum(jnp.sum(present_float * jnp.square(role_adv_std)) / present_count, 0.0))
            max_role_kl_minus_global = max_role_kl - loss_info["approx_kl"]
            max_role_clip_frac_minus_global = max_role_clip_frac - loss_info["clip_frac"]
            max_role_adv_std_over_global = max_role_adv_std / (global_adv_std + 1e-8)
            min_role_count_over_max = present_min_count.astype(jnp.float32) / jnp.maximum(present_max_count.astype(jnp.float32), 1.0)
            role_bits = traj_batch.obs[:, :, -config["NUM_UNIT_TYPES"]:]
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
                r, w, s, tl, ent, agn, cgn, rc, rpm, rbmin, rbmax, mismatch,
                zero_bits, max_diff, mean_diff, ldn, lgn, lpn, ram, ras, rent,
                rkl, rcf, rmr, rxr, rpl, dkl, dcf, adv_ratio, count_ratio,
                res_total, res_loss, res_kl, res_lbr, res_kbr, res_align,
                res_margin, res_lgn, res_bgn, res_ggn, res_cfr, res_rmr,
                res_rxr, gate_mean, gate_std, gate_min, gate_max, ctx_hist,
                order, rsl, rskl, rscf, rsrm, rsrx, corr_mean, corr_std,
                corr_min, corr_max, aun, skipped,
            ):
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
                if config["LOG_ZERO_LORA_EQUIV"] and int(s) == 0:
                    msg += (
                        f" | zero_lora_max_abs_logit_diff: {float(max_diff):.8e}"
                        f" | zero_lora_mean_abs_logit_diff: {float(mean_diff):.8e}"
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
                        f" | role_mean_ratio: {fmt(rmr)}"
                        f" | role_max_ratio: {fmt(rxr)}"
                        f" | role_ppo_loss: {fmt(rpl)}"
                        f" | max_role_kl_minus_global: {float(dkl):.4e}"
                        f" | max_role_clip_frac_minus_global: {float(dcf):.4e}"
                        f" | max_role_adv_std_over_global: {float(adv_ratio):.4e}"
                        f" | min_role_count_over_max: {float(count_ratio):.4e}"
                    )
                if config["LOG_RESIDUAL_DIAGNOSTICS"]:
                    msg += (
                        f" | residual_total_loss: {float(res_total):.4e}"
                        f" | residual_loss: {float(res_loss):.4e}"
                        f" | residual_kl_to_base: {float(res_kl):.4e}"
                        f" | residual_loss_by_role: {fmt(res_lbr)}"
                        f" | residual_kl_to_base_by_role: {fmt(res_kbr)}"
                        f" | residual_adv_alignment_by_role: {fmt(res_align)}"
                        f" | residual_logprob_margin_by_role: {fmt(res_margin)}"
                        f" | residual_lora_grad_norm_by_role: {fmt_all(res_lgn)}"
                        f" | residual_backbone_grad_norm: {float(res_bgn):.4e}"
                        f" | residual_gate_grad_norm: {float(res_ggn):.4e}"
                        f" | residual_clip_frac_by_role: {fmt(res_cfr)}"
                        f" | residual_ratio_mean_by_role: {fmt(res_rmr)}"
                        f" | residual_ratio_max_by_role: {fmt(res_rxr)}"
                    )
                    if config["USE_CONTEXT_GATE"]:
                        msg += (
                            f" | context_gate_mean_by_role: {fmt(gate_mean)}"
                            f" | context_gate_std_by_role: {fmt(gate_std)}"
                            f" | context_gate_min_by_role: {fmt(gate_min)}"
                            f" | context_gate_max_by_role: {fmt(gate_max)}"
                            f" | role_context_histogram_mean: {fmt_all(ctx_hist)}"
                        )
                if config["USE_SEQUENTIAL_ROLE_UPDATES"]:
                    order_str = "[" + " ".join(str(int(x)) for x in np.asarray(order)) + "]"
                    skipped_str = "[" + " ".join(str(int(x)) for x in np.asarray(skipped)) + "]"
                    msg += (
                        f" | role_order: {order_str}"
                        f" | role_seq_loss_by_role: {fmt(rsl)}"
                        f" | role_seq_kl_by_role: {fmt(rskl)}"
                        f" | role_seq_clip_frac_by_role: {fmt(rscf)}"
                        f" | role_seq_ratio_mean_by_role: {fmt(rsrm)}"
                        f" | role_seq_ratio_max_by_role: {fmt(rsrx)}"
                        f" | correction_mean/std/min/max: "
                        f"{float(corr_mean):.4e}/{float(corr_std):.4e}/"
                        f"{float(corr_min):.4e}/{float(corr_max):.4e}"
                        f" | adapter_update_norm_by_role: {fmt(aun)}"
                        f" | role_seq_skipped_by_role: {skipped_str}"
                    )
                print(msg)
                if config["LOG_ROLE_DIAGNOSTIC_TABLE"]:
                    print("role | count | adv_mean | adv_std | entropy | approx_kl | clip_frac | mean_ratio | max_ratio | ppo_loss | lora_grad_norm")
                    lgn_arr = np.asarray(lgn)
                    for idx in range(len(np.asarray(rc))):
                        if not bool(np.asarray(rpm)[idx]):
                            print(f"{idx:4d} | {0:5d} | NA | NA | NA | NA | NA | NA | NA | NA | NA")
                            continue
                        adapter_grad_norm = float(lgn_arr[idx]) if idx < len(lgn_arr) else float("nan")
                        print(
                            f"{idx:4d} | {int(np.asarray(rc)[idx]):5d} "
                            f"| {float(np.asarray(ram)[idx]): .4e} "
                            f"| {float(np.asarray(ras)[idx]): .4e} "
                            f"| {float(np.asarray(rent)[idx]): .4e} "
                            f"| {float(np.asarray(rkl)[idx]): .4e} "
                            f"| {float(np.asarray(rcf)[idx]): .4e} "
                            f"| {float(np.asarray(rmr)[idx]): .4e} "
                            f"| {float(np.asarray(rxr)[idx]): .4e} "
                            f"| {float(np.asarray(rpl)[idx]): .4e} "
                            f"| {adapter_grad_norm: .4e}"
                        )

            step_count = update_steps * config["NUM_ENVS"] * config["NUM_STEPS"]
            jax.experimental.io_callback(
                log_callback, None,
                returns, win_rate, step_count,
                total_loss, entropy, actor_grad_norm, critic_grad_norm,
                role_counts, role_present_mask, role_bit_sum_min, role_bit_sum_max,
                obs_role_mismatch_count, zero_obs_role_bits,
                max_abs_logit_diff, mean_abs_logit_diff,
                lora_delta_norm_by_role, lora_grad_norm_by_role, lora_param_norm_by_role,
                role_adv_mean, role_adv_std, role_entropy, role_approx_kl, role_clip_frac,
                role_mean_ratio, role_max_ratio, role_ppo_loss,
                max_role_kl_minus_global, max_role_clip_frac_minus_global,
                max_role_adv_std_over_global, min_role_count_over_max,
                residual_total_loss, residual_loss_value, residual_kl_to_base,
                residual_loss_by_role, residual_kl_to_base_by_role,
                residual_adv_alignment_by_role, residual_logprob_margin_by_role,
                residual_lora_grad_norm_by_role, residual_backbone_grad_norm,
                residual_gate_grad_norm_value, residual_clip_frac_by_role,
                residual_ratio_mean_by_role, residual_ratio_max_by_role,
                context_gate_mean_by_role, context_gate_std_by_role,
                context_gate_min_by_role, context_gate_max_by_role,
                role_context_histogram_mean,
                role_order, role_seq_loss_by_role, role_seq_kl_by_role,
                role_seq_clip_frac_by_role, role_seq_ratio_mean_by_role,
                role_seq_ratio_max_by_role, correction_mean, correction_std,
                correction_min, correction_max, adapter_update_norm_by_role,
                role_seq_skipped_by_role,
            )
            
            update_steps = update_steps + 1
            runner_state = (
                train_states,
                role_lora_adam_state,
                env_state,
                last_obs,
                last_done,
                hstates,
                rng_after_update,
            )
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            role_lora_adam_state,
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
        "USE_SEQUENTIAL_ROLE_UPDATES": False,
        "USE_ROLE_RESIDUAL_LOSS": False,
        "USE_CONTEXT_GATE": False,
        "ROLE_BALANCED_PPO": False,
        "ROLE_LORA_RANK": 4,
        "ROLE_LORA_SCALE": 1.0,
        "ROLE_LORA_A_INIT_STD": 0.01,
        "LOG_ZERO_LORA_EQUIV": False,
        "LOG_ROLE_DIAGNOSTICS": True,
        "LOG_ROLE_DIAGNOSTIC_TABLE": False,
        "ROLE_ORDERING": "random",
        "CORRECTION_CLIP": 0.2,
        "ROLE_SEQ_CORRECTION_MODE": "env_mean",
        "ROLE_SEQ_ADV_NORM": "role",
        "FREEZE_LORA_IN_SHARED_UPDATE": True,
        "ROLE_SEQ_UPDATE_EPOCHS": 1,
        "ROLE_SEQ_LR_MULT": 0.25,
        "ROLE_SEQ_ADAM_BETA1": 0.9,
        "ROLE_SEQ_ADAM_BETA2": 0.999,
        "ROLE_SEQ_ADAM_EPS": 1e-8,
        "LOG_ROSA_CORRECTIONS": True,
        "RESIDUAL_LOSS_COEF": 0.0,
        "RESIDUAL_KL_COEF": 0.0,
        "RESIDUAL_ADV_NORM": "role",
        "RESIDUAL_STOP_BACKBONE": True,
        "LOG_RESIDUAL_DIAGNOSTICS": True,
        "CONTEXT_SOURCE": "team_role_histogram",
        "CONTEXT_SHUFFLE": False,
        "CONTEXT_GATE_HIDDEN_DIM": 32,
        "RESIDUAL_GATE_INIT_BIAS": 0.0,
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
    with open(model_path, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved ROSA-MAPPO actor checkpoint to {model_path}")
    
    # Can optionally save metrics
    # jnp.save("gru_baseline_metrics.npy", out["metric"])
