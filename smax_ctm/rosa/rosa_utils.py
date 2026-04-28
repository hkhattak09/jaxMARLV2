from typing import Dict

import jax.numpy as jnp


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
