from typing import Dict, Tuple

import distrax
import jax
import jax.numpy as jnp


def _dense(x: jnp.ndarray, params: Dict[str, jnp.ndarray], layer_name: str) -> jnp.ndarray:
    if "kernel" not in params or "bias" not in params:
        raise KeyError(f"{layer_name} params must contain kernel and bias")

    kernel = params["kernel"]
    bias = params["bias"]

    if x.ndim != 2:
        raise ValueError(f"Expected rank-2 input to {layer_name}, got shape {x.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"Expected rank-2 kernel in {layer_name}, got shape {kernel.shape}")
    if bias.ndim != 1:
        raise ValueError(f"Expected rank-1 bias in {layer_name}, got shape {bias.shape}")
    if x.shape[-1] != kernel.shape[0]:
        raise ValueError(
            f"Input dim mismatch in {layer_name}: x={x.shape}, kernel={kernel.shape}"
        )
    if kernel.shape[1] != bias.shape[0]:
        raise ValueError(
            f"Output dim mismatch in {layer_name}: kernel={kernel.shape}, bias={bias.shape}"
        )

    return jnp.dot(x, kernel) + bias


def compute_action_logits(
    synch: jnp.ndarray,
    avail_actions: jnp.ndarray,
    head_params: Dict[str, Dict[str, jnp.ndarray]],
) -> jnp.ndarray:
    if synch.ndim != 2:
        raise ValueError(f"Expected synch shape (num_agents, synch_size), got {synch.shape}")
    if avail_actions.ndim != 2:
        raise ValueError(
            f"Expected avail_actions shape (num_agents, action_dim), got {avail_actions.shape}"
        )
    if synch.shape[0] != avail_actions.shape[0]:
        raise ValueError(
            f"Batch mismatch: synch={synch.shape}, avail_actions={avail_actions.shape}"
        )

    x = _dense(synch, head_params["Dense_0"], "Dense_0")
    x = jax.nn.relu(x)
    x = _dense(x, head_params["Dense_1"], "Dense_1")
    x = jax.nn.relu(x)
    logits = _dense(x, head_params["Dense_2"], "Dense_2")

    if logits.shape != avail_actions.shape:
        raise ValueError(
            f"Logits/action-mask shape mismatch: logits={logits.shape}, avail={avail_actions.shape}"
        )

    unavail_actions = 1 - avail_actions
    return logits - (unavail_actions * 1e10)


def policy_from_synch(
    synch: jnp.ndarray,
    avail_actions: jnp.ndarray,
    head_params: Dict[str, Dict[str, jnp.ndarray]],
) -> distrax.Categorical:
    logits = compute_action_logits(synch, avail_actions, head_params)
    return distrax.Categorical(logits=logits)


def choose_actions(
    pi: distrax.Categorical,
    rng: jax.Array,
    stochastic: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if stochastic:
        actions = pi.sample(seed=rng)
    else:
        actions = pi.mode()
    log_prob = pi.log_prob(actions)
    return actions, log_prob
