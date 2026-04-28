"""JAX ValueNorm implementation ported from MACA/harl/common/valuenorm.py.

This module provides a JAX-compatible, immutable ValueNorm that can be used
inside jit-compiled functions. The state is a PyTree carrying running statistics.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct
from typing import Optional, Tuple


@struct.dataclass
class ValueNormState:
    """Immutable state for ValueNorm running statistics.
    
    Attributes:
        running_mean: Running mean of values.
        running_mean_sq: Running mean of squared values.
        debiasing_term: Debiasing term for EMA updates.
        beta: EMA decay rate (default 0.99999 from MACA).
        epsilon: Small constant for numerical stability (default 1e-5).
        var_clamp_min: Minimum variance clamp value (default 1e-2).
        norm_axes: Number of leading axes that constitute the batch over which
            statistics are computed. 1 means scalar values (batch,), 2 means
            per-agent or per-value-dim stats (batch, value_dim).
    """
    running_mean: jnp.ndarray
    running_mean_sq: jnp.ndarray
    debiasing_term: jnp.ndarray
    beta: float = struct.field(pytree_node=False, default=0.99999)
    epsilon: float = struct.field(pytree_node=False, default=1e-5)
    var_clamp_min: float = struct.field(pytree_node=False, default=1e-2)
    norm_axes: int = struct.field(pytree_node=False, default=1)


def init_value_norm(
    input_shape: Tuple[int, ...],
    beta: float = 0.99999,
    epsilon: float = 1e-5,
    var_clamp_min: float = 1e-2,
    norm_axes: int = 1,
) -> ValueNormState:
    """Initialize a ValueNormState.
    
    Args:
        input_shape: Shape of the input values (e.g., (1,) for scalar values).
        beta: EMA decay rate (default 0.99999 from MACA).
        epsilon: Small constant for numerical stability.
        var_clamp_min: Minimum variance clamp value.
        norm_axes: Number of leading axes treated as batch dimensions.
        
    Returns:
        Initialized ValueNormState.
    """
    return ValueNormState(
        running_mean=jnp.zeros(input_shape, dtype=jnp.float32),
        running_mean_sq=jnp.zeros(input_shape, dtype=jnp.float32),
        debiasing_term=jnp.zeros((), dtype=jnp.float32),
        beta=beta,
        epsilon=epsilon,
        var_clamp_min=var_clamp_min,
        norm_axes=norm_axes,
    )


def value_norm_running_stats(state: ValueNormState) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get debiased running mean and variance.
    
    Args:
        state: Current ValueNormState.
        
    Returns:
        Tuple of (debiased_mean, debiased_var).
    """
    debiased_mean = state.running_mean / jnp.maximum(
        state.debiasing_term, state.epsilon
    )
    debiased_mean_sq = state.running_mean_sq / jnp.maximum(
        state.debiasing_term, state.epsilon
    )
    debiased_var = (debiased_mean_sq - debiased_mean ** 2)
    debiased_var = jnp.maximum(debiased_var, state.var_clamp_min)
    return debiased_mean, debiased_var


def value_norm_update(state: ValueNormState, x: jnp.ndarray) -> ValueNormState:
    """Update running statistics with a new batch of data.
    
    This implements the EMA update from MACA:
    - batch_mean = mean(x)
    - batch_sq_mean = mean(x^2)
    - running_mean = beta * running_mean + (1 - beta) * batch_mean
    - running_mean_sq = beta * running_mean_sq + (1 - beta) * batch_sq_mean
    - debiasing_term = beta * debiasing_term + (1 - beta) * 1.0
    
    Args:
        state: Current ValueNormState.
        x: Batch of values to update with. Shape should be (batch, ...).
        
    Returns:
        Updated ValueNormState.
    """
    # Compute batch statistics over the leading norm_axes dimensions
    axes = tuple(range(state.norm_axes))
    batch_mean = jnp.mean(x, axis=axes)
    batch_sq_mean = jnp.mean(x ** 2, axis=axes)
    
    # EMA update
    weight = state.beta
    new_running_mean = weight * state.running_mean + (1.0 - weight) * batch_mean
    new_running_mean_sq = weight * state.running_mean_sq + (1.0 - weight) * batch_sq_mean
    new_debiasing_term = weight * state.debiasing_term + (1.0 - weight) * 1.0
    
    return ValueNormState(
        running_mean=new_running_mean,
        running_mean_sq=new_running_mean_sq,
        debiasing_term=new_debiasing_term,
        beta=state.beta,
        epsilon=state.epsilon,
        var_clamp_min=state.var_clamp_min,
        norm_axes=state.norm_axes,
    )


def value_norm_normalize(state: ValueNormState, x: jnp.ndarray) -> jnp.ndarray:
    """Normalize input using running statistics.
    
    Args:
        state: Current ValueNormState.
        x: Input to normalize.
        
    Returns:
        Normalized input.
    """
    mean, var = value_norm_running_stats(state)
    prefix = (None,) * state.norm_axes
    return (x - mean[prefix]) / jnp.sqrt(var)[prefix]


def value_norm_denormalize(state: ValueNormState, x: jnp.ndarray) -> jnp.ndarray:
    """Denormalize input back to original scale.
    
    Args:
        state: Current ValueNormState.
        x: Normalized input to denormalize.
        
    Returns:
        Denormalized input.
    """
    mean, var = value_norm_running_stats(state)
    prefix = (None,) * state.norm_axes
    return x * jnp.sqrt(var)[prefix] + mean[prefix]


def create_value_norm_dict(
    use_valuenorm: bool,
    v_shape: Tuple[int, ...] = (1,),
    q_shape: Tuple[int, ...] = (1,),
    eq_shape: Tuple[int, ...] = (1,),
    beta: float = 0.99999,
    epsilon: float = 1e-5,
    var_clamp_min: float = 1e-2,
    norm_axes: int = 1,
) -> Optional[dict]:
    """Create a dictionary of ValueNorm states for v, q, and eq.
    
    Args:
        use_valuenorm: Whether to use ValueNorm. If False, returns None.
        v_shape: Shape for v ValueNorm.
        q_shape: Shape for q ValueNorm.
        eq_shape: Shape for eq ValueNorm.
        beta: EMA decay rate.
        epsilon: Numerical stability constant.
        var_clamp_min: Minimum variance clamp.
        norm_axes: Number of leading axes treated as batch dimensions.
        
    Returns:
        Dictionary with 'v', 'q', 'eq' ValueNormState or None.
    """
    if not use_valuenorm:
        return None
    
    return {
        "v": init_value_norm(v_shape, beta, epsilon, var_clamp_min, norm_axes),
        "q": init_value_norm(q_shape, beta, epsilon, var_clamp_min, norm_axes),
        "eq": init_value_norm(eq_shape, beta, epsilon, var_clamp_min, norm_axes),
    }


def update_value_norm_dict(
    norm_dict: dict, value_targets: jnp.ndarray, q_targets: jnp.ndarray, eq_targets: jnp.ndarray
) -> dict:
    """Update all three ValueNorm states with their respective targets.
    
    Args:
        norm_dict: Dictionary with 'v', 'q', 'eq' ValueNormState.
        value_targets: Targets for v ValueNorm.
        q_targets: Targets for q ValueNorm.
        eq_targets: Targets for eq ValueNorm.
        
    Returns:
        Updated norm_dict.
    """
    if norm_dict is None:
        return None
    
    return {
        "v": value_norm_update(norm_dict["v"], value_targets),
        "q": value_norm_update(norm_dict["q"], q_targets),
        "eq": value_norm_update(norm_dict["eq"], eq_targets),
    }


def normalize_targets(
    norm_dict: dict, value_targets: jnp.ndarray, q_targets: jnp.ndarray, eq_targets: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Normalize targets using ValueNorm states.
    
    Args:
        norm_dict: Dictionary with 'v', 'q', 'eq' ValueNormState.
        value_targets: Raw value targets.
        q_targets: Raw q targets.
        eq_targets: Raw eq targets.
        
    Returns:
        Tuple of (normalized_value_targets, normalized_q_targets, normalized_eq_targets).
        If norm_dict is None, returns the raw targets.
    """
    if norm_dict is None:
        return value_targets, q_targets, eq_targets
    
    return (
        value_norm_normalize(norm_dict["v"], value_targets),
        value_norm_normalize(norm_dict["q"], q_targets),
        value_norm_normalize(norm_dict["eq"], eq_targets),
    )


def denormalize_predictions(
    norm_dict: dict, v_pred: jnp.ndarray, q_pred: jnp.ndarray, eq_pred: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Denormalize predictions using ValueNorm states.
    
    Args:
        norm_dict: Dictionary with 'v', 'q', 'eq' ValueNormState.
        v_pred: Normalized v predictions.
        q_pred: Normalized q predictions.
        eq_pred: Normalized eq predictions.
        
    Returns:
        Tuple of (denormalized_v, denormalized_q, denormalized_eq).
        If norm_dict is None, returns the raw predictions.
    """
    if norm_dict is None:
        return v_pred, q_pred, eq_pred
    
    return (
        value_norm_denormalize(norm_dict["v"], v_pred),
        value_norm_denormalize(norm_dict["q"], q_pred),
        value_norm_denormalize(norm_dict["eq"], eq_pred),
    )
