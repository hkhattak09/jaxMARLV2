import jax
import jax.numpy as jnp

@jax.pmap(axis_name="data")
def step(x):
    y = x - jax.lax.pmean(x, "data")
    return y / (1e-6 + jnp.linalg.norm(y, axis=-1, keepdims=True))
