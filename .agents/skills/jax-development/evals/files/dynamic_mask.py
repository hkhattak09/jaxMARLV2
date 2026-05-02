import jax
import jax.numpy as jnp

@jax.jit
def positives_only(x):
    return x[x > 0]
