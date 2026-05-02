import jax
import jax.numpy as jnp

GLOBAL_KEY = jax.random.key(0)

@jax.jit
def sample_pair():
    a = jax.random.normal(GLOBAL_KEY, (4,))
    b = jax.random.uniform(GLOBAL_KEY, (4,))
    return a, b
