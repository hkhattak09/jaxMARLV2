import jax
import jax.numpy as jnp

@jax.jit
def accumulate(x):
    acc = jnp.zeros_like(x[0])
    for i in range(1000):
        acc = acc + jnp.sin(x[i])
    return acc
