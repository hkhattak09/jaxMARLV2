import numpy as np
import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    y = jnp.sin(x)
    return np.asarray(y).sum()
