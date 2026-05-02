import jax
import jax.numpy as jnp

@jax.jit
def clip_or_negate(x):
    if x.mean() > 0:
        return jnp.clip(x, 0.0, 1.0)
    return -x
