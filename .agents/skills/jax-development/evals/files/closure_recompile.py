import jax
import jax.numpy as jnp

def run(xs, scale):
    outputs = []
    for x in xs:
        f = jax.jit(lambda y: scale * y + 1.0)
        outputs.append(f(x))
    return outputs
