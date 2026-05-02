import jax
import jax.numpy as jnp

def matmul_step(x, w):
    return jnp.tanh(x @ w)

def benchmark_once():
    x = jax.random.normal(jax.random.key(0), (2048, 512))
    w = jax.random.normal(jax.random.key(1), (512, 512))
    fast = jax.jit(matmul_step)
    return fast(x, w)  # caller times this without blocking
