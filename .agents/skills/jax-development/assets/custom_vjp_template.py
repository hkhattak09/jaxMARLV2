#!/usr/bin/env python3
"""Custom VJP starter."""

import jax
import jax.numpy as jnp


@jax.custom_vjp
def clipped_log1p(x):
    return jnp.log1p(jnp.maximum(x, -0.999999))


def clipped_log1p_fwd(x):
    y = clipped_log1p(x)
    return y, x


def clipped_log1p_bwd(residual, g):
    (x,) = (residual,)
    grad = 1.0 / (1.0 + jnp.maximum(x, -0.999999))
    return (g * grad,)


clipped_log1p.defvjp(clipped_log1p_fwd, clipped_log1p_bwd)


def main():
    x = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    print(clipped_log1p(x))
    print(jax.grad(lambda t: jnp.sum(clipped_log1p(t)))(x))


if __name__ == "__main__":
    main()
