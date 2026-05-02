#!/usr/bin/env python3
"""Minimal Pallas skeleton.

This is only a starting point for real kernel work after higher-level JAX
optimisation has been exhausted.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def add_one_kernel(x_ref, y_ref):
    idx = pl.program_id(0)
    y_ref[idx] = x_ref[idx] + 1.0


def main():
    x = jnp.arange(16, dtype=jnp.float32)
    y = pl.pallas_call(
        add_one_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0],),
    )(x)
    print(y)


if __name__ == "__main__":
    main()
