#!/usr/bin/env python3
"""Export / serialisation starter."""

import numpy as np

import jax
import jax.numpy as jnp
from jax import export


def f(x):
    return jnp.sin(x) + 2.0 * x


def main():
    sig = jax.ShapeDtypeStruct((4,), np.float32)
    exported = export.export(jax.jit(f))(sig)
    print("platforms:", exported.platforms)
    print("in_avals:", exported.in_avals)
    print("out_avals:", exported.out_avals)


if __name__ == "__main__":
    main()
