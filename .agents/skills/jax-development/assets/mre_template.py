#!/usr/bin/env python3
"""Minimal reproducible example template for JAX bugs."""

import jax
import jax.numpy as jnp


def f(x):
    # Replace with the smallest function that still reproduces the problem.
    return x + 1


def main():
    x = jnp.arange(4, dtype=jnp.float32)
    print("eager:", f(x))
    print("jit:", jax.jit(f)(x))


if __name__ == "__main__":
    main()
