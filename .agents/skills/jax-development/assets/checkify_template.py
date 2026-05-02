#!/usr/bin/env python3
"""Runtime checks that survive jit via checkify."""

from jax.experimental import checkify
import jax
import jax.numpy as jnp


def f(x, i):
    checkify.check(i >= 0, "index must be non-negative: {i}", i=i)
    checkify.check(i < x.shape[0], "index out of bounds: {i}", i=i)
    y = x[i]
    checkify.check(jnp.isfinite(y).all(), "non-finite output")
    return y


def main():
    x = jnp.arange(8, dtype=jnp.float32)
    checked = checkify.checkify(f, errors=checkify.user_checks | checkify.index_checks)
    err, y = jax.jit(checked)(x, 3)
    err.throw()
    print("y:", y)


if __name__ == "__main__":
    main()
