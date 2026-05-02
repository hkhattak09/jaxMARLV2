#!/usr/bin/env python3
"""Trace and memory-profile starter."""

import jax
import jax.numpy as jnp
import jax.profiler


@jax.jit
def step(x):
    return jnp.tanh(x @ x.T)


def main():
    x = jax.random.normal(jax.random.key(0), (2048, 256))

    with jax.profiler.trace("/tmp/jax-trace"):
        for _ in range(5):
            x = step(x[:, :256])
            x.block_until_ready()

    # Optional: capture device memory profile for OOM / leak work.
    jax.profiler.save_device_memory_profile("/tmp/jax-memory.prof")
    print("Wrote /tmp/jax-trace and /tmp/jax-memory.prof")


if __name__ == "__main__":
    main()
