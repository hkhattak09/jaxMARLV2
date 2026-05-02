#!/usr/bin/env python3
"""Honest JAX benchmarking starter."""

import time

import jax
import jax.numpy as jnp


def workload(x):
    return jnp.tanh(x @ x.T).sum()


def main():
    x = jax.random.normal(jax.random.key(0), (2048, 256))
    workload_jit = jax.jit(workload)

    # Warm-up / compile
    workload_jit(x).block_until_ready()

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        y = workload_jit(x)
        y.block_until_ready()
        times.append((time.perf_counter() - t0) * 1e3)

    print("steady-state times_ms:", times)
    print("mean_ms:", sum(times) / len(times))


if __name__ == "__main__":
    main()
