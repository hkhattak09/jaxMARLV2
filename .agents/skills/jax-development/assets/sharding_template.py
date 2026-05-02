#!/usr/bin/env python3
"""Starter for global-view sharding with Mesh and NamedSharding."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def main():
    devices = np.array(jax.devices())
    if devices.size < 2:
        raise SystemExit("This template expects at least 2 devices.")

    mesh = Mesh(devices.reshape((devices.size,)), ("data",))
    data_sharding = NamedSharding(mesh, P("data", None))
    repl_sharding = NamedSharding(mesh, P())

    x = jax.device_put(jnp.arange(devices.size * 8, dtype=jnp.float32).reshape(devices.size, 8), data_sharding)
    w = jax.device_put(jnp.eye(8, dtype=jnp.float32), repl_sharding)

    @jax.jit
    def f(x, w):
        return x @ w

    y = f(x, w)
    print("input sharding:", x.sharding)
    print("output sharding:", y.sharding)


if __name__ == "__main__":
    main()
