#!/usr/bin/env python3
"""Starter for manual SPMD code with jax.shard_map."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P


def main():
    devices = np.array(jax.devices())
    if devices.size < 2:
        raise SystemExit("This template expects at least 2 devices.")

    mesh = Mesh(devices.reshape((devices.size,)), ("data",))

    @jax.shard_map(mesh=mesh, in_specs=P("data"), out_specs=P("data"))
    def per_shard_scale(x):
        # `x` is the local shard view. Add collectives here if needed.
        return 2.0 * x

    x = jnp.arange(devices.size * 4, dtype=jnp.float32).reshape(devices.size, 4)
    y = per_shard_scale(x)
    print(y)


if __name__ == "__main__":
    main()
