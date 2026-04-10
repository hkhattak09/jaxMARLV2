#!/usr/bin/env python3
"""
One-shot Stage 0 checker for batchify axis ordering.

It mirrors train_mappo_ctm.batchify:
    x = jnp.stack([x[a] for a in agent_list])
    x = x.reshape((num_actors, -1))

Expected ordering is agent-major.
"""

import jax.numpy as jnp


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def main() -> None:
    num_agents = 3
    num_envs = 4
    feat_dim = 1

    agents = [f"a{i}" for i in range(num_agents)]
    x = {
        "a0": jnp.ones((num_envs, feat_dim)) * 0,
        "a1": jnp.ones((num_envs, feat_dim)) * 1,
        "a2": jnp.ones((num_envs, feat_dim)) * 2,
    }

    num_actors = num_agents * num_envs
    flat = batchify(x, agents, num_actors)

    print("flat[:, 0] =", flat[:, 0].tolist())

    agent_major = flat.reshape((num_agents, num_envs, feat_dim))
    print("agent_major[:, :, 0] =")
    print(agent_major[:, :, 0])

    env_major = flat.reshape((num_envs, num_agents, feat_dim))
    print("env_major[:, :, 0] =")
    print(env_major[:, :, 0])

    expected = jnp.concatenate([
        jnp.zeros((num_envs,)),
        jnp.ones((num_envs,)),
        jnp.ones((num_envs,)) * 2,
    ])
    if not jnp.array_equal(flat[:, 0], expected):
        raise AssertionError("Unexpected ordering: flat axis is not agent-major.")

    print("PASS: flat axis is agent-major (agent blocks of length num_envs).")


if __name__ == "__main__":
    main()
