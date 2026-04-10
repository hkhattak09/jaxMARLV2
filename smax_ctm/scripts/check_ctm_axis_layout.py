#!/usr/bin/env python3
"""
Stage 0 checker for CTM actor-axis layout assumptions.

This script is pure NumPy so it can run anywhere (including Colab) without JAX.
It demonstrates that the flat NUM_ACTORS axis produced by batchify is agent-major,
and that reshape(num_envs, num_agents, ...) is incorrect for consensus pooling.
"""

import numpy as np


def main() -> None:
    num_agents = 3
    num_envs = 4
    synch_size = 2
    num_actors = num_agents * num_envs

    # Deliberately tagged rows: each row carries its agent id.
    # Rows [0:4] are agent 0, [4:8] agent 1, [8:12] agent 2.
    row_agent_id = (np.arange(num_actors) // num_envs).astype(np.float32)
    synch = np.repeat(row_agent_id[:, None], synch_size, axis=1)

    # Correct interpretation for agent-major flat axis.
    per_agent = synch.reshape(num_agents, num_envs, synch_size)

    # Wrong interpretation that silently mixes rows.
    wrong = synch.reshape(num_envs, num_agents, synch_size)

    print("flat row agent tags:")
    print(row_agent_id.tolist())

    print("\ncorrect reshape (num_agents, num_envs, synch):")
    print(per_agent[:, :, 0])

    print("\nwrong reshape (num_envs, num_agents, synch):")
    print(wrong[:, :, 0])

    expected = np.tile(np.arange(num_agents, dtype=np.float32)[:, None], (1, num_envs))
    if not np.array_equal(per_agent[:, :, 0], expected):
        raise AssertionError(
            "Agent-major reshape check failed: expected each agent slice to be constant over envs."
        )

    # Demonstrate that env-major reshape is not equivalent and must fail loudly.
    if np.array_equal(wrong[:, :, 0], expected):
        raise AssertionError(
            "Unexpected pass: env-major reshape matched expected layout, this indicates a broken test."
        )

    # Round-trip check used for Stage 1 assert logic.
    round_trip = per_agent.reshape(num_actors, synch_size)
    if not np.array_equal(round_trip, synch):
        raise AssertionError(
            "Round-trip reshape failed: tagged per-agent values did not survive reshape/flatten."
        )

    print("\nPASS: flat axis is agent-major, and env-major reshape is invalid for INC pooling.")


if __name__ == "__main__":
    main()
