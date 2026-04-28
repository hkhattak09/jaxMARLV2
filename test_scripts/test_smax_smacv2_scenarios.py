#!/usr/bin/env python3
"""Smoke/regression tests for SMAX SMACv2 race scenarios.

Run from the repository root, for example in Colab:

    python test_scripts/test_smax_smacv2_scenarios.py

The script assumes JAX/Flax/Chex are installed in the runtime. It does not
require StarCraft II, PySC2, or SMACv2 because it tests the pure-JAX SMAX code.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax
import jax.numpy as jnp

from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.environments.smax.smax_env import SMAX


RACE_SCENARIOS = {
    "protoss_5_vs_5": {
        "n_allies": 5,
        "n_enemies": 5,
        "allowed": {2, 3, 6},
        "weights": {2: 0.45, 3: 0.45, 6: 0.10},
        "exceptions": set(),
    },
    "protoss_10_vs_10": {
        "n_allies": 10,
        "n_enemies": 10,
        "allowed": {2, 3, 6},
        "weights": {2: 0.45, 3: 0.45, 6: 0.10},
        "exceptions": set(),
    },
    "terran_5_vs_5": {
        "n_allies": 5,
        "n_enemies": 5,
        "allowed": {0, 1, 7},
        "weights": {0: 0.45, 1: 0.45, 7: 0.10},
        "exceptions": {7},
    },
    "terran_10_vs_10": {
        "n_allies": 10,
        "n_enemies": 10,
        "allowed": {0, 1, 7},
        "weights": {0: 0.45, 1: 0.45, 7: 0.10},
        "exceptions": {7},
    },
    "zerg_5_vs_5": {
        "n_allies": 5,
        "n_enemies": 5,
        "allowed": {4, 5, 8},
        "weights": {4: 0.45, 5: 0.45, 8: 0.10},
        "exceptions": {8},
    },
    "zerg_10_vs_10": {
        "n_allies": 10,
        "n_enemies": 10,
        "allowed": {4, 5, 8},
        "weights": {4: 0.45, 5: 0.45, 8: 0.10},
        "exceptions": {8},
    },
}


def assert_true(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def as_int_list(x):
    return [int(v) for v in jax.device_get(x).tolist()]


def random_valid_actions(key, env, state):
    avail = env.get_avail_actions(state)
    actions = {}
    for i, agent in enumerate(env.agents):
        key, subkey = jax.random.split(key)
        valid = jnp.nonzero(avail[agent], size=env.action_spaces[agent].n)[0]
        n_valid = jnp.sum(avail[agent])
        choice = jax.random.randint(subkey, shape=(), minval=0, maxval=n_valid)
        actions[agent] = valid[choice].astype(jnp.int32)
    return key, actions


def test_roster_widths():
    base_env = SMAX(scenario=map_name_to_scenario("3m"))
    assert_true(base_env.unit_type_bits == 6, "Old fixed scenarios should keep 6 unit bits")

    v2_env = SMAX(scenario=map_name_to_scenario("terran_5_vs_5"))
    assert_true(v2_env.unit_type_bits == 9, "SMACv2 race scenarios should use 9 unit bits")
    assert_true(v2_env.medivac_type_idx == 7, "Medivac id should be 7")
    assert_true(v2_env.baneling_type_idx == 8, "Baneling id should be 8")


def test_weighted_resets(samples):
    for scenario_name, expected in RACE_SCENARIOS.items():
        env = SMAX(scenario=map_name_to_scenario(scenario_name))
        counts = Counter()
        total_units = 0

        for seed in range(samples):
            _, state = env.reset(jax.random.PRNGKey(seed))
            unit_types = as_int_list(state.unit_types)
            allies = unit_types[: expected["n_allies"]]
            enemies = unit_types[expected["n_allies"] :]

            assert_true(env.num_allies == expected["n_allies"], f"{scenario_name}: ally count mismatch")
            assert_true(env.num_enemies == expected["n_enemies"], f"{scenario_name}: enemy count mismatch")
            assert_true(set(unit_types) <= expected["allowed"], f"{scenario_name}: unexpected unit type {unit_types}")
            assert_true(allies == enemies[: len(allies)], f"{scenario_name}: enemy prefix should mirror allies")

            if expected["exceptions"]:
                assert_true(
                    not set(allies).issubset(expected["exceptions"]),
                    f"{scenario_name}: ally team was all exception units",
                )
                assert_true(
                    not set(enemies).issubset(expected["exceptions"]),
                    f"{scenario_name}: enemy team was all exception units",
                )

            counts.update(unit_types)
            total_units += len(unit_types)

        frequencies = {unit_type: counts[unit_type] / total_units for unit_type in expected["allowed"]}
        for unit_type, target in expected["weights"].items():
            tolerance = 0.08
            assert_true(
                abs(frequencies[unit_type] - target) < tolerance,
                f"{scenario_name}: unit {unit_type} frequency {frequencies[unit_type]:.3f} "
                f"too far from target {target:.3f}",
            )


def test_short_rollouts():
    for scenario_name in RACE_SCENARIOS:
        env = SMAX(scenario=map_name_to_scenario(scenario_name), max_steps=8)
        key = jax.random.PRNGKey(10_000)
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        assert_true(obs["world_state"].shape == (env.state_size + env.num_agents * 2,), f"{scenario_name}: bad world state shape")

        for _ in range(4):
            key, action_key, step_key = jax.random.split(key, 3)
            action_key, actions = random_valid_actions(action_key, env, state)
            obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)
            del rewards, infos
            assert_true(set(obs.keys()) == set(env.agents + ["world_state"]), f"{scenario_name}: bad obs keys")
            assert_true("__all__" in dones, f"{scenario_name}: missing __all__ done")
            assert_true(jnp.all(jnp.isfinite(state.unit_health)), f"{scenario_name}: non-finite health")
            assert_true(jnp.all(state.unit_health >= 0), f"{scenario_name}: negative health")


def test_heuristic_enemy_wrapper():
    env = HeuristicEnemySMAX(
        scenario=map_name_to_scenario("terran_5_vs_5"),
        enemy_shoots=True,
        max_steps=8,
    )
    key = jax.random.PRNGKey(20_000)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    assert_true(set(obs.keys()) == set(env.agents + ["world_state"]), "Heuristic wrapper reset obs keys mismatch")

    for _ in range(3):
        key, action_key, step_key = jax.random.split(key, 3)
        avail = env.get_avail_actions(state)
        actions = {}
        for agent in env.agents:
            action_key, subkey = jax.random.split(action_key)
            valid = jnp.nonzero(avail[agent], size=env.action_spaces[agent].n)[0]
            n_valid = jnp.sum(avail[agent])
            choice = jax.random.randint(subkey, shape=(), minval=0, maxval=n_valid)
            actions[agent] = valid[choice].astype(jnp.int32)
        obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)
        del rewards, infos
        assert_true("__all__" in dones, "Heuristic wrapper missing __all__ done")
        assert_true("world_state" in obs, "Heuristic wrapper missing world_state")


def test_medivac_mechanics():
    env = SMAX(
        scenario=map_name_to_scenario("terran_5_vs_5"),
        world_steps_per_env_step=1,
        walls_cause_death=False,
    )
    _, state = env.reset(jax.random.PRNGKey(30_000))

    unit_types = jnp.array([7, 0, 0, 0, 0, 0, 1, 0, 1, 0], dtype=jnp.uint8)
    positions = jnp.array(
        [
            [10.0, 10.0],
            [11.0, 10.0],
            [14.0, 14.0],
            [14.5, 14.0],
            [15.0, 14.0],
            [20.0, 20.0],
            [21.0, 20.0],
            [22.0, 20.0],
            [23.0, 20.0],
            [24.0, 20.0],
        ],
        dtype=jnp.float32,
    )
    health = env.unit_type_health[unit_types].at[1].set(10.0)
    state = state.replace(
        unit_types=unit_types,
        unit_positions=positions,
        unit_health=health,
        unit_alive=jnp.ones((env.num_agents,), dtype=jnp.bool_),
        unit_weapon_cooldowns=jnp.zeros((env.num_agents,)),
    )

    avail = env.get_avail_actions(state)
    assert_true(int(avail["ally_0"][env.num_movement_actions + 0]) == 0, "Medivac should not heal itself")
    assert_true(int(avail["ally_0"][env.num_movement_actions + 1]) == 1, "Medivac should heal ally_1")

    movements = jnp.zeros((env.num_agents, 2), dtype=jnp.float32)
    attacks = jnp.zeros((env.num_agents,), dtype=jnp.int32).at[0].set(env.num_movement_actions + 1)
    _, new_state, _, _, _ = env.step_env_no_decode(
        jax.random.PRNGKey(30_001),
        state,
        (movements, attacks),
    )

    assert_true(new_state.unit_health[1] > state.unit_health[1], "Medivac did not heal damaged ally")
    assert_true(
        new_state.unit_health[1] <= env.unit_type_health[unit_types[1]],
        "Medivac healing exceeded max health",
    )


def test_baneling_mechanics():
    env = SMAX(
        scenario=map_name_to_scenario("zerg_5_vs_5"),
        world_steps_per_env_step=1,
        walls_cause_death=False,
    )
    _, state = env.reset(jax.random.PRNGKey(40_000))

    unit_types = jnp.array([8, 4, 5, 4, 5, 4, 5, 4, 5, 4], dtype=jnp.uint8)
    positions = jnp.array(
        [
            [10.0, 10.0],
            [13.0, 13.0],
            [14.0, 13.0],
            [15.0, 13.0],
            [16.0, 13.0],
            [10.8, 10.0],
            [11.4, 10.0],
            [20.0, 20.0],
            [21.0, 20.0],
            [22.0, 20.0],
        ],
        dtype=jnp.float32,
    )
    state = state.replace(
        unit_types=unit_types,
        unit_positions=positions,
        unit_health=env.unit_type_health[unit_types],
        unit_alive=jnp.ones((env.num_agents,), dtype=jnp.bool_),
        unit_weapon_cooldowns=jnp.zeros((env.num_agents,)),
    )

    movements = jnp.zeros((env.num_agents, 2), dtype=jnp.float32)
    attacks = jnp.zeros((env.num_agents,), dtype=jnp.int32).at[0].set(env.num_movement_actions)
    _, new_state, _, _, _ = env.step_env_no_decode(
        jax.random.PRNGKey(40_001),
        state,
        (movements, attacks),
    )

    assert_true(new_state.unit_health[0] == 0, "Baneling should die after a valid attack")
    assert_true(new_state.unit_health[5] < state.unit_health[5], "Baneling target did not take damage")
    assert_true(new_state.unit_health[6] < state.unit_health[6], "Baneling splash target did not take damage")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=256, help="Number of resets per scenario for distribution checks")
    args = parser.parse_args()

    print("Testing roster widths...")
    test_roster_widths()
    print("Testing weighted resets and exception sampling...")
    test_weighted_resets(args.samples)
    print("Testing short random rollouts...")
    test_short_rollouts()
    print("Testing HeuristicEnemySMAX wrapper...")
    test_heuristic_enemy_wrapper()
    print("Testing medivac mechanics...")
    test_medivac_mechanics()
    print("Testing baneling mechanics...")
    test_baneling_mechanics()
    print("All SMAX SMACv2 scenario tests passed.")


if __name__ == "__main__":
    main()
