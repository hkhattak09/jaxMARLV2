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
    "protoss_10_vs_11": {
        "n_allies": 10,
        "n_enemies": 11,
        "allowed": {2, 3, 6},
        "weights": {2: 0.45, 3: 0.45, 6: 0.10},
        "exceptions": set(),
    },
    "protoss_20_vs_20": {
        "n_allies": 20,
        "n_enemies": 20,
        "allowed": {2, 3, 6},
        "weights": {2: 0.45, 3: 0.45, 6: 0.10},
        "exceptions": set(),
    },
    "protoss_20_vs_23": {
        "n_allies": 20,
        "n_enemies": 23,
        "allowed": {2, 3, 6},
        "weights": {2: 0.45, 3: 0.45, 6: 0.10},
        "exceptions": set(),
    },
    "terran_10_vs_11": {
        "n_allies": 10,
        "n_enemies": 11,
        "allowed": {0, 1, 7},
        "weights": {0: 0.45, 1: 0.45, 7: 0.10},
        "exceptions": {7},
    },
    "terran_20_vs_20": {
        "n_allies": 20,
        "n_enemies": 20,
        "allowed": {0, 1, 7},
        "weights": {0: 0.45, 1: 0.45, 7: 0.10},
        "exceptions": {7},
    },
    "terran_20_vs_23": {
        "n_allies": 20,
        "n_enemies": 23,
        "allowed": {0, 1, 7},
        "weights": {0: 0.45, 1: 0.45, 7: 0.10},
        "exceptions": {7},
    },
    "zerg_10_vs_11": {
        "n_allies": 10,
        "n_enemies": 11,
        "allowed": {4, 5, 8},
        "weights": {4: 0.45, 5: 0.45, 8: 0.10},
        "exceptions": {8},
    },
    "zerg_20_vs_20": {
        "n_allies": 20,
        "n_enemies": 20,
        "allowed": {4, 5, 8},
        "weights": {4: 0.45, 5: 0.45, 8: 0.10},
        "exceptions": {8},
    },
    "zerg_20_vs_23": {
        "n_allies": 20,
        "n_enemies": 23,
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
        assert_true(
            obs["world_state"].shape == (env.state_size,),
            f"{scenario_name}: bad world state shape "
            f"{obs['world_state'].shape}, expected {(env.state_size,)}",
        )
        for agent in env.agents:
            assert_true(
                obs[agent].shape == (env.obs_size,),
                f"{scenario_name}: bad obs shape for {agent}: "
                f"{obs[agent].shape}, expected {(env.obs_size,)}",
            )

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


def test_smacv2_parity_mode():
    """Validate SMACv2 parity flags for race scenarios."""
    env = SMAX(
        scenario=map_name_to_scenario("protoss_10_vs_10"),
        max_steps=200,
        smacv2_unit_stats=True,
        smacv2_position_parity=True,
        reward_mode="smacv2",
        movement_mode="smacv2",
    )
    assert_true(env.max_steps == 200, "max_steps should be 200")
    assert_true(env.num_movement_actions == 6, "SMACv2 movement mode should have 6 movement actions")
    expected_actions = 6 + max(env.num_allies, env.num_enemies)
    assert_true(
        env.action_spaces["ally_0"].n == expected_actions,
        f"SMACv2 action space should be {expected_actions}, got {env.action_spaces['ally_0'].n}",
    )

    # Unit stats
    colossus_idx = 6
    baneling_idx = 8
    assert_true(
        float(env.unit_type_health[colossus_idx]) == 350.0,
        f"Colossus health should be 350, got {env.unit_type_health[colossus_idx]}",
    )
    assert_true(
        float(env.unit_type_attack_ranges[baneling_idx]) == 2.0,
        f"Baneling range should be 2.0, got {env.unit_type_attack_ranges[baneling_idx]}",
    )

    # Position parity: surround mode inside team at exact center
    # We test via multiple resets and verify inside team has exact center positions
    from jaxmarl.environments.smax.distributions import SMACv2SurroundPositionDistribution
    dist = SMACv2SurroundPositionDistribution(10, 10, 32, 32)
    for seed in range(20):
        pos = dist.generate(jax.random.PRNGKey(seed))
        # ally_inside = True means allies are inside (first 10 units)
        # The generate method randomly decides ally_inside, so we check whichever half is inside
        # Inside units are at positions [0..n_inside-1] or [n_outside..n_outside+n_inside-1]
        # For the SurroundAndReflect wrapper, allies are first 10, enemies next 10.
        # In draw_positions: ally_inside_positions = concat(inside, outside)
        # enemy_inside_positions = concat(outside, inside)
        # Then select based on ally_inside.
        # Since we're testing the distribution directly, we can't easily know which is inside.
        # Instead, verify that some units are exactly at (16, 16).
        center = jnp.array([16.0, 16.0])
        at_center = jnp.all(jnp.isclose(pos, center), axis=-1)
        assert_true(
            jnp.any(at_center),
            f"SMACv2 surround: no unit at exact center for seed {seed}",
        )

    # Avail actions in SMACv2 movement mode
    key = jax.random.PRNGKey(50_000)
    _, state = env.reset(key)
    avail = env.get_avail_actions(state)
    for agent in env.agents:
        # All units alive at reset -> no-op (0) should be unavailable
        assert_true(
            int(avail[agent][0]) == 0,
            f"Alive agent {agent} should not have no-op available in SMACv2 mode",
        )
        # Stop (1) and moves (2..5) should be available
        assert_true(
            int(jnp.sum(avail[agent][1:6])) == 5,
            f"Alive agent {agent} should have stop+moves available in SMACv2 mode",
        )

    # Kill one ally and verify only no-op is available
    dead_health = jnp.copy(state.unit_health)
    dead_health = dead_health.at[0].set(0.0)
    dead_alive = jnp.copy(state.unit_alive)
    dead_alive = dead_alive.at[0].set(False)
    dead_state = state.replace(unit_health=dead_health, unit_alive=dead_alive)
    dead_avail = env.get_avail_actions(dead_state)
    assert_true(
        int(dead_avail["ally_0"][0]) == 1,
        "Dead agent should only have no-op available in SMACv2 mode",
    )
    assert_true(
        int(jnp.sum(dead_avail["ally_0"][1:])) == 0,
        "Dead agent should have no other actions available in SMACv2 mode",
    )

    # SMACv2 reward mode: dummy rollout should produce finite rewards
    key = jax.random.PRNGKey(60_000)
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    for _ in range(4):
        key, action_key, step_key = jax.random.split(key, 3)
        action_key, actions = random_valid_actions(action_key, env, state)
        obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)
        for agent in env.agents:
            assert_true(
                jnp.isfinite(rewards[agent]),
                f"SMACv2 reward should be finite for {agent}",
            )

    # Reward scale should follow SMAC's max_reward / reward_scale_rate:
    # enemy max HP + death rewards + win reward, divided by 20.
    _, state = env.reset(jax.random.PRNGKey(61_000))
    health_before = state.unit_health
    damage = 10.0
    target_idx = env.num_allies
    health_after = health_before.at[target_idx].add(-damage)
    rewards = env.compute_reward(state, health_before, health_after)
    enemy_types = state.unit_types[env.num_allies :]
    max_reward = jnp.sum(env.unit_type_health[enemy_types]) + env.num_enemies * 10.0 + 200.0
    expected_reward = damage / (max_reward / 20.0)
    assert_true(
        jnp.isclose(rewards["ally_0"], expected_reward),
        f"SMACv2 reward scaling mismatch: got {rewards['ally_0']}, expected {expected_reward}",
    )


def test_smacv2_parity_rollout_with_heuristic():
    """SMACv2 parity with heuristic enemy should compile and run."""
    env = HeuristicEnemySMAX(
        scenario=map_name_to_scenario("protoss_10_vs_10"),
        enemy_shoots=True,
        max_steps=8,
        smacv2_unit_stats=True,
        smacv2_position_parity=True,
        reward_mode="smacv2",
        movement_mode="smacv2",
    )
    key = jax.random.PRNGKey(70_000)
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
    print("Testing SMACv2 parity mode...")
    test_smacv2_parity_mode()
    print("Testing SMACv2 parity rollout with heuristic enemy...")
    test_smacv2_parity_rollout_with_heuristic()
    print("All SMAX SMACv2 scenario tests passed.")


if __name__ == "__main__":
    main()
