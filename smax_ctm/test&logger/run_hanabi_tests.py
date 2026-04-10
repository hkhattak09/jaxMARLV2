import os
import sys

# Inject repo root into sys.path so modules are always found regardless of CWD.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import traceback

import jax
import jax.numpy as jnp

from jaxmarl import make


print("=== Running Hanabi JAX 0.7.2 Environment Tests ===")


def assert_dict_has_keys(name, dct, keys):
    for k in keys:
        assert k in dct, f"{name} missing key: {k}"


def _agent_names(num_agents):
    return [f"agent_{i}" for i in range(num_agents)]


def _pick_legal_action_for_player(legal_moves_for_player):
    legal_idx = jnp.where(legal_moves_for_player > 0)[0]
    assert legal_idx.size > 0, "No legal action found for current player"
    return int(legal_idx[0])


def _build_action_dict(env, state, legal_moves):
    cur_player = int(jnp.where(state.cur_player_idx == 1)[0][0])
    noop = env.num_actions - 1
    actions = jnp.full((env.num_agents,), noop, dtype=jnp.int32)

    chosen = _pick_legal_action_for_player(legal_moves[env.agents[cur_player]])
    actions = actions.at[cur_player].set(chosen)

    return {agent: actions[i] for i, agent in enumerate(env.agents)}, cur_player, chosen


print("\n1. Testing make('hanabi') and metadata...")
env = make("hanabi")
assert env.name.lower() == "hanabi", f"Unexpected env name: {env.name}"
assert env.num_agents == 2, f"Expected default num_agents=2, got {env.num_agents}"
assert env.num_actions == 21, f"Expected default num_actions=21, got {env.num_actions}"
assert len(env.agents) == 2, f"Expected 2 agents, got {len(env.agents)}"
print("  Environment creation and metadata checks passed")


print("\n2. Testing reset() output contract...")
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)
assert isinstance(obs, dict), "reset() observations must be a dict"
assert_dict_has_keys("obs", obs, env.agents)

for a in env.agents:
    assert obs[a].ndim == 1, f"Observation for {a} must be 1D, got shape={obs[a].shape}"
    expected_shape = env.observation_space(a).shape
    assert obs[a].shape == expected_shape, (
        f"Observation shape mismatch for {a}: expected {expected_shape}, got {obs[a].shape}"
    )

assert state.cur_player_idx.shape == (env.num_agents,), (
    f"cur_player_idx shape mismatch: {state.cur_player_idx.shape}"
)
assert int(jnp.sum(state.cur_player_idx)) == 1, "Exactly one current player expected"
print("  reset() contract checks passed")


print("\n3. Testing get_legal_moves() contract...")
legal = env.get_legal_moves(state)
assert isinstance(legal, dict), "get_legal_moves() must return a dict"
assert_dict_has_keys("legal_moves", legal, env.agents)

for a in env.agents:
    lm = legal[a]
    assert lm.shape == (env.num_actions,), (
        f"Legal move shape mismatch for {a}: expected {(env.num_actions,)}, got {lm.shape}"
    )
    assert jnp.logical_or(lm == 0, lm == 1).all(), (
        f"Legal move vector for {a} must be binary, got values: {jnp.unique(lm)}"
    )

print("  Legal move checks passed")


print("\n4. Testing one valid environment step...")
actions, cur_player, chosen = _build_action_dict(env, state, legal)
obs2, state2, rewards, dones, infos = env.step(key, state, actions)

assert_dict_has_keys("next_obs", obs2, env.agents)
assert_dict_has_keys("rewards", rewards, env.agents + ["__all__"])
assert_dict_has_keys("dones", dones, env.agents + ["__all__"])
assert isinstance(infos, dict), "infos must be a dict"

for a in env.agents:
    assert obs2[a].shape == env.observation_space(a).shape, (
        f"Step observation shape mismatch for {a}: {obs2[a].shape}"
    )

assert int(jnp.sum(state2.cur_player_idx)) == 1, "Exactly one current player expected after step"
assert state2.turn == state.turn + 1, f"Turn must increment by 1: {state.turn} -> {state2.turn}"
print(f"  Step contract checks passed (current_player={cur_player}, action={chosen})")


print("\n5. Testing short rollout stability (50 steps max)...")
roll_key = jax.random.PRNGKey(123)
roll_obs, roll_state = env.reset(roll_key)
max_steps = 50

for t in range(max_steps):
    roll_key, step_key = jax.random.split(roll_key)
    legal_t = env.get_legal_moves(roll_state)
    action_dict, _, _ = _build_action_dict(env, roll_state, legal_t)
    roll_obs, roll_state, roll_rewards, roll_dones, _ = env.step(step_key, roll_state, action_dict)

    for a in env.agents:
        assert not jnp.any(jnp.isnan(roll_obs[a])), f"NaN in observation for {a} at step {t}"
        assert not jnp.isinf(roll_rewards[a]), f"Inf reward for {a} at step {t}"

    if bool(roll_dones["__all__"]):
        break

print("  Short rollout stability checks passed")


print("\n6. Testing custom num_agents setting (3 players)...")
env3 = make("hanabi", num_agents=3)
obs3, state3 = env3.reset(jax.random.PRNGKey(9))
assert len(env3.agents) == 3, f"Expected 3 agents, got {len(env3.agents)}"
assert env3.num_actions == 31, f"Expected 31 actions for 3 players, got {env3.num_actions}"
assert_dict_has_keys("obs3", obs3, _agent_names(3))
assert state3.cur_player_idx.shape == (3,), f"cur_player_idx shape mismatch for 3p: {state3.cur_player_idx.shape}"
print("  3-player configuration checks passed")


print("\n7. Testing fail-loud unsupported env id...")
try:
    _ = make("hanabi_nonexistent")
    raise AssertionError("Expected make() to raise for unknown env id")
except ValueError:
    pass
print("  Unknown env id fail-loud check passed")


print("\n=== All Hanabi Tests Complete ===")


if __name__ == "__main__":
    try:
        pass
    except Exception as exc:
        print(f"\nERROR: {exc}")
        traceback.print_exc()
        raise
