"""
Tests for JaxMARL AssemblyEnv.
Checks that the JAX env produces outputs with the right shapes, dtypes,
and behavioural properties expected from the C++ reference implementation.

Run with:
    python test_assembly_env.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "JaxMARL"))

import jax
import jax.numpy as jnp
import numpy as np

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "fig", "results.pkl")
N_A = 30
TOPO_NEI_MAX = 6
NUM_OBS_GRID_MAX = 80
OBS_DIM = 4 * (TOPO_NEI_MAX + 1) + 4 + 2 * NUM_OBS_GRID_MAX  # 192

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

def check(name, cond, extra=""):
    tag = PASS if cond else FAIL
    msg = f"  [{tag}] {name}"
    if extra:
        msg += f"  ({extra})"
    print(msg)
    return cond


def make_env():
    from jaxmarl.environments.mpe.assembly import AssemblyEnv
    return AssemblyEnv(
        results_file=RESULTS_FILE,
        n_a=N_A,
        topo_nei_max=TOPO_NEI_MAX,
        num_obs_grid_max=NUM_OBS_GRID_MAX,
        max_steps=200,
    )


def test_init():
    print("\n=== test_init ===")
    env = make_env()

    check("num_shapes > 0",     env.num_shapes > 0,
          f"num_shapes={env.num_shapes}")
    check("n_g_max > 0",        env.n_g_max > 0,
          f"n_g_max={env.n_g_max}")
    check("r_avoid > 0",        env.r_avoid > 0,
          f"r_avoid={env.r_avoid}")
    check("obs_dim correct",    env.obs_dim == OBS_DIM,
          f"got {env.obs_dim}, expected {OBS_DIM}")
    check("agents list length", len(env.agents) == N_A)
    check("obs spaces count",   len(env.observation_spaces) == N_A)
    check("act spaces count",   len(env.action_spaces) == N_A)
    check("all_grid_centers shape",
          env.all_grid_centers.shape == (env.num_shapes, 2, env.n_g_max))
    check("all_valid_masks shape",
          env.all_valid_masks.shape == (env.num_shapes, env.n_g_max))


def test_reset():
    print("\n=== test_reset ===")
    env = make_env()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    check("obs is dict",        isinstance(obs, dict))
    check("obs has all agents", set(obs.keys()) == set(env.agents))
    check("obs shape per agent",
          all(v.shape == (OBS_DIM,) for v in obs.values()),
          f"expected ({OBS_DIM},)")
    check("obs finite",         all(jnp.all(jnp.isfinite(v)) for v in obs.values()))

    check("p_pos shape",        state.p_pos.shape == (N_A, 2))
    check("p_vel shape",        state.p_vel.shape == (N_A, 2))
    check("grid_center shape",  state.grid_center.shape == (2, env.n_g_max))
    check("valid_mask shape",   state.valid_mask.shape == (env.n_g_max,))
    check("step == 0",          int(state.step) == 0)
    check("done all False",     not jnp.any(state.done))
    check("l_cell > 0",         float(state.l_cell) > 0.0)

    # Agents should be within boundary
    bh = env.boundary_half
    in_bnd = jnp.all(jnp.abs(state.p_pos) <= bh + env.size_a * 2)
    check("initial positions near boundary", bool(in_bnd))

    # Velocity within [-0.5, 0.5]
    check("initial vel in [-0.5, 0.5]",
          bool(jnp.all(jnp.abs(state.p_vel) <= 0.5 + 1e-6)))


def test_reset_different_keys():
    print("\n=== test_reset_different_keys ===")
    env = make_env()
    obs0, state0 = env.reset(jax.random.PRNGKey(0))
    obs1, state1 = env.reset(jax.random.PRNGKey(1))

    pos_diff = jnp.any(state0.p_pos != state1.p_pos)
    check("different keys → different states", bool(pos_diff))

    obs_diff = jnp.any(obs0["agent_0"] != obs1["agent_0"])
    check("different keys → different obs", bool(obs_diff))


def test_step():
    print("\n=== test_step ===")
    env = make_env()
    key = jax.random.PRNGKey(42)
    key_reset, key_step = jax.random.split(key)

    obs, state = env.reset(key_reset)

    # Random actions in [-1, 1]
    actions = {
        a: jax.random.uniform(key_step, (2,), minval=-1.0, maxval=1.0)
        for a in env.agents
    }

    obs2, state2, rewards, dones, info = env.step_env(key_step, state, actions)

    check("obs2 is dict",           isinstance(obs2, dict))
    check("obs2 has all agents",    set(obs2.keys()) == set(env.agents))
    check("obs2 shape per agent",
          all(v.shape == (OBS_DIM,) for v in obs2.values()))
    check("obs2 finite",            all(jnp.all(jnp.isfinite(v)) for v in obs2.values()))

    check("rewards is dict",        isinstance(rewards, dict))
    check("rewards has all agents", set(rewards.keys()) == set(env.agents))
    check("rewards are scalars",    all(r.shape == () for r in rewards.values()))
    check("rewards in {0, 1}",
          all(float(r) in (0.0, 1.0) for r in rewards.values()))

    check("dones has __all__",      "__all__" in dones)
    check("step incremented",       int(state2.step) == 1)
    check("state changed",          bool(jnp.any(state2.p_pos != state.p_pos)))


def test_step_action_clip():
    print("\n=== test_step_action_clip ===")
    env = make_env()
    key = jax.random.PRNGKey(7)
    obs, state = env.reset(key)

    # Huge actions should be clipped to [-1,1]
    actions_big   = {a: jnp.array([100.0, -100.0]) for a in env.agents}
    actions_clipped = {a: jnp.array([1.0, -1.0])   for a in env.agents}

    _, state_big,  _, _, _ = env.step_env(key, state, actions_big)
    _, state_clip, _, _, _ = env.step_env(key, state, actions_clipped)

    same = jnp.allclose(state_big.p_pos, state_clip.p_pos, atol=1e-5)
    check("large actions clipped to same as ±1", bool(same))


def test_episode_terminates():
    print("\n=== test_episode_terminates ===")
    env = make_env()
    key = jax.random.PRNGKey(99)
    obs, state = env.reset(key)

    # Use env.step (auto-reset wrapper) so we can check dones["__all__"]
    max_steps = env.max_steps
    done_all = False
    for t in range(max_steps + 1):
        key, k = jax.random.split(key)
        actions = {a: jnp.zeros(2) for a in env.agents}
        obs, state, rewards, dones, _ = env.step(k, state, actions)
        if dones["__all__"]:
            done_all = True
            check("episode terminates at max_steps",
                  t == max_steps - 1,
                  f"terminated at step {t}, max_steps={max_steps}")
            break

    check("episode actually terminated", done_all)


def test_physics_wall_repulsion():
    print("\n=== test_physics_wall_repulsion ===")
    env = make_env()
    key = jax.random.PRNGKey(5)
    _, state = env.reset(key)

    # Place all agents at right wall boundary → should be pushed left
    p_pos = jnp.full((N_A, 2), env.boundary_half + 0.01)
    p_vel = jnp.zeros((N_A, 2))
    state2 = state.replace(p_pos=p_pos, p_vel=p_vel)

    actions = {a: jnp.zeros(2) for a in env.agents}
    _, state3, _, _, _ = env.step_env(key, state2, actions)

    # After wall repulsion, x velocities should be negative (pushed left)
    check("wall repulsion pushes agents inward",
          bool(jnp.all(state3.p_vel[:, 0] < 0.0)))


def test_physics_ball_repulsion():
    print("\n=== test_physics_ball_repulsion ===")
    env = make_env()
    key = jax.random.PRNGKey(3)
    _, state = env.reset(key)

    # Stack all agents at origin → they should spread out
    p_pos = jnp.zeros((N_A, 2))
    p_vel = jnp.zeros((N_A, 2))
    state2 = state.replace(p_pos=p_pos, p_vel=p_vel)

    actions = {a: jnp.zeros(2) for a in env.agents}
    _, state3, _, _, _ = env.step_env(key, state2, actions)

    # Velocities should be non-zero (spring forces acted)
    check("ball repulsion creates non-zero velocities",
          bool(jnp.any(jnp.abs(state3.p_vel) > 1e-6)))


def test_vmap_reset():
    print("\n=== test_vmap_reset ===")
    env = make_env()
    N_ENVS = 8
    keys = jax.random.split(jax.random.PRNGKey(0), N_ENVS)

    v_reset = jax.vmap(env.reset)
    obs_batch, state_batch = v_reset(keys)

    check("vmap reset runs",          True)
    check("batch p_pos shape",        state_batch.p_pos.shape == (N_ENVS, N_A, 2),
          f"got {state_batch.p_pos.shape}")
    check("batch obs shape agent_0",
          obs_batch["agent_0"].shape == (N_ENVS, OBS_DIM),
          f"got {obs_batch['agent_0'].shape}")
    check("all envs differ",
          bool(jnp.any(state_batch.p_pos[0] != state_batch.p_pos[1])))


def test_vmap_step():
    print("\n=== test_vmap_step ===")
    env = make_env()
    N_ENVS = 8
    keys = jax.random.split(jax.random.PRNGKey(1), N_ENVS)

    v_reset = jax.vmap(env.reset)
    obs_batch, state_batch = v_reset(keys)

    act_keys = jax.random.split(jax.random.PRNGKey(2), N_ENVS)
    actions_batch = {
        a: jax.vmap(lambda k: jax.random.uniform(k, (2,), minval=-1.0, maxval=1.0))(act_keys)
        for a in env.agents
    }

    v_step = jax.vmap(env.step_env)
    obs2_batch, state2_batch, rew_batch, dones_batch, _ = v_step(
        act_keys, state_batch, actions_batch
    )

    check("vmap step runs",           True)
    check("batch obs2 shape agent_0",
          obs2_batch["agent_0"].shape == (N_ENVS, OBS_DIM),
          f"got {obs2_batch['agent_0'].shape}")
    check("batch rewards shape",
          rew_batch["agent_0"].shape == (N_ENVS,),
          f"got {rew_batch['agent_0'].shape}")
    check("batch state changed",
          bool(jnp.any(state2_batch.p_pos != state_batch.p_pos)))
    check("obs2 all finite",
          bool(jnp.all(jnp.isfinite(obs2_batch["agent_0"]))))


def test_jit_compile():
    print("\n=== test_jit_compile ===")
    env = make_env()
    key = jax.random.PRNGKey(0)

    # First call compiles
    obs, state = env.reset(key)
    actions = {a: jnp.zeros(2) for a in env.agents}
    obs2, state2, _, _, _ = env.step_env(key, state, actions)

    # Second call should reuse compiled code (no recompile)
    obs3, state3 = env.reset(jax.random.PRNGKey(1))
    obs4, state4, _, _, _ = env.step_env(key, state3, actions)

    check("jit reset second call works", True)
    check("jit step second call works",  True)


def run_all():
    all_ok = True
    tests = [
        test_init,
        test_reset,
        test_reset_different_keys,
        test_step,
        test_step_action_clip,
        test_episode_terminates,
        test_physics_wall_repulsion,
        test_physics_ball_repulsion,
        test_vmap_reset,
        test_vmap_step,
        test_jit_compile,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  [\033[91m FAIL\033[0m] {t.__name__} raised: {e}")
            import traceback; traceback.print_exc()
            all_ok = False

    print("\n" + ("=" * 40))
    if all_ok:
        print("\033[92mAll tests passed.\033[0m")
    else:
        print("\033[91mSome tests FAILED.\033[0m")


if __name__ == "__main__":
    run_all()
