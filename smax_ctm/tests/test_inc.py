import os
import sys

import pytest
from flax import traverse_util

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_SMAX_CTM_DIR = os.path.join(_REPO_ROOT, "smax_ctm")
if _SMAX_CTM_DIR not in sys.path:
    sys.path.insert(0, _SMAX_CTM_DIR)

from smax_ctm.ctm_jax import AgentConsensus, CTMCell
from smax_ctm.train_mappo_ctm import shuffle_and_split_actor_batch_env_grouped


def _build_ctm_cell(**kwargs):
    base = dict(
        d_model=8,
        d_input=4,
        memory_length=3,
        n_synch_out=3,
        iterations=3,
        deep_nlms=False,
        memory_hidden_dims=2,
        obs_dim=5,
        use_sync=True,
        neuron_select_type="first-last",
        do_layernorm_nlm=False,
        num_agents=3,
        inc_enabled=True,
        inc_pooling="mean",
        inc_consensus_dropout=0.0,
        inc_use_alive_mask_from_dones=True,
    )
    base.update(kwargs)
    return CTMCell(**base)


def test_axis_round_trip_agent_major_ordering():
    num_agents = 3
    num_envs = 4
    synch_size = 2
    num_actors = num_agents * num_envs

    # Row i carries its agent id i // num_envs.
    row_agent_id = (jnp.arange(num_actors) // num_envs).astype(jnp.float32)
    flat = jnp.repeat(row_agent_id[:, None], repeats=synch_size, axis=1)

    reshaped = flat.reshape(num_agents, num_envs, synch_size)
    round_trip = reshaped.reshape(num_actors, synch_size)

    assert jnp.array_equal(round_trip, flat)
    assert jnp.array_equal(reshaped[:, 0, 0], jnp.array([0.0, 1.0, 2.0]))


def test_agent_consensus_leave_one_out_mean_all_alive():
    # Agent-major input: (A=3, E=1, S=1)
    sync = jnp.array([[[1.0]], [[2.0]], [[3.0]]])
    alive_mask = jnp.array([[True, True, True]])

    module = AgentConsensus(pooling="mean", dropout_rate=0.0)
    variables = module.init(jax.random.PRNGKey(0), sync, alive_mask=alive_mask, deterministic=True)
    out = module.apply(variables, sync, alive_mask=alive_mask, deterministic=True)

    expected = jnp.array([[[2.5]], [[2.0]], [[1.5]]])
    assert jnp.allclose(out, expected)


def test_agent_consensus_mean_dead_mask():
    # Agent-major input: (A=3, E=1, S=1)
    sync = jnp.array([[[1.0]], [[2.0]], [[3.0]]])

    # env-major alive mask: (E=1, A=3), agent 1 is dead
    alive_mask = jnp.array([[True, False, True]])

    module = AgentConsensus(pooling="mean", dropout_rate=0.0)
    variables = module.init(jax.random.PRNGKey(0), sync, alive_mask=alive_mask, deterministic=True)
    out = module.apply(variables, sync, alive_mask=alive_mask, deterministic=True)

    # For alive agents, pool over alive others only (leave-one-out):
    # agent0 <- agent2 => 3.0
    # agent2 <- agent0 => 1.0
    # dead target agent1 => 0.0
    expected = jnp.array([[[3.0]], [[0.0]], [[1.0]]])
    assert jnp.allclose(out, expected)


def test_no_op_proxy_inc_disabled_matches_single_iteration_enabled():
    # For iterations=1, consensus path is never consumed, so INC on/off must match.
    batch_size = 6  # 3 agents x 2 envs
    obs = jnp.arange(batch_size * 5, dtype=jnp.float32).reshape(batch_size, 5) / 100.0
    dones = jnp.zeros((batch_size,), dtype=bool)
    avail_actions = jnp.ones((batch_size, 4), dtype=jnp.float32)
    carry = CTMCell.initialize_carry(batch_size, d_model=8, memory_length=3)

    cell_off = _build_ctm_cell(iterations=1, inc_enabled=False)
    cell_on = _build_ctm_cell(iterations=1, inc_enabled=True)

    key = jax.random.PRNGKey(7)
    vars_off = cell_off.init(key, carry, (obs, dones, avail_actions))
    vars_on = cell_on.init(key, carry, (obs, dones, avail_actions))

    _, synch_off = cell_off.apply(vars_off, carry, (obs, dones, avail_actions))
    _, synch_on = cell_on.apply(vars_on, carry, (obs, dones, avail_actions))
    assert jnp.allclose(synch_off, synch_on, atol=0.0, rtol=0.0)


def test_gradient_flow_through_consensus_and_synapses():
    batch_size = 6  # 3 agents x 2 envs
    obs = jnp.arange(batch_size * 5, dtype=jnp.float32).reshape(batch_size, 5) / 50.0
    dones = jnp.zeros((batch_size,), dtype=bool)
    avail_actions = jnp.ones((batch_size, 4), dtype=jnp.float32)
    carry = CTMCell.initialize_carry(batch_size, d_model=8, memory_length=3)

    cell = _build_ctm_cell(iterations=3, inc_enabled=True, inc_pooling="gated")
    variables = cell.init(jax.random.PRNGKey(11), carry, (obs, dones, avail_actions))

    def loss_fn(params):
        _, synch = cell.apply({"params": params}, carry, (obs, dones, avail_actions))
        return jnp.sum(synch)

    grads = jax.grad(loss_fn)(variables["params"])
    flat = traverse_util.flatten_dict(grads, sep="/")

    consensus_keys = [k for k in flat if k.startswith("consensus/")]
    assert consensus_keys, "Expected consensus parameters for gated pooling."
    assert any(jnp.any(jnp.abs(flat[k]) > 0) for k in consensus_keys)

    syn_key = "synapses/Dense_0/kernel"
    assert syn_key in flat
    assert jnp.any(jnp.abs(flat[syn_key]) > 0)


def test_env_grouped_minibatch_split_preserves_env_pairs():
    num_agents = 2
    num_envs = 4
    num_minibatches = 2

    # Build a fake actor-axis tensor with agent-major layout.
    # Agent 0 rows: env ids [0,1,2,3], agent 1 rows: [0,1,2,3]
    # Shape: (T=1, B=8, F=1)
    agent0 = jnp.arange(num_envs)
    agent1 = jnp.arange(num_envs)
    flat_actor_ids = jnp.concatenate([agent0, agent1], axis=0)
    x = flat_actor_ids[None, :, None]

    # Permute envs only.
    env_perm = jnp.array([2, 0, 3, 1])

    minibatches = shuffle_and_split_actor_batch_env_grouped(
        (x,),
        env_permutation=env_perm,
        num_agents=num_agents,
        num_envs=num_envs,
        num_minibatches=num_minibatches,
    )[0]

    # minibatches shape: (M=2, T=1, B_mb=4, F=1)
    assert minibatches.shape == (2, 1, 4, 1)

    # Verify each minibatch still contains complete env groups for both agents.
    envs_per_mb = num_envs // num_minibatches
    for mb_idx in range(num_minibatches):
        mb_flat = minibatches[mb_idx, 0, :, 0]
        grouped = mb_flat.reshape(num_agents, envs_per_mb)
        assert jnp.array_equal(grouped[0], grouped[1])


def test_minibatch_permutation_safety_with_agent_major_consensus():
    num_agents = 3
    num_envs = 4
    synch_size = 2

    flat = jnp.arange(num_agents * num_envs * synch_size, dtype=jnp.float32).reshape(
        num_agents * num_envs, synch_size
    )

    module = AgentConsensus(pooling="mean", dropout_rate=0.0)
    alive = jnp.ones((num_envs, num_agents), dtype=bool)
    params = module.init(jax.random.PRNGKey(0), flat.reshape(num_agents, num_envs, synch_size), alive_mask=alive, deterministic=True)

    def pooled_from_flat(flat_sync):
        agent_major = flat_sync.reshape(num_agents, num_envs, synch_size)
        pooled = module.apply(params, agent_major, alive_mask=alive, deterministic=True)
        return pooled.reshape(num_agents * num_envs, synch_size)

    base = pooled_from_flat(flat)

    env_perm = jnp.array([2, 0, 3, 1])
    flat_perm = jnp.concatenate([env_perm + a * num_envs for a in range(num_agents)], axis=0)
    inv_perm = jnp.argsort(flat_perm)

    permuted = flat[flat_perm]
    pooled_permuted = pooled_from_flat(permuted)
    pooled_back = pooled_permuted[inv_perm]

    assert jnp.allclose(base, pooled_back)
