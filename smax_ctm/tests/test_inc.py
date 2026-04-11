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


def test_agent_consensus_all_dead_gradient_is_finite():
    # Regression: on the first rollout step train_mappo_ctm.py seeds last_done=ones
    # to force the CTM reset path, which makes alive_mask all-False. The naive
    # jnp.where(denom>0, numer/denom, 0) gives NaN gradients through the dead
    # branch even though the forward pass looks fine. Safe-divide must fix this.
    num_agents = 3
    num_envs = 2
    synch_size = 4

    sync = jnp.arange(num_agents * num_envs * synch_size, dtype=jnp.float32).reshape(
        num_agents, num_envs, synch_size
    ) / 10.0
    alive_mask = jnp.zeros((num_envs, num_agents), dtype=bool)  # everyone dead

    for pooling in ("mean", "attention", "gated"):
        module = AgentConsensus(pooling=pooling, dropout_rate=0.0)
        variables = module.init(
            jax.random.PRNGKey(0), sync, alive_mask=alive_mask, deterministic=True
        )

        def loss_fn(s):
            out = module.apply(variables, s, alive_mask=alive_mask, deterministic=True)
            return jnp.sum(out)

        grad = jax.grad(loss_fn)(sync)
        assert jnp.all(jnp.isfinite(grad)), (
            f"all-dead mask produced non-finite gradient for pooling={pooling}"
        )


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


# ---------------------------------------------------------------------------
# Stage 2.1 disambiguation tests.
# ---------------------------------------------------------------------------


def test_stage2_1_defaults_are_noop_vs_prerefactor():
    # With both Stage 2.1 flags at their defaults, forward pass must be
    # bit-identical to a cell that doesn't know about them at all.
    batch_size = 6  # 3 agents x 2 envs
    obs = jnp.arange(batch_size * 5, dtype=jnp.float32).reshape(batch_size, 5) / 100.0
    dones = jnp.zeros((batch_size,), dtype=bool)
    avail_actions = jnp.ones((batch_size, 4), dtype=jnp.float32)
    carry = CTMCell.initialize_carry(batch_size, d_model=8, memory_length=3)

    cell_base = _build_ctm_cell(iterations=3, inc_enabled=True)
    cell_defaulted = _build_ctm_cell(
        iterations=3,
        inc_enabled=True,
        ctm_iter_dropout=0.0,
        inc_force_zero_consensus=False,
    )

    key = jax.random.PRNGKey(13)
    vars_base = cell_base.init(key, carry, (obs, dones, avail_actions))
    vars_def = cell_defaulted.init(key, carry, (obs, dones, avail_actions))

    _, s_base = cell_base.apply(vars_base, carry, (obs, dones, avail_actions))
    _, s_def = cell_defaulted.apply(vars_def, carry, (obs, dones, avail_actions))
    assert jnp.allclose(s_base, s_def, atol=0.0, rtol=0.0)


def test_stage2_1_force_zero_consensus_zeroes_pooled_before_dropout():
    # AgentConsensus with force_zero_output=True must return exactly zero
    # (dropout of zero is zero regardless of mask).
    sync = jnp.array([[[1.0]], [[2.0]], [[3.0]]])
    alive_mask = jnp.array([[True, True, True]])

    module = AgentConsensus(pooling="mean", dropout_rate=0.25, force_zero_output=True)
    variables = module.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        sync,
        alive_mask=alive_mask,
        deterministic=False,
    )
    out = module.apply(
        variables,
        sync,
        alive_mask=alive_mask,
        deterministic=False,
        rngs={"dropout": jax.random.PRNGKey(2)},
    )
    assert jnp.all(out == 0.0)


def test_stage2_1_ctm_iter_dropout_deterministic_is_noop():
    # CTM_ITER_DROPOUT > 0 but deterministic=True must match the
    # CTM_ITER_DROPOUT=0 path exactly (dropout scales to identity in eval).
    batch_size = 6
    obs = jnp.arange(batch_size * 5, dtype=jnp.float32).reshape(batch_size, 5) / 100.0
    dones = jnp.zeros((batch_size,), dtype=bool)
    avail_actions = jnp.ones((batch_size, 4), dtype=jnp.float32)
    carry = CTMCell.initialize_carry(batch_size, d_model=8, memory_length=3)

    cell_off = _build_ctm_cell(iterations=3, inc_enabled=False, ctm_iter_dropout=0.0, deterministic=True)
    cell_on = _build_ctm_cell(iterations=3, inc_enabled=False, ctm_iter_dropout=0.25, deterministic=True)

    key = jax.random.PRNGKey(23)
    v_off = cell_off.init(key, carry, (obs, dones, avail_actions))
    v_on = cell_on.init(key, carry, (obs, dones, avail_actions))

    _, s_off = cell_off.apply(v_off, carry, (obs, dones, avail_actions))
    _, s_on = cell_on.apply(v_on, carry, (obs, dones, avail_actions))
    assert jnp.allclose(s_off, s_on, atol=0.0, rtol=0.0)


def test_stage2_1_ctm_iter_dropout_train_changes_with_rng():
    # Train-time: different dropout RNGs must produce different synch outputs.
    batch_size = 6
    obs = jnp.arange(batch_size * 5, dtype=jnp.float32).reshape(batch_size, 5) / 100.0
    dones = jnp.zeros((batch_size,), dtype=bool)
    avail_actions = jnp.ones((batch_size, 4), dtype=jnp.float32)
    carry = CTMCell.initialize_carry(batch_size, d_model=8, memory_length=3)

    cell = _build_ctm_cell(
        iterations=3, inc_enabled=False, ctm_iter_dropout=0.5, deterministic=False
    )
    init_key = jax.random.PRNGKey(31)
    variables = cell.init(
        {"params": init_key, "dropout": jax.random.PRNGKey(32)},
        carry,
        (obs, dones, avail_actions),
    )

    _, s_a = cell.apply(
        variables, carry, (obs, dones, avail_actions), rngs={"dropout": jax.random.PRNGKey(100)}
    )
    _, s_b = cell.apply(
        variables, carry, (obs, dones, avail_actions), rngs={"dropout": jax.random.PRNGKey(200)}
    )
    assert not jnp.allclose(s_a, s_b)


def test_stage2_1_force_zero_gradient_still_flows_to_synapses():
    # With INC_FORCE_ZERO_CONSENSUS=True, the pooled teammate information is
    # zeroed but the widened Synapses first-layer kernel still receives a
    # non-zero gradient (the non-consensus slice of pre_synapse is live).
    # This guards the Stage 2.1 cell E interpretation: cell E strips teammate
    # info WITHOUT also strangling the kernel.
    batch_size = 6
    obs = jnp.arange(batch_size * 5, dtype=jnp.float32).reshape(batch_size, 5) / 50.0
    dones = jnp.zeros((batch_size,), dtype=bool)
    avail_actions = jnp.ones((batch_size, 4), dtype=jnp.float32)
    carry = CTMCell.initialize_carry(batch_size, d_model=8, memory_length=3)

    cell = _build_ctm_cell(
        iterations=3,
        inc_enabled=True,
        inc_pooling="mean",
        inc_consensus_dropout=0.25,
        inc_force_zero_consensus=True,
        deterministic=False,
    )
    variables = cell.init(
        {"params": jax.random.PRNGKey(41), "dropout": jax.random.PRNGKey(42)},
        carry,
        (obs, dones, avail_actions),
    )

    def loss_fn(params):
        _, synch = cell.apply(
            {"params": params},
            carry,
            (obs, dones, avail_actions),
            rngs={"dropout": jax.random.PRNGKey(43)},
        )
        return jnp.sum(synch)

    grads = jax.grad(loss_fn)(variables["params"])
    flat = traverse_util.flatten_dict(grads, sep="/")

    syn_key = "synapses/Dense_0/kernel"
    assert syn_key in flat
    assert jnp.all(jnp.isfinite(flat[syn_key]))
    assert jnp.any(jnp.abs(flat[syn_key]) > 0)


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
