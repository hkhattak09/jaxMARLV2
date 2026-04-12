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
from smax_ctm.train_mappo_ctm import (
    ActorCTM,
    compute_saal_alignment_terms,
    compute_focus_fire_mask,
    shuffle_and_split_actor_batch_env_grouped,
)


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

    _, (synch_off, _) = cell_off.apply(vars_off, carry, (obs, dones, avail_actions))
    _, (synch_on, _) = cell_on.apply(vars_on, carry, (obs, dones, avail_actions))
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
        _, (synch, _) = cell.apply({"params": params}, carry, (obs, dones, avail_actions))
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


def test_compute_focus_fire_mask_no_attacks_all_false():
    # T=2, A=3, E=2; all actions are movement actions (< num_movement_actions).
    actions = jnp.array(
        [
            [0, 1, 2, 0, 1, 2],
            [3, 4, 0, 1, 2, 3],
        ],
        dtype=jnp.int32,
    )
    mask = compute_focus_fire_mask(
        actions,
        num_agents=3,
        num_envs=2,
        num_movement_actions=6,
        num_enemies=3,
    )
    assert mask.shape == (2, 2)
    assert jnp.array_equal(mask, jnp.zeros((2, 2), dtype=bool))


def test_compute_focus_fire_mask_two_agents_same_enemy_true():
    # T=1, A=3, E=2 with agent-major flattening: [a0e0,a0e1,a1e0,a1e1,a2e0,a2e1]
    # Attack index for enemy k is num_movement_actions + k.
    # Env0 targets: [6,6,7] => focus fire True
    # Env1 targets: [7,8,8] => focus fire True
    actions = jnp.array([[6, 7, 6, 8, 7, 8]], dtype=jnp.int32)
    mask = compute_focus_fire_mask(
        actions,
        num_agents=3,
        num_envs=2,
        num_movement_actions=6,
        num_enemies=3,
    )
    expected = jnp.array([[True, True]])
    assert jnp.array_equal(mask, expected)


def test_compute_focus_fire_mask_agents_attack_different_enemies_false():
    # T=1, A=3, E=1. Distinct enemy targets in the single env: [6,7,8] => False.
    actions = jnp.array([[6, 7, 8]], dtype=jnp.int32)
    mask = compute_focus_fire_mask(
        actions,
        num_agents=3,
        num_envs=1,
        num_movement_actions=6,
        num_enemies=3,
    )
    expected = jnp.array([[False]])
    assert jnp.array_equal(mask, expected)


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

    _, (synch_base, act_base) = cell_base.apply(vars_base, carry, (obs, dones, avail_actions))
    _, (synch_def, act_def) = cell_defaulted.apply(vars_def, carry, (obs, dones, avail_actions))
    assert jnp.allclose(synch_base, synch_def, atol=0.0, rtol=0.0)
    assert jnp.allclose(act_base, act_def, atol=0.0, rtol=0.0)


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

    _, (synch_off, act_off) = cell_off.apply(v_off, carry, (obs, dones, avail_actions))
    _, (synch_on, act_on) = cell_on.apply(v_on, carry, (obs, dones, avail_actions))
    assert jnp.allclose(synch_off, synch_on, atol=0.0, rtol=0.0)
    assert jnp.allclose(act_off, act_on, atol=0.0, rtol=0.0)


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

    _, (synch_a, act_a) = cell.apply(
        variables, carry, (obs, dones, avail_actions), rngs={"dropout": jax.random.PRNGKey(100)}
    )
    _, (synch_b, act_b) = cell.apply(
        variables, carry, (obs, dones, avail_actions), rngs={"dropout": jax.random.PRNGKey(200)}
    )
    assert not jnp.allclose(synch_a, synch_b) or not jnp.allclose(act_a, act_b)


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
        _, (synch, _) = cell.apply(
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


# ---------------------------------------------------------------------------
# Stage 5 SAAL tests.
# ---------------------------------------------------------------------------


def test_stage5_alignment_term_finite_and_expected_sign():
    # ff values: 0.9, 0.8 → mean = 0.85; nff values: 0.2, 0.1 → mean = 0.15
    pair_cos_ff = jnp.asarray(0.85, dtype=jnp.float32)
    pair_cos_nff = jnp.asarray(0.15, dtype=jnp.float32)

    l_align, align_pos, align_neg = compute_saal_alignment_terms(
        pair_cos_ff,
        pair_cos_nff,
        align_enabled=True,
        align_alpha=0.05,
        align_beta=0.025,
    )

    expected_pos = -0.05 * 0.85
    expected_neg = 0.025 * 0.15
    expected_l = expected_pos + expected_neg

    assert jnp.isfinite(l_align)
    assert jnp.isfinite(align_pos)
    assert jnp.isfinite(align_neg)
    assert jnp.allclose(align_pos, expected_pos)
    assert jnp.allclose(align_neg, expected_neg)
    assert jnp.allclose(l_align, expected_l)
    assert float(l_align) < 0.0


def test_stage5_alignment_off_switch_returns_exact_zeros():
    pair_cos_ff = jnp.asarray(0.7, dtype=jnp.float32)
    pair_cos_nff = jnp.asarray(0.6, dtype=jnp.float32)
    l_align, align_pos, align_neg = compute_saal_alignment_terms(
        pair_cos_ff,
        pair_cos_nff,
        align_enabled=False,
        align_alpha=0.05,
        align_beta=0.025,
    )
    assert float(l_align) == 0.0
    assert float(align_pos) == 0.0
    assert float(align_neg) == 0.0


def test_stage5_alignment_gradient_reaches_ctm_actor_only():
    config = {
        "CTM_D_MODEL": 8,
        "CTM_D_INPUT": 4,
        "CTM_ITERATIONS": 1,
        "CTM_N_SYNCH_OUT": 4,
        "CTM_MEMORY_LENGTH": 3,
        "CTM_DEEP_NLMS": False,
        "CTM_NLM_HIDDEN_DIM": 2,
        "CTM_ACTOR_HEAD_DIM": 8,
        "CTM_USE_SYNC": True,
        "CTM_NEURON_SELECT": "first-last",
        "CTM_DO_LAYERNORM_NLM": False,
        "INC_NUM_AGENTS": 2,
        "INC_ENABLED": False,
        "INC_POOLING": "mean",
        "INC_CONSENSUS_DROPOUT": 0.0,
        "INC_USE_ALIVE_MASK_FROM_DONES": True,
        "CTM_ITER_DROPOUT": 0.0,
        "INC_FORCE_ZERO_CONSENSUS": False,
    }
    actor = ActorCTM(action_dim=5, config=config)

    seq_len = 3
    num_agents = 2
    num_envs = 2
    batch = num_agents * num_envs
    obs_dim = 6

    hidden = CTMCell.initialize_carry(batch, config["CTM_D_MODEL"], config["CTM_MEMORY_LENGTH"])
    obs = jnp.ones((seq_len, batch, obs_dim), dtype=jnp.float32)
    dones = jnp.zeros((seq_len, batch), dtype=bool)
    avail = jnp.ones((seq_len, batch, 5), dtype=jnp.float32)

    variables = actor.init(jax.random.PRNGKey(0), hidden, (obs, dones, avail), deterministic=True)

    # Agent-major flattened actions: [a0e0,a0e1,a1e0,a1e1]
    # Build ff events in both timesteps/envs by having same enemy target for >=2 agents.
    actions = jnp.array(
        [
            [6, 6, 6, 6],
            [7, 7, 7, 7],
            [6, 6, 7, 7],
        ],
        dtype=jnp.int32,
    )

    params = {
        "actor": variables["params"],
        "critic": {"dummy": jnp.ones((4,), dtype=jnp.float32)},
    }

    def loss_fn(packed):
        _, _, synch = actor.apply(
            {"params": packed["actor"]},
            hidden,
            (obs, dones, avail),
            deterministic=True,
        )
        synch_am = synch.reshape(seq_len, num_agents, num_envs, -1)
        s_norm = synch_am / (jnp.linalg.norm(synch_am, axis=-1, keepdims=True) + 1e-8)
        cos_mat = jnp.einsum("taec,tbec->teab", s_norm, s_norm)
        iu, ju = jnp.triu_indices(num_agents, k=1)
        pair_cos = cos_mat[..., iu, ju].mean(axis=-1)

        ff_mask = compute_focus_fire_mask(
            actions,
            num_agents=num_agents,
            num_envs=num_envs,
            num_movement_actions=6,
            num_enemies=3,
        )
        ff_mask_f = ff_mask.astype(pair_cos.dtype)
        nff_mask_f = (~ff_mask).astype(pair_cos.dtype)
        pc_ff = jnp.sum(pair_cos * ff_mask_f) / (jnp.sum(ff_mask_f) + 1e-8)
        pc_nff = jnp.sum(pair_cos * nff_mask_f) / (jnp.sum(nff_mask_f) + 1e-8)
        l_align, _, _ = compute_saal_alignment_terms(
            pc_ff,
            pc_nff,
            align_enabled=True,
            align_alpha=0.05,
            align_beta=0.025,
        )
        return l_align

    grads = jax.grad(loss_fn)(params)
    actor_flat = traverse_util.flatten_dict(grads["actor"], sep="/")

    synapses_nonzero = any(
        ("synapses/" in k) and jnp.any(jnp.abs(v) > 0)
        for k, v in actor_flat.items()
    )
    nlm_nonzero = any(
        ("nlm/" in k) and jnp.any(jnp.abs(v) > 0)
        for k, v in actor_flat.items()
    )
    decay_nonzero = any(
        ("decay_params_out" in k) and jnp.any(jnp.abs(v) > 0)
        for k, v in actor_flat.items()
    )

    assert synapses_nonzero
    assert nlm_nonzero
    assert decay_nonzero
    assert jnp.allclose(grads["critic"]["dummy"], 0.0)
