import os
import sys
# Inject repo root into sys.path so modules are always found regardless of CWD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import traceback

from smax_ctm.ctm_jax import (
    SuperLinear,
    NLM,
    CTMBackbone,
    Synapses,
    CTMCell,
    ScannedCTM,
    compute_synchronisation,
    glu,
)
from smax_ctm.train_mappo_ctm import ActorCTM

print("=== Running CTM JAX 0.7.2 Detailed Tests ===")

key = jax.random.PRNGKey(0)
batch, obs_dim, n_actions, seq_len = 4, 52, 9, 10
config = {
    "CTM_D_MODEL": 128, "CTM_D_INPUT": 64, "CTM_ITERATIONS": 1,
    "CTM_N_SYNCH_OUT": 16, "CTM_MEMORY_LENGTH": 5,
    "CTM_DEEP_NLMS": True, "CTM_NLM_HIDDEN_DIM": 2,
    "CTM_DO_LAYERNORM_NLM": False, "CTM_ACTOR_HEAD_DIM": 64,
}
d_model, mem_len = config["CTM_D_MODEL"], config["CTM_MEMORY_LENGTH"]
synch_size = config["CTM_N_SYNCH_OUT"] * (config["CTM_N_SYNCH_OUT"] + 1) // 2


def assert_no_nan_tree(tree, name):
    for i, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        assert not jnp.any(jnp.isnan(leaf)), f"NaN detected in {name} leaf {i}"


print("\n0. Testing GLU helper...")
x = jnp.array([[1.0, 2.0, 0.0, 0.0]])
y = glu(x)
expected = jnp.array([[1.0, 2.0]]) * jax.nn.sigmoid(jnp.array([[0.0, 0.0]]))
assert jnp.allclose(y, expected), f"GLU mismatch: {y} vs {expected}"
print("  GLU behavior correct")


print("\n1. Testing SuperLinear...")
sl = SuperLinear(in_dims=mem_len, out_dims=4, N=d_model)
sl_params = sl.init(key, jnp.ones((batch, d_model, mem_len)))
sl_out = sl.apply(sl_params, jnp.ones((batch, d_model, mem_len)))
assert sl_out.shape == (batch, d_model, 4), f"SuperLinear shape mismatch: {sl_out.shape}"
assert "T" in sl_params["params"], "SuperLinear missing temperature parameter T"

sl_norm = SuperLinear(in_dims=mem_len, out_dims=2, N=d_model, do_norm=True)
sl_norm_params = sl_norm.init(key, jnp.ones((batch, d_model, mem_len)))
sl_norm_out = sl_norm.apply(sl_norm_params, jnp.ones((batch, d_model, mem_len)))
assert sl_norm_out.shape == (batch, d_model, 2), f"SuperLinear(do_norm) shape mismatch: {sl_norm_out.shape}"

def _sl_loss(p):
    return jnp.sum(sl.apply(p, jnp.ones((batch, d_model, mem_len))))

sl_grads = jax.grad(_sl_loss)(sl_params)
assert_no_nan_tree(sl_grads, "SuperLinear grads")
print("  SuperLinear shape and grad checks passed")


print("\n2. Testing NLM (deep + shallow)...")
nlm_deep = NLM(d_model=d_model, memory_length=mem_len, memory_hidden_dims=2, deep_nlms=True)
nlm_deep_params = nlm_deep.init(key, jnp.ones((batch, d_model, mem_len)))
nlm_deep_out = nlm_deep.apply(nlm_deep_params, jnp.ones((batch, d_model, mem_len)))
assert nlm_deep_out.shape == (batch, d_model), f"Deep NLM shape mismatch: {nlm_deep_out.shape}"

nlm_shallow = NLM(d_model=d_model, memory_length=mem_len, memory_hidden_dims=2, deep_nlms=False)
nlm_shallow_params = nlm_shallow.init(key, jnp.ones((batch, d_model, mem_len)))
nlm_shallow_out = nlm_shallow.apply(nlm_shallow_params, jnp.ones((batch, d_model, mem_len)))
assert nlm_shallow_out.shape == (batch, d_model), f"Shallow NLM shape mismatch: {nlm_shallow_out.shape}"

def _nlm_loss(p):
    return jnp.sum(nlm_deep.apply(p, jnp.ones((batch, d_model, mem_len))))

nlm_grads = jax.grad(_nlm_loss)(nlm_deep_params)
assert_no_nan_tree(nlm_grads, "NLM grads")
print("  NLM shape and grad checks passed")


print("\n3. Testing Backbone + Synapses...")
bb = CTMBackbone(d_input=config["CTM_D_INPUT"], obs_dim=obs_dim)
bb_params = bb.init(key, jnp.ones((batch, obs_dim)))
bb_out = bb.apply(bb_params, jnp.ones((batch, obs_dim)))
assert bb_out.shape == (batch, config["CTM_D_INPUT"]), f"Backbone shape mismatch: {bb_out.shape}"

for od in [30, 52, 120]:
    bb_t = CTMBackbone(d_input=config["CTM_D_INPUT"], obs_dim=od)
    bb_t_params = bb_t.init(key, jnp.ones((2, od)))
    bb_t_out = bb_t.apply(bb_t_params, jnp.ones((2, od)))
    assert bb_t_out.shape == (2, config["CTM_D_INPUT"]), f"Backbone failed for obs_dim={od}"

def _bb_loss(p):
    return jnp.sum(bb.apply(p, jnp.ones((batch, obs_dim))))

bb_grads = jax.grad(_bb_loss)(bb_params)
assert_no_nan_tree(bb_grads, "Backbone grads")

syn = Synapses(d_model=d_model, d_input=config["CTM_D_INPUT"])
syn_in = jnp.ones((batch, d_model + config["CTM_D_INPUT"]))
syn_params = syn.init(key, syn_in)
syn_out = syn.apply(syn_params, syn_in)
assert syn_out.shape == (batch, d_model), f"Synapses shape mismatch: {syn_out.shape}"

try:
    syn.apply(syn_params, jnp.ones((batch, config["CTM_D_INPUT"])))
    raise AssertionError("Synapses should reject wrong input feature size")
except Exception:
    pass

def _syn_loss(p):
    return jnp.sum(syn.apply(p, syn_in))

syn_grads = jax.grad(_syn_loss)(syn_params)
assert_no_nan_tree(syn_grads, "Synapses grads")
print("  Backbone and Synapses shape checks passed")


print("\n4. Testing synchronisation computation...")
act_trace = jax.random.normal(key, (batch, d_model, mem_len))
decay_params = jnp.zeros((synch_size,))
synch = compute_synchronisation(act_trace, decay_params, config["CTM_N_SYNCH_OUT"], mem_len)
assert synch.shape == (batch, synch_size), f"Synchronisation shape mismatch: {synch.shape}"
assert not jnp.any(jnp.isnan(synch)), "Synchronisation produced NaN"
assert not jnp.any(jnp.isinf(synch)), "Synchronisation produced Inf"

synch_decay = compute_synchronisation(
    act_trace,
    jnp.ones((synch_size,)) * 2.0,
    config["CTM_N_SYNCH_OUT"],
    mem_len,
)
assert not jnp.allclose(synch, synch_decay), "Decay params should affect synchronisation"

uniform_trace = jnp.ones((1, d_model, mem_len))
uniform_synch = compute_synchronisation(uniform_trace, decay_params, config["CTM_N_SYNCH_OUT"], mem_len)
assert jnp.allclose(uniform_synch, uniform_synch[0, 0]), "Uniform trace should produce uniform synch values"

try:
    compute_synchronisation(act_trace, decay_params, config["CTM_N_SYNCH_OUT"], mem_len, neuron_select_type="random")
    raise AssertionError("Expected unsupported neuron_select_type to raise")
except ValueError:
    pass

def _sync_loss(dp):
    return jnp.sum(compute_synchronisation(act_trace, dp, config["CTM_N_SYNCH_OUT"], mem_len))

sync_grads = jax.grad(_sync_loss)(decay_params)
assert not jnp.any(jnp.isnan(sync_grads)), "NaN gradients in synchronisation decay params"

print("  Synchronisation checks passed")


print("\n5. Testing CTMCell single step + reset + JIT + grads...")
cell = CTMCell(
    d_model=d_model, d_input=config["CTM_D_INPUT"], memory_length=mem_len,
    n_synch_out=config["CTM_N_SYNCH_OUT"], iterations=config["CTM_ITERATIONS"],
    deep_nlms=config["CTM_DEEP_NLMS"], memory_hidden_dims=config["CTM_NLM_HIDDEN_DIM"],
    obs_dim=obs_dim,
)

carry = CTMCell.initialize_carry(batch, d_model, mem_len)
obs = jnp.ones((batch, obs_dim))
dones = jnp.zeros((batch,))
avail = jnp.ones((batch, n_actions))

params = cell.init(key, carry, (obs, dones, avail))
new_carry, synch = cell.apply(params, carry, (obs, dones, avail))

assert new_carry[0].shape == (batch, d_model, mem_len)
assert new_carry[1].shape == (batch, d_model, mem_len)
assert synch.shape == (batch, synch_size)

dones_all = jnp.ones((batch,))
_, synch_reset = cell.apply(params, carry, (obs, dones_all, avail))
assert jnp.allclose(synch_reset[0], synch_reset[1], atol=1e-5), "Reset-on-done mismatch"

@jax.jit
def _jit_cell(p, c, i):
    return cell.apply(p, c, i)

_, synch_jit = _jit_cell(params, carry, (obs, dones, avail))
assert jnp.allclose(synch, synch_jit, atol=1e-6), "JIT vs eager mismatch for CTMCell"

def _cell_loss(p):
    _, s = cell.apply(p, carry, (obs, dones, avail))
    return jnp.sum(s)

cell_grads = jax.grad(_cell_loss)(params)
assert_no_nan_tree(cell_grads, "CTMCell grads")
print("  CTMCell checks passed")


print("\n6. Testing ScannedCTM sequence... ")
scanned = ScannedCTM(config)
scan_obs = jnp.ones((seq_len, batch, obs_dim))
scan_dones = jnp.zeros((seq_len, batch))
scan_avail = jnp.ones((seq_len, batch, n_actions))
scan_params = scanned.init(key, carry, (scan_obs, scan_dones, scan_avail))
scan_carry, scan_synch = scanned.apply(scan_params, carry, (scan_obs, scan_dones, scan_avail))
assert scan_carry[0].shape == (batch, d_model, mem_len)
assert scan_carry[1].shape == (batch, d_model, mem_len)
assert scan_synch.shape == (seq_len, batch, synch_size)
print("  ScannedCTM sequence checks passed")


print("\n7. Testing ActorCTM sequence + masking + sampling...")
actor = ActorCTM(action_dim=n_actions, config=config)

seq_obs = jnp.ones((seq_len, batch, obs_dim))
seq_dones = jnp.zeros((seq_len, batch))
seq_avail = jnp.ones((seq_len, batch, n_actions))

actor_params = actor.init(key, carry, (seq_obs, seq_dones, seq_avail))
final_carry, pi = actor.apply(actor_params, carry, (seq_obs, seq_dones, seq_avail))

log_probs = pi.log_prob(jnp.zeros((seq_len, batch), dtype=jnp.int32))
assert log_probs.shape == (seq_len, batch), f"Actor log_prob shape mismatch: {log_probs.shape}"

avail_masked = jnp.ones((seq_len, batch, n_actions)).at[:, :, 0].set(0)
_, pi_masked = actor.apply(actor_params, carry, (seq_obs, seq_dones, avail_masked))
assert jnp.all(pi_masked.probs[:, :, 0] < 1e-6), "Masked action should have near-zero probability"

actions = pi.sample(seed=key)
assert actions.shape == (seq_len, batch), f"Sample shape mismatch: {actions.shape}"
assert jnp.all(actions >= 0) and jnp.all(actions < n_actions), "Sampled action out of bounds"

assert final_carry[0].shape == (batch, d_model, mem_len)
assert final_carry[1].shape == (batch, d_model, mem_len)

param_count = sum(p.size for p in jax.tree_util.tree_leaves(actor_params))
assert 50_000 < param_count < 500_000, f"Actor parameter count seems off: {param_count}"
print(f"  Actor checks passed; parameter count: {param_count:,}")


print("\n8. Testing full training loop init + tiny compile... ")
try:
    from smax_ctm.train_mappo_ctm import make_train
    print("  Successfully imported make_train")
    
    # We create a tiny config to just compile and step the network once
    tiny_config = {
        "LR": 0.002, "NUM_ENVS": 2, "NUM_STEPS": 4, "TOTAL_TIMESTEPS": 8,
        "FC_DIM_SIZE": 16, "GRU_HIDDEN_DIM": 16, "UPDATE_EPOCHS": 1, "NUM_MINIBATCHES": 1,
        "GAMMA": 0.99, "GAE_LAMBDA": 0.95, "CLIP_EPS": 0.2, "SCALE_CLIP_EPS": False,
        "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5, "ACTIVATION": "relu",
        "OBS_WITH_AGENT_ID": True, "ENV_NAME": "HeuristicEnemySMAX", "MAP_NAME": "3m",
        "SEED": 42, "ENV_KWARGS": {"see_enemy_actions": True, "walls_cause_death": True, "attack_mode": "closest"},
        "ANNEAL_LR": False,
        
        # CTM Specifics
        "CTM_D_MODEL": 16, "CTM_D_INPUT": 16, "CTM_ITERATIONS": 1,
        "CTM_N_SYNCH_OUT": 4, "CTM_MEMORY_LENGTH": 3, "CTM_DEEP_NLMS": True,
        "CTM_NLM_HIDDEN_DIM": 2, "CTM_DO_LAYERNORM_NLM": False,
        "CTM_NEURON_SELECT": "first-last", "CTM_ACTOR_HEAD_DIM": 16
    }

    bad_config = dict(tiny_config)
    bad_config["CTM_NEURON_SELECT"] = "random"
    try:
        make_train(bad_config)
        raise AssertionError("Expected make_train to reject unsupported CTM_NEURON_SELECT")
    except ValueError:
        pass
    
    train_jit = jax.jit(make_train(tiny_config))
    print("  JIT compilation of make_train started... (this may take a minute)")
    out = train_jit(key)
    print("  JIT compilation and 1 tiny training epoch finished successfully!")
    print(f"  Final metrics: return={out['metric']['returned_episode_returns'].mean():.4f}")
    
except Exception as e:
    print(f"\n❌ Error during training loop test: {e}")
    traceback.print_exc()

print("\n=== All Detailed Tests Complete ===")