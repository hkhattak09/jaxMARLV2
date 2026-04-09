import os
import sys
# Inject repo root into sys.path so modules are always found regardless of CWD
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import distrax
import traceback

# Assuming `smax_ctm` is in your Python path or working directory in Colab
from smax_ctm.ctm_jax import CTMCell, compute_synchronisation
from smax_ctm.train_mappo_ctm import ActorCTM

print("=== Running CTM JAX 0.7.2 Shape Tests ===")

# Mock Config
key = jax.random.PRNGKey(0)
batch, obs_dim, n_actions, seq_len = 4, 52, 9, 10
config = {
    "CTM_D_MODEL": 128, "CTM_D_INPUT": 64, "CTM_ITERATIONS": 1,
    "CTM_N_SYNCH_OUT": 16, "CTM_MEMORY_LENGTH": 5,
    "CTM_DEEP_NLMS": True, "CTM_NLM_HIDDEN_DIM": 2,
    "CTM_ACTOR_HEAD_DIM": 64,
}
d_model, mem_len = config["CTM_D_MODEL"], config["CTM_MEMORY_LENGTH"]

# --- 1. Test CTMCell Single Step ---
print("\n1. Testing CTMCell Single Step...")
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

print(f"  Carry state_trace shape: {new_carry[0].shape} (Expected: {batch}, {d_model}, {mem_len})")
print(f"  Synch shape: {synch.shape} (Expected: {batch}, 136)")

# --- 2. Test ActorCTM Sequence (ScannedCTM) ---
print("\n2. Testing ActorCTM Sequence (nn.scan)...")
actor = ActorCTM(action_dim=n_actions, config=config)

seq_obs = jnp.ones((seq_len, batch, obs_dim))
seq_dones = jnp.zeros((seq_len, batch))
seq_avail = jnp.ones((seq_len, batch, n_actions))

actor_params = actor.init(key, carry, (seq_obs, seq_dones, seq_avail))
final_carry, pi = actor.apply(actor_params, carry, (seq_obs, seq_dones, seq_avail))

log_probs = pi.log_prob(jnp.zeros((seq_len, batch), dtype=jnp.int32))
print(f"  Log probs shape: {log_probs.shape} (Expected: {seq_len}, {batch})")

param_count = sum(p.size for p in jax.tree.leaves(actor_params))
print(f"  Actor Param Count: {param_count:,} (Expected: ~115K - 120K)")

# --- 3. Test Full Training Loop Initialization ---
print("\n3. Testing Full Training Loop Init...")
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
        "CTM_NLM_HIDDEN_DIM": 2, "CTM_NEURON_SELECT": "first-last", "CTM_ACTOR_HEAD_DIM": 16
    }
    
    train_jit = jax.jit(make_train(tiny_config))
    print("  JIT compilation of make_train started... (this may take a minute)")
    out = train_jit(key)
    print("  JIT compilation and 1 tiny training epoch finished successfully!")
    print(f"  Final metrics: return={out['metric']['returned_episode_returns'].mean():.4f}")
    
except Exception as e:
    print(f"\n❌ Error during training loop test: {e}")
    traceback.print_exc()

print("\n=== All Local Smoke Tests Complete ===")