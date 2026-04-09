# CTM JAX Port Plan — Detailed Step-by-Step

## Error Handling Philosophy: Fail Loud, Never Fake

Prefer a visible failure over a silent fallback.

- **Never silently swallow errors** to keep things "working." Surface the error. Don't substitute placeholder data.
- **Fallbacks are acceptable only when disclosed.** Show a banner, log a warning, annotate the output.
- **Design for debuggability**, not cosmetic stability.

Priority order:
1. Works correctly with real data
2. Falls back visibly — clearly signals degraded mode
3. Fails with a clear error message
4. ~~Silently degrades to look "fine"~~ — **never do this**

This applies everywhere: shape mismatches should crash not reshape silently, NaN losses should halt training not get clipped, prior computation failures should raise not return zeros.

### Execution Environment Protocol
- **Local Device**: All code generation and static syntax checking will be done locally. (JAX is not installed locally and will not be installed).
- **Google Colab**: The user will manually run all actual execution, training, and testing workloads on Google Colab, and share the results/logs back. I will provide precise run instructions, code snippets, and expected output for Colab execution.




## Objective
Port the Continuous Thought Machine (CTM) from PyTorch to JAX/Flax and integrate it as the actor in our existing MAPPO training loop (`train_mappo_gru.py`). The critic remains a GRU. The result is two new files: `ctm_jax.py` (CTM module) and `train_mappo_ctm.py` (training script).

---

## Source Files (PyTorch reference)
| File | What we use from it |
|---|---|
| `other_dirs/continuous-thought-machines/models/ctm_rl.py` | RL-specific CTM: forward pass, synch computation, RL synapses (2-block), backbone, learned start traces |
| `other_dirs/continuous-thought-machines/models/ctm.py` | Base CTM: NLM construction (`get_neuron_level_models`), synch params setup, neuron index init |
| `other_dirs/continuous-thought-machines/models/modules.py` | `SuperLinear` (NLM core), `ClassicControlBackbone`, `SynapseUNET` |
| `other_dirs/continuous-thought-machines/models/utils.py` | `compute_decay` for synch weighting |
| `other_dirs/continuous-thought-machines/tasks/rl/train.py` | Agent class: hidden state init, reset-on-done, `get_states` time-step loop, actor/critic heads |
| `smax_ctm/train_mappo_gru.py` | Our working MAPPO baseline — we modify a copy of this |

---

## Step 1: Create `smax_ctm/ctm_jax.py` — SuperLinear (NLM core)

### What it does
`SuperLinear` is the core of Neuron-Level Models. It applies `d_model` independent linear transforms — one per neuron — to each neuron's temporal history. This is what makes CTM different from a standard RNN: each neuron has its own private MLP that processes its own history.

### PyTorch reference: `modules.py:146-236`
```python
# Weight shape: (memory_length, out_dims, d_model)
# Forward: einsum('BDM,MHD->BDH', x, w1) + b1
# Then squeeze(-1) and divide by learnable temperature T
```

### JAX port
Create a Flax `nn.Module` called `SuperLinear`:
- **Params**: 
  - `w1`: shape `(in_dims, out_dims, N)` — initialized with uniform `[-1/sqrt(in+out), 1/sqrt(in+out)]`
  - `b1`: shape `(1, N, out_dims)` — initialized to zeros
  - `T`: shape `(1,)` — initialized to 1.0 (learnable temperature scalar)
- **Optional**: LayerNorm on last axis of input (controlled by `do_norm` flag; we set False by default)
- **Forward**: 
  ```
  Input x: (batch, d_model, memory_length)
  out = jnp.einsum('BDM,MHD->BDH', x, w1) + b1   # (batch, d_model, out_dims)
  out = out.squeeze(-1) / T                         # (batch, d_model)
  return out
  ```
- **No dropout** for now (dropout=0 in RL config)

### Note on GLU
PyTorch `nn.GLU()` splits the last dimension in half and applies `first_half * sigmoid(second_half)`. JAX has no built-in GLU. Implement as:
```python
def glu(x, axis=-1):
    a, b = jnp.split(x, 2, axis=axis)
    return a * jax.nn.sigmoid(b)
```

### How to Test
```python
# 1. Shape test — verify output shape matches expectations
import jax, jax.numpy as jnp
from ctm_jax import SuperLinear

key = jax.random.PRNGKey(0)
batch, d_model, memory_length, out_dims = 4, 128, 5, 4
sl = SuperLinear(in_dims=memory_length, out_dims=out_dims, N=d_model)
params = sl.init(key, jnp.ones((batch, d_model, memory_length)))
out = sl.apply(params, jnp.ones((batch, d_model, memory_length)))
assert out.shape == (batch, d_model, out_dims), f"Expected (4,128,4), got {out.shape}"
print(f"SuperLinear output shape: {out.shape} ✓")

# 2. Squeeze behavior — when out_dims=1, squeeze should remove last dim
sl1 = SuperLinear(in_dims=memory_length, out_dims=1, N=d_model)
params1 = sl1.init(key, jnp.ones((batch, d_model, memory_length)))
out1 = sl1.apply(params1, jnp.ones((batch, d_model, memory_length)))
# Note: squeeze only happens in NLM's final step, not inside SuperLinear itself
print(f"SuperLinear out_dims=1 shape: {out1.shape}")

# 3. Temperature — check that T param exists and is learnable
flat_params = jax.tree.leaves(params)
print(f"Param count: {sum(p.size for p in flat_params)}")
# Expected: w1 (memory_length * out_dims * d_model) + b1 (d_model * out_dims) + T (1)

# 4. GLU test — verify split-and-gate behavior
from ctm_jax import glu
x = jnp.array([[1.0, 2.0, 0.0, 0.0]])  # first half=[1,2], second half=[0,0]
result = glu(x, axis=-1)
expected = jnp.array([[1.0, 2.0]]) * jax.nn.sigmoid(jnp.array([[0.0, 0.0]]))
assert jnp.allclose(result, expected), f"GLU mismatch: {result} vs {expected}"
print(f"GLU test passed ✓")

# 5. Gradient flow — check gradients are not zero/NaN
def loss_fn(params, x):
    return jnp.sum(sl.apply(params, x))
grads = jax.grad(loss_fn)(params, jnp.ones((batch, d_model, memory_length)))
grad_leaves = jax.tree.leaves(grads)
for i, g in enumerate(grad_leaves):
    assert not jnp.any(jnp.isnan(g)), f"NaN gradient in param {i}"
    assert jnp.any(g != 0), f"Zero gradient in param {i}"
print("Gradient flow test passed ✓")
```

---

## Step 2: Create NLM (trace_processor) in `ctm_jax.py`

### What it does
The NLM takes each neuron's pre-activation history `state_trace[:, d, :]` (a window of length `memory_length`) and produces a single post-activation value per neuron. This is the `g_theta_d` in the paper.

### PyTorch reference: `ctm.py:383-413`
Deep NLMs (default for RL, `deep_nlms=True`, `memory_hidden_dims=2`):
```
SuperLinear(in_dims=memory_length, out_dims=2*memory_hidden_dims, N=d_model)
  -> GLU()                    # halves: out_dims becomes memory_hidden_dims
  -> SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model)
  -> GLU()                    # halves: out_dims becomes 1
  -> Squeeze(-1)              # (batch, d_model, 1) -> (batch, d_model)
```

### JAX port
Create a Flax `nn.Module` called `NLM`:
- **Config**: `d_model`, `memory_length`, `memory_hidden_dims`, `deep_nlms` (bool)
- **Sub-modules**: Two `SuperLinear` layers + GLU between them (if deep), or one `SuperLinear` + GLU (if shallow)
- **Forward**:
  ```
  Input state_trace: (batch, d_model, memory_length)
  
  # Layer 1: (batch, d_model, memory_length) -> (batch, d_model, 2*memory_hidden_dims)
  x = SuperLinear(in_dims=memory_length, out_dims=2*memory_hidden_dims, N=d_model)(state_trace)
  x = glu(x, axis=-1)  # -> (batch, d_model, memory_hidden_dims)
  
  # Layer 2: (batch, d_model, memory_hidden_dims) -> (batch, d_model, 2)
  x = SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model)(x)
  x = glu(x, axis=-1)  # -> (batch, d_model, 1)
  
  x = x.squeeze(-1)    # -> (batch, d_model)
  return x
  ```

### How to Test
```python
# 1. Shape test — full NLM pipeline
from ctm_jax import NLM

key = jax.random.PRNGKey(0)
batch, d_model, memory_length = 4, 128, 5
nlm = NLM(d_model=d_model, memory_length=memory_length, memory_hidden_dims=2, deep_nlms=True)
params = nlm.init(key, jnp.ones((batch, d_model, memory_length)))
out = nlm.apply(params, jnp.ones((batch, d_model, memory_length)))
assert out.shape == (batch, d_model), f"Expected (4,128), got {out.shape}"
print(f"NLM (deep) output shape: {out.shape} ✓")

# 2. Shallow NLM variant — verify it also works
nlm_shallow = NLM(d_model=d_model, memory_length=memory_length, memory_hidden_dims=2, deep_nlms=False)
params_s = nlm_shallow.init(key, jnp.ones((batch, d_model, memory_length)))
out_s = nlm_shallow.apply(params_s, jnp.ones((batch, d_model, memory_length)))
assert out_s.shape == (batch, d_model), f"Shallow NLM shape mismatch: {out_s.shape}"
print(f"NLM (shallow) output shape: {out_s.shape} ✓")

# 3. Different memory_length — verify the NLM adapts
nlm3 = NLM(d_model=64, memory_length=10, memory_hidden_dims=2, deep_nlms=True)
params3 = nlm3.init(key, jnp.ones((2, 64, 10)))
out3 = nlm3.apply(params3, jnp.ones((2, 64, 10)))
assert out3.shape == (2, 64), f"NLM with mem_len=10 shape mismatch: {out3.shape}"
print(f"NLM (mem_len=10) output shape: {out3.shape} ✓")

# 4. Gradient flow through GLU layers
def nlm_loss(params, x):
    return jnp.sum(nlm.apply(params, x))
grads = jax.grad(nlm_loss)(params, jnp.ones((batch, d_model, memory_length)))
grad_leaves = jax.tree.leaves(grads)
for i, g in enumerate(grad_leaves):
    assert not jnp.any(jnp.isnan(g)), f"NaN gradient in NLM param {i}"
print("NLM gradient flow test passed ✓")
```

---

## Step 3: Create Backbone in `ctm_jax.py`

### What it does
Projects raw observation vectors into a fixed-size feature space. SMAX observations are flat vectors, so we use the `ClassicControlBackbone` pattern.

### PyTorch reference: `modules.py:372-386`
```python
nn.Flatten()
nn.LazyLinear(d_input * 2) -> GLU -> LayerNorm(d_input)
nn.LazyLinear(d_input * 2) -> GLU -> LayerNorm(d_input)
```

### JAX port
Create a Flax `nn.Module` called `CTMBackbone`:
- **Config**: `d_input` (output feature dim), `obs_dim` (input obs dim — needed because Flax has no LazyLinear)
- **Forward**:
  ```
  Input obs: (batch, obs_dim)
  x = Dense(d_input * 2)(obs)
  x = glu(x)                      # -> (batch, d_input)
  x = LayerNorm()(x)
  x = Dense(d_input * 2)(x)
  x = glu(x)                      # -> (batch, d_input)
  x = LayerNorm()(x)
  return x                         # (batch, d_input)
  ```

### How to Test
```python
# 1. Shape test
from ctm_jax import CTMBackbone

key = jax.random.PRNGKey(0)
batch, obs_dim, d_input = 4, 52, 64
bb = CTMBackbone(d_input=d_input, obs_dim=obs_dim)
params = bb.init(key, jnp.ones((batch, obs_dim)))
out = bb.apply(params, jnp.ones((batch, obs_dim)))
assert out.shape == (batch, d_input), f"Expected (4,64), got {out.shape}"
print(f"Backbone output shape: {out.shape} ✓")

# 2. Different obs_dim values — test with typical SMAX obs sizes
for od in [30, 52, 120]:  # different SMAX map sizes
    bb_t = CTMBackbone(d_input=d_input, obs_dim=od)
    p = bb_t.init(key, jnp.ones((2, od)))
    o = bb_t.apply(p, jnp.ones((2, od)))
    assert o.shape == (2, d_input), f"Backbone failed for obs_dim={od}"
print("Backbone multi-obs-dim test passed ✓")

# 3. Verify LayerNorm is present — output should be normalized
out_var = jnp.var(out, axis=-1)
# After LayerNorm, variance should be close to 1 (not exactly, due to GLU after)
print(f"Output variance per sample: {out_var}")  # sanity check, not a hard assert

# 4. Gradient flow
def bb_loss(params, x):
    return jnp.sum(bb.apply(params, x))
grads = jax.grad(bb_loss)(params, jnp.ones((batch, obs_dim)))
for g in jax.tree.leaves(grads):
    assert not jnp.any(jnp.isnan(g)), "NaN gradient in Backbone"
print("Backbone gradient flow test passed ✓")
```

---

## Step 4: Create Synapses in `ctm_jax.py`

### What it does
The synapse model is the inter-neuron communication step. It takes the concatenation of `[features, last_activated_state]` and produces a new pre-activation state vector. This is how neurons share information.

### PyTorch reference: `ctm_rl.py:99-119`
For RL, `synapse_depth=1` uses a **2-block** architecture (note: the RL variant overrides the base CTM's single-block `synapse_depth=1` to use two blocks):
```python
nn.Dropout(dropout)
nn.LazyLinear(d_model*2) -> GLU -> LayerNorm(d_model)
nn.LazyLinear(d_model*2) -> GLU -> LayerNorm(d_model)
```

### JAX port
Create a Flax `nn.Module` called `Synapses`:
- **Config**: `d_model`, `d_input` (to compute input dim = `d_input + d_model`)
- **Forward**:
  ```
  Input: (batch, d_input + d_model)   # concat of features and last activated state
  x = Dense(d_model * 2)(input)
  x = glu(x)                          # -> (batch, d_model)
  x = LayerNorm()(x)
  x = Dense(d_model * 2)(x)
  x = glu(x)                          # -> (batch, d_model)
  x = LayerNorm()(x)
  return x                             # (batch, d_model)
  ```
- No dropout (dropout=0 in RL config)

### How to Test
```python
# 1. Shape test
from ctm_jax import Synapses

key = jax.random.PRNGKey(0)
batch, d_model, d_input = 4, 128, 64
syn = Synapses(d_model=d_model, d_input=d_input)
input_dim = d_input + d_model  # 192
params = syn.init(key, jnp.ones((batch, input_dim)))
out = syn.apply(params, jnp.ones((batch, input_dim)))
assert out.shape == (batch, d_model), f"Expected (4,128), got {out.shape}"
print(f"Synapses output shape: {out.shape} ✓")

# 2. Verify input dim = d_input + d_model (the concat of features + last_activated)
try:
    syn.apply(params, jnp.ones((batch, d_input)))  # wrong size
    assert False, "Should have failed with wrong input dim"
except Exception:
    print("Synapses correctly rejects wrong input dim ✓")

# 3. Gradient flow
def syn_loss(params, x):
    return jnp.sum(syn.apply(params, x))
grads = jax.grad(syn_loss)(params, jnp.ones((batch, input_dim)))
for g in jax.tree.leaves(grads):
    assert not jnp.any(jnp.isnan(g)), "NaN gradient in Synapses"
print("Synapses gradient flow test passed ✓")
```

---

## Step 5: Create Synchronisation in `ctm_jax.py`

### What it does
Synchronisation converts the internal neural dynamics into a representation vector. It measures how pairs of neurons co-activate over the time window. This is the CTM's output representation — not the raw neuron states, but a measure of their coordination.

### PyTorch reference: `ctm_rl.py:64-72`
The RL variant uses the `first-last` neuron selection with the full `activated_state_trace`:
```python
# S shape after permute: (batch, memory_length, d_model)
S = activated_state_trace.permute(0, 2, 1)

# Select first n_synch_out neurons: S[:, :, :n_synch_out]  -> (batch, memory_length, n_synch_out)
# Pairwise products: S[..., i] * S[..., j] for all i<=j (upper triangle)
# -> (batch, memory_length, n_synch_out*(n_synch_out+1)/2)

# Decay weighting over time dimension:
# decay = compute_decay(memory_length, decay_params_out)
# -> exponential weights: recent steps weighted more

# synchronisation = sum(decay * pairwise_products, over time) / sqrt(sum(decay, over time))
# -> (batch, synch_size)  where synch_size = n_synch_out*(n_synch_out+1)/2
```

### Detailed math for `compute_decay` (from `utils.py:6-16`)
```python
# For each pair p, compute:
#   indices = [T-1, T-2, ..., 1, 0]  (most recent = index 0)
#   decay[t, p] = exp(-indices[t] * clamp(decay_params[p], 0, 4))
# This creates exponential decay weights — recent activations count more
```

### JAX port
Create a function `compute_synchronisation`:
- **Params**: `decay_params_out` — learnable, shape `(synch_size,)`, initialized to zeros
- **Synch size**: `n_synch_out * (n_synch_out + 1) // 2` (upper triangle of n_synch_out x n_synch_out matrix)
- **Forward**:
  ```
  Input activated_state_trace: (batch, d_model, memory_length)
  
  # 1. Transpose to (batch, memory_length, d_model)
  S = activated_state_trace.transpose(0, 2, 1)
  
  # 2. Select first n_synch_out neurons -> (batch, memory_length, n_synch_out)
  S_sel = S[:, :, :n_synch_out]
  
  # 3. Upper-triangle pairwise products
  # For each time step, compute outer product and take upper triangle
  # S_sel[:, :, i] * S_sel[:, :, j] for all i <= j
  # -> (batch, memory_length, synch_size)
  triu_i, triu_j = jnp.triu_indices(n_synch_out)
  pairwise = S_sel[:, :, triu_i] * S_sel[:, :, triu_j]  # (batch, memory_length, synch_size)
  
  # 4. Compute decay weights
  indices = jnp.arange(memory_length - 1, -1, -1)  # [M-1, M-2, ..., 0]
  clamped_params = jnp.clip(decay_params, 0.0, 4.0)  # (synch_size,)
  decay = jnp.exp(-indices[:, None] * clamped_params[None, :])  # (memory_length, synch_size)
  
  # 5. Weighted sum over time, normalized
  numerator = jnp.sum(decay[None, :, :] * pairwise, axis=1)  # (batch, synch_size)
  denominator = jnp.sqrt(jnp.sum(decay, axis=0))[None, :]    # (1, synch_size)
  synchronisation = numerator / denominator                    # (batch, synch_size)
  
  return synchronisation
  ```

### How to Test
```python
# 1. Shape test with known values
from ctm_jax import compute_synchronisation

key = jax.random.PRNGKey(0)
batch, d_model, memory_length, n_synch_out = 4, 128, 5, 16
synch_size = n_synch_out * (n_synch_out + 1) // 2  # 136

# Create fake activated_state_trace and decay_params
act_trace = jax.random.normal(key, (batch, d_model, memory_length))
decay_params = jnp.zeros((synch_size,))

out = compute_synchronisation(act_trace, decay_params, n_synch_out, memory_length)
assert out.shape == (batch, synch_size), f"Expected (4,136), got {out.shape}"
print(f"Synchronisation output shape: {out.shape} ✓")

# 2. Decay behavior — with zero decay_params, all time steps weighted equally
# With non-zero params, recent steps should dominate
decay_zero = jnp.zeros((synch_size,))
decay_high = jnp.ones((synch_size,)) * 2.0  # strong recency bias

out_flat = compute_synchronisation(act_trace, decay_zero, n_synch_out, memory_length)
out_decay = compute_synchronisation(act_trace, decay_high, n_synch_out, memory_length)
# These should differ — decay_high weights recent steps much more
assert not jnp.allclose(out_flat, out_decay), "Decay params should change output"
print("Decay parameter effect verified ✓")

# 3. Symmetry check — synch should use upper-triangle pairwise products
# If all neurons have same activation, all pairs should give same product
uniform_trace = jnp.ones((1, d_model, memory_length))
out_uniform = compute_synchronisation(uniform_trace, decay_zero, n_synch_out, memory_length)
# All synch values should be identical (since all neuron pairs are 1*1=1)
assert jnp.allclose(out_uniform, out_uniform[0, 0]), "Uniform input should give uniform synch"
print("Symmetry test passed ✓")

# 4. No NaN/Inf
assert not jnp.any(jnp.isnan(out)), "NaN in synchronisation output"
assert not jnp.any(jnp.isinf(out)), "Inf in synchronisation output"
print("No NaN/Inf test passed ✓")

# 5. Gradient flow through decay_params
def synch_loss(decay_p, trace):
    return jnp.sum(compute_synchronisation(trace, decay_p, n_synch_out, memory_length))
grads = jax.grad(synch_loss)(decay_params, act_trace)
assert not jnp.any(jnp.isnan(grads)), "NaN gradient in decay_params"
print("Synchronisation gradient flow test passed ✓")
```

---

## Step 6: Create `CTMCell` in `ctm_jax.py` — full single-step CTM

### What it does
Combines all components into one module that takes an observation and the previous hidden state, and outputs the synchronisation vector + updated hidden state. This is the equivalent of one call to `ContinuousThoughtMachineRL.forward()`.

### PyTorch reference: `ctm_rl.py:151-192` + `train.py:202-209`

### JAX port
Create a Flax `nn.Module` called `CTMCell`:
- **Config**: `d_model`, `d_input`, `memory_length`, `n_synch_out`, `iterations`, `deep_nlms`, `memory_hidden_dims`, `obs_dim`
- **Sub-modules**: `CTMBackbone`, `Synapses`, `NLM`
- **Params** (learned initial states):
  - `start_trace`: shape `(d_model, memory_length)` — initialized uniform `[-1/sqrt(d_model+memory_length), 1/sqrt(d_model+memory_length)]`
  - `start_activated_trace`: shape `(d_model, memory_length)` — initialized same as above
  - `decay_params_out`: shape `(synch_size,)` — initialized to zeros
- **`initialize_carry` static method**:
  ```
  # Returns zeros for now — the actual learned start states are in params
  # and will be used during the reset-on-done logic
  # For init we need: (state_trace, activated_state_trace)
  # Each: (batch_size, d_model, memory_length)
  return (jnp.zeros((batch_size, d_model, memory_length)),
          jnp.zeros((batch_size, d_model, memory_length)))
  ```
- **Forward** `__call__(self, carry, x)`:
  ```
  # carry = (state_trace, activated_state_trace)  each (batch, d_model, memory_length)
  # x = (obs, dones, avail_actions)
  # obs: (batch, obs_dim), dones: (batch,), avail_actions: (batch, n_actions)
  
  obs, dones, avail_actions = x
  state_trace, activated_state_trace = carry
  
  # --- Reset on done ---
  # When an episode ends, replace carry with learned start states
  start_trace = self.param('start_trace', ...)          # (d_model, memory_length)
  start_activated_trace = self.param('start_activated_trace', ...)  # (d_model, memory_length)
  
  # Broadcast start states to batch: (1, d_model, memory_length) -> (batch, d_model, memory_length)
  reset_mask = dones[:, None, None]  # (batch, 1, 1)
  state_trace = jnp.where(reset_mask, start_trace[None], state_trace)
  activated_state_trace = jnp.where(reset_mask, start_activated_trace[None], activated_state_trace)
  
  # --- Backbone ---
  features = CTMBackbone(d_input, obs_dim)(obs)  # (batch, d_input)
  
  # --- Internal ticks (iterations, default=1 for RL) ---
  for _ in range(iterations):
      # Concat features with last activated state
      last_activated = activated_state_trace[:, :, -1]  # (batch, d_model)
      pre_synapse = jnp.concatenate([features, last_activated], axis=-1)  # (batch, d_input + d_model)
      
      # Synapses
      new_state = Synapses(d_model, d_input)(pre_synapse)  # (batch, d_model)
      
      # Shift state_trace left, append new_state
      state_trace = jnp.concatenate([state_trace[:, :, 1:], new_state[:, :, None]], axis=-1)
      
      # NLM
      activated_state = NLM(d_model, memory_length, memory_hidden_dims, deep_nlms)(state_trace)  # (batch, d_model)
      
      # Shift activated_state_trace left, append
      activated_state_trace = jnp.concatenate([activated_state_trace[:, :, 1:], activated_state[:, :, None]], axis=-1)
  
  # --- Synchronisation ---
  synch = compute_synchronisation(activated_state_trace, self.param('decay_params_out', ...), n_synch_out, memory_length)
  # synch: (batch, synch_size)
  
  new_carry = (state_trace, activated_state_trace)
  return new_carry, synch
  ```

### Important: `nn.scan` compatibility
The GRU version uses `ScannedRNN` which wraps GRUCell with `nn.scan` so it can process sequences of time steps (the `NUM_STEPS` rollout). We need the same for CTM.

Create `ScannedCTM`:
```python
class ScannedCTM(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        return CTMCell(...)(carry, x)
```

The carry shape is `(state_trace, activated_state_trace)` — a tuple of two `(batch, d_model, memory_length)` arrays — instead of a single `(batch, hidden_dim)` for GRU.

### How to Test
```python
# 1. Single-step shape test — the most critical test
from ctm_jax import CTMCell

key = jax.random.PRNGKey(0)
batch, obs_dim, n_actions = 4, 52, 9
config = {
    "CTM_D_MODEL": 128, "CTM_D_INPUT": 64, "CTM_ITERATIONS": 1,
    "CTM_N_SYNCH_OUT": 16, "CTM_MEMORY_LENGTH": 5,
    "CTM_DEEP_NLMS": True, "CTM_NLM_HIDDEN_DIM": 2,
}
d_model, mem_len = config["CTM_D_MODEL"], config["CTM_MEMORY_LENGTH"]
synch_size = config["CTM_N_SYNCH_OUT"] * (config["CTM_N_SYNCH_OUT"] + 1) // 2

cell = CTMCell(
    d_model=d_model, d_input=config["CTM_D_INPUT"], memory_length=mem_len,
    n_synch_out=config["CTM_N_SYNCH_OUT"], iterations=config["CTM_ITERATIONS"],
    deep_nlms=config["CTM_DEEP_NLMS"], memory_hidden_dims=config["CTM_NLM_HIDDEN_DIM"],
    obs_dim=obs_dim,
)

# Init carry and inputs
carry = CTMCell.initialize_carry(batch, d_model, mem_len)
obs = jnp.ones((batch, obs_dim))
dones = jnp.zeros((batch,))
avail = jnp.ones((batch, n_actions))

params = cell.init(key, carry, (obs, dones, avail))
new_carry, synch = cell.apply(params, carry, (obs, dones, avail))

state_trace, act_trace = new_carry
assert state_trace.shape == (batch, d_model, mem_len), f"state_trace shape: {state_trace.shape}"
assert act_trace.shape == (batch, d_model, mem_len), f"act_trace shape: {act_trace.shape}"
assert synch.shape == (batch, synch_size), f"synch shape: {synch.shape}"
print(f"CTMCell shapes correct: carry=({state_trace.shape}, {act_trace.shape}), synch={synch.shape} ✓")

# 2. Reset-on-done — verify carry resets to learned start traces
dones_all = jnp.ones((batch,))  # all episodes done
_, synch_reset = cell.apply(params, carry, (obs, dones_all, avail))
# After reset, all agents should produce identical synch (same start state + same obs)
assert jnp.allclose(synch_reset[0], synch_reset[1], atol=1e-5), "Reset should give identical outputs"
print("Reset-on-done test passed ✓")

# 3. State trace sliding window — verify the trace shifts correctly
carry_zero = CTMCell.initialize_carry(batch, d_model, mem_len)
new_carry_1, _ = cell.apply(params, carry_zero, (obs, dones, avail))
new_carry_2, _ = cell.apply(params, new_carry_1, (obs, dones, avail))
# After 2 steps from zeros, the first (mem_len - 2) slots of step2's trace
# should equal the last (mem_len - 2) slots of step1's trace (sliding window)
st1 = new_carry_1[0][:, :, 1:]   # slots [1:] from step 1
st2 = new_carry_2[0][:, :, :-1]  # slots [:-1] from step 2
# These won't be exactly equal because step 2 sees different last_activated,
# but the overlap region (slots that weren't newly written) should match
# Check that slot [1] of step2 == slot [2] of step1 (shifted by one)
# Actually the NEW state goes into the last slot, so:
# step1 trace = [0, 0, 0, 0, s1]  (for mem_len=5, starting from zeros)
# step2 trace = [0, 0, 0, s1, s2]
assert jnp.allclose(new_carry_1[0][:, :, -1], new_carry_2[0][:, :, -2], atol=1e-5), \
    "Sliding window: step1's last slot should appear at step2's second-to-last"
print("Sliding window test passed ✓")

# 4. ScannedCTM — verify it processes a sequence
from ctm_jax import ScannedCTM

seq_len = 10
scan_obs = jnp.ones((seq_len, batch, obs_dim))
scan_dones = jnp.zeros((seq_len, batch))
scan_avail = jnp.ones((seq_len, batch, n_actions))

scanned = ScannedCTM(config)  # or however it's parameterized
# Test that init + apply work with sequence inputs
# The output synch should be (seq_len, batch, synch_size)
# and the final carry should be same shape as initial carry
# (exact API depends on how ScannedCTM wraps CTMCell)
print("ScannedCTM sequence test — verify manually after implementation")

# 5. JIT compilation test — ensure everything traces cleanly
@jax.jit
def jit_step(params, carry, x):
    return cell.apply(params, carry, x)

new_carry_jit, synch_jit = jit_step(params, carry, (obs, dones, avail))
assert jnp.allclose(synch, synch_jit, atol=1e-6), "JIT output should match eager"
print("JIT compilation test passed ✓")

# 6. Full gradient test — can we differentiate through the whole cell?
def full_loss(params, carry, x):
    _, synch = cell.apply(params, carry, x)
    return jnp.sum(synch)
grads = jax.grad(full_loss)(params, carry, (obs, dones, avail))
grad_leaves = jax.tree.leaves(grads)
nan_count = sum(1 for g in grad_leaves if jnp.any(jnp.isnan(g)))
zero_count = sum(1 for g in grad_leaves if jnp.all(g == 0))
print(f"CTMCell gradient test: {len(grad_leaves)} params, {nan_count} NaN, {zero_count} all-zero")
assert nan_count == 0, "NaN gradients in CTMCell"
print("CTMCell full gradient test passed ✓")
```

---

## Step 7: Create `ActorCTM` in `train_mappo_ctm.py`

### What it does
Replaces `ActorRNN`. Uses `ScannedCTM` instead of `ScannedRNN`, and feeds the synchronisation vector (not GRU output) into actor head Dense layers.

### PyTorch reference: `train.py:149-155`
Actor head in PyTorch RL:
```python
Linear(synch_size, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, n_actions)
```

### JAX port
```python
class ActorCTM(nn.Module):
    action_dim: int
    config: Dict
    
    @nn.compact
    def __call__(self, hidden, x):
        # hidden = (state_trace, activated_state_trace)
        # x = (obs, dones, avail_actions)
        obs, dones, avail_actions = x
        
        # CTM processes the sequence
        ctm_in = (obs, dones, avail_actions)
        hidden, synch = ScannedCTM(config)(hidden, ctm_in)
        # synch: (seq_len, batch, synch_size)
        
        # Actor head: synch -> logits
        x = Dense(config["CTM_ACTOR_HEAD_DIM"])(synch)
        x = relu(x)
        x = Dense(self.action_dim)(x)
        
        # Mask unavailable actions
        unavail_actions = 1 - avail_actions
        action_logits = x - (unavail_actions * 1e10)
        
        pi = distrax.Categorical(logits=action_logits)
        return hidden, pi
```

### Synch size calculation
With `n_synch_out=16` and `first-last` selection:
- `synch_size = 16 * (16 + 1) / 2 = 136`
- This is the input dim to the actor head

### How to Test
```python
# 1. Shape test — ActorCTM produces valid policy
from train_mappo_ctm import ActorCTM
from ctm_jax import CTMCell

key = jax.random.PRNGKey(0)
batch, obs_dim, n_actions, seq_len = 4, 52, 9, 10
config = {
    "CTM_D_MODEL": 128, "CTM_D_INPUT": 64, "CTM_ITERATIONS": 1,
    "CTM_N_SYNCH_OUT": 16, "CTM_MEMORY_LENGTH": 5,
    "CTM_DEEP_NLMS": True, "CTM_NLM_HIDDEN_DIM": 2,
    "CTM_ACTOR_HEAD_DIM": 64,
}
d_model, mem_len = config["CTM_D_MODEL"], config["CTM_MEMORY_LENGTH"]

actor = ActorCTM(action_dim=n_actions, config=config)
hidden = CTMCell.initialize_carry(batch, d_model, mem_len)
obs = jnp.ones((seq_len, batch, obs_dim))
dones = jnp.zeros((seq_len, batch))
avail = jnp.ones((seq_len, batch, n_actions))

params = actor.init(key, hidden, (obs, dones, avail))
new_hidden, pi = actor.apply(params, hidden, (obs, dones, avail))

# Verify policy outputs
log_probs = pi.log_prob(jnp.zeros((seq_len, batch), dtype=jnp.int32))
assert log_probs.shape == (seq_len, batch), f"log_prob shape: {log_probs.shape}"
print(f"ActorCTM log_prob shape: {log_probs.shape} ✓")

# 2. Action masking — unavailable actions should have ~zero probability
avail_masked = jnp.ones((seq_len, batch, n_actions))
avail_masked = avail_masked.at[:, :, 0].set(0)  # mask action 0
_, pi_masked = actor.apply(params, hidden, (obs, dones, avail_masked))
probs = pi_masked.probs  # (seq_len, batch, n_actions)
assert jnp.all(probs[:, :, 0] < 1e-6), "Masked action should have ~0 probability"
print("Action masking test passed ✓")

# 3. Sampling works
actions = pi.sample(seed=key)
assert actions.shape == (seq_len, batch), f"Sample shape: {actions.shape}"
assert jnp.all(actions >= 0) and jnp.all(actions < n_actions), "Actions out of range"
print(f"Sampling test passed ✓")

# 4. Hidden state shape preserved through sequence
st, at = new_hidden
assert st.shape == (batch, d_model, mem_len), f"Final state_trace: {st.shape}"
assert at.shape == (batch, d_model, mem_len), f"Final act_trace: {at.shape}"
print("Hidden state shape preservation test passed ✓")

# 5. Param count sanity check
param_count = sum(p.size for p in jax.tree.leaves(params))
print(f"ActorCTM total params: {param_count:,}")
# Expected ~115-120K based on plan estimates
assert 50_000 < param_count < 500_000, f"Param count {param_count} seems off"
print("Param count sanity check passed ✓")
```

---

## Step 8: Create `train_mappo_ctm.py` — modify training script

Copy `train_mappo_gru.py` and make these specific changes:

### 8a. Imports
- Add: `from ctm_jax import ScannedCTM, CTMCell` (or inline if in same file)
- Keep all existing imports

### 8b. Replace `ScannedRNN` + `ActorRNN` with `ScannedCTM` + `ActorCTM`
- Keep `ScannedRNN` and `CriticRNN` unchanged (critic stays GRU)

### 8c. Config additions
Add CTM-specific hyperparameters to the config dict:
```python
"CTM_D_MODEL": 128,          # number of internal neurons
"CTM_D_INPUT": 64,           # backbone output dim
"CTM_ITERATIONS": 1,         # internal ticks per env step
"CTM_N_SYNCH_OUT": 16,       # neurons for output synchronisation
"CTM_MEMORY_LENGTH": 5,      # NLM history window
"CTM_DEEP_NLMS": True,       # deep (2-layer) vs shallow (1-layer) NLMs
"CTM_NLM_HIDDEN_DIM": 2,     # hidden dim for deep NLMs
"CTM_NEURON_SELECT": "first-last",  # neuron selection for synch
"CTM_ACTOR_HEAD_DIM": 64,    # actor head hidden dim
```

### 8d. Network initialization (in `train` function)
Replace actor init block:
```python
# OLD (GRU):
# actor_network = ActorRNN(action_dim, config)
# ac_init_x = (obs_dummy, done_dummy, avail_dummy)
# ac_init_hstate = ScannedRNN.initialize_carry(NUM_ENVS, GRU_HIDDEN_DIM)

# NEW (CTM):
actor_network = ActorCTM(action_dim, config)
ac_init_x = (
    jnp.zeros((1, NUM_ENVS, obs_dim)),      # obs
    jnp.zeros((1, NUM_ENVS)),                 # dones
    jnp.zeros((1, NUM_ENVS, action_dim)),     # avail_actions
)
ac_init_hstate = CTMCell.initialize_carry(NUM_ENVS, config["CTM_D_MODEL"], config["CTM_MEMORY_LENGTH"])
# Returns: (jnp.zeros((NUM_ENVS, d_model, memory_length)),
#           jnp.zeros((NUM_ENVS, d_model, memory_length)))
```
Critic init stays exactly the same (GRU).

### 8e. Hidden state in `_env_step`
The runner_state carries `hstates = (ac_hstate, cr_hstate)`.
- `ac_hstate` changes from `(batch, gru_dim)` to `((batch, d_model, mem_len), (batch, d_model, mem_len))`
- `cr_hstate` stays `(batch, gru_dim)`
- The tuple structure `hstates[0]`, `hstates[1]` is preserved, so the env_step structure stays the same

### 8f. Hidden state in `_update_epoch` minibatch slicing
Currently for GRU:
```python
init_hstates = jax.tree.map(lambda x: jnp.reshape(x, (1, NUM_ACTORS, -1)), init_hstates)
```
For CTM actor hidden state, the reshape needs to handle `(batch, d_model, memory_length)`:
```python
# The CTM hidden state is a tuple of two (batch, d_model, memory_length) arrays
# When reshaping for minibatches, we need to handle this 3D shape
# jax.tree.map will handle the tuple structure automatically
```
The permutation/shuffle along the actor dimension (axis=1) and minibatch splitting will work with `jax.tree.map` since it operates element-wise on the pytree leaves. The CTM carry arrays are `(1, NUM_ACTORS, d_model, memory_length)` instead of `(1, NUM_ACTORS, gru_dim)`, but the shuffle/split logic operates on axis 1 (the actor axis) which is the same.

### 8g. Everything else stays the same
- `CriticRNN`, `SMAXWorldStateWrapper`, `Transition`, `batchify`, `unbatchify`: unchanged
- GAE computation: unchanged
- PPO clipped loss (actor): unchanged (it only uses `pi` from actor, which is produced the same way)
- Value loss (critic): unchanged
- Logging, env stepping, reward computation: unchanged

### How to Test (Step 8 — integration tests before full training)
```python
# These tests verify the training script wiring is correct before committing
# to a full run. Run each on Colab with a tiny config.

# === Test 8.1: Network init compiles ===
# In train_mappo_ctm.py, temporarily add after network init:
print(f"Actor params: {sum(p.size for p in jax.tree.leaves(actor_params)):,}")
print(f"Critic params: {sum(p.size for p in jax.tree.leaves(critic_params)):,}")
print(f"Actor hidden shape: {jax.tree.map(lambda x: x.shape, ac_init_hstate)}")
print(f"Critic hidden shape: {cr_init_hstate.shape}")
# Expected: Actor ~115K params, Critic same as GRU baseline
# Actor hidden: ((NUM_ENVS, 128, 5), (NUM_ENVS, 128, 5))
# Critic hidden: (NUM_ENVS, 128) or whatever GRU dim is

# === Test 8.2: Single env_step runs ===
# Use tiny config and add early exit after 1 step:
config_test = {
    "TOTAL_TIMESTEPS": 1000,
    "NUM_STEPS": 5,
    "NUM_ENVS": 2,
    "CTM_D_MODEL": 16,       # tiny for fast compile
    "CTM_D_INPUT": 8,
    "CTM_MEMORY_LENGTH": 3,
    "CTM_N_SYNCH_OUT": 4,    # synch_size = 10
    "CTM_ITERATIONS": 1,
    "CTM_DEEP_NLMS": True,
    "CTM_NLM_HIDDEN_DIM": 2,
    "CTM_ACTOR_HEAD_DIM": 16,
    # ... keep all other config from GRU baseline
}
# Run and check: no shape errors during JIT compilation

# === Test 8.3: Minibatch slicing works ===
# The most likely failure point is reshaping CTM hidden states for minibatches.
# Add a print inside _update_epoch after the reshape:
# print(f"Minibatch hstate shapes: {jax.tree.map(lambda x: x.shape, init_hstates)}")
# Verify the actor hstate is (1, MINIBATCH_SIZE, d_model, mem_len) not something weird

# === Test 8.4: Loss is finite and decreasing ===
# With the tiny config, run ~1000 timesteps and verify:
# - actor_loss is not NaN
# - critic_loss is not NaN  
# - entropy is positive (not collapsed)
# - returns are changing (not stuck at 0)
# Add print statements in the update function:
print(f"actor_loss={actor_loss:.4f} critic_loss={critic_loss:.4f} entropy={entropy:.4f}")

# === Test 8.5: Gradient magnitudes are reasonable ===
# Inside the update step, after computing grads:
actor_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(actor_grads)))
print(f"Actor grad norm: {actor_grad_norm:.4f}")
# Should be finite, non-zero, and not exploding (< 100 is fine with grad clipping)

# === Test 8.6: Hidden state propagation across rollout ===
# Verify that the actor hidden state actually evolves during a rollout.
# Add a print at the end of _env_step:
# print(f"ac_hstate mean: {jnp.mean(hstates[0][0]):.6f}")
# The mean should change across steps (not stuck at 0 or constant)
```

---

## Step 9: Smoke test on Colab

### 9a. Verify shapes compile
Run `train_mappo_ctm.py` on `3m` with a tiny config first:
```python
"TOTAL_TIMESTEPS": 50_000,   # just enough to see a few log lines
"CTM_D_MODEL": 32,           # small for fast compile
"CTM_MEMORY_LENGTH": 3,
```
Check that:
- JIT compilation succeeds (no shape errors)
- Training loop runs without NaN
- Returns and win rates are printed

### 9b. Full training run
Run with the full config from Step 8c:
```python
"TOTAL_TIMESTEPS": 3_000_000,
"CTM_D_MODEL": 128,
"CTM_MEMORY_LENGTH": 5,
```
Compare against GRU baseline (87% win rate, 1.86 return on 3m).

### How to Test (Step 9 — end-to-end validation checklist)
Run these checks on Colab during the smoke test. Print all diagnostics — don't just check if it "runs."

```
=== Checklist for 9a (tiny config, 50K steps) ===

[ ] JIT compiles without errors (first step takes a while, that's normal)
[ ] No NaN in any logged loss (actor, critic, entropy)
[ ] No NaN in returns or win rates
[ ] Entropy starts high (~log(n_actions)) and stays positive
[ ] Actor loss magnitude is reasonable (not 0, not 1e10)
[ ] Win rate is 0% or very low (expected — only 50K steps)
[ ] Memory usage is reasonable (CTM uses ~2x GRU memory for actor hidden state)
[ ] Print param counts match plan estimates

=== Checklist for 9b (full config, 3M steps) ===

[ ] Training completes without crash
[ ] Win rate curve shows learning (should rise over time)
[ ] Final win rate is > 50% (ideally approaching GRU baseline of 87%)
[ ] Returns are increasing over training
[ ] No entropy collapse (entropy shouldn't drop to near 0)
[ ] No loss spikes or NaN appearing mid-training
[ ] Compare wall-clock time to GRU baseline — CTM should be slower but < 5x

=== Red flags that indicate a bug ===

- Win rate stays at exactly 0% after 1M+ steps → hidden state not propagating, 
  or synch output is degenerate (all same values)
- NaN appears after N steps → numerical instability, check synchronisation 
  denominator (sqrt could hit 0 if decay is extreme)
- Entropy collapses immediately → action masking bug or logits are extreme
- Actor loss is exactly 0 → gradients not flowing through CTM
- Memory keeps growing → state trace arrays being accumulated instead of 
  overwritten (leak in scan carry)
```

---

## JAX 0.7.2 Specific Considerations

1. **No `nn.LazyLinear`**: All Dense layer input dims must be known at init time. We know them:
   - Backbone input: `obs_dim` (from env)
   - Synapses input: `d_input + d_model`
   - SuperLinear input: `memory_length` or `memory_hidden_dims`
   - Actor head input: `synch_size = n_synch_out * (n_synch_out + 1) // 2`

2. **No built-in GLU**: Implement manually as `a, b = split(x, 2); return a * sigmoid(b)`

3. **`nn.scan` with tuple carry**: `nn.scan` supports pytree carries, so `(state_trace, activated_state_trace)` as a tuple works fine — same pattern as the existing `ScannedRNN`

4. **Learned initial params in scanned module**: Use `self.param(name, init_fn, shape)` inside the `nn.compact` method. With `variable_broadcast="params"`, these are shared across scan steps.

5. **`jnp.triu_indices`**: Available in JAX, same API as NumPy/PyTorch

6. **No Python for-loops in JIT for iterations**: Since `iterations=1` for RL, this is a single unrolled step inside JIT, no issue. If we later increase iterations, we may want `jax.lax.fori_loop`, but for now a Python loop unrolled at trace time is fine.

7. **Sliding window update**: `jnp.concatenate([trace[:, :, 1:], new[:, :, None]], axis=-1)` is clean and JIT-compatible.

---

## Parameter Count Estimate

With default config (`d_model=128`, `d_input=64`, `memory_length=5`, `n_synch_out=16`, `memory_hidden_dims=2`):

| Component | Params |
|---|---|
| Backbone: Dense(obs→128) + Dense(128→128) | ~obs*128 + 128*128 ≈ 20-25K |
| Synapses: Dense(192→256) + Dense(128→256) | 192*256 + 128*256 ≈ 82K |
| NLM layer 1: SuperLinear(5→4, N=128) | 5*4*128 + 128*4 = 3K |
| NLM layer 2: SuperLinear(2→2, N=128) | 2*2*128 + 128*2 = 0.8K |
| Start traces (2x) | 128*5 * 2 = 1.3K |
| Decay params | 136 |
| Actor head: Dense(136→64) + Dense(64→n_actions) | ~9K |
| **CTM actor total** | **~115-120K** |
| GRU critic (unchanged) | ~same as before |

For comparison, the GRU actor has ~50-60K params, so the CTM actor is roughly 2x larger. This is expected and fine.

---

## Summary of Files to Create/Modify

| File | Action | Description |
|---|---|---|
| `smax_ctm/ctm_jax.py` | **CREATE** | `SuperLinear`, `NLM`, `CTMBackbone`, `Synapses`, `compute_synchronisation`, `CTMCell`, `ScannedCTM` |
| `smax_ctm/train_mappo_ctm.py` | **CREATE** (copy of `train_mappo_gru.py`) | `ActorCTM` replaces `ActorRNN`; add CTM config; adjust hidden state shapes |
