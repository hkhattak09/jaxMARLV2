# MARL-LLM Data Flow and Tensor Shapes

**Purpose**: Complete reference for tensor shapes and data transformations throughout the pipeline. Critical for debugging shape mismatches and understanding GPU-optimized data flow.

## Overview

Data flows through two main frameworks with GPU-optimized zero-copy transfers:
1. **JAX** (GPU) — Environment simulation
2. **PyTorch** (GPU) — Neural network training
3. **DLPack** — Zero-copy GPU tensor sharing (no CPU intermediate)

**Key Feature**: All data stays on GPU during rollout. Single bulk GPU→CPU transfer only at episode end for buffer storage.

---

## Shape Convention Summary

### Column-Major vs Row-Major

**JAX Environment** (internal):
- Agents: Row-major `[n_a, feature_dim]`
- Grid: Column-major `[2, n_g_max]` (x/y in rows)

**Adapter Output** (GPU tensors):
- Observations: **Column-major** `(obs_dim, n_agents)` — features in rows, agents in columns — `torch.cuda.FloatTensor`
- Actions (input): Row-major `(n_agents, 2)` — `torch.cuda.FloatTensor`
- Actions (output/prior): **Column-major** `(2, n_agents)` — action dims in rows — `torch.cuda.FloatTensor`
- Rewards: `(1, n_agents)` — `torch.cuda.FloatTensor`
- Dones: `(1, n_agents)` — `torch.cuda.BoolTensor`

**Replay Buffer Storage**:
- Internally transposes to row-major: `(n_agents, feature_dim)` on CPU
- Samples as row-major: `(batch_size, feature_dim)`
- **Critical**: Buffer stores on CPU (bulk transfer from GPU at episode end)

**PyTorch Networks**:
- Standard row-major: `(batch_size, feature_dim)`

---

## 1. Environment Reset

### Single Environment (n_envs=1, n_a=30)

```
JAX AssemblyEnv.reset(key)
    ↓
obs_dict: Dict[str, Array]
  {"agent_0": [192], "agent_1": [192], ..., "agent_29": [192]}  (JAX GPU)
    ↓
JaxAssemblyAdapterGPUGPU._obs_dict_to_torch()
    ↓
JIT-compiled dict → array conversion (on GPU):
  jnp.stack([obs_dict[a] for a in agents], axis=0).T
    [30, 192].T → [192, 30] (JAX GPU)
    ↓
DLPack zero-copy transfer (no data movement):
  torch.utils.dlpack.from_dlpack(jax_array)
    ↓
OUTPUT: torch.cuda.FloatTensor shape (192, 30)
```

**Performance**: ~0.1ms for DLPack conversion vs ~1-2ms for GPU→CPU→GPU path

### Parallel Environments (n_envs=4, n_a=30)

```
JAX vmap(env.reset)(keys[4])
    ↓
obs_dict: Dict[str, Array]
  {"agent_0": [4, 192], "agent_1": [4, 192], ...}
    ↓
JaxAssemblyAdapterGPU (JIT-compiled conversion)
    ↓
Stack and reshape:
  jnp.stack([obs_dict[a] for a in agents], axis=1)  # [4, 30, 192]
  .reshape(4*30, 192).T  # [120, 192].T → [192, 120]
    ↓
JAX → NumPy transfer
    ↓
OUTPUT: np.ndarray shape (192, 120) dtype=float32
```

**Memory Layout**: Agents [0:30] from env 0, [30:60] from env 1, etc.

---

## 2. Action Selection (MADDPG Forward Pass - GPU)

### From Observation to Action (All on GPU)

```
INPUT: obs_gpu shape (192, 30)  # torch.cuda.FloatTensor from environment
    ↓
MADDPG.step(obs_gpu, [slice(0, 30)], explore=True)
  # Networks already on GPU, no conversion needed
    ↓
Select agents via slice:
  observations[:, slice(0, 30)]  # (192, 30) GPU tensor
    ↓
Transpose for agent input:
  obs[:, slice].t()  # (30, 192) — agents in rows now, still GPU
    ↓
DDPGAgent.step(obs_t, explore=True)
    ├─ Forward through policy network (on GPU):
    │    MLPNetwork: (30, 192) → (30, 2)
    │    action = policy(obs)  # (30, 2) GPU tensor
    ├─ Add exploration noise if explore=True (on GPU):
    │    action += noise or random
    │    action = action.clamp(-1, 1)
    └─ Transpose output:
         return action.t()  # (2, 30) GPU tensor ← transposed!
    ↓
MADDPG.step returns:
  actions: List[Tensor] = [Tensor(2, 30)]  # torch.cuda.FloatTensor
  log_pis: List[Tensor] = [Tensor(1, 30)]  # torch.cuda.FloatTensor
```

**Critical**: All operations stay on GPU. No .numpy() conversions.

### Action Format for Environment

```
PyTorch GPU actions: List[torch.cuda.FloatTensor(2, 30)]
    ↓
Stack (stays on GPU):
  actions_gpu = torch.column_stack(actions)  # (2, 30) GPU tensor
    ↓
Transpose and detach for env.step:
  env.step(actions_gpu.t().detach())  # (30, 2) GPU tensor
  # .detach() required for DLPack (can't export tensors with gradients)
```
actions, _ = maddpg.step(torch_obs, [slice(0, 30)], explore=True)
# actions is a List with one element (one agent group)
# actions[0] has shape (2, 30) from DDPGAgent.step transpose

# Extract numpy
agent_actions = np.column_stack([ac.data.numpy() for ac in actions])
# ac.data.numpy() for ac in [Tensor(2, 30)] gives [array(2, 30)]
# column_stack on [array(2, 30)] → still (2, 30)
# This is because column_stack treats 1D arrays as columns, but here we have 2D

# Let me check what column_stack actually does with (2, 30):
# np.column_stack([array([[a,b,c,...], [d,e,f,...]])]) 
# Returns the same array (2, 30) - no op for single input

# So agent_actions is (2, 30)

# For env.step, we need (n_a, 2):
env.step(agent_actions.T)  # (30, 2)
```

Actually, wait. Let me look at the actual training code more carefully:

```python
# From train_assembly_jax_gpu.py line 213-220:
torch_agent_actions, _ = maddpg.step(torch_obs, start_stop_num, explore=True)

agent_actions = np.column_stack(
    [ac.data.numpy() for ac in torch_agent_actions]
)  # (N*n_a, 2)

next_obs, rewards, dones, _, agent_actions_prior = env.step(agent_actions.T)

agent_actions_buf = agent_actions  # already (2, N*n_a) for buffer
```

So the comment says `agent_actions` is `(N*n_a, 2)` after column_stack, then it's transposed to `.T` which makes it `(2, N*n_a)` for env.step.

But wait, `DDPGAgent.step` returns `action.t()` which is `(action_dim, batch_size)` = `(2, n_a)`. So `torch_agent_actions[0]` is shape `(2, 30)`.

`column_stack` on a single `(2, 30)` array would give... let me think. Actually, `column_stack` with a list of arrays stacks them horizontally. With a single `(2, 30)` array, it would just return `(2, 30)`.

But the comment says `(N*n_a, 2)`. I think there's confusion here. Let me look at what `.data.numpy()` does on a `(2, 30)` tensor:

```python
ac = Tensor([[1,2,3,...,30], [31,32,33,...,60]])  # (2, 30)
ac.data.numpy() → array([[1,2,3,...,30], [31,32,33,...,60]])  # (2, 30)
```

So `np.column_stack([array(2, 30)])` should return `(2, 30)`.

Hmm, but the comment says `(N*n_a, 2)`. Let me check if maybe each `ac` in the list is already a column vector? No wait, the code shows:

```python
agent_actions = np.column_stack(
    [ac.data.numpy() for ac in torch_agent_actions]
)
```

If there's one agent group and `torch_agent_actions` has one element of shape `(2, 30)`, then we have:
- List comprehension: `[ac.data.numpy()]` = `[array(2, 30)]`
- `column_stack([array(2, 30)])` = `array(2, 30)`

So `agent_actions` is `(2, 30)`, and `.T` makes it `(30, 2)`.

The comment `# (N*n_a, 2)` in the code seems to be describing what shape AFTER transpose? Or maybe it's wrong? Let me just document what actually happens:

---

### Correct Action Flow

```
maddpg.step() returns:
  actions: [Tensor(2, 30)]  # action_dim × n_agents
    ↓
Convert to NumPy:
  [ac.data.numpy() for ac in actions]  # [array(2, 30)]
    ↓
Stack (single array, no change):
  agent_actions = np.column_stack(...)  # (2, 30)
    ↓
Transpose for env.step:
  env.step(agent_actions.T)  # (30, 2) — row-major expected by adapter
```

---

## 3. Environment Step

### Adapter Processing (n_envs=1)

```
INPUT: actions shape (30, 2) NumPy
    ↓
Convert to JAX:
  actions_jax = jnp.asarray(actions)  # (30, 2)
    ↓
Build actions_dict:
  {a: actions_jax[i] for i, a in enumerate(agents)}
  {"agent_0": [2], "agent_1": [2], ..., "agent_29": [2]}
    ↓
JAX env.step_env(key, state, actions_dict)
    ↓
Returns (obs_dict, new_state, rew_dict, done_dict, prior)
  obs_dict: {"agent_i": [192]}
  rew_dict: {"agent_i": scalar}
  done_dict: {"agent_i": bool}
  prior: [30, 2]  # JAX array
    ↓
JIT-compiled conversion:
  obs = jnp.stack([obs_dict[a] for a in agents], axis=0).T  # (30, 192).T → (192, 30)
  rew = jnp.stack([rew_dict[a] for a in agents])[None, :]   # (1, 30)
  done = jnp.stack([done_dict[a] for a in agents])[None, :] # (1, 30)
  prior_out = prior.T  # (30, 2).T → (2, 30)
    ↓
JAX → NumPy transfer (single D2H call)
    ↓
OUTPUT:
  obs: (192, 30) float32
  rew: (1, 30) float32
  done: (1, 30) bool
  prior: (2, 30) float32
```

### Parallel Environments (n_envs=4)

```
INPUT: actions shape (120, 2)  # 4 envs × 30 agents
    ↓
Reshape to per-env:
  actions_jax.reshape(4, 30, 2)  # [n_envs, n_a, 2]
    ↓
Build actions_dict:
  {a: actions_reshaped[:, i, :] for i, a in enumerate(agents)}
  {"agent_0": [4, 2], "agent_1": [4, 2], ...}
    ↓
vmap(env.step_env)(keys[4], states, actions_dict)
    ↓
Returns batched outputs:
  obs_dict: {"agent_i": [4, 192]}
  rew_dict: {"agent_i": [4]}
  done_dict: {"agent_i": [4]}
  prior: [4, 30, 2]
    ↓
JIT-compiled reshape:
  obs = stack → (4, 30, 192) → reshape(120, 192).T → (192, 120)
  rew = stack → (4, 30) → reshape(1, 120)
  done = stack → (4, 30) → reshape(1, 120)
  prior = reshape(120, 2).T → (2, 120)
    ↓
OUTPUT:
  obs: (192, 120) float32
  rew: (1, 120) float32
  done: (1, 120) bool
  prior: (2, 120) float32
```

---

## 4. Replay Buffer Storage

### Push to Buffer

```
INPUT:
  obs: (192, 30)
  actions: (2, 30)  # As prepared above (agent_actions before transpose)
  rewards: (1, 30)
  next_obs: (192, 30)
  dones: (1, 30)
  prior: (2, 30)
  index: slice(0, 30)
    ↓
ReplayBufferAgent.push(obs, actions, rewards, next_obs, dones, index, prior)
    ↓
Select and transpose:
  observations = observations_orig[:, index].T    # (192, 30)[:, 0:30].T → (30, 192)
  actions = actions_orig[:, index].T              # (2, 30)[:, 0:30].T → (30, 2)
  rewards = rewards_orig[:, index].T              # (1, 30)[:, 0:30].T → (30, 1)
  next_observations = next_observations_orig[:, index].T  # (30, 192)
  dones = dones_orig[:, index].T                  # (30, 1)
  actions_prior = actions_prior_orig[:, index].T  # (30, 2)
    ↓
Write to buffer arrays:
  obs_buffs[curr_i:curr_i+30] = observations      # (30, 192)
  ac_buffs[curr_i:curr_i+30] = actions            # (30, 2)
  rew_buffs[curr_i:curr_i+30] = rewards           # (30, 1)
  ...
    ↓
Increment indices:
  curr_i += 30
  filled_i = min(filled_i + 30, max_steps * num_agents)
```

**Key**: Buffer stores data in row-major format `(n_samples, feature_dim)`.

---

## 5. Sampling from Buffer

### Sample Batch for Training

```
ReplayBufferAgent.sample(N=512, to_gpu=True, norm_rews=True)
    ↓
Random sampling:
  inds = np.random.choice(filled_i, size=N, replace=True)  # [512]
    ↓
Gather samples:
  obs = obs_buffs[inds]            # (512, 192)
  acs = ac_buffs[inds]             # (512, 2)
  rews = rew_buffs[inds]           # (512, 1)
  next_obs = next_obs_buffs[inds]  # (512, 192)
  dones = done_buffs[inds]         # (512, 1)
  acs_prior = ac_prior_buffs[inds] # (512, 2)
    ↓
Normalize rewards (if norm_rews=True):
  rews_mean = rews.mean()
  rews_std = rews.std() + 1e-6
  rews = (rews - rews_mean) / rews_std
    ↓
Convert to PyTorch:
  if to_gpu:
    obs = torch.Tensor(obs).cuda()
    acs = torch.Tensor(acs).cuda()
    rews = torch.Tensor(rews).cuda()
    ...
  else:
    obs = torch.Tensor(obs)
    ...
    ↓
OUTPUT: (obs, acs, rews, next_obs, dones, acs_prior)
  All shapes: (512, feature_dim) for respective features
  All types: torch.Tensor (on GPU if to_gpu=True)
```

---

## 6. MADDPG Update

### Critic Update

```
INPUT: (obs, acs, rews, next_obs, dones, acs_prior) all shape (512, ...)
    ↓
Target actions:
  all_trgt_acs = target_policy(next_obs)  # (512, 2)
    ↓
Target Q-value:
  trgt_vf_in = torch.cat([next_obs, all_trgt_acs], dim=1)  # (512, 192+2) = (512, 194)
  target_value = target_critic(trgt_vf_in)  # (512, 1)
  target_value = rews + gamma * target_value * (1 - dones)  # (512, 1)
    ↓
Current Q-value:
  vf_in = torch.cat([obs, acs], dim=1)  # (512, 194)
  actual_value = critic(vf_in)  # (512, 1)
    ↓
Loss:
  vf_loss = MSE(actual_value, target_value.detach())  # scalar
    ↓
Backward and update critic
```

### Actor Update

```
Current policy:
  curr_pol_out = policy(obs)  # (512, 2)
    ↓
Q-value with current policy:
  vf_in = torch.cat([obs, curr_pol_out], dim=1)  # (512, 194)
  q_val = critic(vf_in)  # (512, 1)
    ↓
Policy loss (maximize Q):
  pol_loss = -q_val.mean()  # scalar
    ↓
Regularization (if acs_prior provided):
  # Filter out near-zero prior actions (likely invalid)
  mask = (acs_prior.abs() < 1e-2).all(dim=1)  # (512,)
  valid_mask = ~mask
  filtered_pol = curr_pol_out[valid_mask]     # (N_valid, 2)
  filtered_prior = acs_prior[valid_mask]       # (N_valid, 2)
  
  reg_loss = MSE(filtered_pol, filtered_prior)  # scalar
  pol_loss = pol_loss + 0.3 * alpha * reg_loss
    ↓
Backward and update actor
```

---

## 7. Complete Training Loop Data Flow

```
┌──────────────────────────────────────────────────────────┐
│ 1. RESET ENVIRONMENT                                     │
│    JAX: reset() → obs_dict                               │
│    Adapter: stack + transpose → (192, 30) NumPy          │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│ 2. SELECT ACTIONS (Episode Rollout)                      │
│    NumPy (192, 30) → PyTorch (192, 30)                   │
│    MADDPG: transpose → (30, 192) → policy → (30, 2)      │
│    Output: transpose → (2, 30) NumPy                     │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│ 3. ENVIRONMENT STEP                                      │
│    Actions: (2, 30) → transpose → (30, 2)                │
│    JAX: step() → obs/rew/done/prior dicts                │
│    Adapter: stack + transpose → all (*, 30) NumPy        │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│ 4. STORE IN BUFFER                                       │
│    All inputs (*, 30) column-major                       │
│    Transpose to (30, *) row-major                        │
│    Store 30 experiences sequentially                     │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│ 5. SAMPLE BATCH (after buffer filled)                    │
│    Sample 512 random indices                             │
│    Gather → all (512, *) row-major NumPy                 │
│    Convert → PyTorch, move to GPU                        │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│ 6. UPDATE NETWORKS                                       │
│    Critic: cat(obs, acs) → (512, 194) → Q-value          │
│    Actor: obs → (512, 192) → policy → (512, 2)           │
│    Losses computed, backprop, optimizers step            │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│ 7. SOFT UPDATE TARGET NETWORKS                           │
│    θ_target ← τ*θ + (1-τ)*θ_target                       │
└──────────────────────────────────────────────────────────┘
```

---

## 8. Memory Consumption

### Buffer Memory (n_rollout_threads=1, n_a=30, buffer_length=20000)

```
total_length = buffer_length × n_rollout_threads × n_a
             = 20000 × 1 × 30 = 600,000 experiences

obs_buffs:       (600000, 192) × 4 bytes = 460.8 MB
ac_buffs:        (600000, 2) × 4 bytes   = 4.8 MB
ac_prior_buffs:  (600000, 2) × 4 bytes   = 4.8 MB
rew_buffs:       (600000, 1) × 4 bytes   = 2.4 MB
next_obs_buffs:  (600000, 192) × 4 bytes = 460.8 MB
done_buffs:      (600000, 1) × 1 byte    = 0.6 MB

Total ≈ 934 MB
```

### With Parallel Environments (n_rollout_threads=4)

```
total_length = 20000 × 4 × 30 = 2,400,000

Total ≈ 3.7 GB
```

**Recommendation**: For GPU training with 16GB RAM, keep `buffer_length × n_rollout_threads ≤ 40000`.

---

## 9. Common Shape Errors and Fixes

### Error 1: "Expected (obs_dim, n_agents) but got (n_agents, obs_dim)"

**Cause**: Forgot column-major convention for environment API.

**Fix**: Ensure observations are `.T` transposed when needed:
```python
obs = env.reset()  # Already (192, 30) from adapter
torch_obs = torch.Tensor(obs)  # Keep as (192, 30)
```

### Error 2: "RuntimeError: mat1 and mat2 shapes cannot be multiplied"

**Cause**: Wrong input shape to neural network.

**Fix**: Check network expects `(batch_size, input_dim)`:
```python
# DDPGAgent.step expects (n_agents, obs_dim)
obs_for_agent = observations[:, slice].t()  # (192, 30).t() → (30, 192)
```

### Error 3: "Buffer push expects (2, n_agents) for actions"

**Cause**: Actions not in correct format.

**Fix**:
```python
# maddpg.step returns (action_dim, n_agents)
actions, _ = maddpg.step(obs, [slice(0, 30)])  # List[(2, 30)]
agent_actions = np.column_stack([ac.numpy() for ac in actions])  # (2, 30)
# Buffer expects (2, n_a) ✓
buffer.push(obs, agent_actions, ...)
```

### Error 4: "Invalid action shape for env.step"

**Cause**: Environment expects `(n_agents, action_dim)` row-major.

**Fix**:
```python
# Transpose before passing to env
env.step(agent_actions.T)  # (2, 30).T → (30, 2) ✓
```

---

## 10. Device Transfers

### JAX (GPU) ↔ NumPy (CPU)

**Frequency**: Every `env.reset()` and `env.step()` call.

**Cost**: ~0.5-2ms for 30 agents on PCIe 3.0 × 16.

**Optimization**: Adapter uses JIT-compiled batch conversion to minimize Python overhead.

**Code**:
```python
# Efficient: Single JIT-compiled transfer
obs_jax, rew_jax, done_jax, prior_jax = self._jit_convert(
    obs_dict, rew_dict, done_dict, a_prior_jax
)
# Single D2H call
obs_np, rew_np, done_np, prior_np = map(np.asarray, [obs_jax, rew_jax, done_jax, prior_jax])
```

### PyTorch (CPU) ↔ PyTorch (GPU)

**Frequency**: 
- Rollout: Networks on CPU, no transfer needed
- Training: Data moved to GPU during `buffer.sample(to_gpu=True)`
- Update: Networks on GPU, data already there

**Cost**: ~1-5ms for batch_size=512.

**Optimization**: Move networks in bulk via `prep_training()` and `prep_rollouts()`.

---

## 11. Quick Reference

| Operation | Input Shape | Output Shape | Notes |
|-----------|-------------|--------------|-------|
| `env.reset()` | — | `(192, 30)` | Column-major obs |
| `env.step(actions)` | `(30, 2)` | obs `(192, 30)`, rew `(1, 30)`, prior `(2, 30)` | Row-major in, column-major out |
| `maddpg.step(obs, ...)` | `(192, 30)` | `List[(2, 30)]` | Column-major in, column-major out |
| `buffer.push(obs, acs, ...)` | obs `(192, 30)`, acs `(2, 30)` | — | Column-major inputs, stores row-major |
| `buffer.sample(N)` | — | All `(512, *)` | Row-major outputs |
| `policy(obs)` | `(batch, 192)` | `(batch, 2)` | Standard PyTorch row-major |
| `critic(obs_acs)` | `(batch, 194)` | `(batch, 1)` | Concatenated input |

---

**Next**: See TRAINING_PIPELINE.md for the complete training loop, QUICK_REFERENCE.md for function signatures.
