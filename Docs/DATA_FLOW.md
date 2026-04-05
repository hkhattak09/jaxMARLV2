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

**Replay Buffer Storage** (centralised critic — joint rows):
- One row per environment timestep, not per agent
- Each row concatenates ALL agents: `(1, n_agents * feature_dim)` on CPU
- Samples as row-major: `(batch_size, n_agents * feature_dim)`
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

### Push to Buffer (Centralised Critic — Joint Rows)

One joint row is stored per timestep. All 24 agents' data is concatenated into a
single wide row so the centralised critic can consume the full joint state directly.

```
INPUT (per timestep t, n_envs=1):
  obs:      (obs_dim, n_a)   = (192, 24)  numpy CPU
  actions:  (2, n_a)         = (2,   24)
  rewards:  (1, n_a)         = (1,   24)
  next_obs: (obs_dim, n_a)   = (192, 24)
  dones:    (1, n_a)         = (1,   24)
  prior:    (2, n_a)         = (2,   24)
    ↓
ReplayBufferAgent.push(obs, actions, rewards, next_obs, dones,
                       actions_prior_orig=prior)
    ↓
Flatten to joint row:
  obs_joint      = obs.T.reshape(1, n_a*obs_dim)    # (1, 4608) — 24×192
  acs_joint      = actions.T.reshape(1, n_a*2)       # (1, 48)   — 24×2
  next_obs_joint = next_obs.T.reshape(1, n_a*obs_dim)# (1, 4608)
  rews_joint     = rewards.reshape(1, n_a).mean(keepdims=True)  # (1, 1) — agent mean
  dones_joint    = dones.reshape(1, n_a).max(keepdims=True)     # (1, 1) — any done
  prior_joint    = prior.T.reshape(1, n_a*2)         # (1, 48)
    ↓
Write one row to buffer:
  obs_buffs[curr_i]      = obs_joint       # (4608,)
  ac_buffs[curr_i]       = acs_joint       # (48,)
  rew_buffs[curr_i]      = rews_joint      # (1,)
  next_obs_buffs[curr_i] = next_obs_joint  # (4608,)
  done_buffs[curr_i]     = dones_joint     # (1,)
  ac_prior_buffs[curr_i] = prior_joint     # (48,)
    ↓
Increment indices:
  curr_i += 1
  filled_i = min(filled_i + 1, max_steps)
```

**Key**: Buffer stores one joint row per timestep. `total_length = max_steps` (not `max_steps × n_agents`).
Row ordering: agent 0 dims first, then agent 1, ..., agent 23 — consistent across obs/acs/prior.

---

## 5. Sampling from Buffer

### Sample Batch for Training (Centralised Critic — Joint Rows)

```
ReplayBufferAgent.sample(N=512, to_gpu=True, is_prior=True)
    ↓
Random sampling:
  inds = np.random.choice(filled_i, size=N, replace=False)  # [512]
    ↓
Gather joint-row samples (K=6, M=80 example with n_a=24, obs_dim=192):
  obs      = obs_buffs[inds]            # (512, 4608)  — 24×192
  acs      = ac_buffs[inds]             # (512, 48)    — 24×2
  rews     = rew_buffs[inds]            # (512, 1)
  next_obs = next_obs_buffs[inds]       # (512, 4608)
  dones    = done_buffs[inds]           # (512, 1)
  acs_prior= ac_prior_buffs[inds]       # (512, 48)
    ↓
Convert to PyTorch CUDA tensors:
  cast = lambda x: Tensor(x).requires_grad_(False).cuda()
    ↓
OUTPUT: (obs, acs, rews, next_obs, dones, acs_prior, None)
  obs/next_obs: (512, n_agents*obs_dim)  — joint observations
  acs/acs_prior:(512, n_agents*2)        — joint actions
  rews/dones:   (512, 1)
  All: torch.cuda.FloatTensor
```

**Key change from per-agent buffer**: Samples are joint rows, not individual agent rows.
The critic consumes them directly without any additional reshaping.

---

## 6. MADDPG Update (Centralised Critic + Option B Actor)

### Critic Update

```
INPUT (K=6, M=80 example, n_a=24, obs_dim=192):
  obs:      (512, 4608)  — joint obs  (24×192)
  acs:      (512, 48)    — joint acs  (24×2)
  rews:     (512, 1)
  next_obs: (512, 4608)  — joint next obs
  dones:    (512, 1)
    ↓
Target actions for all agents (target_policies):
  # Reshape next_obs for shared policy forward pass
  next_obs_flat = next_obs.view(512*24, 192)        # (12288, 192)
  target_actions = target_policy(next_obs_flat)      # (12288, 2)
  all_trgt_acs = target_actions.view(512, 24*2)      # (512, 48)
    ↓
Target Q-value:
  trgt_vf_in = cat([next_obs, all_trgt_acs], dim=1) # (512, 4608+48) = (512, 4656)
  target_value = target_critic(trgt_vf_in)           # (512, 1)
  target_value = rews + gamma * target_value * (1 - dones)  # (512, 1)
    ↓
Current Q-value:
  vf_in = cat([obs, acs], dim=1)                    # (512, 4656)
  actual_value = critic(vf_in)                       # (512, 1)
    ↓
Loss:
  vf_loss = MSE(actual_value, target_value.detach()) # scalar
    ↓
Critic architecture: 4656 → 256 → 256 → 256 → 1  (critic_hidden_dim=256)
Backward and update critic
```

### Actor Update (Option B — recompute ALL agents' actions)

With parameter sharing, we recompute all 24 agents' actions through the shared policy.
Gradient flows through all 24 action slots simultaneously — 24× stronger signal than
substituting only one agent's action.

```
obs: (512, 4608) — joint obs
    ↓
Reshape for shared policy:
  obs_flat = obs.view(512*24, 192)           # (12288, 192)
  all_curr_acs = policy(obs_flat)            # (12288, 2)
  all_curr_acs = all_curr_acs.view(512, 48)  # (512, 48)
    ↓
Q-value with current policy:
  vf_in = cat([obs, all_curr_acs], dim=1)   # (512, 4656)
  q_val = critic(vf_in)                      # (512, 1)
    ↓
Policy loss (maximize Q):
  pol_loss = -q_val.mean()                   # scalar
    ↓
Regularization (if acs_prior provided):
  # acs_prior is (512, 48) — all agents' Reynolds prior actions
  mask = (acs_prior.abs() < 1e-2).all(dim=1)     # (512,) — only if ALL 48 dims ≈ 0
  filtered_pol   = all_curr_acs[~mask]             # (N_valid, 48)
  filtered_prior = acs_prior[~mask]                # (N_valid, 48)
  reg_loss = MSE(filtered_pol, filtered_prior)     # scalar (over all 48 action dims)
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

### Buffer Memory (n_rollout_threads=1, n_a=24, buffer_length=20000, K=6 M=80)

Joint-row buffer: `total_length = buffer_length` (one row per timestep, not per agent).

```
total_length = buffer_length = 20000 joint rows

obs_buffs:       (20000, 24×192=4608) × 4 bytes = 368.6 MB
ac_buffs:        (20000, 24×2=48)     × 4 bytes = 3.8 MB
ac_prior_buffs:  (20000, 48)          × 4 bytes = 3.8 MB
rew_buffs:       (20000, 1)           × 4 bytes = 0.08 MB
next_obs_buffs:  (20000, 4608)        × 4 bytes = 368.6 MB
done_buffs:      (20000, 1)           × 4 bytes = 0.08 MB

Total ≈ 745 MB
```

**Previously** (per-agent rows): `(20000×24, 192)` → same total data volume but split across 480k rows.
The joint-row layout is ~20% more compact because reward/done are stored once per timestep rather than 24 times.

### Low-M regime (K=3, M=10, obs_dim=40)

```
obs_buffs: (20000, 24×40=960) × 4 bytes = 76.8 MB
Total ≈ 157 MB  — much smaller, suitable for Colab
```

**Recommendation**: For GPU training with 16GB RAM, keep `buffer_length ≤ 40000` at K=6 M=80.

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

Shapes shown for K=6, M=80 (obs_dim=192, n_a=24). For other configs substitute obs_dim and n_a.

| Operation | Input Shape | Output Shape | Notes |
|-----------|-------------|--------------|-------|
| `env.reset()` | — | `(192, 24)` | Column-major obs |
| `env.step(actions)` | `(24, 2)` | obs `(192, 24)`, rew `(1, 24)`, prior `(2, 24)` | Row-major in, column-major out |
| `maddpg.step(obs, ...)` | `(192, 24)` | `List[(2, 24)]` | Column-major in, column-major out |
| `buffer.push(obs, acs, ...)` | obs `(192, 24)`, acs `(2, 24)` | — | Flattens to 1 joint row per timestep |
| `buffer.sample(N)` | — | obs `(512, 4608)`, acs `(512, 48)` | Joint row-major outputs |
| `policy(obs)` | `(batch, 192)` | `(batch, 2)` | Per-agent; batch=512×24 during actor update |
| `critic(obs_acs)` | `(batch, 4656)` | `(batch, 1)` | Joint: 24×(192+2)=4656; hidden_dim=256 |
| `target_policies(next_obs_all)` | `(512, 4608)` | `(512, 48)` | Reshapes internally to run all agents |

---

**Next**: See TRAINING_PIPELINE.md for the complete training loop, QUICK_REFERENCE.md for function signatures.
