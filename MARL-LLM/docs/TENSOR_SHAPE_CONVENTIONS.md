# Tensor Shape Conventions

This document describes the tensor shape conventions used across the MARL-LLM codebase. Understanding these conventions is critical for debugging shape mismatches.

## Quick Reference

| Component | Actions Shape | Notes |
|-----------|---------------|-------|
| `DDPGAgent.step()` output | `(action_dim, n_agents)` = `(2, 30)` | Returns `action.t()` |
| `env.step()` input | `(n_agents, action_dim)` = `(30, 2)` | JAX env expects agents-first |
| `Buffer.push()` input | `(action_dim, n_agents)` = `(2, 30)` | Internally does `[:, index].T` |

## Detailed Flow

### 1. Policy Forward Pass (MADDPG)

```python
torch_agent_actions, _ = maddpg.step(obs, start_stop_num, explore=True)
```

- `maddpg.step()` calls `DDPGAgent.step()` for each agent group
- `DDPGAgent.step()` returns `action.t()` (transposed)
- Output: list of tensors, each with shape `(action_dim, n_agents)` = `(2, 30)`

### 2. Stacking Actions

```python
agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])
# OR for GPU:
agent_actions_gpu = torch.column_stack(torch_agent_actions)
```

- `column_stack` on a single `(2, 30)` tensor returns `(2, 30)`
- Result: `(action_dim, n_agents)` = `(2, 30)`

### 3. Environment Step

```python
# CORRECT: transpose for env.step
next_obs, rewards, dones, _, prior = env.step(agent_actions.T)
# OR for GPU tensors:
next_obs_gpu, rewards_gpu, dones_gpu, _, prior_gpu = env.step(agent_actions_gpu.t())
```

- `env.step()` expects `(n_agents, action_dim)` = `(30, 2)`
- **Must transpose** before passing to env

### 4. Buffer Storage

```python
# CORRECT: no transpose for buffer
agent_buffer[0].push(
    obs, agent_actions, rewards, next_obs, dones,
    start_stop_num[0], prior
)
```

- `Buffer.push()` expects `(action_dim, n_agents)` = `(2, 30)`
- Internally does `actions_orig[:, index].T` to get `(n_agents, action_dim)` per agent
- **Do NOT transpose** before pushing

## Common Mistakes

### ❌ Wrong: Missing transpose for env.step
```python
next_obs, rewards, dones, _, prior = env.step(agent_actions)  # WRONG!
```

### ❌ Wrong: Extra transpose for buffer
```python
agent_buffer[0].push(obs, agent_actions.T, ...)  # WRONG!
```

### ✅ Correct Pattern
```python
# Stack actions from policy
agent_actions = np.column_stack([ac.data.numpy() for ac in torch_agent_actions])
# Shape: (2, 30) = (action_dim, n_agents)

# Env step: transpose to (30, 2) = (n_agents, action_dim)
next_obs, rewards, dones, _, prior = env.step(agent_actions.T)

# Buffer: keep original (2, 30) shape
agent_buffer[0].push(obs, agent_actions, rewards, next_obs, dones, ...)
```

## GPU Training Pattern

For `train_assembly_jax_gpu.py`:

```python
# Policy step (GPU)
torch_agent_actions, _ = maddpg.step(obs_gpu, start_stop_num, explore=True)
agent_actions_gpu = torch.column_stack(torch_agent_actions)  # (2, N*n_a) on GPU

# Env step: transpose for JAX env
next_obs_gpu, rewards_gpu, dones_gpu, _, prior_gpu = env.step(agent_actions_gpu.t())

# Copy to CPU for buffer (no transpose!)
actions_cpu = agent_actions_gpu.cpu().numpy()  # already (2, N*n_a)
agent_buffer[0].push(obs_cpu, actions_cpu, rewards_cpu, next_obs_cpu, dones_cpu, ...)
```

## JAX Environment Internals

The JAX `AssemblyEnv.step()` method:
- Input: `actions` with shape `(n_agents, action_dim)` = `(30, 2)`
- Internally indexes as `actions[i, :]` for each agent
- Returns observations, rewards, dones, info, prior_actions

## Buffer Internals

`ReplayBuffer.push()` method:
- Input: `actions_orig` with shape `(action_dim, n_agents)` = `(2, 30)`
- For each agent `i`, stores `actions_orig[:, i].T` = `(action_dim,)` → `(1, action_dim)`
- This allows indexing by agent slice: `actions_orig[:, slice(0,30)]`

## JAX ↔ PyTorch Data Transfer

When transferring data between JAX (GPU) and PyTorch:

```python
# JAX array → PyTorch (read-only warning fix)
torch_obs = torch.from_numpy(np.asarray(jax_obs).copy())

# PyTorch → JAX (via DLPack for zero-copy on GPU)
jax_array = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
```

## Debugging Tips

1. **Print shapes** at each step:
   ```python
   print(f"torch_agent_actions[0].shape: {torch_agent_actions[0].shape}")
   print(f"agent_actions.shape: {agent_actions.shape}")
   ```

2. **Check env error messages**: Shape mismatches often cause JAX indexing errors

3. **Verify buffer storage**: Sample from buffer and check action shapes match expected

## Files Reference

| File | Purpose |
|------|---------|
| `train/train_assembly_jax.py` | CPU/JAX training loop |
| `train/train_assembly_jax_gpu.py` | GPU training with DLPack transfers |
| `algorithm/algorithms.py` | MADDPG implementation |
| `algorithm/agents.py` | DDPGAgent with `.step()` returning `action.t()` |
| `algorithm/utils.py` | ReplayBuffer with `.push()` expecting `(action_dim, n_agents)` |
