# JAX Environment Integration

Connects the JAX-based `AssemblyEnv` to the PyTorch MADDPG training loop as a
drop-in replacement for the C++ `AssemblySwarmEnv`.

---

## What was changed

| File | Change |
|---|---|
| `JaxMARL/jaxmarl/environments/mpe/assembly.py` | Added `robot_policy` + `_robot_policy_single` methods |
| `MARL-LLM/cus_gym/gym/wrappers/customized_envs/jax_assembly_wrapper.py` | **New file** — `JaxAssemblyAdapter` class |
| `MARL-LLM/cus_gym/gym/wrappers/__init__.py` | Added `JaxAssemblyAdapter` export |
| `MARL-LLM/marl_llm/train/train_assembly_jax.py` | **New file** — modified training script |

The original `train_assembly.py`, `assembly_wrapper.py`, and all algorithm/buffer
files are untouched.

---

## Prior action: `robot_policy` in `assembly.py`

### What it does

`robot_policy(state)` returns a `[n_a, 2]` array of prior actions computed
entirely on the GPU. It is a JAX port of the C++ `robotPolicy` /
`calculateActionPrior` functions in `AssemblyEnv.cpp`.

Each agent's prior action has three components (hardcoded gains matching C++):

```
total = attraction + repulsion + sync
total = clip(total, -1, 1)
```

| Component | Formula | Gain |
|---|---|---|
| Target attraction | unit vector toward nearest unoccupied grid cell | 2.0 |
| Neighbour repulsion | spring away from neighbours within `r_avoid` | 3.0 × (r_avoid/dist − 1) |
| Velocity sync | difference from mean topological-neighbour velocity | 2.0 |

"Topological neighbours" = top `topo_nei_max` (=6) nearest agents, same set used
for reward and observation — this is consistent with the C++ `neighbor_index`.

### Why it belongs in `assembly.py`

The prior policy reads directly from `AssemblyState` (positions, velocities,
grid centres). Keeping it in the same file as the environment means it has
access to all environment constants (`r_avoid`, `d_sen`, `topo_nei_max`, etc.)
without any extra plumbing. It also stays JIT-compilable and vmappable as a
method of `AssemblyEnv`.

---

## The adapter: `JaxAssemblyAdapter`

### Why a wrapper class instead of modifying the training loop

The MADDPG algorithm, replay buffer, and LLM regularisation code all depend on
a specific environment API:

```python
obs             = env.reset()          # (obs_dim, n_agents)
obs, rew, done, info, a_prior = env.step(actions)
env.alpha                              # regularisation weight
env.num_agents                         # total agent count
env.observation_space.shape[0]         # obs dim for network init
env.action_space.shape[0]              # action dim for network init
```

Rather than changing every call site, a single adapter class absorbs all the
JAX↔NumPy translation in one place.

### Parallel environments via `jax.vmap`

With `n_envs > 1`, the adapter constructs `jax.vmap(env.reset)` and
`jax.vmap(env.step_env)` at init time and compiles them with `jax.jit`. Every
call to `step()` then dispatches a single batched GPU kernel covering all `N`
environments simultaneously — the same compute cost as one env but with `N×`
the experience per wall-clock second.

### GPU–CPU transfer strategy

During rollout the PyTorch policy runs on CPU
(`maddpg.prep_rollouts(device='cpu')`). Therefore JAX results must cross the
PCIe bus once per step regardless. The adapter does this with `np.asarray()`,
which triggers a single D2H copy of the entire batch rather than one copy per
agent. The prior actions are computed on-device (inside `robot_policy`) before
the copy, so no second transfer is needed.

DLPack (zero-copy GPU↔GPU) was considered but ruled out for the initial
integration because the training loop explicitly moves networks to CPU for
rollout. A future optimisation could keep rollout on GPU and use DLPack, but
that would require changes to MADDPG — outside the minimal-change goal.

### PRNG key management

JAX environments are purely functional: `reset(key)` and `step_env(key, state,
actions)` consume a key and produce a new state. The adapter owns a single
`PRNGKey` and advances it with `jax.random.split` on every `reset` and `step`
call, so the caller never needs to manage keys.

---

## The training script: `train_assembly_jax.py`

The original `train_assembly.py` is preserved. The new file is a copy with five
targeted changes:

### 1. Env construction (3 lines replaced)

```python
# Before
scenario_name = 'AssemblySwarm-v0'
base_env = gym.make(scenario_name).unwrapped
env = AssemblySwarmWrapper(base_env, args)

# After
jax_env = AssemblyEnv(results_file=cfg.results_file, n_a=cfg.n_a)
env = JaxAssemblyAdapter(jax_env, n_envs=cfg.n_rollout_threads, seed=cfg.seed)
```

`n_rollout_threads` in the config now directly controls how many parallel
environments run. Setting it to 1 gives behaviour identical to the original.

### 2. Action shape fix before buffer push (1 line added)

```python
agent_actions     = np.column_stack(...)   # (n_a, 2)  ← from maddpg.step
agent_actions_buf = agent_actions.T        # (2, n_a)  ← buffer expects dim-first
agent_buffer[0].push(obs, agent_actions_buf, ...)
```

`maddpg.step` returns actions with shape `(n_agents, action_dim)`. The replay
buffer's `push` method does `actions_orig[:, index].T` internally, which assumes
the input is `(action_dim, n_agents)`. The original code had a latent shape bug
here (invisible only if `n_agents == action_dim`, i.e. 2 agents). The transpose
fixes it.

### 3. Alpha update (1 line)

```python
# Before
env.env.alpha = 0.1   # reached through the C++ env

# After
env.alpha = 0.1       # set directly on the adapter
```

### What was not changed

- `MADDPG` class and all update logic
- `ReplayBufferAgent` and sampling code
- LLM regularisation term (`0.3 * alpha * MSE(policy, prior)`)
- Logging, checkpointing, noise scheduling

---

## Memory layout reference

| Array | Shape | Owner |
|---|---|---|
| `obs` | `(obs_dim, N*n_a)` | env → buffer |
| `rew` | `(1, N*n_a)` | env → buffer |
| `done` | `(1, N*n_a)` | env → buffer |
| `agent_actions` | `(N*n_a, 2)` | maddpg → env |
| `agent_actions_buf` | `(2, N*n_a)` | env → buffer |
| `a_prior` | `(2, N*n_a)` | env → buffer |

All arrays are NumPy on CPU at the buffer boundary. JAX DeviceArrays live only
inside the adapter until `np.asarray()` is called.

---

## Memory note for N > 1

The buffer allocates `buffer_length × N × n_a` rows. With the defaults
(`buffer_length=2e4`, `n_a=30`) and `N=10` that is 6 M rows × 192 floats ≈
4.6 GB. Reduce `buffer_length` proportionally when increasing `n_rollout_threads`.
