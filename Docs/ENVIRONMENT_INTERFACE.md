# MARL-LLM Environment Interface

**Purpose**: Detailed documentation of the JAX-based assembly environment and its PyTorch adapter.

## Overview

The environment simulates a swarm of agents learning to assemble into target shapes (e.g., stars, letters). It consists of:
1. **AssemblyEnv** (JAX) — Core physics and reward computation on GPU
2. **JaxAssemblyAdapterGPU** — Wrapper providing gym-like interface with GPU-optimized data flow using DLPack

**Key Design**: JAX runs parallel environments via `jax.vmap`, returns GPU arrays that are converted to PyTorch CUDA tensors via DLPack zero-copy. All data stays on GPU during rollout.

---

## 1. AssemblyEnv (JAX Core)

**Location**: `JaxMARL/jaxmarl/environments/mpe/assembly.py`

### Purpose
Simulates multi-agent swarm physics with:
- Agent-agent collision forces
- Boundary wall forces
- Rewards for assembling target shapes
- Reynolds flocking prior actions (for regularization)

### Initialization

```python
AssemblyEnv(
    results_file: str,           # Path to fig/results.pkl
    n_a: int = 30,               # Number of agents
    topo_nei_max: int = 6,       # Max neighbors in observation (K)
    num_obs_grid_max: int = 80,  # Max target cells in observation (M)
    grid_obs_fraction: float = None,  # Alternative to num_obs_grid_max: fraction of shape cells visible
    dt: float = 0.1,             # Physics timestep (4 substeps at dt/4=0.025 internally)
    vel_max: float = 0.8,        # Max agent velocity
    k_ball: float = 2000.0,      # Agent-agent repulsion stiffness (was 30 — increased to prevent tunneling)
    k_wall: float = 100.0,       # Wall repulsion stiffness
    c_wall: float = 5.0,         # Wall damping coefficient
    size_a: float = 0.035,       # Agent body radius. Physical contact: dist < 2*size_a = 0.07
    d_sen: float = 0.4,          # Sensing radius (CLI: --d_sen). Controls observability AND sensing_coverage metric.
    r_avoid: float = 0.10,       # Personal space radius (CLI: --r_avoid). Spacing violation: dist < 2*r_avoid = 0.20
    boundary_half: float = 2.4,  # Half-width of square boundary
    max_steps: int = 200,        # Episode length
)
```

**Key parameter changes vs original:**
- `k_ball`: `30.0` → `2000.0` — prevents agent tunneling. With k=30 agents at vel_max=0.8 could pass through each other in a single dt=0.1 step. Now runs 4 substeps at dt_sub=0.025.
- `r_avoid`: previously computed dynamically per shape (`sqrt(4*n_g/(n_a*π)) * l_cell` → gave 0.29 on small shapes, geometrically infeasible). Now a fixed constructor arg defaulting to 0.10, exposed via `--r_avoid` CLI flag.
- `d_sen`: previously hardcoded; now exposed as `--d_sen` CLI flag.

**Key Attributes**:
- `obs_dim`: Observation dimension = `4*(topo_nei_max+1) + 4 + 2*num_obs_grid_max`
  - Default: `4*7 + 4 + 2*80 = 192`

---

## Observability Design — Original Setup and Why It's Being Changed

### Original setup (num_obs_grid_max=80)

In the original configuration each agent observes up to **80 nearest target cells**
(160 of its 192 observation dimensions). The total number of target cells per shape is
typically 80-120, meaning each agent could see essentially the **entire shape** at all times.

This made the task structurally easy: each agent independently knew where all target cells
were and could navigate to the nearest uncovered one without coordinating with teammates.
The problem reduces to *local reactive navigation on a fully visible map* — a task that
can almost be hand-coded without learning. Under this setup varying the number of visible
neighbors (K) had little effect because the hard question ("where do I go?") was already
answered by the shape observation. This is why CTM and MLP performed similarly across all
K values despite the visual difference in agent spacing.

### New direction — shape partial observability

We are reducing `num_obs_grid_max` (M) to make the shape **genuinely unknown** to each agent.
At M=10-15, an agent sees only the nearest 10-15 target cells — a small local patch.
It no longer knows where most of the shape is.

Now the coordination problem is real:
- Agents must spread across unknown territory without a global map
- Flocking-like behaviour (spread out, maintain separation) becomes the correct inductive bias
- The Reynolds prior (cohesion, alignment, separation) is now genuinely useful — it encodes
  exactly the behaviour needed when you can't see the shape
- A prior-seeded CTM that starts iterative computation from the Reynolds prior has a
  structural advantage over a single-pass MLP that has no access to physics knowledge at
  execution time

### Two axes of partial observability

| Parameter | Controls | Current default | Experiment target |
|---|---|---|---|
| `topo_nei_max` (K) | Teammate visibility | 6 | 3 |
| `num_obs_grid_max` (M) | Shape visibility | 80 | 10-15 |

Reducing both creates the regime where architecture differences actually matter.
`topo_nei_max` is already a CLI parameter. `num_obs_grid_max` needs to be wired to cfg.

---
- `agents`: List of agent IDs `["agent_0", "agent_1", ..., "agent_{n_a-1}"]`
- `num_shapes`: Number of target shapes loaded from `results.pkl`

### State Representation

**AssemblyState** (dataclass):
```python
@chex.dataclass
class AssemblyState:
    p_pos: chex.Array        # [n_a, 2]      Agent positions (x, y)
    p_vel: chex.Array        # [n_a, 2]      Agent velocities (vx, vy)
    grid_center: chex.Array  # [2, n_g_max]  Target cell positions (padded)
    valid_mask: chex.Array   # [n_g_max]     True = real cell, False = padding
    l_cell: float            # Grid cell side length
    shape_index: int         # Which target shape is active
    done: chex.Array         # [n_a]         Episode termination flags
    step: int                # Current timestep
```

**Key Insight**: `grid_center` is padded to `n_g_max` (max cells across all shapes) for fixed array size in JIT compilation.

### Observation Space

**Shape**: `(obs_dim,)` per agent (default 192)

**Components** (concatenated):
1. **Self state** (4): `[pos_x, pos_y, vel_x, vel_y]`
2. **Neighbors** (4 × topo_nei_max = 24): For each of K nearest neighbors:
   - Relative position: `[dx, dy]`
   - Relative velocity: `[dvx, dvy]`
   - Padded with zeros if fewer than K neighbors
3. **Target info** (4):
   - Relative position to nearest target cell: `[dx_tgt, dy_tgt]`
   - Relative velocity to target: `[0, 0]` (target is static)
4. **Nearby target cells** (2 × num_obs_grid_max = 160):
   - Up to M nearest target cells (sorted by distance)
   - Each: `[dx_cell, dy_cell]`
   - Padded with zeros if fewer than M cells

**Observation Construction** (optimized):
```python
cached = self._compute_cached_distances(state)  # Compute distances once
obs = self._get_obs_fast(state, cached)         # Use cached data
```

**Cached distances** eliminate redundant argsort calls (4× speedup):
- Agent-to-agent distances + K nearest neighbors
- Agent-to-grid distances + nearest cell per agent

### Action Space

**Shape**: `(2,)` per agent — continuous 2D force in [-1, 1]

**Physical Interpretation**:
- Action is **directly added to force** (sensitivity = 1)
- `F_total = u + F_collision + F_wall + F_damping`
- Velocity update: `v ← v + F * dt` (assuming mass = 1)

**Clipping**: Actions clipped to `[-1, 1]` before physics

### Reset

```python
obs_dict, state = env.reset(key)
```

**Process**:
1. Sample random shape from `all_grid_centers` (domain randomization)
2. Apply random rotation (angle ∈ [-π, π])
3. Apply random offset (within boundary - 1.0 margin)
4. Initialize agent positions:
   - 50% uniform random in boundary
   - 50% clustered around random center
5. Initialize velocities: uniform in [-0.5, 0.5]

**Output**:
- `obs_dict`: Dict mapping `agent_i → obs_array` shape `(obs_dim,)`
- `state`: AssemblyState with initialized values

**Evaluation Variant**:
```python
obs_dict, state = env.reset_eval(key, shape_index=0)
```
- No rotation, no offset (shape centered at origin)
- For reproducible evaluation

### Step

```python
obs_dict, new_state, rew_dict, done_dict, prior = env.step_env(key, state, actions_dict)
```

**Inputs**:
- `key`: PRNG key (for future stochasticity, currently unused)
- `state`: Current AssemblyState
- `actions_dict`: Dict mapping `agent_i → action_array` shape `(2,)`

**Outputs**:
- `obs_dict`: Dict of next observations `{agent_i: obs}`
- `new_state`: Updated AssemblyState
- `rew_dict`: Dict of rewards `{agent_i: reward_scalar}`
- `done_dict`: Dict of done flags `{agent_i: bool, "__all__": bool}`
- `prior`: Prior actions from Reynolds flocking, shape `[n_a, 2]`

**Process** (single timestep):
1. Clip actions to [-1, 1]
2. **Physics update** (`_world_step`):
   - Compute agent-agent collision forces
   - Compute boundary wall forces (spring + damping)
   - Integrate: `v ← v + F*dt`, `p ← p + v*dt`
   - Clip velocity to `[-vel_max, vel_max]`
3. **Compute cached distances** (once per step)
4. **Compute observations** (using cached distances)
5. **Compute rewards** (using cached distances)
6. **Compute prior actions** (Reynolds flocking, using cached distances)
7. Check done (step >= max_steps)

### Physics Details

#### Agent-Agent Collision
```python
F_collision = -k_ball * overlap * direction_away
```
- `overlap = 2*size_a - distance` (positive if colliding)
- Only applied when `distance < 2*size_a`
- Direction: away from other agent

#### Boundary Walls
Wall order: `[x_min, y_max, x_max, y_min]`

**Spring force** (when penetrating):
```python
F_spring = k_wall * penetration_depth * direction_inward
```

**Damping force** (velocity-dependent):
```python
F_damping = -c_wall * velocity_component_toward_wall
```

### Reward Function

**Location**: `_rewards_vectorized(state, cached)` (vectorised) + `_reward_single(i, state)` (reference)

**Previous reward (replaced):**
```python
# Old: binary AND of three conditions, r_avoid dynamically set (0.29 on small shapes)
# is_collision = dist < r_avoid  (wrong threshold — treated radius as diameter)
# reward = where(in_flag & ~is_collision & is_uniform, 1.0, 0.0)
# Problem: r_avoid=0.29 fired constantly (shape too small for 24 agents at that spacing)
# → reward near-zero the entire training run, no learning signal
```

**Current reward structure:**
```python
# Variable definitions:
too_close   = any neighbor dist < 2 * r_avoid (= 0.20)  # spacing policy
is_touching = any agent dist < 2 * size_a (= 0.07)       # physical body contact
n_touching  = count of physically touching neighbors for agent i

# is_uniform fix (saturated case):
# Old: is_uniform = in_flag & any_sensed & (v_exp_norm < 0.05)  ← wrong: penalised agents in full regions
# New:
is_uniform = jnp.where(any_sensed, v_exp_norm < 0.05, True)
# If unoccupied cells visible: must be centred among them
# If none visible (saturated region): already doing job correctly → uniform=True

# Reward:
reward_i = 0.1  × in_flag                              # stepping stone: always positive inside shape
         + 0.9  × (in_flag & ~too_close & is_uniform)  # full reward: inside, spaced, centred
         - 0.07 × n_touching_i                         # physical contact penalty
```

**Stepping stone table:**
| Situation | Reward |
|---|---|
| Outside shape | 0.0 |
| Inside, physically touching 1 neighbour | 0.1 − 0.07 = **0.03** |
| Inside, conditions not fully met | **0.1** |
| Inside, all conditions met | **1.0** |
| Inside, all met + touching 1 | 1.0 − 0.07 = **0.93** |

**Key design rationale:**
- Stepping stone (`+0.1` for `in_flag`) creates gradient at boundary from episode 1 — agents always prefer inside
- `~too_close` (dist < 2×r_avoid) handles spacing policy. `is_touching` (dist < 2×size_a) handles physical contact — two distinct signals with separate weights
- `is_uniform` saturated fix: agents holding a fully-covered patch no longer penalised for "not finding uncovered cells nearby"
- Reynolds flocking prior in actor update reinforces entry further (cohesion pulls outliers toward shape)

### Prior Actions (Reynolds Flocking)

**Location**: `_robot_policy_fast(state, cached)` (vectorised) + `_robot_policy_single(i, state)` (reference)

**Purpose**: Provide physics-based prior action for regularization in MADDPG update.

**Algorithm** (Reynolds rules):
1. **Cohesion**: Move toward average position of nearby grid cells
2. **Alignment**: Match velocities with neighbours
3. **Separation**: Repel from neighbours within personal space

**Updated separation threshold (Group 3 r_avoid fix):**
```python
# Old: rep_factor = where(nei_dists < r_avoid, 3*(r_avoid/dist - 1), 0)
# New: consistent with new r_avoid definition (radius not diameter):
rep_factor = where(nei_dists < 2*r_avoid, 3*(2*r_avoid/dist - 1), 0)
```
Repulsion fires at the same physical spacing (center-to-center 0.20) as `too_close` in the reward — policy and prior use identical thresholds.

**Output**: Shape `[n_a, 2]` — desired actions for each agent

**Usage in Training**:
```python
pol_loss = -Q(s, π(s)) + 0.3 * alpha * MSE(π(s), prior(s))
```
- Bootstraps learning from Reynolds flocking behaviour
- `alpha` fixed at 0.1 (low — prior as regularizer, not behavioural cloning target)

### Evaluation Metrics

**Metrics redesigned in the physics/reward overhaul.** Summary of what changed:

| Old name | New name | Why changed |
|---|---|---|
| `coverage_rate` | **`sensing_coverage`** | Old used `a2g_dist < r_avoid/2` coverage radius → could never reach 1.0 with r_avoid=0.10. New uses `d_sen` (sensing range). |
| `collision_rate` + `count_collisions` | **`r_avoid_violation_count`** | Old used wrong threshold (`dist < r_avoid` not `< 2*r_avoid`) and double-counted. New: pairwise upper-triangle, no double-counting. |
| `coverage_efficiency` | **removed** | Algebraically identical to old `coverage_rate` — same value, different label. Redundant. |

---

**`sensing_coverage`** *(replaces `coverage_rate`)*:
```python
def sensing_coverage(self, state) -> float:
    # Fraction of valid shape cells observed by at least one agent
    # A cell is "sensed" if any agent centre is within d_sen of it
    cell_sensed = jnp.any(a2g_dist < self.d_sen, axis=0)  # [n_g_max]
    return jnp.sum(cell_sensed & valid_mask) / jnp.sum(valid_mask)
```
- No r_avoid dependency — reaches 1.0 when agents cover the shape
- Meaningful for partial observability: captures what fraction of shape the team can collectively perceive
- Wrapper: `env.sensing_coverage()` (no state arg)

**`r_avoid_violation_count`** *(replaces `collision_rate` + `count_collisions`)*:
```python
def r_avoid_violation_count(self, state) -> float:
    # Number of unique agent pairs with centre-to-centre dist < 2 * r_avoid
    # Upper-triangle only — no double counting (same pattern as springboard_collision_count)
    upper = jnp.triu(jnp.ones((n_a, n_a), dtype=bool), k=1)
    is_violation = (dists < 2.0 * self.r_avoid) & upper
    return jnp.sum(is_violation.astype(jnp.float32))
```
- Three mutually-close agents = 3 pairs = 3 events
- Accumulated per step via `r_avoid_violation_count_jax()` (no CPU sync); synced once at episode end
- Wrapper: `env.r_avoid_violation_count()` and `env.r_avoid_violation_count_jax()`

**`distribution_uniformity`** *(unchanged)*:
```python
# Uniformity of nearest-neighbour distances (CoV-based)
# 1 / (1 + std/mean). Range [0,1]. Higher = more uniform spacing.
```

**`voronoi_based_uniformity`** *(unchanged)*:
```python
# CoV of Voronoi cell counts per agent. Range [0,1]. Higher = more balanced territory.
```

**`mean_neighbor_distance`** *(unchanged)*:
```python
# Mean nearest-neighbour distance across all agents.
# Measures absolute spacing magnitude — complements uniformity metrics.
```

**`springboard_collision_count`** *(unchanged)*:
```python
# Unique pairs with dist < 2*size_a = 0.07 (physical body contact).
# Accumulated per step via springboard_collision_count_jax() (no CPU sync).
```

**Note:** All metrics available on `AssemblyEnv` (takes `state` arg) and `JaxAssemblyAdapterGPU`
(uses `self._states`). All appear in: periodic rolling stats, `run_eval`, `run_final_eval`,
and standalone `eval_shapes.py`.

---

## 2. JaxAssemblyAdapterGPU (GPU-Optimized PyTorch Bridge)

**Location**: `MARL-LLM/cus_gym/gym/wrappers/customized_envs/jax_assembly_wrapper_gpu.py`

### Purpose
Wraps JAX `AssemblyEnv` to expose gym-like API with GPU-optimized data flow. Uses DLPack for zero-copy tensor sharing between JAX and PyTorch. Handles:
- **Zero-copy GPU transfers**: DLPack shares memory between JAX and PyTorch on same device
- **PyTorch CUDA tensors**: Returns GPU tensors directly (no CPU intermediate)
- Parallel environment batching via `jax.vmap`
- Observation/action reshaping to match MADDPG expectations

### Initialization

```python
JaxAssemblyAdapterGPU(
    jax_env: AssemblyEnv,    # Constructed AssemblyEnv instance
    n_envs: int = 1,         # Number of parallel environments
    seed: int = 0,           # Random seed
    alpha: float = 1.0,      # Regularization coefficient
)
```

**Key Attributes**:
- `n_a`: Total agents = `n_envs × jax_env.n_a`
- `num_agents`: Alias for `n_a`
- `observation_space`: Dummy space with shape `(obs_dim, num_agents)`
- `action_space`: Dummy space with shape `(2, num_agents)`
- `agents`: List of dummy agent objects (for compatibility)
- `alpha`: Regularization coefficient (read by MADDPG update)

**JIT Compilation**:
- If `n_envs=1`: JIT single env functions + array-based step (no dict overhead)
- If `n_envs>1`: JIT vmapped functions for parallel execution

**DLPack Zero-Copy**:
- JAX GPU arrays converted to PyTorch CUDA tensors with no intermediate copy
- ~0.1ms overhead vs ~1-2ms for GPU→CPU→GPU path
- Requires both frameworks on same CUDA device

### Data Layout Conventions (GPU Tensors)

**Critical**: The adapter returns **PyTorch CUDA tensors** in column-major format for agents:
- Observations: `(obs_dim, n_envs*n_a)` — **features in rows, agents in columns** — `torch.cuda.FloatTensor`
- Actions (input to step): `(n_envs*n_a, 2)` — standard row-major — `torch.cuda.FloatTensor`
- Actions (output from step, prior): `(2, n_envs*n_a)` — **action dims in rows** — `torch.cuda.FloatTensor`
- Rewards: `(1, n_envs*n_a)` — `torch.cuda.FloatTensor`
- Dones: `(1, n_envs*n_a)` — `torch.cuda.BoolTensor`

**Why GPU?** All tensors stay on GPU during rollout for 15-25% speedup. Single bulk GPU→CPU transfer only at episode end for buffer storage.

**Why column-major?** Matches original C++ AssemblySwarmEnv API for drop-in replacement.

### Reset

```python
obs_gpu = adapter.reset()  # Returns (obs_dim, n_envs*n_a) torch.cuda.FloatTensor
```

**Process** (n_envs > 1):
1. Split PRNG key into `n_envs` keys
2. Call `jax.vmap(env.reset)(keys)` — parallel reset
3. Convert obs_dict to stacked array: `[n_envs, n_a, obs_dim] → [obs_dim, n_envs*n_a]`
4. JAX → PyTorch via DLPack (zero-copy GPU transfer)

**Example**:
```python
adapter = JaxAssemblyAdapterGPU(env, n_envs=4, seed=42)
obs_gpu = adapter.reset()  # Shape: (192, 120) torch.cuda.FloatTensor for 4 envs × 30 agents
```

### Step

```python
obs_gpu, rew_gpu, done_gpu, info, a_prior_gpu = adapter.step(actions_gpu)
```

**Inputs**:
- `actions_gpu`: torch.cuda.FloatTensor shape `(n_envs*n_a, 2)` — **must be .detach()'ed**

**Outputs** (all GPU tensors):
- `obs_gpu`: `(obs_dim, n_envs*n_a)` torch.cuda.FloatTensor
- `rew_gpu`: `(1, n_envs*n_a)` torch.cuda.FloatTensor
- `done_gpu`: `(1, n_envs*n_a)` torch.cuda.BoolTensor
- `info`: `{}` (empty dict, for API compatibility)
- `a_prior_gpu`: `(2, n_envs*n_a)` torch.cuda.FloatTensor — **prior actions for regularization**

**Process** (n_envs > 1):
1. Reshape actions: `(n_envs*n_a, 2) → (n_envs, n_a, 2)`
2. Call `jax.vmap(env.step_env_array)(keys, states, actions)` — array-based step, no dict overhead
3. JIT-compiled conversion: arrays → reshaped (on GPU)
4. JAX → PyTorch via DLPack (zero-copy, no PCIe transfer)

**Performance**: Prior actions computed on-device during step. Entire step loop runs on GPU with zero CPU involvement.

### Parallel Environment Batching

**Memory Layout** (n_envs=4, n_a=30):
```
Env 0: agents [0:30]
Env 1: agents [30:60]
Env 2: agents [60:90]
Env 3: agents [90:120]

obs shape: (192, 120)
  ↓
MADDPG sees 120 agents total
  ↓
Replay buffer stores 120 agents per timestep
```

**Key Insight**: Buffer size scales with `n_envs`:
```
buffer_memory = buffer_length × n_envs × n_a × obs_dim × 4 bytes
              = 20000 × 4 × 30 × 192 × 4 bytes
              ≈ 1.8 GB for default config
```

### Render (Optional)

```python
adapter.render()  # Print current state info to console
```

---

## 3. Environment Interaction Pattern (GPU-Optimized)

### Typical Training Loop

```python
# 1. Initialize
jax_env = AssemblyEnv(results_file='fig/results.pkl', n_a=30)
env = JaxAssemblyAdapterGPU(jax_env, n_envs=1, seed=42, alpha=1.0)

# 2. Reset
obs_gpu = env.reset()  # (192, 30) torch.cuda.FloatTensor

# 3. Episode loop - ALL ON GPU
for t in range(200):
    # Get actions from MADDPG (networks on GPU)
    actions, _ = maddpg.step(obs_gpu, [slice(0, 30)], explore=True)
    # actions[0] shape: (2, 30) torch.cuda.FloatTensor (already transposed)
    
    # Stack and transpose for environment
    actions_gpu = torch.column_stack(actions)  # (2, 30) GPU tensor
    
    # Step environment (MUST detach for DLPack)
    next_obs_gpu, rewards_gpu, dones_gpu, _, prior_gpu = env.step(actions_gpu.t().detach())
    # All outputs are torch.cuda tensors: obs (192, 30), rew (1, 30), prior (2, 30)
    
    # Accumulate on GPU (no transfer yet)
    obs_list.append(obs_gpu)
    actions_list.append(actions_gpu)
    rewards_list.append(rewards_gpu)
    # ... etc
    
    obs_gpu = next_obs_gpu  # Stay on GPU

# 4. Single bulk GPU→CPU transfer at episode end
obs_batch = torch.stack(obs_list).cpu().numpy()       # (200, 192, 30)
actions_batch = torch.stack(actions_list).cpu().numpy()  # (200, 2, 30)
rewards_batch = torch.stack(rewards_list).cpu().numpy()  # (200, 1, 30)
# ... etc

# 5. Store in buffer (now on CPU)
for t in range(200):
    buffer.push(obs_batch[t], actions_batch[t], rewards_batch[t], 
                next_obs_batch[t], dones_batch[t], slice(0, 30), prior_batch[t])
```

**Key Points**:
- **No .numpy() conversions during rollout** — all GPU tensors
- **Must .detach()** actions before env.step() (DLPack requirement)
- **Accumulate on GPU** in lists
- **Single bulk transfer** after full episode
- **15-25% faster** than CPU-intermediated version

---

## 4. Configuration and Shapes

### Target Shapes

**Source**: `fig/results.pkl`

**Created by**: `cfg/assembly_cfg.py::process_image()`

**Contents**:
```python
results = {
    'l_cell': [0.061, 0.058, ...],           # Grid cell sizes (scaled)
    'grid_coords': [array([n_g, 2]), ...],   # List of target cell coordinates
    'binary_image': [...],                   # Processed images (not used in JAX)
    'shape_bound_points': [...]              # Bounding boxes (not used)
}
```

**Loading**:
```python
with open(results_file, 'rb') as f:
    loaded = pickle.load(f)
# Padded to [num_shapes, 2, n_g_max] for JAX
```

### Observation Dimension Calculation

```python
obs_dim = 4 * (topo_nei_max + 1) + 4 + 2 * num_obs_grid_max
        = 4 * (6 + 1) + 4 + 2 * 80
        = 28 + 4 + 160
        = 192
```

**Components**:
- Self state: 4
- Neighbors: 4 × 6 = 24
- Target nearest: 4
- Target cells: 2 × 80 = 160

---

## 5. Common Issues and Debugging

### Issue 1: "RuntimeError: Can't export tensor that requires grad"

**Cause**: Trying to pass tensor with gradients to env.step() which uses DLPack

**Solution**: Always `.detach()` before env.step():
```python
env.step(actions_gpu.t().detach())  # ← .detach() is required!
```

### Issue 2: All tensors on CPU instead of GPU

**Cause**: Wrong adapter imported or device not set correctly

**Solution**: 
```python
# Correct
from gym.wrappers.customized_envs.jax_assembly_wrapper_gpu import JaxAssemblyAdapterGPU
env = JaxAssemblyAdapterGPU(jax_env, n_envs=1, seed=42)
obs_gpu = env.reset()  # torch.cuda.FloatTensor

# Check
print(obs_gpu.device)  # Should print: cuda:0
```

### Issue 3: DLPack error "arrays must be on the same device"

**Cause**: JAX and PyTorch not on same GPU

**Solution**:
```bash
# Set both to use cuda:0
export CUDA_VISIBLE_DEVICES=0
# Check JAX sees GPU
python -c "import jax; print(jax.devices())"
# Check PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue 2: Prior Actions All Zero

**Cause**: Reynolds policy only activates when agents are near grid cells.

**Solution**: Normal behavior early in training. Prior becomes more useful as agents approach target.

### Issue 3: Rewards Not Increasing

**Check**:
1. Are agents reaching grid cells? (in_flag counts)
2. Is `r_avoid` too strict? (check agent spacing)
3. Are penalties dominating? (check penalty scales)

### Issue 4: JAX Out of Memory

**Cause**: Too many parallel environments or too large buffer.

**Solution**:
```python
# Reduce JAX memory allocation
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.15'

# Or reduce n_envs
n_rollout_threads = 1  # Start with 1, scale up if memory allows
```

---

## 6. Extending the Environment

### Adding New Target Shapes

1. Create image file (black shape on white background)
2. Run `process_image(image_path)` in `cfg/assembly_cfg.py`
3. Save updated `results.pkl`
4. Environment automatically loads all shapes at initialization

### Modifying Reward Function

**Location**: `JaxMARL/jaxmarl/environments/mpe/assembly.py::_rewards_fast`

**Example**: Add bonus for uniform spacing:
```python
# Compute spacing variance
neighbor_dists_mean = jnp.mean(cached.nei_dists[:, :3])  # Avg of 3 nearest
uniformity_bonus = 0.1 * (1.0 - spacing_variance)
reward += uniformity_bonus
```

### Changing Physics Parameters

**Tunable in init**:
- `k_ball`: Agent repulsion strength (higher = harder collisions)
- `k_wall`: Wall repulsion strength
- `c_wall`: Wall damping (higher = less bouncy)
- `vel_max`: Speed limit
- `dt`: Integration timestep (smaller = more accurate, slower)

---

**Next**: See DATA_FLOW.md for detailed tensor transformations, TRAINING_PIPELINE.md for full training loop.
