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
    topo_nei_max: int = 6,       # Max neighbors in observation
    num_obs_grid_max: int = 80,  # Max target cells in observation
    dt: float = 0.1,             # Physics timestep
    vel_max: float = 0.8,        # Max agent velocity
    k_ball: float = 30.0,        # Agent-agent repulsion stiffness
    k_wall: float = 100.0,       # Wall repulsion stiffness
    c_wall: float = 5.0,         # Wall damping coefficient
    size_a: float = 0.035,       # Agent radius
    d_sen: float = 0.4,          # Neighbor sensing range
    boundary_half: float = 2.4,  # Half-width of square boundary
    max_steps: int = 200,        # Episode length
)
```

**Key Attributes**:
- `obs_dim`: Observation dimension = `4*(topo_nei_max+1) + 4 + 2*num_obs_grid_max`
  - Default: `4*7 + 4 + 2*80 = 192`
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

**Location**: `_rewards_fast(state, cached)`

**Components** (per agent):

1. **Assembly reward** (r_assem):
   - If agent inside a grid cell: `+1.0`
   - Else: `0.0`
   - Uses `in_flag` from cached distances (threshold = √2 * l_cell / 2)

2. **Avoidance penalty** (penalize_interaction):
   - If distance to any neighbor < `r_avoid`: `-0.2`
   - Encourages spacing within grid cells
   - `r_avoid` computed from min shape size

3. **Entering penalty** (penalize_entering):
   - If agent moves from outside → inside occupied cell: `-0.3`
   - Discourages pushing into already-filled cells
   - Tracked via `in_flag` comparison with previous state

4. **Exploration bonus** (penalize_exploration):
   - Small negative reward for being far from target: `-0.05 * normalized_distance`
   - Encourages agents to approach target region early

**Total reward**: Sum of components, clipped to reasonable range

**Example**:
```python
# Agent inside cell, no neighbors nearby, not entering occupied cell
reward = 1.0 + 0.0 - 0.0 - 0.01 = 0.99
```

### Prior Actions (Reynolds Flocking)

**Location**: `_robot_policy_fast(state, cached)`

**Purpose**: Provide expert behavior for regularization in MADDPG update.

**Algorithm** (simplified Reynolds rules):
1. **Cohesion**: Move toward average position of nearby grid cells
2. **Alignment**: Match velocities with neighbors
3. **Separation**: Avoid crowding neighbors

**Output**: Shape `[n_a, 2]` — desired actions for each agent

**Usage in Training**:
```python
pol_loss = -Q(s, π(s)) + 0.3 * alpha * MSE(π(s), prior(s))
```
- Helps bootstrap learning with reasonable swarm behavior
- `alpha` gradually decreased to allow learned policy to dominate

### Evaluation Metrics

**Coverage Rate**:
```python
def coverage_rate(self) -> float:
    # Fraction of target cells occupied by at least one agent
    # Uses in_flag: which agents are in which cells
    n_occupied = count_unique_occupied_cells(state)
    n_total = count_valid_cells(state.valid_mask)
    return n_occupied / n_total
```

**Distribution Uniformity**:
```python
def distribution_uniformity(self) -> float:
    # Uniformity of nearest-neighbor distances across agents
    # Uses coefficient of variation: uniformity = 1 / (1 + std/mean)
    # Range: [0, 1], higher = more uniform spacing
    min_dists = [distance_to_nearest_neighbor(agent) for agent in agents]
    return 1.0 / (1.0 + std(min_dists) / mean(min_dists))
```

**Voronoi Uniformity**:
```python
def voronoi_based_uniformity(self) -> float:
    # Uniformity of Voronoi cell counts across agents
    # Each grid cell is assigned to its nearest agent
    # Uses coefficient of variation: uniformity = 1 / (1 + std/mean)
    # Range: [0, 1], higher = more balanced cell distribution
    cell_counts = [count_cells_assigned_to(agent) for agent in agents]
    return 1.0 / (1.0 + std(cell_counts) / mean(cell_counts))
```

**Mean Neighbor Distance** *(new)*:
```python
def mean_neighbor_distance(self) -> float:
    # Mean nearest-neighbour distance across all agents
    # Measures absolute magnitude of spacing — not just uniformity
    # Higher = agents more spread out (less clustering)
    # Unlike distribution_uniformity, sensitive to overall scale of separation
    min_dists = [distance_to_nearest_neighbor(agent) for agent in agents]
    return mean(min_dists)
```

**Collision Rate** *(new)*:
```python
def collision_rate(self) -> float:
    # Fraction of agents currently in collision (any neighbour within r_avoid)
    # Range: [0, 1]. 0 = no collisions, 1 = all agents colliding
    # Direct measure of physical stacking — cleaner than collision count
    in_collision = [any(dist < r_avoid for dist in neighbor_dists(agent)) for agent in agents]
    return mean(in_collision)
```

**Coverage Efficiency** *(new)*:
```python
def coverage_efficiency(self) -> float:
    # Cells covered / n_agents, normalised by ideal (n_cells / n_agents)
    # Algebraically equivalent to coverage_rate — same number, cleaner label
    # Value of 1.0 = agents spread perfectly with no cell-sharing
    # Useful label when comparing CTM vs MLP stacking behaviour
    return n_occupied / n_valid_cells  # = coverage_rate
```

**Note:** All new metrics are available on both `AssemblyEnv` (JAX, takes `state` arg) and
`JaxAssemblyAdapterGPU` (wrapper, uses `self._states`). All appear in periodic eval logs,
final eval per-shape output, final eval summary, and `eval_shapes.py`.

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
