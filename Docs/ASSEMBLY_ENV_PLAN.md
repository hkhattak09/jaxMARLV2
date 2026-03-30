# Assembly Environment Implementation Plan

Base: `jaxmarl/environments/mpe/simple_spread.py` → new file `jaxmarl/environments/mpe/assembly.py`

---

## 1. State

`simple.py` State has: `p_pos`, `p_vel`, `c`, `done`, `step`

Add to State:
- `grid_center` — `[2, n_g]` target grid cell positions for the current episode
- `n_g` — number of grid cells (varies per shape)
- `shape_index` — which shape is active this episode

The shape data (all shapes from pickle) is loaded once at env init and stored as a static
array on the class, not in the JAX state. Only the per-episode selection goes in State.

---

## 2. Reset

**Current (`simple.py`):** random uniform positions for agents and landmarks.

**Needs to become:**
- Sample `shape_index` randomly from the loaded shapes
- Apply random rotation (`rand_angle ∈ [-π, π]`)
- Apply random scale (`shape_scale`, currently fixed at 1 but designed to vary)
- Apply random offset (`rand_target_offset`) to place shape anywhere in boundary
- Randomise agent starting positions uniformly in boundary
- Zero velocities

**JAX challenge:** pickle loading and variable-length `n_g` per shape cannot be dynamic
inside `jit`. Solution: pad all shapes to `n_g_max` at load time, mask out padding cells
in reward/obs with a validity mask.

---

## 3. Physics — what to replace in `_world_step`

`simple.py` has: action forces + soft contact forces + damping + velocity clipping.

**Remove:**
- Soft contact force between agents (`_get_collision_force` / `_apply_environment_force`) — replace with hard spring model below

**Add (port from C++):**

### Agent–agent spring force (k_ball)
```
delta = p_pos[i] - p_pos[j]
dist = ||delta||
overlap = 2*size_a - dist          # size_a = agent radius
force = k_ball * overlap * (delta/dist)   when overlap > 0, else 0
```

### Wall repulsion force (k_wall, c_wall)
Four walls: x_min, x_max, y_min, y_max
```
d_wall = distance from agent to wall
force = k_wall * (size_a - d_wall) - c_wall * v_normal    when d_wall < size_a
```
Must handle all 4 walls per agent, vmapped.

### Aerodynamic drag (c_aero)
```
f_drag = -c_aero * p_vel[i]
```
Add to total force before integration.

### Integration
Keep `_integrate_state` but use assembly params:
- `dt = 0.1`
- `mass = m_a = 1`
- `Vel_max = 0.8` (max speed clamp)
- No damping constant — drag is explicit via c_aero above, remove the `(1 - damping)` term

---

## 4. Observations — what to replace in `get_obs`

**Current (`simple_spread.py`):** all agents' relative positions + all landmark positions.

**Needs to become (topology-based, topo_nei_max=6):**

For each agent `i`:
1. Own velocity: `p_vel[i]` → `[2]`
2. K nearest neighbours (by distance), up to `topo_nei_max=6`:
   - relative position `p_pos[j] - p_pos[i]` → `[2]` each
   - relative velocity `p_vel[j] - p_vel[i]` → `[2]` each
   - pad with zeros if fewer than K neighbours in sensor range `d_sen`
3. Nearest grid cells within sensor range `d_sen` (up to some cap):
   - relative position to grid cell → `[2]` each
   - binary `in_flag` (is agent already occupying that cell)

**JAX approach:** sort all agents by distance, take top-K — fully differentiable with
`jnp.argsort`. Same for grid cells.

Obs dim per agent: `2 + topo_nei_max*4 + n_obs_grid*2`  (to be fixed at init time)

---

## 5. Reward — what to replace in `rewards`

**Current (`simple_spread.py`):** negative sum of distances to nearest unoccupied landmark.

**Needs to become (port from C++):**

### Coverage reward (positive)
```
for each grid cell g:
    find closest agent i
    if dist(i, g) < l_cell/2:
        reward += coverage_weight / n_g
```
Differentiable JAX version: soft assignment or hard `dist < threshold`.

### Penalty: agent–agent interaction
```
for each pair (i,j):
    if dist(i,j) < r_avoid:
        penalty += penalize_interaction_weight
```

### Penalty: boundary violation
```
for each agent i:
    if outside boundary:
        penalty += penalize_entering_weight
```

### Penalty: exploration (redundant coverage)
```
if two agents cover the same grid cell:
    penalty += penalize_exploration_weight
```

### Prior action regularisation (alpha term)
The existing code has `env.alpha` scaling a prior action penalty — this lives in the
training loop, not the env reward. Keep it in the training loop as-is.

---

## 6. Action Space

Keep continuous 2D: `Box(-1, 1, (2,))` — Cartesian acceleration `[ax, ay]`.
This matches `dynamics_mode = 'Cartesian'` and `(Acc_min, Acc_max) = (-1, 1)`.
Remove the discrete action path entirely for this env.

---

## 7. What does NOT need to change

- `MultiAgentEnv` base class — untouched
- `spaces.py` — untouched
- `step_env` structure in `simple.py` — keep the same flow: set_actions → world_step → rewards → get_obs → dones
- `jit` / `vmap` patterns — reuse exactly
- Registration in `registration.py` — just add `"assembly_v0"` entry

---

## 8. Data loading (pickle → JAX arrays)

At `__init__` time (outside jit):
```python
with open(results_file, 'rb') as f:
    data = pickle.load(f)

# Pad all shapes to n_g_max grid cells
self.all_grid_centers  # [num_shapes, 2, n_g_max]
self.all_valid_masks   # [num_shapes, n_g_max]  bool, True = real cell
self.all_l_cells       # [num_shapes]
self.num_shapes        # int
```

In `reset`, index into these with `shape_index` — all static-shape JAX ops.

---

## 9. File to create

`JaxMARL/jaxmarl/environments/mpe/assembly.py`

Export it from `mpe/__init__.py` and register it in `registration.py`.
