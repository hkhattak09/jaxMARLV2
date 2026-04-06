# Reward & Physics Redesign Decisions

## Context

Identified from GIF visualization at episode 50 (MLP, K=6, M=80) that the task as currently
configured is geometrically ill-posed. The discussion below documents the root causes and
agreed fixes before implementation.

---

## Problem 1 — Physics: Agent Tunneling

### Root Cause
`k_ball=30` with `dt=0.1` cannot prevent tunneling. Two agents closing head-on at `vel_max=0.8`
move `1.6 × 0.1 = 0.16` units in one step — more than the full agent diameter (0.07). The spring
force at first contact is ~0 (overlap ≈ 0), so agents pass straight through each other.

### Fix — Substeps (IMPLEMENTED)
- `k_ball`: `30` → `2000`
- `_world_step`: runs 4 substeps at `dt_sub = dt/4 = 0.025`
- Stability: `k × dt_sub² = 2000 × 0.000625 = 1.25 < 2` ✓
- Max distance per substep: `0.8 × 0.025 = 0.02` < diameter `0.07` — no tunneling
- Residual worst-case penetration: ~0.02 units (29% of diameter), physically acceptable
- vel_max clamp applied every substep — no velocity explosion
- JIT unrolls the Python loop: zero runtime overhead

---

## Problem 2 — r_avoid: Too Large, Geometrically Infeasible

### Root Cause
Current formula:
```python
self.r_avoid = round(np.sqrt(4.0 * min_n_g / (n_a * np.pi)) * min_l_cell, 2)
```
This computes the ideal agent spacing assuming perfect circular tiling of the *smallest* shape.
For n_a=24 on the T-shape (~1.2 sq units), this gives `r_avoid = 0.29`.

Feasibility check — area needed for 24 agents at r_avoid=0.29 spacing:
```
24 × π × 0.29² ≈ 6.35 sq units  >>  shape area ~1.2 sq units
```
The shape is ~5× too small. Agents are **always** within r_avoid of each other just to fit
inside the shape. The inter-agent penalty fires constantly regardless of policy quality.

Maximum feasible r_avoid for 24 agents in ~1.2 sq units:
```
r_avoid_max = sqrt(1.2 / (24 × π)) ≈ 0.13
```

### Fix — Hardcode r_avoid
- Remove the dynamic formula entirely
- Expose `r_avoid` as a constructor argument with a fixed sensible default
- Target value: **~0.10** (coverage radius = 0.05, close to physical body size 0.035)
- `r_avoid` is fixed across all shapes and all episodes — training stability preserved
- Per-episode r_avoid was considered and rejected: different spacing thresholds per shape
  would require different policies per shape, causing training instability

### Why r_avoid is not d_sen
- `d_sen` is not used in the r_avoid formula (confirmed by reading the code)
- They are independent: `d_sen` controls observability, `r_avoid` controls spacing/coverage
- Do not conflate them

---

## Problem 3 — Reward Structure: Binary AND, Impossible Uniformity

### Current reward
```python
return jnp.where(in_flag & ~is_collision & is_uniform, 1.0, 0.0)
```

Three binary conditions, all must hold simultaneously. Zero gradient between "bad" and
"less bad". Because `is_collision` fires almost always (due to r_avoid being too large),
reward is ~0 constantly — DDPG receives almost no signal.

### What each condition is actually trying to express

**`in_flag`**: agent is inside the shape. Correct and necessary.

**`~is_collision`** (really: `~is_too_close`): maintain minimum inter-agent spacing so agents
don't cluster. The name "collision" is misleading — this is a spacing constraint, not a
physical collision. Physical collision is already handled by `k_ball` spring physics.
With r_avoid fixed to a feasible value, this condition becomes meaningful.

**`is_uniform`**: agent is positioned such that the weighted centroid of sensed *unoccupied*
cells around it is near zero (`v_exp_norm < 0.05`). This means the agent is near the medial
axis of uncovered territory — equidistant from uncovered cells in all directions. This is
the right spatial signal for good coverage.

### The uniformity over-packed problem

`is_uniform = in_flag & any_sensed & (v_exp_norm < 0.05)`

When all nearby cells are already claimed by other agents, `any_sensed = False` →
`is_uniform = False` → reward = 0. But an agent in a fully saturated region (no uncovered
cells nearby) is doing its job correctly — it's inside the shape, holding its patch.
Penalizing it is wrong.

Two distinct sub-cases:
1. **Under-packed region** — unoccupied cells exist nearby. Agent should center itself
   among them (move toward medial axis of uncovered territory). `v_exp_norm < 0.05` is correct.
2. **Over-packed region** — no unoccupied cells visible. Agent should hold position inside
   shape. `in_flag` alone should be rewarded.

### Fix — Uniformity condition
```python
# Old
is_uniform = in_flag & any_sensed & (v_exp_norm < 0.05)

# New
is_uniform = jnp.where(any_sensed, v_exp_norm < 0.05, True)
# If unoccupied cells exist: must be centered. If none exist: automatically uniform.
```

---

## d_sen and Uniformity Coupling (note for partial observability discussion)

`d_sen` and the uniformity signal are geometrically coupled. For `v_exp ≈ 0` to correctly
mean "agent is on the shape's medial axis", the sensing disk must reach the shape boundary
on all sides. This requires `d_sen >= half the width of the thinnest shape feature`.

For the T-shape with arm width ~0.3 units: `d_sen >= 0.15` minimum.
Currently `d_sen=0.4` — comfortably covers full arm width on both sides.

**Two regimes:**
- **Full observability:** `d_sen >= thinnest_feature_width / 2` required for `v_exp` to
  encode the true global medial axis (agent sees both edges of thin arms simultaneously)
- **Partial observability:** constraint relaxes. `v_exp` becomes a local centering signal —
  agent centers itself among what it can see, not the global medial axis. Still a valid and
  useful signal. Hard minimum is `d_sen > r_avoid` so the agent can at least sense whether
  its own patch is centered.

**DONE:** `--d_sen` and `--r_avoid` are now CLI flags in `cfg/assembly_cfg.py` (defaults: d_sen=0.4, r_avoid=0.10). Both are passed through to the env constructor in `train_assembly_jax_gpu.py` and `eval_shapes.py`.

---

## Uniformity Logic (AGREED)

### Current behavior
```python
is_uniform = in_flag & any_sensed & (v_exp_norm < 0.05)
```
When `any_sensed=False` (saturated region) → `is_uniform=False` → reward=0. Wrong.

### What each case should do

| Case | Condition | Correct behavior |
|---|---|---|
| Under-packed | `in_flag=True`, `any_sensed=True` | `v_exp` over unoccupied visible cells — pull toward medial axis of uncovered territory |
| Saturated | `in_flag=True`, `any_sensed=False` | No free cell to move to within d_sen — stay put, `is_uniform=True` automatically |
| Spacing violation | `in_flag=True`, `~is_collision=False` | Reward=0 regardless, agent must spread out |
| Outside shape | `in_flag=False` | `is_uniform` irrelevant — `in_flag` gates everything |

### Fix
```python
# If unoccupied cells exist: must be centered among them
# If no unoccupied cells exist: saturated, nowhere better to go, automatically uniform
is_uniform = jnp.where(any_sensed, v_exp_norm < 0.05, True)
```

Full reward condition unchanged:
```python
reward = in_flag & ~is_collision & is_uniform
```
`in_flag` is the outer gate — uniformity logic only applies inside the shape.

### Why ~is_collision is still needed alongside uniformity

Considered removing `~is_collision` and relying on uniformity alone for spacing. Rejected.

- **Under-packed:** uniformity handles spacing naturally — two agents close together both
  see the same unoccupied cells on the far side, `v_exp` pulls both apart. `~is_collision`
  rarely fires here anyway.
- **Saturated:** `is_uniform=True` automatically. If `~is_collision` were removed, two agents
  on top of each other in a saturated region both get full reward — no incentive to separate.
  Physics (`k_ball`) only guarantees `2 * size_a = 0.07` separation, not `2 * r_avoid`.
  The gap between physical contact and desired spacing can only be enforced by the reward.

The two conditions complement each other:
- Under-packed: uniformity does the work, `~is_collision` rarely fires
- Saturated: `~is_collision` does the work, uniformity is automatically True

**The reward structure `in_flag & ~is_collision & is_uniform` is correct.**
The problems are in the parameter values and the saturated case logic, not the structure.

### Deferred
- Binary → continuous reward (v_exp_norm is a rich continuous signal, currently thresholded
  at 0.05 — agent slightly off gets same reward=0 as agent wildly off, no gradient between
  bad and less bad). Fix structure first, continuous later.

---

## Outside Shape Case (AGREED)

### Current behavior
`in_flag=False` → reward=0 everywhere outside. No directional signal. Early training
pathology: if reward inside is also near 0 (conditions not yet learned), no differential
exists and agents have no reason to prefer inside over outside.

### Why special outside logic is not needed
- `v_exp` already points toward shape from just outside (shape cells visible within d_sen)
- Reynolds flocking prior has cohesion — pulls outliers toward neighbor centroid which is
  inside the shape if most agents are there
- Once inside reward is well-posed, reward differential at boundary is sufficient

### Stepping stone reward (AGREED)
Binary all-or-nothing reward creates an early training problem — all conditions must align
simultaneously before any signal exists. Fix with a small stepping stone for `in_flag` alone:

```
reward = 0.0   (outside shape — in_flag=False)
reward = 0.1   (inside shape, conditions not fully met)
reward = 1.0   (inside shape, ~is_collision & is_uniform both satisfied)
```

- Creates clear gradient at boundary from episode 1
- Agent always prefers inside over outside
- Intermediate signal for agents that are inside but not yet well-placed
- Flocking prior reinforces entry further
- Exact weight (0.1) to be tuned — small enough that agents don't cluster inside without
  satisfying spacing/uniformity, large enough to pull agents in from outside

---

## Physical Collision Penalty (AGREED)

### Motivation
Two distinct signals serve two distinct purposes:
- `~is_collision` (dist < 2 * r_avoid = 0.20): **spacing policy** — stay spread out, cover shape
- Physical collision penalty (dist < 2 * size_a = 0.07): **hard constraint** — must never happen

Physical contact is categorically different from spacing violation. In real robotics this means
hardware damage, instability, cascading failures. The reward must reflect this hierarchy:
losing reward is bad, but physical contact is worse. The penalty makes this distinction explicit.

### Detection
Already implemented via `springboard_collision_count_jax()` — counts unique touching pairs
per step as a JAX scalar (no CPU sync). For the reward, per-agent count is needed: how many
agents is agent i physically touching this step.

### Reward structure
```
outside shape:                     reward = 0.0
inside, conditions not met:        reward = 0.1   (stepping stone)
inside, all conditions met:        reward = 1.0
physical collision (per neighbor): reward -= 0.2  (applied on top of above)
```

- One touching neighbor: 1.0 - 0.2 = 0.8 — clearly felt
- Two simultaneous: 1.0 - 0.4 = 0.6 — significant
- Three: 1.0 - 0.6 = 0.4 — very significant
- Agent already lost full reward from ~is_collision; penalty makes physical contact
  categorically worse — "do not let this happen"
- Small enough not to overwhelm stepping stone for agents just entering the shape
- Exact weight (0.2) to be tuned

---

## Variable / Condition Naming (AGREED)

Clean separation between spacing policy and physical contact:

| Old name | New name | Threshold | Meaning |
|---|---|---|---|
| `is_collision` | `too_close` | `dist < 2 * r_avoid` | spacing policy — personal space bubbles overlapping |
| (unnamed) | `is_touching` | `dist < 2 * size_a = 0.07` | physical event — bodies in contact, k_ball fires |
| `n_contacts` | `n_touching` | count of `is_touching` neighbors | how many agents physically touching agent i |

---

## Summary of Changes — All Implemented

| # | What | File | Status |
|---|---|---|---|
| 1 | k_ball=2000, 4 substeps in _world_step | assembly.py | **DONE** |
| 2 | Hardcode r_avoid=0.10 as constructor arg, remove formula | assembly.py | **DONE** |
| 3 | Rename `is_collision` → `too_close`, update threshold: `dist < r_avoid` → `dist < 2 * r_avoid` | assembly.py | **DONE** |
| 4 | Update coverage radius: `r_avoid/2` → `r_avoid` everywhere (obs + sensing_coverage) | assembly.py | **DONE** |
| 5 | Fix `is_uniform` saturated case: `any_sensed=False` → `is_uniform=True` | assembly.py | **DONE** |
| 6 | Stepping stone reward: +0.1 for `in_flag` alone | assembly.py | **DONE** |
| 7 | Physical contact penalty: -0.07 per `is_touching` neighbor (`n_touching`) | assembly.py | **DONE** |
| 8 | Wire `--d_sen` as CLI flag in assembly_cfg.py, pass to env constructor | assembly_cfg.py + assembly.py | **DONE** |
| 9 | Wire `--r_avoid` as CLI flag in assembly_cfg.py, pass to env constructor | assembly_cfg.py + assembly.py | **DONE** |

**Note on penalty weight**: The "Physical Collision Penalty" discussion section above explored 0.2 per neighbor. The final implemented value is **0.07 per touching neighbor** (matching `size_a = 0.035`, i.e. 2×size_a). This gives stepping stone table: inside+touching 1: `0.1 - 0.07 = 0.03`, inside all conditions met: `1.0 - 0.07×n_touching`.

## Reward Structure Summary (target)

```
too_close_i  = any(nei_dist < 2 * r_avoid)   for K nearest within d_sen
is_touching_i = any(agent_dist < 2 * size_a)  all agents, not just K nearest
n_touching_i  = count(agent_dist < 2 * size_a)

reward_i = 0.1  × in_flag                              # stepping stone
         + 0.9  × (in_flag & ~too_close & is_uniform)  # full conditions met
         - 0.07 × n_touching_i                         # physical contact penalty
```

Stepping stone table:
- Outside shape:                    0.0
- Inside, physically touching 1:    0.1 - 0.07 = 0.03  (barely positive)
- Inside, conditions not met:       0.1
- Inside, all conditions met:       1.0

## Metrics Redesign (AGREED)

### coverage_rate — replace with sensing_coverage

Old `coverage_rate` used `a2g_dist < r_avoid/2` as the coverage radius. With r_avoid=0.10
this gives a theoretical maximum of ~63% even with perfect placement — the metric can never
reach 1.0. Meaningless as an absolute measure.

**New metric: `sensing_coverage`**
```python
cell_sensed = jnp.any(a2g_dist < d_sen, axis=0)   # [n_g_max]
sensing_coverage = sum(cell_sensed & valid_mask) / sum(valid_mask)
```
A cell is "covered" if at least one agent can see it (within d_sen). Fraction of shape
collectively visible to the team.

- No r_avoid dependency
- Reaches 1.0 when agents are spread across the shape (achievable)
- Directly meaningful for partial observability experiments — reducing d_sen reduces
  collective sensing coverage, metric captures this cleanly
- Answers: "does the team collectively perceive the entire shape?"

### collision_rate and count_collisions — replace with r_avoid_violation_count

Old metrics used `dist < r_avoid` (wrong threshold) and were agent-wise (double-counting).
Both removed and replaced with a single pairwise event-based metric.

**New metric: `r_avoid_violation_count`**
```python
upper = jnp.triu(jnp.ones((n_a, n_a), dtype=bool), k=1)
violations = (dists < 2.0 * r_avoid) & upper
r_avoid_violation_count = jnp.sum(violations)
```
- Counts unique pairs (i < j) with center-to-center distance < 2 * r_avoid
- One pair too close = 1 event (no double counting)
- Three mutually close agents = 3 pairs = 3 events
- Report as accumulation per episode, averaged across envs (same pattern as springboard_collision_count)
- Same upper-triangle pattern already used in springboard_collision_count

### coverage_efficiency — remove
Mathematically identical to coverage_rate (= n_occupied / n_g). Redundant. Remove.

### Metrics that are correct as-is
- `distribution_uniformity` — CoV of nearest-neighbor distances. No r_avoid dependency.
- `voronoi_based_uniformity` — CoV of Voronoi cell counts. No r_avoid dependency.
- `mean_neighbor_distance` — mean nearest-neighbor distance. No r_avoid dependency.
- `springboard_collision_count` — physical contact pairs (dist < 2*size_a). Correct.

---

## Full r_avoid Audit — Every Occurrence in assembly.py

All occurrences categorized by what changes with new definition.

### Group 1 — spacing check (old: `dist < r_avoid`, new: `dist < 2 * r_avoid`)
- Reward `too_close` (vectorised): `nei_dists < r_avoid` → `nei_dists < 2.0 * r_avoid`
- Reward `too_close` (single agent): `nei_dists_topo < r_avoid` → `nei_dists_topo < 2.0 * r_avoid`
- `count_collisions`: remove entirely (replaced by `r_avoid_violation_count`)
- `collision_rate`: remove entirely (replaced by `r_avoid_violation_count`)

### Group 2 — coverage radius (old: `r_avoid/2`, new: `r_avoid`)
- Obs occupancy (vectorised + single agent): `a2g_dist < r_avoid/2` → `a2g_dist < r_avoid`
- `coverage_rate`: remove entirely (replaced by `sensing_coverage`)
- `coverage_efficiency`: remove entirely (redundant)

### Group 3 — Reynolds prior repulsion (update to new definition)
Old:
```python
rep_factor = where(nei_dists < r_avoid, 3.0 * (r_avoid/dist - 1), 0)
```
New (both vectorised and single-agent versions):
```python
rep_factor = where(nei_dists < 2.0 * r_avoid, 3.0 * (2.0 * r_avoid/dist - 1), 0)
```
Physical meaning preserved — repulsion fires at same center-to-center spacing threshold,
magnitude formula has same shape (0 at boundary, grows as dist shrinks).

### Group 4 — is_nearby lookahead (old: `d_sen + r_avoid/2`, new: `d_sen + r_avoid`)
Agent j's territory (radius `r_avoid`) could be visible to agent i (sensing radius `d_sen`)
if j is within `d_sen + r_avoid` of i. All four occurrences:
```python
is_nearby = agent_dists < (d_sen + r_avoid/2)   # old
is_nearby = agent_dists < (d_sen + r_avoid)      # new
```

### Group 5 — formula removal + constructor arg
```python
# Old: computed from shape geometry
self.r_avoid = round(np.sqrt(4.0 * min_n_g / (n_a * np.pi)) * min_l_cell, 2)

# New: hardcoded constructor arg with default 0.10
# Preferred value: floor(size_a + 2*size_a) = 3 * 0.035 = 0.105 → 0.10
# Gives minimum center-to-center spacing 2*r_avoid=0.20, one full body diameter
# clearance between personal space boundaries
r_avoid: float = 0.10
```

---

## Variables Still To Decide

- r_avoid=0.10 — **implemented**. Tuning deferred until MLP baseline run confirms reward is well-posed.
- Binary → continuous reward — deferred, structure fixed first
- d_sen reduction for partial observability — deferred until MLP baseline confirms fixed reward works

---

## Canonical Definition of r_avoid (AGREED)

**`r_avoid` is the personal space radius of each agent.**

- Each agent has a personal space bubble of radius `r_avoid` centred on itself
- Two agents are "too close" when their bubbles overlap: center-to-center `dist < 2 * r_avoid`
- Minimum allowable center-to-center distance between any two agents = `2 * r_avoid`
- A grid cell is "covered" by an agent if the agent centre is within `r_avoid` of the cell

**Previous code interpretation (before this redesign):**
- `is_collision = dist < r_avoid` — treated r_avoid as the full diameter (not radius)
- `cell_occupied = a2g_dist < r_avoid/2` — coverage radius was r_avoid/2

**What changes with the new definition:**
- Spacing check: `dist < r_avoid` → `dist < 2 * r_avoid`
- Coverage radius: `r_avoid/2` → `r_avoid`
- r_avoid value itself will be smaller (~0.10) since it now represents a radius not diameter

---

## Geometric Reference

| Quantity | Value |
|---|---|
| Agent radius (size_a) | 0.035 |
| Agent diameter | 0.07 |
| Physical contact threshold | 0.07 (k_ball activates here) |
| Current r_avoid (old definition, diameter) | 0.29 (too large) |
| Proposed r_avoid (new definition, radius) | ~0.10 |
| d_sen (sensing range) | 0.40 |
| T-shape area (approx) | ~1.2 sq units |
| n_a | 24 |
