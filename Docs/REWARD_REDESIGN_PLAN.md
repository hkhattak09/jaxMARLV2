# Reward Redesign Plan

## Problem Summary

1. **K=6 in reward hides clusters**: `too_close` only checks K=6 nearest neighbors. Full pairwise `agent_dists [n_a, n_a]` is already computed but not used in reward. Result: r_avoid violations reported as less than physical collisions (impossible).
2. **All binary gates**: `in_flag`, `too_close`, `is_uniform` are booleans. No gradient for the policy — flat plateaus with cliff edges.
3. **Reward doesn't measure what we care about**: Goal is Voronoi uniformity (equal territory per agent). Reward uses `is_uniform` (local cell-balance check via v_exp_norm < 0.05) which is a loose proxy.
4. **`is_uniform` is opaque**: Weighted centroid of sensed unoccupied cells, binary threshold, saturated case handling. Replaced by territory balance.

## New Reward — Three Continuous Components

### Component 1: Shape Proximity (continuous attraction)

Replaces binary `in_flag` gating and `-0.03` outside penalty.

```python
dist = cached.nearest_grid_dist                    # [n_a]
cell_thresh = jnp.sqrt(2.0) * state.l_cell / 2.0  # ~0.042

# Broad pull: attracts agents toward shape from up to d_sen away
broad = jnp.clip(1.0 - dist / self.d_sen, 0.0, 1.0)

# Sharp on-cell bonus: high reward for sitting on a cell
sharp = jnp.clip(1.0 - dist / cell_thresh, 0.0, 1.0)

proximity = 0.3 * broad + 0.7 * sharp  # [n_a], range [0, 1]
```

- Two scales: `broad` pulls from far (up to d_sen=0.3), `sharp` rewards precision on-cell
- Continuous everywhere, replaces binary in_flag

### Component 2: Crowding Penalty (all-pairs, continuous, density-aware)

Replaces K-limited binary `too_close`. Uses full `cached.agent_dists [n_a, n_a]`.

```python
dists_excl = jnp.where(jnp.eye(n_a, dtype=bool), jnp.inf, cached.agent_dists)

# Quadratic overlap: smooth at boundary, scales with proximity AND density
raw_overlap = jnp.maximum(0.0, 2.0 * self.r_avoid - dists_excl) / (2.0 * self.r_avoid)
overlap = raw_overlap ** 2                      # [n_a, n_a], smooth gradient at boundary
crowding = jnp.sum(overlap, axis=1)             # [n_a]
```

- Uses ALL pairwise distances, not K=6
- Quadratic: smooth zero-gradient at boundary (dist = 2*r_avoid), increasing as agents get closer
- Sum scales with cluster size: 10 neighbors at dist 0.10 gives 10x penalty vs 1 neighbor
- Penalizes all agents everywhere (not gated on in_flag) — prevents pre-clustering off-shape

### Component 3: Territory Balance (direct Voronoi proxy)

Replaces binary `is_uniform`. Directly measures per-agent Voronoi territory share.

```python
# Which agent is closest to each shape cell? (god-view, full a2g_dist)
a2g_masked = jnp.where(state.valid_mask[None, :], cached.a2g_dist, jnp.inf)
nearest_agent = jnp.argmin(a2g_masked, axis=0)        # [n_g_max]

# Count cells "owned" by each agent
agent_ids = jnp.arange(n_a)[:, None]                   # [n_a, 1]
owns = (agent_ids == nearest_agent[None, :]) & state.valid_mask[None, :]  # [n_a, n_g_max]
territory = jnp.sum(owns, axis=1).astype(jnp.float32)  # [n_a]

# Ideal: equal share
n_valid = jnp.sum(state.valid_mask).astype(jnp.float32)
ideal = n_valid / n_a

# Score: 1.0 at ideal, linear falloff, 0.0 when >=50% deviation
territory_score = jnp.clip(1.0 - jnp.abs(territory - ideal) / (0.5 * ideal), 0.0, 1.0)
```

- Cheap: argmin over already-computed a2g_dist + one-hot comparison. ~10K ops for N=20, n_g_max=500.
- argmin not differentiable but doesn't need to be — critic learns to predict reward, actor optimizes through critic.
- Gated on proximity in the combined reward (territory only matters when on/near shape).

### Combined Reward

```python
reward = (0.20 * proximity
        + 0.60 * proximity * territory_score
        - 0.15 * crowding
        - 0.05 * n_touching)
```

- Territory (60%) is primary objective, gated on proximity
- Proximity (20%) gets agents to the shape
- Crowding (15%) prevents pileups using full pairwise info
- Touching (5%) safety penalty for physical contact

### What We Remove

| Old component | Replaced by | Why |
|---|---|---|
| Binary `in_flag` gating | Continuous `proximity` | Gradient everywhere |
| K-limited `too_close` | All-pairs `crowding` | Sees real cluster density |
| Binary `is_uniform` (v_exp_norm < 0.05) | `territory_score` | Directly measures Voronoi territory |
| `-0.03` outside penalty | Subsumed by `proximity` = 0 when far | Cleaner |
| `is_nearby` / `is_occupied` / `is_sensed_unoccupied` / `psi` / `v_exp` computation | Removed | All part of old is_uniform |

## Prior Repulsion Threshold Change

Change repulsion threshold from `4 * r_avoid` to `2 * r_avoid` in both:

### `_robot_policy_vectorized` (line ~1156-1158)
```python
# OLD:
rep_factor = jnp.where(
    (nei_dists > 0) & (nei_dists < 4.0 * self.r_avoid),
    3.0 * (4.0 * self.r_avoid / safe_nei_dist - 1.0),
    0.0
)

# NEW:
rep_factor = jnp.where(
    (nei_dists > 0) & (nei_dists < 2.0 * self.r_avoid),
    3.0 * (2.0 * self.r_avoid / safe_nei_dist - 1.0),
    0.0
)
```

### `_robot_policy_single` (line ~1081-1084)
Same change: `4.0 * self.r_avoid` → `2.0 * self.r_avoid` in both the condition and the factor formula.

**Rationale**: `4 * r_avoid = 0.40 > d_sen = 0.3` — prior was repelling from agents beyond sensing range. `2 * r_avoid = 0.20` = the spacing violation threshold. Prior and reward now agree: "too close" means `dist < 2 * r_avoid`.

## Files to Change

1. **`JaxMARL/jaxmarl/environments/mpe/assembly.py`**:
   - `_rewards_vectorized()`: full rewrite with new 3-component reward
   - `_reward_single()`: matching rewrite (reference implementation)
   - `_robot_policy_vectorized()`: repulsion threshold 4*r_avoid → 2*r_avoid
   - `_robot_policy_single()`: same threshold change
   - Old `is_uniform` helper code (`_rho_cos_dec`, occupancy computation) can stay — may be used elsewhere. Only remove from reward functions.

2. **No other files need changes** — reward is computed inside assembly.py and returned as a scalar per agent. All downstream code (wrapper, buffer, training loop) just consumes the reward value.

## Not Changing

- K=6 in observations — stays as partial observability design choice
- K=6 in prior — stays as local controller representation  
- Reynolds prior structure (attraction + repulsion + velocity sync) — only threshold changes
- No team-level Voronoi bonus for now — per-agent territory_score should suffice
- Weights are a starting point (0.20, 0.60, 0.15, 0.05) — may need tuning
