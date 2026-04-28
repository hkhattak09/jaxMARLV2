# JaxMARL SMAX Environment

Pure-JAX reimplementation of the SMAC (StarCraft Multi-Agent Challenge) micro-management environment. SMAX is fully JIT-compilable and differentiable, making it ideal for large-scale multi-agent RL experiments.

## Quick Start

```python
from jaxmarl.environments.smax import SMAX, map_name_to_scenario
import jax

# Load a scenario
scenario = map_name_to_scenario("3m")
env = SMAX(scenario=scenario)

# Reset
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

# Sample random valid actions
avail = env.get_avail_actions(state)
actions = {agent: env.action_space(agent).sample(jax.random.PRNGKey(i))
           for i, agent in enumerate(env.agents)}

# Step (auto-reset on episode end)
obs, state, reward, done, info = env.step(key, state, actions)
```

## Core API

### Environment Creation

```python
SMAX(
    scenario: Scenario,           # Scenario object (see below)
    action_type="discrete",        # "discrete" or "continuous"
    observation_type="unit_list",  # "unit_list" or "conic"
    use_self_play_agent=False,
    max_steps=100,
    walls_cause_death=True,
    see_enemy_actions=False,
    world_steps_per_env_step=1,
    attack_mode="closest",         # "closest" or "random"
)
```

### Return Types & State Flow

All methods are JAX-transformable and accept/return JAX arrays.

```python
from jaxmarl.environments.smax.smax_env import State as SMAXState

# Raw env
obs, state      = env.reset(key)          # state: SMAXState
obs, next_state, reward, done, info = env.step(key, state, actions)
                                            # next_state: SMAXState
```

**Important**: `obs["world_state"]` is computed **fresh** inside `get_obs(state)` from the *current* `state` passed to it. It is never cached or stale.

| Method | Exact Return Type | Description |
|--------|-------------------|-------------|
| `reset` | `(obs: Dict[str, Array], state: SMAXState)` | Reset the environment |
| `step` | `(obs: Dict[str, Array], next_state: SMAXState, reward: Dict[str, float], done: Dict[str, bool], info: Dict)` | Step + **auto-reset** when `done["__all__"]` |
| `step_env` | Same as `step` | Step **without** auto-reset |
| `get_obs` | `Dict[str, Array]` | Per-agent observations (includes `"world_state"` key) |
| `get_world_state` | `Array` | Global state vector shaped `(state_size,)` |
| `get_avail_actions` | `Dict[str, Array]` | Binary mask per agent (uint8) |

**Wrapper state types**: Wrappers like `SMAXLogWrapper` wrap the raw state in their own dataclass (e.g. `SMAXLogEnvState` with a `.env_state` field). When you see `state` in a wrapper's `step()`, it refers to the **wrapper's state object**, not the raw env state. The underlying env state is extracted internally. This is why `world_state_fn(obs, state)` in `SMAXWorldStateWrapper` works correctly — it reads `obs["world_state"]` (already fresh) and ignores the `state` argument entirely.

### Observations

Each agent receives:
- **`obs[agent]`**: Per-agent observation shaped `(obs_size,)`
- **`obs["world_state"]`**: Global state shaped `(state_size,)` — concatenation of all unit features, teams, and types

Observations contain normalized health, relative positions, weapon cooldown, unit type one-hot, and last actions.

### Actions

Discrete action space per agent:
- **0-3**: Move in cardinal directions (N, S, E, W after rotation)
- **4**: Stop
- **5+**: Target actions — attack enemies or heal allies (for medivacs)

The number of target actions is `max(num_allies, num_enemies)`, so asymmetric scenarios (e.g., 10 vs 11) share a consistent action space width.

`get_avail_actions(state)` returns a binary mask where `1 = valid action`. The stop action is always valid.

### State Object

```python
@struct.dataclass
class State:
    unit_positions: Array        # (num_agents, 2)
    unit_alive: Array            # (num_agents,) bool
    unit_teams: Array            # (num_agents,) int
    unit_health: Array           # (num_agents,)
    unit_types: Array            # (num_agents,) uint8
    unit_weapon_cooldowns: Array # (num_agents,)
    prev_movement_actions: Array # (num_agents, 2)
    prev_attack_actions: Array   # (num_agents,)
    time: int
    terminal: bool
```

## Scenarios

### Built-in Maps

Classic SMAC scenarios: `3m`, `2s3z`, `25m`, `3s5z`, `8m`, `5m_vs_6m`, `10m_vs_11m`, `27m_vs_30m`, `3s5z_vs_3s6z`, `3s_vs_5z`, `6h_vs_8z`.

SMACv2 procedural scenarios: `smacv2_5_units`, `smacv2_10_units`, `smacv2_20_units`.

### SMACv2 Race Scenarios

Weighted unit-type distributions with procedural positions:

| Scenario | Races | Unit Types | Weights |
|----------|-------|------------|---------|
| `protoss_5_vs_5`, `protoss_10_vs_10`, `protoss_10_vs_11`, `protoss_20_vs_20`, `protoss_20_vs_23` | Protoss | stalker, zealot, colossus | 45%, 45%, 10% |
| `terran_5_vs_5`, `terran_10_vs_10`, `terran_10_vs_11`, `terran_20_vs_20`, `terran_20_vs_23` | Terran | marine, marauder, medivac | 45%, 45%, 10% |
| `zerg_5_vs_5`, `zerg_10_vs_10`, `zerg_10_vs_11`, `zerg_20_vs_20`, `zerg_20_vs_23` | Zerg | zergling, hydralisk, baneling | 45%, 45%, 10% |

*Medivacs (Terran) and banelings (Zerg) are "exception" units — a team cannot be composed entirely of them.*

### Custom Scenarios

```python
from jaxmarl.environments.smax.smax_env import Scenario, register_scenario

my_scenario = Scenario(
    unit_types=jnp.zeros((10,), dtype=jnp.uint8),
    num_allies=5,
    num_enemies=5,
    smacv2_position_generation=True,
    smacv2_unit_type_generation=True,
    unit_type_indices=jnp.array([0, 1, 7], dtype=jnp.uint8),
    unit_type_weights=jnp.array([0.45, 0.45, 0.10], dtype=jnp.float32),
    exception_unit_type_indices=jnp.array([7], dtype=jnp.uint8),
    use_smacv2_unit_types=True,
)
register_scenario("my_map", my_scenario)
```

## Unit Types

### Base Units (6 types)

| Index | Name | Health | Attack | Range | Speed |
|-------|------|--------|--------|-------|-------|
| 0 | marine | 45 | 9 | 5 | 3.15 |
| 1 | marauder | 125 | 10 | 6 | 2.25 |
| 2 | stalker | 160 | 13 | 6 | 4.13 |
| 3 | zealot | 150 | 8 | 2 | 3.15 |
| 4 | zergling | 35 | 5 | 2 | 4.13 |
| 5 | hydralisk | 80 | 12 | 5 | 3.15 |

### SMACv2 Units (9 types)

| Index | Name | Health | Attack | Range | Notes |
|-------|------|--------|--------|-------|-------|
| 6 | colossus | 200 | 15 | 7 | Protoss splash |
| 7 | medivac | 150 | 9 | 6 | Heals allies; exception unit |
| 8 | baneling | 30 | 80 | 1 | Splash on death; exception unit |

## Wrappers

### HeuristicEnemySMAX

Single-team environment where the enemy team is controlled by a built-in heuristic (attack closest / random target).

```python
from jaxmarl.environments.smax import HeuristicEnemySMAX

env = HeuristicEnemySMAX(
    scenario=map_name_to_scenario("terran_5_vs_5"),
    enemy_shoots=True,
    attack_mode="closest",
)
```

### World State Wrapper

Adds a centralized `world_state` observation for critic networks.

```python
from smax_ctm.smax_wrappers import SMAXWorldStateWrapper

env = SMAXWorldStateWrapper(env, obs_with_agent_id=True)
# world_state_size = state_size + num_allies (one-hot agent IDs)
```

### Log Wrapper

Tracks episode returns, lengths, and win rates.

```python
from jaxmarl.wrappers.baselines import SMAXLogWrapper

env = SMAXLogWrapper(env)
# info["returned_episode_returns"], info["returned_won_episode"]
```

## Key Differences from SMAC / SMACv2

| Feature | SMAC (PySC2) | SMAX |
|---------|-------------|------|
| Backend | PySC2 + StarCraft II | Pure JAX |
| JIT | No | Yes |
| Batch / Vmap | Slow | Native |
| SMACv2 races | Protoss/Terran/Zerg | Protoss/Terran/Zerg |
| Medivac healing | Supported | Supported |
| Baneling splash | Supported | Supported |
| Continuous actions | No | Yes (optional) |

## Tips for Training

- **Always mask invalid actions**: Use `env.get_avail_actions(state)` to zero out unavailable actions in your policy logits.
- **Action space size varies by scenario**: Access via `env.action_space(agent).n`. For asymmetric maps, all agents share the same padded action width (`max(num_allies, num_enemies) + num_movement_actions`).
- **Use `jax.vmap`** for batched environment rollouts:
  ```python
  keys = jax.random.split(key, num_envs)
  obs, states = jax.vmap(env.reset)(keys)
  ```
- **World state is already in obs**: After wrapping with `SMAXWorldStateWrapper`, critic inputs are available at `obs["world_state"]`.

## Files

| File | Purpose |
|------|---------|
| `smax_env.py` | Core environment |
| `heuristic_enemy_smax_env.py` | Single-team wrapper with heuristic enemies |
| `heuristic_enemy.py` | Heuristic policy implementation |
| `distributions.py` | Start position and weighted unit-type generators |
| `smax_env.py::MAP_NAME_TO_SCENARIO` | Scenario registry |
