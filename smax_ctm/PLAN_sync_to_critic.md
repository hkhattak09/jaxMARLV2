# Plan: Sync Vectors to Critic (Rapid Prototype)

## Summary
Move sync vectors out of the actor head and into the critic. The actor uses only
`last_activated` (128d) for action selection — no sync bottleneck. The critic
receives sync vectors concatenated after its GRU output, enabling coordination-aware
value estimates.

See `DISCUSSION_sync_to_critic.md` for full rationale.

---

## Changes Required

### File 1: `smax_ctm/train_mappo_ctm.py`

#### 1a. ActorCTM.__call__ — simplify head input (line ~149)

Current:
```python
head_input = jnp.concatenate([synch, last_activated], axis=-1)
```

Change to:
```python
head_input = last_activated
```

Everything else in ActorCTM stays the same. It still unpacks `(synch, last_activated)`
from ScannedCTM and still returns `(hidden, pi, synch)` — synch is needed for SAAL
and now for the critic.

#### 1b. Transition — add synch field (line ~200)

Current:
```python
class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
```

Add after `avail_actions`:
```python
    synch: jnp.ndarray
```

#### 1c. Rollout _env_step — capture and store synch (lines ~437-450)

Current (both branches):
```python
ac_hstate, pi, _ = actor_network.apply(...)
```

Change to:
```python
ac_hstate, pi, synch = actor_network.apply(...)
```

In the Transition construction (line ~470), add:
```python
synch=synch.squeeze(),
```

The squeeze removes the time axis (1, num_actors, 528) → (num_actors, 528), matching
the shape convention of other Transition fields.

#### 1d. Rollout carry — add last_synch for bootstrap (line ~420, ~482)

Current carry:
```python
train_states, env_state, last_obs, last_done, hstates, rng = runner_state
```

Add `last_synch` to carry. Initialize before scan with zeros:
```python
init_synch = jnp.zeros((config["NUM_ACTORS"], synch_size))
```
where `synch_size = config["CTM_N_SYNCH_OUT"] * (config["CTM_N_SYNCH_OUT"] + 1) // 2`.

Update carry to include `last_synch` at end of _env_step:
```python
runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), synch.squeeze(), rng)
```

Unpack after scan (line ~507):
```python
train_states, env_state, last_obs, last_done, hstates, last_synch, rng = runner_state
```

#### 1e. Bootstrap value — pass last_synch to critic (lines ~509-512)

Current:
```python
cr_in = (last_world_state[None, :], last_done[np.newaxis, :])
_, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
```

Change to:
```python
cr_in = (last_world_state[None, :], last_done[np.newaxis, :], last_synch[np.newaxis, :])
_, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
```

Note: `last_synch` is one step stale (from the last action, not the current
observation). This is acceptable — the bootstrap is already an approximation.

#### 1f. Rollout _env_step — pass synch to critic during collection (line ~460)

Current:
```python
cr_in = (world_state[None, :], last_done[np.newaxis, :])
```

Change to:
```python
cr_in = (world_state[None, :], last_done[np.newaxis, :], synch)
```

Here `synch` already has shape (1, num_actors, 528) from the actor call — no
newaxis needed.

#### 1g. CriticRNN.__call__ — accept and use synch (lines ~178-197)

Current:
```python
def __call__(self, hidden, x):
    world_state, dones = x
    embedding = nn.Dense(
        self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
    )(world_state)
    embedding = nn.relu(embedding)

    rnn_in = (embedding, dones)
    hidden, embedding = ScannedRNN()(hidden, rnn_in)

    critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
        embedding
    )
    critic = nn.relu(critic)
    critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
        critic
    )

    return hidden, jnp.squeeze(critic, axis=-1)
```

Change to:
```python
def __call__(self, hidden, x):
    world_state, dones, synch = x

    embedding = nn.Dense(
        self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
    )(world_state)
    embedding = nn.relu(embedding)

    rnn_in = (embedding, dones)
    hidden, embedding = ScannedRNN()(hidden, rnn_in)

    # Concat GRU output with sync vectors for coordination-aware value estimate
    value_input = jnp.concatenate([embedding, synch], axis=-1)

    critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
        value_input
    )
    critic = nn.relu(critic)
    critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
        critic
    )

    return hidden, jnp.squeeze(critic, axis=-1)
```

#### 1h. Critic loss function — pass synch from traj_batch (line ~632)

Current:
```python
def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
    _, value = critic_network.apply(
        critic_params,
        jax.tree.map(lambda x: x[0], init_hstate),
        (traj_batch.world_state, traj_batch.done),
    )
```

Change to:
```python
    (traj_batch.world_state, traj_batch.done, traj_batch.synch),
```

#### 1i. Critic init — pass dummy synch for param initialization (around line ~400)

Find where `critic_network.init` is called. It currently passes a dummy
`(world_state, done)` tuple. Add a dummy synch array with the right shape:
```python
synch_size = config["CTM_N_SYNCH_OUT"] * (config["CTM_N_SYNCH_OUT"] + 1) // 2
dummy_synch = jnp.zeros((1, config["NUM_ACTORS"], synch_size))
```

Pass `(dummy_world_state, dummy_done, dummy_synch)` to `critic_network.init`.

---

### File 2: `smax_ctm/ctm_jax.py`

**No changes needed.** CTMCell already returns `(synch, last_activated)` from the
previous plan's implementation.

---

### File 3: Config

**No new config keys needed.** The change is structural. Existing keys suffice:
- `CTM_D_MODEL: 128` — activated state size (actor head input)
- `CTM_N_SYNCH_OUT: 32` — sync size (critic input, 528d)
- `CTM_ACTOR_HEAD_DIM: 64` — actor head hidden dim (unchanged)
- `GRU_HIDDEN_DIM: 512` — critic value MLP hidden dim (unchanged, absorbs wider input)

---

## Summary of All Changes

| Location | Change | Lines (approx) |
|----------|--------|----------------|
| ActorCTM.__call__ | head_input = last_activated (drop sync) | ~149 (1 line) |
| Transition | add synch field | ~210 (1 line) |
| _env_step | capture synch from actor, store in Transition | ~437,445,470 (~4 lines) |
| _env_step carry | add last_synch to carry for bootstrap | ~420,482 (~3 lines) |
| post-scan unpack | unpack last_synch | ~507 (1 line) |
| bootstrap cr_in | add last_synch | ~511 (1 line) |
| rollout cr_in | add synch | ~460 (1 line) |
| CriticRNN.__call__ | unpack synch, concat after GRU | ~179-196 (~4 lines) |
| _critic_loss_fn | pass traj_batch.synch | ~636 (1 line) |
| critic init | add dummy synch | ~400 (2 lines) |

Total: ~20 lines changed across one file.

---

## What About PLAN_ctm_head_augmentation.md?

That plan has been implemented (CTMCell returns `(synch, last_activated)`, actor
head uses `concat(synch, last_activated)`). This new plan **supersedes** the actor
head portion — we now use `last_activated` only. The CTMCell changes from that plan
remain useful (we still need `last_activated` returned). The old plan file can be
deleted.

---

## Testing Plan

### Test 1: Shape Smoke Test (local, before Colab)
Add a temporary print in CriticRNN.__call__ on first call:
```python
jax.debug.print("critic value_input shape: {}", value_input.shape)
```
Expected: `(T, num_actors, 512 + 528)` = `(T, num_actors, 1040)`

### Test 2: Existing Tests
Run `smax_ctm/tests/test_inc.py` — these test CTMCell output shapes. They should
still pass since CTMCell is unchanged. The critic tests (if any) will need the new
synch argument.

### Test 3: Short Training Run (Colab)
Run 3m SMAX for 1M steps. Check:
- [ ] Loss decreases (no NaN/Inf)
- [ ] Win rate improves from random (~0.0)
- [ ] Entropy decreases gradually (not collapse)
- [ ] pair_cos metrics still logged correctly (SAAL path unchanged)

### Test 4: Comparison (Colab)
Run three configs on 3m for 3M steps:
- [ ] **A**: Current code (actor=concat(sync,activated), critic=world_state only)
- [ ] **B**: New code (actor=activated only, critic=world_state+sync)
- [ ] **C**: GRU baseline

Compare win rate curves. Hypothesis: B >= A, and B closer to C than A is.

### Test 5: Stretch — 10-unit SMAX
If 3m results are promising, run on 10v10 where the GRU gap is largest (0.12 vs 0.45).
This is where coordination-aware critic value should matter most.
