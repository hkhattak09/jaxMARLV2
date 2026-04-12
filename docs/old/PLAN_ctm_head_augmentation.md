# Plan: Augment CTM Actor Head with Activated State

## Problem
The CTM actor head receives only the synchronization vector (528 dims) — pairwise
products of 32 output neurons across the memory window. This discards individual
neuron magnitudes. The GRU baseline passes its full 128-dim hidden state directly
to its head. The CTM is stuck at ~0.12 win rate on SMAX 10-unit while GRU reaches
~0.45.

## Goal
Concatenate the last activated state (128 dims) alongside the sync vector (528 dims)
before feeding the actor head. The sync still contributes relational information.
The activated state provides direct magnitude information. SAAL alignment (which
operates on the sync vector between agents) is untouched.

---

## Changes Required

### File 1: `smax_ctm/ctm_jax.py`

#### CTMCell.__call__ (line ~448)
Currently returns:
```python
return new_carry, synch
```
Change to return both synch AND the last activated state:
```python
last_activated = activated_state_trace[:, :, -1]  # (batch, d_model)
return new_carry, (synch, last_activated)
```

#### ScannedCTM.__call__ (line ~463)
Currently returns whatever CTMCell returns. The `nn.scan` with `out_axes=0` will
automatically stack both tuple elements along the time axis. No change needed to
ScannedCTM itself — the tuple will propagate through scan correctly.

**Verify:** `nn.scan` with `out_axes=0` handles tuple outputs by stacking each
element independently. This is standard Flax behavior. The output becomes
`(synch_stacked, last_activated_stacked)` where both have shape `(T, B, ...)`.

---

### File 2: `smax_ctm/train_mappo_ctm.py`

#### ActorCTM.__call__ (lines 134-159)
Currently:
```python
hidden, synch = ScannedCTM(self.config, deterministic=deterministic)(
    hidden, (obs, dones, avail_actions)
)
x_head = nn.Dense(self.config["CTM_ACTOR_HEAD_DIM"])(synch)
x_head = nn.relu(x_head)
x_head = nn.Dense(self.config["CTM_ACTOR_HEAD_DIM"])(x_head)
x_head = nn.relu(x_head)
x_head = nn.Dense(self.action_dim)(x_head)
```

Change to:
```python
hidden, (synch, last_activated) = ScannedCTM(self.config, deterministic=deterministic)(
    hidden, (obs, dones, avail_actions)
)

# Concat sync (528) + activated state (128) = 656 dims for head input
head_input = jnp.concatenate([synch, last_activated], axis=-1)

x_head = nn.Dense(
    self.config["CTM_ACTOR_HEAD_DIM"],
    kernel_init=orthogonal(np.sqrt(2)),
    bias_init=constant(0.0),
)(head_input)
x_head = nn.relu(x_head)
x_head = nn.Dense(
    self.config["CTM_ACTOR_HEAD_DIM"],
    kernel_init=orthogonal(np.sqrt(2)),
    bias_init=constant(0.0),
)(x_head)
x_head = nn.relu(x_head)
x_head = nn.Dense(
    self.action_dim,
    kernel_init=orthogonal(0.01),
    bias_init=constant(0.0),
)(x_head)
```

Note: `orthogonal` and `constant` are already imported in this file (used by CriticRNN).
The final layer uses `orthogonal(0.01)` — this produces near-zero initial logits,
giving a near-uniform initial policy. This matches the GRU baseline and is important
for PPO exploration.

#### Return value of ActorCTM.__call__
Currently returns `(hidden, pi, synch)`. Keep returning `synch` (not `head_input`)
because downstream SAAL pair_cos computation needs the pure sync vector.
```python
return hidden, pi, synch
```
No change here.

#### SAAL pair_cos computation (lines ~556-566)
This code reshapes `synch` for cross-agent cosine similarity. It receives `synch`
from the ActorCTM return value, which is still the pure sync vector. **No change needed.**

#### Both call sites of actor_network.apply (lines ~532, ~540)
These unpack `(hidden, pi, synch)` from ActorCTM. The return signature hasn't
changed, so **no change needed** at these call sites.

---

### File 3: Config (no changes needed)

The 3m config you showed does not need any new keys. The change is structural
(concat before the head), not configurable. The existing keys are sufficient:

- `CTM_D_MODEL: 128` — determines activated state size (128 dims)
- `CTM_N_SYNCH_OUT: 32` — determines sync size (528 dims)
- `CTM_ACTOR_HEAD_DIM: 64` — head hidden dim stays the same

The first Dense layer in the head will automatically adapt to the larger input
(656 vs 528) because Flax Dense infers input dims.

---

## Summary of touched lines

| File | What changes | Lines |
|------|-------------|-------|
| `ctm_jax.py` | CTMCell.__call__ return | ~448 (1 line becomes 2) |
| `train_mappo_ctm.py` | ActorCTM.__call__ unpack + head input + init | ~145-153 (~10 lines) |

Everything else (ScannedCTM, SAAL, critic, config, carry shape) stays the same.

## Compute cost
- One extra `jnp.concatenate` per forward pass (negligible)
- First Dense layer: 656×64 = 42K params instead of 528×64 = 34K params (+8K)
- Total extra: ~8K multiply-adds per step. Unmeasurable impact on wall time.

## Tests to verify
- Existing tests in `smax_ctm/tests/` should still pass (the synch output is
  unchanged, just accompanied by last_activated now)
- Check that `test_inc.py` tests unpack correctly — they may need to unpack
  `(synch, last_activated)` instead of just `synch` from CTMCell output
- Run a short 3m training (500K steps) to verify loss decreases and no NaN
