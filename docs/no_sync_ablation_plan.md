# No-Sync Ablation Plan (Stage 5a)

## Goal
Establish **causal** evidence that `compute_synchronisation` is the mechanism driving cross-agent neural coupling and coordination — not just a correlated byproduct of parameter sharing. This is the single highest-value experiment for the paper; without it, the lag-lead finding from Stage 4 is purely correlational and a reviewer will reject.

## Hypothesis
If sync is functional:
- **Win rate drops** significantly on 3m when sync is replaced by a plain projection of the flattened activated trace.
- **Lag-lead signal disappears** — the projected-trace vector shows no predictive rise before coordination events.
- **Event-conditional delta (grouping)** drops or vanishes.

If sync is epiphenomenal:
- Win rate stays the same and the projected-trace vector reproduces all Stage 3–4 findings. (In that case, the paper pivots — see "Decision Gate" below.)

---

## Scientific Design

### What we swap out
The current pipeline at [smax_ctm/ctm_jax.py:187-197](smax_ctm/ctm_jax.py#L187-L197) computes:
```
synch = compute_synchronisation(activated_state_trace, decay_params_out, n_synch_out=32, memory_length=5)
# synch shape: (B, 528)  where 528 = 32*33/2
```
and the actor head downstream consumes this 528-dim vector.

### What we replace it with
A single `Dense(528)` projection applied to the flattened activated trace:
```
flat_trace = activated_state_trace.reshape(B, d_model * memory_length)   # (B, 640)
proxy = nn.Dense(synch_size)(flat_trace)                                  # (B, 528)
```
Everything downstream (actor head, critic, loss, training loop) stays identical.

### Why this specific design (and not alternatives)
- **Match output dimension (528):** Keeps actor head architecture and parameter count comparable. Rules out "sync works because it reduces dimensionality nicely."
- **Single Dense layer (no nonlinearity):** Minimal change. A deeper MLP would conflate "we removed sync" with "we added capacity." A Dense is the same capacity class as the learnable sync (which is essentially a weighted sum of neuron products).
- **Feed raw flattened trace, not `state_trace`:** The activated trace is the thing sync itself consumes, so the ablation's projection has access to the same information — only the specific quadratic/temporal-decay operator is removed. This isolates the mechanism.
- **Keep `activated_state_trace` computation (NLMs, Synapses):** All CTM internal dynamics are preserved. We are *only* removing `compute_synchronisation`.

### What this ablation is NOT
- Not "CTM vs MLP" — backbone, synapses, NLM, and temporal state all remain.
- Not "parameter-sharing ablation" — agents still share all weights.
- Not "dimensionality reduction ablation" — the projection output is the same 528-dim as sync.

---

## Implementation Plan

### Step 1: Add `CTM_USE_SYNC` config flag

**File:** [smax_ctm/ctm_jax.py](smax_ctm/ctm_jax.py)

In `CTMCell`:
1. Add a new dataclass field `use_sync: bool = True` to `CTMCell`.
2. At the end of `__call__`, branch on `self.use_sync`:
   - If `True` (default, baseline): existing `compute_synchronisation` path.
   - If `False`: flatten `activated_state_trace` to `(B, d_model * memory_length)` and apply `nn.Dense(self.n_synch_out * (self.n_synch_out + 1) // 2)` (no activation).
3. The learnable `decay_params_out` must **only** be created when `use_sync=True`. Otherwise the param tree diverges meaninglessly.
4. Name the new Dense layer explicitly: `nn.Dense(..., name="trace_proj")` — this makes the ablation param path stable for analysis.

In `ScannedCTM.__call__`, propagate the flag:
```python
use_sync=self.config.get("CTM_USE_SYNC", True),
```

### Step 2: Update the training script

**File:** [smax_ctm/train_mappo_ctm.py](smax_ctm/train_mappo_ctm.py)

1. Add `"CTM_USE_SYNC": False` to the config dict at [train_mappo_ctm.py:516](smax_ctm/train_mappo_ctm.py#L516) area. **Important:** default in baseline runs should be `True`; we only flip it for the ablation.
2. Change the checkpoint output filename when ablating. Simplest: derive from the flag.
   ```python
   suffix = "_nosync" if not config["CTM_USE_SYNC"] else ""
   model_path = os.path.join(model_dir, f"smax_mappo_ctm_actor{suffix}.pkl")
   ```
3. Keep **every other hyperparameter identical** to the 3m baseline:
   - `SEED=42`, `LR=0.002`, `NUM_ENVS=128`, `NUM_STEPS=128`
   - `TOTAL_TIMESTEPS=3e6`, `ENT_COEF=0.01`
   - `CTM_D_MODEL=128`, `CTM_D_INPUT=64`, `CTM_ITERATIONS=1`
   - `CTM_MEMORY_LENGTH=5`, `CTM_DEEP_NLMS=True`
   - `MAP_NAME="3m"`
4. Save `CTM_USE_SYNC` inside the checkpoint's `config` dict (already happens automatically since the full config is pickled). The analysis script can then detect ablation mode from the checkpoint.

### Step 3: Update the analysis collector

**File:** [smax_ctm/analysis/collector.py](smax_ctm/analysis/collector.py)

Minimal surgical change — the existing collector already calls `ctm_cell.apply(...)` and receives `(new_carry, synch)` on [collector.py:184](smax_ctm/analysis/collector.py#L184). It will work unchanged **if** we ensure the branch in `CTMCell.__call__` still returns a tuple of `(carry, vector)` where `vector` has the same shape `(B, 528)`.

Required change:
1. When constructing `CTMCell` on [collector.py:135](smax_ctm/analysis/collector.py#L135), pass `use_sync=config.get("CTM_USE_SYNC", True)`. This reads the flag out of the checkpoint's saved config.
2. Add a clear log line at load time:
   ```python
   if not config.get("CTM_USE_SYNC", True):
       print("[analyse_sync] NO-SYNC ablation mode — 'synch' vector below is a Dense projection of the activated trace, NOT compute_synchronisation output.")
   ```
3. In the saved diagnostics output, rename the field semantically via a new key `"coord_vector"` alongside `"synch"` for backward compatibility — OR keep the key `"synch"` and document in the output `metadata` dict that it refers to the projection when `use_sync=False`. **Go with the latter** (less churn); just store `use_sync` in the metadata.

### Step 4: Update `analyse_sync.py` to label ablation runs

**File:** [smax_ctm/analyse_sync.py](smax_ctm/analyse_sync.py)

1. After loading the checkpoint, detect `config["CTM_USE_SYNC"]` and print the mode.
2. Append `_nosync` suffix to figure titles and the output directory default if not user-specified. Titles should read e.g. "Event-Conditional Coord Vector Correlation Delta (No-Sync Ablation)" so plots don't get confused with baseline runs.
3. No other logic changes — the metric code treats the 528-dim vector as an opaque feature vector, so correlations/lags still make sense.

### Step 5: Sanity checks before launching training

Before kicking off the 3M-step training run:
1. **Run `jax.tree.map(lambda x: x.shape, params)` on a freshly initialized model** with `use_sync=False`. Confirm:
   - `decay_params_out` is **absent**.
   - `trace_proj/kernel` has shape `(640, 528)` and `trace_proj/bias` has shape `(528,)`.
2. **Do a 10k-step smoke train** (`TOTAL_TIMESTEPS=1e4`) to verify the training loop runs without shape errors. Check that the loss decreases at all.
3. **Unit-check the branch with `use_sync=True`** hasn't changed — baseline checkpoint should still load and evaluate identically. Load the existing `smax_mappo_ctm_actor.pkl` via the updated code path and re-run `analyse_sync.py --num-episodes 5` — results should match a prior small run exactly.

---

## Training Run Specification

| Hyperparameter | Value |
|---|---|
| Map | `3m` |
| Seed | `42` |
| Total env steps | `3e6` |
| `CTM_USE_SYNC` | `False` |
| All other CTM/training hyperparams | Identical to baseline (see [train_mappo_ctm.py:498-535](smax_ctm/train_mappo_ctm.py#L498-L535)) |
| Output checkpoint | `model/smax_mappo_ctm_actor_nosync.pkl` |
| Runtime target | Same wall-clock as baseline (~few minutes on Colab GPU) |

**Reproducibility:** Run with the exact same seed (42) as the baseline. If the no-sync model still trains to comparable win rate, seed variance is not the issue — repeat with 2–3 additional seeds only if the result is surprising.

---

## Analysis Run Specification

Run `analyse_sync.py` on the new checkpoint with the **same arguments** used for the baseline Stage-4 run so results are directly comparable:

```bash
python smax_ctm/analyse_sync.py \
    --checkpoint ../model/smax_mappo_ctm_actor_nosync.pkl \
    --output-dir ../analysis_results_nosync \
    --num-episodes 30 \
    --seed 42 \
    --max-lag 12 \
    --lead-window 3 \
    --num-permutations 5000 \
    --non-strict-outcomes
```

Expected outputs (mirrors baseline):
```
analysis_results_nosync/
  episode_traces.pkl
  sync_metrics.pkl
  event_stats.pkl
  event_lag_profiles.pkl
  outcome_diagnostics.pkl
  figures/
    sync_timeseries_ep{0,1,2}.png
    sync_heatmap_ep0.png
    neuron_activation_heatmap_ep0.png
    event_conditional.png
    event_lag_profiles.png
    sync_vs_outcome.png
```

---

## Comparison Table (to fill in after both runs)

| Metric | Baseline (3m, sync) | No-Sync (3m) | Delta |
|---|---|---|---|
| Win rate | ~84% | ? | ? |
| Mean return | 1.82 | ? | ? |
| Mean cross-agent coord-vec correlation | 0.63 | ? | ? |
| Grouping event delta | +0.097 (p=0.0002) | ? | ? |
| Focus fire lead-lag delta | 0.126 (p=0.0002) | ? | ? |
| Grouping lead-lag delta | 0.044 (p=0.0002) | ? | ? |
| Enemy kill lead-lag delta | 0.171 (p=0.0002) | ? | ? |

---

## Decision Gate (after analysis completes)

### Case A: Win rate drops AND lead-lag signal disappears
**Best case.** Clean causal evidence. Proceed to:
- Write the ablation results into [docs/stage1to3_results.md](docs/stage1to3_results.md) (or a new `stage5a_results.md`).
- Move to 5m_vs_6m baseline training (Step 2 of the overall plan).

### Case B: Win rate drops but lead-lag signal persists
**Interesting but confusing.** Sync helps performance but isn't the only mechanism producing predictive coupling. Investigate:
- Maybe the projected trace still captures enough temporal structure.
- Consider a stronger ablation (freeze `trace_proj` at random init, or replace with identity → 640-dim actor head).

### Case C: Win rate unchanged
**The 3m task doesn't require the sync mechanism.** This is plausible given 3m is trivially easy. Do **not** conclude sync is useless — instead:
- Immediately pivot to 5m_vs_6m ablation. Harder tasks stress coordination harder and should reveal the mechanism if it's real.
- Optionally re-run 3m ablation with reduced capacity (`d_model=64`) to see if the ablation only matters under pressure.

### Case D: Win rate drops AND all sync metrics drop
**Also good.** Confirms sync is load-bearing both for performance and for cross-agent coupling. Proceed as in Case A.

---

## Risk & Mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| 3m is too easy to distinguish ablation effects | Medium | Pre-commit to running 5m_vs_6m ablation next regardless of 3m outcome |
| Param tree change breaks checkpoint loading | Low | Sanity check in Step 5 — load baseline ckpt with new code before training |
| The 528-dim Dense projection is "too powerful" and can emulate sync | Low | Single linear layer cannot express the quadratic pairwise products of sync; mathematically distinct |
| Colab training variance hides effect | Low | Same seed as baseline; add 2 extra seeds only if result is within noise band |
| Analysis script mislabels ablation figures | Low | Explicit print + title suffix in Step 4 |

---

## Implementation Order (checklist)

1. [ ] Edit [ctm_jax.py](smax_ctm/ctm_jax.py): add `use_sync` field to `CTMCell`, add branch, propagate via `ScannedCTM`.
2. [ ] Edit [train_mappo_ctm.py](smax_ctm/train_mappo_ctm.py): add `CTM_USE_SYNC` to config, add filename suffix logic.
3. [ ] Edit [collector.py](smax_ctm/analysis/collector.py): propagate `use_sync` from loaded config into `CTMCell` construction + add log line.
4. [ ] Edit [analyse_sync.py](smax_ctm/analyse_sync.py): detect ablation mode + label figures.
5. [ ] **Sanity check:** verify `use_sync=True` baseline checkpoint still loads and produces identical metrics.
6. [ ] **Smoke test:** 10k-step training run with `CTM_USE_SYNC=False` to verify shapes.
7. [ ] **Full training run:** 3M steps, save to `smax_mappo_ctm_actor_nosync.pkl`.
8. [ ] **Analysis run:** `analyse_sync.py` with same args as baseline Stage-4 run.
9. [ ] **Write up results** in `docs/stage5a_results.md` using the comparison table above.
10. [ ] **Decision:** proceed to 5m_vs_6m based on outcome (see Decision Gate).

---

## Notes
- This ablation does NOT require touching the critic, the loss, the rollout logic, or the GRU baseline.
- All changes are gated behind a config flag — `CTM_USE_SYNC=True` must remain the default everywhere so existing checkpoints and analysis runs are untouched.
- Preserve the "sync" naming in output files for backward compatibility; use metadata + figure titles to disambiguate ablation runs.
