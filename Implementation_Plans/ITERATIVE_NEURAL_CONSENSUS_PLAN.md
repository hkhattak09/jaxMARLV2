# Sync-Based Cross-Agent Coordination — Implementation Plan

**Paper working title:** *Sync-Alignment Losses for Coordinated Multi-Agent Policies*
**Primary method:** **Sync-Alignment Auxiliary Loss (SAAL).** A training-time loss that
encourages per-environment cosine similarity between teammates' CTM sync vectors
to be higher on coordination events than on non-coordination steps. **Fully
decentralised at execution** — the loss only touches gradients during training,
and at test time each agent runs a vanilla CTM-MAPPO with zero inter-agent
bandwidth.
**Comparison method:** **Iterative Neural Consensus (INC).** Between-iteration
cross-agent pooling of sync vectors *inside* a single env step. Implemented in
Stages 0–2.5. Useful as a comparison condition — it exposes how much of the
coordination gain comes from an execution-time channel vs. a training-time
alignment signal.
**Benchmarks:** SMAX 3m (fast iteration) → SMAX 2s3z + 5m_vs_6m (main matrix) →
Hanabi 2p + 4p (held back until SAAL lands on SMAX — see Stage 7).

**How this document relates to the research direction doc:**
[docs/iterative_neural_consensus.md](../docs/iterative_neural_consensus.md) is the
"what we're building and why" for a new collaborator. This document is the
"exactly what to do next and which files to touch" plan.

---

## History before the pivot (done and still valid)

Stages 0–2.5 and the completed parts of Stage 3 built the infrastructure — CTM
refactor, `AgentConsensus` module, Hanabi env wiring, unit tests, and the
Stage 2.1 disambiguation experiment. **None of that work is being thrown away.**
The SAAL implementation sits on top of the same CTM code path; Hanabi wiring is
reused verbatim for the eventual Hanabi runs; the Stage 2.1 analysis framework
(per-seed final WR, frac-over-threshold, curve-slope comparisons) is the same
methodology we'll use for SAAL on SMAX.

What changed: **INC is no longer the paper's headline method**, because of a
cheating concern on info-restricted benchmarks (Hanabi) that we did not surface
at the start of the project. See
[docs/iterative_neural_consensus.md](../docs/iterative_neural_consensus.md) §4
for the full pivot rationale. Stage 2.1 data is still load-bearing — it showed
the pooled channel is real on SMAX — and now frames the motivation for SAAL
("the same alignment phenomenon, but trained into the model rather than piped
between models at execution").

---

## Stage 0 — Pre-flight checks ✅ DONE

**Goal:** confirm assumptions and baseline numbers before touching the architecture.

- [x] Re-run SMAX 3m CTM-MAPPO baseline with `CTM_ITERATIONS=1` and confirm the
  ~84% WR number. Canonical baseline seeded and checkpointed.
- [x] Verify `CTMCell.__call__` batch layout: inside `ScannedCTM`, the flat batch
  dim is `NUM_ACTORS = num_agents * num_envs`, **agent-major**.
  `check_batchify_order.py` and `check_ctm_axis_layout.py` both pass; notes in
  [docs/inc_axis_convention.md](../docs/inc_axis_convention.md).
- [x] Read and document upstream JaxMARL Hanabi env; obs/action/reward shapes in
  [docs/hanabi_env_notes.md](../docs/hanabi_env_notes.md).
- [x] Identify the upstream MAPPO/IPPO Hanabi baseline config as the reference
  number to beat.

**Exit criterion (met):** baseline reproduces; agent/env axis convention
documented.

---

## Stage 1 — CTMCell refactor: per-iteration sync + optional consensus input ✅ DONE

**Goal:** iterate-by-iterate internal loop emitting per-iteration sync. No
behavioural change when `consensus_in=None`.

**Files touched:** [smax_ctm/ctm_jax.py](../smax_ctm/ctm_jax.py),
[smax_ctm/train_mappo_ctm.py](../smax_ctm/train_mappo_ctm.py),
[smax_ctm/train_mappo_ctm_hanabi.py](../smax_ctm/train_mappo_ctm_hanabi.py).

- [x] `CTMCell._single_iter(state_trace, activated_state_trace, features,
  consensus_in)` helper added; returns `(new_state_trace,
  new_activated_state_trace, synch)`.
- [x] Consensus input is concatenated with `[features, last_activated]` before
  synapses. Seeded with zeros on iteration 0 (when `INC_ENABLED=True`) so the
  `nn.Dense` kernel shape is stable across iterations.
- [x] No-op equivalence: with `INC_ENABLED=False` and `CTM_ITERATIONS=1`, the
  refactor reproduces pre-change numbers bit-for-bit.

**Exit criterion (met):** refactor is a no-op at the Stage 1 config.

---

## Stage 2 — Cross-agent pooling via `AgentConsensus` ✅ DONE

**Goal:** introduce the INC consensus mechanism behind `INC_ENABLED`. Off by
default; fully Stage-1-compatible when off.

**Files touched:** [smax_ctm/ctm_jax.py](../smax_ctm/ctm_jax.py),
[smax_ctm/train_mappo_ctm.py](../smax_ctm/train_mappo_ctm.py),
[smax_ctm/train_mappo_ctm_hanabi.py](../smax_ctm/train_mappo_ctm_hanabi.py).

- [x] Config keys: `INC_ENABLED`, `INC_POOLING ∈ {mean, attention, gated}`,
  `INC_NUM_AGENTS` (auto from env), `INC_CONSENSUS_DROPOUT` (default 0).
- [x] `AgentConsensus` Flax module: leave-one-out pooling on agent-major tensors,
  dead-agent masking, three pooling variants.
- [x] `CTMCell` wires the pool between iterations, agent-major reshape verified
  by unit tests.
- [x] Minibatch permutation preserves agent grouping (regression-tested).

**Exit criterion (met):** at `INC_ENABLED=True, INC_POOLING="mean",
CTM_ITERATIONS=3`, SMAX 3m training runs without shape errors. Observed: plain
INC is slightly better than baseline; INC + 0.25 consensus dropout is
substantially better (final WR 0.822 vs 0.793, sustained-WR frac 41.5% vs 4.9%).

---

## Stage 2.1 — Disambiguating the dropout win ✅ DONE

**Goal:** identify whether the cell-C win comes from (a) the pooled channel, (b)
stochastic regularisation on the iteration loop, or (c) both.

**Files touched:** [smax_ctm/ctm_jax.py](../smax_ctm/ctm_jax.py) (added
`CTM_ITER_DROPOUT`, `INC_FORCE_ZERO_CONSENSUS` flags),
[smax_ctm/scripts/run_stage2_1_disambig.py](../smax_ctm/scripts/run_stage2_1_disambig.py),
[smax_ctm/tests/test_inc.py](../smax_ctm/tests/test_inc.py),
`analysis_results_inc/stage2_1/`,
[docs/inc_stage2_1_findings.md](../docs/inc_stage2_1_findings.md).

**Cells run** (3 seeds each, SMAX 3m, 3M timesteps):

| Cell | `INC_ENABLED` | `INC_CONSENSUS_DROPOUT` | `CTM_ITER_DROPOUT` | `FORCE_ZERO_CONSENSUS` | Final WR | Frac ≥ 0.8 |
|---|---|---|---|---|---|---|
| A | False | — | 0.0 | — | 0.793 | 4.9% |
| B | True  | 0.0 | 0.0 | False | 0.811 | 12.0% |
| C | True  | 0.25 | 0.0 | False | **0.822** | **41.5%** |
| D | False | — | 0.25 | — | 0.745 | 1.1% |
| E | True | 0.25 | 0.0 | True | 0.802 | 14.9% |

**Outcome:** Story 1 ("consensus channel is load-bearing") supported. Cell D
regresses *below* the no-INC baseline — pure iteration-loop stochasticity is a
net negative, not a neutral control. Cell E matches B on final WR and stays well
below C on curve-quality metrics, reverting to the no-INC regime when teammate
information is stripped. Full analysis in
[docs/inc_stage2_1_findings.md](../docs/inc_stage2_1_findings.md).

**Consequence for the new plan:** the pooled channel is demonstrably load-bearing
on SMAX for *sustained* high-performance coordination. On Hanabi, the same
channel is what raises the cheating concern. SAAL is the natural way to exploit
this alignment phenomenon without an execution-time channel — see Stages 4–6.

---

## Stage 2.5 — Unit tests for INC plumbing ✅ DONE

All invariants locked into
[smax_ctm/tests/test_inc.py](../smax_ctm/tests/test_inc.py): axis round-trip,
leave-one-out mean, no-op equivalence, dead-agent masking, gradient flow,
minibatch permutation safety, Stage 2.1 force-zero + gradient-still-flows test.

---

## Stage 3 — Hanabi environment wiring ✅ DONE (training held back)

**Goal:** CTM-MAPPO and GRU-MAPPO runnable on Hanabi.

**Files touched:** [jaxmarl/environments/hanabi/](../jaxmarl/environments/hanabi/),
[smax_ctm/train_mappo_ctm_hanabi.py](../smax_ctm/train_mappo_ctm_hanabi.py),
[smax_ctm/train_mappo_gru_hanabi.py](../smax_ctm/train_mappo_gru_hanabi.py),
[smax_ctm/test_and_logger/run_hanabi_tests.py](../smax_ctm/test_and_logger/run_hanabi_tests.py).

- [x] Upstream FLAIROx/JaxMARL Hanabi env imported and registered.
- [x] Hanabi-specific CTM and GRU training scripts with `HanabiWorldStateWrapper`,
  dict-returning `get_legal_moves` handling, score-out-of-25 logging, CTM
  reset-on-done `last_done` initialisation.
- [x] Env contract tests pass (reset/step shapes, legal-moves shape, 3-player
  config).
- [x] Sanity smoke tests: 90k-timestep INC-on and INC-on-with-dropout runs
  complete without shape/crash errors; score is still ~0 at that budget (expected
  — Hanabi needs tens of millions of steps).

**Why training is held back:** a full Hanabi run at `CTM_ITERATIONS=3` costs
~6.5h per seed. Running the Hanabi matrix before SAAL exists on SMAX would spend
compute refining a story we are pivoting away from. Hanabi training resumes in
Stage 7 once SAAL is validated on SMAX and has a Hanabi-specific coord-event
detector.

**Exit criterion (met):** Hanabi wiring runs end-to-end; both GRU and CTM code
paths load, step, and train. Full-budget training deferred to Stage 7.

---

## Stage 4 — SAAL pair-cosine logging pass ✅ DONE

**Goal:** before committing to loss weights, measure the baseline cross-agent
`pair_cos` distribution on a vanilla CTM-MAPPO SMAX 3m run. With parameter
sharing, sync vectors already have some natural similarity; we need to know
whether that baseline is 0.1 (lots of headroom) or 0.9 (no headroom) before
picking `ALIGN_ALPHA`.

**This stage writes no loss.** It adds logging only. Zero risk of affecting
training dynamics. Outcome is a number (or a histogram) that unblocks Stage 5.

**Files touched:**
- [smax_ctm/train_mappo_ctm.py](../smax_ctm/train_mappo_ctm.py): expose sync from
  `ActorCTM`, compute and log `pair_cos` in `_actor_loss_fn`.

**Implementation:**

- [ ] **Expose sync from `ActorCTM`.** Change
  [train_mappo_ctm.py:139-159](../smax_ctm/train_mappo_ctm.py#L139) so
  `ActorCTM.__call__` returns `(hidden, pi, synch)` instead of `(hidden, pi)`.
  `synch` is already locally available — it's fed to the policy head on line 149.
  Update every call site of `actor_network.apply` to unpack three values
  (rollout step and `_actor_loss_fn`).
- [ ] **Capture env constants into config.** In `make_train` after
  [train_mappo_ctm.py:279](../smax_ctm/train_mappo_ctm.py#L279), stash
  `config["NUM_MOVEMENT_ACTIONS"] = int(env.num_movement_actions)` and
  `config["NUM_ENEMIES"] = int(env.num_enemies)`. These are needed for the
  focus-fire mask helper.
- [ ] **Helper function `compute_focus_fire_mask(actions, num_agents, num_envs,
  num_movement_actions, num_enemies)`.** Pure `jnp`. Input: action array shape
  `(T, num_agents * num_envs)`. Reshape to `(T, num_agents, num_envs)`, mark
  entries where action ≥ `num_movement_actions` and ≤
  `num_movement_actions + num_enemies - 1` as "attacking enemy k". For each
  `(t, env)`, return True iff ≥2 agents target the same enemy. Logic mirrors
  [analysis/collector.py:73-89](../smax_ctm/analysis/collector.py#L73). Place in
  [train_mappo_ctm.py](../smax_ctm/train_mappo_ctm.py) near the other helpers.
- [ ] **Compute `pair_cos` in the actor loss fn.** In `_actor_loss_fn`
  ([train_mappo_ctm.py:462](../smax_ctm/train_mappo_ctm.py#L462)), after
  unpacking `synch`:
  ```python
  T = synch.shape[0]
  num_agents = config["INC_NUM_AGENTS"]
  num_envs_mb = synch.shape[1] // num_agents
  synch_am = synch.reshape(T, num_agents, num_envs_mb, -1)
  s_norm = synch_am / (jnp.linalg.norm(synch_am, axis=-1, keepdims=True) + 1e-8)
  cos_mat = jnp.einsum("taec,tbec->teab", s_norm, s_norm)  # (T, E, A, A)
  iu, ju = jnp.triu_indices(num_agents, k=1)
  pair_cos = cos_mat[..., iu, ju].mean(axis=-1)            # (T, E)

  ff_mask = compute_focus_fire_mask(
      traj_batch.action, num_agents, num_envs_mb,
      config["NUM_MOVEMENT_ACTIONS"], config["NUM_ENEMIES"],
  )                                                        # (T, E) bool

  pair_cos_all  = pair_cos.mean()
  pair_cos_ff   = jnp.where(ff_mask, pair_cos, 0.0).sum() / (ff_mask.sum() + 1e-8)
  pair_cos_nff  = jnp.where(~ff_mask, pair_cos, 0.0).sum() / ((~ff_mask).sum() + 1e-8)
  ff_frac       = ff_mask.mean()
  ```
  **Crucial:** do NOT add `pair_cos` to `actor_loss`. This stage is logging only.
- [ ] **Add fields to `loss_info`** at
  [train_mappo_ctm.py:519](../smax_ctm/train_mappo_ctm.py#L519): `pair_cos_all`,
  `pair_cos_ff`, `pair_cos_nff`, `ff_frac`. Confirm they print to stdout at each
  log step.
- [ ] **Unit test `compute_focus_fire_mask`** in
  [smax_ctm/tests/test_inc.py](../smax_ctm/tests/test_inc.py) on a hand-crafted
  action array with known events. 2–3 cases: no attacks (all False), 2 agents
  target same enemy (True), 2 agents target different enemies (False).
- [ ] **Run SMAX 3m, 1 seed, same budget as Stage 2 cell A** with
  `INC_ENABLED=False, CTM_ITERATIONS=1`. This is the baseline CTM-MAPPO. Collect
  the `pair_cos_all / pair_cos_ff / pair_cos_nff` curves throughout training.
- [ ] **Write interpretation note** at
  [docs/saal_stage4_baseline_paircos.md](../docs/saal_stage4_baseline_paircos.md):
  what are the baseline values, do `ff` and `nff` diverge spontaneously (the
  Stage 1-3 phenomenon showing up in the training loop directly), how much
  headroom is there for an alignment loss to push.

**Exit criterion:**
- Logging fields appear in training output and are finite across the whole run.
- A number for baseline `pair_cos` at converged training exists.
- A decision on `ALIGN_ALPHA` / `ALIGN_BETA` starting points is written into
  Stage 5.

**Decision rule after Stage 4:**
- If baseline `pair_cos_ff` is already very close to 1.0 (say ≥ 0.9), SAAL has
  almost no room to push — revisit the loss formulation (maybe align to a target
  subspace, not a scalar cosine).
- If baseline `pair_cos_ff ≈ pair_cos_nff`, the model is not spontaneously
  distinguishing coord from non-coord steps — SAAL has a clear job to do and
  Story 1-3 is intact in the training distribution.
- If baseline `pair_cos_ff` > `pair_cos_nff` by a meaningful margin, the model is
  already learning the alignment on its own — SAAL may only sharpen the effect.
  Set `ALIGN_ALPHA` conservatively.

**Observations (run on 2026-04-11, 1 seed, ~3M steps, `INC_ENABLED=False`,
`CTM_ITERATIONS=1`, final WR 0.82 — reproduces Stage 0 baseline):**

| Phase | WR | `pair_cos_all` | `pair_cos_ff` | `pair_cos_nff` | gap (ff−nff) | ff_frac |
|---|---|---|---|---|---|---|
| init (step 0) | 0.00 | 0.763 | 0.854 | 0.763 | **+0.091** | 0.003 |
| early (<0.5M) | 0.30 | 0.624 | 0.673 | 0.612 | +0.060 | 0.22 |
| mid (0.5–1.5M) | 0.68 | 0.566 | 0.583 | 0.554 | +0.029 | 0.42 |
| late (≥2.5M) | 0.80 | 0.652 | 0.662 | 0.642 | +0.020 | 0.47 |
| final10 | 0.80 | 0.650 | 0.662 | 0.639 | **+0.023** | 0.47 |

1. **Headroom exists.** Converged `pair_cos_ff ≈ 0.66`, nowhere near the 0.9
   ceiling. SAAL has room to push.
2. **Parameter-sharing floor.** At init, both `ff` and `nff` sit at ~0.76 — the
   non-zero baseline similarity is a weight-sharing artefact, not coordination.
3. **`ff > nff` holds almost everywhere** (181/183 log points). The Stage 1–3
   alignment phenomenon *does* surface in the training distribution of a vanilla
   CTM-MAPPO — but the margin is small.
4. **The gap shrinks over training** (0.06 → 0.02). Whatever spontaneous
   alignment the model discovers early is partially *unlearned* as entropy
   collapses and the policy sharpens. This is a general PG / parameter-sharing
   homogenisation pressure, independent of INC — and it reframes the Stage 2.1
   `INC_CONSENSUS_DROPOUT=0.25` win: dropout helps precisely because it *resists
   this collapse* by preventing agents from leaning on a constant pooled
   teammate vector. SAAL is the training-time analogue — its β term exists to
   push back on exactly this homogenisation.
5. **Non-monotonic trajectory.** All three cosines dip to ~0.55 mid-training
   (~1M steps) then rise back to ~0.65 — an entropy/exploration artefact rather
   than a coordination signal. SAAL should be robust to this U-shape.
6. **`ff_frac` tracks WR cleanly** (0.003 → 0.47). Focus-fire detector fires
   meaningfully once agents start winning, so the `ff` bucket is not empty
   during the informative part of training.

**Decision rule match:** case 3 (small-but-real positive gap). Starting
hyperparameters for Stage 5: **`ALIGN_ALPHA = 0.05, ALIGN_BETA = 0.025`**
(conservative as the decision rule prescribes). Rationale: the existing gap is
only ~0.02, so a loss term that dominates the PG signal would over-steer. See
Stage 5 for the full hyperparameter discussion.

---

## Stage 5 — SAAL loss implementation + SMAX 3m validation 🟢 NEXT

**Goal:** add the sync-alignment auxiliary loss and validate on SMAX 3m. Single
seed sanity run first, then 3-seed comparison against the Stage 1 baseline
(Stage 2.1 cell A).

**Prerequisite:** Stage 4 logging pass complete (done). Hyperparameters locked
post-Stage-4 (see "Locked hyperparameters" below).

**Files touched:**
- [smax_ctm/train_mappo_ctm.py](../smax_ctm/train_mappo_ctm.py) — loss assembly
  and config keys.
- [smax_ctm/tests/test_inc.py](../smax_ctm/tests/test_inc.py) — loss-gate unit
  test, gradient-flow test.
- [smax_ctm/scripts/run_saal_stage5.py](../smax_ctm/scripts/run_saal_stage5.py)
  (new) — sanity + 3-seed runner, modelled on
  [run_stage2_1_disambig.py](../smax_ctm/scripts/run_stage2_1_disambig.py).
- [docs/saal_stage5_findings.md](../docs/saal_stage5_findings.md) (new) —
  results write-up.

**Loss definition:**

```
L_align  = - ALIGN_ALPHA * mean_{(t,e) ∈ focus_fire} pair_cos(t, e)
           + ALIGN_BETA  * mean_{(t,e) ∉ focus_fire} pair_cos(t, e)

actor_loss_total = loss_actor - ENT_COEF * entropy + L_align
```

The β term prevents the degenerate solution of making sync vectors
agent-invariant under parameter sharing. It is also doing load-bearing work
against the Stage-4 shrinking-gap phenomenon (PG / parameter sharing actively
*homogenises* sync vectors over training; the β term resists this). See
[docs/iterative_neural_consensus.md](../docs/iterative_neural_consensus.md) §5
for the derivation.

**Locked hyperparameters (post-Stage-4):**

- `ALIGN_ALPHA = 0.05`
- `ALIGN_BETA  = 0.025` (i.e. β = α/2)

Rationale: Stage 4 measured a converged `pair_cos_ff − pair_cos_nff` gap of
only ~0.02. A loss term on the order of the actor PG loss (~0.03) is a
meaningful but non-dominant nudge — `α = 0.05` multiplying a cosine in [0, 1]
lands exactly in that range. `β = α/2` is conservative: `nff ≈ 0.64` is not
dangerously close to `ff ≈ 0.66`, so anti-collapse pressure can be smaller
than the positive term for the sanity run. A wider β/α ratio sweep is
deferred to Stage 6.

**Explicitly deferred to Stage 6:**

- β/α ratio sweep (`0.0, 0.25, 0.5, 1.0`).
- `ALIGN_ALPHA` magnitude sweep (`0.01, 0.05, 0.1`).
- Any curriculum / ramp on α. (Judgement call: the Stage-4 U-shape suggests a
  ramp might help, but adding a schedule hyperparameter before we know
  constant-α works is premature. Revisit only if constant-α fails on SMAX 3m
  *and* the simpler fallback list below doesn't recover it. May not be done
  at all.)

**Implementation:**

- [ ] **New config keys** in `make_train`:
  - `ALIGN_ENABLED` (bool, default `False`)
  - `ALIGN_ALPHA` (float, default `0.0`)
  - `ALIGN_BETA` (float, default `0.0`)
- [ ] **Loss assembly.** In `_actor_loss_fn`, behind `if config["ALIGN_ENABLED"]:`:
  ```python
  align_pos = jnp.where(ff_mask, pair_cos, 0.0).sum() / (ff_mask.sum() + 1e-8)
  align_neg = jnp.where(~ff_mask, pair_cos, 0.0).sum() / ((~ff_mask).sum() + 1e-8)
  L_align   = -config["ALIGN_ALPHA"] * align_pos + config["ALIGN_BETA"] * align_neg
  actor_loss = actor_loss + L_align
  ```
  `align_pos` and `align_neg` are the **same quantities** Stage 4 already
  computes for logging — reuse them; do not recompute. When
  `ALIGN_ENABLED=False`, `L_align` is not computed at all (keeps the jit graph
  identical to Stage 4 so pre- and post-Stage-5 baselines remain bit-for-bit
  comparable).
- [ ] **Logging.** Add to `loss_info`:
  - `L_align` (total weighted term)
  - `align_pos` (raw weighted-positive scalar: `-α · pair_cos_ff`)
  - `align_neg` (raw weighted-negative scalar: `+β · pair_cos_nff`)
  - **Keep Stage 4's raw `pair_cos_all / pair_cos_ff / pair_cos_nff / ff_frac`
    live during SAAL runs** (they remain unchanged — we already log them). This
    is the load-bearing diagnostic: SAAL should visibly *move the underlying
    cosine*, not just the weighted sum. If `L_align` drops but `pair_cos_ff`
    doesn't rise, something is wrong with the gradient path.
- [ ] **Loss-gate unit test** in
  [tests/test_inc.py](../smax_ctm/tests/test_inc.py):
  - Forward-pass with `ALIGN_ENABLED=True, ALIGN_ALPHA=0.05, ALIGN_BETA=0.025`,
    confirm `L_align` is finite.
  - Confirm sign: with a synthetic batch where `pair_cos_ff > pair_cos_nff`,
    `L_align = -α·pos + β·neg` should be negative (we want to *minimise* this,
    which *increases* the ff term).
  - Gradient-flow test: `jax.grad` of `L_align` w.r.t. actor params produces
    non-zero gradients on `Synapses`, `NLMs`, and `decay_params_out` (the three
    CTM param groups that feed the sync vector). Cross-check: gradient on
    `critic` params is zero (SAAL must not leak into the critic).
  - Off-switch test: with `ALIGN_ENABLED=False`, the loss value and gradient
    match the Stage-4 baseline exactly (regression guard against accidental
    jit-graph divergence).
- [ ] **Sanity training run.** SMAX 3m, 1 seed (seed `200`), same budget as
  Stage 2.1 cell A (≈3M timesteps), `ALIGN_ENABLED=True`, `ALIGN_ALPHA=0.05`,
  `ALIGN_BETA=0.025`, `INC_ENABLED=False`, `CTM_ITERATIONS=1`. Confirm:
  - Run completes without NaN and without entropy collapse.
  - `pair_cos_ff` at convergence rises **meaningfully above the Stage-4
    baseline of 0.66** (target: ≥ 0.70).
  - `L_align` is monotonic-ish downward over training and is the expected
    magnitude (|L_align| < actor_loss throughout).
  - Final WR is at least not *worse* than Stage-2.1 cell A (0.793) —
    regression guard.
  - If any of these fail, halt before the 3-seed run and diagnose.
- [ ] **3-seed comparison run.** Same config, seeds `[201, 202, 203]`
  (non-overlapping with Stage 2.1's `[103, 104, 105]`). Compare against the
  3-seed Stage-2.1 cell A baseline (already on disk). Runner should mirror
  [run_stage2_1_disambig.py](../smax_ctm/scripts/run_stage2_1_disambig.py): one
  cell ("SAAL-α05β025"), 3 seeds sequentially, per-seed WR/loss/cosine dumps
  into `analysis_results_inc/stage5/`.
- [ ] **Metrics reported** (same framework as Stage 2.1 for direct
  comparability):
  - Final WR (mean ± std across seeds).
  - Frac of updates with WR ≥ 0.8.
  - First update where rolling-20 WR ≥ 0.8.
  - Final `pair_cos_ff`, `pair_cos_nff`, and gap at convergence.
  - `L_align` trajectory plot (average across seeds).
- [ ] **Write findings note** at
  [docs/saal_stage5_findings.md](../docs/saal_stage5_findings.md). Report the
  3-seed table, plot the cosine trajectories overlaid on the Stage-4 baseline,
  state explicitly which pre-registered bar was hit.

**Pre-registered success bars** (locked before the 3-seed run — written here
so post-hoc reinterpretation is visibly dishonest):

| Outcome | Criterion (both conditions required) |
|---|---|
| **Clear win** | final WR ≥ 0.822 **and** frac(WR ≥ 0.8) > 20% |
| **Minimum viable** | final WR ≥ 0.81 **and** `pair_cos_ff` at convergence ≥ 0.70 |
| **Null** | final WR within ±0.01 of cell A (0.793) **and** `pair_cos_ff` lift < 0.02 |
| **Regression** | final WR < cell A − 0.01 **or** NaN / entropy collapse during training |

- **Clear win** → proceed directly to Stage 6.
- **Minimum viable** → proceed to Stage 6 but prioritise the α/β sweep; a
  stronger α may be leaving performance on the table.
- **Null** → enter the fallback list below. Do not proceed to Stage 6 until a
  retune produces at least "minimum viable".
- **Regression** → hard stop; debug the loss wiring or revisit the formulation.

**Exit criterion:** SAAL sanity run completes without NaN/entropy collapse;
3-seed SMAX 3m comparison hits at least the "minimum viable" bar, and the
findings doc records which bar was hit with per-seed numbers. A go/no-go for
Stage 6 is written at the bottom of the findings doc.

**If SAAL fails on SMAX 3m (null or regression):** options to try in order:

1. Hyperparameter retune: sweep `ALIGN_ALPHA ∈ {0.01, 0.05, 0.1, 0.2}` with
   `ALIGN_BETA = ALIGN_ALPHA / 2`. Cheapest fix — no code changes.
2. Ratio retune: with whichever `ALIGN_ALPHA` did best in (1), sweep
   `ALIGN_BETA / ALIGN_ALPHA ∈ {0.0, 0.5, 1.0, 2.0}`. Informs whether the
   collapse-prevention term is doing the work or the positive alignment term is.
3. Event-detector widening: add grouping and enemy-kill events (requires
   plumbing raw SMAX state through `Transition` — ~50 lines).
4. Advantage-gated formulation: replace `ff_mask` with `advantage > 0` as a
   soft gate. Cheaper than plumbing state; not tied to hand-crafted events.
5. (Last resort) α curriculum: start `ALIGN_ALPHA` at 0, ramp linearly to 0.05
   over the first 500k steps. Only try if (1)–(4) fail — this adds a schedule
   hyperparameter that's hard to defend in the paper.

If none of these work, SAAL is dead as an idea and we return to the direction
doc for a re-plan before spending more compute.

---

## Stage 6 — SAAL SMAX main matrix + hyperparameter sweep

**Goal:** lock in SAAL's performance envelope on SMAX before touching Hanabi.

**Prerequisite:** Stage 5 shows a statistically meaningful SAAL > baseline win on
SMAX 3m.

**Cells (SMAX, 3 seeds each unless noted):**

| Map | Methods |
|---|---|
| 3m | CTM baseline, CTM + SAAL, CTM + INC(cell C), CTM + SAAL + INC |
| 2s3z | same four methods |
| 5m_vs_6m | same four methods |

The `CTM + SAAL + INC` cell is the "does SAAL and INC combine additively" test —
useful regardless of whether INC ends up in the paper, because it tells us
whether the two interventions are tapping the same underlying coordination
signal or different ones.

**Hyperparameter sweep (on SMAX 3m only, 2 seeds per cell):**
- `ALIGN_ALPHA ∈ {0.01, 0.05, 0.1}` (keeping `ALIGN_BETA = ALIGN_ALPHA / 2`)
- `ALIGN_BETA / ALIGN_ALPHA ∈ {0.0, 0.25, 0.5, 1.0}` at chosen `ALIGN_ALPHA`

Pick a single `(ALIGN_ALPHA, ALIGN_BETA)` and lock it in before running 2s3z and
5m_vs_6m.

- [ ] Write runner `smax_ctm/scripts/run_saal_smax_matrix.py`.
- [ ] Run the sweep on 3m, pick the best `(alpha, beta)`.
- [ ] Run the full 2-map × 4-method × 3-seed matrix.
- [ ] Write findings note at
  [docs/saal_stage6_smax_matrix.md](../docs/saal_stage6_smax_matrix.md).

**Exit criterion:** SAAL's SMAX performance across three maps is characterised;
`(ALIGN_ALPHA, ALIGN_BETA)` is locked; a go/no-go decision on porting SAAL to
Hanabi is made.

---

## Stage 7 — SAAL Hanabi port

**Goal:** port SAAL to Hanabi. Needs a Hanabi-specific coord-event detector
because focus-fire does not exist there.

**Prerequisite:** Stage 6 shows SAAL wins on SMAX across at least two maps.

**Files touched:**
[smax_ctm/train_mappo_ctm_hanabi.py](../smax_ctm/train_mappo_ctm_hanabi.py), new
Hanabi event detector helper.

**Hanabi coord event candidates** (pick one or both for V1):

1. **Hint-then-play.** An agent gives a hint on turn `t`, the recipient plays a
   card on turn `t+1` (or within `k` turns). The `(t, t+1)` pair is a coord event;
   label both steps as "in coord event" for the current hand.
2. **Successful-play reward.** On any step where `reward > 0`, mark the previous
   `k` steps as "in coord event." Captures the "something good just happened,
   the reasoning was probably shared" intuition without needing to parse hint
   semantics.

V1 proposal: option 2 (reward-lookback). Simpler, environment-agnostic, works
with the data already in `Transition`. Option 1 is a cleaner signal but requires
parsing Hanabi's action space into hint vs play vs discard.

- [ ] Implement `compute_hanabi_coord_event_mask(rewards, num_agents, num_envs,
  lookback_k)` as a pure `jnp` helper.
- [ ] Wire SAAL into
  [train_mappo_ctm_hanabi.py](../smax_ctm/train_mappo_ctm_hanabi.py) with the
  same structure as Stage 5. Reuse `ALIGN_ALPHA / ALIGN_BETA` from Stage 6 as
  defaults; expect to retune.
- [ ] Unit tests for the event mask.
- [ ] **Sanity run:** Hanabi 2p, 1 seed, half budget (~3h). Confirm no crash,
  entropy reasonable, `pair_cos_*` logs look sane.
- [ ] **Main runs:** Hanabi 2p, 3 seeds, full budget. Cells: baseline Hanabi
  CTM-MAPPO, CTM + SAAL. `ALIGN_ALPHA / ALIGN_BETA` sweep if time permits.

**Exit criterion:** SAAL trains stably on Hanabi 2p and either beats or matches
the Hanabi CTM-MAPPO baseline on score-out-of-25.

---

## Stage 8 — INC Hanabi discriminating tests

**Goal:** run the INC-on-Hanabi cells that tell us whether INC's
coordination gain comes from a legitimate mechanism or from smuggling card
information through the execution-time channel. This is the "negative result
with teeth" that supports the pivot and makes SAAL the defensible recommendation
rather than just the author's preference.

**Prerequisite:** Stage 7 in hand. This stage is effectively "add the INC cells
to the Hanabi results table."

**Files touched:**
[smax_ctm/train_mappo_ctm_hanabi.py](../smax_ctm/train_mappo_ctm_hanabi.py) (no
code changes needed — uses existing INC flags), new cross-play eval script.

**Cells (Hanabi 2p, 3 seeds each):**

| Cell | Config | Purpose |
|---|---|---|
| H-A | baseline (already from Stage 7) | No-channel control |
| H-C | `INC_ENABLED=True, INC_CONSENSUS_DROPOUT=0.25` | INC at its best SMAX config |
| H-E | H-C + `INC_FORCE_ZERO_CONSENSUS=True` | Zero-consensus control |

Plus cross-play eval between H-C seed-0 and H-C seed-1 checkpoints.

- [ ] Write `smax_ctm/scripts/run_hanabi_inc_discrim.py` to launch H-C and H-E
  with the same hyperparameters and budget as the Stage 7 baseline.
- [ ] **Cross-play eval plumbing.** New script
  [smax_ctm/eval_hanabi_crossplay.py](../smax_ctm/eval_hanabi_crossplay.py) that:
  1. Loads two `HanabiCTM` actor checkpoints from different seeds.
  2. In rollout, dispatches actor params by agent index — agent 0 uses
     `params_A`, agent 1 uses `params_B`.
  3. Logs mean / median / std score over N eval episodes.
  4. Reports score alongside self-play (same checkpoint on both agents).
- [ ] **Pre-registered decision rules:**
  - "H-C above H-A" ≡ mean-score gap ≥ 2 points.
  - "H-E drops" ≡ H-E mean within 1 point of H-A.
  - "Cross-play collapses" ≡ cross-play mean ≥ 3 points below self-play mean.
- [ ] Write findings note at
  [docs/inc_stage8_hanabi_discrim.md](../docs/inc_stage8_hanabi_discrim.md).

**Interpretation matrix:**

| H-C vs H-A | H-E vs H-A | Cross-play | Verdict | Paper role |
|---|---|---|---|---|
| H-C ≫ A | H-E ≈ A | collapses | INC cheats via channel + private convention | Negative result, INC reported as cautionary case |
| H-C ≫ A | H-E ≈ A | holds | Channel load-bearing but conventions are generic | Mixed result; keep INC as secondary method |
| H-C ≫ A | H-E ≈ H-C | — | Channel not load-bearing, INC gain is other | INC survives as parallel method to SAAL |
| H-C ≈ A | — | — | INC doesn't transfer to Hanabi | Drop INC from Hanabi table |

**Exit criterion:** all three cells + cross-play numbers logged;
[docs/inc_stage8_hanabi_discrim.md](../docs/inc_stage8_hanabi_discrim.md) states
which verdict cell fires and how the paper's Hanabi section is framed in
consequence.

---

## Stage 9 — Ablations

**Goal:** answer the obvious reviewer questions before they're asked.

- [ ] **SAAL α/β sensitivity curve.** Already partially covered in Stage 6 sweep.
  Produce a single plot.
- [ ] **SAAL event-detector sensitivity.** If Stage 5 used focus-fire-only, re-run
  3m with the grouping and enemy-kill events added. Does the signal get stronger
  or does the extra mask dilute it?
- [ ] **SAAL under no parameter sharing.** Train with separate networks per agent
  (if feasible). Tests whether the loss is really about cross-agent alignment or
  about shared-param regularisation.
- [ ] **INC pooling type.** `mean / attention / gated` on SMAX 2s3z and Hanabi 2p,
  3 seeds each. Only if Stage 8 shows INC is surviving as a secondary method.
- [ ] **INC iteration count.** `K ∈ {1, 2, 3, 5}` with INC on.
- [ ] **Consensus dropout rate.** `{0, 0.1, 0.25, 0.5}` — already partially
  covered by Stage 2 but extend.
- [ ] **Decentralised-at-test INC.** Train centralised, eval with each agent
  pooling only over observable teammates.
- [ ] **No-sync ablation.** Replace the sync readout with a linear projection of
  the flat activated-state trace, keeping SAAL on. Tests whether the benefit
  comes specifically from the sync signal or any internal representation.

**Exit criterion:** each ablation has a clean table or plot, interpretable in one
sentence.

---

## Stage 10 — Analysis & paper figures

**Goal:** turn training logs into figures and narrative.

Figure list (tentative, in pitch order):

1. **Architecture + method diagram.** CTM sync readout with the SAAL gradient
   arrow on training-time coord events. Optional INC variant in a grey box.
2. **Motivation from Stage 1-3.** Sync rises during coord events (existing plot).
3. **SAAL main results.** Sample efficiency on SMAX 3m / 2s3z / 5m_vs_6m and
   Hanabi 2p / 4p. Four methods: GRU-MAPPO, CTM iter=1, CTM + SAAL, CTM + INC.
4. **SAAL mechanism verification.** `pair_cos_ff` vs `pair_cos_nff` over training
   for baseline vs SAAL. Shows the loss does what it says.
5. **SAAL ablation heatmap.** α × β grid on SMAX 3m.
6. **Hanabi INC cheating signature.** Cross-play vs self-play bar chart for H-C.
   The "INC fails Hanabi correctly" figure.
7. **Reliability / rliable aggregate.** IQM + probability-of-improvement over
   baseline, all envs pooled.

Tables:
- **Table 1:** full matrix, mean ± 95% CI.
- **Table 2:** compute / memory / wall-clock comparison.

Writing tasks:
- [ ] Method section: SAAL derivation, β collapse-prevention argument, focus-fire
  detector, Hanabi reward-lookback detector.
- [ ] Related work: sync-based coordination, aux losses for MARL coordination
  (Jakob Foerster's ZSC line, SAD), CommNet/TarMAC/IC3Net for contrast, CTM
  background.
- [ ] Discussion: when SAAL helps, when it doesn't, cost of parameter sharing
  assumption, Hanabi INC case study as a cautionary note on execution-time
  information channels.

**Exit criterion:** draft figures + writing ready for internal review.

---

## Stage 11 — Robustness & negative controls

- [ ] **Negative control.** Run SAAL on a task that does not require coordination
  (single-agent MPE or an independent-reward setting). SAAL should *not* help.
  If it does, the story is wrong.
- [ ] **Compute-matched GRU baseline.** Same FLOPs per step as CTM + SAAL.
  Defends against "you just have more compute."
- [ ] **Parameter-matched GRU baseline.**
- [ ] **Fresh seeds for headline numbers**, not used during development.
- [ ] **Hanabi 4p scaling.**

---

## Dependencies between stages

```
Stage 0 ──► 1 ──► 2 ──► 2.1 ──► 2.5 ──► [pivot]
                                              │
                                              ▼
                                          Stage 4 (SAAL logging)
                                              │
                                              ▼
                                          Stage 5 (SAAL SMAX 3m validation)
                                              │
                                              ▼
                                          Stage 6 (SAAL SMAX matrix)
                                              │
                                              ▼
                                          Stage 7 (SAAL Hanabi port)
                                              │
                                              ▼
                                          Stage 8 (INC Hanabi discrim + cross-play)
                                              │
                                              ▼
                                          Stage 9 (ablations)
                                              │
                                              ▼
                                          Stage 10 (figures + writing)
                                              │
                                              ▼
                                          Stage 11 (robustness)

Stage 3 (Hanabi env wiring) sits outside this line — done in parallel during
Stages 1-2, completed, and is a prerequisite for Stages 7 and 8.
```

**Bailout points:**
- End of Stage 4: if baseline `pair_cos_ff` is already saturated, reformulate SAAL before Stage 5.
- End of Stage 5: if SAAL doesn't beat baseline on SMAX 3m after hyperparameter retune, return to direction doc.
- End of Stage 6: if SAAL doesn't generalise beyond 3m, decide whether to port to Hanabi or re-plan.
- End of Stage 7: if SAAL doesn't work on Hanabi, the Hanabi section becomes negative-only (Stage 8 only).

---

## Config keys (quick reference)

| Key | Type | Default | Added in | Meaning |
|---|---|---|---|---|
| `INC_ENABLED` | bool | `False` | Stage 2 | Master switch for INC pooling |
| `INC_POOLING` | str | `"mean"` | Stage 2 | `mean` / `attention` / `gated` |
| `INC_NUM_AGENTS` | int | auto | Stage 2 | Set from env at `make_train` |
| `INC_CONSENSUS_DROPOUT` | float | `0.0` | Stage 2 | Dropout on pooled consensus |
| `CTM_ITER_DROPOUT` | float | `0.0` | Stage 2.1 | Dropout on activated trace between iters |
| `INC_FORCE_ZERO_CONSENSUS` | bool | `False` | Stage 2.1 | Zero pooled vector (cell-E control) |
| `ALIGN_ENABLED` | bool | `False` | Stage 5 | Master switch for SAAL |
| `ALIGN_ALPHA` | float | `0.0` | Stage 5 | SAAL pull weight on coord events |
| `ALIGN_BETA` | float | `0.0` | Stage 5 | SAAL push weight on non-coord steps |
| `NUM_MOVEMENT_ACTIONS` | int | auto | Stage 4 | Captured from env for focus-fire mask |
| `NUM_ENEMIES` | int | auto | Stage 4 | Captured from env for focus-fire mask |

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Baseline `pair_cos` is already saturated | Stage 4 measures this before any loss is added; reformulate SAAL as subspace alignment if needed |
| SAAL drives sync vectors to agent-invariant collapse under parameter sharing | β push-term on non-coord steps; Stage 5 gradient-flow test; entropy monitoring during sanity run |
| Focus-fire is too sparse a signal | Stage 5 bailout option 2 (widen event set) or option 3 (advantage-gated) |
| SAAL helps SMAX but not Hanabi | Stage 7 has a Hanabi-specific event detector; bailout is Hanabi section becomes negative-only (INC cheating case only) |
| INC cheating test is ambiguous (H-C ≫ A but H-E ≈ C) | Cross-play is the decisive second line of evidence; if both tests disagree, report the ambiguity in the paper rather than picking a side |
| Hanabi budget explodes (full matrix > available compute) | Stage 7 runs 1 seed per cell first ("signal exists"), 3 seeds only if warranted; same staging rule as Stage 8 |
| Cross-play plumbing has a bug that underreports scores | Unit test: self-pair (same seed on both agents) must match standard eval score within noise |
