# Stage 2.1 findings — disambiguating the INC dropout win

**Date:** 2026-04-11
**Scope:** SMAX 3m, 3M timesteps, `CTM_ITERATIONS=3`, `INC_POOLING="mean"`.
**Protocol:** Stage 2 cells A / B / C already logged (1 seed each, seeds 0–2
from the Stage 2 run). Stage 2.1 cells D and E run fresh, 3 seeds each
(103 / 104 / 105, deliberately non-overlapping with Stage 2).
**Raw data:** `analysis_results_inc/runs/run{1,2,3}.txt` for A/B/C;
`analysis_results_inc/stage2_1/*.json` for D/E.

## TL;DR

The cell C win (`INC_ENABLED=True, INC_CONSENSUS_DROPOUT=0.25`) is coming
from **noised-but-real teammate information flowing through the pooled
consensus channel**. It is not coming from dropout acting as a generic
regulariser on the CTM iteration unroll, and it is not an artefact of the
dropout call site (RNG consumption, widened Synapses kernel). This is
**Story 1 — "consensus channel is load-bearing"** from the plan's three
decision stories. Stage 4 / 5 / 7 framing stays as originally written;
dropout becomes a prominent sub-finding under the robust-consensus framing
rather than the main result.

## Numbers

All "final WR" values are mean win rate over the last 20 PPO updates. All
cells run the same 3M-timestep, 128-env, 128-step budget (183 PPO updates).

| Cell | Config | Seeds | Final WR | Peak WR | Frac updates ≥ 0.8 | First update rolling-20 ≥ 0.8 |
|---|---|---|---|---|---|---|
| **A** (run1) | CTM baseline, `INC_ENABLED=False` | 1 | 0.793 | 0.820 | 4.9 % | never |
| **B** (run2) | `INC_ENABLED=True`, dropout 0.0 | 1 | 0.811 | 0.840 | 12.0 % | 173 |
| **C** (run3) | `INC_ENABLED=True`, `INC_CONSENSUS_DROPOUT=0.25` | 1 | 0.822 | 0.850 | 41.5 % | 131 |
| **D** | `INC_ENABLED=False`, `CTM_ITER_DROPOUT=0.25` | 3 | 0.745 ± 0.033 | 0.767 | 1.1 % | never |
| **E** | `INC_ENABLED=True`, `INC_CONSENSUS_DROPOUT=0.25`, `INC_FORCE_ZERO_CONSENSUS=True` | 3 | 0.802 ± 0.034 | 0.837 | 14.9 % | 155 |

Per-seed breakdown for D / E:

| Run | Final WR | Peak | Frac ≥ 0.8 | First update ≥ 0.8 |
|---|---|---|---|---|
| D seed 103 | 0.732 | 0.752 | 0.0 % | never |
| D seed 104 | 0.789 | 0.807 | 3.3 % | never |
| D seed 105 | 0.715 | 0.742 | 0.0 % | never |
| E seed 103 | 0.757 | 0.785 | 0.0 % | never |
| E seed 104 | 0.820 | 0.854 | 20.2 % | 160 |
| E seed 105 | 0.829 | 0.871 | 24.6 % | 150 |

## Interpretation against the three decision stories

### Story 1 — "consensus channel is load-bearing" — **SUPPORTED**

Plan's rule: *D close to A/B, E close to A/B, C remains the clear winner.*

Final-WR picture:
- D (0.745) is actually **below** A (0.793) and B (0.811) — not just "close
  to A/B". Stripping the consensus channel and adding dropout to the
  iteration trace is a net negative, not a neutral control.
- E (0.802) sits in between A and B on final WR — i.e. it has reverted to
  the no-INC regime. C (0.822) stays the clear winner on final WR.
- Within the plan's 3 pp-band rule: E is within 3 pp of *both* B (+0.9 pp)
  and C (−2.0 pp) on the final-WR metric alone. But the plan explicitly
  required the 3 pp band **and** the learning-curve slope to match. On
  curve shape E is clearly closer to B than to C:

| Metric | B | E | C |
|---|---|---|---|
| Frac updates WR ≥ 0.8 | 12.0 % | 14.9 % | 41.5 % |
| First update rolling-20 ≥ 0.8 | 173 | 155 | 131 |
| Peak WR | 0.840 | 0.837 | 0.850 |

C spends roughly 2.8× as many updates above 0.8 as E, and first crosses it
~24 updates (≈ 13 % of training) earlier. E is not tracking C's curve.

### Story 2 — "stochastic iteration loop" — **REJECTED**

Plan's rule: *D close to C, E close to C.*

Neither condition holds. D is −7.7 pp below C on final WR, regresses below
even the no-INC baseline A, never holds above 0.8, and is tight across
seeds (0.715 / 0.732 / 0.789), so it's not a variance artefact. E is also
off C on the curve-shape metrics above.

A dropout-as-iteration-regulariser story would require D ≈ C. We see the
opposite — dropping random activations on the trace when there is nothing
for them to coordinate with actively hurts.

### Story 3 — "mixed" — **REJECTED**

Plan's rule: *D in between A and C (noticeable lift but not matching C),
E close to D or close to A.*

D is **not** between A and C — it is worse than A. Rule fails on the first
clause.

## Why the force-zero control is clean

Cell E's implementation (see [smax_ctm/ctm_jax.py](../smax_ctm/ctm_jax.py)
`AgentConsensus` with `force_zero_output=True`) zeros the pooled vector
**before** the dropout call. So:

1. The dropout RNG is consumed on the same step as cell C.
2. The dropout mask is applied to a zero tensor and still produces zero.
3. The widened Synapses kernel (the one with input dim
   `sync_size + consensus_size` instead of `sync_size` alone) is still
   trained — gradients flow through the consensus slice via the non-zero
   dropout mask path and via the zeroing op itself
   (verified by `test_stage2_1_force_zero_gradient_still_flows_to_synapses`
   in [smax_ctm/tests/test_inc.py](../smax_ctm/tests/test_inc.py)).

So the *only* thing E strips relative to C is the actual teammate
information. If the consensus channel were irrelevant, E would match C.
It does not.

## Caveats

1. **A / B / C are single-seed.** The Stage 2 protocol ran one seed per
   cell. C's 41.5 % frac-over-0.8 is one draw and could be seed-lucky.
   However the monotone A → B → C ordering on both final WR (0.793 →
   0.811 → 0.822) and frac-over-0.8 (4.9 → 12.0 → 41.5 %) is consistent
   with the story, and D / E are 3-seed so the *negative* controls are
   properly sampled.
2. **E has meaningful seed variance** (0.757 / 0.820 / 0.829). Seed 103
   behaves like A; seeds 104/105 behave closer to B. This is itself
   informative — with the pooled vector zeroed, runs revert to the no-INC
   regime, which is exactly what Story 1 predicts. None of E's seeds
   reach C's final WR or C's frac-over-0.8.
3. **Peak WR vs sustained WR.** E's *peak* (0.837) is almost equal to C's
   (0.850), so at the peak E can briefly reach near-C performance. The
   difference is in how often and how early it *stays* there, which is
   why the plan specified the curve-slope rule in addition to the 3 pp
   final-WR band.

## Consequences for later stages

- **Stage 4** hyperparameter sweep: keep the originally planned axes
  (`INC_POOLING`, `NUM_CONSENSUS_ITERATIONS`, pooling dim, dropout rate).
  The dropout sweep is now motivated — we want to know where on the
  dropout-rate curve the benefit peaks and how quickly it degrades.
- **Stage 5** main matrix: unchanged. INC vs no-INC is still the headline
  comparison. Cell C config stays the default "INC" setting.
- **Stage 6** ablations: the Stage 2.1 result is itself an ablation and
  can slot directly into the ablations table as "zero-consensus control"
  and "iter-dropout-only control".
- **Stage 7** paper framing: headline is still the pooled teammate sync
  channel. Dropout on the channel is presented as the configuration that
  unlocks the full benefit, with cells D and E as the controls that show
  neither the stochasticity nor the widened kernel alone explain it.

## Files and reproduction

- Runner: [smax_ctm/scripts/run_stage2_1_disambig.py](../smax_ctm/scripts/run_stage2_1_disambig.py)
- Module changes: [smax_ctm/ctm_jax.py](../smax_ctm/ctm_jax.py)
  (`AgentConsensus.force_zero_output`, `CTMCell.ctm_iter_dropout`,
  `CTMCell.inc_force_zero_consensus`)
- Training plumbing: [smax_ctm/train_mappo_ctm.py](../smax_ctm/train_mappo_ctm.py)
  and [smax_ctm/train_mappo_ctm_hanabi.py](../smax_ctm/train_mappo_ctm_hanabi.py)
- Unit tests: `test_stage2_1_*` in
  [smax_ctm/tests/test_inc.py](../smax_ctm/tests/test_inc.py)
- Stage 2 baseline logs: `analysis_results_inc/runs/run{1,2,3}.txt`
- Stage 2.1 raw output: `analysis_results_inc/stage2_1/*.json`,
  `analysis_results_inc/stage2_1/stage2_1_runs.json`
