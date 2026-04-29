# Riemannian LoRASA-EGGROLL TODO

Living checklist for the LoRASA phase-3 work. Keep this file current: mark
completed tasks with `[x]`, add short notes under tasks when results arrive,
and remove or rewrite tasks that become obsolete.

## Current Decision State

```text
Main phase-2 candidate: no-recurrent LoRASA
Active adapter slots on protoss_10_vs_10: 2, 3, 6
Main phase-3 direction: Riemannian LoRASA-EGGROLL
Immediate next experiment: post-hoc rank compression of no-recurrent LoRASA
```

## Completed

- [x] Read EGGROLL paper and identify relevance to RL/MARL and hyperscale ES.
- [x] Run deep research on fixed-rank ES for LoRA.
- [x] Choose tangent-projection + retraction as the theoretically grounded update.
- [x] Create dependency-light LoRASA adapter spectral diagnostic.
- [x] Run diagnostics on full LoRASA checkpoint.
- [x] Run no-recurrent-GRU-LoRA ablation.
- [x] Run diagnostics on no-recurrent LoRASA checkpoint.
- [x] Promote no-recurrent LoRASA to the current main phase-2 candidate.

## Immediate Next Tasks

- [x] Implement post-hoc LoRASA adapter compression tool.
  - Input: no-recurrent LoRASA checkpoint.
  - Output: compressed checkpoint variants for rank schedules A-C.
  - Must preserve checkpoint structure and metadata.
  - Must not retrain.
  - Verified with `py_compile`, `--self_test`, and a fake-checkpoint dry run.

- [ ] Evaluate compressed checkpoint variants.
  - Compare against original no-recurrent LoRASA checkpoint.
  - Use the same evaluation protocol and seeds where possible.
  - Record win rate, return, episode length, and any timeout/death metrics available.

- [ ] Update `riemannian_lorasa_eggroll.md` with compression results.
  - Which schedule preserves performance?
  - Which rank schedule should phase-3 ES use?
  - Did compression reveal hidden rank dependence?

## Rank Compression Schedules

- [ ] Schedule A: all active blocks rank 4.
- [ ] Schedule B: base layers rank 4, GRU input gates rank 2, action_out rank 4.
- [ ] Schedule C: base layers rank 4, input_candidate rank 4, input_reset/input_update rank 2, action_out rank 4.

## Phase-3 ES Design Tasks

- [ ] Decide final active-rank schedule from compression results.
- [ ] Decide active-slot mask source.
  - Hard-code slots `2, 3, 6` only for first experiment if needed.
  - Later: derive active slots from rollout/eval metadata.

- [ ] Decide ES fitness formula.
  - Start with win rate plus shaped-return tie-breaker.
  - Consider timeout/death penalties only if logging is reliable.

- [ ] Decide variance-reduction defaults.
  - Antithetic pairs.
  - Common random numbers.
  - Centered ranks.
  - Held-out evaluation seeds.
  - Optional per-seed baseline.

- [ ] Inspect official EGGROLL implementation for systems patterns.
  - Population-axis layout.
  - Low-rank perturbation generation.
  - Antithetic pairing.
  - Batched forward-pass mechanics.
  - RNG discipline.
  - Aggregation and logging.
  - Do not copy the ambient unconstrained update rule as the main method.

- [ ] Implement Riemannian LoRASA-EGGROLL prototype.
  - Start with no-recurrent LoRASA.
  - Active slots only.
  - Tangent projection.
  - SVD retraction.
  - Norm-scaled sigma.

## Ablations To Keep

- [ ] No-recurrent LoRASA continuation vs Riemannian LoRASA-EGGROLL.
- [ ] Full LoRASA vs no-recurrent LoRASA on additional seeds/maps if time allows.
- [ ] Riemannian update vs ambient EGGROLL + SVD compression.
- [ ] Fixed-rank manifold ES vs rank-expanded residual ES.
- [ ] Independent role/layer perturbations vs partially shared perturbations.
- [ ] Factor-space ES baseline if needed for reviewer/scientific comparison.

## Notes To Fill In

Compression result notes:

```text
TBD
```

Final chosen ES rank schedule:

```text
TBD
```

Known risks:

```text
SMAX/SMACv2 objective is stochastic.
Pure win rate can be too binary for early ES ranking.
GRU input gates have higher condition numbers than MLP/action adapters.
Unused adapter slots must be masked out.
```
