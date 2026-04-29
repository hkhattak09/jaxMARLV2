# Riemannian LoRASA-EGGROLL TODO

Living checklist for the LoRASA phase-3 work. Keep this file current: mark
completed tasks with `[x]`, add short notes under tasks when results arrive,
and remove or rewrite tasks that become obsolete.

## Current Decision State

```text
Main phase-2 candidate: no-recurrent LoRASA
Active adapter slots on protoss_10_vs_10: 2, 3, 6
Main phase-3 direction: Riemannian LoRASA-EGGROLL
Current ES adapter target: Schedule A compressed no-recurrent LoRASA, rank 4 everywhere
Immediate next experiment: prototype adapter-only Riemannian ES on protoss_10_vs_10
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
- [x] Implement post-hoc LoRASA adapter compression tool.
- [x] Evaluate compressed checkpoint variants A/B/C.
- [x] Choose Schedule A as the working compressed adapter state.
- [x] Inspect local official EGGROLL implementation in `HyperscaleES/`.
  - Useful patterns: Noiser abstraction, low-rank matmul injection, antithetic
    `thread_id // 2` direction keys, vmap/shard_map population axis,
    shape-bucketed update compilation, process_allgather fitness collection,
    timing/logging/checkpoint structure.
  - Do not copy the official unconstrained ambient update rule as the main
    LoRASA method.

## Immediate Next Tasks

- [x] Write a concrete implementation plan for the Riemannian LoRASA-EGGROLL prototype.
  - Start from Schedule A compressed no-recurrent LoRASA.
  - Active slots only: `2, 3, 6`.
  - Rank 4 for all active blocks.
  - Tangent projection.
  - SVD retraction.
  - Norm-scaled sigma.

- [ ] Implement Riemannian LoRASA-EGGROLL prototype.
  - [x] Add adapter-tree helpers for discovering active `lora_a` / `lora_b` leaves.
    - Added `smax_ctm/lorasa_eggroll.py`.
  - [x] Add Riemannian fixed-rank helpers: balanced factorization, tangent
    projection, truncated-SVD retraction, perturbation regeneration.
  - [ ] Run Colab smoke test for helper module.
  - [ ] Add an ES evaluator/trainer entry point based on `smax_ctm/eval_smax.py`.
  - [ ] Keep backbone and critic frozen; optimize actor adapters only.
  - [ ] Use deterministic actions for ES evaluation.

- [ ] Run a smoke test on a tiny population and a small number of SMAX envs.
  - Validate checkpoint load/save.
  - Validate active slots `2, 3, 6` change and unused slots remain unchanged.
  - Validate antithetic candidates use identical perturbation directions with
    opposite signs.
  - Validate rank-4 adapters remain rank 4 after retraction.

- [ ] Run first protoss_10_vs_10 ES experiment.
  - Report train-bundle fitness, held-out deterministic win rate, update norms,
    sigma/eta, active slot norms, and singular values.

## Rank Compression Schedules

- [x] Schedule A: all active blocks rank 4.
  - Result: mean win rate 0.9440, std 0.2300, sem 0.0073, episodes 1000.
  - Decision: chosen as working ES target.

- [x] Schedule B: base layers rank 4, GRU input gates rank 2, action_out rank 4.
  - Result: mean win rate 0.9320, std 0.2519, sem 0.0080, episodes 1000.

- [x] Schedule C: base layers rank 4, input_candidate rank 4, input_reset/input_update rank 2, action_out rank 4.
  - Result: mean win rate 0.9390, std 0.2395, sem 0.0076, episodes 1000.

## Phase-3 ES Design Tasks

- [x] Decide final active-rank schedule from compression results.
  - Use Schedule A: all active no-recurrent LoRA blocks rank 4.

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

- [ ] Implement Riemannian LoRASA-EGGROLL prototype.
  - Start with Schedule A compressed no-recurrent LoRASA.
  - Active slots only.
  - Rank 4 all active blocks.
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
Schedule A: 0.9440 mean win rate over 1000 episodes.
Schedule B: 0.9320 mean win rate over 1000 episodes.
Schedule C: 0.9390 mean win rate over 1000 episodes.

Schedule A is selected because it is simplest and performed best. Mixed rank-2
GRU input schedules are not worth the added complexity right now.
```

Final chosen ES rank schedule:

```text
No-recurrent LoRASA, active slots 2/3/6, rank 4 for every active LoRA block.
```

Known risks:

```text
SMAX/SMACv2 objective is stochastic.
Pure win rate can be too binary for early ES ranking.
GRU input gates have higher condition numbers than MLP/action adapters.
Unused adapter slots must be masked out.
```
