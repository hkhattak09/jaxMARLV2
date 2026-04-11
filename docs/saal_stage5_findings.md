# SAAL Stage 5 Findings

## Setup

- Environment: SMAX 3m
- Method: CTM-MAPPO + SAAL
- Config: `INC_ENABLED=False`, `CTM_ITERATIONS=1`, `ALIGN_ENABLED=True`, `ALIGN_ALPHA=0.05`, `ALIGN_BETA=0.025`
- Sanity seed: 200
- Main seeds: 201, 202, 203
- Output directory: `analysis_results_inc/stage5/`

## Pre-Registered Bars

- Clear win: final WR >= 0.822 and frac(WR >= 0.8) > 20%
- Minimum viable: final WR >= 0.81 and converged pair_cos_ff >= 0.70
- Null: final WR within +/-0.01 of 0.793 and pair_cos_ff lift < 0.02
- Regression: final WR < 0.783 or NaN/entropy collapse

## Sanity Run (Seed 200)

- Status: pending
- NaN/entropy collapse: pending
- final WR (last 20 updates): pending
- pair_cos_ff final20: pending
- pair_cos_nff final20: pending
- pair_cos gap final20: pending
- L_align final20: pending

## Main 3-Seed Results (201/202/203)

| Seed | final WR | frac WR >= 0.8 | pair_cos_ff final20 | pair_cos_nff final20 | gap | L_align final20 |
|---|---:|---:|---:|---:|---:|---:|
| 201 | pending | pending | pending | pending | pending | pending |
| 202 | pending | pending | pending | pending | pending | pending |
| 203 | pending | pending | pending | pending | pending | pending |
| mean +/- std | pending | pending | pending | pending | pending | pending |

## Go/No-Go

- Stage 5 verdict: pending
- Decision for Stage 6: pending

## Notes

- Populate this document from `analysis_results_inc/stage5/stage5_runs.json` plus per-run metric pickles.
- Keep Stage 4 baseline values side-by-side for direct comparison.
