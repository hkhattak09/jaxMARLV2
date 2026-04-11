# SAAL Stage 4 Baseline Pair-Cos Note

## Scope
Stage 4 is logging-only on vanilla CTM-MAPPO (SMAX 3m):
- INC disabled
- CTM iterations fixed to 1
- no alignment term added to actor loss

Goal: estimate baseline cross-agent synchronisation cosine statistics before choosing SAAL loss weights.

## Run Setup
- Map: 3m
- Seeds: 1
- Budget: same as Stage 2 Cell A
- Required config:
  - INC_ENABLED=False
  - CTM_ITERATIONS=1

## Logged Metrics
From training stdout and metric dict:
- pair_cos_all
- pair_cos_ff
- pair_cos_nff
- ff_frac

Definitions:
- pair_cos_all: mean pairwise cosine over all timesteps and envs.
- pair_cos_ff: mean pairwise cosine restricted to focus-fire steps.
- pair_cos_nff: mean pairwise cosine restricted to non-focus-fire steps.
- ff_frac: fraction of focus-fire steps in sampled rollout.

## Colab Command
Run from repository root:

```bash
python -m smax_ctm.train_mappo_ctm
```

Override config in file or launch variant consistent with Stage 2 Cell A.

Expected logging line includes:

```text
pair_cos(all/ff/nff): <float>/<float>/<float> | ff_frac: <float>
```

## Results
Fill after run:
- Final pair_cos_all:
- Final pair_cos_ff:
- Final pair_cos_nff:
- Final ff_frac:
- Any NaN encountered: yes/no
- Observed trend notes:

## Interpretation
Use this decision rule:
1. If pair_cos_ff >= 0.90 at convergence, SAAL has little scalar-cos headroom. Prefer revisiting objective (for example target subspace alignment instead of raw cosine push).
2. If pair_cos_ff approximately equals pair_cos_nff, spontaneous separation is weak and SAAL should have a clear job.
3. If pair_cos_ff is meaningfully above pair_cos_nff, baseline already carries coordination signal and SAAL should be applied conservatively.

## Stage 5 Initialization Recommendation
Choose a conservative starting point from observed headroom:
- High baseline similarity (pair_cos_ff >= 0.90): ALIGN_ALPHA in [0.0, 0.01], ALIGN_BETA in [0.0, 0.05]
- Medium baseline similarity (0.70 to 0.90): ALIGN_ALPHA in [0.01, 0.05], ALIGN_BETA in [0.05, 0.20]
- Low baseline similarity (< 0.70): ALIGN_ALPHA in [0.05, 0.20], ALIGN_BETA in [0.10, 0.30]

Selected starting values for Stage 5:
- ALIGN_ALPHA:
- ALIGN_BETA:
- Rationale:
