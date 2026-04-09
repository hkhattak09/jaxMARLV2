# CTM Synchronisation Analysis Infrastructure

This package implements Stage 1-3 analysis infrastructure from `docs/analysis_plan.md` using a split, modular layout.

## Directory Structure

- `smax_ctm/analysis/checkpoint.py`: checkpoint loading and strict param-tree extraction
- `smax_ctm/analysis/collector.py`: manual CTM unroll and per-step diagnostics collection
- `smax_ctm/analysis/policy_head.py`: actor head forward pass from `synch` to policy
- `smax_ctm/analysis/metrics.py`: pairwise sync/observation correlation metrics
- `smax_ctm/analysis/plotting.py`: quick-look timeseries and heatmap figures
- `smax_ctm/analysis/io_utils.py`: output directory and pickle IO helpers
- `smax_ctm/analyse_sync.py`: thin CLI entrypoint

## Execution Protocol

- Local machine: code generation and static checks only.
- Colab: run scripts that need JAX runtime and checkpoints.
- No virtual environment setup is required by this package.

## Colab Run Command

Run from repository root:

```bash
python smax_ctm/analyse_sync.py \
  --checkpoint model/smax_mappo_ctm_actor.pkl \
  --output-dir analysis_results \
  --num-episodes 20 \
  --seed 42
```

## Outputs

- `analysis_results/episode_traces.pkl`: raw per-step diagnostics and event flags
- `analysis_results/sync_metrics.pkl`: computed pairwise metrics and summaries
- `analysis_results/figures/*.png`: quick-look correlation plots

## Fail-Loud Behavior

The implementation raises clear errors for missing keys, incompatible shapes, or malformed checkpoints. Undefined correlations from constant vectors are not hidden; they are stored as `NaN` and reported as warnings.
