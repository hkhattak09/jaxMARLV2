# Temporary Compaction Handoff

Current repo: `/Users/hassan/repos/new_marl_llm_implementation`.
User runs runtime tests in Colab; do not run local runtime tests. Static checks are okay.

## Current Branch / GitHub State

- `main` was pushed to GitHub after reverting early-stop eval:
  - `f206e31 Revert early-stop population eval`
  - Keeps `225ee1b chunking on gpu` device candidate builder.
  - Removes `80de204 eval stops early when episodes end` behavior from trainer.
- After that push, a new local uncommitted edit was made:
  - `smax_ctm/train_lorasa_eggroll_pop.py` now sets
    `os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")`
    before importing JAX.
  - `riemannian_lorasa_eggroll_todo.md` notes this.
- Need to run static checks, commit, and push this memory-fraction default unless user changes their mind.

## Important User Preferences

- Colab commands must include repo-prefixed script paths, e.g.
  `python jaxMARLV2/smax_ctm/train_lorasa_eggroll_pop.py ...`
- Checkpoint path in Colab is currently:
  `/content/jaxMARLV2/r4_no_recurrent.pkl`
- Diagnostics paths should be prefixed with `jaxMARLV2/diagnostics/...`.
- User wants to maximize A100 utilization; willing to probe capacity.
- User does not want vague small pilots now; focus on scale and throughput.

## Project State

- Task: Riemannian LoRASA-EGGROLL population-axis ES for SMAX `protoss_10_vs_10`.
- Active LoRASA slots: `2,3,6`.
- Active blocks:
  - `params/action_out`
  - `params/base_0`
  - `params/base_1`
  - `params/base_2`
  - `params/rnn/gru_cell/input_candidate`
  - `params/rnn/gru_cell/input_reset`
  - `params/rnn/gru_cell/input_update`
- Target active rank: `4`.
- Sequential trainer `smax_ctm/train_lorasa_eggroll.py` remains an off-hot-path correctness oracle only.
- Population trainer: `smax_ctm/train_lorasa_eggroll_pop.py`.

## Validated Trainer Features

1. CPU sequential correctness path passed earlier.
2. Population-axis trainer works on one GPU.
3. Device/JAX candidate builder added and validated:
   - `--candidate_build device` default.
   - Builds per-chunk candidate LoRA factors with JAX batched SVD/retraction.
   - Uses same JAX noise backend for update, keeping candidate scores and update directions consistent.
   - CPU fallback remains: `--candidate_build cpu`.
4. Device builder smoke:
   - Run dir `lorasa_eggroll_pop_runs/lorasa_eggroll_pop_20260430_103646`
   - Chunk 1 compile: `build=25.74s eval=27.42s`
   - Chunk 2 steady: `build=0.14s eval=0.21s`
5. Device builder structural validation passed:
   - `passed=true`, `num_violations=0`
   - `active_slot_pairs_changed=21`
   - `changed_non_active_leaves=0`
   - `active_rank_violations=0`

## Early-Stop Eval Experiment

- Early-stop eval was tried and pushed as `80de204`, then reverted by `f206e31`.
- Reason: on A100 scale, early stop became straggler/while-loop limited and did not improve eval time.
- Do not use early-stop flags; they no longer exist in the restored trainer.

## A100 Capacity Observations

Old CPU candidate build big run:
```text
num_directions=2048, candidates=4096, population_batch_size=256,
num_envs_per_candidate=128, episodes_per_candidate=256
train episodes/epoch=1,048,576
CPU build steady ~=15-18s/chunk
eval steady ~=10.7s/chunk
epoch ~=455s
JAX allocated ~30.1GB of 40GB due to default ~75% memory fraction
```

Device builder tiny smoke showed build bottleneck is gone.
Need large A100 device-builder probe next.

## Immediate Next Static Work

Run:
```bash
python -m py_compile smax_ctm/train_lorasa_eggroll_pop.py
git diff --check
git status --short --branch --untracked-files=all
```
Then commit and push the memory-fraction default:
```bash
git add smax_ctm/train_lorasa_eggroll_pop.py riemannian_lorasa_eggroll_todo.md
git commit -m "Raise JAX memory fraction for population trainer"
git push origin main
```
Need escalation for git add/commit/push if sandbox blocks index/network.

## Next Colab Capacity Probes

First confirm device builder with fixed horizon and 0.95 memory fraction default:
```bash
python jaxMARLV2/smax_ctm/train_lorasa_eggroll_pop.py \
  --checkpoint /content/jaxMARLV2/r4_no_recurrent.pkl \
  --num_epochs 1 \
  --num_directions 2048 \
  --population_batch_size 256 \
  --num_envs_per_candidate 128 \
  --episodes_per_candidate 256 \
  --heldout_num_envs 256 \
  --heldout_episodes 2048 \
  --sigma 0.05 \
  --eta 0.0005 \
  --eval_every 1 \
  --save_every 1 \
  --candidate_build device
```

Then push concurrency:
```bash
python jaxMARLV2/smax_ctm/train_lorasa_eggroll_pop.py \
  --checkpoint /content/jaxMARLV2/r4_no_recurrent.pkl \
  --num_epochs 1 \
  --num_directions 2048 \
  --population_batch_size 512 \
  --num_envs_per_candidate 256 \
  --episodes_per_candidate 256 \
  --heldout_num_envs 512 \
  --heldout_episodes 2048 \
  --sigma 0.05 \
  --eta 0.0005 \
  --eval_every 1 \
  --save_every 1 \
  --candidate_build device
```

If it fits and improves throughput, try:
```bash
python jaxMARLV2/smax_ctm/train_lorasa_eggroll_pop.py \
  --checkpoint /content/jaxMARLV2/r4_no_recurrent.pkl \
  --num_epochs 1 \
  --num_directions 2048 \
  --population_batch_size 1024 \
  --num_envs_per_candidate 256 \
  --episodes_per_candidate 256 \
  --heldout_num_envs 512 \
  --heldout_episodes 2048 \
  --sigma 0.05 \
  --eta 0.0005 \
  --eval_every 1 \
  --save_every 1 \
  --candidate_build device
```

Capacity logic:
- Peak concurrent lanes ~= `population_batch_size * num_envs_per_candidate`.
- Current big baseline: `256*128=32,768` lanes.
- Probe 1: `512*256=131,072` lanes, 8 chunks, 1 rollout batch.
- Probe 2: `1024*256=262,144` lanes, 4 chunks, 1 rollout batch.

Main run will be chosen after capacity probe, likely using best chunk shape with larger `num_directions` (possibly 4096) and 4-6 epochs.
