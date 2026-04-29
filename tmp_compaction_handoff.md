# Temporary Compaction Handoff

This file is a restart note for the next chat. It can be deleted after the next
assistant has read it.

## User Preference

- Do not run local runtime tests. The user runs Colab commands and pastes
  output back.
- It is okay to inspect files, run static checks, and read diffs locally.
- Do not include `XLA_PYTHON_CLIENT_PREALLOCATE=false` in run commands unless
  the user explicitly switches to memory-debug mode.

## Current Direction

The sequential trainer has completed its job as a correctness oracle:

```text
smax_ctm/train_lorasa_eggroll.py
```

It proved geometry/checkpoint plumbing but is not the real optimizer. It should
not be used in the training hot path.

The active next path is a single-GPU population-axis trainer:

```text
smax_ctm/train_lorasa_eggroll_pop.py
```

No `shard_map`, no `process_allgather`, no distributed assumptions. Use
HyperscaleES ideas only where appropriate:

```text
thread_id // 2 -> direction id
thread_id % 2 -> antithetic sign
vmap over population chunks
deterministic regenerated perturbations
```

Do not copy HyperscaleES geometry: our method is fixed-rank LoRASA manifold ES
with tangent projection + SVD retraction.

## Implemented Since Last Compact

1. Added `smax_ctm/train_lorasa_eggroll_pop.py`
   - User-facing rollout scale:
     `episodes_per_candidate`, not `num_loops`.
   - One epoch means:
     population eval -> antithetic weights -> one Riemannian adapter update.
   - Population axis lives only on selected LoRA leaves:
     selected `lora_a` / `lora_b` leaves have `in_axes=0`; frozen leaves are
     broadcast with `in_axes=None`.
   - Candidate LoRA factors are built per population chunk and discarded after
     evaluation.
   - Defaults:
     `num_directions=64`, `population_batch_size=8`,
     `num_envs_per_candidate=16`, `episodes_per_candidate=64`,
     `sigma=0.05`, `eta=0.0015`,
     `fitness_mode=win_rate_return_tiebreak`.
   - Evaluator returns wins, episode returns, episode lengths, and recorded
     fractions.
   - Supports `--print_candidates`.

2. Updated `smax_ctm/lorasa_eggroll.py`
   - `apply_weighted_tangent_update(...)` now accepts optional
     `direction_normalizer`.
   - Population trainer passes `direction_normalizer=args.num_directions` so
     update scale is stable even when many direction weights are zero/tied.

3. Updated docs:
   - `riemannian_lorasa_eggroll.md`
   - `riemannian_lorasa_eggroll_todo.md`

## Colab Result Right Before Compact

User ran the no-op population smoke:

```bash
python smax_ctm/train_lorasa_eggroll_pop.py \
  --checkpoint /path/to/schedule_A/checkpoint_final_compressed_A.pkl \
  --num_epochs 1 \
  --num_directions 2 \
  --population_batch_size 4 \
  --num_envs_per_candidate 4 \
  --episodes_per_candidate 4 \
  --heldout_num_envs 4 \
  --heldout_episodes 4 \
  --sigma 0.0 \
  --eta 0.0 \
  --fitness_mode win_rate \
  --print_candidates
```

It failed at first with:

```text
TypeError: vmap in_axes must be an int, None, or a tuple ...
```

Cause:

```python
in_axes=population_in_axes
```

was passed directly for a one-positional-argument vmapped function.

Patch applied:

```python
in_axes=(population_in_axes,)
```

File:

```text
smax_ctm/train_lorasa_eggroll_pop.py
```

Static `git diff --check` passed after the patch.

## Immediate Next Command

Ask the user to rerun the no-op population smoke without any XLA env prefix:

```bash
python smax_ctm/train_lorasa_eggroll_pop.py \
  --checkpoint /path/to/schedule_A/checkpoint_final_compressed_A.pkl \
  --num_epochs 1 \
  --num_directions 2 \
  --population_batch_size 4 \
  --num_envs_per_candidate 4 \
  --episodes_per_candidate 4 \
  --heldout_num_envs 4 \
  --heldout_episodes 4 \
  --sigma 0.0 \
  --eta 0.0 \
  --fitness_mode win_rate \
  --print_candidates
```

Expected signal:

```text
one chunk with 4 candidates
candidate rows print
direction weights can be zero or tie-derived
update summary has zero singular shift because eta=0
heldout line prints
checkpoint saves
no crash
```

If that passes, next command is the small nonzero population smoke:

```bash
python smax_ctm/train_lorasa_eggroll_pop.py \
  --checkpoint /path/to/schedule_A/checkpoint_final_compressed_A.pkl \
  --num_epochs 1 \
  --num_directions 4 \
  --population_batch_size 4 \
  --num_envs_per_candidate 8 \
  --episodes_per_candidate 16 \
  --heldout_num_envs 8 \
  --heldout_episodes 16 \
  --sigma 0.05 \
  --eta 0.0015 \
  --print_candidates
```

If nonzero direction weights are reported, validate:

```bash
python smax_ctm/lorasa_eggroll.py \
  --reference_checkpoint /path/to/schedule_A/checkpoint_final_compressed_A.pkl \
  --checkpoint lorasa_eggroll_pop_runs/<run_id>/checkpoint_final.pkl \
  --require_active_change \
  --validation_json diagnostics/lorasa_eggroll_pop_update_validation.json
```

## Important Workspace Notes

`git status --short --untracked-files=all` before this handoff showed:

```text
 M riemannian_lorasa_eggroll_todo.md
 M smax_ctm/train_lorasa_eggroll_pop.py
```

Before that, there had also been unrelated user/workspace changes:

```text
 D no_recurrent_lora_final.pkl
?? r4_no_recurrent.pkl
?? r8_no_recurrent.pkl
```

Do not revert or touch these unrelated checkpoint files unless the user asks.

## Research State

Schedule A compressed no-recurrent LoRASA remains the source checkpoint:

```text
map: protoss_10_vs_10
active adapter slots: 2, 3, 6
target active rank: 4
blocks:
  params/action_out
  params/base_0
  params/base_1
  params/base_2
  params/rnn/gru_cell/input_candidate
  params/rnn/gru_cell/input_reset
  params/rnn/gru_cell/input_update
```

Previous structural validations passed:

```text
num_selected_blocks=7
active_slot_pairs_changed=21 for nonzero update
changed_non_active_leaves=0
active_rank_violations=0
```

Sequential small-pop pilot did not improve Schedule A and is now considered a
correctness reference only, not the optimizer.
