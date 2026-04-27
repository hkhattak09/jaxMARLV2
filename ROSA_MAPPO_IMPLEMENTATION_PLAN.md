# ROSA-MAPPO Implementation Plan

This plan is for implementing **ROSA-MAPPO: Role-Ordered Sequential Adapter MAPPO** as a separate experiment path.

Do **not** modify `smax_ctm/train_mappo_gru.py`. Treat it as the untouched baseline. Create a new file:

```text
smax_ctm/train_rosa_mappo.py
```

Start by copying the structure of `smax_ctm/train_mappo_gru.py`, then make all ROSA changes only in the new file.

The most important rule: every stage must produce information that can invalidate or redirect the idea. Do not build all stages first and only then run experiments.

---

## Execution Environment Assumptions

The implementation agent should assume that training will **not** be executed locally and the agent will **not** run Colab itself. Local work is limited to code editing and, at most, syntax/import checks.

The real training runs will be launched from a notebook/Colab-style environment with commands like:

```python
!python /content/jaxMARLV2/smax_ctm/train_mappo_gru.py
```

Use full paths in the stage run commands. If the repo is mounted at a different path, replace `/content/jaxMARLV2` with the actual repo root.

Expected Colab repo root used in this plan:

```text
/content/jaxMARLV2
```

Expected local repo root while editing:

```text
/Users/hassan/repos/new_marl_llm_implementation
```

Because training is not run locally, every stage should include handoff instructions for the user:

```text
1. the code changes required
2. a local syntax-check command
3. the exact config values or code switches the user should set
4. the full notebook/Colab command the user should run
5. the logs/metrics the user should send back after the run
```

The agent should not say "run the stage" vaguely. It should give the user a concrete command such as:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py
```

If the stage requires config changes, the agent should list the exact keys and values to set in the `config` dictionary before giving the command.

---

## Error Handling Philosophy: Fail Loud, Never Fake

Prefer a visible failure over a silent fallback.

- Never silently swallow errors to keep things "working."
- Surface the error. Do not substitute placeholder data.
- Fallbacks are acceptable only when disclosed. Print a warning, log a warning, or annotate the output.
- Design for debuggability, not cosmetic stability.

Priority order:

```text
1. Works correctly with real data
2. Falls back visibly and clearly signals degraded mode
3. Fails with a clear error message
4. Silently degrades to look fine: never do this
```

Examples:

```text
If role IDs cannot be extracted, raise ValueError with obs shape and expected unit-type slice.
If a role has zero samples, log that it was skipped.
If correction factors are clipped, log correction min/mean/max before and after clipping.
If an optional metric cannot be computed inside jit, explicitly mark it unavailable rather than returning zeros.
```

Do not fake metrics with zeros unless zero is mathematically correct. If a metric is unavailable, name it as unavailable.

---

## Per-Stage Handoff Template

At the end of each implementation stage, the agent should give the user a short handoff in this shape:

```text
Stage N handoff

Files changed:
- ...

Local syntax check:
python -m py_compile /Users/hassan/repos/new_marl_llm_implementation/smax_ctm/train_rosa_mappo.py

Config to set:
- KEY = VALUE
- KEY = VALUE

Notebook command to run:
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py

Please send back:
- the last 30-50 log lines
- any role-wise diagnostic table printed
- whether the run NaN'd or crashed
```

For Stage 0, the command uses the untouched baseline:

```python
!python /content/jaxMARLV2/smax_ctm/train_mappo_gru.py
```

For all ROSA stages, the command uses:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py
```

If the script later supports CLI overrides, prefer explicit CLI commands. Until then, list the exact config dictionary edits required before the user runs the command.

---

## Research Hypothesis

Standard shared recurrent MAPPO is strong on fixed SMAX maps, but it is a poor fit for heterogeneous or randomized-role settings because it uses:

```text
one shared policy
one simultaneous PPO actor update
one global policy drift signal
```

ROSA-MAPPO tests whether we can recover some of HAPPO's sequential-update stability inside a mostly shared MAPPO policy by adding lightweight **role-specific LoRA adapters** and updating those adapters **sequentially by role**.

The core hypothesis has three parts:

1. Role-conditioned adapters can learn useful specialization without abandoning the shared GRU backbone.
2. Role-wise policy metrics reveal instability that global MAPPO metrics hide.
3. Sequential role-adapter updates improve stability or sample efficiency over normal simultaneous adapter training.

If any of these fail, the implementation should tell us early.

---

## Main Files

Create:

```text
smax_ctm/train_rosa_mappo.py
```

Optional later files:

```text
smax_ctm/run_rosa_sweep.py
smax_ctm/analyze_rosa_metrics.py
```

Recommended logs:

```text
logs/rosa_stage0_baseline_<map>_<seed>.txt
logs/rosa_stage1_role_plumbing_<map>_<seed>.txt
logs/rosa_stage2_zero_lora_<map>_<seed>.txt
logs/rosa_stage3_role_lora_<map>_<seed>.txt
logs/rosa_stage5_rosa_<map>_<seed>.txt
```

---

## Stage 0: Baseline Reproducibility And Harness

### Goal

Confirm the untouched GRU MAPPO baseline behavior and establish the exact comparison setup.

### Implementation

Do not change `smax_ctm/train_mappo_gru.py`.

Run baseline on:

```text
3m
2s3z
smacv2_5_units
smacv2_10_units
```

Use short runs first for smoke tests, then longer runs for real comparison.

Recommended initial smoke config:

```text
TOTAL_TIMESTEPS = 3e5 to 1e6
NUM_ENVS = 64 or 128
NUM_STEPS = 128
SEEDS = [0]
```

Recommended real comparison:

```text
TOTAL_TIMESTEPS = 3e6+
SEEDS = [0, 1, 2] if time allows
```

### Required Information

Record:

```text
win_rate curve
return curve
entropy
actor_grad_norm
critic_grad_norm
approx_kl
wall-clock time
```

### User Run Command

The user should run the untouched baseline from the notebook/Colab environment:

```python
!python /content/jaxMARLV2/smax_ctm/train_mappo_gru.py
```

For each map, set in the baseline script's `config` before running:

```python
"MAP_NAME": "3m"              # then repeat for "2s3z", "smacv2_5_units", "smacv2_10_units"
"TOTAL_TIMESTEPS": int(3e5)   # smoke; use int(3e6) for real comparison
"SEED": 0
```

Ask the user to send back:

```text
last 30-50 log lines
final win rate
whether training completed or crashed
rough wall-clock time
```

### Success Criteria

Continue if:

```text
baseline trains successfully
logs are reproducible enough to compare against
smacv2 maps show a meaningful weakness or slower learning
```

### Stop Or Pivot Criteria

Stop and debug before ROSA if:

```text
baseline does not train
metrics are missing or unusable
baseline already solves smacv2_5_units and smacv2_10_units quickly and reliably
```

If baseline solves the randomized maps, the ROSA hypothesis needs a different testbed or a harder generalization split.

---

## Stage 1: Create ROSA File With Role Plumbing Only

### Goal

Create `train_rosa_mappo.py` and add role-id tracking without changing behavior.

### Implementation

Copy `smax_ctm/train_mappo_gru.py` to:

```text
smax_ctm/train_rosa_mappo.py
```

Add config keys:

```python
"NUM_UNIT_TYPES": 6,
"ROLE_ID_SOURCE": "own_obs_unit_type",
"USE_ROLE_LORA": False,
"USE_SEQUENTIAL_ROLE_UPDATES": False,
```

Add `role_id` to `Transition`:

```python
class Transition(NamedTuple):
    ...
    role_id: jnp.ndarray
```

In SMAX local observations, own unit type is at the end of the observation because `get_obs_unit_list` concatenates other unit features followed by own features. Own features end with unit-type bits. Extract role id as:

```python
own_type_bits = obs_batch[:, -config["NUM_UNIT_TYPES"]:]
role_id = jnp.argmax(own_type_bits, axis=-1).astype(jnp.int32)
```

Store `role_id` in each transition.

Do not use `role_id` in the actor loss yet.

### Required Information

Log per update:

```text
role_count_0 ... role_count_5
role_present_mask
```

For fixed maps, role counts should match the scenario:

```text
3m: only marine role
2s3z: stalker and zealot roles
smacv2_*: multiple randomized roles
```

### Success Criteria

Continue if:

```text
train_rosa_mappo.py matches baseline performance when all ROSA options are off
role counts match expected unit types for fixed maps
role counts vary on smacv2 maps
```

### Stop Or Pivot Criteria

Stop if:

```text
role IDs are wrong
role counts do not match known maps
baseline-equivalent mode changes learning behavior
```

This stage must pass before any adapter work. Bad role IDs make every later result meaningless.

### Local Syntax Check

After creating `train_rosa_mappo.py`, the implementation agent can check syntax locally:

```bash
python -m py_compile /Users/hassan/repos/new_marl_llm_implementation/smax_ctm/train_rosa_mappo.py
```

### User Run Command

The user should run:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py
```

Set:

```python
"MAP_NAME": "2s3z"               # also test "3m" and "smacv2_5_units"
"TOTAL_TIMESTEPS": int(3e5)
"USE_ROLE_LORA": False
"USE_SEQUENTIAL_ROLE_UPDATES": False
"SEED": 0
```

Ask the user to send back:

```text
role_count logs
last 30-50 training log lines
any role extraction error
whether baseline-equivalent mode appears to train normally
```

---

## Stage 2: Add Zero-Initialized Role LoRA, Verify Equivalence

### Goal

Add role-specific LoRA adapters to the actor while preserving exact or near-exact baseline behavior at initialization.

### Recommended First Placement

Start with LoRA on the final action-logit layer only. This is the safest initial implementation.

Current actor head:

```text
GRU output -> Dense hidden -> ReLU -> Dense action logits
```

Change final logits to:

```text
base_logits = W_shared h
delta_logits = B_role A_role h
logits = base_logits + lora_scale * delta_logits
```

Use:

```text
rank = 4 or 8
lora_scale = 1.0 / rank or config value
```

Parameter shapes:

```text
lora_A: (NUM_UNIT_TYPES, rank, hidden_dim)
lora_B: (NUM_UNIT_TYPES, action_dim, rank)
```

Initialize:

```text
lora_A: normal or orthogonal small init
lora_B: zeros
```

Zero-initialized `lora_B` ensures the adapter initially contributes zero.

The actor input must now include `role_id`:

```text
obs, dones, avail_actions, role_id
```

When `USE_ROLE_LORA = False`, skip the adapter path.

When `USE_ROLE_LORA = True`, include it.

### Required Information

Before training or at first update, compare:

```text
baseline logits
zero-LoRA logits
max_abs_logit_diff
mean_abs_logit_diff
```

During training, log:

```text
lora_delta_norm_by_role
lora_param_norm_by_role
lora_grad_norm_by_role
role_count_by_role
```

### Local Syntax Check

```bash
python -m py_compile /Users/hassan/repos/new_marl_llm_implementation/smax_ctm/train_rosa_mappo.py
```

### User Run Command

The user should run:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py
```

Set:

```python
"MAP_NAME": "2s3z"                # also test "3m" after this
"TOTAL_TIMESTEPS": int(3e5)
"USE_ROLE_LORA": True
"USE_SEQUENTIAL_ROLE_UPDATES": False
"ROLE_LORA_RANK": 4
"ROLE_LORA_SCALE": 1.0
"LOG_ZERO_LORA_EQUIV": True
"SEED": 0
```

Ask the user to send back:

```text
zero-LoRA max_abs_logit_diff and mean_abs_logit_diff
role_count logs
lora_delta_norm_by_role
lora_grad_norm_by_role
last 30-50 training log lines
```

### Success Criteria

Continue if:

```text
zero-LoRA logits match baseline up to numerical noise
training with USE_ROLE_LORA=True and zero init does not immediately destabilize
adapter gradients are nonzero for roles that appear
```

### Stop Or Pivot Criteria

Stop and debug if:

```text
zero-LoRA changes initial policy materially
adapter gradients are always zero
unavailable-action masking breaks
single-role maps such as 3m behave unexpectedly
```

If gradients are too small, move LoRA one layer earlier, for example onto the hidden actor Dense layer instead of only the final logits.

---

## Stage 3: Normal Role-LoRA MAPPO Baseline

### Goal

Test whether role-specific adapters alone are useful before adding sequential updates.

This stage answers:

```text
Is role specialization itself useful?
```

### Implementation

Train with:

```python
"USE_ROLE_LORA": True,
"USE_SEQUENTIAL_ROLE_UPDATES": False,
```

Use normal simultaneous MAPPO actor update. The actor params include both shared parameters and role-LoRA parameters.

Compare against:

```text
GRU MAPPO baseline
GRU MAPPO + role LoRA
```

Optional but useful:

```text
GRU MAPPO + agent-index LoRA
```

Agent-index LoRA is a strong diagnostic. If agent-index LoRA beats role-LoRA on randomized maps, our role-specialization claim is weaker than expected.

### Required Information

Log aggregate metrics:

```text
win_rate
return
entropy
actor_loss
value_loss
actor_grad_norm
critic_grad_norm
```

Log role metrics:

```text
role_count
role_adv_mean
role_adv_std
role_entropy
role_approx_kl
role_clip_frac
role_lora_delta_norm
role_lora_grad_norm
```

### Local Syntax Check

```bash
python -m py_compile /Users/hassan/repos/new_marl_llm_implementation/smax_ctm/train_rosa_mappo.py
```

### User Run Command

The user should run:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py
```

Set:

```python
"MAP_NAME": "smacv2_5_units"      # then "smacv2_10_units" if smoke passes
"TOTAL_TIMESTEPS": int(1e6)       # smoke; use int(3e6) for real comparison
"USE_ROLE_LORA": True
"USE_SEQUENTIAL_ROLE_UPDATES": False
"ROLE_LORA_RANK": 4
"ROLE_LORA_SCALE": 1.0
"LOG_ROLE_DIAGNOSTICS": True
"LOG_ZERO_LORA_EQUIV": False
"SEED": 0
```

Ask the user to send back:

```text
final win rate and return
last 50 log lines
role_count
role_approx_kl
role_clip_frac
role_lora_grad_norm
whether any role has zero or tiny gradients despite appearing
```

### Success Criteria

Continue if at least one is true:

```text
role LoRA improves sample efficiency on smacv2 maps
role LoRA improves final win rate on smacv2 maps
role LoRA reveals meaningful per-role differences in KL, advantage scale, entropy, or gradient norms
```

### Stop Or Pivot Criteria

If role LoRA does nothing:

```text
check whether adapter gradients are too small
increase rank from 4 to 8
move LoRA to actor hidden layer
try warmup before enabling adapters
```

If role LoRA hurts:

```text
reduce lora learning rate
reduce lora_scale
increase entropy coefficient slightly
turn on adapters only after shared-policy warmup
```

If role LoRA helps strongly and role-wise metrics are stable:

```text
sequential updates may be less necessary
the project may become role-conditioned LoRA MAPPO unless Stage 4 shows hidden instability
```

Do not proceed blindly to ROSA if this stage shows no role-specific learning signal.

---

## Stage 4: Role-Wise Instability Diagnostics

### Goal

Before implementing sequential role updates, verify that there is actually a role-wise optimization problem for ROSA to solve.

This stage answers:

```text
Does global MAPPO hide role-specific policy drift or role imbalance?
```

### Implementation

Add diagnostic computations inside the actor loss path.

For each role:

```text
count_r
adv_mean_r
adv_std_r
entropy_r
approx_kl_r
clip_frac_r
mean_ratio_r
max_ratio_r
ppo_loss_r
```

Also compute global-vs-role discrepancy:

```text
max_role_kl - global_kl
max_role_clip_frac - global_clip_frac
max_role_adv_std / global_adv_std
min_role_count / max_role_count
```

### Required Information

Produce a short diagnostic table per run:

```text
role | count | adv_mean | adv_std | entropy | approx_kl | clip_frac | lora_grad_norm
```

For smacv2 maps, inspect whether:

```text
some roles have much higher KL
some roles have much lower entropy
some roles receive much weaker gradients
global approx_kl looks safe while max_role_kl is high
```

### Local Syntax Check

```bash
python -m py_compile /Users/hassan/repos/new_marl_llm_implementation/smax_ctm/train_rosa_mappo.py
```

### User Run Command

The user should run:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py
```

Set:

```python
"MAP_NAME": "smacv2_5_units"
"TOTAL_TIMESTEPS": int(1e6)
"USE_ROLE_LORA": True
"USE_SEQUENTIAL_ROLE_UPDATES": False
"ROLE_LORA_RANK": 4
"ROLE_LORA_SCALE": 1.0
"LOG_ROLE_DIAGNOSTICS": True
"LOG_ROLE_DIAGNOSTIC_TABLE": True
"SEED": 0
```

Ask the user to send back:

```text
the printed role diagnostic table
global approx_kl and max role approx_kl
global clip_frac and max role clip_frac
role advantage means/stds
role gradient norms
last 50 log lines
```

### Success Criteria

Proceed to Stage 5 if:

```text
role-wise metrics differ meaningfully
or role LoRA helps but shows unstable role-specific KL/clip_frac
or randomized maps show role imbalance that global metrics obscure
```

### Stop Or Pivot Criteria

If all role metrics are nearly identical:

```text
ROSA's sequential role-update justification is weak
consider pivoting to role-conditioned LoRA only, entity critic, or a harder environment
```

If instability exists only in the critic/value loss:

```text
consider role-aware critic or role-wise advantage normalization before sequential adapter updates
```

This stage is the main early-warning system. Do not skip it.

---

## Stage 5: ROSA Sequential Role-Adapter Updates

### Goal

Implement the core ROSA algorithm:

```text
shared MAPPO backbone update
then sequential role-specific adapter updates
```

This stage answers:

```text
Does HAPPO-style sequential conditioning improve role-adapter learning?
```

### Implementation Strategy

Keep implementation conservative.

Use two actor training phases:

```text
Phase A: standard actor update
Phase B: sequential role-adapter update
```

For the first working version, consider these options:

```text
Option 1:
Phase A updates shared actor params and LoRA params with normal MAPPO.
Phase B performs extra sequential LoRA-only updates.

Option 2, cleaner:
Phase A updates shared actor params with LoRA gradients masked/frozen.
Phase B updates only LoRA params sequentially.
```

Option 2 is conceptually cleaner and preferred if param masking is manageable.

### Parameter Freezing

During sequential role updates:

```text
freeze shared actor backbone
freeze critic
update only current role's LoRA parameters
```

This is necessary. If the shared backbone changes during one role update, all other role policies change too, and the sequential interpretation becomes muddy.

### Random Role Order

For version 1:

```text
sample random order over roles present in the batch
```

Use random ordering because it is simple and consistent with HAPPO-like random sequential updates.

### Conditioned Advantage

Start with a conservative correction.

For current role `r`, compute:

```text
A_conditioned = correction_previous_roles * A
```

where `correction_previous_roles` is based on previous role policy ratios.

To avoid instability:

```text
correction_previous_roles = stop_gradient(correction_previous_roles)
correction_previous_roles = clip(correction_previous_roles, 1 - CORRECTION_CLIP, 1 + CORRECTION_CLIP)
```

Recommended initial config:

```python
"CORRECTION_CLIP": 0.2,
"ROLE_SEQ_UPDATE_EPOCHS": 1,
"ROLE_SEQ_LR_MULT": 0.5,
"ROLE_ORDERING": "random",
```

Important practical note:

The exact HAPPO product over previous agents can be high-variance. For a stable first implementation, use one of these approximations:

```text
sample-wise correction:
for each sample, use ratio if its role was previously updated, else 1

role-mean correction:
multiply by mean previous-role ratio over the minibatch

clipped log-ratio correction:
sum clipped previous-role log ratios, then exponentiate
```

Start with clipped log-ratio correction or role-mean correction. Log all correction statistics.

### Required Information

Log:

```text
role_order
role_seq_loss_by_role
role_seq_kl_by_role
role_seq_clip_frac_by_role
role_seq_ratio_mean_by_role
role_seq_ratio_max_by_role
correction_mean
correction_std
correction_min
correction_max
adapter_update_norm_by_role
```

Also log whether each role was skipped due to zero samples.

### Local Syntax Check

```bash
python -m py_compile /Users/hassan/repos/new_marl_llm_implementation/smax_ctm/train_rosa_mappo.py
```

### User Run Command

The user should run:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py
```

Set:

```python
"MAP_NAME": "smacv2_5_units"
"TOTAL_TIMESTEPS": int(5e5)       # first ROSA smoke; increase only after stable
"USE_ROLE_LORA": True
"USE_SEQUENTIAL_ROLE_UPDATES": True
"ROLE_ORDERING": "random"
"ROLE_LORA_RANK": 4
"ROLE_LORA_SCALE": 1.0
"CORRECTION_CLIP": 0.2
"ROLE_SEQ_UPDATE_EPOCHS": 1
"ROLE_SEQ_LR_MULT": 0.5
"LOG_ROLE_DIAGNOSTICS": True
"LOG_ROSA_CORRECTIONS": True
"SEED": 0
```

Ask the user to send back:

```text
whether the run completed or NaN'd
role_order samples from logs
correction_mean/std/min/max
role_seq_kl_by_role
role_seq_clip_frac_by_role
adapter_update_norm_by_role
last 50 log lines
```

### Success Criteria

Continue if:

```text
ROSA trains without NaNs
correction factors stay bounded
role-wise KL is more controlled than normal role-LoRA MAPPO
ROSA matches or beats normal role-LoRA on smacv2 maps in sample efficiency or final win rate
```

### Stop Or Pivot Criteria

If correction explodes:

```text
lower CORRECTION_CLIP
use role-mean correction instead of sample-wise product
reduce ROLE_SEQ_LR_MULT
use only one sequential role epoch
```

If ROSA is stable but no better than role-LoRA:

```text
the role adapter is doing most of the work
sequential conditioning may not be necessary
try Stage 6 ordering diagnostics before abandoning
```

If ROSA hurts performance:

```text
disable conditioned advantage but keep sequential role updates
compare "sequential no-correction" vs "sequential correction"
if no-correction works better, the HAPPO correction approximation is wrong for this setting
```

This stage should produce an honest answer about whether ROSA's algorithmic part helps.

---

## Stage 6: Ordering Strategy Ablations

### Goal

Test whether role update order matters. Keep this after random-order ROSA is working.

### First Ordering

Use:

```text
random
```

This is the default and should be the first complete version.

### Footnote Strategies To Try Later

Implement these only after Stage 5 works:

```text
rare-role-first:
roles with fewer samples update earlier

KL-risk:
roles with high previous approx_kl or clip_frac update earlier

influence-first:
roles that appear most interaction-heavy update earlier
cheap proxies: visible unit count, proximity to enemies, damage taken/dealt proxy,
or mean absolute advantage

fixed tactical order:
frontline/melee roles first, ranged roles second, support roles last
```

### Required Information

For each ordering:

```text
same seed
same map
same config
win_rate curve
role-wise KL curve
correction stats
adapter update norms
```

### Local Syntax Check

```bash
python -m py_compile /Users/hassan/repos/new_marl_llm_implementation/smax_ctm/train_rosa_mappo.py
```

### User Run Commands

Run one ordering at a time. For the default random order:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py
```

Set:

```python
"MAP_NAME": "smacv2_5_units"
"TOTAL_TIMESTEPS": int(1e6)
"USE_ROLE_LORA": True
"USE_SEQUENTIAL_ROLE_UPDATES": True
"ROLE_ORDERING": "random"
"CORRECTION_CLIP": 0.2
"ROLE_SEQ_UPDATE_EPOCHS": 1
"ROLE_SEQ_LR_MULT": 0.5
"SEED": 0
```

Then repeat with one changed key:

```python
"ROLE_ORDERING": "rare_first"
```

Later candidates:

```python
"ROLE_ORDERING": "kl_risk_first"
"ROLE_ORDERING": "influence_first"
```

Ask the user to send back for each ordering:

```text
final win rate
area under win-rate curve if available
role-wise KL summary
correction stats
last 50 log lines
```

### Success Criteria

Ordering is a useful research axis if:

```text
random works but another simple order is consistently better
or different maps prefer different orderings
or KL-risk/influence order reduces role-specific policy drift
```

### Stop Or Pivot Criteria

If all orders perform the same:

```text
ordering is a footnote, not a contribution
focus the final story on role adapters and sequential updates
```

---

## Stage 7: Evaluation Matrix

### Goal

Build the final evidence package.

### Algorithms

At minimum:

```text
1. GRU MAPPO baseline
2. Role-LoRA MAPPO, simultaneous update
3. ROSA-MAPPO, random role order
```

If time:

```text
4. Agent-index LoRA MAPPO
5. ROSA no-correction
6. ROSA with alternate orderings
```

### Maps

Primary:

```text
smacv2_5_units
smacv2_10_units
```

Secondary:

```text
2s3z
3s5z
3m as sanity/anomaly check
```

### Metrics

Report:

```text
final win rate
area under win-rate curve
steps to 50% win rate
steps to 70% win rate
variance across seeds
wall-clock time
parameter count
```

ROSA-specific:

```text
per-role KL
per-role clip fraction
per-role entropy
per-role adapter norm
correction statistics
ordering statistics
```

### User Run Commands

The agent should give the user one command/config block per algorithm.

Baseline:

```python
!python /content/jaxMARLV2/smax_ctm/train_mappo_gru.py
```

Set in `train_mappo_gru.py`:

```python
"MAP_NAME": "smacv2_5_units"      # repeat for "smacv2_10_units"
"TOTAL_TIMESTEPS": int(3e6)
"SEED": 0                         # repeat for 1 and 2 if time allows
```

Role-LoRA MAPPO:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py
```

Set:

```python
"MAP_NAME": "smacv2_5_units"
"TOTAL_TIMESTEPS": int(3e6)
"USE_ROLE_LORA": True
"USE_SEQUENTIAL_ROLE_UPDATES": False
"ROLE_LORA_RANK": 4
"ROLE_LORA_SCALE": 1.0
"LOG_ROLE_DIAGNOSTICS": True
"SEED": 0
```

ROSA-MAPPO:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py
```

Set:

```python
"MAP_NAME": "smacv2_5_units"
"TOTAL_TIMESTEPS": int(3e6)
"USE_ROLE_LORA": True
"USE_SEQUENTIAL_ROLE_UPDATES": True
"ROLE_ORDERING": "random"
"ROLE_LORA_RANK": 4
"ROLE_LORA_SCALE": 1.0
"CORRECTION_CLIP": 0.2
"ROLE_SEQ_UPDATE_EPOCHS": 1
"ROLE_SEQ_LR_MULT": 0.5
"LOG_ROLE_DIAGNOSTICS": True
"LOG_ROSA_CORRECTIONS": True
"SEED": 0
```

Repeat each command for:

```text
smacv2_5_units
smacv2_10_units
```

and seeds:

```text
0, 1, 2 if time allows
```

Ask the user to send back:

```text
full log files or final 100 lines per run
final win rate
best win rate
time-to-threshold if visible
role-wise diagnostic summaries
any crash/NaN reports
```

### Success Criteria For The Idea

Strong positive result:

```text
ROSA > Role-LoRA > GRU MAPPO on randomized maps
ROSA controls role-wise KL better than Role-LoRA
fixed maps are not harmed
```

Moderate positive result:

```text
Role-LoRA > GRU MAPPO
ROSA matches Role-LoRA but improves stability metrics
```

Weak result:

```text
Role-LoRA helps, ROSA does not
```

In this case, pivot the story to role-conditioned specialization and keep ROSA as an attempted sequential extension.

Negative result:

```text
Role-LoRA does not help
ROSA does not help
role-wise metrics show no hidden instability
```

In this case, the hypothesis is likely wrong for this testbed. Do not keep stacking complexity.

---

## Implementation Notes For The Agent

### Preserve Baseline

Never modify:

```text
smax_ctm/train_mappo_gru.py
```

All changes go into:

```text
smax_ctm/train_rosa_mappo.py
```

### Keep Config Switches

Every feature should be switchable:

```python
"USE_ROLE_LORA": False,
"USE_SEQUENTIAL_ROLE_UPDATES": False,
"ROLE_ORDERING": "random",
"ROLE_LORA_RANK": 4,
"ROLE_LORA_SCALE": 1.0,
"LOG_ZERO_LORA_EQUIV": False,
"LOG_ROLE_DIAGNOSTICS": False,
"LOG_ROLE_DIAGNOSTIC_TABLE": False,
"LOG_ROSA_CORRECTIONS": False,
"CORRECTION_CLIP": 0.2,
"ROLE_SEQ_LR_MULT": 0.5,
"ROLE_SEQ_UPDATE_EPOCHS": 1,
```

This allows ablations without rewriting code.

### Avoid A Huge First Rewrite

Do not start by adding entity encoders, transformer memory, or critic redesign.

First implement:

```text
role_id tracking
action-head LoRA
role-wise diagnostics
sequential adapter update
```

### NaN Guards

Add checks or logs for:

```text
ratio max
correction max
loss finite
grad norm finite
adapter norm finite
```

If NaNs appear, first reduce:

```text
ROLE_SEQ_LR_MULT
CORRECTION_CLIP
ROLE_SEQ_UPDATE_EPOCHS
ROLE_LORA_SCALE
```

### Expected Failure Modes

1. **Role IDs are wrong**

Everything after Stage 1 becomes invalid.

2. **LoRA path changes initial policy**

Zero-init is wrong or masking is wrong.

3. **LoRA gradients are tiny**

Move LoRA earlier or increase rank.

4. **Sequential correction is too noisy**

Use role-mean correction or no-correction ablation.

5. **ROSA helps fixed maps but not randomized maps**

Likely learning capacity, not role generalization.

6. **Role-LoRA helps but ROSA does not**

Keep role specialization as the useful result; do not oversell sequential updates.

---

## Recommended Stage Order

```text
Stage 0: baseline harness
Stage 1: role-id plumbing, no behavior change
Stage 2: zero-init role LoRA, equivalence test
Stage 3: normal role-LoRA MAPPO
Stage 4: role-wise instability diagnostics
Stage 5: ROSA sequential role-adapter updates
Stage 6: ordering ablations
Stage 7: final evaluation matrix
```

Do not skip stages. Each stage is designed to answer whether the next stage is worth implementing.
