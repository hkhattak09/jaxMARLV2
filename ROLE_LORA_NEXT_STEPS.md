# Role-LoRA MAPPO Project Plan

This is the single canonical planning file for the Role-LoRA / ROSA / residual MAPPO project.

It replaces:

```text
PROJECT_MEMORY_ROLE_CONTEXT_RESIDUAL.md
ROSA_MAPPO_IMPLEMENTATION_PLAN.md
```

Do not maintain separate project-memory or implementation-plan copies. If the project changes, update this file.

---

## Current State

Repository:

```text
/Users/hassan/repos/new_marl_llm_implementation
```

Untouched baseline:

```text
smax_ctm/train_mappo_gru.py
```

Experimental file:

```text
smax_ctm/train_rosa_mappo.py
```

Primary benchmark:

```text
smacv2_10_units with heuristic enemies
```

Training setup:

```text
Training runs happen in Colab/notebook, not locally.
Local work should be limited to code edits and syntax/import checks.
Smoke tests use total_timesteps=300000.
Full comparison runs use total_timesteps=3000000.
```

Local syntax check:

```bash
python -m py_compile /Users/hassan/repos/new_marl_llm_implementation/smax_ctm/train_rosa_mappo.py
```

Baseline command:

```python
!python /content/jaxMARLV2/smax_ctm/train_mappo_gru.py
```

Experiment command pattern:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode role_lora --total_timesteps 3000000 --run_name role_lora_smacv2_10_seed42
```

Error-handling policy:

```text
Fail loud, never fake.
Prefer a visible failure over a silent fallback.
Do not substitute placeholder data that looks valid.
If a metric is unavailable, say unavailable rather than returning fake zeros.
Do not modify smax_ctm/train_mappo_gru.py.
```

---

## Validated Result

The strongest validated result is:

```text
Role-LoRA MAPPO
```

Actor:

```text
local observation -> Dense/ReLU -> GRU -> Dense/ReLU -> base action logits
role id -> select role-specific LoRA adapter
final logits = base logits + role LoRA residual logits
```

Formula:

```text
h_i,t      = GRU_theta(o_i,t, h_i,t-1)
z_base_i,t = f_theta(h_i,t)
delta_i,t  = B_role_i A_role_i h_i,t
z_i,t      = z_base_i,t + alpha * delta_i,t
pi_i,t     = softmax(mask(z_i,t))
```

Main result on `smacv2_10_units`, five seeds:

| mode | seeds | final WR mean | final WR std | best WR mean | final return mean | AUC WR mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAPPO baseline | 5 | 0.4620 | 0.0259 | 0.4940 | 1.3100 | 0.2386 |
| Role-LoRA | 5 | 0.5060 | 0.0152 | 0.5300 | 1.3640 | 0.2596 |
| sequential polish | 5 | 0.5120 | 0.0164 | 0.5300 | 1.3760 | 0.2665 |

Interpretation:

```text
Role-LoRA beats baseline on final win rate in all five seeds.
Role-LoRA improves AUC and lowers final-win-rate variance.
Sequential polish gives a small average gain, but the clean effect is Role-LoRA itself.
```

Recommended wording:

```text
Across five seeds on randomized smacv2_10_units, role-conditioned LoRA improves final win rate from 46.2% to 50.6% and AUC from 0.2386 to 0.2596. A lightweight role-local polishing step gives a small additional gain to 51.2%, but the larger and cleaner effect is the role-conditioned adapter parameterization.
```

Earlier seed-42 ablation:

| mode | final WR | best WR | AUC WR |
| --- | ---: | ---: | ---: |
| none | 0.4900 | 0.5500 | 0.2956 |
| role_lora | 0.5400 | 0.5500 | 0.3101 |
| global_lora | 0.4900 | 0.5400 | 0.2840 |
| agent_lora | 0.5200 | 0.5500 | 0.2992 |
| sequential_polish | 0.5500 | 0.5700 | 0.3185 |

Interpretation:

```text
Role-LoRA beats global LoRA and agent-ID LoRA in the seed-42 ablation.
This helps answer the "you just added parameters" criticism.
Agent-ID LoRA is weaker than semantic role conditioning on randomized teams.
```

---

## Explored Negative Evidence

Do not headline these as the main contribution.

### Sequential ROSA

Original idea:

```text
Role-Ordered Sequential Adapter MAPPO
```

Mechanism:

```text
shared recurrent MAPPO backbone
role-specific LoRA adapters
HAPPO-style sequential adapter updates by role
```

Result:

```text
Mechanically valid, but did not clearly beat joint Role-LoRA.
Sequential polish is useful as a small practical refinement, not as a HAPPO-level contribution.
```

Interpretation:

```text
HAPPO-style sequential correction does not transfer cleanly when applied only as an adapter-local post-step after a shared MAPPO update.
```

### Role-Residual

Mechanism:

```text
L_total = L_MAPPO + lambda_residual * L_residual + beta_kl * KL(pi_base || pi_full)
```

Clean version:

```text
LoRA frozen out of normal MAPPO gradients.
LoRA trained only through residual role-normalized PPO loss.
Backbone stopped/frozen for residual loss.
Hinge KL target around 0.02.
```

Important implementation fix:

```text
Actor optimizer must call apply_gradients exactly once per update.
A second Adam step can move parameters even when residual gradients are zero because momentum state is active.
Residual gradients must be combined with actor gradients, then applied in one optimizer step.
```

Empirical status:

```text
Mechanically clean.
Residual LoRA gradients > 0.
Residual backbone gradients = 0.
Hinge KL mostly inactive.
Did not clearly beat Role-LoRA or sequential polish.
```

Seed-1 comparison:

| mode | seed-1 final WR | seed-1 best WR |
| --- | ---: | ---: |
| none | 0.49 | 0.51 |
| role_lora | 0.53 | 0.54 |
| sequential_polish | 0.54 | 0.54 |
| role_residual | 0.50 | 0.53 |

### Role-Context Residual

Mechanism:

```text
final logits = base logits + context_gate * role_lora_delta
context = team role histogram / own-role count
```

Empirical status:

```text
Mechanically healthy.
Gate gradients were nonzero.
Backbone residual gradients stayed zero.
Full seed-42 run underperformed Role-Residual.
Gate saturated high, roughly learning "turn residual on" rather than meaningful context modulation.
```

Decision:

```text
Drop context gating from the core method.
Do not run shuffled_context_residual or retained-context variants unless the branch is explicitly reopened later.
```

---

## Research Synthesis

The deep research review and the MACA paper support this pivot:

```text
The missing ingredient is probably not more residual capacity.
The missing ingredient is a stronger advantage / credit-assignment signal.
```

The literature pattern:

| Theme | Representative sources | Useful lesson |
| --- | --- | --- |
| Strong PPO/MAPPO baseline | [MAPPO](https://openreview.net/forum?id=YVXaxB6L2Pl) | PPO/MAPPO can be very strong when details are right; do not assume architecture novelty is needed. |
| Sequential trust-region MARL | [HARL/HAPPO/HATRPO](https://jmlr.org/papers/v25/23-0488.html), [A2PO](https://openreview.net/forum?id=Q-neeWNVv1) | Simultaneous updates can hide agent/role instability; per-agent or per-role update budgets are defensible. |
| Credit assignment | [PRD-MAPPO](https://arxiv.org/abs/2408.04295), [Dr.REINFORCE](https://www.microsoft.com/en-us/research/publication/dr-reinforce/), [Shapley Counterfactual Credits](https://arxiv.org/abs/2106.00285) | Team reward is often too blunt; better credit routing can matter more than extra capacity. |
| Multi-level advantage credit | [MACA](https://arxiv.org/abs/2508.06836) | MAPPO's joint advantage and COMA's individual advantage are both incomplete; strong tasks need credit at individual, team, and correlated-subgroup levels. |
| Role specialization | [ROMA](https://proceedings.mlr.press/v119/wang20f.html), [RODE](https://openreview.net/forum?id=TTUVg6vkNjK), [ACORM](https://arxiv.org/abs/2312.04819) | Roles need identifiable responsibilities or behaviorally meaningful separation, not just labels. |
| Partial sharing/adapters | [LoRASA](https://arxiv.org/abs/2502.05573), [Kaleidoscope](https://arxiv.org/abs/2410.08540) | Shared backbones plus small specialist modules are reasonable, but specialization needs a signal. |
| Conditional capacity | Switch/ST-MoE/Soft MoE style results | Extra branches or experts help only when routing and regularization prevent collapse or wasted capacity. |

Diagnosis of current failure modes:

```text
Role-LoRA works because semantic unit type gives the actor a useful specialization key.

Role-Residual did not clearly improve because role-normalized team advantage changes scale but does not create new role credit.

Freezing LoRA out of normal MAPPO made the residual mechanism clean, but it also removed the strongest PPO signal from the adapter.

The residual KL was either weak or mostly inactive, so it did not create a strong trust-region story.

The context gate saturated because the static team-composition context was too weak to tell the model when a residual should matter.
```

MACA changes the mental map:

```text
Role-Trust fixes actor-side update weighting, but it still uses the same underlying team advantage.

MACA shows that the larger improvement likely comes from changing the advantage itself:
    A_i = Q(s, a) - counterfactual baseline_i

The baseline should reason at multiple levels:
    joint/team action
    individual agent action
    correlated subgroup action
```

For this project, the correlated subgroup should initially be role-based, not learned with a transformer:

```text
CorrSet C_i = agents in the same semantic unit role as agent i, including i.
```

This gives a role-aware MACA-lite path that uses our existing environment role ids and avoids a large attention-critic implementation at first.

---

## Updated Direction

There are now two different next steps:

```text
Immediate cheap probe:
    Role-Trust LoRA MAPPO

Main next research direction:
    Role-MACA-Lite MAPPO
```

Role-Trust is already implemented and should still be smoke-tested because it is cheap. But the MACA paper suggests that if Role-Trust does not clearly improve, we should not tune it for long. Move to MACA-lite credit assignment.

Role-Trust short pitch:

```text
Role-Trust LoRA MAPPO keeps the validated Role-LoRA actor, but replaces the global actor objective with an equal-role PPO objective and a soft per-role KL budget so rare or unstable roles cannot be drowned out by the average MAPPO update.
```

Role-MACA-Lite short pitch:

```text
Role-MACA-Lite keeps the Role-LoRA actor, but replaces the blunt MAPPO advantage with a multi-level counterfactual advantage that credits individual agents, the full team, and same-role subgroups separately.
```

Decision:

```text
Do not treat Role-Trust as the final algorithm unless it clearly beats Role-LoRA.
Treat Role-Trust as a low-cost diagnostic.
Treat Role-MACA-Lite as the stronger literature-backed improvement path.
```

---

## Role-Trust Objective

Actor remains Role-LoRA:

```text
h_i,t      = GRU_theta(o_i,t, h_i,t-1)
z_base_i,t = f_theta(h_i,t)
delta_i,t  = B_role_i A_role_i h_i,t
z_i,t      = z_base_i,t + alpha * delta_i,t
pi_i,t     = softmax(mask(z_i,t))
```

For each role `r` present in the minibatch:

```text
M_r = 1[role_i,t = r]

A_r = (A - mean(A | M_r)) / (std(A | M_r) + eps)

rho_i,t = pi_new(a_i,t | o_i,t) / pi_old(a_i,t | o_i,t)

L_ppo_r =
    - mean_{M_r} min(
        rho_i,t * A_r,
        clip(rho_i,t, 1 - eps, 1 + eps) * A_r
    )

KL_r = mean_{M_r} [(rho_i,t - 1) - log(rho_i,t)]
```

Equal-role PPO objective:

```text
L_role_ppo = mean_{r in present_roles} L_ppo_r
```

Soft per-role KL budget:

```text
L_role_kl =
    mean_{r in present_roles} max(0, KL_r - role_kl_target)^2
```

Actor loss:

```text
L_actor =
    L_role_ppo
    + role_kl_coef * L_role_kl
    - ent_coef * mean_entropy
```

Critic remains standard MAPPO:

```text
L_critic = clipped value loss on centralized world-state value
```

Total update:

```text
L_total = L_actor + vf_coef * L_critic
```

Parameters updated:

```text
shared actor backbone:
    updated by L_actor

role LoRA adapters:
    updated by L_actor

critic:
    updated by L_critic

residual auxiliary branch:
    not used

context gate:
    not used
```

Difference from Role-LoRA:

```text
Same actor.
Different PPO objective.
The main actor loss is role-balanced and per-role trust-region aware.
```

Difference from Role-Residual:

```text
No separate weak auxiliary objective.
No adapter-only residual loss.
The adapter receives the main PPO signal, but that signal is now role-aware.
```

---

## Role-MACA-Lite Objective

This is the new main research direction suggested by the MACA paper.

Full MACA:

```text
A_i^MACA(s, a) = Q(s, a) - b_i^MACA(s, a)

b_i^MACA =
    psi_i^Jnt * b^Jnt
  + psi_i^Ind * b_i^Ind
  + psi_i^Cor * b_i^Cor
```

Where:

```text
b^Jnt:
    baseline marginalizing the full joint action under the policy

b_i^Ind:
    baseline marginalizing only agent i's action

b_i^Cor:
    baseline marginalizing the action subset C_i of agents correlated with i
```

Project-specific MACA-lite:

```text
C_i = {j : role_j = role_i}
```

So the three credit levels become:

```text
joint/team:
    did the whole joint action help?

individual:
    did this agent's action help?

same-role subgroup:
    did this unit type's coordinated action help?
```

Minimal fixed-weight version:

```text
b_i^RoleMACA =
    w_jnt  * b^Jnt
  + w_ind  * b_i^Ind
  + w_role * b_i^Role

A_i^RoleMACA = stop_gradient(Q_taken - b_i^RoleMACA)
```

Start with fixed weights to avoid MACA's CMA-ES coefficient optimization:

```text
w_jnt = 1/3
w_ind = 1/3
w_role = 1/3
```

Then ablate:

```text
joint only:
    approximates MAPPO-style advantage

individual only:
    approximates COMA-style individual counterfactual credit

role only:
    tests whether same-role subgroup credit is the useful addition

joint + individual + role:
    main Role-MACA-Lite method
```

Why this is better aligned than Role-Trust:

```text
Role-Trust changes how PPO averages role losses.
Role-MACA-Lite changes the advantage signal before PPO sees it.

The MACA ablations show that stronger critic architecture alone is not enough.
The multi-level advantage is the key mechanism.
```

Implementation warning:

```text
Do not implement Role-MACA-Lite as just another value head V(s).
It needs an action-conditioned Q critic or an equivalent way to compute counterfactual baselines.
```

Minimal critic design:

```text
Q_phi(world_state, joint_action_features, agent_id_or_role_id) -> scalar Q_i
```

For each transition:

```text
Q_taken:
    Q_phi(s, onehot_taken_joint_action, i)

b_i^Ind:
    Q_phi(s, joint_action_features where only agent i's action is replaced by pi_i(.|o_i), i)

b_i^Role:
    Q_phi(s, joint_action_features where all same-role agents' actions are replaced by their policy distributions, i)

b^Jnt:
    Q_phi(s, joint_action_features where all agents' actions are replaced by policy distributions, i)
```

Actor update:

```text
Use A_i^RoleMACA in the PPO surrogate.
Keep Role-LoRA actor unchanged.
```

Critic training:

```text
Train Q_taken toward rollout returns / GAE targets first.
Keep the first implementation simple and observable.
Do not add transformer attention or learned psi weights in the first version.
```

Diagnostics:

```text
q_taken_mean_by_role
baseline_joint_by_role
baseline_individual_by_role
baseline_role_by_role
adv_maca_mean_by_role
adv_maca_std_by_role
component_gap_ind = Q_taken - b_i^Ind
component_gap_role = Q_taken - b_i^Role
component_gap_joint = Q_taken - b^Jnt
role_maca_weight_jnt/ind/role
```

Decision gate:

```text
If Role-Trust does not beat Role-LoRA in the first two full seeds, prioritize Role-MACA-Lite.
If Role-Trust improves AUC/final WR, still consider Role-MACA-Lite as the stronger paper-backed extension.
```

---

## Implementation Status And Checklist

Status:

```text
Implemented in smax_ctm/train_rosa_mappo.py.
```

The implementation deliberately keeps existing `role_balanced` behavior unchanged as an older ablation. The new mode is separate:

```text
adapter_mode=role_trust_lora
```

Completed code changes:

```text
1. Added role_trust_lora to SUPPORTED_ADAPTER_MODES.
2. Added CLI args:
   --role_kl_target
   --role_kl_coef
   --role_kl_penalty_mode

3. For adapter_mode=role_trust_lora, apply_cli_overrides sets:
   USE_ROLE_LORA=True
   ROLE_BALANCED_PPO=True
   ROLE_EQUALIZE_PPO_LOSS=True
   ROLE_KL_BUDGET=True
   FREEZE_LORA_IN_SHARED_UPDATE=False
   USE_ROLE_RESIDUAL_LOSS=False
   USE_CONTEXT_GATE=False

4. Added config defaults:
   ROLE_EQUALIZE_PPO_LOSS=False
   ROLE_KL_BUDGET=False
   ROLE_KL_TARGET=0.02
   ROLE_KL_COEF=0.5
   ROLE_KL_PENALTY_MODE="hinge_squared"

5. In _actor_loss_fn:
   role_trust_lora role-normalizes advantages,
   computes PPO loss per sample,
   computes role_ppo_loss as mean loss per role,
   averages present-role losses equally,
   adds a soft per-role KL budget penalty.

6. Added logging for:
   role_actor_loss_weight
   role_kl_penalty_by_role
   role_kl_penalty

7. Added fail-loud runtime check for role ids outside [0, NUM_UNIT_TYPES).
```

Critical implementation invariant:

```text
role_trust_lora must NOT be implemented as:
    role-normalize advantages, then global sample mean

It must be:
    compute PPO loss per sample
    compute mean PPO loss within each present role
    average present-role PPO losses equally
```

Current code path:

```text
role_ppo_loss = role_mean_metric(ppo_loss_per_sample, role_id, NUM_UNIT_TYPES)
policy_loss = present_role_mean(role_ppo_loss, role_id, NUM_UNIT_TYPES)
```

The `role_actor_loss_weight` diagnostic should show equal weights for present roles under `role_trust_lora`.

Config additions:

```python
"ROLE_EQUALIZE_PPO_LOSS": True
"ROLE_KL_BUDGET": True
"ROLE_KL_TARGET": 0.02
"ROLE_KL_COEF": 0.5
"ROLE_KL_PENALTY_MODE": "hinge_squared"
```

Optional later:

```python
"ROLE_MIN_COUNT_FOR_LOSS": 8
"ROLE_ENTROPY_BALANCED": False
"ROLE_ADAPTIVE_CLIP": False
```

CLI additions:

```text
--role_kl_target
--role_kl_coef
--role_kl_penalty_mode
```

Startup logging must print:

```text
resolved config
adapter_mode
map_name
seed
run_name
role trust config values
residual/context config values, if any are enabled
```

Fail-loud checks:

```text
If adapter_mode is unsupported, raise.
If role ids fall outside [0, NUM_UNIT_TYPES), raise.
If role bits in observation disagree with env-state role ids, raise.
If ROLE_KL_BUDGET=True but per-role KL is unavailable, raise.
```

Diagnostics to log:

```text
role_count
role_present_mask
role_actor_loss_weight
role_adv_mean
role_adv_std
role_ppo_loss
role_approx_kl
role_kl_penalty
role_clip_frac
role_mean_ratio
role_max_ratio
max_role_kl_minus_global
max_role_clip_frac_minus_global
min_role_count_over_max
lora_delta_norm_by_role
lora_grad_norm_by_adapter
lora_param_norm_by_adapter
```

Expected diagnostics if Role-Trust is working:

```text
role_actor_loss_weight gives each present role a meaningful contribution.
max_role_kl_minus_global shrinks.
role_clip_frac becomes less spiky.
lora_grad_norm_by_adapter is less dominated by common roles.
Win-rate AUC improves before final win rate does.
```

Main risks:

```text
Equal role weighting may overemphasize rare/noisy roles.
The KL penalty may be too conservative.
Role-normalized advantages may amplify noise when role counts are tiny.
The current map may already be stable enough that Role-Trust only matches Role-LoRA.
```

---

## Minimal Experiment Plan

Do not run a large grid.

### Smoke

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode role_trust_lora --total_timesteps 300000 --run_name smoke_role_trust_lora_smacv2_10_seed42
```

Smoke success criteria:

```text
no crash or NaN
role ids pass validation
role_actor_loss_weight prints and is sensible
role_kl_penalty is finite
lora_grad_norm_by_adapter is nonzero
```

### Full Runs

Start with two seeds:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode role_trust_lora --total_timesteps 3000000 --run_name role_trust_lora_smacv2_10_seed42
```

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 1 --adapter_mode role_trust_lora --total_timesteps 3000000 --run_name role_trust_lora_smacv2_10_seed1
```

If either seed is unstable or worse than Role-LoRA by more than normal wobble, try exactly one softer KL setting:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode role_trust_lora --role_kl_coef 0.1 --total_timesteps 3000000 --run_name role_trust_lora_kl010_smacv2_10_seed42
```

If seed 42 and seed 1 look promising, add:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 0 --adapter_mode role_trust_lora --total_timesteps 3000000 --run_name role_trust_lora_smacv2_10_seed0
```

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 2 --adapter_mode role_trust_lora --total_timesteps 3000000 --run_name role_trust_lora_smacv2_10_seed2
```

Success criteria:

```text
Minimum:
    match Role-LoRA final WR while improving AUC or reducing variance

Good:
    beat Role-LoRA by >= 0.02 final WR or AUC on at least two checked seeds

Strong:
    beat sequential_polish mean without adding new architecture
```

Drop criteria:

```text
Drop if seed 42 and seed 1 both fail to beat Role-LoRA/sequential polish.
Drop if role KL penalty stays zero and diagnostics look identical to plain Role-LoRA.
Drop if role KL penalty dominates and clip fraction collapses.
Drop if rare-role weighting causes noisy oscillation without AUC gain.
```

### MACA-Lite Gate

Do not tune Role-Trust for many runs.

```text
If the 300k smoke passes, run at most two full Role-Trust seeds first: 42 and 1.
```

Then decide:

```text
If Role-Trust clearly improves:
    finish the 4-seed check and report it as a lightweight actor-side credit/stability improvement.

If Role-Trust only matches or underperforms:
    stop tuning Role-Trust and implement Role-MACA-Lite.
```

Minimal Role-MACA-Lite implementation order:

```text
1. Add action-conditioned Q critic that can score:
   Q(s, taken joint action, i)
   Q(s, counterfactual joint-action distribution, i)

2. Add fixed-weight baselines:
   b_jnt
   b_ind
   b_role

3. Construct:
   A_i = stop_gradient(Q_taken - (b_jnt + b_ind + b_role) / 3)

4. Use A_i in the existing PPO actor loss.

5. Only after this works, consider learned weights or attention CorrSets.
```

First Role-MACA-Lite ablations:

```text
role_lora:
    current validated baseline

role_trust_lora:
    actor-side balancing probe

role_maca_lite_jnt:
    joint baseline only, should behave closest to MAPPO-style credit

role_maca_lite_ind:
    individual baseline only, COMA-like credit

role_maca_lite_role:
    same-role subgroup baseline only

role_maca_lite:
    fixed average of joint + individual + same-role baselines
```

---

## Candidate Directions After Role-Trust

Ranked by implementation priority:

### 1. Role-MACA-Lite

```text
Multi-level counterfactual advantage using:
    joint/team baseline
    individual-agent baseline
    same-role subgroup baseline

This is now the strongest paper-backed direction.
It directly addresses the advantage credit-assignment bottleneck that Role-Residual and Role-Trust only touch indirectly.
```

Minimal implementation:

```text
Keep Role-LoRA actor.
Add action-conditioned Q critic.
Use fixed weights over joint / individual / same-role baselines.
Use same-role agents as the first CorrSet approximation.
```

### 2. PRD-Lite Role Credit Critic

Add a learned relevance or role-pair critic that estimates which roles or teammates matter for each agent's return.

Sketch:

```text
w_i,j,t = softmax(attn(q_i,t, k_j,t))
A_i,t^credit = GAE using relevance-filtered rewards/value signal
```

Why it is interesting:

```text
This is close to MACA's learned CorrSet idea, but less directly tied to multi-level counterfactual baselines.
```

Why it is not first:

```text
Role-MACA-Lite gives us a cleaner equation and clearer ablations.
```

### 3. Role-Diverse LoRA

Add a light behavior-diversity term between role-conditioned policies.

For the same hidden states, compute action distributions under each role adapter:

```text
pi_r(a | h) = softmax(z_base(h) + delta_r(h))
```

Add:

```text
L_div = - lambda_js * mean_{r < s} JS(pi_r || pi_s)
```

Use tiny coefficients only:

```text
lambda_js in {0.001, 0.003}
```

Risk:

```text
Forced diversity can hurt when roles should sometimes behave similarly.
```

### 4. Role-Seq-Trust LoRA

Revisit sequential role updates only after Role-Trust.

Mechanism:

```text
Update one role adapter at a time.
Use role-normalized PPO objective.
Use explicit per-role KL budgets.
Do not use residual auxiliary loss.
```

Risk:

```text
Prior sequential polish gains were small, so this is not the first bet.
```

---

## Presentation Framing

Clean story:

```text
1. Recurrent MAPPO is a strong baseline, but it uses one shared actor update for randomized heterogeneous unit roles.

2. Role-LoRA improves MAPPO by adding semantic role-conditioned low-rank policy adapters.

3. We tested natural extensions: sequential role polishing, role residual training, and context-gated residuals.

4. These extensions were mechanically valid but did not clearly beat Role-LoRA.

5. This negative result is useful: the bottleneck is probably not more residual capacity.

6. MACA sharpens the diagnosis: MAPPO's joint advantage is too blunt, and good cooperative MARL needs multi-level credit assignment.

7. Role-Trust LoRA is a cheap already-implemented probe that makes PPO's actor update role-aware.

8. The stronger next method is Role-MACA-Lite: keep the successful Role-LoRA actor, but replace the advantage with a multi-level counterfactual advantage over individual, team, and same-role subgroup credit.
```

Expected contribution:

```text
Role-LoRA shows that semantic unit roles are useful specialization keys in randomized SMAX.

Role-Trust LoRA tests whether actor-side role-balanced PPO is enough.

Role-MACA-Lite tests the stronger MACA-inspired hypothesis: the actor needs role-specific capacity and multi-level credit assignment.
```

Meeting-safe claim before Role-Trust results:

```text
Our immediate next step is tightly scoped: keep the successful Role-LoRA actor and test whether role-balanced PPO with per-role trust budgets improves stability. The stronger follow-up, motivated by MACA, is to replace the MAPPO advantage with a multi-level role-aware counterfactual advantage.
```

Meeting-safe claim if Role-Trust fails:

```text
Role-LoRA remains the robust contribution. Residual auxiliary losses, context gates, and role-balanced trust budgets did not consistently add signal on this benchmark, which supports the MACA diagnosis: actor-side loss shaping is not enough, and the next improvement should change the advantage credit-assignment signal itself.
```

---

## Handoff Template

After any implementation stage, hand off in this format:

```text
Files changed:
- ...

Local syntax check:
python -m py_compile /Users/hassan/repos/new_marl_llm_implementation/smax_ctm/train_rosa_mappo.py

Notebook command:
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py ...

Please send back:
- last 30-50 log lines
- any role-wise diagnostic table printed
- whether the run NaN'd or crashed
```

Every stage must answer a question. Do not build all stages first and only then learn that the hypothesis was wrong.
