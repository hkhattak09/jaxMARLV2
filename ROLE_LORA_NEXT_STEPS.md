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

### Role-Trust LoRA

Mechanism:

```text
Keep the validated Role-LoRA actor.
Use role-normalized advantages.
Compute PPO loss per role.
Average present-role PPO losses equally.
Add a hinge-squared per-role KL budget.
```

Empirical status:

```text
Mechanically healthy.
Smoke run passed.
Full seed-42 run underperformed plain Role-LoRA.
Final WR was around 0.45, with near-end WR around 0.49 before dropping.
This is below seed-42 Role-LoRA, which reached about 0.54 final WR and 0.55 best WR.
```

Key diagnostics from the full run:

```text
role_actor_loss_weight = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    Equal role weighting was active.

lora_grad_norm_by_adapter > 0
lora_delta_norm_by_role became large and role-specific.
    Role-LoRA adapters were alive and training.

role_approx_kl late in training was around 1e-05 to 5e-05.
ROLE_KL_TARGET = 0.02.
role_kl_penalty_by_role = 0 for all roles.
    The trust-region penalty was inactive.

role_clip_frac was near zero late in training.
role_mean_ratio was about 1.0.
    PPO updates were already tiny near the end.
```

Interpretation:

```text
Role-Trust mostly tested equal role weighting on top of the same MAPPO/team advantage.
It did not add a genuinely new credit signal.
The KL-budget part did not matter because the per-role KL stayed far below target.
Equal role weighting may have distorted useful gradients by forcing all roles to contribute equally even when their tactical importance differs.
```

Decision:

```text
Stop tuning Role-Trust.
Do not spend more full runs on Role-Trust unless a reproduction check is explicitly needed.
Keep it as a negative actor-side reweighting result.
Move priority to Role-MACA-Lite, because it changes the advantage/credit signal itself.
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

Role-Trust did not improve because it still used the same blunt team/global advantage.
Its per-role KL budget stayed inactive, and equal role weighting alone was not a useful replacement for credit assignment.
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

The direction is now:

```text
Completed cheap probe:
    Role-Trust LoRA MAPPO
    Result: underperformed Role-LoRA; stop tuning.

Main next implementation:
    Role-MACA-Lite MAPPO
```

Role-Trust was useful as a diagnostic. It showed that role-balanced actor-side PPO reweighting is not enough when the underlying advantage is still the same blunt team/global signal.

Role-Trust short pitch:

```text
Role-Trust LoRA MAPPO keeps the validated Role-LoRA actor, but replaces the global actor objective with an equal-role PPO objective and a soft per-role KL budget so rare or unstable roles cannot be drowned out by the average MAPPO update.
```

Role-MACA-Lite short pitch:

```text
Role-MACA-Lite keeps the Role-LoRA actor, but blends MAPPO's GAE advantage with a multi-level counterfactual advantage that credits individual agents, the full team, and same-role subgroups separately.
```

Decision:

```text
Do not treat Role-Trust as the final algorithm.
Treat Role-Trust as completed negative evidence.
Treat Role-MACA-Lite as the stronger literature-backed improvement path and the next implementation target.
```

---

## Role-Trust Objective

This section is kept as implementation reference only. Role-Trust is implemented and tested, but it underperformed Role-LoRA and should not be tuned further.

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
Use a conservative blend in the PPO surrogate:

A_i^actor =
    (1 - alpha_maca) * A_i^GAE
  + alpha_maca       * A_i^RoleMACA

alpha_maca = 0.15

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
Resolved.
Role-Trust did not beat Role-LoRA.
Prioritize Role-MACA-Lite.
```

---

## Active Training File Status

The active experimental file is now intentionally cleaned up:

```text
smax_ctm/train_rosa_mappo.py
```

Currently supported adapter modes:

```text
none
role_lora
global_lora
agent_lora
role_maca_lite_jnt
role_maca_lite_ind
role_maca_lite_role
role_maca_lite
```

Removed from the active training file:

```text
role_trust_lora
role_balanced
sequential_polish
role_residual
role_residual_clean
role_context_residual
global_residual
agent_residual
shuffled_context_residual
```

Why these were removed:

```text
Role-Trust underperformed Role-LoRA and mostly tested equal role weighting with inactive KL.
Role-Residual and context-gated residual paths were mechanically valid but weak/negative.
Sequential polish was a small refinement, not the next research direction.
The next stage needs a clean Role-LoRA baseline surface for Role-MACA-Lite credit assignment.
```

The active training file should now be treated as:

```text
MAPPO baseline pathway
+ Role-LoRA actor pathway
+ global/agent LoRA ablations
+ Role-MACA-Lite first implementation
+ compact role diagnostics
```

Do not re-add deprecated actor-side reweighting/residual/context/sequential paths unless explicitly reopening those experiments.

Role-MACA-Lite implementation status:

```text
Implemented first pass in smax_ctm/train_rosa_mappo.py.
Uses a separate action-conditioned feed-forward Q critic.
Precomputes counterfactual advantages on the full rollout before minibatching.
Uses blended actor advantages, not pure MACA advantages:
    A_actor = 0.85 * A_GAE + 0.15 * A_RoleMACA
Uses fixed MACA weights:
    role_maca_lite_jnt  = joint only
    role_maca_lite_ind  = individual only
    role_maca_lite_role = same-role subgroup only
    role_maca_lite      = 1/3 joint + 1/3 individual + 1/3 same-role
```

Important implementation choice:

```text
Counterfactual joint-action features are built before PPO minibatching because minibatches contain shuffled subsets of agents and cannot reconstruct full joint actions safely.
The computed A_i^RoleMACA is stop-gradient before entering the PPO actor loss.
Pure A_i^RoleMACA was too weak in the 300k smoke run: finite, but small/collapsing and it caused fast entropy collapse.
The active implementation therefore blends it with normal GAE using ROLE_MACA_BLEND_ALPHA = 0.15.
The Q critic is trained toward the same return targets used by the existing value critic.
```

---

## Minimal Experiment Plan

Do not run a large grid.

### Completed Role-Trust Gate

```text
Role-Trust smoke:
    passed

Role-Trust full seed 42:
    underperformed Role-LoRA

Decision:
    stop Role-Trust
    do not run seed 1 or KL variants unless explicitly checking reproducibility
    implement Role-MACA-Lite next
```

### Current Implementation: Blended Role-MACA-Lite

Implementation target:

```text
Keep the Role-LoRA actor.
Add an action-conditioned Q critic.
Use Q counterfactual baselines to add a small multi-level credit correction to the normal MAPPO/GAE advantage.
Do not implement learned attention CorrSets or learned MACA weights in the first version.
```

Minimal Role-MACA-Lite implementation order:

```text
1. Add adapter modes:
   role_maca_lite_jnt
   role_maca_lite_ind
   role_maca_lite_role
   role_maca_lite

2. Keep Role-LoRA actor unchanged for all role_maca_lite modes.

3. Add an action-conditioned Q critic:
   Q_phi(world_state, joint_action_features, agent_or_role_id) -> Q_i

4. Store or reconstruct joint action features:
   taken joint action one-hots
   current policy action probabilities for counterfactual baselines

5. Train Q_taken toward the same return target used by the existing critic first.
   Keep the existing V critic/value loss if that is easier for bootstrapping.

6. Add fixed-weight counterfactual baselines:
   b_jnt
   b_ind
   b_role

7. Construct:
   A_i = stop_gradient(Q_taken - (b_jnt + b_ind + b_role) / 3)

8. Blend with normal GAE before the existing PPO actor loss:
   A_actor = 0.85 * A_GAE + 0.15 * A_i
   Start with the same advantage normalization style as the current Role-LoRA run.

9. Only after this works, consider learned weights or attention CorrSets.
```

First Role-MACA-Lite ablations:

```text
role_lora:
    current validated baseline

role_trust_lora:
    completed negative actor-side balancing probe; keep only as reference

role_maca_lite_jnt:
    joint baseline only, should behave closest to MAPPO-style credit

role_maca_lite_ind:
    individual baseline only, COMA-like credit

role_maca_lite_role:
    same-role subgroup baseline only

role_maca_lite:
    fixed average of joint + individual + same-role baselines
```

Smoke command after implementation:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode role_maca_lite --total_timesteps 300000 --run_name smoke_role_maca_lite_smacv2_10_seed42
```

Smoke success criteria:

```text
no crash or NaN
Q loss finite
Q_taken finite
b_jnt / b_ind / b_role finite
A_i^RoleMACA finite and non-collapsed
lora_grad_norm_by_adapter nonzero
role_maca diagnostics print by role
```

First full runs if smoke passes:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode role_maca_lite --total_timesteps 3000000 --run_name role_maca_lite_smacv2_10_seed42
```

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 1 --adapter_mode role_maca_lite --total_timesteps 3000000 --run_name role_maca_lite_smacv2_10_seed1
```

Initial success criteria:

```text
Minimum:
    not worse than Role-LoRA on seed 42 by more than normal seed wobble

Good:
    beat Role-LoRA seed-42 final WR or AUC

Strong:
    beat Role-LoRA on both seed 42 and seed 1, then expand to more seeds
```

Drop / revise criteria:

```text
Drop the current Role-MACA-Lite version if Q loss is unstable or A_i collapses.
Revise baseline construction if one component dominates all advantage variance.
Revise same-role CorrSet if role baseline is uninformative or clearly harmful.
Do not add attention CorrSets until fixed same-role CorrSet has been tested cleanly.
```

---

## Candidate Directions Now

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

Low-priority fallback. Do not revisit until Role-MACA-Lite has been tested.

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
Role-Trust also underperformed, so more actor-side trust-region shaping is less attractive than credit assignment.
```

---

## Presentation Framing

Clean story:

```text
1. Recurrent MAPPO is a strong baseline, but it uses one shared actor update for randomized heterogeneous unit roles.

2. Role-LoRA improves MAPPO by adding semantic role-conditioned low-rank policy adapters.

3. We tested natural extensions: sequential role polishing, role residual training, context-gated residuals, and Role-Trust role-balanced PPO.

4. These extensions were mechanically valid but did not clearly beat Role-LoRA. Role-Trust specifically underperformed because it reweighted the same blunt team advantage and its KL budget stayed inactive.

5. This negative result is useful: the bottleneck is probably not more residual capacity or actor-side loss weighting.

6. MACA sharpens the diagnosis: MAPPO's joint advantage is too blunt, and good cooperative MARL needs multi-level credit assignment.

7. The next method is Role-MACA-Lite: keep the successful Role-LoRA actor, but blend the normal GAE advantage with a multi-level counterfactual advantage over individual, team, and same-role subgroup credit.

8. This gives a clean literature-backed progression from role-specific capacity to role-specific credit assignment.
```

Expected contribution:

```text
Role-LoRA shows that semantic unit roles are useful specialization keys in randomized SMAX.

Role-Trust LoRA shows that actor-side role-balanced PPO is not enough.

Role-MACA-Lite tests the stronger MACA-inspired hypothesis: the actor needs role-specific capacity and multi-level credit assignment.
```

Current meeting-safe claim:

```text
Role-LoRA remains the robust contribution. Residual auxiliary losses, context gates, and role-balanced trust budgets did not consistently add signal on this benchmark, which supports the MACA diagnosis: actor-side loss shaping is not enough. The next implementation keeps the successful Role-LoRA actor and blends MAPPO's GAE advantage with Role-MACA-Lite multi-level credit assignment.
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
