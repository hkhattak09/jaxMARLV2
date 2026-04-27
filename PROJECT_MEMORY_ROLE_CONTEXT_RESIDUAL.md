# Project Memory: Role-LoRA and Residual MAPPO

This file is a compact memory for future work on the MAPPO/Role-LoRA/Role-Context Residual project.

## Current Date Context

Work is happening in:

```text
/Users/hassan/repos/new_marl_llm_implementation
```

The original GRU MAPPO baseline is:

```text
smax_ctm/train_mappo_gru.py
```

Do not modify it.

The experimental file is:

```text
smax_ctm/train_rosa_mappo.py
```

The main planning file is:

```text
ROLE_LORA_NEXT_STEPS.md
```

Training is run by the user in Colab/notebook, not locally. Local work should be limited to code edits and syntax checks.

Local syntax check:

```bash
python -m py_compile /Users/hassan/repos/new_marl_llm_implementation/smax_ctm/train_rosa_mappo.py
```

Typical Colab command:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py
```

Baseline command:

```python
!python /content/jaxMARLV2/smax_ctm/train_mappo_gru.py
```

Error handling philosophy:

```text
Fail loud, never fake.
Prefer clear failure over silent fallback.
Do not substitute placeholder data that looks valid.
If a metric is unavailable, say unavailable rather than returning fake zeros.
```

Current operational state:

```text
The user is running experiments in Colab.
Do not ask the user to run large grids unless a single run has justified it.
Smoke tests should use total_timesteps=300000.
Full comparisons should use total_timesteps=3000000.
Stage 4B full role_context_residual seed-42 run is complete.
It did not beat Stage 4A role_residual, so context gating should not be the core method.
Stage 4A role_residual seed-1 validation run is complete.
Using best/peak win rate as the less noisy signal, it tied seed-1 role_lora at 0.53 but did not beat sequential_polish at 0.54.
Do not expand Stage 4 residual experiments right now.
Current strongest evidence is the Stage 3 Role-LoRA multi-seed result, with sequential_polish as a small optional refinement.
```

---

## Research Trajectory So Far

Original direction:

```text
ROSA-MAPPO = Role-Ordered Sequential Adapter MAPPO
```

It used:

```text
shared GRU MAPPO backbone
role-specific LoRA adapters
HAPPO-style sequential role adapter updates
```

Empirical result:

```text
Full sequential ROSA did not improve over joint Role-LoRA.
Sequential correction was active mechanically, but it did not give a clear learning benefit.
```

Interpretation:

```text
HAPPO-style sequencing does not transfer cleanly to adapter-only updates after a shared MAPPO backbone update.
The shared PPO update already keeps role-wise KL/clip metrics controlled.
The useful ingredient is role-conditioned residual specialization, not full sequential correction.
```

Current direction after the latest checks:

```text
Role-LoRA MAPPO as the main validated method.
Role-Residual MAPPO is a mechanically valid explored extension. It is competitive by peak win rate, but not clearly better than the Stage 3 methods.
```

Plain Role-LoRA is now the strongest validated result. The residual objective is interesting and mechanically correct, but the seed-1 check only tied Role-LoRA by peak win rate and did not justify making it the main method yet.

---

## Important Empirical Result: Stage 3B Multi-Seed

Map:

```text
smacv2_10_units
```

Seeds:

```text
0, 1, 2, 42, 43
```

Modes:

```text
none
role_lora
sequential_polish
```

Aggregate summary:

| mode | num_seeds | final_win_rate_mean | final_win_rate_std | best_win_rate_mean | final_return_mean | auc_win_rate_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 5 | 0.4620 | 0.0259 | 0.4940 | 1.3100 | 0.2386 |
| role_lora | 5 | 0.5060 | 0.0152 | 0.5300 | 1.3640 | 0.2596 |
| sequential_polish | 5 | 0.5120 | 0.0164 | 0.5300 | 1.3760 | 0.2665 |

Key interpretation:

```text
Role-LoRA beats baseline on final win rate in all 5 seeds.
Role-LoRA also improves AUC and lowers final-win-rate std.
Sequential polish gives a small average gain over Role-LoRA, but the gain is modest.
```

Numerical deltas:

```text
role_lora final WR gain over none: +0.044
sequential_polish final WR gain over none: +0.050
sequential_polish final WR gain over role_lora: +0.006

role_lora AUC gain over none: +0.0210
sequential_polish AUC gain over none: +0.0279
sequential_polish AUC gain over role_lora: +0.0069
```

Recommended wording:

```text
Across five seeds on randomized smacv2_10_units, role-conditioned LoRA improves final win rate from 46.2% to 50.6% and AUC from 0.2386 to 0.2596. A lightweight role-local polishing step gives a small further gain to 51.2%, but the larger and cleaner effect is the role-conditioned residual parameterization.
```

---

## Earlier Single-Seed Ablation

Map:

```text
smacv2_10_units
```

Seed:

```text
42
```

Summary:

| mode | final_win_rate | best_win_rate | auc_win_rate |
| --- | ---: | ---: | ---: |
| none | 0.4900 | 0.5500 | 0.2956 |
| role_lora | 0.5400 | 0.5500 | 0.3101 |
| global_lora | 0.4900 | 0.5400 | 0.2840 |
| agent_lora | 0.5200 | 0.5500 | 0.2992 |
| sequential_polish | 0.5500 | 0.5700 | 0.3185 |

Key interpretation:

```text
Role-LoRA beats global LoRA and agent-ID LoRA.
Global LoRA is worse than baseline in this seed.
This helps answer the "you just added parameters" criticism.
Agent-ID LoRA is weaker than role conditioning on randomized teams.
```

Need follow-up:

```text
Multi-seed ablation for global_lora and agent_lora is still useful if time allows.
```

---

## Latest Stage 4 Evidence

Important implementation fixes already made:

```text
1. Fixed a NameError caused by shadowing the residual_gate_grad_norm helper with a local variable.
2. Fixed a serious optimizer bug: actor_train_state.apply_gradients must be called exactly once per update.
   A second Adam step with zero residual gradients still moved parameters via optimizer momentum and caused collapse.
3. Residual gradients are now combined with normal actor gradients, then one optimizer step is applied.
```

Stage 4A.0 zero-coefficient smoke, seed 42:

```text
mode: role_residual
map: smacv2_10_units
total_timesteps: 300000
residual_loss_coef: 0.0
residual_kl_coef: 0.0
result: passed after the single-apply_gradients fix
```

Interpretation:

```text
Zero-coefficient residual mode no longer collapses.
residual_lora_grad_norm_by_role stayed 0.
residual_backbone_grad_norm stayed 0.
This validates the residual wiring enough to run nonzero residual loss.
```

Stage 4A full run, seed 42:

```text
mode: role_residual
map: smacv2_10_units
total_timesteps: 3000000
residual_loss_coef: 0.1
residual_kl_coef: 0.01
final win rate: 0.52
best observed late win rate in pasted logs: 0.56
final return: 1.38
```

Interpretation:

```text
Role-Residual is mechanically stable and gives a small seed-42 lift over the prior Stage 3/seq-polish references.
The auxiliary residual gradients update LoRA only: residual_backbone_grad_norm stayed 0.
Late residual KL to base is around 0.017-0.019, while residual_loss is tiny, so the KL term restrains the residual strongly late in training.
```

Stage 4A validation run, seed 1:

```text
mode: role_residual
map: smacv2_10_units
total_timesteps: 3000000
residual_loss_coef: 0.1
residual_kl_coef: 0.01
final pasted step: 2981888
final win rate: 0.50
final return: 1.37
best observed in pasted final logs: 0.53
```

Compare against existing seed-1 Stage 3 results. In this stochastic environment, best/peak win rate is more informative than the final checkpoint alone, because the final evaluation can wobble:

| mode | seed 1 final win rate | seed 1 best/peak win rate |
| --- | ---: | ---: |
| none | 0.49 | 0.51 |
| role_lora | 0.53 | 0.54 |
| sequential_polish | 0.54 | 0.54 |
| role_residual | 0.50 | 0.53 |

Interpretation:

```text
Role-Residual is still mechanically healthy on seed 1: residual_lora_grad_norm_by_role is nonzero, residual_backbone_grad_norm stays 0, and residual_gate_grad_norm is 0 because no context gate is used.
Behaviorally, seed 1 makes Role-Residual competitive but not clearly better.
By peak win rate it reaches 0.53, roughly tying Role-LoRA but not beating sequential_polish at 0.54.
By final win rate it is worse, but final checkpoint is not the preferred signal under high environment stochasticity.
Do not run more role_residual seeds, global_residual, agent_residual, or role_residual coefficient sweeps unless the project explicitly reopens Stage 4 later.
```

Stage 4B smoke run, seed 42:

```text
mode: role_context_residual
map: smacv2_10_units
total_timesteps: 300000
residual_loss_coef: 0.1
residual_kl_coef: 0.01
final pasted win rate: 0.02
best pasted smoke win rate: 0.04
```

Interpretation:

```text
Context gate wiring is mechanically safe: residual_gate_grad_norm is nonzero, residual_backbone_grad_norm stays 0, and no collapse occurs.
The gate opens steadily from about 0.50 to roughly 0.75-0.80 by 278k steps, with max values near 0.99.
This means the gate is alive, but it may be learning "turn residual on" rather than selective team-context modulation.
This justified one full 3e6 context-gated run as a decisive check; that full run is recorded below.
```

Stage 4B full run, seed 42:

```text
mode: role_context_residual
map: smacv2_10_units
total_timesteps: 3000000
residual_loss_coef: 0.1
residual_kl_coef: 0.01
final pasted step: 2981888
final win rate: 0.50
final return: 1.34
best observed in pasted final logs: 0.51
```

Interpretation:

```text
The full context-gated residual run is mechanically healthy, but behaviorally worse than Stage 4A.
It does not beat Stage 4A role_residual, which had final win rate 0.52 and best observed late win rate 0.56.
residual_backbone_grad_norm stayed 0, residual_gate_grad_norm was nonzero, and residual_lora_grad_norm_by_role stayed nonzero.
Late context_gate_mean_by_role saturated high, roughly 0.83-0.93, with max values at 1.0.
This suggests the gate mostly learned to turn the residual on rather than provide useful selective team-context modulation.
Drop context gating from the core method.
Do not run shuffled_context_residual or retained-context Stage 4C unless the project later reopens the context-gating branch.
```

---

## Current Best Framing

Do not headline full ROSA or the context-gated residual branch.

Use:

```text
Role-LoRA MAPPO
```

Important status:

```text
Stage 3 Role-LoRA has the strongest multi-seed evidence.
Sequential polish gives a small additional gain but should not be oversold as HAPPO.
Stage 4A Role-Residual was promising on seed 42 and competitive on seed 1 by peak win rate, but it did not clearly improve over Stage 3.
Stage 4B context gating was tested and did not beat Stage 4A.
Role-Residual MAPPO and Role-Context Residual MAPPO should be described as explored extensions, not the main contribution.
```

Short pitch:

```text
Role-LoRA MAPPO improves shared recurrent MAPPO by adding role-specific low-rank policy adapters, allowing randomized unit-type roles to specialize while preserving the shared recurrent MAPPO backbone.
```

Plain Role-LoRA is:

```text
validated main component
```

Sequential ROSA is:

```text
negative/redundant ablation
```

Sequential polish is:

```text
small role-local adapter refinement
keep as optional current-best practical variant
do not oversell as HAPPO
```

Role-Residual is:

```text
mechanically valid explored extension
promising on seed 42 and competitive on seed 1 by peak win rate
not clearly better than role_lora/sequential_polish
do not expand right now
```

---

## Current Idea From Start To Finish

Shared GRU MAPPO actor:

```text
h_i,t = GRU_theta(o_i,t, h_i,t-1)
z_base_i,t = f_theta(h_i,t)
```

Role-LoRA residual:

```text
delta_i,t = B_role_i A_role_i h_i,t
```

Context gate:

```text
gate_i,t = sigmoid(MLP([h_i,t, onehot(role_i), c_t]))
```

Final logits:

```text
z_i,t = z_base_i,t + gate_i,t * delta_i,t
```

Policy:

```text
pi_full(a_i,t | o_i,t, role_i, c_t) = softmax(z_i,t)
pi_base(a_i,t | o_i,t) = softmax(z_base_i,t)
```

Sable-inspired retained team context:

```text
x_t = team role histogram / own-role count / alive role counts
c_t = (1 - done_t) * (rho * c_{t-1} + phi(x_t))
```

Borrowed from Sable:

```text
persistent temporal multi-agent context matters
memory must reset at episode boundaries
avoid fixed agent-slot bias
do not simply add a larger sequence model
```

Not borrowed:

```text
full Sable/RetNet architecture
Mamba replacement
autoregressive action decoder
full sequence-model MARL replacement
```

Loss:

```text
L_total = L_MAPPO + lambda_residual * L_residual + beta_kl * KL(stop_gradient(pi_base) || pi_full)
```

Normal MAPPO actor loss:

```text
L_MAPPO = -E[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)]
```

Role-normalized advantage:

```text
A_role_i,t = (A_i,t - mean_role_i(A)) / (std_role_i(A) + eps)
```

Adapter-only residual loss:

```text
ratio_residual = pi_full(a_i,t) / pi_old(a_i,t)
L_residual = -E[min(ratio_residual * A_role,
                    clip(ratio_residual, 1 - eps, 1 + eps) * A_role)]
```

Important:

```text
For L_residual, shared hidden features and base logits must be stop_gradient.
Only role LoRA and context gate should receive residual-loss gradients.
Residual gradients must be added to normal actor gradients and applied in one optimizer step.
Do not call actor_train_state.apply_gradients twice with the same Adam state.
Even zero residual gradients can move parameters on a second Adam step because momentum is nonzero.
```

KL:

```text
L_KL = E[KL(stop_gradient(pi_base) || pi_full)]
Do not let this KL term update the base/backbone path.
```

Purpose:

```text
L_MAPPO learns general cooperation.
L_residual teaches each role adapter how to deviate from the shared policy.
L_KL keeps deviations controlled.
The context gate was explored, but it did not improve the seed-42 full run and should not be part of the core method.
```

---

## Next Recommended Experiments

Immediate next step:

```text
Stop Stage 4 expansion for now.
Consolidate the Stage 3 Role-LoRA / sequential_polish story.
```

Question now:

```text
How should the final writeup explain why role-conditioned adapters help over the MAPPO baseline?
```

Current key comparison:

```text
none
role_lora
sequential_polish
```

Recommended map:

```text
smacv2_10_units
```

Recommended experiments:

```text
Do not run more experiments immediately.
If time later allows exactly one useful ablation, prefer a multi-seed global_lora or agent_lora comparison against role_lora to answer the "just extra parameters" objection.
Do not run more role_residual seeds unless Stage 4 is explicitly reopened.
```

Stage 4 residual branch:

```text
Do not run more role_residual seeds now.
Do not run global_residual or agent_residual now.
Do not run shuffled_context_residual now.
Do not tune residual coefficients now.
```

Retained-context Stage 4C:

```text
Do not implement or run retained team-context gate now.
Only revisit if there is a new reason to reopen context gating.
```

Possible later ablations for Role-Residual:

```text
global_residual
agent_residual
role_balanced
```

---

## Diagnostics To Log In Stage 4

Must-have:

```text
residual_loss_by_role
residual_kl_to_base_by_role
residual_adv_alignment_by_role
residual_logprob_margin_by_role
residual_lora_grad_norm_by_role
lora_delta_norm_by_role
```

Definitions:

```text
residual_logprob_margin = log pi_full(a) - log pi_base(a)
residual_adv_alignment = mean(A_role * residual_logprob_margin)
```

For context gate:

```text
context_gate_mean_by_role
context_gate_std_by_role
context_gate_min_by_role
context_gate_max_by_role
```

For retained context:

```text
retained_context_norm
retained_context_reset_check
retained_context_delta_norm
```

Fail loudly if:

```text
context does not reset on done
role IDs cannot be extracted
residual gradients leak into backbone when they should not
metrics are unavailable but printed as zeros
```

---

## Actor/Critic Loss Detail In Baseline

In `smax_ctm/train_mappo_gru.py`, actor and critic are separate networks with separate train states.

Actor loss:

```text
actor_loss = PPO clipped policy loss - ENT_COEF * entropy
```

Current config:

```text
ENT_COEF = 0.0
```

So entropy is logged but does not affect actor updates.

Critic loss:

```text
critic_loss = VF_COEF * clipped value loss
VF_COEF = 0.5
```

Value loss is not included in actor gradients because the actor and critic do not share parameters.

`total_loss = actor_loss + critic_loss` is only for logging.

---

## Stop / Proceed Criteria

Proceed if:

```text
Role-Residual beats Role-LoRA or sequential_polish
residual_adv_alignment is positive and meaningful
Role-Context Residual beats Role-Residual
retained context beats static context
real context beats shuffled context
global residual does not match role residual
agent residual does not match role residual on randomized maps
```

Stop or narrow claim if:

```text
Role-Residual matches or hurts Role-LoRA
residual_adv_alignment stays near zero
residual KL explodes
context gates collapse to constants
shuffled context performs the same as real context
global residual matches role-context residual
agent-ID residual matches role-context residual on randomized maps
gains are single-seed only
```
