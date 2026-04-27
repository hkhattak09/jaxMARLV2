# Project Memory: Role-Context Residual MAPPO

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

Current direction:

```text
Role-Context Residual MAPPO
```

Plain Role-LoRA is useful but too weak as a headline. The stronger method should make the adapter a controlled residual learner with a defined objective.

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

## Current Best Framing

Do not headline plain Role-LoRA or full ROSA.

Use:

```text
Role-Context Residual MAPPO
```

Short pitch:

```text
Role-Context Residual MAPPO improves shared recurrent MAPPO by learning controlled role-specific residual policies, trained with role-normalized adapter-only advantage signals and gated by retained team-composition context.
```

Plain Role-LoRA is:

```text
validated base component
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
L_total = L_MAPPO + lambda_residual * L_residual + beta_kl * KL(pi_full || pi_base)
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
L_residual = -E[A_role_i,t * log pi_full(a_i,t)]
```

Important:

```text
For L_residual, shared hidden features and base logits must be stop_gradient.
Only role LoRA and context gate should receive residual-loss gradients.
```

KL:

```text
L_KL = E[KL(pi_full || pi_base)]
```

Purpose:

```text
L_MAPPO learns general cooperation.
L_residual teaches each role adapter how to deviate from the shared policy.
L_KL keeps deviations controlled.
Context gate decides when the residual should matter.
```

---

## Next Recommended Experiments

Immediate next experiment:

```text
Stage 4A: Role-Residual MAPPO
```

Question:

```text
Does giving the role adapter an adapter-only role-normalized advantage objective improve over plain Role-LoRA?
```

Compare:

```text
role_lora
sequential_polish
role_residual
```

Recommended map:

```text
smacv2_10_units
```

Recommended seeds:

```text
0, 1, 2 first
then 42, 43 if promising
```

Stage 4B:

```text
Add static team-context gate.
```

Question:

```text
Does team composition help decide when role residuals should be active?
```

Stage 4C:

```text
Add retained team-context gate inspired by Sable.
```

Question:

```text
Does retained temporal team context beat static context?
```

Ablations:

```text
shuffled_context_residual
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
residual_grad_norm_by_role
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
