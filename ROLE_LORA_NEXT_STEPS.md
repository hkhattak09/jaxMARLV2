# Role-Context Residual MAPPO Next Steps

This file records the next plan after the Stage 3 Role-LoRA and sequential ROSA experiments.

Current evidence:

- Stage 3, **joint Role-LoRA MAPPO**, is the best current variant.
- Sequential ROSA did not beat Stage 3.
- Sequential polishing, where normal LoRA training remains enabled and a tiny role-local step is added afterward, matched Stage 3 but did not improve it.
- Therefore, sequential role correction should be treated as an ablation, not the main contribution.

The main weakness of plain Role-LoRA is presentation and novelty:

```text
If the whole method is "we attached a LoRA adapter to MAPPO", it is too easy to dismiss.
```

The new plan is to keep Stage 3 as the base component, but add a stronger algorithmic idea:

```text
Role-Context Residual MAPPO
```

The adapter should not merely be extra capacity. It should have a defined job:

```text
learn controlled role-specific deviations from the shared policy,
using role-normalized advantage,
and modulate those deviations by team-composition context.
```

---

## Research Claim

Shared recurrent MAPPO learns one cooperative policy for all agents. This is efficient, but in heterogeneous teams it can entangle two different things:

```text
1. general cooperative behavior that should be shared across roles
2. role-specific action preferences that should differ by unit type
```

Plain Role-LoRA adds specialization capacity, but it does not explicitly say what the adapter should learn.

Role-Context Residual MAPPO makes the decomposition explicit:

```text
shared GRU backbone:
    learns general cooperative behavior

role adapter:
    learns role-specific residual deviations from the shared policy

context gate:
    decides how strongly the role residual should be used under the current team composition
```

Formula for agent `i`:

```text
base_logits_i  = f_shared(o_i, h_i)
delta_i        = LoRA_role_i(h_i)
gate_i         = sigmoid(g(h_i, role_i, team_context))
final_logits_i = base_logits_i + gate_i * delta_i
```

Training objective:

```text
L_total = L_MAPPO + lambda_residual * L_residual + beta_kl * KL(pi_full || pi_base)
```

Where:

```text
L_MAPPO:
    normal clipped PPO loss

L_residual:
    adapter-only role-normalized advantage loss

KL(pi_full || pi_base):
    keeps the residual policy from completely overriding the shared base policy
```

Important implementation rule:

```text
The auxiliary residual loss must not push role-specific gradients back into the shared backbone.
```

For the residual loss, the implementation must either:

```text
1. stop_gradient on shared hidden features and base logits before computing the residual policy
```

or:

```text
2. compute residual gradients separately and zero every non-adapter/non-gate gradient before applying them
```

Do not implement a naive combined loss where `L_residual` freely updates the whole actor.

---

## Non-Negotiables

- Do not modify `smax_ctm/train_mappo_gru.py`.
- Use `smax_ctm/train_rosa_mappo.py` or a copied experiment file for all new work.
- Training will not be run locally by the implementation agent.
- Local work is limited to editing and syntax checks.
- Colab/notebook commands should use full paths from `/content`.
- Follow the error handling policy: **fail loud, never fake**.

Local syntax check:

```bash
python -m py_compile /Users/hassan/repos/new_marl_llm_implementation/smax_ctm/train_rosa_mappo.py
```

Baseline Colab command:

```python
!python /content/jaxMARLV2/smax_ctm/train_mappo_gru.py
```

Role experiment Colab command:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py
```

Every stage must answer a question. Do not build all stages first and only then learn that the hypothesis was wrong.

---

## Required Run Modes

Before adding new methods, make experiments easy to run.

Add CLI overrides to `smax_ctm/train_rosa_mappo.py`.

Minimum arguments:

```text
--map_name
--seed
--total_timesteps
--adapter_mode
--role_lora_rank
--role_lora_scale
--run_name
```

Required `adapter_mode` values:

```text
none
role_lora
global_lora
agent_lora
sequential_polish
role_balanced
role_residual
role_context_residual
global_residual
agent_residual
shuffled_context_residual
```

Fail loudly if an unsupported mode is passed.

At startup, print:

```text
resolved config
adapter_mode
map_name
seed
run_name
all residual/context config values
```

Save the resolved config with the checkpoint.

Decision gate:

- If CLI overrides are silently ignored, stop and fix this first.
- If the printed config does not make the active method obvious, stop and fix logging.

---

## Stage 3: Freeze The Current Base Method

Goal: preserve the current best method as the clean baseline for all future comparisons.

Config:

```python
"USE_ROLE_LORA": True
"USE_SEQUENTIAL_ROLE_UPDATES": False
"ROLE_LORA_RANK": 4
"ROLE_LORA_SCALE": 1.0
"ROLE_ID_SOURCE": "env_state_unit_type"
"LOG_ROLE_DIAGNOSTICS": True
```

Suggested CLI:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name 3m --seed 42 --adapter_mode role_lora --total_timesteps 3000000 --run_name stage3_role_lora_3m_seed42
```

Collect:

```text
win_rate curve
return curve
wall-clock time
lora_delta_norm_by_role
lora_grad_norm_by_role
lora_param_norm_by_role
role_approx_kl
role_clip_frac
role_adv_mean
role_adv_std
role_count
```

Question answered:

```text
What does the strongest simple Role-LoRA baseline look like?
```

Decision gate:

- If Stage 3 is unstable, do not build on it.
- If Stage 3 is stable but only marginally better than baseline, future stages must show clear additional value.

---

## Stage 3B: Capacity And Identity Ablations

Goal: defend against the criticism that Stage 3 only adds parameters.

Run:

```text
1. baseline GRU MAPPO
2. Role-LoRA MAPPO
3. global LoRA MAPPO
4. agent-ID LoRA MAPPO
5. sequential polishing
```

Commands:

```python
!python /content/jaxMARLV2/smax_ctm/train_mappo_gru.py
```

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode role_lora --total_timesteps 3000000 --run_name stage3_role_lora_smacv2_10_seed42
```

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode global_lora --total_timesteps 3000000 --run_name global_lora_smacv2_10_seed42
```

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode agent_lora --total_timesteps 3000000 --run_name agent_lora_smacv2_10_seed42
```

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode sequential_polish --total_timesteps 3000000 --run_name seq_polish_smacv2_10_seed42
```

Question answered:

```text
Is role conditioning doing anything beyond adding adapter capacity?
```

Decision gate:

- If global LoRA matches Role-LoRA, plain role adapters are probably not enough.
- If agent-ID LoRA matches Role-LoRA on randomized maps, semantic roles are not yet proven useful.
- If sequential polishing matches Role-LoRA, keep it as a negative/redundant ablation.

---

## Stage 4A: Role-Residual MAPPO

Goal: test whether an adapter-only role-advantage objective improves over plain Stage 3.

This is the first real post-Stage-3 algorithmic change.

Actor:

```text
base_logits_i  = shared actor logits
delta_i        = role LoRA residual
final_logits_i = base_logits_i + delta_i
```

Training:

```text
L_MAPPO:
    normal PPO on the final policy

L_residual:
    adapter-only role-normalized advantage loss

L_KL:
    small KL from final policy to base policy
```

Residual loss:

```text
A_role = normalize advantage within each role
L_residual = - A_role * log pi_full(a)
```

But for this loss:

```text
shared hidden features are stop_gradient
base logits are stop_gradient
only role LoRA params receive residual gradients
```

Suggested config:

```python
"USE_ROLE_LORA": True
"USE_ROLE_RESIDUAL_LOSS": True
"USE_CONTEXT_GATE": False
"ROLE_BALANCED_PPO": False
"RESIDUAL_LOSS_COEF": 0.1
"RESIDUAL_KL_COEF": 0.01
"RESIDUAL_ADV_NORM": "role"
"RESIDUAL_STOP_BACKBONE": True
"LOG_RESIDUAL_DIAGNOSTICS": True
```

Suggested CLI:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode role_residual --total_timesteps 3000000 --run_name role_residual_smacv2_10_seed42
```

New diagnostics to log:

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

Question answered:

```text
Does giving the adapter a role-specific advantage objective improve over simply attaching Role-LoRA?
```

Decision gate:

- If Role-Residual beats Stage 3, continue to context gating.
- If Role-Residual matches Stage 3 but diagnostics show positive residual alignment, it may still be useful but needs context to matter.
- If Role-Residual hurts performance or residual KL explodes, reduce `RESIDUAL_LOSS_COEF` or increase `RESIDUAL_KL_COEF`.
- If residual gradients leak into the backbone, stop and fix implementation before interpreting results.

---

## Stage 4B: Role-Context Residual MAPPO

Goal: test whether role residuals should depend on team composition.

This borrows the useful intuition from Sable and Multi-Agent Mamba:

```text
agent behavior should depend on agent/team context, not only local role identity
```

Do not implement a full Sable, retention, or Mamba architecture for the first version. Use a cheap context gate.

Actor:

```text
base_logits_i  = f_shared(o_i, h_i)
delta_i        = LoRA_role_i(h_i)
context_i      = team role histogram and own-role count
gate_i         = sigmoid(MLP([stopgrad(h_i), role_onehot_i, context_i]))
final_logits_i = base_logits_i + gate_i * delta_i
```

Recommended first context:

```text
team role histogram
own role count
ally role counts if easy to extract cleanly
```

Avoid using mean teammate hidden states in the first version unless it is explicitly framed as communication. Team-composition context is easier to defend under decentralized execution.

Suggested config:

```python
"USE_ROLE_LORA": True
"USE_ROLE_RESIDUAL_LOSS": True
"USE_CONTEXT_GATE": True
"CONTEXT_SOURCE": "team_role_histogram"
"CONTEXT_SHUFFLE": False
"CONTEXT_GATE_HIDDEN_DIM": 32
"RESIDUAL_LOSS_COEF": 0.1
"RESIDUAL_KL_COEF": 0.01
"RESIDUAL_ADV_NORM": "role"
"RESIDUAL_STOP_BACKBONE": True
"LOG_RESIDUAL_DIAGNOSTICS": True
```

Suggested CLI:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode role_context_residual --total_timesteps 3000000 --run_name role_context_residual_smacv2_10_seed42
```

New diagnostics to log:

```text
context_gate_mean_by_role
context_gate_std_by_role
context_gate_min_by_role
context_gate_max_by_role
residual_kl_to_base_by_role
residual_adv_alignment_by_role
role_context_histogram_mean
```

Question answered:

```text
Does team composition/context help decide when role-specific residuals should be active?
```

Decision gate:

- If Role-Context Residual beats Role-Residual, context is useful.
- If gate values collapse to all zeros, residual is being ignored.
- If gate values saturate to all ones, the context gate is not doing useful modulation.
- If context improves random-team maps but not fixed maps, that supports the generalization story.

---

## Stage 4C: Context And Residual Ablations

Goal: prove the context and residual mechanisms are doing real work.

Run:

```text
1. Stage 3 Role-LoRA
2. Role-Residual
3. Role-Context Residual
4. shuffled-context residual
5. global residual
6. agent-ID residual
7. role-balanced PPO without residual
```

Suggested commands:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode shuffled_context_residual --total_timesteps 3000000 --run_name shuffled_context_residual_smacv2_10_seed42
```

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode global_residual --total_timesteps 3000000 --run_name global_residual_smacv2_10_seed42
```

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode agent_residual --total_timesteps 3000000 --run_name agent_residual_smacv2_10_seed42
```

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 42 --adapter_mode role_balanced --total_timesteps 3000000 --run_name role_balanced_smacv2_10_seed42
```

Question answered:

```text
Is the gain caused by role residual learning, meaningful context, semantic role conditioning, or just extra parameters?
```

Decision gate:

- If Role-Residual beats Stage 3, the residual objective matters.
- If Role-Context Residual beats Role-Residual, context matters.
- If real context beats shuffled context, the context signal is meaningful.
- If global residual matches Role-Context Residual, role conditioning is not the cause.
- If agent-ID residual matches Role-Context Residual on randomized maps, semantic role identity is not proven.
- If role-balanced PPO alone matches the residual methods, the core improvement may be role-balanced optimization rather than adapters.

---

## Stage 5: Generalization Stress Tests

Goal: make the project about robustness under heterogeneous or randomized team composition.

Priority maps:

```text
smacv2_10_units
2s3z
3s5z
6h_vs_8z
```

Most important:

```text
smacv2_10_units
```

This is the primary randomized-unit benchmark for the project. It is cheap enough to run, and existing baselines are already available, so do not spend the main ablation budget on `smacv2_5_units`.

First multi-seed set:

```text
seeds: 0, 1, 2
methods: Stage 3 Role-LoRA, Role-Residual, Role-Context Residual, global residual
timesteps: 3e6
```

Suggested command pattern:

```python
!python /content/jaxMARLV2/smax_ctm/train_rosa_mappo.py --map_name smacv2_10_units --seed 0 --adapter_mode role_context_residual --total_timesteps 3000000 --run_name role_context_residual_smacv2_10_seed0
```

Question answered:

```text
Does the method improve robustness on randomized heterogeneous teams, not just one fixed map?
```

Decision gate:

- If gains appear on `smacv2_10_units` across multiple seeds, the randomized heterogeneous-team story is plausible.
- If gains appear only on `3m`, do not claim heterogeneous generalization.
- If gains are seed-fragile, reduce the claim to exploratory evidence.

---

## Stage 6: Optional Composition-Shift Evaluation

Goal: test whether role/context factorization survives distribution shift.

Only do this after Stages 4A-5 show promise.

Possible design:

```text
Train distribution:
    unit types sampled from a restricted or balanced distribution

Eval distribution:
    held-out, skewed, or rare-role-heavy team compositions
```

Implementation warning:

- Do not fake this by changing labels.
- The actual environment unit-type sampling must change.
- If clean sampling control cannot be implemented quickly, skip this stage and say so.

Question answered:

```text
Does role-context residual learning help when team composition at evaluation differs from training?
```

Decision gate:

- If Role-Context Residual beats Stage 3 and global residual under composition shift, this is the strongest result.
- If all methods fail under shift, report the failure and keep the main claim narrower.

---

## Minimum Meeting Package

Aim to have:

```text
1. Clear statement of the failure of sequential ROSA:
   HAPPO-style sequential correction did not improve over joint Role-LoRA.

2. Clear statement of the new idea:
   adapters are trained as role-specific residual advantage learners, not merely added as capacity.

3. One architecture diagram:
   shared GRU -> base logits
   role LoRA -> residual logits
   context gate -> residual strength
   final logits = base + gate * residual

4. One plot:
   Stage 3 Role-LoRA vs Role-Residual vs Role-Context Residual.

5. One ablation table:
   Stage 3, global residual, agent residual, shuffled context, role-balanced PPO.

6. One diagnostic table:
   residual_adv_alignment_by_role
   residual_kl_to_base_by_role
   context_gate_mean_by_role
```

Minimum acceptable conclusion:

```text
Sequential role correction was not useful in our shared-backbone MAPPO setting.
The next useful direction is to treat role adapters as controlled residual advantage learners.
```

Strong conclusion, if experiments support it:

```text
Role-Context Residual MAPPO improves shared recurrent MAPPO in randomized heterogeneous SMAX settings by separating shared cooperative behavior from context-conditioned role-specific residual behavior.
```

---

## Stop Conditions

Stop pursuing the post-Stage-3 direction if:

```text
Role-Residual does not beat or meaningfully diagnose Stage 3
residual advantage alignment stays near zero
residual KL to base explodes or collapses to zero with no performance gain
context gates collapse to constant values
shuffled context performs the same as real context
global residual consistently matches Role-Context Residual
agent-ID residual matches Role-Context Residual on randomized maps
the gains appear on one seed only
```

Proceed if:

```text
Role-Residual beats Stage 3 or gives clearly positive residual alignment
Role-Context Residual beats Role-Residual
real context beats shuffled context
Role-Context Residual beats global residual
gains appear on smacv2_10_units across multiple seeds
```

---

## Current Best Framing

Working name:

```text
Role-Context Residual MAPPO
```

Short pitch:

```text
Role-Context Residual MAPPO improves shared recurrent MAPPO by learning controlled role-specific residual policies, trained with role-normalized adapter-only advantage signals and gated by team-composition context.
```

Avoid making plain Role-LoRA or sequential ROSA the headline. Plain Role-LoRA is the base component. Sequential ROSA is a negative ablation. The post-Stage-3 research bet is residual advantage learning plus context-dependent role specialization.
