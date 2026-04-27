**Role-Context Residual MAPPO**

Standard GRU MAPPO uses one shared recurrent actor for all agents. This is efficient, but in heterogeneous teams it can mix conflicting gradients from different roles. A shared policy may learn general coordination, but the action preference that is good for one unit type may be bad for another.

Plain Role-LoRA helps by adding role-specific adapters, but by itself it is weak as a research idea because it can look like “we attached a small adapter to MAPPO.”

The stronger idea is to make the adapter a **controlled role-specific residual learner**.

The actor is decomposed into:

```text
shared GRU backbone -> base policy
role-specific LoRA adapter -> role residual policy
team-context gate -> decides how much the role residual should matter
```

For agent `i`:

```text
base_logits_i = f_shared(o_i, h_i)

delta_i = LoRA_role_i(h_i)

gate_i = sigmoid(g(h_i, role_i, team_context))

final_logits_i = base_logits_i + gate_i * delta_i
```

Where:

```text
base_logits_i:
    general shared MAPPO policy

delta_i:
    role-specific residual correction

gate_i:
    context-dependent strength of the role residual

team_context:
    simple team-composition signal, e.g. role histogram / own-role count
```

The key idea is:

```text
The adapter should not merely add extra capacity.
It should learn when and how a role should deviate from the shared policy.
```

Training uses two losses.

First, the normal MAPPO PPO loss:

```text
L_MAPPO
```

This trains the full actor normally:

```text
shared backbone + role adapters
```

Second, an auxiliary adapter-only residual loss:

```text
L_residual
```

For this loss, the shared backbone is stopped/frozen. Only the role adapter and context gate receive gradients.

Compute:

```text
pi_base = policy from shared backbone only
pi_full = policy from shared backbone + gated role residual
A_role  = advantage normalized within each role
```

If `A_role` is positive, the adapter should make the taken action more likely than the base policy would.

If `A_role` is negative, the adapter should make the taken action less likely than the base policy would.

So:

```text
L_residual = - A_role * log pi_full(a)
```

with:

```text
stop_gradient(base_logits)
stop_gradient(shared_hidden)
```

This prevents the auxiliary loss from contaminating the shared backbone with role-specific gradients.

Add a small KL regularizer:

```text
L_KL = KL(pi_full || pi_base)
```

so the residual cannot completely override the shared policy.

The total objective is:

```text
L_total = L_MAPPO + λ L_residual + β L_KL
```

Interpretation:

```text
L_MAPPO:
    learns general cooperative behavior

L_residual:
    trains role-specific deviations from the shared policy

L_KL:
    keeps deviations controlled

context gate:
    lets the policy decide when a role-specific deviation is useful
```

This borrows the useful intuition from Sable / Multi-Agent Mamba without replacing the whole model:

```text
agent behavior should depend on team/sequence context, not only local role identity
```

But instead of implementing a full retention or Mamba sequence model, the first version uses a cheap team-context signal:

```text
team role histogram
own role count
possibly ally role counts
```

So the final contribution is:

> Role-Context Residual MAPPO explicitly decomposes shared MAPPO into a general recurrent cooperative policy and a controlled, context-conditioned, role-specific residual policy trained with role-normalized advantage.

**Stages**

```text
Stage 3:
    Role-LoRA MAPPO
    final_logits = base_logits + delta_role

Stage 4A:
    Role-Residual MAPPO
    add adapter-only role-normalized residual advantage loss

Stage 4B:
    Role-Context Residual MAPPO
    add context gate:
    final_logits = base_logits + gate * delta_role

Stage 4C:
    context ablations
    real context vs shuffled context vs no context
```

**Main ablations**

```text
1. GRU MAPPO baseline
2. Stage 3 Role-LoRA MAPPO
3. Role-balanced MAPPO without LoRA
4. Role-Residual MAPPO without context gate
5. Role-Context Residual MAPPO
6. Global-LoRA residual
7. Agent-ID residual
8. Shuffled-context residual
```

**What would count as success**

```text
Role-Residual > Stage 3 Role-LoRA:
    residual advantage objective matters

Role-Context Residual > Role-Residual:
    team context matters

Role-Context Residual > Global-LoRA residual:
    semantic role conditioning matters

Role-Context Residual > Agent-ID residual on randomized maps:
    role identity is better than fixed agent identity

Real context > shuffled context:
    the context gate is using meaningful team information
```

**One-sentence pitch**

Role-Context Residual MAPPO improves shared recurrent MAPPO by learning controlled role-specific residual policies, trained with role-normalized adapter-only advantage signals and gated by team-composition context, so heterogeneous agents can specialize without abandoning parameter sharing.