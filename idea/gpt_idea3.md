**ROSA-MAPPO: Role-Ordered Sequential Adapter MAPPO**

Standard MAPPO uses one shared actor for all agents and updates that actor with one simultaneous PPO objective. This works well on many fixed SMAX maps, but it becomes less natural when agents are heterogeneous or when unit roles change across episodes, as in SMAX/SMACv2-style randomized team composition.

The issue is that a fully shared policy has no clean way to specialize by role, while fully separate policies are expensive and data-inefficient. Existing LoRA/LoRASA-style methods solve part of this by adding small low-rank adapters, but they usually specialize by fixed agent identity:

```text
agent_0 -> adapter_0
agent_1 -> adapter_1
agent_2 -> adapter_2
```

That is fragile when roles are randomized, because `agent_0` may be a marine in one episode, a stalker in another, and a hydralisk in another.

ROSA-MAPPO instead specializes by **role**, not by agent index.

Each role or unit type receives a lightweight LoRA adapter:

```text
marine    -> adapter_marine
stalker   -> adapter_stalker
zealot    -> adapter_zealot
hydralisk -> adapter_hydralisk
```

The actor becomes:

```text
observation -> shared encoder -> shared GRU hidden h
role_id / unit_type -> role-specific LoRA adapter
shared policy output + role adapter correction -> action distribution
```

For a policy layer:

```text
y = W_shared h + ΔW_role h
```

where:

```text
ΔW_role = B_role A_role
```

and `A_role`, `B_role` are small low-rank matrices.

The shared GRU learns general cooperative behavior. The role adapter learns role-specific deviations.

The algorithmic part is that these role adapters are not updated all at once. They are updated **sequentially by role**, inspired by HAPPO.

During each PPO update, ROSA-MAPPO samples a random role order:

```text
[zealot, stalker, marine, hydralisk]
```

Then it updates one role adapter at a time.

For the current role, the shared backbone is frozen and only that role’s adapter is updated. Earlier role adapters in the sequence may already have changed. Therefore, the current role’s advantage is conditioned on the policy-ratio corrections of previously updated roles:

```text
A_conditioned = correction_previous_roles * A
```

where:

```text
correction_previous_roles =
    product of pi_new(a | o) / pi_old(a | o)
    for samples belonging to roles updated earlier
```

In practice, this correction can be clipped for stability:

```text
correction_previous_roles =
    clip(correction_previous_roles, 1 - c, 1 + c)
```

The current role is then updated with the normal PPO clipped objective:

```text
ratio_current = pi_new_current / pi_old_current

L_role = min(
    ratio_current * A_conditioned,
    clip(ratio_current, 1 - eps, 1 + eps) * A_conditioned
)
```

This keeps the method PPO-compatible. There is no Hessian, no TRPO, and no conjugate-gradient optimization.

Training can be structured as:

```text
1. Shared backbone update:
   standard MAPPO update over all agents

2. Role adapter update:
   sequential HAPPO-style PPO update over role-specific LoRA adapters
```

or, initially:

```text
train shared MAPPO normally for warmup
then enable role-specific sequential adapter updates
```

**Why This Is Interesting**

ROSA-MAPPO combines three useful ideas:

```text
MAPPO:
efficient shared recurrent policy

LoRA/LoRASA:
cheap specialization without fully separate policies

HAPPO:
sequential policy updates that account for previous policy changes
```

But it applies them at the role level:

```text
not fully shared
not fully separate
not agent-index-specific
but role-specialized and sequentially updated
```

The research question is:

```text
Can HAPPO-style sequential improvement be recovered inside a mostly shared MAPPO policy by applying it only to lightweight role-specific adapters?
```

This is especially relevant when roles are heterogeneous or randomized.

**Expected Results**

On fixed SMAX maps:

```text
standard GRU MAPPO may already solve the task
ROSA-MAPPO should match it and possibly improve sample efficiency
```

On heterogeneous or randomized maps:

```text
role-specific adapters should improve specialization
sequential role updates should improve stability
random role order should avoid hardcoded domain assumptions
```

**Baselines**

```text
1. GRU MAPPO
2. GRU MAPPO + normal role LoRA adapters
3. GRU MAPPO + agent-index LoRA adapters
4. ROSA-MAPPO: role LoRA + sequential role updates
```

**Footnote: Future Ordering Strategies**

For the first version, use **random role ordering**, because random ordering is already a reasonable and proven choice in HAPPO-like sequential update methods.

Later, test more structured role orders:

```text
rare-role-first:
update underrepresented roles before common roles

KL-risk order:
use per-role KL or clip fraction to prioritize unstable roles

influence-first:
update roles that most affect other agents first,
using proxies such as visibility, proximity, advantage impact,
damage dealt/taken, or learned interaction scores

fixed tactical order:
frontline/melee roles first, ranged roles second,
support roles last
```

**One-Sentence Pitch**

ROSA-MAPPO keeps MAPPO’s shared GRU backbone, adds lightweight role-specific LoRA adapters, and updates those adapters sequentially with HAPPO-style conditioned PPO advantages, giving heterogeneous agents stable role specialization without abandoning PPO efficiency.