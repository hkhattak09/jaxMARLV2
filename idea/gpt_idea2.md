**Idea: RA-MAPPO, Role-Aware MAPPO for Randomized Team Composition**

Standard MAPPO uses one shared actor for all agents and applies one PPO update over all agent-timesteps. This works well on fixed SMAX maps, but randomized SMAX/SMACv2 maps introduce heterogeneous unit roles that change every episode. A shared policy update can therefore be safe on average while being harmful for a particular role.

The key issue is that standard MAPPO mixes all roles into one optimization signal:

```text
global advantage normalization
global PPO loss average
global approximate KL / clip fraction
```

This can cause common or high-advantage-scale roles to dominate the update, while rare or unstable roles receive poor gradients. It can also hide role-specific policy collapse: the average KL may look fine even if one role’s policy changes too aggressively.

RA-MAPPO keeps PPO’s cheap clipped objective. It does **not** use TRPO, Hessians, or conjugate gradients. The change is to make PPO’s update role-aware.

Each agent has a current role id, usually its unit type:

```text
marine, stalker, zealot, hydralisk, ...
```

Then MAPPO’s actor update is modified in three small ways.

**1. Role-Wise Advantage Normalization**

Instead of normalizing advantages over all samples:

```text
A_norm = (A - mean(A_all)) / std(A_all)
```

normalize them within each role:

```text
A_norm_i = (A_i - mean(A_role_i)) / std(A_role_i)
```

This prevents one role’s advantage scale from dominating the shared policy update.

**2. Role-Balanced PPO Loss**

Standard MAPPO averages PPO loss over all agent-timesteps:

```text
L_actor = mean(L_i over all samples)
```

RA-MAPPO first averages within each role, then averages across roles:

```text
L_actor = mean_over_roles(
    mean(L_i for samples with role_i = role)
)
```

This gives each role a fair contribution to the shared actor update, even if some roles appear less often.

**3. Role-Wise Trust-Region Guard**

Instead of relying only on global approximate KL or global clip fraction, RA-MAPPO monitors them per role:

```text
KL_role = mean(KL_i for role_i = role)
clip_frac_role = mean(clip_i for role_i = role)
```

Then add a cheap PPO-style penalty if any role moves too far:

```text
L_total = L_actor
        + beta * mean_over_roles(max(0, KL_role - target_kl)^2)
```

This keeps the spirit of trust-region methods like HAPPO, but without expensive TRPO machinery. The shared policy is not allowed to silently collapse for a rare or unstable role just because the average KL looks acceptable.

Optional extension:

```text
RA-MAPPO + role-conditioned adapters
```

The role-aware objective improves the update signal. The adapters improve specialization. The main algorithmic contribution, however, is the role-aware PPO objective.

**Research Question**

```text
Does standard MAPPO fail under randomized team composition because its shared PPO update mixes heterogeneous roles into one global advantage distribution and one global trust-region signal?
```

**Expected Result**

On fixed SMAX maps, RA-MAPPO may match standard MAPPO because roles are stable and the task is mostly solved.

On randomized SMAX/SMACv2 maps, RA-MAPPO should improve stability and sample efficiency by preventing role-specific undertraining or policy collapse.

**One-Sentence Pitch**

RA-MAPPO keeps MAPPO’s efficient PPO update but makes it role-aware: advantages are normalized per role, policy losses are balanced across roles, and approximate KL is constrained per role so randomized unit types cannot be drowned out by the global shared-policy update.