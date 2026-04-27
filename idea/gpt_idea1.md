**Idea: Role-Conditioned Specialization for MAPPO Under Randomized Team Composition**

Standard MAPPO uses a shared policy for all agents. This is efficient, but it assumes that all agents can be handled well by the same actor. In fixed SMAX maps, this often works because each agent slot has a stable meaning: for example, `agent_0` may always be a stalker and `agent_3` may always be a zealot. The policy can exploit this fixed slot-to-role structure.

However, in randomized SMAX/SMACv2-style maps, unit types change every episode. The same agent slot may be a marine in one episode, a stalker in another, and a hydralisk in another. This breaks fixed agent identity as a useful specialization signal.

Existing agent-specific adaptation methods, such as LoRASA-style low-rank adapters, specialize by agent identity:

```text
agent_0 -> adapter_0
agent_1 -> adapter_1
agent_2 -> adapter_2
```

This is useful when agent identity is stable, but it is a poor inductive bias when roles are randomized. `adapter_0` cannot cleanly specialize if `agent_0` changes unit type across episodes.

We propose **role-conditioned adapters** for MAPPO. Instead of giving each agent slot its own adapter, we give each unit role/type its own lightweight adapter:

```text
marine    -> adapter_marine
stalker   -> adapter_stalker
zealot    -> adapter_zealot
hydralisk -> adapter_hydralisk
```

At each timestep, an agent selects or generates its adapter from its current role id/unit type. The shared GRU policy learns general cooperative combat behavior, while the role adapter learns small role-specific deviations.

The actor becomes:

```text
observation -> shared encoder -> GRU hidden h
unit_type / role_id -> role adapter
h + adapter_role(h) -> action logits
```

Or, in low-rank form:

```text
W_role = W_shared + ΔW_role
ΔW_role = B_role A_role
```

where `ΔW_role` is a small role-specific low-rank update.

The central research question is:

```text
In cooperative MARL with randomized team composition, should specialization be tied to fixed agent identity or to the agent’s current role?
```

Expected result:

```text
On fixed SMAX maps:
agent-specific adapters and role-conditioned adapters may both work.

On randomized SMAX/SMACv2 maps:
role-conditioned adapters should generalize better because specialization follows the unit type rather than the arbitrary agent slot.
```

This directly targets the observed weakness of GRU MAPPO: it solves most fixed SMAX maps but struggles when unit roles are randomized. The method preserves MAPPO’s shared-policy efficiency while adding dynamic, role-aware specialization.