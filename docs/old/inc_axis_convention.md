# INC axis convention (Stage 0)

This note records the actor-batch axis ordering used by SMAX/Hanabi MAPPO scripts, and the reshape rule that INC must follow.

## Source of truth

In smax_ctm/train_mappo_ctm.py, batchify is:

```python
def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))
```

The stack creates shape (num_agents, num_envs, F), then flattening to (num_agents * num_envs, F) preserves C-order row blocks.

## Consequence

The flat actor axis is agent-major:

- rows 0..num_envs-1 are agent 0 across all envs
- rows num_envs..2*num_envs-1 are agent 1 across all envs
- and so on

So the INC reshape from flat sync must be:

```python
synch_per_agent = synch.reshape(num_agents, num_envs, synch_size)
```

Then pool over axis 0 (agent axis). Do not reshape as (num_envs, num_agents, synch_size); that mixes agents silently.

## Round-trip check to add in Stage 1

Use a tagged flat tensor where each row equals row_index // num_envs (agent id), reshape to (num_agents, num_envs, ...), then flatten back and assert exact equality. This catches accidental env-major reshapes.
