# Sources and maintenance notes

This skill was rebuilt from two inputs:

1. the provided agent-skill authoring guides
2. current JAX documentation plus the provided `jax-main.zip` source snapshot

## JAX topics explicitly refreshed for this version

- installation and platform guidance
- asynchronous dispatch and honest benchmarking
- typed PRNG keys and key-reuse considerations
- control-flow primitives and `scan` / `fori_loop`
- modern sharding APIs and `pmap` migration
- export / serialisation and AOT lowering
- profiling and memory-tooling guidance
- Pallas and advanced extension points

## Maintenance checklist

When updating the skill for a newer JAX release:

1. re-check:
   - `docs/changelog.md`
   - `docs/installation.md`
   - `docs/random-numbers.md`
   - `docs/debugging.md`
   - `docs/benchmarking.md`
   - `docs/sharded-computation.md`
   - `docs/migrate_pmap.md`
   - `docs/export/export.md`
   - `docs/device_memory_profiling.md`

2. revisit:
   - `jax/_src/api.py`
   - `jax/_src/random.py`
   - `jax/_src/debugging.py`
   - `jax/_src/pjit.py`
   - `jax/_src/sharding.py`
   - `jax/_src/pallas/`

3. refresh the eval prompts if terminology or recommended APIs shift

## Notes for future editors

- keep `SKILL.md` focused on workflow and escalation logic
- push deep detail into the reference files
- prefer scripts that produce structured output and avoid interactive prompts
- keep claims about performance and backend behaviour tied to evidence
- treat `pmap`, export, and Pallas guidance as likely to drift over time
