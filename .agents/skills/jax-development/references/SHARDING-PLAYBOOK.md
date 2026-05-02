# Sharding playbook

Use this file for multi-device, multi-host, `NamedSharding`, `PartitionSpec`, `Mesh`, `shard_map`, or `pmap` migration work.

## Start with the three modes

Think about JAX parallelism in this order:

### 1. Automatic parallelism with `jit`

You write global-view code for one logical array. The compiler chooses a partitioned execution strategy.

Good when:
- you want the simplest high-level design
- the compiler can infer a good strategy
- you do not need explicit collectives

### 2. Explicit sharding

You still write global-view code, but sharding becomes part of the array/type-level story. This is the right level for many serious multi-device programs.

Good when:
- placement matters
- you want predictable data layout
- you want the compiler constrained by your sharding choices

### 3. Manual parallelism with `shard_map`

You write per-device code and explicit collectives.

Good when:
- you need manual SPMD control
- compiler-driven partitioning is not enough
- you need exact collective semantics inside the mapped function

## Default design sequence

1. make single-device code correct
2. state the logical global shape
3. define the mesh
4. define `PartitionSpec`
5. construct `NamedSharding`
6. place inputs explicitly if needed
7. benchmark and inspect resharding
8. only then consider `shard_map`

## Core abstractions

### Mesh

A mesh names device axes. The names matter because they are how you express sharding and collectives.

### `PartitionSpec`

Maps logical array axes onto mesh axes. Mentioning a mesh axis means sharding along that array dimension; omitting it means replication along that mesh axis.

### `NamedSharding`

Combines a mesh and a `PartitionSpec` into an explicit placement object.

## Review questions for sharded code

- What is the logical global array shape?
- Which array axes are sharded?
- Over which mesh axes?
- Which values are replicated?
- Are collectives using the correct axis names?
- Are local vs global semantics explicit?

## Common mistakes

### Accidental resharding

Symptoms:
- unnecessary communication
- unexpected slowdown
- outputs or intermediates moving between layouts

Fix:
- inspect input and output shardings
- make placement explicit
- align adjacent computations on compatible shardings

### Confusing local and global data

In multi-host code, remember:
- local addressable shards are not the full global array
- indexing a global array can trigger unexpected movement or resharding
- when you truly want local data, use local-shard APIs intentionally

### Rank-reduction confusion in `pmap`

Legacy `pmap` habits can lead to wrong assumptions during migration. Modern sharding APIs are more explicit and usually easier to reason about.

## `pmap` migration stance

For new code:
- start with modern sharding APIs
- use `pmap` only if compatibility or migration cost strongly argues for it

For existing code:
- preserve semantics first
- migrate incrementally
- check:
  - implicit mapped axis assumptions
  - collectives
  - local vs global views
  - donation behaviour
  - indexing semantics

## Multi-host checklist

When `jax.process_count() > 1`, make these explicit:

- process-local devices vs global device set
- which data is loaded on which host
- whether a value is local, global, or replicated
- whether all processes execute the same collectives in the same order

## Memory and host offload notes

Sharding interacts with memory. If the task involves host offloading or memory kinds:

- be explicit about placement
- do not hide data movement
- confirm whether the problem is true device memory pressure or accidental replication

## Practical advice

If the user asks for “make it multi-GPU” and the current code is still impure or shape-unstable, do not jump to sharding. Fix the single-device design first. Distributed JAX amplifies weak assumptions.
