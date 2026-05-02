# Advanced extensions

Use this file only after ordinary JAX design, transforms, and sharding have been exhausted or ruled out.

## Escalation ladder

Prefer the simplest option that solves the real problem:

1. rewrite in plain JAX
2. use a different transform or control-flow primitive
3. add `checkpoint` / donation / explicit sharding
4. use `custom_jvp` or `custom_vjp`
5. use export / AOT lowering
6. use `custom_partitioning`
7. use Pallas
8. use FFI
9. modify JAX internals directly

## `custom_jvp` / `custom_vjp`

Use when:
- default autodiff is mathematically wrong for the desired abstraction
- a stable forward formula and stable backward formula differ
- the primitive is too expensive to differentiate naively

Ask:
- is the derivative truly custom, or does the forward pass just need stabilising?
- can the tangent/cotangent rules be tested independently?
- does the custom rule preserve batching / compilation expectations?

Use templates:
- `assets/custom_vjp_template.py`

## `jax.export` and AOT lowering

Use when:
- tracing/lowering and execution must be decoupled
- you need a serialisable staged computation
- you need shape-polymorphic export
- you need to inspect lowered programs without executing them immediately

Use `assets/export_template.py` as a starter.

## `custom_partitioning`

Use when:
- a function needs specialised sharding-aware lowering
- compiler suggestions are not enough
- the function should participate in distributed compilation with custom rules

This is already advanced. Only recommend it when the sharding semantics are clearly understood.

## Pallas

Use when:
- profiling points to a real kernel-level bottleneck
- plain JAX and XLA optimisations are not enough
- the target backend and kernel semantics are understood well enough

Do **not** recommend Pallas just because code is “slow”. It is the right answer only when:
- the hotspot is real
- the algorithm is stable
- higher-level rewrites have been exhausted
- the user can tolerate backend-specific complexity

Use:
- `assets/pallas_kernel_skeleton.py`

## FFI

Use when:
- a custom kernel or external library is already available
- Pallas and other extension points are insufficient
- integration cost is justified by the performance or functionality gain

FFI is powerful but expensive in complexity, portability, and maintenance.

## JAX internals

Use source-level work only when:
- the user is modifying JAX itself
- the public API is insufficient
- the issue is truly an internal bug or design change

When reasoning about internals, think in terms of:
- primitives
- tracers
- jaxpr
- abstract evaluation
- batching rules
- autodiff rules
- MLIR / XLA lowering

Use `references/REPO-MAP.md` and `scripts/jax_repo_locator.py`.

## Advanced-work checklist

Before proposing an advanced extension, verify:

- the simpler alternatives were considered and rejected for a concrete reason
- the user actually needs the extra power
- the backend and portability trade-offs were stated
- the testing plan is clear
- the performance or correctness case is specific, not hand-wavy
