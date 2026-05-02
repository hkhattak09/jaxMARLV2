# Transform decision matrix

Use this file when choosing the right primitive or transformation.

## First choose the base representation

Start with a pure function over arrays and pytrees. Do not add transformations until the eager version is conceptually clean.

## Choose the transform by intent

### `jax.jit`

Use when:
- the same computation will run repeatedly
- Python overhead matters
- shapes/dtypes/shardings are reasonably stable

Do not use first when:
- the function still contains hidden side-effects
- the data-dependent control flow has not been rewritten
- the user is still trying to understand basic correctness

### `jax.grad` / `jax.value_and_grad`

Use when:
- the function returns a scalar loss
- reverse-mode is appropriate
- you want end-to-end training-step compilation

Prefer `value_and_grad(..., has_aux=True)` when the forward pass should also return metrics or updated non-differentiated state.

### `jax.vmap`

Use when:
- the computation is the same for many independent examples
- you currently have a Python loop over batch elements
- you want batched Jacobians / per-example gradients

Red flags:
- the body is not truly independent across the batched axis
- memory blows up because batching the whole computation is too large

### `lax.scan`

Use when:
- you have many iterations with a fixed-shape carry
- the loop body is uniform
- compile time or jaxpr size is large because of Python unrolling

Typical wins:
- RNNs
- time stepping
- optimisation loops
- sequential simulation

### `lax.fori_loop`

Use when:
- you want a counted loop primitive
- bounds are known or simple
- you do not need to materialise all intermediate outputs as with `scan`

Important detail:
- static trip counts can lower to `scan`, which improves reverse-mode support
- dynamic trip counts behave more like `while_loop`

### `lax.while_loop`

Use when:
- the number of iterations is data-dependent
- you need loop semantics inside compiled code
- the carry has a fixed pytree structure and fixed leaf shapes/dtypes

### `lax.cond` / `lax.switch`

Use when:
- the branch depends on an array value
- Python branching would try to concretise a tracer

### `jnp.where`

Use when:
- the choice is elementwise
- both branches are arrays with compatible shapes

### `jax.checkpoint` / `jax.remat`

Use when:
- peak memory is the bottleneck
- recomputation is cheaper than storing intermediates

Do not assume this is a free win. Measure wall time and memory after applying it.

### `jit` + `scan`

This is the default solution for “I have a long compiled loop”.
Prefer:
```python
@jax.jit
def run(carry, xs):
    return jax.lax.scan(step, carry, xs)
```
over:
```python
@jax.jit
def run(...):
    for ...:
        ...
```

### `jit` + `value_and_grad`

This is the default solution for “I need a fast training step”.
Put the whole step under one compilation boundary when shapes are stable.

### `jit` + `vmap`

This is the default solution for “same computation over a batch”.
Usually prefer:
```python
fast_batched = jax.jit(jax.vmap(fn))
```
or compile the full caller that contains the `vmap`.

### `shard_map`

Use when:
- you need explicit per-device code
- you need explicit collectives
- automatic / explicit sharding in global-view code is not enough

Think of it as the manual-transmission option. Powerful, but more demanding.

### `pmap`

Use when:
- the codebase already uses it heavily and migration churn would be high
- you need compatibility with existing patterns

For new work, prefer modern sharding APIs unless there is a strong reason not to.

### `jax.export` / AOT lowering

Use when:
- staging and lowering must be separated from execution
- you need a serialisable compiled representation
- you need shape-polymorphic exported programs

### `custom_jvp` / `custom_vjp`

Use when:
- default autodiff is wrong, unstable, or too expensive
- you need to hide a numerically stabilised forward pass behind a custom derivative

## Quick decision rules

- Independent examples? `vmap`
- Long sequential loop with fixed carry? `scan`
- Data-dependent branch? `cond`
- Data-dependent loop count? `while_loop`
- Scalar loss gradient? `value_and_grad`
- Memory issue, compute is cheap? `checkpoint`
- Multi-device, compiler should decide? `jit` with sharding
- Multi-device, you want explicit collectives? `shard_map`
- Need serialisation / offline compile? `export`

## Anti-pattern replacements

- Python loop in `jit` -> `scan` / `fori_loop`
- Python `if` on array -> `cond` / `where`
- Global RNG -> explicit key threading
- Small jitted functions created in a loop -> hoist `jit`
- Huge per-example Python loop -> `vmap`
- `pmap` for new global-view code -> modern sharding APIs first

## Questions to answer before picking a primitive

- Is the iteration count static or dynamic?
- Are the batched computations independent?
- Is the carry shape fixed?
- Does the branch depend on array data?
- Do I need outputs from every step or only the final state?
- Is this about code clarity, compile time, runtime, memory, or distributed semantics?

Getting those right is usually more important than the specific primitive name.
