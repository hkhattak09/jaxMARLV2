# Mental model

This file is about how JAX actually executes programs. Use it whenever the user is confused about tracing, staging, compilation, or why JAX behaves differently from NumPy.

## The big idea

JAX is not just “NumPy on accelerators”. It is a system for tracing pure numerical Python functions into a simpler intermediate program, transforming that program, and then lowering it for execution.

The three most important phases to keep separate are:

1. **Tracing / specialisation**
   - Python runs once with tracer objects instead of ordinary arrays.
   - JAX learns the abstract program: shapes, dtypes, control-flow structure, constants, and primitive operations.

2. **Compilation / lowering**
   - The abstract program is lowered to backend IR and compiled.

3. **Execution**
   - Compiled code runs on CPU, GPU, or TPU, often asynchronously with respect to Python.

Confusion comes from mixing these phases.

## What Python sees vs what JAX sees

Python sees:
- objects
- control flow
- lists, dicts, classes
- side-effects
- concrete integers and booleans

JAX sees, during tracing:
- abstract arrays with shape/dtype information
- pure primitive operations
- structured control flow
- pytrees as containers

If Python asks a traced value for something concrete, you get tracer/concretisation errors.

## Static versus dynamic

Treat these as potentially part of the compile signature:

- input shapes
- input dtypes
- sharding
- static arguments
- captured Python objects

Treat these as dynamic runtime values:

- array contents
- per-example data
- values carried through `scan` or `while_loop`

When a value is truly compile-time configuration, make it static or keep it on the Python side. When it is data, keep the logic in JAX space.

## Purity is not optional

The reliable mental model is:

```python
out = f(inputs)
```

not:

```python
f(mutates_globals, consumes_hidden_rng, appends_to_list, logs_everything)
```

Side-effects can appear to work in eager mode, but become misleading or broken under transforms because tracing sees the program structure, not ordinary step-by-step imperative execution.

## Why Python control flow breaks

Python `if`, `while`, and short-circuiting `and`/`or` operate on concrete truth values. Inside `jit`, array-dependent branches need JAX control flow:

- `lax.cond`
- `lax.switch`
- `lax.fori_loop`
- `lax.while_loop`
- `lax.scan`

Elementwise selection often wants `jnp.where`, not a branch.

## Why shapes matter so much

Compilation usually specialises to shapes and dtypes. That means:

- changing shape often means a new compile
- dynamically sized outputs are difficult in compiled JAX
- boolean masking that changes result size is a common anti-pattern
- padded fixed-shape representations are often the right solution

If compile time is exploding, inspect shape variation first.

## Async dispatch

JAX frequently returns control to Python before device execution has completed. That means:

- timing without blocking is misleading
- printing or converting arrays can introduce hidden synchronisation
- host-side inspection changes the program’s performance behaviour

This is why honest benchmarks call `.block_until_ready()` or `jax.block_until_ready(...)`.

## Pytrees are the right structured-data model

A pytree is JAX’s way of handling nested structured inputs and outputs. It is usually the right abstraction for:

- model parameters
- optimiser state
- training state
- nested batches or metadata

Use pytrees instead of custom mutation-heavy object graphs whenever possible.

## Randomness is explicit by design

JAX’s PRNG model is functional:

- a key is an input value
- using the same key twice gives the same result
- new randomness comes from `split` or `fold_in`
- keys should be threaded through the computation

This design supports reproducibility and parallelism, but it punishes hidden global RNG state.

## `jax.Array` and sharding

Modern JAX uses `jax.Array` as the unified array abstraction, including arrays that span multiple devices. For many workflows, you should think in terms of a logical global array and then specify or inspect its sharding.

This leads to three increasingly explicit modes:

- automatic parallelism with `jit`
- explicit sharding in the array type
- fully manual per-device code with `shard_map`

## JAX code review questions

When reviewing any JAX function, ask:

- Which values are static?
- Which values are dynamic?
- Is randomness explicit?
- Where is the compile boundary?
- Are any Python decisions being made on traced values?
- Are there host-device conversions in the hot path?
- Could the program be expressed with fewer, larger compiled regions?

If you can answer these clearly, most JAX bugs become much easier to fix.
