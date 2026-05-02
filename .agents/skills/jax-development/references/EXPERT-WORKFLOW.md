# Expert workflow

This file is the control tower for the whole skill. Use it when you need to decide what to do first, what evidence to gather, and what a strong JAX answer should contain.

## The five-step loop

### 1. Model the program before touching the code

Ask these questions immediately, even if only implicitly:

- What are the true inputs and outputs?
- What state is being threaded through the computation?
- Which values are arrays, and which are Python configuration?
- Which axes are batch, time, feature, or device axes?
- Where does randomness enter, and how long should it live?
- Where is the intended compilation boundary?
- Is the user describing a compile problem, a runtime problem, or a scaling problem?

If you cannot answer those, do not start “optimising” yet.

### 2. Make a tiny reproducible slice

For JAX, a tiny reproducer should expose:

- explicit shapes and dtypes
- explicit key handling
- a single transformation boundary when possible
- the smallest loop or branch that still fails
- backend assumptions if they matter

Use `assets/mre_template.py` if you need to create one quickly.

### 3. Gather the cheapest useful evidence

Use this ladder. Climb only as high as needed.

1. **Static reasoning**
   - Does the code violate JAX rules about purity, control flow, or host/device boundaries?

2. **Environment evidence**
   - `python3 scripts/jax_env_report.py --format json`

3. **Static project scan**
   - `python3 scripts/jax_project_scan.py PATH --format json`

4. **Lowering evidence**
   - `python3 scripts/jax_compile_probe.py ...`

5. **Timing evidence**
   - `python3 scripts/jax_benchmark_harness.py ...`

6. **Cross-case compile evidence**
   - `python3 scripts/jax_recompile_explorer.py ...`

7. **Profiler / memory evidence**
   - use the profile and memory templates from `assets/`

Do not jump straight to “use Pallas” or “donate buffers” unless the evidence points there.

### 4. Apply the minimum structural fix

Prefer the smallest change that resolves the real issue.

Examples:
- tracer bool error -> replace data-dependent Python `if` with `lax.cond` or `jnp.where`
- compile storm -> stabilise shapes / static args / closure capture before touching kernels
- long Python loop -> `scan` or `fori_loop`
- repeated RNG -> explicit key plumbing
- slow multi-device code -> make sharding explicit before rewriting the algorithm

### 5. Prove the fix

A strong answer does not stop at “here is the patch”. It explains how to verify:

- what should now compile once instead of many times
- which timings should be compared
- which values or shapes should stay constant
- where to look in profiler traces
- which caveats still depend on backend or scale

## Default response shape

Unless the user asked for something very different, structure the answer like this:

1. **Root cause or recommendation**
2. **Runnable code / patch**
3. **Why it works**
4. **How to verify**
5. **Risks or caveats**

## Common lane-specific workflows

### Porting or refactoring

1. Write pure eager `jax.numpy`.
2. Remove mutation and hidden state.
3. Thread keys explicitly.
4. Add `jit`.
5. Add batching (`vmap`) or loop primitives (`scan`) if needed.
6. Only then consider sharding.

### Debugging

1. Reproduce outside the full training loop.
2. Identify whether the failure is:
   - tracing
   - control flow
   - shape/dtype
   - randomness
   - side-effects
   - sharding / runtime
   - numerics
3. Use the appropriate debug tool:
   - `jax.debug.print`
   - `checkify`
   - `make_jaxpr`
   - lowering inspection
   - debug flags
4. Rewrite the offending structure rather than piling on print statements.

### Performance

1. Confirm there is a real bottleneck.
2. Separate:
   - transfer
   - first-call compile
   - steady-state execution
   - synchronisation / materialisation
3. Look for:
   - compile storms
   - Python loops
   - host round trips
   - unintentional replication
   - missing donation
   - poor sharding
4. Benchmark and profile before any major redesign.

### Sharding or distributed work

1. Make the single-device version correct.
2. Express the logical global array shape.
3. Choose sharding mode:
   - automatic via `jit`
   - explicit sharding
   - manual `shard_map`
4. Check local versus global semantics in multi-host code.
5. Only drop to manual collectives when global-view code is insufficient.

## What “expert JAX style” usually means

- pure functions
- explicit pytrees
- explicit randomness
- shapes and dtypes treated as first-class design constraints
- minimal host interaction
- loop and branch primitives chosen intentionally
- performance claims backed by timing or IR evidence
- distributed semantics stated clearly rather than implied

## When to say “don’t do this in JAX”

Be willing to push back when the design is a bad match:

- extremely irregular variable-length outputs in the hot path
- heavy Python object mutation at every step
- host callbacks every iteration
- tiny scalar-heavy logic where compilation cost dominates
- requests for low-level accelerator tuning before basic JAX issues are fixed

In those cases, offer a hybrid design instead of blindly forcing everything through `jit`.
