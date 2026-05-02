# Performance playbook

Use this file for compile-time blowups, slow steady-state execution, hidden synchronisation, memory pressure, or distributed performance work.

## First rule: separate the costs

Never talk about “JAX performance” as one number. Separate:

- host-to-device transfer
- first-call trace and compile
- steady-state device execution
- synchronisation / materialisation
- communication or resharding
- memory pressure / OOM behaviour

Most bad optimisation advice comes from mixing these.

## Honest benchmark pattern

Use `assets/benchmark_template.py` or `scripts/jax_benchmark_harness.py`.

Checklist:
- warm up first
- block before stopping the timer
- report first-call and steady-state separately
- say whether data transfer is inside or outside the timer
- state backend and key shapes/dtypes

## Compile-time problems

Symptoms:
- first call is extremely slow
- every call looks like a first call
- `make_jaxpr` is huge
- CPU-side time dominates

Check:
- changing shape or dtype
- changing sharding
- static arguments changing every call
- creating new lambdas/partials/jitted functions inside loops
- long Python loops inside `jit`
- giant captured constants / closures

Tools:
- `scripts/jax_compile_probe.py`
- `scripts/jax_recompile_explorer.py`

Typical fixes:
- stabilise shapes
- hoist `jit`
- use `scan` or `fori_loop`
- make static args explicit and small
- avoid rebuilding objects each call

## Steady-state execution problems

Symptoms:
- first call is fine, repeated calls are still slow
- GPU/TPU utilisation is poor
- code is fast in theory but not in practice

Check:
- host round-trips
- tiny compiled kernels separated by Python
- poor batching
- accidental replication or resharding
- heavy callbacks or printing
- slow data pipeline starving the device

Typical fixes:
- larger compiled regions
- `vmap` / `scan`
- keep arrays on device
- explicit sharding
- reduce logging / callbacks in the hot path

## Memory problems

Symptoms:
- OOM
- high peak memory
- code only works with tiny batch sizes

Check:
- large intermediates being materialised
- duplication across branches or batches
- unnecessary outputs retained
- no donation where it would help
- sharding causing replication
- activations that could be recomputed

Typical fixes:
- buffer donation
- rematerialisation (`jax.checkpoint`)
- better sharding
- smaller live ranges
- structured loops instead of unrolled Python

Do not suggest donation or remat automatically; tie them to evidence.

## Donation

Donation is useful when:
- an input buffer is dead after the call
- the output can reuse its storage
- memory pressure is real

Donation is not a magic speed-up knob. Use it primarily for memory and only after correctness and API semantics are clear.

## Persistent compilation cache

Consider it when:
- the same program is compiled repeatedly across runs
- compile time is a real user pain point
- the environment is stable enough for cache reuse to matter

This is especially relevant for development loops and repeated workloads, not as a first response to every slowdown.

## Profiling strategy

Use a profiler before deep optimisation when:
- the user wants serious speed work
- compile time is not obviously the whole story
- memory or communication may dominate

Start with:
- trace collection (`assets/profile_template.py`)
- device memory profiling for OOM or leaks
- lowering inspection if compile structure seems wrong

## Performance review questions

Ask these in order:

1. Is the code timing dispatch or actual execution?
2. Is the slow path compile, execute, transfer, or communication?
3. Are shapes/dtypes/shardings stable?
4. Is there a Python loop or callback in the hot path?
5. Is the data already on device?
6. Is sharding aligned with the algorithm?
7. Is the memory footprint forcing a bad design?

## “Fast JAX code” defaults

These defaults are often right:

- compile coarse-grained steps, not every small helper
- batch independent work with `vmap`
- express long loops with `scan`
- keep hot arrays on device
- measure with blocking
- reduce shape churn
- prefer global-view sharding before manual per-device code
- use profiler traces rather than intuition for serious tuning

## What to report back to the user

When performance is discussed, try to report:

- backend and device count
- input shapes and dtypes
- first-call time
- repeated-call summary
- what was inside the timer
- the most likely remaining bottleneck

That level of honesty is more useful than a vague “should be faster”.
