# Debugging triage

Use this file for tracer errors, NaNs, shape bugs, mysterious slowdowns, or “works eagerly, fails under `jit`” reports.

## Fast triage order

1. Is the function pure?
2. Is randomness explicit?
3. Is Python control flow touching traced values?
4. Are shapes/dtypes changing between calls?
5. Is there a host boundary in the hot path?
6. Is the failure really a runtime/backend issue instead?

## Minimal debugging toolbox

### `jax.debug.print`

Use when you need runtime values inside staged code.

Pattern:
```python
jax.debug.print("loss={loss}", loss=loss)
```

Use this instead of plain `print` for traced values.

### `checkify`

Use when you need assertions or runtime checks that survive compilation.

Good uses:
- bounds checks
- finiteness checks
- user invariants

Template: `assets/checkify_template.py`

### `jax.make_jaxpr`

Use when you need to see what program JAX is tracing. Helpful for:
- giant unrolled loops
- unexpected primitives
- captured constants or shape logic

### Lowering / compiler IR

Use `scripts/jax_compile_probe.py` when the issue may be:
- compile storm
- unexpected lowering
- huge jaxpr / IR
- sharding or export confusion

### Debug flags

Temporarily consider:
- `jax_disable_jit`
- `jax_debug_nans`
- `jax_debug_infs`
- `jax_debug_key_reuse`

These are debugging tools, not production fixes.

## Error families and likely causes

### Concretisation / tracer conversion

Symptoms:
- `ConcretizationTypeError`
- `TracerBoolConversionError`
- `TracerArrayConversionError`
- `NonConcreteBooleanIndexError`

Think:
- Python asked for a concrete value too early
- boolean masking is changing shape
- a NumPy helper touched a traced value
- a Python branch depends on data

Default fixes:
- `cond`, `switch`, `scan`, `while_loop`, `fori_loop`, `where`
- keep fixed shapes
- move host work out of the transform boundary

### Unexpected tracer leaks

Symptoms:
- values stored in globals or object fields
- `UnexpectedTracerError`
- transforms fail after refactoring stateful code

Think:
- a transformed function is not actually pure
- traced values escaped through mutation or closures

Default fix:
- return values explicitly
- store persistent state as pytrees, not hidden side-effects

### Numerical failures

Symptoms:
- NaNs only after `jit`
- gradients become zero or explode
- eager and jitted outputs diverge too much

Think:
- fused algebra changed evaluation order
- dtype assumptions differ by backend
- masked expressions still evaluate unstable branches
- division, log, exp, norm, or softmax math is unstable

Default fixes:
- stabilise algebra
- inspect dtypes
- use debug flags to catch NaNs early
- compare with tolerances, not exact equality

### PRNG bugs

Symptoms:
- same dropout mask every step
- suspiciously identical random samples
- nondeterminism when code “should” be reproducible

Think:
- key reuse
- hidden global key
- missing `fold_in` for step or process index
- mixing legacy and typed key conventions poorly

Default fixes:
- make key lifetime explicit
- split once per consumer
- return the updated key
- enable key-reuse checking when useful

### Runtime/backend failures

Symptoms:
- `JaxRuntimeError`
- device OOM
- backend-specific crashes
- only fails on multi-device or multi-host

Think:
- wrong installation/runtime pairing
- sharding mismatch
- collective mismatch
- memory pressure
- backend limitation

Default steps:
- run `scripts/jax_env_report.py`
- reduce to CPU or single-device if possible
- make sharding explicit
- profile memory if OOM is involved

## Debugging patterns that often work

### “Disable, isolate, reintroduce”

1. run eagerly
2. add `jit`
3. add `grad`
4. add `vmap`
5. add sharding

The step that breaks tells you what semantics changed.

### “Inspect shape before value”

Many JAX bugs are about shape, dtype, or sharding, not the actual numbers. Check those first.

### “Rewrite, don’t patch around”

If the root cause is:
- Python branch on array
- Python loop in the hot path
- dynamic result shape
- hidden global RNG
- host round-trip

then structural rewrite beats another debug print.

## Useful commands

Environment:
```bash
python3 scripts/jax_env_report.py --format text
```

Project scan:
```bash
python3 scripts/jax_project_scan.py PATH --format text
```

Lowering:
```bash
python3 scripts/jax_compile_probe.py --help
```

## What to include in the final answer

- exact failure class or root cause
- a minimal fixed example
- why the bug appears only under tracing / compilation if relevant
- how to re-test after the fix
