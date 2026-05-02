# Code review rubric

Use this file before finalising an answer. It helps the agent self-review JAX code the way an experienced maintainer would.

## Must-fix categories

### 1. Purity and state

Check:
- any hidden global state?
- any in-place mutation that should be a functional update?
- any traced values leaking into globals or object fields?

If yes, fix before anything else.

### 2. Randomness

Check:
- is the key explicit?
- is it split exactly where needed?
- is the updated key returned when the state continues?
- is the same key accidentally reused?

If key handling is muddy, the code is not production-ready.

### 3. Static vs dynamic boundary

Check:
- are Python decisions being made on traced values?
- are array shapes or sizes created from runtime data?
- are large Python objects or lambdas changing across calls?

If yes, expect tracer errors or compile storms.

### 4. Host-device boundary

Check:
- `np.asarray`
- `.item()`
- `.tolist()`
- `device_get`
- frequent printing or callbacks
- Python control flow around array values

If any of these sit in the claimed hot path, call them out.

## Performance review categories

### 5. Compilation structure

Check:
- is `jit` hoisted out of loops?
- are many tiny compiled regions being created?
- is the whole step compiled instead of many subpieces when that makes sense?
- are shapes, dtypes, and static args stable across calls?

### 6. Loop and batch structure

Check:
- Python loop that should be `scan`?
- Python per-example loop that should be `vmap`?
- huge unrolled jaxpr likely causing compile blowup?

### 7. Memory and numerics

Check:
- dtype expectations explicit?
- unnecessary materialisation?
- remat or donation worth considering?
- stable algebra around divisions, logs, exps, masks, and softmax-like code?

### 8. Benchmark honesty

Check:
- warm-up?
- blocking?
- compile time separated from steady-state?
- host transfer excluded or included intentionally?
- device/backend stated?

No benchmark claim should survive without these.

## Distributed review categories

### 9. Sharding semantics

Check:
- what is the logical global shape?
- which axes are sharded and over what mesh axes?
- which values are replicated?
- are collectives aligned with the right axis names?
- are local vs global semantics explicit?

### 10. Legacy API choices

Check:
- is `pmap` being used because it is genuinely appropriate, or just because the code is old?
- would modern sharding APIs clarify the logic?

## Response-quality categories

### 11. Explanation quality

A strong answer explains:
- what was wrong
- why the fix works under tracing / compilation / sharding semantics
- what to measure or test next

### 12. Verification quality

A strong answer gives:
- a small correctness check
- a timing / profiler plan when performance is discussed
- backend caveats if behaviour may differ on GPU/TPU

## Rapid scorecard

For each category, grade the draft answer mentally as:

- **green**: solid
- **amber**: acceptable but mention caveat
- **red**: fix before sending

If purity, randomness, static/dynamic boundaries, or benchmark honesty are red, the answer should not be sent yet.
