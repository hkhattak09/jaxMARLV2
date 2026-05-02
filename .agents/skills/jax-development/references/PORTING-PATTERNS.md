# Porting patterns

Use this file when rewriting NumPy, SciPy, TensorFlow, or PyTorch-style code into idiomatic JAX.

## The safe order

1. make it correct in eager `jax.numpy`
2. remove mutation and hidden state
3. make randomness explicit
4. add `jit`
5. add batching or loop primitives
6. add sharding only after single-device correctness is clear

## Pattern: mutation to functional updates

NumPy / PyTorch style:
```python
x[i] += y
```

JAX style:
```python
x = x.at[i].add(y)
```

If the update pattern is large and performance-sensitive, re-think the algorithm rather than translating mutation mechanically.

## Pattern: module state to pytrees

Object-heavy code often wants to become a pytree:

- parameters
- optimiser state
- model buffers
- recurrent carry
- RNG state

Avoid burying arrays inside objects that are mutated in place.

## Pattern: global RNG to explicit keys

Bad:
```python
GLOBAL_KEY = jax.random.key(0)
```

Better:
```python
def step(state, key, batch):
    key, subkey = jax.random.split(key)
    ...
    return new_state, key
```

If code is legacy and expects `PRNGKey`, preserve compatibility, but prefer typed keys in new code.

## Pattern: Python data-dependent branch to JAX control flow

Bad:
```python
@jax.jit
def f(x):
    if x.sum() > 0:
        return x
    return -x
```

Better:
```python
@jax.jit
def f(x):
    return jax.lax.cond(x.sum() > 0, lambda y: y, lambda y: -y, x)
```

For elementwise choice, prefer `jnp.where`.

## Pattern: long Python loop to `scan`

Bad:
```python
@jax.jit
def run(state, xs):
    for x in xs:
        state = step(state, x)
    return state
```

Better:
```python
@jax.jit
def run(state, xs):
    state, ys = jax.lax.scan(step, state, xs)
    return state, ys
```

This often improves compile time dramatically.

## Pattern: per-example Python loop to `vmap`

Bad:
```python
outs = [f(x) for x in batch]
```

Better:
```python
outs = jax.vmap(f)(batch)
```

## Pattern: host logging in the hot path

Bad:
```python
@jax.jit
def step(...):
    print(loss)
```

Better:
- use `jax.debug.print` when debugging
- aggregate metrics and log outside the hot path in production

## Pattern: boolean masking that changes shape

Bad inside `jit`:
```python
x = x[x > 0]
```

Better:
- keep the full tensor and mask with `where`
- or pad to fixed shape and track a validity mask

## Pattern: NumPy helpers inside transformed code

Bad:
```python
np.asarray(x)
float(x)
len(x)
```

Better:
- stay in `jax.numpy`
- use array ops for shape-aware logic
- keep Python-only work outside transforms

## Pattern: training step design

Strong default:
- params and opt state are pytrees
- batch is a pytree
- key is explicit
- one `value_and_grad` call inside one `jit`
- metrics are returned as aux
- donation is considered only after correctness and measurement

## Pattern: shape-polymorphic ambitions

Before trying shape polymorphism or export, first ask:
- can I bucket or pad shapes?
- can I separate compile-time configuration from runtime values?
- is the real issue a compile storm caused by accidental shape churn?

Often a simpler fixed-shape design beats a complicated polymorphic one.

## Porting checklist

Before calling the port “done”, verify:

- no hidden mutation
- no global RNG dependence
- no host round-trips in the hot path
- data-dependent loops/branches use JAX primitives
- tests compare with tolerances rather than exact floating-point equality
- timings exclude first-call compilation unless compile time is part of the question
