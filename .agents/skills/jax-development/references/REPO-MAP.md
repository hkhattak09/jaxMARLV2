# Repository map

This file assumes a local checkout similar to the provided `jax-main.zip` snapshot.

## High-value documentation paths

Start with docs before implementation.

- `docs/installation.md`
- `docs/benchmarking.md`
- `docs/async_dispatch.rst`
- `docs/control-flow.md`
- `docs/random-numbers.md`
- `docs/debugging.md`
- `docs/debugging/checkify_guide.md`
- `docs/debugging/print_breakpoint.md`
- `docs/device_memory_profiling.md`
- `docs/profiling.md`
- `docs/persistent_compilation_cache.md`
- `docs/buffer_donation.md`
- `docs/sharded-computation.md`
- `docs/notebooks/explicit-sharding.md`
- `docs/migrate_pmap.md`
- `docs/aot.md`
- `docs/export/export.md`
- `docs/export/shape_poly.md`
- `docs/gpu_performance_tips.md`
- `docs/faq.rst`
- `docs/changelog.md`
- `docs/notebooks/shard_map.md`
- `docs/pallas/` and `docs/jax.experimental.pallas*.rst`

## High-value source modules

These are the first places to inspect in the Python sources:

- `jax/_src/api.py`
- `jax/_src/api_util.py`
- `jax/_src/array.py`
- `jax/_src/random.py`
- `jax/_src/debugging.py`
- `jax/_src/checkify.py`
- `jax/_src/errors.py`
- `jax/_src/custom_derivatives.py`
- `jax/_src/ad_checkpoint.py`
- `jax/_src/stages.py`
- `jax/_src/export/`
- `jax/_src/pjit.py`
- `jax/_src/sharding.py`
- `jax/_src/sharding_impls.py`
- `jax/_src/mesh.py`
- `jax/_src/mesh_utils.py`
- `jax/_src/pallas/`

## Tests worth checking early

Tests are often the fastest truth source for current behaviour.

- `tests/api_test.py`
- `tests/errors_test.py`
- `tests/lax_control_flow_test.py`
- `tests/random_test.py`
- `tests/checkify_test.py`
- `tests/debugging_primitives_test.py`
- `tests/pjit_test.py`
- `tests/pmap_test.py`
- `tests/shard_map_test.py`
- `tests/profiler_test.py`
- `tests/export_test.py`

## Search recipes

Control-flow and tracer errors:
```bash
rg "ConcretizationTypeError|TracerBoolConversionError|NonConcreteBooleanIndexError" docs tests jax
```

Randomness:
```bash
rg "random.key|PRNGKey|key reuse|fold_in|split" docs tests jax
```

Sharding and migration:
```bash
rg "NamedSharding|PartitionSpec|Mesh|shard_map|pmap|migrate_pmap" docs tests jax
```

Debugging and profiling:
```bash
rg "debug.print|checkify|block_until_ready|profiler|compiler_ir" docs tests jax
```

Custom autodiff / export:
```bash
rg "custom_vjp|custom_jvp|export|ShapeDtypeStruct|shape polymorphism" docs tests jax
```

Pallas:
```bash
rg "pallas|mosaic|triton" docs tests jax
```

## Recommended protocol for repo questions

1. read the relevant docs page
2. inspect the nearest `_src` implementation
3. confirm with a nearby test
4. only then make a behaviour claim

If docs and code look out of sync, prefer current tests plus changelog or migration docs.
