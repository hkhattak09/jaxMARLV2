\
#!/usr/bin/env python3
"""Benchmark a Python callable with optional JAX JIT and proper blocking."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def load_module(module_name: str | None, file_path: str | None) -> Any:
    if bool(module_name) == bool(file_path):
        raise ValueError("Exactly one of --module or --file is required.")
    if module_name:
        return importlib.import_module(module_name)

    path = Path(file_path or "")
    if not path.exists():
        raise FileNotFoundError(f"Module file not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_attr(obj: Any, dotted: str) -> Any:
    current = obj
    for part in dotted.split("."):
        current = getattr(current, part)
    return current


def load_json_arg(raw: str | None, file_path: str | None, default: Any) -> Any:
    if raw is not None and file_path is not None:
        raise ValueError("Choose either the inline JSON form or the file form, not both.")
    if raw is not None:
        return json.loads(raw)
    if file_path is not None:
        return json.loads(Path(file_path).read_text(encoding="utf-8"))
    return default


def numeric_tree(value: Any) -> bool:
    if isinstance(value, (int, float, bool)):
        return True
    if isinstance(value, list):
        return all(numeric_tree(v) for v in value)
    return False


def maybe_import_jax():
    try:
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        return jax, jnp
    except Exception:
        return None, None


def tree_arrayify(value: Any, jnp_module: Any) -> Any:
    if isinstance(value, dict):
        return {k: tree_arrayify(v, jnp_module) for k, v in value.items()}
    if isinstance(value, list):
        if numeric_tree(value):
            return jnp_module.array(value)
        return [tree_arrayify(v, jnp_module) for v in value]
    return value


def tree_device_put(value: Any, jax_module: Any) -> Any:
    if isinstance(value, dict):
        return {k: tree_device_put(v, jax_module) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_device_put(v, jax_module) for v in value]
    try:
        return jax_module.device_put(value)
    except Exception:
        return value


def parse_int_tuple(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or raw == "":
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def block_until_ready(value: Any, jax_module: Any | None) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    if jax_module is not None:
        try:
            return jax_module.block_until_ready(value)
        except Exception:
            pass
    return value


def run_once(fn: Any, args: list[Any], kwargs: dict[str, Any], jax_module: Any | None) -> Any:
    out = fn(*args, **kwargs)
    block_until_ready(out, jax_module)
    return out


def summary(times_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "stdev_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def benchmark(fn: Any, args: list[Any], kwargs: dict[str, Any], repeat: int, jax_module: Any | None) -> list[float]:
    times_ms = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        run_once(fn, args, kwargs, jax_module)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    return times_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a Python callable with optional JAX JIT and proper blocking.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Exit codes:
  0 success
  2 operational error

Examples:
  python3 scripts/jax_benchmark_harness.py --file evals/files/naive_benchmark.py --function matmul_step \\
      --args-json '[[[1.0, 2.0], [3.0, 4.0]], [[1.0], [2.0]]]' --arrayify --jit --compare-eager

  python3 scripts/jax_benchmark_harness.py --module mypkg.train --function step \\
      --args-file args.json --kwargs-file kwargs.json --jit --repeat 20

  python3 scripts/jax_benchmark_harness.py --file train.py --function step \\
      --args-file args.json --jit --static-argnums 2 --donate-argnums 0,1
""",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--module", help="Import path for the module containing the callable.")
    source.add_argument("--file", help="Path to a Python file containing the callable.")
    parser.add_argument("--function", required=True, help="Callable name or dotted attribute path.")
    parser.add_argument("--args-json", help="JSON list of positional arguments.")
    parser.add_argument("--args-file", help="Path to a JSON file containing positional arguments.")
    parser.add_argument("--kwargs-json", help="JSON object of keyword arguments.")
    parser.add_argument("--kwargs-file", help="Path to a JSON file containing keyword arguments.")
    parser.add_argument("--arrayify", action="store_true", help="Convert numeric JSON lists to `jax.numpy.array` when JAX is available.")
    parser.add_argument("--device-put", action="store_true", help="Apply `jax.device_put` to arguments before timing when JAX is available.")
    parser.add_argument("--jit", action="store_true", help="Wrap the callable with `jax.jit`.")
    parser.add_argument("--static-argnums", help="Comma-separated positional indices to mark static when using --jit.")
    parser.add_argument("--donate-argnums", help="Comma-separated positional indices to donate when using --jit.")
    parser.add_argument("--compare-eager", action="store_true", help="Also benchmark the original eager callable.")
    parser.add_argument("--repeat", type=int, default=10, help="Number of timed steady-state repetitions. Default: 10")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up calls before timed loops. Default: 1")
    parser.add_argument("--format", choices=("json", "text"), default="json", help="Output format. Default: json")
    parser.add_argument("--output", help="Write the report to this file instead of stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        module = load_module(args.module, args.file)
        fn = resolve_attr(module, args.function)
        if not callable(fn):
            raise TypeError(f"Resolved object is not callable: {args.function}")

        raw_args = load_json_arg(args.args_json, args.args_file, [])
        raw_kwargs = load_json_arg(args.kwargs_json, args.kwargs_file, {})
        if not isinstance(raw_args, list):
            raise TypeError("Positional arguments JSON must decode to a list.")
        if not isinstance(raw_kwargs, dict):
            raise TypeError("Keyword arguments JSON must decode to an object.")

        jax, jnp = maybe_import_jax()

        proc_args = list(raw_args)
        proc_kwargs = dict(raw_kwargs)
        if args.arrayify:
            if jax is None or jnp is None:
                raise RuntimeError("--arrayify requires JAX to be importable.")
            proc_args = [tree_arrayify(v, jnp) for v in proc_args]
            proc_kwargs = {k: tree_arrayify(v, jnp) for k, v in proc_kwargs.items()}

        if args.device_put:
            if jax is None:
                raise RuntimeError("--device-put requires JAX to be importable.")
            proc_args = [tree_device_put(v, jax) for v in proc_args]
            proc_kwargs = {k: tree_device_put(v, jax) for k, v in proc_kwargs.items()}

        static_argnums = parse_int_tuple(args.static_argnums)
        donate_argnums = parse_int_tuple(args.donate_argnums)

        report: dict[str, Any] = {
            "source": args.module or args.file,
            "callable": args.function,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "jax_available": jax is not None,
            "jit_requested": args.jit,
            "compare_eager": args.compare_eager,
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums,
        }

        if args.compare_eager or not args.jit:
            for _ in range(args.warmup):
                run_once(fn, proc_args, proc_kwargs, jax)
            eager_times = benchmark(fn, proc_args, proc_kwargs, args.repeat, jax)
            report["eager"] = {
                "times_ms": eager_times,
                "summary": summary(eager_times),
            }

        if args.jit:
            if jax is None:
                raise RuntimeError("--jit requires JAX to be importable.")
            jit_kwargs = {}
            if static_argnums is not None:
                jit_kwargs["static_argnums"] = static_argnums
            if donate_argnums is not None:
                jit_kwargs["donate_argnums"] = donate_argnums
            fn_jit = jax.jit(fn, **jit_kwargs)

            t0 = time.perf_counter()
            run_once(fn_jit, proc_args, proc_kwargs, jax)
            first_call_ms = (time.perf_counter() - t0) * 1e3

            for _ in range(max(args.warmup - 1, 0)):
                run_once(fn_jit, proc_args, proc_kwargs, jax)

            jit_times = benchmark(fn_jit, proc_args, proc_kwargs, args.repeat, jax)
            report["jit"] = {
                "first_call_ms": first_call_ms,
                "times_ms": jit_times,
                "summary": summary(jit_times),
            }

        if "eager" in report and "jit" in report:
            eager_mean = report["eager"]["summary"]["mean_ms"]
            jit_mean = report["jit"]["summary"]["mean_ms"]
            report["speedup_vs_eager_mean"] = (eager_mean / jit_mean) if jit_mean else None

    except Exception as exc:
        sys.stderr.write(f"Error: {type(exc).__name__}: {exc}\n")
        return 2

    if args.format == "json":
        text = json.dumps(report, indent=2, sort_keys=True)
    else:
        lines = [
            f"Callable: {report['source']}::{report['callable']}",
            f"Repeat: {report['repeat']}",
            f"Warmup: {report['warmup']}",
            f"JAX available: {report['jax_available']}",
            f"JIT requested: {report['jit_requested']}",
            f"Static argnums: {report['static_argnums']}",
            f"Donate argnums: {report['donate_argnums']}",
            "",
        ]
        if "eager" in report:
            lines.append("Eager")
            for key, value in report["eager"]["summary"].items():
                lines.append(f"  {key}: {value:.3f}")
            lines.append("")
        if "jit" in report:
            lines.append("JIT")
            lines.append(f"  first_call_ms: {report['jit']['first_call_ms']:.3f}")
            for key, value in report["jit"]["summary"].items():
                lines.append(f"  {key}: {value:.3f}")
            lines.append("")
        if "speedup_vs_eager_mean" in report:
            lines.append(f"speedup_vs_eager_mean: {report['speedup_vs_eager_mean']:.3f}")
        text = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
