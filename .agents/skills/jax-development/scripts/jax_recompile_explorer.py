\
#!/usr/bin/env python3
"""Run several input cases through a jitted function and flag likely recompiles.

This script uses a combination of lowering hashes and first-vs-second-call timing.
It is heuristic, but very useful for catching shape/static-argument churn.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
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


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def block_until_ready(value: Any, jax_module: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    try:
        return jax_module.block_until_ready(value)
    except Exception:
        return value


def parse_int_tuple(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or raw == "":
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def to_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    for getter in (
        lambda: obj.as_text(),
        lambda: obj.operation.get_asm(enable_debug_info=True),
        lambda: obj.operation.get_asm(),
    ):
        try:
            return getter()
        except Exception:
            pass
    return str(obj)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def case_signature(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return {"shape": tuple(int(v) for v in shape), "dtype": str(dtype)}
    if isinstance(value, dict):
        return {k: case_signature(v) for k, v in value.items()}
    if isinstance(value, list):
        return [case_signature(v) for v in value]
    return {"type": type(value).__name__, "value_repr": repr(value)[:120]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run several input cases through a jitted function and flag likely recompiles.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Case file format:
[
  {"name": "case_a", "args": [[1, 2, 3]], "kwargs": {}},
  {"name": "case_b", "args": [[1, 2, 3, 4]], "kwargs": {}}
]

Exit codes:
  0 success
  2 operational error

Examples:
  python3 scripts/jax_recompile_explorer.py --file train.py --function step --cases-file cases.json --arrayify
  python3 scripts/jax_recompile_explorer.py --module pkg.mod --function step --cases-file cases.json --arrayify --static-argnums 2
""",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--module", help="Import path for the module containing the callable.")
    source.add_argument("--file", help="Path to a Python file containing the callable.")
    parser.add_argument("--function", required=True, help="Callable name or dotted attribute path.")
    parser.add_argument("--cases-file", required=True, help="Path to a JSON file containing a list of cases.")
    parser.add_argument("--arrayify", action="store_true", help="Convert numeric lists to `jax.numpy.array` when JAX is available.")
    parser.add_argument("--static-argnums", help="Comma-separated positional indices to mark static when jitting.")
    parser.add_argument("--donate-argnums", help="Comma-separated positional indices to donate when jitting.")
    parser.add_argument("--ratio-threshold", type=float, default=3.0, help="Flag a likely compile when first_call_ms / second_call_ms exceeds this ratio. Default: 3.0")
    parser.add_argument("--min-first-call-ms", type=float, default=1.0, help="Ignore compile suspicion for tiny first calls below this threshold. Default: 1.0")
    parser.add_argument("--format", choices=("json", "text"), default="json", help="Output format. Default: json")
    parser.add_argument("--output", help="Write output to a file instead of stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        jax, jnp = maybe_import_jax()
        if jax is None or jnp is None:
            raise RuntimeError("JAX is not importable in this environment.")

        module = load_module(args.module, args.file)
        fn = resolve_attr(module, args.function)
        if not callable(fn):
            raise TypeError(f"Resolved object is not callable: {args.function}")

        raw_cases = load_json(args.cases_file)
        if not isinstance(raw_cases, list):
            raise TypeError("Case file must decode to a list of case objects.")

        static_argnums = parse_int_tuple(args.static_argnums)
        donate_argnums = parse_int_tuple(args.donate_argnums)
        jit_kwargs = {}
        if static_argnums is not None:
            jit_kwargs["static_argnums"] = static_argnums
        if donate_argnums is not None:
            jit_kwargs["donate_argnums"] = donate_argnums
        fn_jit = jax.jit(fn, **jit_kwargs)

        seen_lowerings: dict[str, str] = {}
        case_reports = []

        for idx, raw_case in enumerate(raw_cases):
            if not isinstance(raw_case, dict):
                raise TypeError(f"Case {idx} is not an object.")
            name = raw_case.get("name", f"case_{idx}")
            case_args = raw_case.get("args", [])
            case_kwargs = raw_case.get("kwargs", {})
            if not isinstance(case_args, list) or not isinstance(case_kwargs, dict):
                raise TypeError(f"Case {name} must contain `args` (list) and `kwargs` (object).")

            if args.arrayify:
                case_args = [tree_arrayify(v, jnp) for v in case_args]
                case_kwargs = {k: tree_arrayify(v, jnp) for k, v in case_kwargs.items()}

            lower_hash = None
            lower_error = None
            try:
                lowered = fn_jit.lower(*case_args, **case_kwargs)
                lower_hash = sha256_text(to_text(lowered))
            except Exception as exc:
                lower_error = f"{type(exc).__name__}: {exc}"

            first_call_ms = None
            second_call_ms = None
            first_call_error = None
            second_call_error = None
            try:
                t0 = time.perf_counter()
                out1 = fn_jit(*case_args, **case_kwargs)
                block_until_ready(out1, jax)
                first_call_ms = (time.perf_counter() - t0) * 1e3
            except Exception as exc:
                first_call_error = f"{type(exc).__name__}: {exc}"

            try:
                t1 = time.perf_counter()
                out2 = fn_jit(*case_args, **case_kwargs)
                block_until_ready(out2, jax)
                second_call_ms = (time.perf_counter() - t1) * 1e3
            except Exception as exc:
                second_call_error = f"{type(exc).__name__}: {exc}"

            ratio = (first_call_ms / second_call_ms) if (first_call_ms is not None and second_call_ms not in (None, 0)) else None
            compile_suspected = (
                ratio is not None
                and first_call_ms is not None
                and first_call_ms >= args.min_first_call_ms
                and ratio >= args.ratio_threshold
            )

            reused_from = None
            if lower_hash is not None and lower_hash in seen_lowerings:
                reused_from = seen_lowerings[lower_hash]
            elif lower_hash is not None:
                seen_lowerings[lower_hash] = name

            case_reports.append(
                {
                    "name": name,
                    "args_signature": case_signature(case_args),
                    "kwargs_signature": case_signature(case_kwargs),
                    "lower_hash": lower_hash,
                    "lower_error": lower_error,
                    "reused_lowering_from": reused_from,
                    "first_call_ms": first_call_ms,
                    "second_call_ms": second_call_ms,
                    "first_call_error": first_call_error,
                    "second_call_error": second_call_error,
                    "ratio_first_to_second": ratio,
                    "compile_suspected": compile_suspected,
                }
            )

        report = {
            "source": args.module or args.file,
            "callable": args.function,
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums,
            "ratio_threshold": args.ratio_threshold,
            "min_first_call_ms": args.min_first_call_ms,
            "unique_lowerings": len({c["lower_hash"] for c in case_reports if c["lower_hash"] is not None}),
            "cases": case_reports,
        }

    except Exception as exc:
        sys.stderr.write(f"Error: {type(exc).__name__}: {exc}\n")
        return 2

    if args.format == "json":
        text = json.dumps(report, indent=2, sort_keys=True)
    else:
        lines = [
            f"Callable: {report['source']}::{report['callable']}",
            f"Static argnums: {report['static_argnums']}",
            f"Donate argnums: {report['donate_argnums']}",
            f"Unique lowerings: {report['unique_lowerings']}",
            "",
        ]
        for case in report["cases"]:
            lines.append(f"Case: {case['name']}")
            lines.append(f"  first_call_ms: {case['first_call_ms']}")
            lines.append(f"  second_call_ms: {case['second_call_ms']}")
            if case["first_call_error"]:
                lines.append(f"  first_call_error: {case['first_call_error']}")
            if case["second_call_error"]:
                lines.append(f"  second_call_error: {case['second_call_error']}")
            lines.append(f"  ratio_first_to_second: {case['ratio_first_to_second']}")
            lines.append(f"  compile_suspected: {case['compile_suspected']}")
            lines.append(f"  reused_lowering_from: {case['reused_lowering_from']}")
            if case["lower_error"]:
                lines.append(f"  lower_error: {case['lower_error']}")
            lines.append("")
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
