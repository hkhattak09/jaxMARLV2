\
#!/usr/bin/env python3
"""Inspect JAX tracing/lowering artefacts for a callable."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
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


def parse_int_tuple(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or raw == "":
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def sha256_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def preview_text(text: str, max_lines: int, max_chars: int) -> str:
    lines = text.splitlines()
    clipped = "\n".join(lines[:max_lines])
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars]
    return clipped


def write_artifact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception as exc:  # pragma: no cover - best effort only
        return f"<repr failed: {type(exc).__name__}: {exc}>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect `eval_shape`, jaxpr, lowering, and compiler IR for a callable.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Exit codes:
  0 success
  2 operational error

Examples:
  python3 scripts/jax_compile_probe.py --file my_module.py --function step --args-file args.json --arrayify
  python3 scripts/jax_compile_probe.py --module pkg.train --function step --args-file args.json --jit --write-dir probe_out
  python3 scripts/jax_compile_probe.py --file my_module.py --function fn --args-json '[[1, 2, 3]]' --arrayify --dialects stablehlo,hlo
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
    parser.add_argument("--jit", action="store_true", help="Wrap the callable with `jax.jit` before lowering.")
    parser.add_argument("--static-argnums", help="Comma-separated positional indices to mark static when using --jit.")
    parser.add_argument("--donate-argnums", help="Comma-separated positional indices to donate when using --jit.")
    parser.add_argument("--compile", action="store_true", help="Compile the lowered program and report compile timing when possible.")
    parser.add_argument("--dialects", default="stablehlo", help="Comma-separated compiler IR dialects to attempt. Default: stablehlo")
    parser.add_argument("--max-preview-lines", type=int, default=80, help="Maximum preview lines per artefact. Default: 80")
    parser.add_argument("--max-preview-chars", type=int, default=12000, help="Maximum preview characters per artefact. Default: 12000")
    parser.add_argument("--write-dir", help="Optional directory for full artefacts such as jaxpr and IR text files.")
    parser.add_argument("--format", choices=("json", "text"), default="json", help="Output format. Default: json")
    parser.add_argument("--output", help="Write the report to a file instead of stdout.")
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

        raw_args = load_json_arg(args.args_json, args.args_file, [])
        raw_kwargs = load_json_arg(args.kwargs_json, args.kwargs_file, {})
        if not isinstance(raw_args, list):
            raise TypeError("Positional arguments JSON must decode to a list.")
        if not isinstance(raw_kwargs, dict):
            raise TypeError("Keyword arguments JSON must decode to an object.")

        proc_args = list(raw_args)
        proc_kwargs = dict(raw_kwargs)
        if args.arrayify:
            proc_args = [tree_arrayify(v, jnp) for v in proc_args]
            proc_kwargs = {k: tree_arrayify(v, jnp) for k, v in proc_kwargs.items()}

        static_argnums = parse_int_tuple(args.static_argnums)
        donate_argnums = parse_int_tuple(args.donate_argnums)

        report: dict[str, Any] = {
            "source": args.module or args.file,
            "callable": args.function,
            "jit_requested": args.jit,
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums,
            "dialects_requested": [d.strip() for d in args.dialects.split(",") if d.strip()],
        }

        try:
            eval_shape = jax.eval_shape(fn, *proc_args, **proc_kwargs)
            report["eval_shape_repr"] = safe_repr(eval_shape)
        except Exception as exc:
            report["eval_shape_error"] = f"{type(exc).__name__}: {exc}"

        try:
            jaxpr_obj = jax.make_jaxpr(fn)(*proc_args, **proc_kwargs)
            jaxpr_text = str(jaxpr_obj)
            report["jaxpr"] = {
                "lines": len(jaxpr_text.splitlines()),
                "sha256": sha256_text(jaxpr_text),
                "preview": preview_text(jaxpr_text, args.max_preview_lines, args.max_preview_chars),
            }
            if args.write_dir:
                write_artifact(Path(args.write_dir) / "jaxpr.txt", jaxpr_text)
        except Exception as exc:
            report["jaxpr_error"] = f"{type(exc).__name__}: {exc}"

        lowered_fn = fn
        if args.jit:
            jit_kwargs = {}
            if static_argnums is not None:
                jit_kwargs["static_argnums"] = static_argnums
            if donate_argnums is not None:
                jit_kwargs["donate_argnums"] = donate_argnums
            lowered_fn = jax.jit(fn, **jit_kwargs)

        try:
            if not hasattr(lowered_fn, "lower"):
                raise TypeError("Target does not support `.lower(...)`; try using --jit.")
            lowered = lowered_fn.lower(*proc_args, **proc_kwargs)
            lowered_text = to_text(lowered)
            report["lowering"] = {
                "sha256": sha256_text(lowered_text),
                "preview": preview_text(lowered_text, args.max_preview_lines, args.max_preview_chars),
            }
            if args.write_dir:
                write_artifact(Path(args.write_dir) / "lowering.txt", lowered_text)

            ir_reports = {}
            for dialect in report["dialects_requested"]:
                try:
                    ir_obj = lowered.compiler_ir(dialect=dialect)
                    ir_text = to_text(ir_obj)
                    ir_reports[dialect] = {
                        "sha256": sha256_text(ir_text),
                        "preview": preview_text(ir_text, args.max_preview_lines, args.max_preview_chars),
                    }
                    if args.write_dir:
                        suffix = "mlir" if dialect in {"stablehlo", "mhlo"} else "txt"
                        write_artifact(Path(args.write_dir) / f"{dialect}.{suffix}", ir_text)
                except Exception as exc:
                    ir_reports[dialect] = {"error": f"{type(exc).__name__}: {exc}"}
            report["compiler_ir"] = ir_reports

            if args.compile:
                import time

                t0 = time.perf_counter()
                compiled = lowered.compile()
                compile_ms = (time.perf_counter() - t0) * 1e3
                report["compile"] = {
                    "ok": compiled is not None,
                    "elapsed_ms": compile_ms,
                }
        except Exception as exc:
            report["lowering_error"] = f"{type(exc).__name__}: {exc}"

    except Exception as exc:
        sys.stderr.write(f"Error: {type(exc).__name__}: {exc}\n")
        return 2

    if args.format == "json":
        text = json.dumps(report, indent=2, sort_keys=True)
    else:
        lines = [
            f"Callable: {report['source']}::{report['callable']}",
            f"JIT requested: {report['jit_requested']}",
            f"Static argnums: {report['static_argnums']}",
            f"Donate argnums: {report['donate_argnums']}",
            "",
        ]
        if "eval_shape_repr" in report:
            lines.append("eval_shape:")
            lines.append(report["eval_shape_repr"])
            lines.append("")
        if "eval_shape_error" in report:
            lines.append(f"eval_shape_error: {report['eval_shape_error']}")
            lines.append("")
        if "jaxpr" in report:
            lines.append(f"jaxpr lines: {report['jaxpr']['lines']}")
            lines.append(report["jaxpr"]["preview"])
            lines.append("")
        if "lowering" in report:
            lines.append("lowering preview:")
            lines.append(report["lowering"]["preview"])
            lines.append("")
        if "compiler_ir" in report:
            for dialect, info in report["compiler_ir"].items():
                lines.append(f"{dialect}:")
                if "error" in info:
                    lines.append(f"  error: {info['error']}")
                else:
                    lines.append(info["preview"])
                lines.append("")
        if "compile" in report:
            lines.append(f"compile elapsed_ms: {report['compile']['elapsed_ms']:.3f}")
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
