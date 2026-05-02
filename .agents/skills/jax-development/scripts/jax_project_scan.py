\
#!/usr/bin/env python3
"""Static scanner for common JAX sharp bits.

The scanner is intentionally conservative:
- it reports likely review targets
- it does not claim to prove a bug
- it is useful even without JAX installed
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_EXCLUDES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


@dataclass
class Finding:
    file: str
    line: int
    column: int
    severity: str
    kind: str
    message: str
    snippet: str | None


def dotted_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = dotted_name(node.value)
        return f"{head}.{node.attr}" if head else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    return None


def parse_int_literal(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


class JaxScanner(ast.NodeVisitor):
    def __init__(self, filename: Path, source: str):
        self.filename = filename
        self.lines = source.splitlines()
        self.findings: list[Finding] = []
        self.loop_depth = 0
        self.module_level = True

        self.jit_aliases = {"jax.jit", "jit"}
        self.transform_aliases = {
            "jax.jit",
            "jax.pmap",
            "jax.vmap",
            "jax.shard_map",
            "jax.smap",
            "jax.experimental.pjit.pjit",
            "jit",
            "pmap",
            "vmap",
            "shard_map",
            "smap",
            "pjit",
        }
        self.numpy_aliases = {"np", "numpy"}
        self.random_key_aliases = {
            "jax.random.key",
            "jax.random.PRNGKey",
            "random.key",
            "random.PRNGKey",
            "key",
            "PRNGKey",
        }
        self.in_transformed_function_stack: list[str] = []

    def add(self, node: ast.AST, severity: str, kind: str, message: str) -> None:
        lineno = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        snippet = None
        if 1 <= lineno <= len(self.lines):
            snippet = self.lines[lineno - 1].strip()
        self.findings.append(
            Finding(
                file=str(self.filename),
                line=lineno,
                column=col,
                severity=severity,
                kind=kind,
                message=message,
                snippet=snippet,
            )
        )

    def _is_transform_decorator(self, node: ast.AST) -> bool:
        name = dotted_name(node)
        if name in self.jit_aliases or name in self.transform_aliases:
            return True
        if isinstance(node, ast.Call):
            func_name = dotted_name(node.func)
            if func_name in {"functools.partial", "partial"} and node.args:
                first = dotted_name(node.args[0])
                return first in self.jit_aliases or first in self.transform_aliases
        return False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            if name == "jax":
                self.jit_aliases.add(f"{asname}.jit")
                self.transform_aliases.update(
                    {f"{asname}.jit", f"{asname}.vmap", f"{asname}.pmap", f"{asname}.shard_map", f"{asname}.smap"}
                )
            if name == "numpy":
                self.numpy_aliases.add(asname)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            target = alias.asname or alias.name
            full = f"{mod}.{alias.name}" if mod else alias.name
            if full == "jax.jit":
                self.jit_aliases.add(target)
                self.transform_aliases.add(target)
            elif full in {
                "jax.vmap",
                "jax.pmap",
                "jax.shard_map",
                "jax.smap",
                "jax.experimental.pjit.pjit",
            }:
                self.transform_aliases.add(target)
            elif full in {"jax.random.key", "jax.random.PRNGKey"}:
                self.random_key_aliases.add(target)
            elif full == "numpy":
                self.numpy_aliases.add(target)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.module_level:
            rhs = dotted_name(node.value)
            if rhs in self.random_key_aliases:
                self.add(
                    node,
                    "warning",
                    "global-prng-key",
                    "Module-level PRNG key detected. Prefer explicit key threading rather than persistent hidden RNG state.",
                )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        transformed = any(self._is_transform_decorator(d) for d in node.decorator_list)
        prev_module_level = self.module_level
        self.module_level = False
        if transformed:
            self.in_transformed_function_stack.append(node.name)
        self.generic_visit(node)
        if transformed:
            self.in_transformed_function_stack.pop()
        self.module_level = prev_module_level

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        if self.in_transformed_function_stack:
            self.add(
                node,
                "warning",
                "python-if-in-transformed-function",
                "Python `if` inside a transformed function may fail if the condition depends on traced values. Consider `jax.lax.cond` or `jnp.where`.",
            )
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        if self.in_transformed_function_stack:
            msg = "Python `for` loop inside a transformed function will execute at trace time or be unrolled. Consider `lax.scan` or `lax.fori_loop`."
            trip_count = None
            if isinstance(node.iter, ast.Call) and dotted_name(node.iter.func) == "range":
                if node.iter.args:
                    trip_count = parse_int_literal(node.iter.args[0])
            if trip_count is not None and trip_count <= 4:
                msg += " This loop is tiny, so it may be acceptable; still review it intentionally."
            self.add(node, "info", "python-for-in-transformed-function", msg)
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self.loop_depth += 1
        if self.in_transformed_function_stack:
            self.add(
                node,
                "warning",
                "python-while-in-transformed-function",
                "Python `while` loop inside transformed code often needs `jax.lax.while_loop`.",
            )
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_ListComp(self, node: ast.ListComp) -> None:
        if self.in_transformed_function_stack:
            self.add(
                node,
                "info",
                "list-comprehension-in-transformed-function",
                "List comprehension inside transformed code may signal Python-side work. Review whether this should be array programming or `vmap`.",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)

        if self.loop_depth > 0 and name in self.jit_aliases:
            self.add(
                node,
                "warning",
                "jit-created-inside-loop",
                "A jitted callable is being created inside a loop. Hoist `jax.jit(...)` out of the loop to avoid repeated tracing and confusing cache behaviour.",
            )

        if name == "jax.random.PRNGKey" or name == "random.PRNGKey" or name == "PRNGKey":
            self.add(
                node,
                "info",
                "legacy-key-api",
                "Legacy `PRNGKey` API detected. New code often prefers typed keys via `jax.random.key(...)`, unless compatibility requires legacy keys.",
            )

        if name == "jax.pmap" or name == "pmap":
            self.add(
                node,
                "info",
                "pmap-usage",
                "`pmap` detected. Review whether modern sharding APIs would be clearer for new code or major refactors.",
            )

        if self.in_transformed_function_stack:
            if name == "print":
                self.add(
                    node,
                    "warning",
                    "print-in-transformed-function",
                    "Use `jax.debug.print` for traced runtime values. Plain `print` only sees trace-time information.",
                )
            if name in {"open", "logging.info", "logging.warning", "logging.debug"}:
                self.add(
                    node,
                    "warning",
                    "side-effect-in-transformed-function",
                    "Host-side side effects inside transformed functions are a common source of confusion and tracer leaks.",
                )
            if name in {"np.asarray", "numpy.asarray", "np.array", "numpy.array"} or any(
                name == f"{alias}.asarray" or name == f"{alias}.array" for alias in self.numpy_aliases
            ):
                self.add(
                    node,
                    "warning",
                    "numpy-conversion-in-transformed-function",
                    "NumPy conversion inside transformed code can force host conversion or fail on tracers. Stay in `jax.numpy` inside the compiled region.",
                )
            if name in {"jax.device_get", "device_get"}:
                self.add(
                    node,
                    "warning",
                    "device-get-in-transformed-function",
                    "Device-to-host transfer inside transformed code is usually a performance or correctness smell.",
                )

        if isinstance(node.func, ast.Attribute) and node.func.attr in {"item", "tolist"}:
            self.add(
                node,
                "warning" if self.in_transformed_function_stack else "info",
                "host-conversion",
                f"Array conversion via `.{node.func.attr}()` may force synchronisation or host conversion. Review whether this belongs outside the hot path.",
            )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = dotted_name(node)
        if name in {"jax.device_get", "jax.device_put", "jax.block_until_ready"}:
            # Merely using these is not wrong, but it often matters during review.
            self.add(
                node,
                "info",
                "runtime-boundary-api",
                f"`{name}` appears in the code. Review whether it represents an intentional device boundary.",
            )
        self.generic_visit(node)


def iter_python_files(root: Path, excludes: set[str]) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".py" else []

    files = []
    for path in root.rglob("*.py"):
        if any(part in excludes for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def scan_file(path: Path) -> tuple[list[Finding], str | None]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="latin-1")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [], f"SyntaxError: {exc}"
    scanner = JaxScanner(path, source)
    scanner.visit(tree)
    return scanner.findings, None


def build_suggestions(findings: list[Finding]) -> list[str]:
    kinds = {f.kind for f in findings}
    suggestions = []
    if "python-if-in-transformed-function" in kinds or "python-while-in-transformed-function" in kinds:
        suggestions.append("Review data-dependent Python control flow and replace it with `lax.cond`, `lax.while_loop`, or `jnp.where` where appropriate.")
    if "python-for-in-transformed-function" in kinds:
        suggestions.append("Check long Python loops under `jit`; many should be `lax.scan` or `lax.fori_loop`.")
    if "global-prng-key" in kinds or "legacy-key-api" in kinds:
        suggestions.append("Review PRNG handling. Prefer explicit key threading and typed keys for new code unless compatibility requires legacy keys.")
    if "numpy-conversion-in-transformed-function" in kinds or "host-conversion" in kinds:
        suggestions.append("Review host/device boundaries. Keep hot-path computations in `jax.numpy` and move host conversions to the program edge.")
    if "jit-created-inside-loop" in kinds:
        suggestions.append("Hoist `jax.jit` construction out of loops to reduce repeated tracing and cache churn.")
    if "pmap-usage" in kinds:
        suggestions.append("For major refactors, compare current `pmap` usage against modern sharding APIs and `shard_map` migration guidance.")
    return suggestions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Python files for common JAX sharp bits and migration targets.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Exit codes:
  0 success
  2 operational error

Examples:
  python3 scripts/jax_project_scan.py .
  python3 scripts/jax_project_scan.py src/ --format text
  python3 scripts/jax_project_scan.py my_file.py --max-findings 50
""",
    )
    parser.add_argument("path", help="File or directory to scan.")
    parser.add_argument("--exclude-dir", action="append", default=[], help="Directory names to exclude. May be passed multiple times.")
    parser.add_argument("--format", choices=("json", "text"), default="json", help="Output format. Default: json")
    parser.add_argument("--max-findings", type=int, default=200, help="Maximum number of findings to include. Default: 200")
    parser.add_argument("--output", help="Write output to a file instead of stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.path)
    if not root.exists():
        sys.stderr.write(f"Error: path not found: {root}\n")
        return 2

    excludes = set(DEFAULT_EXCLUDES) | set(args.exclude_dir)
    files = iter_python_files(root, excludes)

    all_findings: list[Finding] = []
    parse_errors: list[dict[str, str]] = []
    for path in files:
        findings, parse_error = scan_file(path)
        if parse_error is not None:
            parse_errors.append({"file": str(path), "error": parse_error})
        all_findings.extend(findings)

    severity_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for finding in all_findings:
        severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
        kind_counts[finding.kind] = kind_counts.get(finding.kind, 0) + 1

    all_findings_sorted = sorted(all_findings, key=lambda f: (f.file, f.line, f.column, f.kind))
    limited_findings = all_findings_sorted[: max(args.max_findings, 0)]

    report = {
        "path": str(root),
        "files_scanned": len(files),
        "parse_errors": parse_errors,
        "summary": {
            "total_findings": len(all_findings),
            "shown_findings": len(limited_findings),
            "severity_counts": severity_counts,
            "kind_counts": dict(sorted(kind_counts.items())),
        },
        "suggestions": build_suggestions(all_findings),
        "findings": [asdict(f) for f in limited_findings],
    }

    if args.format == "json":
        text = json.dumps(report, indent=2, sort_keys=True)
    else:
        lines = [
            f"Path: {report['path']}",
            f"Files scanned: {report['files_scanned']}",
            f"Parse errors: {len(report['parse_errors'])}",
            f"Total findings: {report['summary']['total_findings']}",
            "",
            "Severity counts:",
        ]
        for sev, count in sorted(report["summary"]["severity_counts"].items()):
            lines.append(f"  {sev}: {count}")
        lines.append("")
        lines.append("Suggestions:")
        if report["suggestions"]:
            for item in report["suggestions"]:
                lines.append(f"  - {item}")
        else:
            lines.append("  - No specific suggestions.")
        lines.append("")
        lines.append("Findings:")
        if limited_findings:
            for f in limited_findings:
                lines.append(
                    f"  - {f.file}:{f.line}:{f.column} [{f.severity}] {f.kind}: {f.message}"
                )
                if f.snippet:
                    lines.append(f"      {f.snippet}")
        else:
            lines.append("  (none)")
        if parse_errors:
            lines.append("")
            lines.append("Parse errors:")
            for err in parse_errors:
                lines.append(f"  - {err['file']}: {err['error']}")
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
