\
#!/usr/bin/env python3
"""Search a local JAX checkout for relevant docs, tests, and source files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_DIRS = ("docs", "jax", "tests", "benchmarks")
TEXT_EXTS = {".py", ".md", ".rst", ".txt", ".bzl", ".yaml", ".yml"}


def tokenize_query(query: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9_]+", query.lower()) if len(tok) >= 2]


def iter_candidate_files(repo: Path, include_dirs: tuple[str, ...], kind: str) -> list[Path]:
    dirs = []
    if kind == "all":
        dirs = include_dirs
    elif kind == "docs":
        dirs = ("docs",)
    elif kind == "source":
        dirs = ("jax",)
    elif kind == "tests":
        dirs = ("tests",)
    elif kind == "benchmarks":
        dirs = ("benchmarks",)

    out = []
    for d in dirs:
        root = repo / d
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if path.suffix.lower() in TEXT_EXTS:
                out.append(path)
    return sorted(out)


def score_file(path: Path, repo: Path, query_tokens: list[str], phrase: str, max_bytes: int) -> dict[str, Any] | None:
    rel = path.relative_to(repo)
    rel_str = rel.as_posix().lower()

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    if len(content) > max_bytes:
        content = content[:max_bytes]

    content_lower = content.lower()

    path_hits = {tok: rel_str.count(tok) for tok in query_tokens if tok in rel_str}
    content_hits = {tok: content_lower.count(tok) for tok in query_tokens if tok in content_lower}

    score = 0
    score += sum(min(count, 3) * 5 for count in path_hits.values())
    score += sum(min(count, 5) * 1 for count in content_hits.values())

    if phrase and phrase in rel_str:
        score += 20
    if phrase and phrase in content_lower:
        score += 10

    if score == 0:
        return None

    preview_lines = []
    if query_tokens:
        line_matches = 0
        for lineno, line in enumerate(content.splitlines(), start=1):
            line_lower = line.lower()
            if any(tok in line_lower for tok in query_tokens):
                preview_lines.append({"line": lineno, "text": line.strip()[:240]})
                line_matches += 1
                if line_matches >= 5:
                    break

    return {
        "path": rel.as_posix(),
        "score": score,
        "path_hits": path_hits,
        "content_hits": content_hits,
        "preview_lines": preview_lines,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a local JAX checkout for relevant docs, tests, or source files by topic.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Exit codes:
  0 success
  2 operational error

Examples:
  python3 scripts/jax_repo_locator.py --repo /path/to/jax --query "custom vjp batching"
  python3 scripts/jax_repo_locator.py --repo . --query "shard_map pmap migration" --kind docs
  python3 scripts/jax_repo_locator.py --repo . --query "debug.print compiler_ir" --kind all --format text
""",
    )
    parser.add_argument("--repo", required=True, help="Path to a local JAX checkout.")
    parser.add_argument("--query", required=True, help="Search query.")
    parser.add_argument("--kind", choices=("all", "docs", "source", "tests", "benchmarks"), default="all", help="Subset of the repo to search. Default: all")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of results to return. Default: 20")
    parser.add_argument("--max-bytes", type=int, default=200000, help="Maximum bytes of each file to score. Default: 200000")
    parser.add_argument("--format", choices=("json", "text"), default="json", help="Output format. Default: json")
    parser.add_argument("--output", help="Write output to a file instead of stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo)
    if not repo.exists():
        sys.stderr.write(f"Error: repo path not found: {repo}\n")
        return 2

    query_tokens = tokenize_query(args.query)
    phrase = args.query.lower().strip()

    files = iter_candidate_files(repo, DEFAULT_DIRS, args.kind)
    scored = []
    for path in files:
        result = score_file(path, repo, query_tokens, phrase, args.max_bytes)
        if result is not None:
            scored.append(result)

    scored.sort(key=lambda item: (-item["score"], item["path"]))
    scored = scored[: max(args.limit, 0)]

    report = {
        "repo": str(repo),
        "query": args.query,
        "kind": args.kind,
        "query_tokens": query_tokens,
        "results": scored,
    }

    if args.format == "json":
        text = json.dumps(report, indent=2, sort_keys=True)
    else:
        lines = [
            f"Repo: {report['repo']}",
            f"Query: {report['query']}",
            f"Kind: {report['kind']}",
            "",
        ]
        if not scored:
            lines.append("No matches.")
        else:
            for item in scored:
                lines.append(f"- {item['path']} (score={item['score']})")
                if item["path_hits"]:
                    lines.append(f"    path_hits: {item['path_hits']}")
                if item["content_hits"]:
                    lines.append(f"    content_hits: {item['content_hits']}")
                for line in item["preview_lines"]:
                    lines.append(f"    L{line['line']}: {line['text']}")
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
