\
#!/usr/bin/env python3
"""Emit a structured report about the local JAX environment.

Design goals:
- non-interactive
- JSON by default
- standard-library only
- useful even when JAX is not importable
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from typing import Any

PACKAGE_NAMES = (
    "jax",
    "jaxlib",
    "numpy",
    "scipy",
    "flax",
    "optax",
    "equinox",
    "orbax-checkpoint",
)

CONFIG_KEYS = (
    "jax_enable_x64",
    "jax_default_matmul_precision",
    "jax_debug_nans",
    "jax_debug_infs",
    "jax_debug_key_reuse",
    "jax_platform_name",
    "jax_default_prng_impl",
    "jax_transfer_guard",
    "jax_compilation_cache_dir",
)

ENV_PREFIXES = (
    "JAX_",
    "XLA_",
    "CUDA_",
    "NVIDIA_",
    "ROCM",
    "HIP_",
    "TPU",
    "NCCL_",
)

ENV_NAMES = {
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "XLA_FLAGS",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "PATH",
}


def package_info(name: str) -> dict[str, Any]:
    out: dict[str, Any] = {"installed": False}
    try:
        out["version"] = importlib.metadata.version(name)
        out["installed"] = True
    except importlib.metadata.PackageNotFoundError:
        out["version"] = None
    except Exception as exc:  # pragma: no cover - best effort only
        out["version"] = None
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def selected_environment() -> dict[str, str]:
    env = {}
    for key, value in os.environ.items():
        if key in ENV_NAMES or any(key.startswith(prefix) for prefix in ENV_PREFIXES):
            env[key] = value
    return dict(sorted(env.items()))


def read_config_value(jax_module: Any, key: str) -> Any:
    cfg = getattr(jax_module, "config", None)
    if cfg is None:
        return None

    # Try the most stable public-ish access patterns first.
    for getter in (
        lambda: getattr(cfg, "values", {}).get(key),
        lambda: cfg.read(key),  # type: ignore[attr-defined]
        lambda: getattr(cfg, key),
    ):
        try:
            value = getter()
            if value is not None:
                return value
        except Exception:
            pass
    return None


def maybe_key(jax_module: Any, seed: int):
    try:
        return jax_module.random.key(seed)
    except Exception:
        return jax_module.random.PRNGKey(seed)


def smoke_test(jax_module: Any, jnp_module: Any) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": False}
    t0 = time.perf_counter()

    key = maybe_key(jax_module, 0)
    x = jax_module.random.normal(key, (256, 256), dtype=jnp_module.float32)
    y = (x @ x.T).block_until_ready()

    @jax_module.jit
    def loss_fn(z):
        return jnp_module.sum(jnp_module.tanh(z @ z.T))

    loss = loss_fn(x)
    loss.block_until_ready()
    grad = jax_module.grad(lambda z: jnp_module.sum(jnp_module.sin(z)))(x)
    jax_module.block_until_ready(grad)

    report["ok"] = True
    report["elapsed_ms"] = (time.perf_counter() - t0) * 1e3
    report["matmul_shape"] = tuple(int(v) for v in y.shape)
    report["loss_dtype"] = str(loss.dtype)
    return report


def load_jax_report(run_smoke_test: bool) -> dict[str, Any]:
    report: dict[str, Any] = {"imported": False}
    try:
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
    except Exception as exc:
        report["import_error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["imported"] = True
    report["version"] = getattr(jax, "__version__", None)

    try:
        report["default_backend"] = jax.default_backend()
    except Exception:
        report["default_backend"] = None

    try:
        report["process_count"] = int(jax.process_count())
        report["process_index"] = int(jax.process_index())
        report["device_count"] = int(jax.device_count())
        report["local_device_count"] = int(jax.local_device_count())
    except Exception as exc:
        report["process_error"] = f"{type(exc).__name__}: {exc}"

    devices = []
    try:
        for dev in jax.devices():
            devices.append(
                {
                    "id": getattr(dev, "id", None),
                    "platform": getattr(dev, "platform", None),
                    "device_kind": getattr(dev, "device_kind", None),
                    "process_index": getattr(dev, "process_index", None),
                    "memory_limit": getattr(dev, "memory_limit", None),
                }
            )
    except Exception as exc:
        report["devices_error"] = f"{type(exc).__name__}: {exc}"
    report["devices"] = devices

    cfg = {}
    for key in CONFIG_KEYS:
        cfg[key] = read_config_value(jax, key)
    report["config"] = cfg

    if run_smoke_test:
        try:
            report["smoke_test"] = smoke_test(jax, jnp)
        except Exception as exc:
            report["smoke_test"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return report


def build_report(run_smoke_test: bool) -> dict[str, Any]:
    return {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": {name: package_info(name) for name in PACKAGE_NAMES},
        "environment": selected_environment(),
        "jax": load_jax_report(run_smoke_test),
    }


def format_text(report: dict[str, Any]) -> str:
    lines = []
    lines.append("Python")
    lines.append(f"  version: {report['python']['version']}")
    lines.append(f"  executable: {report['python']['executable']}")
    lines.append("")
    lines.append("Platform")
    for key, value in report["platform"].items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Packages")
    for name, info in report["packages"].items():
        status = info.get("version") if info.get("installed") else "not installed"
        lines.append(f"  {name}: {status}")
    lines.append("")
    lines.append("Environment")
    for key, value in report["environment"].items():
        lines.append(f"  {key}={value}")
    lines.append("")

    jax_info = report["jax"]
    lines.append("JAX")
    if not jax_info.get("imported"):
        lines.append(f"  import_failed: {jax_info.get('import_error')}")
        return "\n".join(lines)

    lines.append(f"  version: {jax_info.get('version')}")
    lines.append(f"  default_backend: {jax_info.get('default_backend')}")
    lines.append(f"  process_count: {jax_info.get('process_count')}")
    lines.append(f"  process_index: {jax_info.get('process_index')}")
    lines.append(f"  device_count: {jax_info.get('device_count')}")
    lines.append(f"  local_device_count: {jax_info.get('local_device_count')}")
    lines.append("")
    lines.append("Devices")
    for device in jax_info.get("devices", []):
        lines.append(
            "  - id={id} platform={platform} kind={device_kind} process_index={process_index} memory_limit={memory_limit}".format(
                **device
            )
        )
    lines.append("")
    lines.append("Config")
    for key, value in jax_info.get("config", {}).items():
        lines.append(f"  {key}: {value}")

    smoke = jax_info.get("smoke_test")
    if smoke:
        lines.append("")
        lines.append("Smoke test")
        for key, value in smoke.items():
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit a structured report about the local JAX environment.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Exit codes:
  0 success
  2 operational error

Examples:
  python3 scripts/jax_env_report.py
  python3 scripts/jax_env_report.py --smoke-test --format text
  python3 scripts/jax_env_report.py --output env.json
""",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a small JAX smoke test if JAX is importable.")
    parser.add_argument("--format", choices=("json", "text"), default="json", help="Output format. Default: json")
    parser.add_argument("--output", help="Write output to a file instead of stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args.smoke_test)
        text = json.dumps(report, indent=2, sort_keys=True) if args.format == "json" else format_text(report)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
        else:
            sys.stdout.write(text)
            if not text.endswith("\n"):
                sys.stdout.write("\n")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {type(exc).__name__}: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
