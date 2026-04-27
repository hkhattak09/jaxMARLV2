"""
Run and summarize Stage 3B Role-LoRA ablations.

This script is intended for Colab/notebook use. It launches the ROSA MAPPO
experiment script with explicit CLI overrides, stores one log per ablation, and
builds a compact metrics table from the logged learning curves.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev


LOG_LINE_RE = re.compile(
    r"Step\s+(?P<step>\d+)\s+\|\s+Return:\s+(?P<return>[-+0-9.eE]+)\s+"
    r"\|\s+Win Rate:\s+(?P<win_rate>[-+0-9.eE]+)"
)


DEFAULT_MODES = (
    "none",
    "role_lora",
    "global_lora",
    "agent_lora",
    "sequential_polish",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stage 3B Role-LoRA ablations and summarize learning curves."
    )
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--map_name", type=str, default="smacv2_5_units")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--role_lora_rank", type=int, default=4)
    parser.add_argument("--role_lora_scale", type=float, default=1.0)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument("--target_win_rate", type=float, default=0.8)
    return parser.parse_args()


def run_name_for(mode, map_name, seed):
    if mode == "none":
        prefix = "baseline_none"
    elif mode == "role_lora":
        prefix = "stage3_role_lora"
    elif mode == "sequential_polish":
        prefix = "seq_polish"
    else:
        prefix = mode
    return f"{prefix}_{map_name}_seed{seed}"


def parse_curve(log_path):
    points = []
    if not log_path.exists():
        return points
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = LOG_LINE_RE.search(line)
            if not match:
                continue
            points.append(
                {
                    "step": int(match.group("step")),
                    "return": float(match.group("return")),
                    "win_rate": float(match.group("win_rate")),
                }
            )
    return points


def trapezoid_auc(points, key):
    if len(points) < 2:
        return None
    area = 0.0
    for prev, cur in zip(points, points[1:]):
        width = cur["step"] - prev["step"]
        if width < 0:
            raise ValueError(f"Learning curve steps are not monotonic near step {cur['step']}.")
        area += width * (prev[key] + cur[key]) / 2.0
    total_width = points[-1]["step"] - points[0]["step"]
    if total_width <= 0:
        return None
    return area / total_width


def summarize_curve(mode, seed, run_name, log_path, target_win_rate):
    points = parse_curve(log_path)
    if not points:
        return {
            "mode": mode,
            "seed": seed,
            "run_name": run_name,
            "status": "missing_or_unparsed",
            "num_points": 0,
            "final_step": "",
            "final_return": "",
            "final_win_rate": "",
            "best_win_rate": "",
            "best_win_rate_step": "",
            "auc_win_rate": "",
            "auc_return": "",
            "steps_to_target_win_rate": "",
            "log_path": str(log_path),
        }

    best = max(points, key=lambda point: point["win_rate"])
    first_target = next(
        (point for point in points if point["win_rate"] >= target_win_rate),
        None,
    )
    final = points[-1]
    return {
        "mode": mode,
        "seed": seed,
        "run_name": run_name,
        "status": "ok",
        "num_points": len(points),
        "final_step": final["step"],
        "final_return": final["return"],
        "final_win_rate": final["win_rate"],
        "best_win_rate": best["win_rate"],
        "best_win_rate_step": best["step"],
        "auc_win_rate": trapezoid_auc(points, "win_rate"),
        "auc_return": trapezoid_auc(points, "return"),
        "steps_to_target_win_rate": first_target["step"] if first_target else "",
        "log_path": str(log_path),
    }


def write_csv(rows, path):
    fieldnames = [
        "mode",
        "seed",
        "run_name",
        "status",
        "num_points",
        "final_step",
        "final_return",
        "final_win_rate",
        "best_win_rate",
        "best_win_rate_step",
        "auc_win_rate",
        "auc_return",
        "steps_to_target_win_rate",
        "log_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value):
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows, path):
    columns = [
        "mode",
        "seed",
        "status",
        "final_step",
        "final_return",
        "final_win_rate",
        "best_win_rate",
        "best_win_rate_step",
        "auc_win_rate",
        "steps_to_target_win_rate",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_value(row[column]) for column in columns) + " |\n")


def numeric_values(rows, mode, key):
    values = []
    for row in rows:
        if row["mode"] != mode or row["status"] != "ok" or row[key] == "":
            continue
        values.append(float(row[key]))
    return values


def summarize_by_mode(rows, modes):
    aggregate_rows = []
    for mode in modes:
        final_win_rates = numeric_values(rows, mode, "final_win_rate")
        best_win_rates = numeric_values(rows, mode, "best_win_rate")
        final_returns = numeric_values(rows, mode, "final_return")
        auc_win_rates = numeric_values(rows, mode, "auc_win_rate")
        aggregate_rows.append(
            {
                "mode": mode,
                "num_seeds": len(final_win_rates),
                "final_win_rate_mean": mean(final_win_rates) if final_win_rates else "",
                "final_win_rate_std": stdev(final_win_rates) if len(final_win_rates) > 1 else "",
                "best_win_rate_mean": mean(best_win_rates) if best_win_rates else "",
                "final_return_mean": mean(final_returns) if final_returns else "",
                "auc_win_rate_mean": mean(auc_win_rates) if auc_win_rates else "",
            }
        )
    return aggregate_rows


def write_aggregate_markdown(rows, path):
    columns = [
        "mode",
        "num_seeds",
        "final_win_rate_mean",
        "final_win_rate_std",
        "best_win_rate_mean",
        "final_return_mean",
        "auc_win_rate_mean",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_value(row[column]) for column in columns) + " |\n")


def tee_process(command, log_path):
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
        log_file.write(f"\n[ablation_runner] return_code={return_code}\n")
        log_file.write(f"[ablation_runner] elapsed_minutes={(time.time() - start) / 60.0:.2f}\n")
    return return_code


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    seeds = args.seeds if args.seeds is not None else [args.seed]
    train_script = repo_root / "smax_ctm" / "train_rosa_mappo.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Could not find training script: {train_script}")

    output_dir = args.output_dir
    if output_dir is None:
        if len(seeds) == 1:
            seed_suffix = f"seed{seeds[0]}"
        else:
            seed_suffix = "seeds_" + "_".join(str(seed) for seed in seeds)
        output_dir = repo_root / "logs" / f"stage3b_{args.map_name}_{seed_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        for mode in args.modes:
            run_name = run_name_for(mode, args.map_name, seed)
            log_path = output_dir / f"{run_name}.log"
            if not args.summarize_only:
                command = [
                    args.python,
                    str(train_script),
                    "--map_name",
                    args.map_name,
                    "--seed",
                    str(seed),
                    "--adapter_mode",
                    mode,
                    "--total_timesteps",
                    str(args.total_timesteps),
                    "--num_envs",
                    str(args.num_envs),
                    "--num_steps",
                    str(args.num_steps),
                    "--num_minibatches",
                    str(args.num_minibatches),
                    "--update_epochs",
                    str(args.update_epochs),
                    "--role_lora_rank",
                    str(args.role_lora_rank),
                    "--role_lora_scale",
                    str(args.role_lora_scale),
                    "--run_name",
                    run_name,
                ]
                print(f"\n[ablation_runner] Running seed={seed} mode={mode}: {' '.join(command)}")
                return_code = tee_process(command, log_path)
                if return_code != 0 and not args.keep_going:
                    raise RuntimeError(
                        f"Ablation seed={seed} mode={mode!r} failed with return code {return_code}. "
                        f"See log: {log_path}"
                    )
            rows.append(summarize_curve(mode, seed, run_name, log_path, args.target_win_rate))

    csv_path = output_dir / "stage3b_ablation_summary.csv"
    md_path = output_dir / "stage3b_ablation_summary.md"
    aggregate_md_path = output_dir / "stage3b_ablation_aggregate.md"
    aggregate_rows = summarize_by_mode(rows, args.modes)
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    write_aggregate_markdown(aggregate_rows, aggregate_md_path)

    print(f"\n[ablation_runner] Wrote CSV summary: {csv_path}")
    print(f"[ablation_runner] Wrote Markdown summary: {md_path}")
    print(f"[ablation_runner] Wrote aggregate Markdown summary: {aggregate_md_path}")
    print("\n[ablation_runner] Aggregate summary:")
    print(aggregate_md_path.read_text(encoding="utf-8"))
    print("\n[ablation_runner] Per-run summary:")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
