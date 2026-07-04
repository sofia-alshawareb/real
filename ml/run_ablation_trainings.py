#!/usr/bin/env python3
"""Run all ablation training configs with limited GPU parallelism.

Uses GPU 1 by default (GPU 0 is often occupied). Adjust --gpus and --max-jobs
as needed. Skips experiments whose summary.json already exists unless --force.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = _ROOT / "configs" / "ablations"
TRAIN_SCRIPT = _ROOT / "ml" / "dino_image_classifier.py"
OUTPUT_ROOT = _ROOT / "outputs" / "dino_image_classifier"
LOG_DIR = OUTPUT_ROOT / "_ablation_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classifier ablation trainings.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[],
        help="Specific config paths (default: all configs/ablations/*.yaml).",
    )
    parser.add_argument(
        "--gpus",
        default="1",
        help="Comma-separated GPU ids to use (default: 1).",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=2,
        help="Max parallel training jobs per GPU (default: 2).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if summary.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs without launching.",
    )
    return parser.parse_args()


def experiment_name(config_path: Path) -> str:
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return str(cfg["output"]["experiment_name"])


def is_done(config_path: Path) -> bool:
    name = experiment_name(config_path)
    summary = OUTPUT_ROOT / name / "summary.json"
    return summary.is_file()


def launch(config_path: Path, gpu_id: str, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "train",
        "--config",
        str(config_path),
        "--set",
        "model.freeze_dino=true",
        "--set",
        "training.device=cuda",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(_ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def collect_results(configs: list[Path]) -> list[dict]:
    import yaml

    rows: list[dict] = []
    for cfg_path in configs:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        name = cfg["output"]["experiment_name"]
        summary_path = OUTPUT_ROOT / name / "summary.json"
        row = {
            "experiment": name,
            "config": str(cfg_path.relative_to(_ROOT)),
            "head_type": cfg["head"].get("type", "structure"),
            "pooling": cfg["head"]["pooling"],
            "num_blocks": cfg["model"]["num_blocks"],
            "status": "missing",
            "best_val_accuracy": None,
        }
        if summary_path.is_file():
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            row["status"] = "done"
            row["best_val_accuracy"] = summary.get("best_val_accuracy")
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise SystemExit("No GPUs specified")

    if args.configs:
        configs = [Path(c) if Path(c).is_absolute() else _ROOT / c for c in args.configs]
    else:
        configs = sorted(
            p for p in ABLATION_DIR.glob("*.yaml") if not p.name.startswith("_")
        )

    pending = [c for c in configs if args.force or not is_done(c)]
    skipped = [c for c in configs if c not in pending]

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Total configs: {len(configs)}")
    print(f"To run: {len(pending)} | Skip (already done): {len(skipped)}")
    print(f"GPUs: {gpu_ids} | Max parallel per GPU: {args.max_jobs}")

    if args.dry_run:
        for c in pending:
            print(f"  would run: {c.name} -> {experiment_name(c)}")
        return

    max_parallel = len(gpu_ids) * args.max_jobs
    running: list[tuple[subprocess.Popen, Path, str, Path]] = []
    queue = list(pending)
    failed: list[str] = []

    def reap_finished() -> None:
        nonlocal running
        still: list[tuple[subprocess.Popen, Path, str, Path]] = []
        for proc, cfg_path, gpu_id, log_path in running:
            code = proc.poll()
            if code is None:
                still.append((proc, cfg_path, gpu_id, log_path))
                continue
            name = experiment_name(cfg_path)
            if code == 0:
                print(f"Finished: {name} (gpu {gpu_id})")
            else:
                print(f"FAILED: {name} (gpu {gpu_id}) — see {log_path}")
                failed.append(name)
        running = still

    def gpu_slots_in_use() -> dict[str, int]:
        counts = {g: 0 for g in gpu_ids}
        for _proc, _cfg, gpu_id, _log in running:
            counts[gpu_id] += 1
        return counts

    while queue or running:
        reap_finished()
        counts = gpu_slots_in_use()

        started_any = False
        for gpu_id in gpu_ids:
            while queue and counts[gpu_id] < args.max_jobs and len(running) < max_parallel:
                cfg_path = queue.pop(0)
                name = experiment_name(cfg_path)
                log_path = LOG_DIR / f"{name}.log"
                print(f"Starting: {name} on GPU {gpu_id}")
                proc = launch(cfg_path, gpu_id, log_path)
                running.append((proc, cfg_path, gpu_id, log_path))
                counts[gpu_id] += 1
                started_any = True

        if running and (queue or started_any):
            time.sleep(15)

    results = collect_results(configs)
    results_path = OUTPUT_ROOT / "ablation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    done = [r for r in results if r["status"] == "done"]
    done.sort(key=lambda r: r["best_val_accuracy"] or 0.0, reverse=True)
    print("\n=== Ablation leaderboard (val accuracy) ===")
    for row in done:
        acc = row["best_val_accuracy"]
        print(
            f"  {acc:.4f}  {row['experiment']}  "
            f"type={row['head_type']} pool={row['pooling']} blocks={row['num_blocks']}"
        )
    print(f"\nFull results: {results_path}")

    if failed:
        raise SystemExit(f"{len(failed)} run(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
