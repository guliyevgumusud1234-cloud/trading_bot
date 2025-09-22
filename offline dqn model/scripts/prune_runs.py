#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def prune_runs(root: Path, keep: int) -> None:
    if not root.exists():
        return
    experiments = [p for p in root.iterdir() if p.is_dir()]
    for exp in experiments:
        runs = sorted([p for p in exp.iterdir() if p.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        for old in runs[keep:]:
            print(f"Removing {old}")
            shutil.rmtree(old, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove old run directories, keeping the most recent ones")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--keep", type=int, default=5)
    args = parser.parse_args()

    prune_runs(Path(args.runs_dir), max(1, args.keep))


if __name__ == "__main__":
    main()
