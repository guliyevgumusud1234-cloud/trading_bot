#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
COIN_CONFIG = PROJECT_ROOT / "config" / "coins.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "market"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_LOG = LOG_DIR / "train_summary.jsonl"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def select_symbols(group: str, config: dict) -> list[str]:
    if group == "core":
        return list(config["symbols"]["core"])
    if group == "extended":
        return list(config["symbols"]["extended"])
    merged = list(config["symbols"]["core"])
    merged.extend(config["symbols"]["extended"])
    return merged


def ensure_data(symbol: str) -> Path:
    path = DATA_DIR / f"{symbol.lower()}_1h.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data missing for {symbol}: {path}")
    return path


def window_data(source: Path, days: int) -> Tuple[Path, bool]:
    if days <= 0:
        return source, False
    df = pd.read_csv(source, parse_dates=["timestamp"])
    cutoff = df["timestamp"].max() - timedelta(days=days)
    truncated = df[df["timestamp"] >= cutoff]
    if len(truncated) < len(df):
        tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix="_window.csv")
        truncated.to_csv(tmp.name, index=False)
        tmp.close()
        return Path(tmp.name), True
    return source, False


def build_symbol_config(base_cfg: dict, symbol: str, data_path: Path) -> dict:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy via yaml
    base_name = cfg.get("experiment", {}).get("name", "experiment")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = f"{symbol.lower()}_{base_name}_{timestamp}"
    cfg["experiment"]["output_dir"] = str(Path(cfg["experiment"].get("output_dir", "runs")))
    cfg.setdefault("data", {})
    cfg["data"]["csv_path"] = str(data_path)
    return cfg


def run_training(config_path: Path, log_path: Path) -> int:
    cmd = [sys.executable, str(PROJECT_ROOT / "train.py"), "--config", str(config_path)]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
        return_code = process.wait()
    return return_code


def extract_run_dir(log_path: Path) -> str | None:
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                line = line.strip()
                if line.startswith("Training finished. Run directory:"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential training loop for configured symbols")
    parser.add_argument("--config", type=str, default=str(BASE_CONFIG), help="Base config path")
    parser.add_argument("--coins", type=str, default=str(COIN_CONFIG), help="Coin config path")
    parser.add_argument("--group", choices=["core", "extended", "all"], default="core", help="Which symbol group to train")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of symbols to train (0=all)")
    parser.add_argument("--train-window-days", type=int, default=0, help="Train using only the most recent N days of data")
    args = parser.parse_args()

    base_cfg = load_yaml(Path(args.config))
    coin_cfg = load_yaml(Path(args.coins))
    symbols = select_symbols(args.group, coin_cfg)
    if args.limit > 0:
        symbols = symbols[: args.limit]
    if not symbols:
        print("No symbols selected", file=sys.stderr)
        sys.exit(1)

    summary = []
    for symbol in symbols:
        try:
            data_path = ensure_data(symbol)
        except FileNotFoundError as exc:
            summary.append((symbol, "missing", str(exc)))
            continue
        window_path, is_temp_data = window_data(data_path, args.train_window_days)
        if is_temp_data:
            print(f"[train_loop] Using last {args.train_window_days} days for {symbol}")
        cfg = build_symbol_config(base_cfg, symbol, window_path)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=f"_{symbol.lower()}.yaml") as tmp_cfg:
            save_yaml(Path(tmp_cfg.name), cfg)
            tmp_cfg_path = Path(tmp_cfg.name)
        log_path = LOG_DIR / f"train_{symbol.lower()}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.log"
        print(f"[train_loop] Training {symbol} (log: {log_path})")
        code = run_training(tmp_cfg_path, log_path)
        run_dir = extract_run_dir(log_path)
        if code == 0:
            details = str(run_dir) if run_dir else str(log_path)
            summary.append((symbol, "ok", details))
        else:
            summary.append((symbol, "error", f"exit code {code}"))
        tmp_cfg_path.unlink(missing_ok=True)
        if is_temp_data:
            window_path.unlink(missing_ok=True)

    print("[train_loop] summary:")
    for symbol, status, info in summary:
        print(f"  - {symbol}: {status} ({info})")

    try:
        import json

        with SUMMARY_LOG.open("a", encoding="utf-8") as f:
            for symbol, status, info in summary:
                record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": symbol,
                    "status": status,
                    "info": info,
                }
                f.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[train_loop] failed to write summary log: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
