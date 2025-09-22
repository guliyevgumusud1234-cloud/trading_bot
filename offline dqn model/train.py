from __future__ import annotations

import argparse
from pathlib import Path

from trader.config import load_config
from trader.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN trading agent")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)
    result = run_training(config, config_path)
    print("Training finished. Run directory:", result["run_dir"])


if __name__ == "__main__":
    main()
