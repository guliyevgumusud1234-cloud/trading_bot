#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from trader.config import load_config
from trader.env import make_environment
from trader.evaluation import evaluate_policy
from trader.features import prepare_datasets
from trader.model import QNetwork
from trader.utils import resolve_device


def evaluate_checkpoint(config_path: Path, checkpoint_path: Path, split: str, device: torch.device) -> dict:
    cfg = load_config(config_path)
    datasets, _, _ = prepare_datasets(cfg.data)
    data_split = datasets[split]
    env = make_environment(
        split_features=data_split.features,
        split_prices=data_split.prices,
        window=cfg.data.window,
        action_levels=cfg.environment.action_levels,
        cost_bps=cfg.environment.cost_bps,
        reward_scale=cfg.environment.reward_scale,
        random_start=False,
        max_episode_steps=len(data_split.features) - cfg.data.window - 1,
        cooldown_bars=cfg.environment.cooldown_bars,
        max_drawdown=cfg.environment.max_drawdown,
        trend_filter_abs_return=cfg.environment.trend_filter_abs_return,
        volatility_ceiling=cfg.environment.volatility_ceiling,
    )
    model = QNetwork(
        input_dim=env.observation_size,
        hidden_sizes=cfg.model.hidden_sizes,
        num_actions=len(cfg.environment.action_levels),
        dueling=cfg.model.dueling,
        dropout=cfg.model.dropout,
        use_gru=cfg.model.use_gru,
        window=cfg.data.window,
        feature_dim=data_split.features.shape[1],
        gru_hidden_size=cfg.model.gru_hidden_size,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    result = evaluate_policy(env, model, device, episodes=1)
    return {
        "checkpoint": str(checkpoint_path),
        "total_return": result.total_return,
        "sharpe": result.sharpe,
        "max_drawdown": result.max_drawdown,
    }


def discover_checkpoints(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            paths.extend(sorted(p.rglob("model_best.pt")))
        elif any(ch in pattern for ch in "*?[]"):
            paths.extend(p.parent.glob(p.name))
        else:
            paths.append(p)
    unique = []
    seen = set()
    for path in paths:
        path = path.resolve()
        if path.exists() and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and build ensemble manifest")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoints", type=str, nargs="+", help="Checkpoint paths, directories or globs")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--top", type=int, default=5, help="Top-N models by Sharpe to keep")
    parser.add_argument("--output", type=str, default="ensemble_manifest.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoints = discover_checkpoints(args.checkpoints)
    if not checkpoints:
        print("No checkpoints found")
        return

    results = []
    for ckpt in checkpoints:
        print(f"Evaluating {ckpt}")
        try:
            res = evaluate_checkpoint(Path(args.config), ckpt, args.split, device)
            results.append(res)
            print(f"  Sharpe={res['sharpe']:.3f} Return={res['total_return']:.2%}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Failed: {exc}")

    if not results:
        print("No successful evaluations")
        return

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "split": args.split,
        "top": results[: args.top],
    }
    Path(args.output).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {args.output}")


if __name__ == "__main__":
    main()
