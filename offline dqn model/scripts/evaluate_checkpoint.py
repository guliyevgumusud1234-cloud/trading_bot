from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trader.config import load_config
from trader.env import make_environment
from trader.evaluation import evaluate_policy
from trader.features import prepare_datasets
from trader.model import QNetwork
from trader.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g. runs/.../checkpoints/model_best.pt)",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--episodes", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device to run evaluation on (cpu/cuda/auto)")
    return parser.parse_args()


def resolve_window(datasets, desired_window: int) -> int:
    min_len = min(len(split.features) for split in datasets.values())
    effective = min(desired_window, max(2, min_len - 2))
    if effective < 2:
        raise ValueError("Dataset too small for evaluation; check your data splits or window size")
    return effective


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    datasets, scaler, feature_names = prepare_datasets(config.data)
    window = resolve_window(datasets, config.data.window)
    if window != config.data.window:
        print(f"Adjusted window from {config.data.window} to {window} for evaluation")

    split = datasets[args.split]
    if len(split.features) <= window + 1:
        raise ValueError("Selected split is too short for the evaluation window; adjust configuration")

    device = resolve_device(args.device)
    num_actions = len(config.environment.action_levels)

    env = make_environment(
        split_features=split.features,
        split_prices=split.prices,
        window=window,
        action_levels=config.environment.action_levels,
        cost_bps=config.environment.cost_bps,
        reward_scale=config.environment.reward_scale,
        random_start=False,
        max_episode_steps=len(split.features) - window - 1,
        cooldown_bars=config.environment.cooldown_bars,
        max_drawdown=config.environment.max_drawdown,
    )

    model = QNetwork(
        input_dim=env.observation_size,
        hidden_sizes=config.model.hidden_sizes,
        num_actions=num_actions,
        dueling=config.model.dueling,
        dropout=config.model.dropout,
        use_gru=config.model.use_gru,
        window=window,
        feature_dim=split.features.shape[1],
        gru_hidden_size=config.model.gru_hidden_size,
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    result = evaluate_policy(env, model, device, episodes=args.episodes)
    print("Split:", args.split)
    print("Checkpoint:", checkpoint_path)
    print(f"Episodes: {args.episodes}")
    print(f"Total return: {result.total_return:.4%}")
    print(f"Sharpe: {result.sharpe:.4f}")
    print(f"Max drawdown: {result.max_drawdown:.4%}")
    print(f"Average reward: {result.avg_reward:.6f}")
    print(f"Reward volatility: {result.reward_vol:.6f}")


if __name__ == "__main__":
    main()
