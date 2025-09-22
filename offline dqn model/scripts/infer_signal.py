#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trader.config import load_config
from trader.features import build_feature_matrix
from trader.model import QNetwork
from trader.utils import resolve_device


def prepare_input(csv_path: Path, config, feature_names, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
    feature_df, prices, _ = build_feature_matrix(df, config.data)
    feature_df = feature_df[feature_names]
    features = feature_df.to_numpy(dtype=np.float32)
    features = (features - mean) / (std + 1e-8)
    return features, prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer next action from trained checkpoint")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True, help="CSV with latest market data")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--position", type=float, default=0.0, help="Current position value to append (e.g. -1,0,1)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    cfg = load_config(Path(args.config))
    checkpoint = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)

    feature_names = checkpoint.get("feature_names")
    if feature_names is None:
        raise ValueError("Checkpoint missing feature_names")
    mean = np.asarray(checkpoint.get("scaler_mean"))
    std = np.asarray(checkpoint.get("scaler_std"))
    if mean.size == 0 or std.size == 0:
        raise ValueError("Checkpoint missing scaler statistics")
    mean = mean.reshape(-1)
    std = std.reshape(-1)

    features, prices = prepare_input(Path(args.csv), cfg, feature_names, mean, std)
    window = checkpoint.get("window", cfg.data.window)
    if len(features) <= window:
        raise ValueError("Not enough rows for specified window")
    obs_features = features[-window:]
    position = np.array([args.position], dtype=np.float32)
    obs = np.concatenate([obs_features.reshape(-1), position], axis=0)

    model = QNetwork(
        input_dim=window * len(feature_names) + 1,
        hidden_sizes=cfg.model.hidden_sizes,
        num_actions=len(cfg.environment.action_levels),
        dueling=cfg.model.dueling,
        dropout=cfg.model.dropout,
        use_gru=cfg.model.use_gru,
        window=window,
        feature_dim=len(feature_names),
        gru_hidden_size=cfg.model.gru_hidden_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        q_values = model(obs_tensor)
        action_idx = int(torch.argmax(q_values, dim=1).item())
        action = cfg.environment.action_levels[action_idx]

    print("Action:", action)
    print("Q-values:", q_values.cpu().numpy().tolist())


if __name__ == "__main__":
    main()
