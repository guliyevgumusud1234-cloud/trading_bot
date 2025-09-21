from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
import yaml

from trader.config import DataConfig, DataSplits, load_config
from trader.env import make_environment
from trader.evaluation import evaluate_policy
from trader.features import build_feature_matrix, load_dataframe
from trader.model import QNetwork
from trader.utils import resolve_device

BINANCE_ENDPOINT = "https://api.binance.com/api/v3/klines"
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "DOTUSDT",
]


def fetch_symbol_klines(symbol: str, interval: str = "1h", limit: int = 720) -> pd.DataFrame:
    url = f"{BINANCE_ENDPOINT}?symbol={symbol}&interval={interval}&limit={limit}"
    with urlopen(url) as resp:
        data = json.load(resp)
    if not data:
        raise RuntimeError(f"No data returned for {symbol}")
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(data, columns=columns)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    out = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
        }
    )
    out.sort_values("timestamp", inplace=True)
    return out


def ensure_symbol_csv(symbol: str, data_dir: Path, force: bool = False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{symbol.lower()}_1h.csv"
    if not path.exists() or force:
        df = fetch_symbol_klines(symbol)
        df.to_csv(path, index=False)
    return path


def build_environment_for_symbol(
    csv_path: Path,
    base_config_path: Path,
    checkpoint_state: dict,
    device: torch.device,
):
    config = load_config(base_config_path)
    data_cfg = replace(
        config.data,
        csv_path=str(csv_path),
        splits=DataSplits(**{
            "train": config.data.splits.train,
            "val": config.data.splits.val,
            "test": config.data.splits.test,
        }),
        purge_bars=0,
    )
    df = load_dataframe(data_cfg)
    feature_df, prices, _ = build_feature_matrix(df, data_cfg)
    feature_names = checkpoint_state["feature_names"]
    if not all(name in feature_df.columns for name in feature_names):
        missing = [name for name in feature_names if name not in feature_df.columns]
        raise RuntimeError(f"Missing features in data for {csv_path.name}: {missing}")
    feature_matrix = feature_df[feature_names].to_numpy(dtype=np.float32)
    mean = checkpoint_state["scaler_mean"].astype(np.float32)
    std = checkpoint_state["scaler_std"].astype(np.float32)
    feature_matrix = (feature_matrix - mean) / (std + 1e-8)
    window = checkpoint_state.get("window", config.data.window)
    if len(feature_matrix) <= window + 1:
        raise RuntimeError(f"Not enough samples ({len(feature_matrix)}) for window {window} in {csv_path.name}")
    env = make_environment(
        split_features=feature_matrix,
        split_prices=prices,
        window=window,
        action_levels=config.environment.action_levels,
        cost_bps=config.environment.cost_bps,
        reward_scale=config.environment.reward_scale,
        random_start=False,
        max_episode_steps=len(feature_matrix) - window - 1,
        cooldown_bars=config.environment.cooldown_bars,
        max_drawdown=config.environment.max_drawdown,
    )
    model_cfg = config.model
    model = QNetwork(
        input_dim=env.observation_size,
        hidden_sizes=model_cfg.hidden_sizes,
        num_actions=len(config.environment.action_levels),
        dueling=model_cfg.dueling,
        dropout=model_cfg.dropout,
        use_gru=model_cfg.use_gru,
        window=window,
        feature_dim=feature_matrix.shape[1],
        gru_hidden_size=model_cfg.gru_hidden_size,
    ).to(device)
    model.load_state_dict(checkpoint_state["model_state_dict"])
    return env, model


def evaluate_symbols(
    symbols: Iterable[str],
    base_config_path: Path,
    checkpoint_path: Path,
    data_dir: Path,
    initial_capital: float,
    device: torch.device,
    force_download: bool = False,
):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    results = []
    for symbol in symbols:
        try:
            csv_path = ensure_symbol_csv(symbol, data_dir, force=force_download)
            env, model = build_environment_for_symbol(csv_path, base_config_path, state, device)
            eval_result = evaluate_policy(env, model, device, episodes=1)
            final_equity = initial_capital * (1.0 + eval_result.total_return)
            results.append(
                {
                    "symbol": symbol,
                    "total_return": eval_result.total_return,
                    "sharpe": eval_result.sharpe,
                    "max_drawdown": eval_result.max_drawdown,
                    "final_equity": final_equity,
                }
            )
            print(
                f"{symbol}: total_return={eval_result.total_return:.2%}, "
                f"sharpe={eval_result.sharpe:.2f}, max_dd={eval_result.max_drawdown:.2%}, "
                f"final_equity={final_equity:,.2f}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error evaluating {symbol}: {exc}", file=sys.stderr)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate symbols using trained DQN")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Base config path")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g. runs/.../model_best.pt)",
    )
    parser.add_argument("--symbols", type=str, nargs="*", default=None, help="Symbols to evaluate")
    parser.add_argument("--coins-config", type=str, default=str(Path("config/coins.yaml")), help="Coin config file")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to store downloaded CSVs")
    parser.add_argument("--initial", type=float, default=10000.0, help="Initial capital for reporting")
    parser.add_argument("--device", type=str, default="auto", help="Device to run evaluation on")
    parser.add_argument("--force", action="store_true", help="Force re-download of market data")
    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    else:
        coins_cfg_path = Path(args.coins_config)
        if coins_cfg_path.exists():
            with coins_cfg_path.open("r", encoding="utf-8") as f:
                coins_cfg = yaml.safe_load(f)
            symbols = list(coins_cfg["symbols"].get("core", []))
            symbols.extend(coins_cfg["symbols"].get("extended", []))
        else:
            symbols = DEFAULT_SYMBOLS

    device = resolve_device(args.device)
    results = evaluate_symbols(
        symbols=symbols,
        base_config_path=Path(args.config),
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        initial_capital=args.initial,
        device=device,
        force_download=args.force,
    )
    if not results:
        print("No successful evaluations")
        return
    total_final = sum(item["final_equity"] for item in results)
    avg_return = np.mean([item["total_return"] for item in results])
    print("\nSummary:")
    for item in results:
        print(
            f" - {item['symbol']}: final_equity={item['final_equity']:,.2f} (return {item['total_return']:.2%})"
        )
    print(f"Average return: {avg_return:.2%}")
    print(f"Average final equity (per symbol): {total_final / len(results):,.2f}")


if __name__ == "__main__":
    main()
