from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import DataConfig


@dataclass
class DatasetSplit:
    features: np.ndarray
    prices: np.ndarray
    timestamps: np.ndarray | None


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std


def load_dataframe(cfg: DataConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)
    if cfg.time_col and cfg.time_col in df.columns:
        df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
        df = df.sort_values(cfg.time_col)
    if cfg.price_col not in df.columns:
        raise ValueError(f"CSV must include price column '{cfg.price_col}'")
    df = df.reset_index(drop=True)
    if cfg.fill_method:
        if cfg.fill_method == "ffill":
            df = df.ffill()
        elif cfg.fill_method == "bfill":
            df = df.bfill()
        else:
            df = df.fillna(method=cfg.fill_method)
    df = df.fillna(0.0)
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df.get("high")
    low = df.get("low")
    close = df.get("close")
    if high is None or low is None or close is None:
        return pd.Series(np.zeros(len(df)), index=df.index)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff().fillna(0.0)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)


def _bollinger_bandwidth(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std().fillna(0.0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    bandwidth = (upper - lower) / (ma + 1e-12)
    return bandwidth.fillna(0.0)


def build_feature_matrix(df: pd.DataFrame, cfg: DataConfig) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    close = df[cfg.price_col].astype(float)
    ret_1 = close.pct_change().fillna(0.0)
    ret_4 = close.pct_change(4).fillna(0.0)
    ret_6 = close.pct_change(6).fillna(0.0)
    ret_24 = close.pct_change(24).fillna(0.0)
    ret_72 = close.pct_change(72).fillna(0.0)
    mom_12 = close.pct_change(12).fillna(0.0)
    mom_24 = close.pct_change(24).fillna(0.0)
    mom_72 = close.pct_change(72).fillna(0.0)
    vol_24 = ret_1.rolling(window=24, min_periods=1).std().fillna(0.0)
    vol_72 = ret_1.rolling(window=72, min_periods=1).std().fillna(0.0)
    ema_fast = _ema(close, span=20)
    ema_mid = _ema(close, span=50)
    ema_slow = _ema(close, span=100)
    ema_long = _ema(close, span=200)
    ema_ratio = (ema_fast / (ema_slow + 1e-12) - 1.0).fillna(0.0)
    ema_long_ratio = (ema_mid / (ema_long + 1e-12) - 1.0).fillna(0.0)
    atr_norm = (_atr(df, period=14) / (close + 1e-12)).fillna(0.0)
    rsi_7 = _rsi(close, period=7).fillna(50.0) / 100.0 - 0.5
    rsi_14 = _rsi(close, period=14).fillna(50.0) / 100.0 - 0.5
    rsi_21 = _rsi(close, period=21).fillna(50.0) / 100.0 - 0.5
    bb_width = _bollinger_bandwidth(close, period=20).fillna(0.0)

    volume = df.get("volume")
    if volume is not None:
        vol_z = (volume.pct_change().fillna(0.0)).rolling(window=24, min_periods=1).std().fillna(0.0)
        obv = (ret_1.pipe(np.sign).fillna(0.0) * volume.fillna(0.0)).cumsum().fillna(0.0)
        obv_norm = (obv - obv.rolling(window=72, min_periods=1).mean()).fillna(0.0)
    else:
        vol_z = pd.Series(np.zeros(len(df)), index=df.index)
        obv_norm = pd.Series(np.zeros(len(df)), index=df.index)

    trend_strength = ema_ratio.abs()
    vol_state = vol_24 / (vol_72 + 1e-8)
    regime_trend = np.tanh(trend_strength * 5.0)
    regime_vol = np.tanh(vol_state - 1.0)

    features = pd.DataFrame(
        {
            "ret_1": ret_1,
            "ret_4": ret_4,
            "ret_6": ret_6,
            "ret_24": ret_24,
            "ret_72": ret_72,
            "mom_12": mom_12,
            "mom_24": mom_24,
            "mom_72": mom_72,
            "vol_24": vol_24,
            "vol_72": vol_72,
            "ema_ratio": ema_ratio,
            "ema_long_ratio": ema_long_ratio,
            "atr_norm": atr_norm,
            "rsi_7": rsi_7,
            "rsi_14": rsi_14,
            "rsi_21": rsi_21,
            "bb_width": bb_width,
            "volume_vol": vol_z,
            "obv_norm": obv_norm,
            "regime_trend": regime_trend,
            "regime_vol": regime_vol,
        }
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    timestamps = df[cfg.time_col].to_numpy(dtype="datetime64[ns]") if cfg.time_col and cfg.time_col in df.columns else None
    return features, close.to_numpy(dtype=np.float32), timestamps


def _split_indices(n_samples: int, cfg: DataConfig) -> Dict[str, slice]:
    splits = cfg.splits
    train_len = int(n_samples * splits.train)
    val_len = int(n_samples * splits.val)
    test_len = n_samples - train_len - val_len
    if min(train_len, val_len, test_len) <= 0:
        raise ValueError("Dataset too small for requested splits")

    purge = cfg.purge_bars
    remainder = n_samples - (train_len + val_len + test_len)
    if remainder < 0:
        raise ValueError("Split proportions exceed dataset length")
    if purge > 0:
        max_purge = remainder // 2
        if purge > max_purge:
            purge = max_purge
    train_slice = slice(0, train_len)
    val_start = train_len + purge
    if val_start >= n_samples:
        val_start = max(train_len, n_samples - val_len - test_len)
    val_end = min(val_start + val_len, n_samples)
    if val_end <= val_start:
        val_start = max(train_len, n_samples - val_len - test_len)
        val_end = min(val_start + val_len, n_samples)
    test_start = val_end + purge
    if test_start >= n_samples:
        test_start = max(val_end, n_samples - test_len)
    test_end = n_samples
    if test_end - test_start <= 0:
        test_start = max(val_end, n_samples - max(test_len, 1))
    if test_end - test_start <= 0:
        raise ValueError("Dataset too small after adjustments; consider reducing window or changing splits")
    return {
        "train": train_slice,
        "val": slice(val_start, val_end),
        "test": slice(test_start, test_end),
    }


def prepare_datasets(cfg: DataConfig) -> Tuple[Dict[str, DatasetSplit], FeatureScaler, list[str]]:
    df = load_dataframe(cfg)
    feature_df, prices, timestamps = build_feature_matrix(df, cfg)
    feature_matrix = feature_df.to_numpy(dtype=np.float32)
    feature_names = list(feature_df.columns)
    n_samples = len(feature_df)
    indices = _split_indices(n_samples, cfg)

    train_slice = indices["train"]
    train_features = feature_matrix[train_slice]
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    scaler = FeatureScaler(mean=mean, std=std)

    datasets: Dict[str, DatasetSplit] = {}
    for key, idx in indices.items():
        feats = scaler.transform(feature_matrix[idx])
        pr = prices[idx]
        ts = timestamps[idx] if timestamps is not None else None
        datasets[key] = DatasetSplit(features=feats.astype(np.float32), prices=pr.astype(np.float32), timestamps=ts)
    return datasets, scaler, feature_names
