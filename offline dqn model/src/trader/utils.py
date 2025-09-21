from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_pref: str) -> torch.device:
    if device_pref == "cpu":
        return torch.device("cpu")
    if device_pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_pref)
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_scaler(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    np.savez(path, mean=mean, std=std)


def load_scaler(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["mean"], data["std"]


def rolling_window_view(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) < window:
        raise ValueError("Array shorter than window")
    shape = (len(arr) - window + 1, window) + arr.shape[1:]
    strides = (arr.strides[0],) + arr.strides
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def batch_to_device(items: Iterable[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    return [item.to(device) for item in items]
