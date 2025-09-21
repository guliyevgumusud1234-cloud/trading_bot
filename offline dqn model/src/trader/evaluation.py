from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .env import TradingEnv


@dataclass
class EvaluationResult:
    total_return: float
    sharpe: float
    max_drawdown: float
    equity_curve: list[float]
    avg_reward: float
    reward_vol: float


def _max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-12)
    return float(drawdown.min())


def evaluate_policy(env: TradingEnv, model: torch.nn.Module, device: torch.device, episodes: int = 1) -> EvaluationResult:
    model.eval()
    rewards_all = []
    equity_all = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        rewards = []
        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(obs_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
        equity_curve = np.array(env.summary.equity_curve, dtype=np.float32)
        equity_all.append(equity_curve)
        rewards_all.extend(rewards)
    rewards_arr = np.array(rewards_all, dtype=np.float32)
    equity_arr = equity_all[-1]
    total_return = float(equity_arr[-1] - 1.0)
    reward_vol = float(rewards_arr.std())
    annualization = math.sqrt(24 * 252)
    if reward_vol < 1e-6:
        sharpe = 0.0
        reward_vol = 0.0
    else:
        sharpe = float(rewards_arr.mean() / reward_vol * annualization)
    max_dd = _max_drawdown(equity_arr)
    return EvaluationResult(
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_dd,
        equity_curve=list(map(float, equity_arr)),
        avg_reward=float(rewards_arr.mean()),
        reward_vol=reward_vol,
    )
