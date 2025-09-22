from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _flatten_observation(features_window: np.ndarray, position: float) -> np.ndarray:
    flat = features_window.astype(np.float32).reshape(-1)
    return np.concatenate([flat, np.array([position], dtype=np.float32)], axis=0)


@dataclass
class EpisodeSummary:
    equity_curve: list[float]
    rewards: list[float]
    positions: list[float]


class TradingEnv:
    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        window: int,
        action_levels: Sequence[float],
        cost_bps: float,
        reward_scale: float = 1.0,
        random_start: bool = True,
        max_episode_steps: int | None = None,
        cooldown_bars: int = 0,
        max_drawdown: float = 1.0,
        trend_filter_abs_return: float = 0.0,
        volatility_ceiling: float = 0.0,
    ) -> None:
        if len(features) != len(prices):
            raise ValueError("Features and prices length mismatch")
        if len(features) <= window + 1:
            raise ValueError("Not enough samples for the requested window size")
        self.features = features
        self.prices = prices
        self.window = window
        self.action_levels = np.asarray(action_levels, dtype=np.float32)
        self.cost = cost_bps * 1e-4
        self.reward_scale = reward_scale
        self.random_start = random_start
        self.max_episode_steps = max_episode_steps or (len(features) - window - 1)
        self.cooldown_bars = max(0, int(cooldown_bars))
        self.max_drawdown = max_drawdown
        self.trend_filter_abs_return = max(0.0, trend_filter_abs_return)
        self.volatility_ceiling = max(0.0, volatility_ceiling)
        self._t = 0
        self._pos = 0.0
        self._done = False
        self._step_count = 0
        self._start_idx = window
        self._equity = 1.0
        self._peak_equity = 1.0
        self._cooldown_remaining = 0
        self.summary = EpisodeSummary([], [], [])

    @property
    def observation_size(self) -> int:
        feature_dim = self.features.shape[1]
        return feature_dim * self.window + 1

    def reset(self) -> np.ndarray:
        max_start = len(self.features) - self.window - 2
        if max_start <= self.window:
            start_idx = self.window
        elif self.random_start:
            start_idx = random.randint(self.window, max_start)
        else:
            start_idx = self.window
        self._start_idx = start_idx
        self._t = start_idx
        self._pos = 0.0
        self._done = False
        self._step_count = 0
        self._equity = 1.0
        self._peak_equity = 1.0
        self._cooldown_remaining = 0
        self.summary = EpisodeSummary(equity_curve=[self._equity], rewards=[], positions=[self._pos])
        return self._current_observation()

    def _current_observation(self) -> np.ndarray:
        window_slice = slice(self._t - self.window, self._t)
        window_feats = self.features[window_slice]
        return _flatten_observation(window_feats, self._pos)

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, dict]:
        if self._done:
            raise RuntimeError("step() called on finished episode")
        if action_idx < 0 or action_idx >= len(self.action_levels):
            raise ValueError("Invalid action index")
        prev_pos = self._pos
        requested_pos = float(self.action_levels[action_idx])
        if self.cooldown_bars > 0 and self._cooldown_remaining > 0 and requested_pos != prev_pos:
            new_pos = self._pos
        else:
            new_pos = requested_pos
        trade_cost = self.cost * abs(new_pos - self._pos)
        price_now = self.prices[self._t]
        price_next = self.prices[self._t + 1]
        price_return = (price_next - price_now) / (price_now + 1e-12)
        if self.trend_filter_abs_return > 0.0 and abs(price_return) < self.trend_filter_abs_return:
            new_pos = 0.0
        if self.volatility_ceiling > 0.0 and abs(price_return) > self.volatility_ceiling:
            price_return = math.copysign(self.volatility_ceiling, price_return)
        reward = new_pos * price_return - trade_cost
        scaled_reward = reward * self.reward_scale
        self._equity *= (1.0 + scaled_reward)
        self._pos = new_pos
        self._t += 1
        self._step_count += 1
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = (self._equity - self._peak_equity) / (self._peak_equity + 1e-12)
        drawdown_limit_hit = self.max_drawdown < 1.0 and drawdown < -abs(self.max_drawdown)
        done = (
            self._t >= len(self.prices) - 1
            or self._step_count >= self.max_episode_steps
            or drawdown_limit_hit
        )
        self._done = done
        if self.cooldown_bars > 0:
            if new_pos != prev_pos and self._cooldown_remaining == 0:
                self._cooldown_remaining = self.cooldown_bars
            if self._cooldown_remaining > 0:
                self._cooldown_remaining = max(0, self._cooldown_remaining - 1)
        obs = self._current_observation() if not done else np.zeros(self.observation_size, dtype=np.float32)
        info = {
            "equity": self._equity,
            "raw_reward": reward,
            "scaled_reward": scaled_reward,
            "position": self._pos,
            "price_return": price_return,
            "drawdown": drawdown,
            "cooldown": self._cooldown_remaining,
        }
        self.summary.equity_curve.append(self._equity)
        self.summary.rewards.append(float(scaled_reward))
        self.summary.positions.append(self._pos)
        return obs, float(scaled_reward), done, info

    def run_greedy_episode(self, policy) -> EpisodeSummary:
        obs = self.reset()
        done = False
        while not done:
            q_values = policy(obs)
            action = int(np.argmax(q_values))
            obs, _, done, _ = self.step(action)
        return self.summary


def make_environment(
    split_features: np.ndarray,
    split_prices: np.ndarray,
    window: int,
    action_levels: Sequence[float],
    cost_bps: float,
    reward_scale: float = 1.0,
    random_start: bool = True,
    max_episode_steps: int | None = None,
    cooldown_bars: int = 0,
    max_drawdown: float = 1.0,
    trend_filter_abs_return: float = 0.0,
    volatility_ceiling: float = 0.0,
) -> TradingEnv:
    return TradingEnv(
        features=split_features,
        prices=split_prices,
        window=window,
        action_levels=action_levels,
        cost_bps=cost_bps,
        reward_scale=reward_scale,
        random_start=random_start,
        max_episode_steps=max_episode_steps,
        cooldown_bars=cooldown_bars,
        max_drawdown=max_drawdown,
        trend_filter_abs_return=trend_filter_abs_return,
        volatility_ceiling=volatility_ceiling,
    )
