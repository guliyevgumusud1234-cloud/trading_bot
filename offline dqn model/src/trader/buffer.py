from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            np.stack(states).astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            np.stack(next_states).astype(np.float32),
            dones.astype(np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float, beta_start: float, beta_frames: int) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = max(beta_frames, 1)
        self.buffer: list[Tuple[np.ndarray, int, float, np.ndarray, bool]] = [None] * capacity  # type: ignore
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.frame = 0

    def __len__(self) -> int:
        return self.size

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def beta(self) -> float:
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("No elements in buffer")
        priorities = self.priorities[: self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        idxs = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in idxs]
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        beta = self.beta()
        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max() + 1e-8
        self.frame = min(self.frame + 1, self.beta_frames)
        return (
            np.stack(states).astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            np.stack(next_states).astype(np.float32),
            dones.astype(np.float32),
            weights.astype(np.float32),
            idxs.astype(np.int64),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.asarray(priorities, dtype=np.float32)
        priorities = np.power(priorities + 1e-6, 1.0)
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
