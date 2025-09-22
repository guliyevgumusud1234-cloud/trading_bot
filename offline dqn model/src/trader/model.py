from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int],
        num_actions: int,
        dueling: bool = True,
        dropout: float = 0.0,
        use_gru: bool = False,
        window: int | None = None,
        feature_dim: int | None = None,
        gru_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.use_gru = use_gru
        self.window = window
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.dueling = dueling

        layers = []
        if self.use_gru:
            if window is None or feature_dim is None:
                raise ValueError("window and feature_dim must be provided when use_gru=True")
            self.gru = nn.GRU(input_size=feature_dim, hidden_size=gru_hidden_size, batch_first=True)
            in_dim = gru_hidden_size + 1  # concat with current position
        else:
            in_dim = input_dim

        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = size
        self.feature = nn.Sequential(*layers) if layers else nn.Identity()

        if dueling:
            self.value_stream = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, 1))
            self.advantage_stream = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, num_actions))
        else:
            self.head = nn.Linear(in_dim, num_actions)

    def _split_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_gru:
            raise RuntimeError("_split_sequence should only be called when use_gru=True")
        seq_length = self.window * self.feature_dim
        seq = x[:, :seq_length].view(-1, self.window, self.feature_dim)
        pos = x[:, seq_length:].view(-1, 1)
        return seq, pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gru:
            seq, pos = self._split_sequence(x)
            gru_out, _ = self.gru(seq)
            base = torch.cat([gru_out[:, -1, :], pos], dim=1)
        else:
            base = x
        features = self.feature(base)
        if not self.dueling:
            return self.head(features)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    def act(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.forward(x)
        return q_values.argmax(dim=1)


def calculate_td_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_q_values: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    return rewards + gamma * (1.0 - dones) * next_q_values
