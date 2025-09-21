from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

import torch


@dataclass(order=True)
class _CheckpointRecord:
    metric: float
    path: Path = field(compare=False)


class CheckpointManager:
    def __init__(self, directory: Path, top_k: int) -> None:
        self.directory = directory
        self.top_k = top_k
        self.directory.mkdir(parents=True, exist_ok=True)
        self._best: List[_CheckpointRecord] = []

    def save(self, name: str, state: dict, metric: float | None = None) -> Path:
        path = self.directory / f"{name}.pt"
        torch.save(state, path)
        if metric is not None and self.top_k > 0:
            heapq.heappush(self._best, _CheckpointRecord(metric, path))
            if len(self._best) > self.top_k:
                removed = heapq.heappop(self._best)
                if removed.path.exists() and removed.path != path:
                    removed.path.unlink()
        return path

    def best_checkpoint(self) -> Path | None:
        if not self._best:
            return None
        return max(self._best).path

    def export_metadata(self, path: Path) -> None:
        data: List[Tuple[float, str]] = [(rec.metric, rec.path.name) for rec in sorted(self._best)]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
