"""Define model training contracts."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from idiolect.config import TrainConfig, TrainDataConfig
from idiolect.model import ModelSpec
from idiolect.types import DatasetRef, RunRef, TrainResult


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """Keep one verified training run and its fixed policy."""

    ref: RunRef
    model: ModelSpec
    model_digest: str
    data: TrainDataConfig
    adapter_path: Path
    adapter_digest: str
    policy: Mapping[str, Any]
    seed: int
    max_seq_length: int


class Trainer(Protocol):
    """Train one model adapter."""

    def train(self, dataset: DatasetRef, config: TrainConfig) -> TrainResult:
        """Train and return the configured model runs."""
        ...
