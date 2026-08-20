"""Define the model training port."""

from typing import Protocol

from idiolect.config import TrainConfig
from idiolect.types import DatasetRef, TrainResult


class Trainer(Protocol):
    """Train one model adapter."""

    def train(self, dataset: DatasetRef, config: TrainConfig) -> TrainResult:
        """Train and return the configured model runs."""
        ...
