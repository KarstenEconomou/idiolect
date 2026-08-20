"""Define the model training port."""

from typing import Protocol

from idiolect.config import TrainConfig
from idiolect.types import DatasetRef, RunRef


class Trainer(Protocol):
    """Train one model adapter."""

    def train(self, dataset: DatasetRef, config: TrainConfig) -> RunRef:
        """Train and return one model run."""
        ...
