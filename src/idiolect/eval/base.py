"""Define the model test port."""

from collections.abc import Sequence
from typing import Protocol

from idiolect.config import EvalConfig
from idiolect.types import DatasetRef, Metric, RunRef


class Evaluator(Protocol):
    """Test one trained model adapter."""

    def evaluate(
        self,
        run: RunRef,
        dataset: DatasetRef,
        config: EvalConfig,
    ) -> Sequence[Metric]:
        """Test one run and return its results."""
        ...
