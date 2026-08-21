"""Define local model evaluation contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from idiolect.config import EvalConfig, InferConfig
from idiolect.infer.base import ModelTarget
from idiolect.prompt import ModelInput
from idiolect.train.base import LoadedRun
from idiolect.types import DatasetRef, EvaluationRef


@dataclass(frozen=True, slots=True)
class CompletionScore:
    """Keep one completion likelihood result."""

    prompt_tokens: int
    tokens: int
    negative_log_likelihood: float


class ScoreSession(Protocol):
    """Score completions with one loaded model."""

    def score(self, prompt: ModelInput, completion: str) -> CompletionScore:
        """Return the likelihood cost for one completion."""
        ...

    def close(self) -> None:
        """Release the loaded model."""
        ...


class ScoreBackend(Protocol):
    """Load one local completion scoring backend."""

    @property
    def version(self) -> str:
        """Return the backend version."""
        ...

    def load(self, target: ModelTarget) -> ScoreSession:
        """Load one verified target and return its scoring session."""
        ...


class Evaluator(Protocol):
    """Compare one complete training policy with its base."""

    def evaluate(
        self,
        runs: Sequence[LoadedRun],
        dataset: DatasetRef,
        config: EvalConfig,
        infer: InferConfig,
    ) -> EvaluationRef:
        """Evaluate one policy and return its fixed artifact."""
        ...
