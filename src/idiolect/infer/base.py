"""Define local text generation contracts."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from idiolect.config import InferConfig, TrainDataConfig
from idiolect.prompt import ModelInput


class TargetMode(StrEnum):
    """Name one inference target mode."""

    CONFIG_BASE = "config-base"
    RUN_BASE = "run-base"
    RUN_ADAPTER = "run-adapter"


@dataclass(frozen=True, slots=True)
class ModelTarget:
    """Identify one verified model target."""

    id: str
    mode: TargetMode
    model_path: Path
    model_digest: str
    data: TrainDataConfig
    trust_remote_code: bool = False
    adapter_path: Path | None = None
    adapter_digest: str | None = None
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class BackendResult:
    """Keep one result from a generation backend."""

    text: str
    finish_reason: str
    prompt_tokens: int
    generated_tokens: int


@dataclass(frozen=True, slots=True)
class Prediction:
    """Keep one generated prediction."""

    example_id: str
    index: int
    seed: int
    rng_seed: int
    text: str
    finish_reason: str
    prompt_tokens: int
    generated_tokens: int


class Session(Protocol):
    """Generate text with one loaded model."""

    def count_tokens(self, value: ModelInput) -> int:
        """Return the formatted prompt token count."""
        ...

    def generate(
        self,
        value: ModelInput,
        seed: int,
        config: InferConfig,
    ) -> BackendResult:
        """Generate one result from one formatted input."""
        ...

    def close(self) -> None:
        """Release the loaded model."""
        ...


class Backend(Protocol):
    """Load one local generation backend."""

    @property
    def version(self) -> str:
        """Return the backend version."""
        ...

    def load(self, target: ModelTarget) -> Session:
        """Load one verified target and return its session."""
        ...
