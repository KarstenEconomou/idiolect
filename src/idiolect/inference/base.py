"""Define local text generation contracts."""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from idiolect.config import GenerationConfig, InferenceConfig, TrainDataConfig
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
    prompt_throughput: float | None = None
    generation_throughput: float | None = None
    peak_memory: float | None = None


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    """Keep one streaming text delta or final generation result."""

    text: str = ""
    result: BackendResult | None = None


class Cancellation(Protocol):
    """Report whether one generation must stop."""

    def is_set(self) -> bool:
        """Return true when cancellation is requested."""
        ...


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
        config: InferenceConfig,
    ) -> BackendResult:
        """Generate one result from one formatted input."""
        ...

    def close(self) -> None:
        """Release the loaded model."""
        ...


class StreamingSession(Session, Protocol):
    """Generate streaming text with one loaded model."""

    def stream(
        self,
        value: ModelInput,
        seed: int,
        config: GenerationConfig,
        cancel: Cancellation | None = None,
    ) -> Iterator[GenerationEvent]:
        """Yield text deltas and one final result."""
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
