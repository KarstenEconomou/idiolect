"""Define settings for each pipeline stage."""

from dataclasses import dataclass, field
from pathlib import Path

from idiolect.types import ChatId


@dataclass(frozen=True, slots=True)
class SignalConfig:
    """Set the Signal input options."""

    live: bool = False
    archive: Path | None = None
    chats: tuple[ChatId, ...] = ()


@dataclass(frozen=True, slots=True)
class StoreConfig:
    """Set the local store paths."""

    root: Path = Path("var")
    database: str = "idiolect.duckdb"


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Set the dataset build options."""

    context: int = 32
    valid_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Set the model training options."""

    base_model: str = ""
    batch_size: int = 1
    epochs: int = 1
    learning_rate: float = 0.0002


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Set the model test options."""

    max_examples: int | None = None


@dataclass(frozen=True, slots=True)
class InferConfig:
    """Set the text generation options."""

    max_tokens: int = 128
    temperature: float = 0.7


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Group all Idiolect settings."""

    signal: SignalConfig = field(default_factory=SignalConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    infer: InferConfig = field(default_factory=InferConfig)
