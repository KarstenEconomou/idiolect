"""Load and define settings for each pipeline stage."""

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from idiolect.types import ChatId


class ConfigError(ValueError):
    """Report an invalid configuration."""


@dataclass(frozen=True, slots=True)
class SignalConfig:
    """Set the Signal input options."""

    account: str | None = None
    binary: str = "signal-cli"
    data_dir: Path | None = None
    chats: tuple[ChatId, ...] = ()
    timeout: int = 5
    max_messages: int | None = None


@dataclass(frozen=True, slots=True)
class StoreConfig:
    """Set the local store paths."""

    root: Path = Path("var")
    database: str = "idiolect.duckdb"

    @property
    def database_path(self) -> Path:
        """Return the full database path."""
        path = Path(self.database)
        return path if path.is_absolute() else self.root / path


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


def load_config(
    path: Path,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load settings from one TOML file and the environment."""
    try:
        with path.open("rb") as stream:
            values = tomllib.load(stream)
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file does not exist: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"Configuration file is not valid: {error}") from error

    env = os.environ if environ is None else environ
    signal_values = _section(values, "signal")
    store_values = _section(values, "store")
    data_values = _section(values, "data")
    train_values = _section(values, "train")
    eval_values = _section(values, "eval")
    infer_values = _section(values, "infer")
    _check_keys(values, {"signal", "store", "data", "train", "eval", "infer"}, "root")
    _check_keys(
        signal_values,
        {"account", "binary", "data_dir", "chats", "timeout", "max_messages"},
        "signal",
    )
    _check_keys(store_values, {"root", "database"}, "store")
    _check_keys(data_values, {"context", "valid_ratio", "test_ratio"}, "data")
    _check_keys(
        train_values,
        {"base_model", "batch_size", "epochs", "learning_rate"},
        "train",
    )
    _check_keys(eval_values, {"max_examples"}, "eval")
    _check_keys(infer_values, {"max_tokens", "temperature"}, "infer")

    signal = SignalConfig(
        account=env.get("IDIOLECT_SIGNAL_ACCOUNT") or _optional_str(signal_values, "account"),
        binary=env.get("IDIOLECT_SIGNAL_BIN") or _str(signal_values, "binary", "signal-cli"),
        data_dir=_optional_path(
            env.get("IDIOLECT_SIGNAL_DATA_DIR") or signal_values.get("data_dir")
        ),
        chats=tuple(ChatId(value) for value in _str_list(signal_values, "chats")),
        timeout=_int(signal_values, "timeout", 5),
        max_messages=_optional_int(signal_values, "max_messages"),
    )
    store = StoreConfig(
        root=Path(_str(store_values, "root", "var")),
        database=_str(store_values, "database", "idiolect.duckdb"),
    )
    data = DataConfig(
        context=_int(data_values, "context", 32),
        valid_ratio=_float(data_values, "valid_ratio", 0.1),
        test_ratio=_float(data_values, "test_ratio", 0.1),
    )
    train = TrainConfig(
        base_model=_str(train_values, "base_model", ""),
        batch_size=_int(train_values, "batch_size", 1),
        epochs=_int(train_values, "epochs", 1),
        learning_rate=_float(train_values, "learning_rate", 0.0002),
    )
    eval_config = EvalConfig(max_examples=_optional_int(eval_values, "max_examples"))
    infer = InferConfig(
        max_tokens=_int(infer_values, "max_tokens", 128),
        temperature=_float(infer_values, "temperature", 0.7),
    )
    _validate_signal(signal)
    return AppConfig(signal, store, data, train, eval_config, infer)


def _section(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = values.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration section must be a table: {name}")
    return value


def _check_keys(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigError(f"Configuration section {section} has unknown values: {names}")


def _str(values: Mapping[str, Any], name: str, default: str) -> str:
    value = values.get(name, default)
    if not isinstance(value, str):
        raise ConfigError(f"Configuration value must be text: {name}")
    return value


def _optional_str(values: Mapping[str, Any], name: str) -> str | None:
    value = values.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"Configuration value must be text: {name}")
    return value


def _int(values: Mapping[str, Any], name: str, default: int) -> int:
    value = values.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"Configuration value must be an integer: {name}")
    return value


def _optional_int(values: Mapping[str, Any], name: str) -> int | None:
    value = values.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"Configuration value must be an integer: {name}")
    return value


def _float(values: Mapping[str, Any], name: str, default: float) -> float:
    value = values.get(name, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"Configuration value must be a number: {name}")
    return float(value)


def _str_list(values: Mapping[str, Any], name: str) -> list[str]:
    value = values.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"Configuration value must be a list of text: {name}")
    return value


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError("Signal data directory must be text")
    return Path(value)


def _validate_signal(config: SignalConfig) -> None:
    if config.timeout < -1:
        raise ConfigError("Signal timeout must be -1 or greater")
    if config.max_messages is not None and config.max_messages < 1:
        raise ConfigError("Signal max_messages must be greater than zero")
