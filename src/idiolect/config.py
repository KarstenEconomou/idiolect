"""Load and define settings for each pipeline stage."""

import json
import math
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

    output: Path | None = None
    context: int = 32
    valid_ratio: float = 0.1
    test_ratio: float = 0.1


type ConfigScalar = str | int | float | bool
type ConfigValue = ConfigScalar | tuple[ConfigScalar, ...]


@dataclass(frozen=True, slots=True)
class TrainDataConfig:
    """Set the training data export options."""

    format: str = ""
    system_prompt: str = ""
    prompt_role: str = ""
    completion_role: str = ""
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    completion_prefix: str = ""
    completion_suffix: str = ""


@dataclass(frozen=True, slots=True)
class LoraConfig:
    """Set the low-rank adapter options."""

    keys: tuple[str, ...] = ()
    rank: int = 0
    scale: float = 0.0
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Set the model training options."""

    base_model: str = ""
    model_source: str = ""
    model_revision: str = ""
    model_cache: Path | None = None
    output: Path | None = None
    adapter_file: str = ""
    command: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    fine_tune_type: str = ""
    optimizer: str = ""
    optimizer_options: tuple[tuple[str, ConfigValue], ...] = ()
    batch_size: int = 1
    epochs: int | None = None
    iterations: int | None = None
    learning_rate: float = 0.0002
    num_layers: int = 0
    val_batches: int = 0
    test: bool = False
    test_batches: int = 0
    max_seq_length: int = 0
    grad_checkpoint: bool = False
    grad_accumulation_steps: int = 0
    clear_cache_threshold: int = 0
    steps_per_report: int = 0
    steps_per_eval: int = 0
    save_every: int = 0
    mask_prompt: bool = False
    trust_remote_code: bool = False
    schedule: str = ""
    report_to: str = ""
    project_name: str = ""
    data: TrainDataConfig = field(default_factory=TrainDataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    specified: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Set the model test options."""

    output: Path | None = None
    backend: str = ""
    suite: str = ""
    split: str = ""
    max_examples: int = 0
    bootstrap_seed: int = 0
    bootstrap_samples: int = 0
    confidence_level: float = 0.0
    long_match_chars: int = 0
    max_empty_rate: float = 0.0
    max_format_violation_rate: float = 0.0
    max_truncation_rate: float = 0.0
    max_memorization_rate_delta: float = 0.0
    ballot_seed: int = 0
    ballots_per_rater: int = 0
    control_fraction: float = 0.0
    min_panel_raters: int = 0
    min_primary_comparisons: int = 0
    specified: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Set generation-only model options."""

    backend: str = ""
    max_prompt_tokens: int = 0
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    repetition_penalty: float = 1.0
    repetition_context_size: int = 20


@dataclass(frozen=True, slots=True)
class InferConfig(GenerationConfig):
    """Set batch inference and generation options."""

    output: Path | None = None
    seeds: tuple[int, ...] = ()
    max_examples: int = 0
    specified: frozenset[str] = frozenset()

    @property
    def generation(self) -> GenerationConfig:
        """Return the generation-only part of this policy."""
        return GenerationConfig(
            backend=self.backend,
            max_prompt_tokens=self.max_prompt_tokens,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            min_tokens_to_keep=self.min_tokens_to_keep,
            repetition_penalty=self.repetition_penalty,
            repetition_context_size=self.repetition_context_size,
        )


@dataclass(frozen=True, slots=True)
class ChatConfig:
    """Set the local interactive chat policy."""

    output: Path | None = None
    seed: int = 0
    participant_name: str = ""
    context_policy: str = ""
    history: str = ""
    specified: frozenset[str] = frozenset()
    unknown: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Group all Idiolect settings."""

    signal: SignalConfig = field(default_factory=SignalConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)


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
    chat_values = _section(values, "chat")
    _check_keys(
        values,
        {"signal", "store", "data", "train", "eval", "infer", "chat"},
        "root",
    )
    _check_keys(
        signal_values,
        {"binary", "data_dir", "timeout", "max_messages"},
        "signal",
    )
    _check_keys(store_values, {"root", "database"}, "store")
    _check_keys(data_values, {"output", "context", "valid_ratio", "test_ratio"}, "data")
    _check_keys(
        train_values,
        {
            "base_model",
            "model_source",
            "model_revision",
            "model_cache",
            "output",
            "adapter_file",
            "command",
            "seeds",
            "fine_tune_type",
            "optimizer",
            "optimizer_options",
            "batch_size",
            "epochs",
            "iterations",
            "learning_rate",
            "num_layers",
            "val_batches",
            "test",
            "test_batches",
            "max_seq_length",
            "grad_checkpoint",
            "grad_accumulation_steps",
            "clear_cache_threshold",
            "steps_per_report",
            "steps_per_eval",
            "save_every",
            "mask_prompt",
            "trust_remote_code",
            "schedule",
            "report_to",
            "project_name",
            "data",
            "lora",
        },
        "train",
    )
    train_data_values = _section(train_values, "data")
    lora_values = _section(train_values, "lora")
    optimizer_values = _section(train_values, "optimizer_options")
    _check_keys(
        train_data_values,
        {
            "format",
            "system_prompt",
            "prompt_role",
            "completion_role",
            "prompt_prefix",
            "prompt_suffix",
            "completion_prefix",
            "completion_suffix",
        },
        "train.data",
    )
    _check_keys(lora_values, {"keys", "rank", "scale", "dropout"}, "train.lora")
    _check_keys(
        eval_values,
        {
            "output",
            "backend",
            "suite",
            "split",
            "max_examples",
            "bootstrap_seed",
            "bootstrap_samples",
            "confidence_level",
            "long_match_chars",
            "max_empty_rate",
            "max_format_violation_rate",
            "max_truncation_rate",
            "max_memorization_rate_delta",
            "ballot_seed",
            "ballots_per_rater",
            "control_fraction",
            "min_panel_raters",
            "min_primary_comparisons",
        },
        "eval",
    )
    _check_keys(
        infer_values,
        {
            "output",
            "backend",
            "seeds",
            "max_examples",
            "max_prompt_tokens",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "min_tokens_to_keep",
            "repetition_penalty",
            "repetition_context_size",
        },
        "infer",
    )

    signal = SignalConfig(
        account=env.get("IDIOLECT_SIGNAL_ACCOUNT"),
        binary=env.get("IDIOLECT_SIGNAL_BIN")
        or _str(signal_values, "binary", "signal-cli"),
        data_dir=_optional_path(
            env.get("IDIOLECT_SIGNAL_DATA_DIR") or signal_values.get("data_dir")
        ),
        chats=_signal_chats(env.get("IDIOLECT_SIGNAL_CHATS")),
        timeout=_int(signal_values, "timeout", 5),
        max_messages=_optional_int(signal_values, "max_messages"),
    )
    store = StoreConfig(
        root=Path(_str(store_values, "root", "var")),
        database=_str(store_values, "database", "idiolect.duckdb"),
    )
    data = DataConfig(
        output=_optional_path(data_values.get("output")),
        context=_int(data_values, "context", 32),
        valid_ratio=_float(data_values, "valid_ratio", 0.1),
        test_ratio=_float(data_values, "test_ratio", 0.1),
    )
    train = TrainConfig(
        base_model=_str(train_values, "base_model", ""),
        model_source=_str(train_values, "model_source", ""),
        model_revision=_str(train_values, "model_revision", ""),
        model_cache=_optional_path(train_values.get("model_cache")),
        output=_optional_path(train_values.get("output")),
        adapter_file=_str(train_values, "adapter_file", ""),
        command=tuple(_str_list(train_values, "command")),
        seeds=tuple(_int_list(train_values, "seeds")),
        fine_tune_type=_str(train_values, "fine_tune_type", ""),
        optimizer=_str(train_values, "optimizer", ""),
        optimizer_options=tuple(sorted(_scalar_map(optimizer_values).items())),
        batch_size=_int(train_values, "batch_size", 1),
        epochs=_optional_int(train_values, "epochs"),
        iterations=_optional_int(train_values, "iterations"),
        learning_rate=_float(train_values, "learning_rate", 0.0002),
        num_layers=_int(train_values, "num_layers", 0),
        val_batches=_int(train_values, "val_batches", 0),
        test=_bool(train_values, "test", False),
        test_batches=_int(train_values, "test_batches", 0),
        max_seq_length=_int(train_values, "max_seq_length", 0),
        grad_checkpoint=_bool(train_values, "grad_checkpoint", False),
        grad_accumulation_steps=_int(train_values, "grad_accumulation_steps", 0),
        clear_cache_threshold=_int(train_values, "clear_cache_threshold", 0),
        steps_per_report=_int(train_values, "steps_per_report", 0),
        steps_per_eval=_int(train_values, "steps_per_eval", 0),
        save_every=_int(train_values, "save_every", 0),
        mask_prompt=_bool(train_values, "mask_prompt", False),
        trust_remote_code=_bool(train_values, "trust_remote_code", False),
        schedule=_str(train_values, "schedule", ""),
        report_to=_str(train_values, "report_to", ""),
        project_name=_str(train_values, "project_name", ""),
        data=TrainDataConfig(
            format=_str(train_data_values, "format", ""),
            system_prompt=_str(train_data_values, "system_prompt", ""),
            prompt_role=_str(train_data_values, "prompt_role", ""),
            completion_role=_str(train_data_values, "completion_role", ""),
            prompt_prefix=_str(train_data_values, "prompt_prefix", ""),
            prompt_suffix=_str(train_data_values, "prompt_suffix", ""),
            completion_prefix=_str(train_data_values, "completion_prefix", ""),
            completion_suffix=_str(train_data_values, "completion_suffix", ""),
        ),
        lora=LoraConfig(
            keys=tuple(_str_list(lora_values, "keys")),
            rank=_int(lora_values, "rank", 0),
            scale=_float(lora_values, "scale", 0.0),
            dropout=_float(lora_values, "dropout", 0.0),
        ),
        specified=_train_keys(train_values, train_data_values, lora_values),
    )
    eval_config = EvalConfig(
        output=_optional_path(eval_values.get("output")),
        backend=_str(eval_values, "backend", ""),
        suite=_str(eval_values, "suite", ""),
        split=_str(eval_values, "split", ""),
        max_examples=_int(eval_values, "max_examples", 0),
        bootstrap_seed=_int(eval_values, "bootstrap_seed", 0),
        bootstrap_samples=_int(eval_values, "bootstrap_samples", 0),
        confidence_level=_float(eval_values, "confidence_level", 0.0),
        long_match_chars=_int(eval_values, "long_match_chars", 0),
        max_empty_rate=_float(eval_values, "max_empty_rate", 0.0),
        max_format_violation_rate=_float(eval_values, "max_format_violation_rate", 0.0),
        max_truncation_rate=_float(eval_values, "max_truncation_rate", 0.0),
        max_memorization_rate_delta=_float(
            eval_values, "max_memorization_rate_delta", 0.0
        ),
        ballot_seed=_int(eval_values, "ballot_seed", 0),
        ballots_per_rater=_int(eval_values, "ballots_per_rater", 0),
        control_fraction=_float(eval_values, "control_fraction", 0.0),
        min_panel_raters=_int(eval_values, "min_panel_raters", 0),
        min_primary_comparisons=_int(eval_values, "min_primary_comparisons", 0),
        specified=frozenset(eval_values),
    )
    infer = InferConfig(
        output=_optional_path(infer_values.get("output")),
        backend=_str(infer_values, "backend", ""),
        seeds=tuple(_int_list(infer_values, "seeds")),
        max_examples=_int(infer_values, "max_examples", 0),
        max_prompt_tokens=_int(infer_values, "max_prompt_tokens", 0),
        max_tokens=_int(infer_values, "max_tokens", 128),
        temperature=_float(infer_values, "temperature", 0.7),
        top_p=_float(infer_values, "top_p", 0.8),
        top_k=_int(infer_values, "top_k", 20),
        min_p=_float(infer_values, "min_p", 0.0),
        min_tokens_to_keep=_int(infer_values, "min_tokens_to_keep", 1),
        repetition_penalty=_float(infer_values, "repetition_penalty", 1.0),
        repetition_context_size=_int(infer_values, "repetition_context_size", 20),
        specified=frozenset(infer_values),
    )
    chat = ChatConfig(
        output=_optional_path(chat_values.get("output")),
        seed=_int(chat_values, "seed", 0),
        participant_name=_str(chat_values, "participant_name", ""),
        context_policy=_str(chat_values, "context_policy", ""),
        history=_str(chat_values, "history", ""),
        specified=frozenset(chat_values),
        unknown=frozenset(
            set(chat_values)
            - {"output", "seed", "participant_name", "context_policy", "history"}
        ),
    )
    _validate_signal(signal)
    _validate_train(train)
    _validate_eval(eval_config)
    _validate_infer(infer)
    return AppConfig(signal, store, data, train, eval_config, infer, chat)


def _section(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = values.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration section must be a table: {name}")
    return value


def _check_keys(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigError(
            f"Configuration section {section} has unknown values: {names}"
        )


def _str(values: Mapping[str, Any], name: str, default: str) -> str:
    value = values.get(name, default)
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


def _bool(values: Mapping[str, Any], name: str, default: bool) -> bool:
    value = values.get(name, default)
    if not isinstance(value, bool):
        raise ConfigError(f"Configuration value must be true or false: {name}")
    return value


def _str_list(values: Mapping[str, Any], name: str) -> list[str]:
    value = values.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"Configuration value must be a list of text: {name}")
    return value


def _signal_chats(value: str | None) -> tuple[ChatId, ...]:
    """Load the private Signal chat whitelist."""
    if value is None:
        return ()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ConfigError(
            "IDIOLECT_SIGNAL_CHATS must be a JSON list of nonempty text values"
        ) from error
    if not isinstance(decoded, list) or not all(
        isinstance(item, str) and item.strip() for item in decoded
    ):
        raise ConfigError(
            "IDIOLECT_SIGNAL_CHATS must be a JSON list of nonempty text values"
        )
    chats = decoded
    if len(chats) != len(set(chats)):
        raise ConfigError("IDIOLECT_SIGNAL_CHATS must not contain duplicate values")
    return tuple(ChatId(chat) for chat in chats)


def _int_list(values: Mapping[str, Any], name: str) -> list[int]:
    value = values.get(name, [])
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ConfigError(f"Configuration value must be a list of integers: {name}")
    return value


def _scalar_map(values: Mapping[str, Any]) -> dict[str, ConfigValue]:
    result: dict[str, ConfigValue] = {}
    for name, value in values.items():
        if isinstance(value, list) and all(
            isinstance(item, (str, int, float, bool)) for item in value
        ):
            result[name] = tuple(value)
            continue
        if not isinstance(value, (str, int, float, bool)):
            raise ConfigError(
                "Configuration value must be a scalar or scalar list: "
                f"optimizer_options.{name}"
            )
        result[name] = value
    return result


def _train_keys(
    train: Mapping[str, Any],
    data: Mapping[str, Any],
    lora: Mapping[str, Any],
) -> frozenset[str]:
    keys = {name for name in train if name not in {"data", "lora"}}
    keys.update(f"data.{name}" for name in data)
    keys.update(f"lora.{name}" for name in lora)
    return frozenset(keys)


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError("Configuration path must be text")
    return Path(value)


def _validate_signal(config: SignalConfig) -> None:
    if config.timeout < -1:
        raise ConfigError("Signal timeout must be -1 or greater")
    if config.max_messages is not None and config.max_messages < 1:
        raise ConfigError("Signal max_messages must be greater than zero")


def _validate_train(config: TrainConfig) -> None:
    if config.epochs is not None and config.epochs < 1:
        raise ConfigError("Training epochs must be greater than zero")
    if config.iterations is not None and config.iterations < 1:
        raise ConfigError("Training iterations must be greater than zero")
    if config.epochs is not None and config.iterations is not None:
        raise ConfigError("Set training epochs or iterations, not both")
    if config.seeds and len(set(config.seeds)) != len(config.seeds):
        raise ConfigError("Training seeds must be unique")
    if not 0.0 <= config.lora.dropout < 1.0:
        raise ConfigError("LoRA dropout must be at least zero and less than one")


def _validate_infer(config: InferConfig) -> None:
    if config.seeds and len(set(config.seeds)) != len(config.seeds):
        raise ConfigError("Inference seeds must be unique")
    if config.max_examples < 0:
        raise ConfigError("Inference max_examples must not be negative")
    if config.max_prompt_tokens < 0 or config.max_tokens < 1:
        raise ConfigError("Inference token limits are not valid")
    values = (
        config.temperature,
        config.top_p,
        config.min_p,
        config.repetition_penalty,
    )
    if any(not math.isfinite(value) for value in values):
        raise ConfigError("Inference sampling values must be finite")
    if config.temperature < 0:
        raise ConfigError("Inference temperature must not be negative")
    if not 0.0 <= config.top_p <= 1.0 or not 0.0 <= config.min_p <= 1.0:
        raise ConfigError("Inference probability limits must be from zero to one")
    if config.top_k < 0 or config.min_tokens_to_keep < 1:
        raise ConfigError("Inference token sampling limits are not valid")
    if config.repetition_penalty <= 0 or config.repetition_context_size < 1:
        raise ConfigError("Inference repetition settings are not valid")


def _validate_eval(config: EvalConfig) -> None:
    if config.max_examples < 0:
        raise ConfigError("Evaluation max_examples must not be negative")
    if config.bootstrap_samples < 0 or config.long_match_chars < 0:
        raise ConfigError("Evaluation sample limits must not be negative")
    if config.ballots_per_rater < 0:
        raise ConfigError("Evaluation ballots_per_rater must not be negative")
    if config.min_panel_raters < 0 or config.min_primary_comparisons < 0:
        raise ConfigError("Evaluation panel limits must not be negative")
    rates = (
        config.confidence_level,
        config.max_empty_rate,
        config.max_format_violation_rate,
        config.max_truncation_rate,
        config.max_memorization_rate_delta,
        config.control_fraction,
    )
    if any(not 0.0 <= value <= 1.0 for value in rates):
        raise ConfigError("Evaluation rates must be from zero to one")
