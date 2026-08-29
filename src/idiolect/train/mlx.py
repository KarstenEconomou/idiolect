"""Validate model tokens and train local adapters with MLX-LM."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from idiolect.artifact import (
    canonical_json_bytes,
    file_hashes,
    is_digest,
    write_json,
    write_jsonl,
)
from idiolect.config import TrainConfig, TrainDataConfig
from idiolect.data.local import BuildResult, DataError, load_dataset
from idiolect.model import (
    ModelError,
    ModelSpec,
    directory_digest,
    resolve_model,
    verify_model,
)
from idiolect.prompt import (
    ModelExample,
    PromptError,
    completed_turns,
    format_example,
    validate_prompt_config,
)
from idiolect.render import ChatTemplateRenderer, ModelRenderer, RenderError, Tokenizer
from idiolect.train.base import LoadedRun
from idiolect.types import DatasetId, DatasetRef, RunId, RunRef, Split, TrainResult

_RUN_VERSION = 1
_REQUIRED_TOML = frozenset(
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
        "data.format",
        "data.system_prompt",
        "data.prompt_role",
        "data.completion_role",
        "data.prompt_prefix",
        "data.prompt_suffix",
        "data.completion_prefix",
        "data.completion_suffix",
        "lora.keys",
        "lora.rank",
        "lora.scale",
        "lora.dropout",
    }
)


class TrainError(RuntimeError):
    """Report an invalid or failed training operation."""


class CommandRunner(Protocol):
    """Run one local backend command."""

    def __call__(
        self,
        command: Sequence[str],
        log_path: Path,
        /,
    ) -> int:
        """Run the command and return its exit code."""
        ...


class ModelResolver(Protocol):
    """Resolve one configured model to a local path."""

    def __call__(self, config: TrainConfig, /) -> Path:
        """Return the local model path."""
        ...


class DatasetLoader(Protocol):
    """Load and verify one immutable dataset."""

    def __call__(self, path: Path, /) -> BuildResult:
        """Return the verified dataset reference."""
        ...


class TokenizerLoader(Protocol):
    """Load one tokenizer without loading model weights."""

    def __call__(self, path: Path, trust_remote_code: bool, /) -> Tokenizer:
        """Return the tokenizer for one local model."""
        ...


class MlxTrainer:
    """Train content-addressed MLX-LM adapters."""

    def __init__(
        self,
        runner: CommandRunner | None = None,
        resolver: ModelResolver | None = None,
        tokenizer_loader: TokenizerLoader | None = None,
        loader: DatasetLoader | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Set the external boundaries for model training."""
        self._runner = _run_command if runner is None else runner
        self._resolver = _resolve_model if resolver is None else resolver
        self._tokenizer_loader = (
            _load_tokenizer if tokenizer_loader is None else tokenizer_loader
        )
        self._loader = load_dataset if loader is None else loader
        self._clock = _utc_now if clock is None else clock

    def train(self, dataset: DatasetRef, config: TrainConfig) -> TrainResult:
        """Train every configured seed and return the fixed runs."""
        _validate(config)
        model_path: Path | None = None
        try:
            model_path = self._resolver(config)
            model_digest = verify_model(model_path, _model_spec(config))
            tokenizer = self._tokenizer_loader(
                model_path,
                config.trust_remote_code,
            )
        except ModelError as error:
            raise TrainError(str(error)) from error
        except TrainError:
            raise
        except Exception as error:
            location = str(model_path) if model_path is not None else config.base_model
            raise TrainError(f"Cannot load training tokenizer: {location}") from error
        if not tokenizer.has_chat_template:
            raise TrainError("Training tokenizer does not have a chat template")
        try:
            verified = self._loader(dataset.path)
        except DataError as error:
            raise TrainError(str(error)) from error
        if verified.dataset.id != dataset.id:
            raise TrainError(
                f"Dataset identity does not match the requested dataset: {dataset.path}"
            )
        dataset_digest = directory_digest(dataset.path)
        prepared = _prepare_data(
            dataset.path,
            config,
            ChatTemplateRenderer(tokenizer),
        )
        runs = tuple(
            self._train_one(
                dataset,
                config,
                model_path,
                model_digest,
                dataset_digest,
                seed,
                prepared,
            )
            for seed in config.seeds
        )
        return TrainResult(runs)

    def _train_one(
        self,
        dataset: DatasetRef,
        config: TrainConfig,
        model_path: Path,
        model_digest: str,
        dataset_digest: str,
        seed: int,
        prepared: Mapping[Split, tuple[Mapping[str, Any], ...]],
    ) -> RunRef:
        """Train or return one content-addressed seed run."""
        recipe = _recipe(dataset, config, model_digest, dataset_digest, seed)
        run_id = RunId(hashlib.sha256(canonical_json_bytes(recipe)).hexdigest())
        output = _required_path(config.output, "Training output")
        destination = output / str(run_id)
        if destination.exists():
            return _load_run(destination, run_id, dataset)

        output.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".train-", dir=output))
        try:
            data_path = temporary / "data"
            counts = _export_data(data_path, prepared)
            adapter_path = temporary / "adapter"
            adapter_path.mkdir(mode=0o700)
            request = _request(
                config, model_path, data_path, adapter_path, seed, counts
            )
            request_path = temporary / "request.json"
            write_json(request_path, request)
            log_path = temporary / "train.log"
            command = (*config.command, "--config", str(request_path))
            exit_code = self._runner(command, log_path)
            if exit_code != 0:
                raise TrainError(f"MLX-LM training failed with exit code {exit_code}")
            artifact = adapter_path / config.adapter_file
            if not artifact.is_file():
                raise TrainError(f"MLX-LM did not create the adapter file: {artifact}")

            created_at = self._clock()
            files = file_hashes(temporary)
            manifest = {
                "run_id": str(run_id),
                "dataset_id": str(dataset.id),
                "created_at": created_at.isoformat(),
                "recipe": recipe,
                "counts": counts,
                "files": files,
            }
            write_json(temporary / "manifest.json", manifest)
            temporary.rename(destination)
            return RunRef(run_id, dataset.id, destination, created_at)
        except KeyboardInterrupt:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        except TrainError:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        except (OSError, TypeError, ValueError) as error:
            shutil.rmtree(temporary, ignore_errors=True)
            raise TrainError(f"Cannot create training run: {destination}") from error


def _validate(config: TrainConfig) -> None:
    missing = []
    if config.specified:
        missing.extend(sorted(_REQUIRED_TOML - config.specified))
        limits = {"epochs", "iterations"} & config.specified
        if len(limits) != 1:
            missing.append("exactly one of epochs or iterations")
    required_text = {
        "base_model": config.base_model,
        "model_source": config.model_source,
        "output": str(config.output) if config.output is not None else "",
        "adapter_file": config.adapter_file,
        "fine_tune_type": config.fine_tune_type,
        "optimizer": config.optimizer,
        "schedule": config.schedule,
        "train.data.format": config.data.format,
    }
    missing.extend(name for name, value in required_text.items() if not value)
    if not config.command:
        missing.append("command")
    if not config.seeds:
        missing.append("seeds")
    if config.epochs is None and config.iterations is None:
        missing.append("epochs or iterations")
    if not config.lora.keys:
        missing.append("train.lora.keys")
    positive = {
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "max_seq_length": config.max_seq_length,
        "grad_accumulation_steps": config.grad_accumulation_steps,
        "steps_per_report": config.steps_per_report,
        "steps_per_eval": config.steps_per_eval,
        "save_every": config.save_every,
        "train.lora.rank": config.lora.rank,
        "train.lora.scale": config.lora.scale,
    }
    missing.extend(name for name, value in positive.items() if value <= 0)
    if missing:
        raise TrainError(f"Training configuration is incomplete: {', '.join(missing)}")
    if config.model_source not in {"hub", "path"}:
        raise TrainError("Training model_source must be hub or path")
    if config.model_source == "hub" and not config.model_revision:
        raise TrainError("Hub models require model_revision")
    if config.model_source == "hub" and config.model_cache is None:
        raise TrainError("Hub models require model_cache")
    if config.num_layers == 0 or config.num_layers < -1:
        raise TrainError("Training num_layers must be -1 or greater than zero")
    if config.fine_tune_type != "lora":
        raise TrainError("This training backend requires fine_tune_type = lora")
    if config.schedule != "constant":
        raise TrainError("This training backend supports schedule = constant")
    try:
        validate_prompt_config(config.data)
    except PromptError as error:
        raise TrainError(str(error)) from error
    if config.val_batches == 0 or config.val_batches < -1:
        raise TrainError("Training val_batches must be -1 or greater than zero")
    if config.test_batches == 0 or config.test_batches < -1:
        raise TrainError("Training test_batches must be -1 or greater than zero")
    if config.clear_cache_threshold < 0:
        raise TrainError("Training clear_cache_threshold must not be negative")
    if not config.mask_prompt:
        raise TrainError("Training must mask the prompt for target-only supervision")


def _resolve_model(config: TrainConfig) -> Path:
    try:
        return resolve_model(_model_spec(config))
    except ModelError as error:
        raise TrainError(str(error)) from error


def _load_tokenizer(path: Path, trust_remote_code: bool) -> Tokenizer:
    try:
        from mlx_lm.utils import load_tokenizer
    except ImportError as error:
        raise TrainError(
            "Training packages are not installed. Run: uv sync --extra train"
        ) from error
    return load_tokenizer(
        path,
        tokenizer_config_extra={"trust_remote_code": trust_remote_code},
    )


def _model_spec(config: TrainConfig) -> ModelSpec:
    return ModelSpec(
        config.base_model,
        config.model_source,
        config.model_revision,
        config.model_cache,
        config.trust_remote_code,
    )


def _run_command(command: Sequence[str], log_path: Path) -> int:
    with log_path.open("xb") as log:
        os.chmod(log_path, 0o600)
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def _prepare_data(
    source: Path,
    config: TrainConfig,
    renderer: ModelRenderer,
) -> Mapping[Split, tuple[Mapping[str, Any], ...]]:
    prepared: dict[Split, tuple[Mapping[str, Any], ...]] = {}
    for split in Split:
        source_path = source / f"{split.value}.jsonl"
        if not source_path.exists():
            continue
        rows = []
        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                value = json.loads(line)
                prompt = value["prompt"]
                completion = value["completion"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise TrainError(
                    f"Dataset row is not valid: {source_path}:{line_number}"
                ) from error
            if not isinstance(prompt, str) or not isinstance(completion, str):
                raise TrainError(
                    f"Dataset row text is not valid: {source_path}:{line_number}"
                )
            example = format_example(prompt, completion, config.data)
            try:
                rendered = renderer.render_example(example)
            except RenderError as error:
                raise TrainError(f"{error}: {source_path}:{line_number}") from error
            if len(rendered.token_ids) > config.max_seq_length:
                raise TrainError(
                    "Dataset row exceeds max_seq_length at "
                    f"{source_path}:{line_number}: {len(rendered.token_ids)} > "
                    f"{config.max_seq_length}"
                )
            row = _mlx_lm_row(example, config.data)
            rows.append(row)
        prepared[split] = tuple(rows)
    if len(prepared.get(Split.TRAIN, ())) == 0:
        raise TrainError("Dataset does not contain training examples")
    if len(prepared[Split.TRAIN]) < config.batch_size:
        raise TrainError("Dataset has fewer training examples than batch_size")
    if len(prepared.get(Split.VALID, ())) == 0:
        raise TrainError("Dataset does not contain validation examples")
    if config.test and len(prepared.get(Split.TEST, ())) == 0:
        raise TrainError("Dataset does not contain test examples")
    return prepared


def _export_data(
    destination: Path,
    prepared: Mapping[Split, tuple[Mapping[str, Any], ...]],
) -> dict[str, int]:
    destination.mkdir(mode=0o700)
    counts = {}
    for split in Split:
        rows = prepared.get(split)
        if rows is None:
            continue
        write_jsonl(destination / f"{split.value}.jsonl", rows)
        counts[split.value] = len(rows)
    return counts


def _mlx_lm_row(
    example: ModelExample,
    config: TrainDataConfig,
) -> Mapping[str, Any]:
    """Return one private MLX-LM training row."""
    turns = completed_turns(example)
    if config.format == "completion":
        return {
            "prompt": turns[0].content,
            "completion": turns[-1].content,
        }
    return {
        "messages": [{"role": turn.role, "content": turn.content} for turn in turns]
    }


def _request(
    config: TrainConfig,
    model_path: Path,
    data_path: Path,
    adapter_path: Path,
    seed: int,
    counts: Mapping[str, int],
) -> Mapping[str, Any]:
    iterations = config.iterations
    if iterations is None:
        epochs = config.epochs
        if epochs is None:
            raise TrainError("Training epochs are not configured")
        iterations = math.ceil(counts[Split.TRAIN.value] / config.batch_size) * epochs
    return {
        "model": str(model_path),
        "train": True,
        "fine_tune_type": config.fine_tune_type,
        "optimizer": config.optimizer,
        "optimizer_config": {config.optimizer: dict(config.optimizer_options)},
        "data": str(data_path),
        "seed": seed,
        "num_layers": config.num_layers,
        "batch_size": config.batch_size,
        "iters": iterations,
        "val_batches": config.val_batches,
        "learning_rate": config.learning_rate,
        "steps_per_report": config.steps_per_report,
        "steps_per_eval": config.steps_per_eval,
        "grad_accumulation_steps": config.grad_accumulation_steps,
        "resume_adapter_file": None,
        "adapter_path": str(adapter_path),
        "save_every": config.save_every,
        "test": config.test,
        "test_batches": config.test_batches,
        "max_seq_length": config.max_seq_length,
        "grad_checkpoint": config.grad_checkpoint,
        "clear_cache_threshold": config.clear_cache_threshold,
        "mask_prompt": config.mask_prompt,
        "report_to": config.report_to or None,
        "project_name": config.project_name or None,
        "lr_schedule": None,
        "lora_parameters": asdict(config.lora),
    }


def _recipe(
    dataset: DatasetRef,
    config: TrainConfig,
    model_digest: str,
    dataset_digest: str,
    seed: int,
) -> Mapping[str, Any]:
    # One run's identity must depend only on that run's own inputs, so the
    # policy records only the seed of this run.
    policy = dict(training_policy(config))
    policy["seeds"] = [seed]
    return {
        "version": _RUN_VERSION,
        "backend": "mlx-lm",
        "dataset_id": str(dataset.id),
        "dataset_digest": dataset_digest,
        "model_digest": model_digest,
        "seed": seed,
        "config": policy,
    }


def training_policy(config: TrainConfig) -> Mapping[str, Any]:
    """Return the recorded value for one training policy."""
    value = asdict(config)
    value.pop("specified")
    value["model_cache"] = str(config.model_cache) if config.model_cache else None
    value["output"] = str(config.output) if config.output else None
    value["optimizer_options"] = dict(config.optimizer_options)
    return json.loads(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )


def _load_run(path: Path, run_id: RunId, dataset: DatasetRef) -> RunRef:
    run = load_run(path)
    if run.ref.id != run_id:
        raise TrainError(f"Run manifest does not match its path: {path}")
    if run.ref.dataset_id != dataset.id:
        raise TrainError(f"Run manifest does not match the dataset: {path}")
    return run.ref


def load_run(path: Path) -> LoadedRun:
    """Load and verify one fixed training run."""
    try:
        run_id = RunId(path.name)
        if not is_digest(path.name):
            raise TrainError(f"Run path does not contain an ID: {path}")
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if value["run_id"] != str(run_id):
            raise TrainError(f"Run manifest does not match its path: {path}")
        actual_id = hashlib.sha256(canonical_json_bytes(value["recipe"])).hexdigest()
        if actual_id != str(run_id):
            raise TrainError(f"Run recipe does not match its ID: {path}")
        files = value["files"]
        if not isinstance(files, dict):
            raise TypeError
        actual_names = {
            item.relative_to(path).as_posix()
            for item in path.rglob("*")
            if item.is_file() and item != path / "manifest.json"
        }
        if actual_names != set(files):
            raise TrainError(f"Run files do not match its manifest: {path}")
        for name, expected in files.items():
            file_path = _run_file(path, name)
            actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if actual != expected:
                raise TrainError(f"Run file does not match its manifest: {file_path}")
        created_at = datetime.fromisoformat(value["created_at"])
        dataset_id = value["dataset_id"]
        if not isinstance(dataset_id, str):
            raise TypeError
        recipe = value["recipe"]
        config = recipe["config"]
        data = config["data"]
        model_cache = config["model_cache"]
        if model_cache is not None and not isinstance(model_cache, str):
            raise TypeError
        model = ModelSpec(
            _manifest_text(config, "base_model"),
            _manifest_text(config, "model_source"),
            _manifest_text(config, "model_revision"),
            Path(model_cache) if model_cache is not None else None,
            _manifest_bool(config, "trust_remote_code"),
        )
        data_config = _run_data_config(data)
        model_digest = _manifest_text(recipe, "model_digest")
        adapter_path = path / "adapter"
        adapter_name = (
            Path("adapter") / _manifest_text(config, "adapter_file")
        ).as_posix()
        _run_file(path, adapter_name)
        adapter_digest = directory_digest(adapter_path)
        seed = recipe["seed"]
        max_seq_length = config["max_seq_length"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError
        if not isinstance(max_seq_length, int) or isinstance(max_seq_length, bool):
            raise TypeError
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise TrainError(f"Cannot read existing run: {path}") from error
    ref = RunRef(run_id, DatasetId(dataset_id), path, created_at)
    return LoadedRun(
        ref,
        model,
        model_digest,
        data_config,
        adapter_path,
        adapter_digest,
        config,
        seed,
        max_seq_length,
    )


def _run_file(root: Path, name: object) -> Path:
    if not isinstance(name, str):
        raise TypeError
    root_path = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root_path) or not path.is_file():
        raise TrainError(f"Run manifest contains an invalid file path: {name}")
    return path


def _manifest_text(value: object, name: str) -> str:
    if not isinstance(value, dict) or not isinstance(value.get(name), str):
        raise TypeError
    return value[name]


def _manifest_bool(value: object, name: str) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get(name), bool):
        raise TypeError
    return value[name]


def _run_data_config(value: object) -> TrainDataConfig:
    if not isinstance(value, dict):
        raise TypeError
    names = (
        "format",
        "system_prompt",
        "prompt_role",
        "completion_role",
        "prompt_prefix",
        "prompt_suffix",
        "completion_prefix",
        "completion_suffix",
    )
    values = {name: _manifest_text(value, name) for name in names}
    return TrainDataConfig(**values)


def _required_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise TrainError(f"{name} is not configured")
    return path.expanduser().resolve()


def _utc_now() -> datetime:
    return datetime.now(UTC)
