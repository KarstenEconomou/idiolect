"""Train local adapters with MLX-LM."""

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

from idiolect.config import TrainConfig, TrainDataConfig
from idiolect.model import (
    ModelError,
    ModelSpec,
    directory_digest,
    resolve_model,
    verify_model,
)
from idiolect.prompt import format_row
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


class MlxTrainer:
    """Train content-addressed MLX-LM adapters."""

    def __init__(
        self,
        runner: CommandRunner | None = None,
        resolver: ModelResolver | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Set the external boundaries for model training."""
        self._runner = _run_command if runner is None else runner
        self._resolver = _resolve_model if resolver is None else resolver
        self._clock = _utc_now if clock is None else clock

    def train(self, dataset: DatasetRef, config: TrainConfig) -> TrainResult:
        """Train every configured seed and return the fixed runs."""
        _validate(config)
        try:
            model_path = self._resolver(config)
            model_digest = verify_model(model_path, _model_spec(config))
        except ModelError as error:
            raise TrainError(str(error)) from error
        dataset_digest = directory_digest(dataset.path)
        runs = tuple(
            self._train_one(
                dataset,
                config,
                model_path,
                model_digest,
                dataset_digest,
                seed,
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
    ) -> RunRef:
        """Train or return one content-addressed seed run."""
        recipe = _recipe(dataset, config, model_digest, dataset_digest, seed)
        run_id = RunId(hashlib.sha256(_json_bytes(recipe)).hexdigest())
        output = _required_path(config.output, "Training output")
        destination = output / str(run_id)
        if destination.exists():
            return _load_run(destination, run_id, dataset)

        output.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".train-", dir=output))
        try:
            data_path = temporary / "data"
            counts = _export_data(dataset.path, data_path, config)
            adapter_path = temporary / "adapter"
            adapter_path.mkdir(mode=0o700)
            request = _request(config, model_path, data_path, adapter_path, seed, counts)
            request_path = temporary / "request.json"
            _write_json(request_path, request)
            log_path = temporary / "train.log"
            command = (*config.command, "--config", str(request_path))
            exit_code = self._runner(command, log_path)
            if exit_code != 0:
                raise TrainError(f"MLX-LM training failed with exit code {exit_code}")
            artifact = adapter_path / config.adapter_file
            if not artifact.is_file():
                raise TrainError(f"MLX-LM did not create the adapter file: {artifact}")

            created_at = self._clock()
            files = _file_hashes(temporary)
            manifest = {
                "run_id": str(run_id),
                "dataset_id": str(dataset.id),
                "created_at": created_at.isoformat(),
                "recipe": recipe,
                "counts": counts,
                "files": files,
            }
            _write_json(temporary / "manifest.json", manifest)
            temporary.rename(destination)
            return RunRef(run_id, dataset.id, destination, created_at)
        except KeyboardInterrupt:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            TrainError,
        ) as error:
            shutil.rmtree(temporary, ignore_errors=True)
            if isinstance(error, TrainError):
                raise
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
    if config.data.format not in {"chat", "completion"}:
        raise TrainError("Training data format must be chat or completion")
    if config.data.format == "chat":
        if not config.data.prompt_role:
            missing.append("train.data.prompt_role")
        if not config.data.completion_role:
            missing.append("train.data.completion_role")
        if missing:
            raise TrainError(
                f"Training configuration is incomplete: {', '.join(missing)}"
            )
        if config.data.prompt_role not in {"system", "user", "assistant"}:
            raise TrainError("Training prompt_role is not valid")
        if config.data.completion_role not in {"system", "user", "assistant"}:
            raise TrainError("Training completion_role is not valid")
    if config.val_batches == 0 or config.val_batches < -1:
        raise TrainError("Training val_batches must be -1 or greater than zero")
    if config.test_batches == 0 or config.test_batches < -1:
        raise TrainError("Training test_batches must be -1 or greater than zero")
    if config.clear_cache_threshold < 0:
        raise TrainError("Training clear_cache_threshold must not be negative")


def _resolve_model(config: TrainConfig) -> Path:
    try:
        return resolve_model(_model_spec(config))
    except ModelError as error:
        raise TrainError(str(error)) from error


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


def _export_data(source: Path, destination: Path, config: TrainConfig) -> dict[str, int]:
    destination.mkdir(mode=0o700)
    counts: dict[str, int] = {}
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
            rows.append(format_row(prompt, completion, config.data))
        target = destination / source_path.name
        _write_jsonl(target, rows)
        counts[split.value] = len(rows)
    if counts.get(Split.TRAIN.value, 0) == 0:
        raise TrainError("Dataset does not contain training examples")
    return counts


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
    return {
        "version": _RUN_VERSION,
        "backend": "mlx-lm",
        "dataset_id": str(dataset.id),
        "dataset_digest": dataset_digest,
        "model_digest": model_digest,
        "seed": seed,
        "config": training_policy(config),
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
        if len(path.name) != 64 or any(
            character not in "0123456789abcdef" for character in path.name
        ):
            raise TrainError(f"Run path does not contain an ID: {path}")
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if value["run_id"] != str(run_id):
            raise TrainError(f"Run manifest does not match its path: {path}")
        actual_id = hashlib.sha256(_json_bytes(value["recipe"])).hexdigest()
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
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, TrainError):
            raise
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


def _file_hashes(root: Path) -> Mapping[str, str]:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.chmod(path, 0o600)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            json.dump(row, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
    os.chmod(path, 0o600)


def _required_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise TrainError(f"{name} is not configured")
    return path.expanduser().resolve()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _utc_now() -> datetime:
    return datetime.now(UTC)
