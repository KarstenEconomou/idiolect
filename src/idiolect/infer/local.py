"""Run private local inference operations."""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from idiolect.config import InferConfig, TrainConfig
from idiolect.data.local import load_dataset
from idiolect.infer.base import (
    Backend,
    ModelTarget,
    Prediction,
    TargetMode,
)
from idiolect.model import (
    ModelError,
    ModelSpec,
    directory_digest,
    resolve_model,
    verify_model,
)
from idiolect.prompt import format_prompt
from idiolect.train.mlx import TrainError, load_run
from idiolect.types import DatasetRef, InferenceId, InferenceRef, Split

_ARTIFACT_VERSION = 1
_REQUIRED_TOML = frozenset(
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
    }
)


class InferenceError(RuntimeError):
    """Report an invalid or failed inference operation."""


class LocalInferencer:
    """Run reproducible inference with one local backend."""

    def __init__(
        self,
        backend: Backend,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Set the backend and clock for inference."""
        self._backend = backend
        self._clock = _utc_now if clock is None else clock

    def text(
        self,
        target: ModelTarget,
        prompt: str,
        config: InferConfig,
    ) -> tuple[Prediction, ...]:
        """Generate one private prompt without an artifact."""
        _validate(config, self._backend.version)
        if not prompt.strip():
            raise InferenceError("Inference prompt must not be empty")
        example_id = hashlib.sha256(prompt.encode()).hexdigest()
        return self._generate(target, ((0, example_id, prompt),), config)

    def dataset(
        self,
        target: ModelTarget,
        dataset: DatasetRef,
        split: Split,
        config: InferConfig,
    ) -> InferenceRef:
        """Generate one fixed dataset split and return its artifact."""
        _validate(config, self._backend.version)
        verified = load_dataset(dataset.path).dataset
        rows = _dataset_rows(verified, split)
        selected = _select(rows, config.max_examples)
        dataset_digest = directory_digest(verified.path)
        recipe = _recipe(
            target,
            verified,
            dataset_digest,
            split,
            config,
            self._backend.version,
            selected,
        )
        inference_id = InferenceId(hashlib.sha256(_json_bytes(recipe)).hexdigest())
        output = _required_output(config)
        destination = output / str(inference_id)
        if destination.exists():
            return _load_artifact(destination, inference_id)

        predictions = self._generate(target, selected, config)
        created_at = self._clock()
        output.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".infer-", dir=output))
        try:
            rows_value = [_prediction_value(value) for value in predictions]
            prediction_path = temporary / "pred.jsonl"
            _write_jsonl(prediction_path, rows_value)
            manifest = {
                "inference_id": str(inference_id),
                "created_at": created_at.isoformat(),
                "recipe": recipe,
                "counts": {
                    "examples": len(selected),
                    "predictions": len(predictions),
                },
                "files": {
                    "pred.jsonl": hashlib.sha256(
                        prediction_path.read_bytes()
                    ).hexdigest()
                },
            }
            _write_json(temporary / "manifest.json", manifest)
            temporary.rename(destination)
        except KeyboardInterrupt:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        except (OSError, TypeError, ValueError) as error:
            shutil.rmtree(temporary, ignore_errors=True)
            raise InferenceError(
                f"Cannot create inference artifact: {destination}"
            ) from error
        return InferenceRef(inference_id, destination, created_at, len(predictions))

    def _generate(
        self,
        target: ModelTarget,
        examples: Sequence[tuple[int, str, str]],
        config: InferConfig,
    ) -> tuple[Prediction, ...]:
        session = self._backend.load(target)
        predictions = []
        try:
            for index, example_id, prompt in examples:
                model_input = format_prompt(prompt, target.data)
                prompt_tokens = session.count_tokens(model_input)
                if prompt_tokens > config.max_prompt_tokens:
                    raise InferenceError(
                        "Inference prompt exceeds max_prompt_tokens at "
                        f"example {index}: {prompt_tokens} > "
                        f"{config.max_prompt_tokens}"
                    )
                for seed in config.seeds:
                    rng_seed = _rng_seed(seed, example_id)
                    result = session.generate(model_input, rng_seed, config)
                    if result.prompt_tokens != prompt_tokens:
                        raise InferenceError(
                            f"Inference backend changed prompt tokens at example {index}"
                        )
                    predictions.append(
                        Prediction(
                            example_id,
                            index,
                            seed,
                            rng_seed,
                            result.text,
                            result.finish_reason,
                            result.prompt_tokens,
                            result.generated_tokens,
                        )
                    )
        except KeyboardInterrupt:
            raise
        except InferenceError:
            raise
        except Exception as error:
            raise InferenceError("Inference backend failed") from error
        finally:
            session.close()
        return tuple(predictions)


def configured_target(
    config: TrainConfig,
    resolver: Callable[[ModelSpec], Path] = resolve_model,
) -> ModelTarget:
    """Resolve and verify the configured base target."""
    spec = _model_spec(config)
    try:
        path = resolver(spec)
        digest = verify_model(path, spec)
    except ModelError as error:
        raise InferenceError(str(error)) from error
    value = {
        "mode": TargetMode.CONFIG_BASE.value,
        "model_digest": digest,
        "trust_remote_code": config.trust_remote_code,
        "data": asdict(config.data),
    }
    target_id = hashlib.sha256(_json_bytes(value)).hexdigest()
    return ModelTarget(
        target_id,
        TargetMode.CONFIG_BASE,
        path,
        digest,
        config.data,
        trust_remote_code=config.trust_remote_code,
    )


def recorded_target(
    path: Path,
    adapter: bool,
    resolver: Callable[[ModelSpec], Path] = resolve_model,
) -> ModelTarget:
    """Resolve and verify one target from a fixed training run."""
    try:
        run = load_run(path)
        model_path = resolver(run.model)
        verify_model(model_path, run.model, run.model_digest)
    except (ModelError, TrainError) as error:
        raise InferenceError(str(error)) from error
    mode = TargetMode.RUN_ADAPTER if adapter else TargetMode.RUN_BASE
    return ModelTarget(
        id=f"{run.ref.id}:{mode.value}",
        mode=mode,
        model_path=model_path,
        model_digest=run.model_digest,
        data=run.data,
        trust_remote_code=run.model.trust_remote_code,
        adapter_path=run.adapter_path if adapter else None,
        adapter_digest=run.adapter_digest if adapter else None,
        run_id=str(run.ref.id),
    )


def load_inference(path: Path) -> InferenceRef:
    """Load and verify one immutable inference artifact."""
    name = path.name
    if len(name) != 64 or any(
        character not in "0123456789abcdef" for character in name
    ):
        raise InferenceError(f"Inference path does not contain an ID: {path}")
    return _load_artifact(path, InferenceId(name))


def _validate(config: InferConfig, backend_version: str) -> None:
    missing = sorted(_REQUIRED_TOML - config.specified) if config.specified else []
    if missing:
        raise InferenceError(
            f"Inference configuration is incomplete: {', '.join(missing)}"
        )
    if config.output is None:
        raise InferenceError("Inference output is not configured")
    if config.backend != "mlx-lm":
        raise InferenceError("Inference backend must be mlx-lm")
    if not backend_version:
        raise InferenceError("Inference backend version is not available")
    if not config.seeds:
        raise InferenceError("Inference seeds are not configured")
    if config.max_prompt_tokens < 1:
        raise InferenceError("Inference max_prompt_tokens must be greater than zero")


def _model_spec(config: TrainConfig) -> ModelSpec:
    return ModelSpec(
        config.base_model,
        config.model_source,
        config.model_revision,
        config.model_cache,
        config.trust_remote_code,
    )


def _dataset_rows(
    dataset: DatasetRef,
    split: Split,
) -> tuple[tuple[int, str, str], ...]:
    path = dataset.path / f"{split.value}.jsonl"
    if not path.is_file():
        raise InferenceError(f"Dataset split does not exist: {split.value}")
    result = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
            prompt = value["prompt"]
            completion = value["completion"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise InferenceError(f"Dataset row is not valid: {path}:{index + 1}") from error
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise InferenceError(f"Dataset row text is not valid: {path}:{index + 1}")
        identity = {
            "dataset_id": str(dataset.id),
            "split": split.value,
            "index": index,
            "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        example_id = hashlib.sha256(_json_bytes(identity)).hexdigest()
        result.append((index, example_id, prompt))
    if not result:
        raise InferenceError(f"Dataset split is empty: {split.value}")
    return tuple(result)


def _select(
    rows: Sequence[tuple[int, str, str]],
    limit: int,
) -> tuple[tuple[int, str, str], ...]:
    if limit == 0 or limit >= len(rows):
        return tuple(rows)
    selected = sorted(rows, key=lambda value: value[1])[:limit]
    return tuple(sorted(selected, key=lambda value: value[0]))


def _rng_seed(seed: int, example_id: str) -> int:
    value = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    return int.from_bytes(value[:8], "big") & 0x7FFF_FFFF


def _recipe(
    target: ModelTarget,
    dataset: DatasetRef,
    dataset_digest: str,
    split: Split,
    config: InferConfig,
    backend_version: str,
    rows: Sequence[tuple[int, str, str]],
) -> Mapping[str, Any]:
    policy = asdict(config)
    policy.pop("output")
    policy.pop("specified")
    return {
        "version": _ARTIFACT_VERSION,
        "backend": config.backend,
        "backend_version": backend_version,
        "dataset_id": str(dataset.id),
        "dataset_digest": dataset_digest,
        "split": split.value,
        "examples": [value[1] for value in rows],
        "target": {
            "id": target.id,
            "mode": target.mode.value,
            "run_id": target.run_id,
            "model_digest": target.model_digest,
            "adapter_digest": target.adapter_digest,
            "trust_remote_code": target.trust_remote_code,
            "data": asdict(target.data),
        },
        "config": policy,
    }


def _prediction_value(value: Prediction) -> Mapping[str, Any]:
    return asdict(value)


def _required_output(config: InferConfig) -> Path:
    if config.output is None:
        raise InferenceError("Inference output is not configured")
    return config.output.expanduser().resolve()


def _load_artifact(path: Path, inference_id: InferenceId) -> InferenceRef:
    try:
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if value["inference_id"] != str(inference_id):
            raise InferenceError(f"Inference manifest does not match its path: {path}")
        actual_id = hashlib.sha256(_json_bytes(value["recipe"])).hexdigest()
        if actual_id != str(inference_id):
            raise InferenceError(f"Inference recipe does not match its ID: {path}")
        files = value["files"]
        if not isinstance(files, dict):
            raise TypeError
        actual_names = {
            item.relative_to(path).as_posix()
            for item in path.rglob("*")
            if item.is_file() and item != path / "manifest.json"
        }
        if actual_names != set(files):
            raise InferenceError(f"Inference files do not match its manifest: {path}")
        for name, expected in files.items():
            file_path = _artifact_file(path, name)
            actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if actual != expected:
                raise InferenceError(
                    f"Inference file does not match its manifest: {file_path}"
                )
        created_at = datetime.fromisoformat(value["created_at"])
        predictions = int(value["counts"]["predictions"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, InferenceError):
            raise
        raise InferenceError(f"Cannot read inference artifact: {path}") from error
    return InferenceRef(inference_id, path, created_at, predictions)


def _artifact_file(root: Path, name: object) -> Path:
    if not isinstance(name, str):
        raise TypeError
    root_path = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root_path) or not path.is_file():
        raise InferenceError(f"Inference manifest contains an invalid file path: {name}")
    return path


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


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _utc_now() -> datetime:
    return datetime.now(UTC)
