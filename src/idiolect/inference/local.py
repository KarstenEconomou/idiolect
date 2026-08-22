"""Run private local inference operations."""

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard

from idiolect.config import InferenceConfig, TrainConfig
from idiolect.data.local import load_dataset
from idiolect.inference.base import (
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
from idiolect.prompt import PromptError, format_prompt, validate_prompt_config
from idiolect.train.base import LoadedRun
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

    def validate(self, config: InferenceConfig) -> None:
        """Verify one complete inference policy."""
        _validate(config, self._backend.version)

    def text(
        self,
        target: ModelTarget,
        prompt: str,
        config: InferenceConfig,
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
        config: InferenceConfig,
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
        output = _required_output(config)
        existing = _find_artifact(output, recipe)
        if existing is not None:
            return existing

        predictions = self._generate(target, selected, config)
        created_at = self._clock()
        output.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".inference-", dir=output))
        destination: Path | None = None
        try:
            rows_value = [_prediction_value(value) for value in predictions]
            prediction_path = temporary / "pred.jsonl"
            _write_jsonl(prediction_path, rows_value)
            counts = {
                "examples": len(selected),
                "predictions": len(predictions),
            }
            files = {
                "pred.jsonl": hashlib.sha256(prediction_path.read_bytes()).hexdigest()
            }
            identity = {
                "recipe": recipe,
                "counts": counts,
                "files": files,
            }
            inference_id = InferenceId(
                hashlib.sha256(_json_bytes(identity)).hexdigest()
            )
            destination = output / str(inference_id)
            manifest = {
                "inference_id": str(inference_id),
                "created_at": created_at.isoformat(),
                **identity,
            }
            _write_json(temporary / "manifest.json", manifest)
            temporary.rename(destination)
        except KeyboardInterrupt:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        except (OSError, TypeError, ValueError) as error:
            shutil.rmtree(temporary, ignore_errors=True)
            if destination is not None and destination.exists():
                return _load_artifact(
                    destination,
                    InferenceId(destination.name),
                    expected_recipe=recipe,
                )
            raise InferenceError(
                f"Cannot create inference artifact: {destination or output}"
            ) from error
        return InferenceRef(inference_id, destination, created_at, len(predictions))

    def _generate(
        self,
        target: ModelTarget,
        examples: Sequence[tuple[int, str, str]],
        config: InferenceConfig,
    ) -> tuple[Prediction, ...]:
        try:
            validate_prompt_config(target.data)
        except PromptError as error:
            raise InferenceError(str(error)) from error
        session = self._backend.load(target)
        predictions = []
        try:
            for index, example_id, prompt in examples:
                model_input = format_prompt(prompt, target.data)
                prompt_tokens = session.count_tokens(model_input)
                if (
                    not isinstance(prompt_tokens, int)
                    or isinstance(prompt_tokens, bool)
                    or prompt_tokens < 1
                ):
                    raise InferenceError(
                        f"Inference backend returned invalid prompt tokens at example {index}"
                    )
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
                    prediction = Prediction(
                        example_id,
                        index,
                        seed,
                        rng_seed,
                        result.text,
                        result.finish_reason,
                        result.prompt_tokens,
                        result.generated_tokens,
                    )
                    _validate_prediction(prediction, config)
                    predictions.append(prediction)
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
    expected_digest: str | None = None,
) -> ModelTarget:
    """Resolve and verify the configured base target."""
    try:
        validate_prompt_config(config.data)
    except PromptError as error:
        raise InferenceError(str(error)) from error
    if not config.base_model:
        raise InferenceError("Inference base model is not configured")
    spec = _model_spec(config)
    try:
        path = resolver(spec)
        digest = verify_model(path, spec, expected_digest)
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
        return RecordedTargetResolver(resolver).target(run, adapter)
    except (InferenceError, TrainError) as error:
        raise InferenceError(str(error)) from error


class RecordedTargetResolver:
    """Resolve verified targets and reuse model verification."""

    def __init__(
        self,
        resolver: Callable[[ModelSpec], Path] = resolve_model,
    ) -> None:
        """Set the model resolver and create an empty verification cache."""
        self._resolver = resolver
        self._models: dict[tuple[ModelSpec, str], Path] = {}

    def target(self, run: LoadedRun, adapter: bool) -> ModelTarget:
        """Build one target from one verified training run."""
        try:
            validate_prompt_config(run.data)
            key = (run.model, run.model_digest)
            model_path = self._models.get(key)
            if model_path is None:
                model_path = self._resolver(run.model)
                verify_model(model_path, run.model, run.model_digest)
                self._models[key] = model_path
        except (ModelError, PromptError) as error:
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
    if not _is_digest(name):
        raise InferenceError(f"Inference path does not contain an ID: {path}")
    return _load_artifact(path, InferenceId(name))


def _validate(config: InferenceConfig, backend_version: str) -> None:
    missing = sorted(_REQUIRED_TOML - config.specified)
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
    if len(set(config.seeds)) != len(config.seeds):
        raise InferenceError("Inference seeds must be unique")
    if config.max_examples < 0:
        raise InferenceError("Inference max_examples must not be negative")
    if config.max_prompt_tokens < 1 or config.max_tokens < 1:
        raise InferenceError("Inference token limits are not valid")
    sampling = (
        config.temperature,
        config.top_p,
        config.min_p,
        config.repetition_penalty,
    )
    if any(not math.isfinite(value) for value in sampling):
        raise InferenceError("Inference sampling values must be finite")
    if config.temperature < 0:
        raise InferenceError("Inference temperature must not be negative")
    if not 0.0 <= config.top_p <= 1.0 or not 0.0 <= config.min_p <= 1.0:
        raise InferenceError("Inference probability limits must be from zero to one")
    if config.top_k < 0 or config.min_tokens_to_keep < 1:
        raise InferenceError("Inference token sampling limits are not valid")
    if config.repetition_penalty <= 0 or config.repetition_context_size < 1:
        raise InferenceError("Inference repetition settings are not valid")


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
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise InferenceError(f"Cannot read dataset split: {split.value}") from error
    result = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
            prompt = value["prompt"]
            completion = value["completion"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise InferenceError(
                f"Dataset row is not valid: {path}:{index + 1}"
            ) from error
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
    config: InferenceConfig,
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
        "examples": [{"index": value[0], "example_id": value[1]} for value in rows],
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


def _required_output(config: InferenceConfig) -> Path:
    if config.output is None:
        raise InferenceError("Inference output is not configured")
    return config.output.expanduser().resolve()


def _find_artifact(
    output: Path,
    recipe: Mapping[str, Any],
) -> InferenceRef | None:
    if not output.is_dir():
        return None
    matches = []
    try:
        paths = sorted(output.iterdir())
    except OSError as error:
        raise InferenceError(f"Cannot inspect inference output: {output}") from error
    for path in paths:
        if not path.is_dir() or not _is_digest(path.name):
            continue
        try:
            value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        except OSError, ValueError, json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or not _same_json(value.get("recipe"), recipe):
            continue
        matches.append(
            _load_artifact(
                path,
                InferenceId(path.name),
                expected_recipe=recipe,
            )
        )
    if len(matches) > 1:
        raise InferenceError("More than one inference artifact has the same recipe")
    return matches[0] if matches else None


def _load_artifact(
    path: Path,
    inference_id: InferenceId,
    expected_recipe: Mapping[str, Any] | None = None,
) -> InferenceRef:
    try:
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {
            "inference_id",
            "created_at",
            "recipe",
            "counts",
            "files",
        }:
            raise TypeError
        if value["inference_id"] != str(inference_id):
            raise InferenceError(f"Inference manifest does not match its path: {path}")
        recipe = value["recipe"]
        counts = _artifact_counts(value["counts"])
        files = _artifact_files(value["files"])
        identity = {"recipe": recipe, "counts": counts, "files": files}
        actual_id = hashlib.sha256(_json_bytes(identity)).hexdigest()
        if actual_id != str(inference_id):
            raise InferenceError(f"Inference content does not match its ID: {path}")
        if expected_recipe is not None and not _same_json(recipe, expected_recipe):
            raise InferenceError(f"Inference recipe does not match its request: {path}")
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
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise TypeError
        predictions = _read_predictions(path / "pred.jsonl")
        _validate_artifact_predictions(recipe, counts, predictions)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, InferenceError):
            raise
        raise InferenceError(f"Cannot read inference artifact: {path}") from error
    return InferenceRef(inference_id, path, created_at, len(predictions))


def _artifact_counts(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"examples", "predictions"}:
        raise TypeError
    if not all(_is_nonnegative_int(item) for item in value.values()):
        raise TypeError
    return {"examples": value["examples"], "predictions": value["predictions"]}


def _artifact_files(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"pred.jsonl"}:
        raise TypeError
    digest = value["pred.jsonl"]
    if not isinstance(digest, str) or not _is_digest(digest):
        raise TypeError
    return {"pred.jsonl": digest}


def _read_predictions(path: Path) -> tuple[Prediction, ...]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != {
            "example_id",
            "index",
            "seed",
            "rng_seed",
            "text",
            "finish_reason",
            "prompt_tokens",
            "generated_tokens",
        }:
            raise TypeError
        values.append(Prediction(**value))
    return tuple(values)


def _validate_artifact_predictions(
    recipe: object,
    counts: Mapping[str, int],
    predictions: Sequence[Prediction],
) -> None:
    if not isinstance(recipe, dict):
        raise TypeError
    examples = recipe.get("examples")
    config = recipe.get("config")
    if not isinstance(examples, list) or not isinstance(config, dict):
        raise TypeError
    seeds = config.get("seeds")
    max_prompt_tokens = config.get("max_prompt_tokens")
    max_tokens = config.get("max_tokens")
    if (
        not isinstance(seeds, list)
        or not all(_is_int(seed) for seed in seeds)
        or not _is_positive_int(max_prompt_tokens)
        or not _is_positive_int(max_tokens)
    ):
        raise TypeError
    expected = []
    for example in examples:
        if not isinstance(example, dict) or set(example) != {"index", "example_id"}:
            raise TypeError
        index = example["index"]
        example_id = example["example_id"]
        if (
            not _is_nonnegative_int(index)
            or not isinstance(example_id, str)
            or not _is_digest(example_id)
        ):
            raise TypeError
        expected.extend((index, example_id, seed) for seed in seeds)
    if len(set(expected)) != len(expected):
        raise TypeError
    if counts != {"examples": len(examples), "predictions": len(expected)}:
        raise InferenceError("Inference counts do not match its recipe")
    if len(predictions) != len(expected):
        raise InferenceError("Inference predictions do not match its counts")
    for prediction, (index, example_id, seed) in zip(
        predictions, expected, strict=True
    ):
        _validate_prediction_values(
            prediction,
            max_prompt_tokens,
            max_tokens,
        )
        if (
            prediction.index != index
            or prediction.example_id != example_id
            or prediction.seed != seed
            or prediction.rng_seed != _rng_seed(seed, example_id)
        ):
            raise InferenceError("Inference prediction does not match its recipe")


def _validate_prediction(value: Prediction, config: InferenceConfig) -> None:
    _validate_prediction_values(
        value,
        config.max_prompt_tokens,
        config.max_tokens,
    )


def _validate_prediction_values(
    value: Prediction,
    max_prompt_tokens: int,
    max_tokens: int,
) -> None:
    if not isinstance(value.example_id, str) or not _is_digest(value.example_id):
        raise InferenceError("Inference backend returned an invalid example ID")
    if not _is_nonnegative_int(value.index) or not _is_int(value.seed):
        raise InferenceError("Inference backend returned invalid prediction identity")
    if not _is_nonnegative_int(value.rng_seed) or value.rng_seed > 0x7FFF_FFFF:
        raise InferenceError("Inference backend returned an invalid random seed")
    if not isinstance(value.text, str):
        raise InferenceError("Inference backend returned invalid text")
    if value.finish_reason not in {"stop", "length"}:
        raise InferenceError("Inference backend returned an invalid finish reason")
    if (
        not _is_positive_int(value.prompt_tokens)
        or value.prompt_tokens > max_prompt_tokens
        or not _is_positive_int(value.generated_tokens)
        or value.generated_tokens > max_tokens
    ):
        raise InferenceError("Inference backend returned invalid token counts")


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonnegative_int(value: object) -> bool:
    return _is_int(value) and value >= 0


def _is_positive_int(value: object) -> bool:
    return _is_int(value) and value > 0


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _same_json(first: object, second: object) -> bool:
    try:
        return _json_bytes(first) == _json_bytes(second)
    except TypeError, ValueError:
        return False


def _artifact_file(root: Path, name: object) -> Path:
    if not isinstance(name, str):
        raise TypeError
    root_path = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root_path) or not path.is_file():
        raise InferenceError(
            f"Inference manifest contains an invalid file path: {name}"
        )
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
