"""Run immutable local policy evaluations."""

import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from idiolect.config import EvalConfig, InferConfig
from idiolect.data.local import load_dataset
from idiolect.eval.base import CompletionScore, ScoreBackend
from idiolect.eval.text import TrainingMatchIndex, normalize_text
from idiolect.infer.base import ModelTarget, Prediction
from idiolect.infer.local import RecordedTargetResolver, load_inference
from idiolect.model import directory_digest
from idiolect.prompt import format_prompt
from idiolect.train.base import LoadedRun
from idiolect.types import (
    DatasetRef,
    EvaluationId,
    EvaluationRef,
    EvaluationReport,
    GateResult,
    InferenceRef,
    Interval,
    Split,
)

_ARTIFACT_VERSION = 1
_SUITE = "fidelity"
_REQUIRED_TOML = frozenset(
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
    }
)
_MENTION = re.compile(r"@[\w.-]+", re.UNICODE)
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_ROLE = re.compile(r"(^|\n)\s*(assistant|system|user)\s*:", re.IGNORECASE)
_FEATURES = (
    "characters",
    "words",
    "lines",
    "punctuation_density",
    "uppercase_ratio",
    "emoji_rate",
    "mention_rate",
    "url_rate",
    "question_rate",
    "exclamation_rate",
    "terminal_punctuation_rate",
    "starts_lowercase_rate",
    "repeated_character_rate",
)


class EvaluationError(RuntimeError):
    """Report an invalid or failed evaluation operation."""


@dataclass(frozen=True, slots=True)
class EvalRow:
    """Keep one selected held-out example."""

    index: int
    example_id: str
    prompt: str
    completion: str


class DatasetInferencer(Protocol):
    """Generate one fixed dataset split."""

    def dataset(
        self,
        target: ModelTarget,
        dataset: DatasetRef,
        split: Split,
        config: InferConfig,
    ) -> InferenceRef:
        """Generate one split and return its artifact."""
        ...


class LocalEvaluator:
    """Compare one complete adapter policy with its recorded base."""

    def __init__(
        self,
        scorer: ScoreBackend,
        inferencer: DatasetInferencer,
        target_loader: Callable[[LoadedRun, bool], ModelTarget] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Set the scoring and generation boundaries."""
        self._scorer = scorer
        self._inferencer = inferencer
        self._target_loader = target_loader
        self._clock = _utc_now if clock is None else clock

    def evaluate(
        self,
        runs: Sequence[LoadedRun],
        dataset: DatasetRef,
        config: EvalConfig,
        infer: InferConfig,
    ) -> EvaluationRef:
        """Evaluate one policy and return its fixed artifact."""
        _validate_config(config, infer, self._scorer.version)
        ordered = _validate_policy(runs, dataset)
        verified = load_dataset(dataset.path).dataset
        rows = _select(_load_rows(verified, Split.VALID), config.max_examples)
        train_completions = tuple(
            row.completion for row in _load_rows(verified, Split.TRAIN)
        )
        effective_infer = replace(infer, max_examples=config.max_examples)

        target_loader = self._target_loader
        if target_loader is None:
            target_loader = RecordedTargetResolver().target
        base_target = target_loader(ordered[0], False)
        adapter_targets = tuple(
            target_loader(run, True) for run in ordered
        )
        base_inference = self._inferencer.dataset(
            base_target, verified, Split.VALID, effective_infer
        )
        adapter_inferences = tuple(
            self._inferencer.dataset(target, verified, Split.VALID, effective_infer)
            for target in adapter_targets
        )
        base_predictions = _load_predictions(base_inference.path)
        run_predictions = tuple(
            _load_predictions(reference.path) for reference in adapter_inferences
        )
        _check_alignment(rows, base_predictions, run_predictions, infer.seeds)

        dataset_digest = directory_digest(verified.path)
        recipe = {
            "version": _ARTIFACT_VERSION,
            "suite": config.suite,
            "dataset_id": str(verified.id),
            "dataset_digest": dataset_digest,
            "split": config.split,
            "examples": [row.example_id for row in rows],
            "runs": [str(run.ref.id) for run in ordered],
            "run_seeds": [run.seed for run in ordered],
            "model_digest": ordered[0].model_digest,
            "adapter_digests": [run.adapter_digest for run in ordered],
            "inferences": {
                "base": str(base_inference.id),
                "runs": [str(value.id) for value in adapter_inferences],
            },
            "backend": config.backend,
            "backend_version": self._scorer.version,
            "eval_config": _config_value(config),
            "infer_config": _infer_value(effective_infer),
        }
        evaluation_id = EvaluationId(hashlib.sha256(_json_bytes(recipe)).hexdigest())
        output = _required_output(config)
        destination = output / str(evaluation_id)
        if destination.exists():
            return load_evaluation(destination)

        base_scores = self._score(base_target, rows, ordered[0].max_seq_length)
        run_scores = tuple(
            self._score(target, rows, run.max_seq_length)
            for target, run in zip(adapter_targets, ordered, strict=True)
        )
        report, example_values = _report(
            rows,
            train_completions,
            ordered,
            base_scores,
            run_scores,
            base_predictions,
            run_predictions,
            config,
        )
        created_at = self._clock()
        output.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".eval-", dir=output))
        try:
            metrics_path = temporary / "metrics.json"
            examples_path = temporary / "examples.jsonl"
            report_path = temporary / "report.md"
            _write_json(metrics_path, report)
            _write_jsonl(examples_path, example_values)
            _write_text(report_path, _markdown_report(report, ordered))
            files = _file_hashes(temporary)
            manifest = {
                "evaluation_id": str(evaluation_id),
                "created_at": created_at.isoformat(),
                "eligible": report["eligible"],
                "recipe": recipe,
                "sources": {
                    "dataset": str(verified.path.resolve()),
                    "base_inference": str(base_inference.path.resolve()),
                    "run_inferences": [
                        str(value.path.resolve()) for value in adapter_inferences
                    ],
                },
                "files": files,
            }
            _write_json(temporary / "manifest.json", manifest)
            temporary.rename(destination)
        except KeyboardInterrupt:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        except (OSError, TypeError, ValueError) as error:
            shutil.rmtree(temporary, ignore_errors=True)
            raise EvaluationError(
                f"Cannot create evaluation artifact: {destination}"
            ) from error
        return EvaluationRef(evaluation_id, destination, created_at, report["eligible"])

    def _score(
        self,
        target: ModelTarget,
        rows: Sequence[EvalRow],
        max_seq_length: int,
    ) -> tuple[CompletionScore, ...]:
        session = self._scorer.load(target)
        values = []
        try:
            for row in rows:
                value = session.score(
                    format_prompt(row.prompt, target.data),
                    f"{row.completion}{target.data.completion_suffix}",
                )
                total = value.prompt_tokens + value.tokens
                if total > max_seq_length:
                    raise EvaluationError(
                        "Evaluation row exceeds the recorded max_seq_length at "
                        f"example {row.index}: {total} > {max_seq_length}"
                    )
                if value.tokens < 1 or not math.isfinite(
                    value.negative_log_likelihood
                ):
                    raise EvaluationError(
                        f"Evaluation backend returned an invalid score at example {row.index}"
                    )
                values.append(value)
        except KeyboardInterrupt:
            raise
        except EvaluationError:
            raise
        except Exception as error:
            raise EvaluationError("Evaluation scoring backend failed") from error
        finally:
            session.close()
        return tuple(values)


def load_evaluation(path: Path) -> EvaluationRef:
    """Load and verify one immutable evaluation artifact."""
    try:
        evaluation_id = EvaluationId(path.name)
        _valid_digest(path.name, "Evaluation")
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if value["evaluation_id"] != str(evaluation_id):
            raise EvaluationError(f"Evaluation manifest does not match its path: {path}")
        if hashlib.sha256(_json_bytes(value["recipe"])).hexdigest() != str(
            evaluation_id
        ):
            raise EvaluationError(f"Evaluation recipe does not match its ID: {path}")
        files = value["files"]
        if not isinstance(files, dict):
            raise TypeError
        actual_names = {
            item.relative_to(path).as_posix()
            for item in path.rglob("*")
            if item.is_file() and item != path / "manifest.json"
        }
        if actual_names != set(files):
            raise EvaluationError(f"Evaluation files do not match its manifest: {path}")
        for name, expected in files.items():
            file_path = _artifact_file(path, name)
            if hashlib.sha256(file_path.read_bytes()).hexdigest() != expected:
                raise EvaluationError(
                    f"Evaluation file does not match its manifest: {file_path}"
                )
        created_at = datetime.fromisoformat(value["created_at"])
        eligible = value["eligible"]
        if not isinstance(eligible, bool):
            raise TypeError
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        if not isinstance(metrics, dict) or metrics.get("eligible") != eligible:
            raise EvaluationError(
                f"Evaluation result does not match its manifest: {path}"
            )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, EvaluationError):
            raise
        raise EvaluationError(f"Cannot read evaluation artifact: {path}") from error
    return EvaluationRef(evaluation_id, path, created_at, eligible)


def _validate_config(
    config: EvalConfig,
    infer: InferConfig,
    backend_version: str,
) -> None:
    missing = sorted(_REQUIRED_TOML - config.specified) if config.specified else []
    if missing:
        raise EvaluationError(
            f"Evaluation configuration is incomplete: {', '.join(missing)}"
        )
    if config.output is None:
        raise EvaluationError("Evaluation output is not configured")
    if config.backend != "mlx-lm" or infer.backend != config.backend:
        raise EvaluationError("Evaluation and inference backends must be mlx-lm")
    if config.suite != _SUITE:
        raise EvaluationError(f"Evaluation suite must be {_SUITE}")
    if config.split != Split.VALID.value:
        raise EvaluationError("Evaluation split must be valid")
    if config.bootstrap_samples < 1:
        raise EvaluationError("Evaluation bootstrap_samples must be greater than zero")
    if not 0.0 < config.confidence_level < 1.0:
        raise EvaluationError("Evaluation confidence_level must be between zero and one")
    if config.long_match_chars < 8:
        raise EvaluationError("Evaluation long_match_chars must be at least eight")
    if config.ballots_per_rater < 1:
        raise EvaluationError("Evaluation ballots_per_rater must be greater than zero")
    if config.min_panel_raters < 1 or config.min_primary_comparisons < 1:
        raise EvaluationError("Evaluation panel minimums must be greater than zero")
    if not infer.seeds:
        raise EvaluationError("Inference seeds are not configured")
    if not backend_version:
        raise EvaluationError("Evaluation backend version is not available")


def _validate_policy(
    runs: Sequence[LoadedRun],
    dataset: DatasetRef,
) -> tuple[LoadedRun, ...]:
    if not runs:
        raise EvaluationError("Evaluation requires at least one training run")
    ordered = tuple(sorted(runs, key=lambda value: value.seed))
    first = ordered[0]
    expected_seeds = first.policy.get("seeds")
    if not isinstance(expected_seeds, list) or not all(
        isinstance(seed, int) and not isinstance(seed, bool) for seed in expected_seeds
    ):
        raise EvaluationError("Run policy does not contain valid training seeds")
    actual_seeds = [run.seed for run in ordered]
    if actual_seeds != sorted(expected_seeds):
        raise EvaluationError("Evaluation runs must contain every configured training seed")
    if len(actual_seeds) != len(set(actual_seeds)):
        raise EvaluationError("Evaluation runs contain duplicate training seeds")
    for run in ordered:
        if run.ref.dataset_id != dataset.id:
            raise EvaluationError("Evaluation run does not match the dataset")
        if run.model_digest != first.model_digest or run.data != first.data:
            raise EvaluationError("Evaluation runs do not use the same model and text format")
        if run.policy != first.policy:
            raise EvaluationError("Evaluation runs do not use the same training policy")
        if run.max_seq_length != first.max_seq_length:
            raise EvaluationError("Evaluation runs do not use the same sequence limit")
    return ordered


def _load_rows(dataset: DatasetRef, split: Split) -> tuple[EvalRow, ...]:
    path = dataset.path / f"{split.value}.jsonl"
    if not path.is_file():
        raise EvaluationError(f"Dataset split does not exist: {split.value}")
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
            prompt = value["prompt"]
            completion = value["completion"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise EvaluationError(f"Dataset row is not valid: {path}:{index + 1}") from error
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise EvaluationError(f"Dataset row text is not valid: {path}:{index + 1}")
        identity = {
            "dataset_id": str(dataset.id),
            "split": split.value,
            "index": index,
            "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        example_id = hashlib.sha256(_json_bytes(identity)).hexdigest()
        rows.append(EvalRow(index, example_id, prompt, completion))
    if not rows:
        raise EvaluationError(f"Dataset split is empty: {split.value}")
    return tuple(rows)


def _select(rows: Sequence[EvalRow], limit: int) -> tuple[EvalRow, ...]:
    if limit == 0 or limit >= len(rows):
        return tuple(rows)
    selected = sorted(rows, key=lambda value: value.example_id)[:limit]
    return tuple(sorted(selected, key=lambda value: value.index))


def _load_predictions(path: Path) -> tuple[Prediction, ...]:
    result = []
    try:
        load_inference(path)
        for line in (path / "pred.jsonl").read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            result.append(Prediction(**value))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Cannot read inference predictions: {path}") from error
    return tuple(result)


def _check_alignment(
    rows: Sequence[EvalRow],
    base: Sequence[Prediction],
    runs: Sequence[Sequence[Prediction]],
    seeds: Sequence[int],
) -> None:
    expected = [(row.example_id, seed) for row in rows for seed in seeds]
    for name, values in (("base", base), *(
        (f"run {index}", predictions)
        for index, predictions in enumerate(runs, start=1)
    )):
        actual = [(value.example_id, value.seed) for value in values]
        if actual != expected:
            raise EvaluationError(f"Evaluation {name} predictions are not aligned")


def _report(
    rows: Sequence[EvalRow],
    train: Sequence[str],
    runs: Sequence[LoadedRun],
    base_scores: Sequence[CompletionScore],
    run_scores: Sequence[Sequence[CompletionScore]],
    base_predictions: Sequence[Prediction],
    run_predictions: Sequence[Sequence[Prediction]],
    config: EvalConfig,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    reference = [row.completion for row in rows]
    base_text = [value.text for value in base_predictions]
    run_text = [[value.text for value in values] for values in run_predictions]
    policy_text = [text for values in run_text for text in values]
    match_index = TrainingMatchIndex.build(train, config.long_match_chars)

    likelihood = _likelihood_report(
        rows, runs, base_scores, run_scores, config
    )
    reference_profile = _profile(reference)
    profiles: dict[str, Any] = {
        "reference": reference_profile,
        "base": _profile_comparison(base_text, reference_profile, reference),
        "policy": _profile_comparison(policy_text, reference_profile, reference),
        "runs": {
            str(run.ref.id): _profile_comparison(text, reference_profile, reference)
            for run, text in zip(runs, run_text, strict=True)
        },
    }
    base_behavior = _behavior(
        base_predictions, rows, match_index, config
    )
    run_behavior = {
        str(run.ref.id): _behavior(
            values, rows, match_index, config
        )
        for run, values in zip(runs, run_predictions, strict=True)
    }
    policy_predictions = tuple(
        value for predictions in run_predictions for value in predictions
    )
    policy_behavior = _behavior(
        policy_predictions, rows, match_index, config
    )
    reference_memorization = _memorization_rate(
        reference, match_index
    )
    incremental = max(
        0.0,
        policy_behavior["long_training_match_rate"]
        - max(base_behavior["long_training_match_rate"], reference_memorization),
    )
    gates = {
        "empty_output": _gate(
            policy_behavior["empty_rate"], config.max_empty_rate
        ),
        "format_violation": _gate(
            policy_behavior["format_violation_rate"],
            config.max_format_violation_rate,
        ),
        "truncation": _gate(
            policy_behavior["truncation_rate"], config.max_truncation_rate
        ),
        "incremental_memorization": _gate(
            incremental, config.max_memorization_rate_delta
        ),
    }
    eligible = all(value.passed for value in gates.values())
    report = asdict(
        EvaluationReport(
            suite=_SUITE,
            eligible=eligible,
            examples=len(rows),
            training_runs=len(runs),
            generation_seeds=len({value.seed for value in base_predictions}),
            gates=gates,
            likelihood=likelihood,
            voice_profiles=profiles,
            behavior={
            "reference_long_training_match_rate": reference_memorization,
            "incremental_memorization_rate": incremental,
            "base": base_behavior,
            "policy": policy_behavior,
            "runs": run_behavior,
            },
        )
    )
    examples = []
    by_base = _by_example(base_predictions)
    by_runs = [_by_example(values) for values in run_predictions]
    for position, row in enumerate(rows):
        examples.append(
            {
                "example_id": row.example_id,
                "index": row.index,
                "base_score": asdict(base_scores[position]),
                "run_scores": {
                    str(run.ref.id): asdict(scores[position])
                    for run, scores in zip(runs, run_scores, strict=True)
                },
                "base_outputs": [
                    _example_diagnostic(
                        value, row, match_index, config
                    )
                    for value in by_base[row.example_id]
                ],
                "run_outputs": {
                    str(run.ref.id): [
                        _example_diagnostic(
                            value, row, match_index, config
                        )
                        for value in values[row.example_id]
                    ]
                    for run, values in zip(runs, by_runs, strict=True)
                },
            }
        )
    return report, tuple(examples)


def _likelihood_report(
    rows: Sequence[EvalRow],
    runs: Sequence[LoadedRun],
    base: Sequence[CompletionScore],
    run_scores: Sequence[Sequence[CompletionScore]],
    config: EvalConfig,
) -> Mapping[str, Any]:
    base_values = [_mean_nll(value) for value in base]
    run_values = [[_mean_nll(value) for value in values] for values in run_scores]
    policy_values = [
        sum(values[index] for values in run_values) / len(run_values)
        for index in range(len(rows))
    ]
    policy_delta = [
        policy_values[index] - base_values[index] for index in range(len(rows))
    ]
    result_runs = {}
    for run, scores, values in zip(runs, run_scores, run_values, strict=True):
        deltas = [value - baseline for value, baseline in zip(values, base_values, strict=True)]
        corpus_mean_nll = _corpus_mean_nll(scores)
        result_runs[str(run.ref.id)] = {
            "seed": run.seed,
            "macro_mean_nll": _mean(values),
            "corpus_mean_nll": corpus_mean_nll,
            "corpus_perplexity": _safe_exp(corpus_mean_nll),
            "delta_macro_mean_nll": _interval_value(deltas, config),
            "examples_improved_rate": sum(value < 0 for value in deltas) / len(deltas),
        }
    policy_scores = tuple(value for values in run_scores for value in values)
    return {
        "base": {
            "macro_mean_nll": _mean(base_values),
            "corpus_mean_nll": _corpus_mean_nll(base),
            "corpus_perplexity": _safe_exp(_corpus_mean_nll(base)),
        },
        "policy": {
            "macro_mean_nll": _mean(policy_values),
            "corpus_mean_nll": _corpus_mean_nll(policy_scores),
            "corpus_perplexity": _safe_exp(_corpus_mean_nll(policy_scores)),
            "delta_macro_mean_nll": _interval_value(policy_delta, config),
            "examples_improved_rate": sum(value < 0 for value in policy_delta)
            / len(policy_delta),
            "run_delta_range": [
                min(_mean([v - b for v, b in zip(values, base_values, strict=True)]) for values in run_values),
                max(_mean([v - b for v, b in zip(values, base_values, strict=True)]) for values in run_values),
            ],
        },
        "runs": result_runs,
    }


def _profile(texts: Sequence[str]) -> Mapping[str, float]:
    values = [_text_features(text) for text in texts]
    return {
        name: _mean([value[name] for value in values]) for name in _FEATURES
    }


def _profile_comparison(
    texts: Sequence[str],
    reference_profile: Mapping[str, float],
    reference: Sequence[str],
) -> Mapping[str, Any]:
    profile = _profile(texts)
    return {
        "values": profile,
        "absolute_differences": {
            name: abs(profile[name] - reference_profile[name]) for name in _FEATURES
        },
        "character_3gram_js_divergence": _js_divergence(
            _ngrams(texts, 3), _ngrams(reference, 3)
        ),
    }


def _text_features(text: str) -> Mapping[str, float]:
    stripped = text.strip()
    letters = [character for character in stripped if character.isalpha()]
    punctuation = [
        character for character in stripped if unicodedata.category(character).startswith("P")
    ]
    emoji = [character for character in stripped if _is_emoji(character)]
    return {
        "characters": float(len(stripped)),
        "words": float(len(stripped.split())),
        "lines": float(len(stripped.splitlines()) or 1),
        "punctuation_density": len(punctuation) / max(1, len(stripped)),
        "uppercase_ratio": sum(character.isupper() for character in letters)
        / max(1, len(letters)),
        "emoji_rate": float(bool(emoji)),
        "mention_rate": float(bool(_MENTION.search(stripped))),
        "url_rate": float(bool(_URL.search(stripped))),
        "question_rate": float("?" in stripped),
        "exclamation_rate": float("!" in stripped),
        "terminal_punctuation_rate": float(
            bool(stripped) and stripped[-1] in ".!?…"
        ),
        "starts_lowercase_rate": float(bool(stripped) and stripped[0].islower()),
        "repeated_character_rate": float(
            bool(re.search(r"(.)\1{2,}", stripped, re.DOTALL))
        ),
    }


def _behavior(
    predictions: Sequence[Prediction],
    rows: Sequence[EvalRow],
    match_index: TrainingMatchIndex,
    config: EvalConfig,
) -> Mapping[str, float]:
    row_map = {row.example_id: row for row in rows}
    diagnostics = [
        _example_diagnostic(
            value,
            row_map[value.example_id],
            match_index,
            config,
        )
        for value in predictions
    ]
    texts = [value.text for value in predictions]
    return {
        "empty_rate": _rate(diagnostics, "empty"),
        "format_violation_rate": _rate(diagnostics, "format_violation"),
        "unknown_mention_rate": _rate(diagnostics, "unknown_mention"),
        "truncation_rate": _rate(diagnostics, "truncated"),
        "long_training_match_rate": _rate(diagnostics, "long_training_match"),
        "exact_training_match_rate": _rate(diagnostics, "exact_training_match"),
        "cross_prompt_duplicate_rate": _duplicate_rate(texts),
        "within_prompt_duplicate_rate": _within_prompt_duplicate_rate(predictions),
    }


def _example_diagnostic(
    prediction: Prediction,
    row: EvalRow,
    match_index: TrainingMatchIndex,
    config: EvalConfig,
) -> Mapping[str, Any]:
    normalized = normalize_text(prediction.text)
    match = match_index.longest(prediction.text)
    allowed = set(_MENTION.findall(row.prompt))
    mentions = set(_MENTION.findall(prediction.text))
    return {
        "seed": prediction.seed,
        "finish_reason": prediction.finish_reason,
        "empty": not bool(prediction.text.strip()),
        "format_violation": _format_violation(prediction.text),
        "unknown_mention": bool(mentions - allowed),
        "truncated": prediction.finish_reason in {"length", "max_tokens"},
        "long_training_match": match >= config.long_match_chars,
        "longest_training_match_chars": match,
        "exact_training_match": bool(normalized) and normalized in match_index.exact,
    }


def _format_violation(text: str) -> bool:
    lowered = text.casefold()
    return bool(
        _ROLE.search(text)
        or "<think>" in lowered
        or "</think>" in lowered
        or "[next response]" in lowered
        or "conversation:" in lowered
    )


def _memorization_rate(
    texts: Sequence[str],
    index: TrainingMatchIndex,
) -> float:
    return sum(
        index.longest(text) >= index.threshold
        for text in texts
    ) / len(texts)


def _normalize(text: str) -> str:
    return normalize_text(text)


def _ngrams(texts: Sequence[str], size: int) -> Counter[str]:
    result: Counter[str] = Counter()
    for text in texts:
        normalized = _normalize(text)
        result.update(
            normalized[index : index + size]
            for index in range(max(0, len(normalized) - size + 1))
        )
    return result


def _js_divergence(left: Counter[str], right: Counter[str]) -> float:
    keys = set(left) | set(right)
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not keys or left_total == 0 or right_total == 0:
        return 0.0
    result = 0.0
    for key in keys:
        p = left[key] / left_total
        q = right[key] / right_total
        midpoint = (p + q) / 2
        if p:
            result += 0.5 * p * math.log2(p / midpoint)
        if q:
            result += 0.5 * q * math.log2(q / midpoint)
    return result


def _interval_value(values: Sequence[float], config: EvalConfig) -> Mapping[str, float]:
    rng = random.Random(config.bootstrap_seed)
    estimates = []
    for _ in range(config.bootstrap_samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(_mean(sample))
    estimates.sort()
    tail = (1.0 - config.confidence_level) / 2
    lower = estimates[min(len(estimates) - 1, int(tail * len(estimates)))]
    upper = estimates[min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))]
    return asdict(Interval(_mean(values), lower, upper))


def _by_example(predictions: Sequence[Prediction]) -> Mapping[str, tuple[Prediction, ...]]:
    values: dict[str, list[Prediction]] = {}
    for prediction in predictions:
        values.setdefault(prediction.example_id, []).append(prediction)
    return {key: tuple(items) for key, items in values.items()}


def _duplicate_rate(texts: Sequence[str]) -> float:
    normalized = [_normalize(text) for text in texts if _normalize(text)]
    if not normalized:
        return 0.0
    counts = Counter(normalized)
    return sum(count - 1 for count in counts.values()) / len(normalized)


def _within_prompt_duplicate_rate(predictions: Sequence[Prediction]) -> float:
    grouped = _by_example(predictions)
    duplicated = 0
    for values in grouped.values():
        texts = [_normalize(value.text) for value in values]
        duplicated += len(texts) > 1 and len(set(texts)) < len(texts)
    return duplicated / len(grouped)


def _rate(values: Sequence[Mapping[str, Any]], name: str) -> float:
    return sum(bool(value[name]) for value in values) / len(values)


def _gate(value: float, limit: float) -> GateResult:
    return GateResult(value, limit, value <= limit)


def _mean_nll(value: CompletionScore) -> float:
    return value.negative_log_likelihood / value.tokens


def _corpus_mean_nll(values: Sequence[CompletionScore]) -> float:
    return sum(value.negative_log_likelihood for value in values) / sum(
        value.tokens for value in values
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _safe_exp(value: float) -> float:
    return math.exp(min(value, 700.0))


def _is_emoji(character: str) -> bool:
    code = ord(character)
    return 0x1F000 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF


def _markdown_report(
    report: Mapping[str, Any], runs: Sequence[LoadedRun]
) -> str:
    policy = report["likelihood"]["policy"]
    delta = policy["delta_macro_mean_nll"]
    lines = [
        "# Idiolect Evaluation",
        "",
        f"Eligible: {'yes' if report['eligible'] else 'no'}",
        f"Validation examples: {report['examples']}",
        f"Training runs: {len(runs)}",
        "",
        "## Predictive fidelity",
        "",
        (
            "Base macro mean NLL: "
            f"{report['likelihood']['base']['macro_mean_nll']:.6f}"
        ),
        (
            "Base corpus perplexity: "
            f"{report['likelihood']['base']['corpus_perplexity']:.6f}"
        ),
        f"Policy macro mean NLL: {policy['macro_mean_nll']:.6f}",
        f"Policy corpus perplexity: {policy['corpus_perplexity']:.6f}",
        (
            "Paired macro-NLL policy delta: "
            f"{delta['value']:.6f} [{delta['lower']:.6f}, {delta['upper']:.6f}]"
        ),
        "",
        "## Gates",
        "",
    ]
    for name, gate in report["gates"].items():
        state = "pass" if gate["passed"] else "fail"
        lines.append(
            f"- {name}: {state} ({gate['value']:.6f} <= {gate['limit']:.6f})"
        )
    lines.extend(
        (
            "",
            (
                "The JSON report contains per-run likelihood, voice-profile, behavior, "
                "diversity, and memorization evidence. It does not combine these values "
                "into one fidelity score."
            ),
            "",
        )
    )
    return "\n".join(lines)


def _config_value(config: EvalConfig) -> Mapping[str, Any]:
    value = asdict(config)
    value.pop("specified")
    value["output"] = str(config.output) if config.output else None
    return value


def _infer_value(config: InferConfig) -> Mapping[str, Any]:
    value = asdict(config)
    value.pop("specified")
    value["output"] = str(config.output) if config.output else None
    return value


def _required_output(config: EvalConfig) -> Path:
    if config.output is None:
        raise EvaluationError("Evaluation output is not configured")
    return config.output.expanduser().resolve()


def _file_hashes(root: Path) -> Mapping[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _artifact_file(root: Path, name: object) -> Path:
    if not isinstance(name, str):
        raise TypeError
    root_path = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root_path) or not path.is_file():
        raise EvaluationError(f"Evaluation manifest contains an invalid file path: {name}")
    return path


def _valid_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise EvaluationError(f"{label} path does not contain an ID")


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


def _write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)
    os.chmod(path, 0o600)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _utc_now() -> datetime:
    return datetime.now(UTC)
