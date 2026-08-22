"""Collect and summarize private familiar-panel judgments."""

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from idiolect.config import EvalConfig
from idiolect.eval.local import (
    EvalRow,
    EvaluationError,
    _load_predictions,
    _load_rows,
    _select,
    load_evaluation,
)
from idiolect.inference.local import load_inference
from idiolect.types import (
    EvaluationId,
    EvaluationRef,
    Interval,
    JudgmentId,
    JudgmentRef,
    PanelId,
    PanelRef,
    Split,
)

_JUDGMENT_VERSION = 1
_PANEL_VERSION = 1
_DIMENSIONS = (
    ("target_likeness", "Which reply would the target be more likely to send here?"),
    ("voice", "Which reply sounds more like the target?"),
    ("context_fit", "Which reply fits the conversation better?"),
)
_CHOICES = frozenset({"a", "b", "tie", "neither"})
_CANONICAL_CHOICES = frozenset({"policy", "base", "human", "tie", "neither"})
_MATCHUP_IDENTITIES = {
    "policy-base": frozenset({"policy", "base"}),
    "human-policy": frozenset({"human", "policy"}),
    "human-base": frozenset({"human", "base"}),
}
_JUDGMENT_KEYS = frozenset(
    {"ballot_id", "example_id", "matchup", "dimension", "choice", "position"}
)
_RATER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def collect_judgments(
    evaluation_path: Path,
    rater_id: str,
    config: EvalConfig,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    clock: Callable[[], datetime] | None = None,
) -> JudgmentRef:
    """Run one blind terminal session and store its judgments."""
    evaluation = load_evaluation(evaluation_path)
    rater = rater_id.strip()
    if _RATER_ID.fullmatch(rater) is None:
        raise EvaluationError("Rater pseudonym contains invalid characters")
    ballots = _evaluation_ballots(evaluation, config, rater)
    write(
        "Private familiar-panel evaluation. Continue only if this rater has "
        "consent and permission to view these conversations."
    )
    judgments = []
    for number, ballot in enumerate(ballots, start=1):
        write(f"\nBallot {number}/{len(ballots)}\n")
        write(ballot["prompt"])
        write(f"\nA:\n{ballot['a_text']}\n\nB:\n{ballot['b_text']}")
        for dimension, question in _DIMENSIONS:
            answer = _answer(read, question)
            judgments.append(
                {
                    "ballot_id": ballot["ballot_id"],
                    "example_id": ballot["example_id"],
                    "matchup": ballot["matchup"],
                    "dimension": dimension,
                    "choice": _canonical_choice(
                        answer, ballot["a_identity"], ballot["b_identity"]
                    ),
                    "position": answer if answer in {"a", "b"} else None,
                }
            )
    created_at = (clock or _utc_now)()
    artifact_recipe = {
        "version": _JUDGMENT_VERSION,
        "evaluation_id": str(evaluation.id),
        "rater_id": rater,
        "ballot_ids": [value["ballot_id"] for value in ballots],
        "config": {
            "ballot_seed": config.ballot_seed,
            "ballots_per_rater": config.ballots_per_rater,
            "control_fraction": config.control_fraction,
        },
        "judgments": judgments,
    }
    judgment_id = JudgmentId(hashlib.sha256(_json_bytes(artifact_recipe)).hexdigest())
    root = _evaluation_root(config) / "judgments"
    destination = root / str(judgment_id)
    if destination.exists():
        return load_judgment(destination)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".judgment-", dir=root))
    try:
        judgment_path = temporary / "judgments.jsonl"
        _write_jsonl(judgment_path, judgments)
        manifest_value = {
            "judgment_id": str(judgment_id),
            "evaluation_id": str(evaluation.id),
            "rater_id": rater,
            "created_at": created_at.isoformat(),
            "recipe": artifact_recipe,
            "files": {
                "judgments.jsonl": hashlib.sha256(judgment_path.read_bytes()).hexdigest()
            },
        }
        _write_json(temporary / "manifest.json", manifest_value)
        temporary.rename(destination)
    except (OSError, TypeError, ValueError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise EvaluationError(f"Cannot create judgment artifact: {destination}") from error
    return JudgmentRef(
        judgment_id,
        evaluation.id,
        destination,
        created_at,
        len(judgments),
    )


def create_panel(
    evaluation_path: Path,
    judgment_paths: Sequence[Path],
    config: EvalConfig,
    clock: Callable[[], datetime] | None = None,
) -> PanelRef:
    """Create one immutable report from familiar-rater judgments."""
    evaluation = load_evaluation(evaluation_path)
    if not judgment_paths:
        raise EvaluationError("Panel evaluation requires at least one judgment artifact")
    loaded = tuple(load_judgment(path) for path in judgment_paths)
    if any(value.evaluation_id != evaluation.id for value in loaded):
        raise EvaluationError("Panel judgments do not match the evaluation")
    evaluation_manifest = _read_manifest(evaluation.path)
    _validate_panel_config(evaluation_manifest["recipe"], config)
    manifests = [_read_manifest(value.path) for value in loaded]
    raters = [_required_text(value.get("rater_id")) for value in manifests]
    if len(raters) != len(set(raters)):
        raise EvaluationError("Panel contains more than one judgment set for a rater")
    judgments = []
    for reference, manifest in zip(loaded, manifests, strict=True):
        rater = _required_text(manifest["rater_id"])
        rows = _read_jsonl(reference.path / "judgments.jsonl")
        expected = _evaluation_ballots(evaluation, config, rater)
        _validate_judgment_schedule(manifest, rows, expected, config)
        judgments.extend(
            {**row, "rater_id": rater} for row in rows
        )
    primary = [
        value
        for value in judgments
        if value["matchup"] == "policy-base"
        and value["dimension"] == "target_likeness"
    ]
    complete = (
        len(raters) >= config.min_panel_raters
        and len(primary) >= config.min_primary_comparisons
    )
    summary = {
        "complete": complete,
        "raters": len(raters),
        "primary_comparisons": len(primary),
        "preferences": _preference_summary(judgments, config),
        "human_controls": _control_summary(judgments, config),
        "position_a_decisive_rate": _position_rate(judgments),
        "krippendorff_alpha": {
            dimension: _agreement(judgments, dimension)
            for dimension, _ in _DIMENSIONS
        },
    }
    recipe = {
        "version": _PANEL_VERSION,
        "evaluation_id": str(evaluation.id),
        "judgment_ids": sorted(str(value.id) for value in loaded),
        "minimums": {
            "raters": config.min_panel_raters,
            "primary_comparisons": config.min_primary_comparisons,
        },
    }
    panel_id = PanelId(hashlib.sha256(_json_bytes(recipe)).hexdigest())
    root = _evaluation_root(config) / "panels"
    destination = root / str(panel_id)
    if destination.exists():
        return load_panel(destination)
    created_at = (clock or _utc_now)()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".panel-", dir=root))
    try:
        report_path = temporary / "panel.json"
        _write_json(report_path, summary)
        manifest = {
            "panel_id": str(panel_id),
            "evaluation_id": str(evaluation.id),
            "created_at": created_at.isoformat(),
            "complete": complete,
            "recipe": recipe,
            "files": {
                "panel.json": hashlib.sha256(report_path.read_bytes()).hexdigest()
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except (OSError, TypeError, ValueError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise EvaluationError(f"Cannot create panel artifact: {destination}") from error
    return PanelRef(panel_id, evaluation.id, destination, created_at, complete)


def load_judgment(path: Path) -> JudgmentRef:
    """Load and verify one familiar-rater judgment artifact."""
    value = _verify_simple_artifact(path, "judgment_id", "Judgment")
    try:
        judgment_id = JudgmentId(value["judgment_id"])
        evaluation_id = EvaluationId(value["evaluation_id"])
        rater = _required_text(value["rater_id"])
        if _RATER_ID.fullmatch(rater) is None:
            raise EvaluationError("Judgment rater pseudonym is not valid")
        recipe = value["recipe"]
        rows = _read_jsonl(path / "judgments.jsonl")
        if (
            not isinstance(recipe, dict)
            or recipe.get("version") != _JUDGMENT_VERSION
            or recipe.get("evaluation_id") != str(evaluation_id)
            or recipe.get("rater_id") != rater
            or recipe.get("judgments") != list(rows)
        ):
            raise EvaluationError(f"Judgment content does not match its recipe: {path}")
        _validate_judgment_rows(rows)
        created_at = datetime.fromisoformat(value["created_at"])
        count = len(rows)
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"Cannot read judgment artifact: {path}") from error
    return JudgmentRef(judgment_id, evaluation_id, path, created_at, count)


def load_panel(path: Path) -> PanelRef:
    """Load and verify one familiar-panel artifact."""
    value = _verify_simple_artifact(path, "panel_id", "Panel")
    try:
        panel_id = PanelId(value["panel_id"])
        evaluation_id = EvaluationId(value["evaluation_id"])
        recipe = value["recipe"]
        if (
            not isinstance(recipe, dict)
            or recipe.get("version") != _PANEL_VERSION
            or recipe.get("evaluation_id") != str(evaluation_id)
        ):
            raise EvaluationError(f"Panel content does not match its recipe: {path}")
        created_at = datetime.fromisoformat(value["created_at"])
        complete = value["complete"]
        if not isinstance(complete, bool):
            raise TypeError
        summary = json.loads((path / "panel.json").read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or summary.get("complete") != complete:
            raise EvaluationError(f"Panel result does not match its manifest: {path}")
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"Cannot read panel artifact: {path}") from error
    return PanelRef(panel_id, evaluation_id, path, created_at, complete)


def _evaluation_ballots(
    evaluation: EvaluationRef,
    config: EvalConfig,
    rater_id: str,
) -> tuple[Mapping[str, Any], ...]:
    manifest = _read_manifest(evaluation.path)
    recipe = manifest["recipe"]
    _validate_panel_config(recipe, config)
    sources = manifest["sources"]
    dataset_path = _source_path(sources, "dataset")
    dataset = _dataset_reference(dataset_path)
    if str(dataset.id) != recipe.get("dataset_id"):
        raise EvaluationError("Evaluation dataset source does not match its recipe")
    rows = _select(
        _load_rows(dataset, Split.VALID),
        int(recipe["eval_config"]["max_examples"]),
    )
    selected_ids = set(recipe["examples"])
    rows = tuple(row for row in rows if row.example_id in selected_ids)
    if [row.example_id for row in rows] != recipe["examples"]:
        raise EvaluationError("Evaluation examples do not match the dataset source")

    base_path = _source_path(sources, "base_inference")
    base_ref = load_inference(base_path)
    inference_recipe = recipe.get("inferences")
    if not isinstance(inference_recipe, dict) or str(base_ref.id) != inference_recipe.get(
        "base"
    ):
        raise EvaluationError("Evaluation base inference source does not match its recipe")
    base = _load_predictions(base_path)
    run_paths = sources.get("run_inferences")
    if not isinstance(run_paths, list) or not run_paths:
        raise EvaluationError("Evaluation does not contain run inference sources")
    expected_runs = inference_recipe.get("runs")
    if not isinstance(expected_runs, list) or len(expected_runs) != len(run_paths):
        raise EvaluationError("Evaluation run inference sources do not match its recipe")
    verified_run_paths = tuple(Path(_required_text(value)) for value in run_paths)
    for path, expected in zip(verified_run_paths, expected_runs, strict=True):
        if str(load_inference(path).id) != expected:
            raise EvaluationError(
                "Evaluation run inference source does not match its recipe"
            )
    run_predictions = tuple(_load_predictions(path) for path in verified_run_paths)
    return _ballots(rows, base, run_predictions, config, rater_id)


def _ballots(
    rows: Sequence[EvalRow],
    base_predictions: Sequence[Any],
    run_predictions: Sequence[Sequence[Any]],
    config: EvalConfig,
    rater_id: str,
) -> tuple[Mapping[str, Any], ...]:
    base = _prediction_map(base_predictions)
    runs = [_prediction_map(values) for values in run_predictions]
    ordered_rows = list(rows)
    random.Random(config.ballot_seed).shuffle(ordered_rows)
    count = min(config.ballots_per_rater, len(ordered_rows))
    controls = round(count * config.control_fraction)
    primary = count - controls
    matchups = ["policy-base"] * primary
    matchups.extend(
        "human-policy" if index % 2 == 0 else "human-base"
        for index in range(controls)
    )
    random.Random(config.ballot_seed + 1).shuffle(matchups)
    order_rng = random.Random(_derived_seed(config.ballot_seed, rater_id))
    result = []
    for position, (row, matchup) in enumerate(
        zip(ordered_rows[:count], matchups, strict=True)
    ):
        available_seeds = sorted(base[row.example_id])
        seed = available_seeds[position % len(available_seeds)]
        run_index = position % len(runs)
        if matchup == "policy-base":
            candidates = (
                ("policy", runs[run_index][row.example_id][seed]),
                ("base", base[row.example_id][seed]),
            )
        elif matchup == "human-policy":
            candidates = (
                ("human", row.completion),
                ("policy", runs[run_index][row.example_id][seed]),
            )
        else:
            candidates = (
                ("human", row.completion),
                ("base", base[row.example_id][seed]),
            )
        canonical = {
            "example_id": row.example_id,
            "matchup": matchup,
            "seed": seed,
            "run_index": run_index if "policy" in {value[0] for value in candidates} else None,
        }
        ballot_id = hashlib.sha256(_json_bytes(canonical)).hexdigest()
        left, right = candidates
        if order_rng.randrange(2):
            left, right = right, left
        result.append(
            {
                "ballot_id": ballot_id,
                "example_id": row.example_id,
                "matchup": matchup,
                "prompt": row.prompt,
                "a_identity": left[0],
                "a_text": left[1],
                "b_identity": right[0],
                "b_text": right[1],
            }
        )
    return tuple(result)


def _prediction_map(values: Sequence[Any]) -> Mapping[str, Mapping[int, str]]:
    result: dict[str, dict[int, str]] = {}
    for value in values:
        result.setdefault(value.example_id, {})[value.seed] = value.text
    return result


def _answer(read: Callable[[str], str], question: str) -> str:
    while True:
        try:
            value = read(f"{question} [a/b/tie/neither]: ").strip().casefold()
        except EOFError as error:
            raise EvaluationError("Rating input ended before the session was complete") from error
        if value in _CHOICES:
            return value


def _canonical_choice(answer: str, left: str, right: str) -> str:
    if answer == "a":
        return left
    if answer == "b":
        return right
    return answer


def _validate_judgment_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    expected_dimensions = {name for name, _ in _DIMENSIONS}
    units: set[tuple[str, str]] = set()
    for row in rows:
        if set(row) != _JUDGMENT_KEYS:
            raise EvaluationError("Judgment row fields are not valid")
        ballot_id = _required_text(row["ballot_id"])
        example_id = _required_text(row["example_id"])
        matchup = _required_text(row["matchup"])
        dimension = _required_text(row["dimension"])
        choice = _required_text(row["choice"])
        if not _is_digest(ballot_id) or not _is_digest(example_id):
            raise EvaluationError("Judgment row identifiers are not valid")
        if matchup not in _MATCHUP_IDENTITIES or dimension not in expected_dimensions:
            raise EvaluationError("Judgment row policy is not valid")
        if choice not in _CANONICAL_CHOICES or choice not in (
            _MATCHUP_IDENTITIES[matchup] | {"tie", "neither"}
        ):
            raise EvaluationError("Judgment row choice is not valid")
        position = row["position"]
        if position not in {"a", "b", None}:
            raise EvaluationError("Judgment row position is not valid")
        if (choice in {"tie", "neither"}) != (position is None):
            raise EvaluationError("Judgment row choice and position do not match")
        unit = (ballot_id, dimension)
        if unit in units:
            raise EvaluationError("Judgment artifact contains duplicate ballot answers")
        units.add(unit)


def _validate_judgment_schedule(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    ballots: Sequence[Mapping[str, Any]],
    config: EvalConfig,
) -> None:
    recipe = manifest.get("recipe")
    if not isinstance(recipe, dict):
        raise EvaluationError("Judgment recipe is not valid")
    expected_ids = [ballot["ballot_id"] for ballot in ballots]
    if recipe.get("ballot_ids") != expected_ids:
        raise EvaluationError("Judgment ballots do not match the evaluation schedule")
    expected_config = {
        "ballot_seed": config.ballot_seed,
        "ballots_per_rater": config.ballots_per_rater,
        "control_fraction": config.control_fraction,
    }
    if recipe.get("config") != expected_config:
        raise EvaluationError("Judgment ballot configuration is not valid")
    expected = {
        (ballot["ballot_id"], dimension): (ballot, dimension)
        for ballot in ballots
        for dimension, _ in _DIMENSIONS
    }
    actual = {(row["ballot_id"], row["dimension"]): row for row in rows}
    if set(actual) != set(expected):
        raise EvaluationError("Judgment answers do not match the evaluation schedule")
    for unit, row in actual.items():
        ballot, dimension = expected[unit]
        if (
            row["example_id"] != ballot["example_id"]
            or row["matchup"] != ballot["matchup"]
            or row["dimension"] != dimension
        ):
            raise EvaluationError("Judgment answer does not match its ballot")
        position = row["position"]
        if position == "a":
            expected_choice = ballot["a_identity"]
        elif position == "b":
            expected_choice = ballot["b_identity"]
        else:
            expected_choice = row["choice"]
        if row["choice"] != expected_choice:
            raise EvaluationError("Judgment answer position does not match its choice")


def _preference_summary(
    judgments: Sequence[Mapping[str, Any]], config: EvalConfig
) -> Mapping[str, Any]:
    result = {}
    for dimension, _ in _DIMENSIONS:
        values = [
            value
            for value in judgments
            if value["matchup"] == "policy-base" and value["dimension"] == dimension
        ]
        counts = Counter(value["choice"] for value in values)
        result[dimension] = {
            "policy": counts["policy"],
            "base": counts["base"],
            "tie": counts["tie"],
            "neither": counts["neither"],
            "policy_decisive_rate": _cluster_interval(
                values, "policy", "base", config
            ),
        }
    return result


def _control_summary(
    judgments: Sequence[Mapping[str, Any]], config: EvalConfig
) -> Mapping[str, Any]:
    result = {}
    for matchup in ("human-policy", "human-base"):
        values = [
            value
            for value in judgments
            if value["matchup"] == matchup
            and value["dimension"] == "target_likeness"
        ]
        counts = Counter(value["choice"] for value in values)
        model = "policy" if matchup == "human-policy" else "base"
        result[matchup] = {
            "human": counts["human"],
            model: counts[model],
            "tie": counts["tie"],
            "neither": counts["neither"],
            "human_decisive_rate": _cluster_interval(
                values, "human", model, config
            ),
        }
    return result


def _position_rate(judgments: Sequence[Mapping[str, Any]]) -> float | None:
    decisive = [value for value in judgments if value["position"] in {"a", "b"}]
    if not decisive:
        return None
    return sum(value["position"] == "a" for value in decisive) / len(decisive)


def _agreement(
    judgments: Sequence[Mapping[str, Any]], dimension: str
) -> float | None:
    if len({value["rater_id"] for value in judgments}) < 2:
        return None
    units: dict[str, list[str]] = {}
    for value in judgments:
        if (
            value["matchup"] != "policy-base"
            or value["dimension"] != dimension
        ):
            continue
        units.setdefault(value["ballot_id"], []).append(value["choice"])
    comparable = [values for values in units.values() if len(values) > 1]
    if not comparable:
        return None
    categories: Counter[str] = Counter()
    observed_disagreement = 0.0
    total = 0
    for values in comparable:
        counts = Counter(values)
        size = len(values)
        categories.update(counts)
        total += size
        observed_disagreement += (
            size * size - sum(count * count for count in counts.values())
        ) / (size - 1)
    observed = observed_disagreement / total
    if total < 2:
        return None
    expected = (
        total * total - sum(count * count for count in categories.values())
    ) / (total * (total - 1))
    return 1.0 - observed / expected if expected else 1.0


def _cluster_interval(
    values: Sequence[Mapping[str, Any]],
    positive: str,
    negative: str,
    config: EvalConfig,
) -> Mapping[str, Any] | None:
    decisive = [value for value in values if value["choice"] in {positive, negative}]
    if not decisive:
        return None
    examples = sorted({_required_text(value["example_id"]) for value in values})
    raters = sorted({_required_text(value["rater_id"]) for value in values})
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for value in values:
        key = (
            _required_text(value["example_id"]),
            _required_text(value["rater_id"]),
        )
        cells.setdefault(key, []).append(value)
    rng = random.Random(config.bootstrap_seed)
    estimates = []
    attempts = 0
    while len(estimates) < config.bootstrap_samples:
        attempts += 1
        example_sample = [examples[rng.randrange(len(examples))] for _ in examples]
        rater_sample = [raters[rng.randrange(len(raters))] for _ in raters]
        sample = [
            value
            for example in example_sample
            for rater in rater_sample
            for value in cells.get((example, rater), ())
            if value["choice"] in {positive, negative}
        ]
        if sample:
            estimates.append(
                sum(value["choice"] == positive for value in sample) / len(sample)
            )
        elif attempts >= config.bootstrap_samples * 20:
            break
    if not estimates:
        estimates.append(
            sum(value["choice"] == positive for value in decisive) / len(decisive)
        )
    estimates.sort()
    tail = (1.0 - config.confidence_level) / 2
    lower = estimates[min(len(estimates) - 1, int(tail * len(estimates)))]
    upper = estimates[
        min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))
    ]
    result: dict[str, Any] = asdict(
        Interval(
            sum(value["choice"] == positive for value in decisive) / len(decisive),
            lower,
            upper,
        )
    )
    result["method"] = "two-way-cluster-bootstrap"
    result["example_clusters"] = len(examples)
    result["rater_clusters"] = len(raters)
    return result


def _validate_panel_config(recipe: object, config: EvalConfig) -> None:
    if not isinstance(recipe, dict) or not isinstance(recipe.get("eval_config"), dict):
        raise EvaluationError("Evaluation recipe does not contain panel policy")
    recorded = recipe["eval_config"]
    names = (
        "bootstrap_seed",
        "bootstrap_samples",
        "confidence_level",
        "ballot_seed",
        "ballots_per_rater",
        "control_fraction",
        "min_panel_raters",
        "min_primary_comparisons",
    )
    for name in names:
        if recorded.get(name) != getattr(config, name):
            raise EvaluationError("Selected configuration does not match the evaluation")


def _verify_simple_artifact(
    path: Path, id_key: str, label: str
) -> Mapping[str, Any]:
    try:
        value = _read_manifest(path)
        identifier = _required_text(value[id_key])
        if path.name != identifier or not _is_digest(identifier):
            raise EvaluationError(f"{label} manifest does not match its path: {path}")
        if hashlib.sha256(_json_bytes(value["recipe"])).hexdigest() != identifier:
            raise EvaluationError(f"{label} recipe does not match its ID: {path}")
        files = value["files"]
        if not isinstance(files, dict):
            raise TypeError
        actual = {
            item.relative_to(path).as_posix()
            for item in path.rglob("*")
            if item.is_file() and item != path / "manifest.json"
        }
        if actual != set(files):
            raise EvaluationError(f"{label} files do not match its manifest: {path}")
        for name, expected in files.items():
            file_path = (path / _required_text(name)).resolve()
            if not file_path.is_relative_to(path.resolve()) or not file_path.is_file():
                raise EvaluationError(f"{label} manifest contains an invalid file path")
            if hashlib.sha256(file_path.read_bytes()).hexdigest() != expected:
                raise EvaluationError(f"{label} file does not match its manifest: {file_path}")
        return value
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, EvaluationError):
            raise
        raise EvaluationError(f"Cannot read {label.casefold()} artifact: {path}") from error


def _dataset_reference(path: Path) -> Any:
    from idiolect.data.local import load_dataset

    return load_dataset(path).dataset


def _read_manifest(path: Path) -> Mapping[str, Any]:
    value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError
    return value


def _source_path(values: object, name: str) -> Path:
    if not isinstance(values, dict):
        raise EvaluationError("Evaluation sources are not valid")
    return Path(_required_text(values.get(name)))


def _required_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError
    return value


def _evaluation_root(config: EvalConfig) -> Path:
    if config.output is None:
        raise EvaluationError("Evaluation output is not configured")
    return config.output.expanduser().resolve()


def _derived_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    try:
        values = tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Cannot read JSON Lines: {path}") from error
    if not all(isinstance(value, dict) for value in values):
        raise EvaluationError(f"JSON Lines rows are not objects: {path}")
    return values


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
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _utc_now() -> datetime:
    return datetime.now(UTC)
