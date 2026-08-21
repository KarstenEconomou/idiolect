"""Test local model policy evaluation."""

import hashlib
import json
import math
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

from idiolect.config import EvalConfig, InferConfig, TrainDataConfig
from idiolect.eval.base import CompletionScore
from idiolect.eval.local import EvaluationError, LocalEvaluator, load_evaluation
from idiolect.eval.panel import collect_judgments, create_panel
from idiolect.eval.text import TrainingMatchIndex
from idiolect.infer.base import ModelTarget, Prediction, TargetMode
from idiolect.model import ModelSpec
from idiolect.train.base import LoadedRun
from idiolect.types import (
    DatasetId,
    DatasetRef,
    InferenceId,
    InferenceRef,
    PersonId,
    RunId,
    RunRef,
    Split,
)

_NOW = datetime(2026, 8, 20, tzinfo=UTC)
_EVAL_KEYS = frozenset(
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


class FakeScoreSession:
    """Return fixed completion costs for one target."""

    def __init__(self, cost: float) -> None:
        """Set the per-token cost."""
        self.cost = cost
        self.closed = False

    def score(self, prompt: object, completion: str) -> CompletionScore:
        """Return one deterministic completion cost."""
        assert completion
        return CompletionScore(8, 2, self.cost * 2)

    def close(self) -> None:
        """Record that the session closed."""
        self.closed = True


class FakeScoreBackend:
    """Create deterministic scoring sessions."""

    def __init__(self) -> None:
        """Create an empty session record."""
        self.sessions: list[FakeScoreSession] = []

    @property
    def version(self) -> str:
        """Return one fixed backend version."""
        return "test-score-1"

    def load(self, target: ModelTarget) -> FakeScoreSession:
        """Return a better score for adapters than for the base."""
        cost = 2.0 if target.adapter_path is None else 1.0
        session = FakeScoreSession(cost)
        self.sessions.append(session)
        return session


class VariableScoreSession:
    """Return completion costs with unequal token counts."""

    def __init__(self, adapter: bool) -> None:
        """Set whether this session scores an adapter."""
        self.adapter = adapter

    def score(self, prompt: object, completion: str) -> CompletionScore:
        """Return one score that separates macro and corpus means."""
        tokens = 1 if completion.endswith("0") else 9
        cost = 2.0 if self.adapter else 1.0 if tokens == 1 else 3.0
        return CompletionScore(8, tokens, cost * tokens)

    def close(self) -> None:
        """Release no resources."""


class VariableScoreBackend:
    """Create scoring sessions for a corpus-perplexity check."""

    @property
    def version(self) -> str:
        """Return one fixed backend version."""
        return "test-variable-score-1"

    def load(self, target: ModelTarget) -> VariableScoreSession:
        """Return one session for the selected target."""
        return VariableScoreSession(target.adapter_path is not None)


class FakeInferencer:
    """Write aligned synthetic inference artifacts."""

    def __init__(self, root: Path, leaked: bool = False) -> None:
        """Set the private output directory."""
        self.root = root
        self.leaked = leaked

    def dataset(
        self,
        target: ModelTarget,
        dataset: DatasetRef,
        split: Split,
        config: InferConfig,
    ) -> InferenceRef:
        """Write fixed predictions for every selected row and seed."""
        rows = _dataset_rows(dataset)
        recipe = {"target": target.id}
        identity = hashlib.sha256(_json_bytes(recipe)).hexdigest()
        path = self.root / identity
        path.mkdir(parents=True, exist_ok=True)
        values = []
        for index, prompt, _ in rows:
            example_id = _example_id(dataset, index, prompt)
            for seed in config.seeds:
                if target.adapter_path is None:
                    text = "Base response"
                elif self.leaked:
                    text = "private training phrase that is deliberately very long"
                else:
                    text = "personal response!"
                values.append(
                    Prediction(
                        example_id,
                        index,
                        seed,
                        seed,
                        text,
                        "stop",
                        8,
                        3,
                    )
                )
        prediction_path = path / "pred.jsonl"
        with prediction_path.open("w", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(asdict(value)) + "\n")
        manifest = {
            "inference_id": identity,
            "created_at": _NOW.isoformat(),
            "recipe": recipe,
            "counts": {"examples": len(rows), "predictions": len(values)},
            "files": {
                "pred.jsonl": hashlib.sha256(prediction_path.read_bytes()).hexdigest()
            },
        }
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return InferenceRef(InferenceId(identity), path, _NOW, len(values))


def test_policy_evaluation_is_paired_private_and_content_addressed(
    tmp_path: Path,
) -> None:
    """Check policy evidence, gates, reuse, and artifact verification."""
    dataset = _dataset(tmp_path)
    runs = _runs(tmp_path, dataset)
    scorer = FakeScoreBackend()
    evaluator = LocalEvaluator(
        scorer,
        FakeInferencer(tmp_path / "infer"),
        target_loader=_target,
        clock=lambda: _NOW,
    )

    first = evaluator.evaluate(runs, dataset, _eval_config(tmp_path), _infer_config(tmp_path))
    second = evaluator.evaluate(runs, dataset, _eval_config(tmp_path), _infer_config(tmp_path))

    assert first == second
    assert first.eligible is True
    assert len(scorer.sessions) == 3
    assert all(session.closed for session in scorer.sessions)
    report = json.loads((first.path / "metrics.json").read_text(encoding="utf-8"))
    assert (
        report["likelihood"]["policy"]["delta_macro_mean_nll"]["value"]
        == -1.0
    )
    assert report["behavior"]["policy"]["empty_rate"] == 0.0
    artifact_text = "".join(
        path.read_text(encoding="utf-8") for path in first.path.iterdir()
    )
    assert "private validation prompt" not in artifact_text
    assert first.path.stat().st_mode & 0o777 == 0o700

    (first.path / "metrics.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="does not match its manifest"):
        load_evaluation(first.path)


def test_policy_requires_complete_seed_set_and_gates_new_memorization(
    tmp_path: Path,
) -> None:
    """Check policy grouping and adapter-only training text leakage."""
    dataset = _dataset(tmp_path)
    runs = _runs(tmp_path, dataset)
    evaluator = LocalEvaluator(
        FakeScoreBackend(),
        FakeInferencer(tmp_path / "infer", leaked=True),
        target_loader=_target,
        clock=lambda: _NOW,
    )

    with pytest.raises(EvaluationError, match="every configured training seed"):
        evaluator.evaluate(
            runs[:1], dataset, _eval_config(tmp_path), _infer_config(tmp_path)
        )

    result = evaluator.evaluate(
        runs, dataset, _eval_config(tmp_path), _infer_config(tmp_path)
    )
    assert result.eligible is False
    report = json.loads((result.path / "metrics.json").read_text(encoding="utf-8"))
    assert report["gates"]["incremental_memorization"]["passed"] is False


def test_perplexity_uses_all_completion_tokens(tmp_path: Path) -> None:
    """Check that corpus perplexity weights each completion token equally."""
    dataset = _dataset(tmp_path)
    result = LocalEvaluator(
        VariableScoreBackend(),
        FakeInferencer(tmp_path / "infer"),
        target_loader=_target,
        clock=lambda: _NOW,
    ).evaluate(
        _runs(tmp_path, dataset),
        dataset,
        _eval_config(tmp_path),
        _infer_config(tmp_path),
    )

    report = json.loads((result.path / "metrics.json").read_text(encoding="utf-8"))
    base = report["likelihood"]["base"]
    assert base["macro_mean_nll"] == 2.0
    assert base["corpus_mean_nll"] == 2.8
    assert base["corpus_perplexity"] == pytest.approx(math.exp(2.8))


def test_sparse_match_index_finds_unaligned_long_text() -> None:
    """Check a long match whose start is not a sampled seed boundary."""
    shared = "a distinctive private phrase with enough characters"
    training = "prefix material before " + shared + " and material after"
    candidate = "new opening " + shared + " with a different ending"

    index = TrainingMatchIndex.build((training,), 32)

    assert index.longest(candidate) >= len(shared)


def test_familiar_panel_keeps_separate_immutable_rater_artifacts(
    tmp_path: Path,
) -> None:
    """Check blind sessions, pseudonymous judgments, and panel completion."""
    dataset = _dataset(tmp_path)
    config = _eval_config(tmp_path)
    evaluation = LocalEvaluator(
        FakeScoreBackend(),
        FakeInferencer(tmp_path / "infer"),
        target_loader=_target,
        clock=lambda: _NOW,
    ).evaluate(_runs(tmp_path, dataset), dataset, config, _infer_config(tmp_path))

    answers = ["a", "a", "a", "b", "b", "b"]
    first = collect_judgments(
        evaluation.path,
        "rater-01",
        config,
        read=lambda _: answers.pop(0),
        write=lambda _: None,
        clock=lambda: _NOW,
    )
    answers = ["a", "a", "a", "b", "b", "b"]
    second = collect_judgments(
        evaluation.path,
        "rater-02",
        config,
        read=lambda _: answers.pop(0),
        write=lambda _: None,
        clock=lambda: _NOW,
    )
    panel = create_panel(
        evaluation.path,
        (first.path, second.path),
        config,
        clock=lambda: _NOW,
    )

    assert first.path != second.path
    assert panel.complete is True
    judgment_text = (first.path / "judgments.jsonl").read_text(encoding="utf-8")
    assert "private validation prompt" not in judgment_text
    summary = json.loads((panel.path / "panel.json").read_text(encoding="utf-8"))
    assert summary["raters"] == 2
    assert summary["primary_comparisons"] == 2
    interval = summary["preferences"]["target_likeness"][
        "policy_decisive_rate"
    ]
    assert interval["method"] == "two-way-cluster-bootstrap"
    assert interval["example_clusters"] == 1
    assert interval["rater_clusters"] == 2
    assert set(summary["krippendorff_alpha"]) == {
        "target_likeness",
        "voice",
        "context_fit",
    }


def test_panel_rejects_a_self_consistent_judgment_off_schedule(
    tmp_path: Path,
) -> None:
    """Check that artifact hashes cannot authorize a changed ballot answer."""
    dataset = _dataset(tmp_path)
    config = _eval_config(tmp_path)
    evaluation = LocalEvaluator(
        FakeScoreBackend(),
        FakeInferencer(tmp_path / "infer"),
        target_loader=_target,
        clock=lambda: _NOW,
    ).evaluate(_runs(tmp_path, dataset), dataset, config, _infer_config(tmp_path))
    answers = ["a", "a", "a", "b", "b", "b"]
    judgment = collect_judgments(
        evaluation.path,
        "rater-01",
        config,
        read=lambda _: answers.pop(0),
        write=lambda _: None,
        clock=lambda: _NOW,
    )
    forged = _forge_judgment(tmp_path, judgment.path)

    with pytest.raises(EvaluationError, match="does not match its ballot"):
        create_panel(evaluation.path, (forged,), config, clock=lambda: _NOW)


def _dataset(tmp_path: Path) -> DatasetRef:
    recipe = {"schema_version": 1, "target_id": "target", "source": "synthetic"}
    dataset_id = hashlib.sha256(_json_bytes(recipe)).hexdigest()
    path = tmp_path / "data" / dataset_id
    path.mkdir(parents=True)
    split_rows = {
        "train": [
            {
                "prompt": "private training prompt",
                "completion": "private training phrase that is deliberately very long",
            }
        ],
        "valid": [
            {
                "prompt": f"private validation prompt {index}",
                "completion": f"human reply {index}",
            }
            for index in range(2)
        ],
    }
    files = {}
    for name, rows in split_rows.items():
        content = "".join(f"{json.dumps(row, separators=(',', ':'))}\n" for row in rows)
        (path / f"{name}.jsonl").write_text(content, encoding="utf-8")
        files[f"{name}.jsonl"] = hashlib.sha256(content.encode()).hexdigest()
    manifest = {
        "dataset_id": dataset_id,
        "created_at": _NOW.isoformat(),
        "recipe": recipe,
        "counts": {"train": 1, "valid": 2, "test": 0},
        "files": files,
        "pseudonyms": {},
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return DatasetRef(DatasetId(dataset_id), PersonId("target"), path, _NOW)


def _runs(tmp_path: Path, dataset: DatasetRef) -> tuple[LoadedRun, ...]:
    data = TrainDataConfig(
        format="chat",
        prompt_role="user",
        completion_role="assistant",
    )
    policy = {"seeds": [17, 42], "max_seq_length": 128}
    model = ModelSpec("model", "path", "fixed", None, False)
    values = []
    for seed in (17, 42):
        path = tmp_path / f"run-{seed}"
        adapter = path / "adapter"
        adapter.mkdir(parents=True)
        values.append(
            LoadedRun(
                RunRef(RunId(f"run-{seed}"), dataset.id, path, _NOW),
                model,
                "model-digest",
                data,
                adapter,
                f"adapter-{seed}",
                policy,
                seed,
                128,
            )
        )
    return tuple(values)


def _target(path: Path, adapter: bool) -> ModelTarget:
    return ModelTarget(
        f"{path.name}:{adapter}",
        TargetMode.RUN_ADAPTER if adapter else TargetMode.RUN_BASE,
        path,
        "model-digest",
        TrainDataConfig(
            format="chat",
            prompt_role="user",
            completion_role="assistant",
        ),
        adapter_path=path / "adapter" if adapter else None,
    )


def _eval_config(tmp_path: Path) -> EvalConfig:
    return EvalConfig(
        output=tmp_path / "eval",
        backend="mlx-lm",
        suite="fidelity",
        split="valid",
        max_examples=2,
        bootstrap_seed=7,
        bootstrap_samples=20,
        confidence_level=0.95,
        long_match_chars=12,
        max_empty_rate=0.0,
        max_format_violation_rate=0.0,
        max_truncation_rate=0.0,
        max_memorization_rate_delta=0.0,
        ballot_seed=11,
        ballots_per_rater=2,
        control_fraction=0.5,
        min_panel_raters=2,
        min_primary_comparisons=2,
        specified=_EVAL_KEYS,
    )


def _infer_config(tmp_path: Path) -> InferConfig:
    return InferConfig(
        output=tmp_path / "infer",
        backend="mlx-lm",
        seeds=(101, 202),
        max_examples=2,
        max_prompt_tokens=100,
        max_tokens=16,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        min_tokens_to_keep=1,
        repetition_penalty=1.0,
        repetition_context_size=20,
    )


def _dataset_rows(dataset: DatasetRef) -> list[tuple[int, str, str]]:
    return [
        (index, value["prompt"], value["completion"])
        for index, line in enumerate(
            (dataset.path / "valid.jsonl").read_text(encoding="utf-8").splitlines()
        )
        for value in (json.loads(line),)
    ]


def _example_id(dataset: DatasetRef, index: int, prompt: str) -> str:
    identity = {
        "dataset_id": str(dataset.id),
        "split": "valid",
        "index": index,
        "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    return hashlib.sha256(_json_bytes(identity)).hexdigest()


def _forge_judgment(tmp_path: Path, source: Path) -> Path:
    staging = tmp_path / "forged-judgment"
    shutil.copytree(source, staging)
    judgment_path = staging / "judgments.jsonl"
    rows = [
        json.loads(line)
        for line in judgment_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["example_id"] = "0" * 64
    content = "".join(
        f"{json.dumps(row, separators=(',', ':'))}\n" for row in rows
    )
    judgment_path.write_text(content, encoding="utf-8")
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recipe"]["judgments"] = rows
    identity = hashlib.sha256(_json_bytes(manifest["recipe"])).hexdigest()
    manifest["judgment_id"] = identity
    manifest["files"]["judgments.jsonl"] = hashlib.sha256(
        content.encode()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "forged" / identity
    destination.parent.mkdir()
    staging.rename(destination)
    return destination


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
