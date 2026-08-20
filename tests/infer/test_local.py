"""Test reproducible local inference orchestration."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from idiolect.config import InferConfig, TrainDataConfig
from idiolect.infer.base import (
    BackendResult,
    ModelInput,
    ModelTarget,
    TargetMode,
)
from idiolect.infer.local import InferenceError, LocalInferencer
from idiolect.prompt import Turn
from idiolect.types import DatasetId, DatasetRef, PersonId, Split

_NOW = datetime(2026, 8, 20, tzinfo=UTC)
_INFER_KEYS = frozenset(
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


class FakeSession:
    """Record formatted prompts without model work."""

    def __init__(self) -> None:
        """Create an empty request record."""
        self.requests: list[tuple[ModelInput, int]] = []
        self.closed = False

    def count_tokens(self, value: ModelInput) -> int:
        """Return one stable synthetic token count."""
        return sum(len(turn.content) for turn in value.turns)

    def generate(
        self,
        value: ModelInput,
        seed: int,
        config: InferConfig,
    ) -> BackendResult:
        """Return one synthetic generation result."""
        self.requests.append((value, seed))
        return BackendResult(
            f"reply-{seed}",
            "stop",
            self.count_tokens(value),
            config.max_tokens,
        )

    def close(self) -> None:
        """Record that the session was closed."""
        self.closed = True


class FakeBackend:
    """Create recorded local generation sessions."""

    def __init__(self) -> None:
        """Create an empty session record."""
        self.sessions: list[FakeSession] = []

    @property
    def version(self) -> str:
        """Return one stable synthetic backend version."""
        return "test-1"

    def load(self, target: ModelTarget) -> FakeSession:
        """Create one session for the verified target."""
        session = FakeSession()
        self.sessions.append(session)
        return session


def test_text_uses_training_format_and_order_independent_seeds(tmp_path: Path) -> None:
    """Check one-shot formatting, seed records, and session reuse."""
    backend = FakeBackend()
    inferencer = LocalInferencer(backend, clock=lambda: _NOW)
    config = _config(tmp_path, seeds=(101, 202))
    target = _target(tmp_path)

    predictions = inferencer.text(target, "context", config)

    assert [value.seed for value in predictions] == [101, 202]
    assert predictions[0].rng_seed != predictions[1].rng_seed
    assert len(backend.sessions) == 1
    session = backend.sessions[0]
    assert session.closed is True
    assert [request[0] for request in session.requests] == [
        ModelInput(
            turns=(
                Turn("system", "Write one reply."),
                Turn("user", "[context]\n/no_think"),
                Turn("assistant", "<think>\n\n</think>\n\n"),
            ),
            has_prefill=True,
        )
    ] * 2


def test_dataset_writes_private_fixed_artifact_and_rejects_changes(
    tmp_path: Path,
) -> None:
    """Check selection, reuse, privacy, permissions, and file verification."""
    dataset = _dataset(tmp_path)
    backend = FakeBackend()
    inferencer = LocalInferencer(backend, clock=lambda: _NOW)
    config = _config(tmp_path, seeds=(101, 202), max_examples=2)

    first = inferencer.dataset(_target(tmp_path), dataset, Split.TEST, config)
    second = inferencer.dataset(_target(tmp_path), dataset, Split.TEST, config)

    assert first == second
    assert first.predictions == 4
    assert len(backend.sessions) == 1
    assert first.path.stat().st_mode & 0o777 == 0o700
    prediction_path = first.path / "pred.jsonl"
    assert prediction_path.stat().st_mode & 0o777 == 0o600
    artifact_text = "".join(
        path.read_text(encoding="utf-8") for path in first.path.iterdir()
    )
    assert "private prompt" not in artifact_text
    assert "expected reply" not in artifact_text

    prediction_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(InferenceError, match="does not match its manifest"):
        inferencer.dataset(_target(tmp_path), dataset, Split.TEST, config)


def test_prompt_overflow_closes_session_without_an_artifact(tmp_path: Path) -> None:
    """Check that inference rejects input instead of changing its context."""
    backend = FakeBackend()
    config = replace(_config(tmp_path), max_prompt_tokens=4)

    with pytest.raises(InferenceError, match="example 0"):
        LocalInferencer(backend).text(_target(tmp_path), "long context", config)

    assert backend.sessions[0].closed is True
    assert not (tmp_path / "infer").exists()


def _config(
    tmp_path: Path,
    seeds: tuple[int, ...] = (101,),
    max_examples: int = 0,
) -> InferConfig:
    return InferConfig(
        output=tmp_path / "infer",
        backend="mlx-lm",
        seeds=seeds,
        max_examples=max_examples,
        max_prompt_tokens=100,
        max_tokens=12,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        min_tokens_to_keep=1,
        repetition_penalty=1.0,
        repetition_context_size=20,
        specified=_INFER_KEYS,
    )


def _target(tmp_path: Path) -> ModelTarget:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    data = TrainDataConfig(
        format="chat",
        system_prompt="Write one reply.",
        prompt_role="user",
        completion_role="assistant",
        prompt_prefix="[",
        prompt_suffix="]\n/no_think",
        completion_prefix="<think>\n\n</think>\n\n",
    )
    return ModelTarget("target", TargetMode.CONFIG_BASE, model, "model-digest", data)


def _dataset(tmp_path: Path) -> DatasetRef:
    recipe = {
        "schema_version": 1,
        "target_id": "target",
        "source_digest": "synthetic",
    }
    dataset_id = hashlib.sha256(_json_bytes(recipe)).hexdigest()
    path = tmp_path / "data" / dataset_id
    path.mkdir(parents=True)
    rows = [
        {"prompt": f"private prompt {index}", "completion": "expected reply"}
        for index in range(4)
    ]
    content = "".join(
        f"{json.dumps(row, ensure_ascii=False, separators=(',', ':'))}\n"
        for row in rows
    )
    (path / "test.jsonl").write_text(content, encoding="utf-8")
    manifest = {
        "dataset_id": dataset_id,
        "created_at": _NOW.isoformat(),
        "recipe": recipe,
        "counts": {"test": 4},
        "files": {"test.jsonl": hashlib.sha256(content.encode()).hexdigest()},
        "pseudonyms": {},
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return DatasetRef(DatasetId(dataset_id), PersonId("target"), path, _NOW)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
