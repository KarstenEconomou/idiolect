"""Test reproducible local inference orchestration."""

import hashlib
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from idiolect.config import InferenceConfig, TrainConfig, TrainDataConfig
from idiolect.inference.base import (
    BackendResult,
    ModelInput,
    ModelTarget,
    TargetMode,
)
from idiolect.inference.local import (
    InferenceError,
    LocalInferencer,
    RecordedTargetResolver,
    configured_target,
    load_inference,
)
from idiolect.model import ModelSpec, directory_digest
from idiolect.prompt import Turn
from idiolect.train.base import LoadedRun
from idiolect.types import DatasetId, DatasetRef, PersonId, RunId, RunRef, Split

_NOW = datetime(2026, 8, 20, tzinfo=UTC)
_INFERENCE_KEYS = frozenset(
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
        config: InferenceConfig,
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


class InvalidSession(FakeSession):
    """Return invalid generation metadata."""

    def generate(
        self,
        value: ModelInput,
        seed: int,
        config: InferenceConfig,
    ) -> BackendResult:
        """Return one result with invalid finish and token values."""
        return BackendResult(
            "reply",
            "unknown",
            self.count_tokens(value),
            config.max_tokens + 1,
        )


class InvalidBackend(FakeBackend):
    """Create a backend that returns an invalid generation result."""

    def load(self, target: ModelTarget) -> FakeSession:
        """Create one session with an invalid generation method."""
        session = InvalidSession()
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
    assert not (tmp_path / "inference").exists()


def test_artifact_id_rejects_coordinated_prediction_change(tmp_path: Path) -> None:
    """Check that a changed prediction and checksum cannot keep one artifact ID."""
    result = LocalInferencer(FakeBackend(), clock=lambda: _NOW).dataset(
        _target(tmp_path),
        _dataset(tmp_path),
        Split.TEST,
        _config(tmp_path),
    )
    prediction_path = result.path / "pred.jsonl"
    prediction_path.write_text("changed\n", encoding="utf-8")
    manifest_path = result.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["pred.jsonl"] = hashlib.sha256(b"changed\n").hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InferenceError, match="content does not match its ID"):
        load_inference(result.path)


def test_artifact_reader_rejects_invalid_self_consistent_prediction(
    tmp_path: Path,
) -> None:
    """Check prediction schema after content identity verification."""
    result = LocalInferencer(FakeBackend(), clock=lambda: _NOW).dataset(
        _target(tmp_path),
        _dataset(tmp_path),
        Split.TEST,
        _config(tmp_path),
    )
    prediction_path = result.path / "pred.jsonl"
    rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["finish_reason"] = "invalid"
    content = "".join(
        f"{json.dumps(row, ensure_ascii=False, separators=(',', ':'))}\n"
        for row in rows
    )
    prediction_path.write_text(content, encoding="utf-8")
    manifest_path = result.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["pred.jsonl"] = hashlib.sha256(content.encode()).hexdigest()
    identity = {
        "recipe": manifest["recipe"],
        "counts": manifest["counts"],
        "files": manifest["files"],
    }
    inference_id = hashlib.sha256(_json_bytes(identity)).hexdigest()
    manifest["inference_id"] = inference_id
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path = result.path.rename(result.path.parent / inference_id)

    with pytest.raises(InferenceError, match="invalid finish reason"):
        load_inference(path)


def test_backend_result_validation_stops_invalid_artifact(tmp_path: Path) -> None:
    """Check that invalid backend metadata cannot enter one artifact."""
    backend = InvalidBackend()

    with pytest.raises(InferenceError, match="invalid finish reason"):
        LocalInferencer(backend).dataset(
            _target(tmp_path),
            _dataset(tmp_path),
            Split.TEST,
            _config(tmp_path),
        )

    assert backend.sessions[0].closed is True
    assert not (tmp_path / "inference").exists()


def test_configured_target_rejects_invalid_text_format_before_resolution() -> None:
    """Check base inference policy before model access."""
    called = False

    def resolver(spec: ModelSpec) -> Path:
        nonlocal called
        called = True
        return Path("unused")

    config = TrainConfig(
        base_model="model",
        model_source="path",
        data=TrainDataConfig(format="misspelled"),
    )

    with pytest.raises(InferenceError, match="format must be chat or completion"):
        configured_target(config, resolver)

    assert called is False


def test_recorded_target_resolver_verifies_shared_model_once(tmp_path: Path) -> None:
    """Check that one evaluation policy hashes its common model once."""
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "weights.bin").write_bytes(b"fixed")
    digest = directory_digest(model_path)
    model = ModelSpec("model", "path", "fixed", None, False)
    data = TrainDataConfig(
        format="chat",
        prompt_role="user",
        completion_role="assistant",
    )
    calls = 0

    def resolver(spec: ModelSpec) -> Path:
        nonlocal calls
        calls += 1
        assert spec == model
        return model_path

    target_resolver = RecordedTargetResolver(resolver)
    runs = []
    for seed in (17, 42):
        run_path = tmp_path / f"run-{seed}"
        adapter_path = run_path / "adapter"
        adapter_path.mkdir(parents=True)
        runs.append(
            LoadedRun(
                RunRef(RunId(str(seed)), DatasetId("dataset"), run_path, _NOW),
                model,
                digest,
                data,
                adapter_path,
                f"adapter-{seed}",
                {},
                seed,
                128,
            )
        )

    target_resolver.target(runs[0], False)
    target_resolver.target(runs[0], True)
    target_resolver.target(runs[1], True)

    assert calls == 1


def test_concurrent_artifact_publication_returns_verified_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check recovery when another process publishes equal content first."""
    dataset = _dataset(tmp_path)
    original_rename = Path.rename

    def race(source: Path, destination: Path) -> Path:
        if source.name.startswith(".inference-"):
            shutil.copytree(source, destination)
            raise FileExistsError
        return original_rename(source, destination)

    monkeypatch.setattr(Path, "rename", race)

    result = LocalInferencer(FakeBackend(), clock=lambda: _NOW).dataset(
        _target(tmp_path),
        dataset,
        Split.TEST,
        _config(tmp_path),
    )

    assert load_inference(result.path) == result


def _config(
    tmp_path: Path,
    seeds: tuple[int, ...] = (101,),
    max_examples: int = 0,
) -> InferenceConfig:
    return InferenceConfig(
        output=tmp_path / "inference",
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
        specified=_INFERENCE_KEYS,
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
        "target_name": "TARGET",
        "context": 4,
        "burst_gap_seconds": 120.0,
        "unit": "response-episode-v1",
        "source_digest": "synthetic",
    }
    content = "".join(
        f"{json.dumps(row, ensure_ascii=False, separators=(',', ':'))}\n"
        for row in (
            {"prompt": f"private prompt {index}", "completion": "expected reply"}
            for index in range(4)
        )
    )
    index_content = "".join(
        f"{json.dumps(row, ensure_ascii=False, separators=(',', ':'))}\n"
        for number, index in enumerate(range(4))
        for row in [
            {
                "split": "test",
                "index": index,
                "chat_id": "chat",
                "episode_id": f"message-{number:02d}",
                "target_message_ids": [f"message-{number:02d}"],
                "target_sent_at": (_NOW + timedelta(minutes=number)).isoformat(),
                "target_end_sent_at": (_NOW + timedelta(minutes=number)).isoformat(),
                "reply_parent_message_id": None,
                "thread_anchor_message_ids": [],
                "context_message_ids": [],
                "context_reaction_event_ids": [],
            }
        ]
    )
    counts = {"train": 0, "valid": 0, "test": 4}
    total = sum(counts.values())
    selection = {
        "attachment": 0,
        "deleted": 0,
        "edited": 0,
        "no_text": 0,
        "no_visible_text": 0,
        "target_episodes": total,
        "included": total,
        "unusable_episodes": 0,
        "authored_messages": total,
        "episode_messages_included": total,
        "episode_messages_excluded": 0,
    }
    files = {
        "test.jsonl": hashlib.sha256(content.encode()).hexdigest(),
        "index.jsonl": hashlib.sha256(index_content.encode()).hexdigest(),
    }
    identity = {
        "recipe": recipe,
        "counts": counts,
        "selection": selection,
        "files": files,
        "pseudonyms": {},
    }
    dataset_id = hashlib.sha256(_json_bytes(identity)).hexdigest()
    path = tmp_path / "data" / dataset_id
    path.mkdir(parents=True)
    (path / "test.jsonl").write_text(content, encoding="utf-8")
    (path / "index.jsonl").write_text(index_content, encoding="utf-8")
    manifest = {
        "dataset_id": dataset_id,
        "created_at": _NOW.isoformat(),
        **identity,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return DatasetRef(DatasetId(dataset_id), PersonId("target"), path, _NOW)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
