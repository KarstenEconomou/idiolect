"""Test immutable local chat snapshots."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from idiolect.artifact import canonical_json_bytes
from idiolect.chat.discovery import Assistant
from idiolect.chat.state import ChatSession, TurnTelemetry
from idiolect.chat.storage import ChatStorageError, ChatStore
from idiolect.config import ChatConfig, GenerationConfig, TrainDataConfig
from idiolect.data.local import BuildResult
from idiolect.model import ModelSpec
from idiolect.train.base import LoadedRun
from idiolect.types import DatasetId, DatasetRef, PersonId, RunId, RunRef


def test_snapshot_is_private_idempotent_and_creates_lineage(
    tmp_path, monkeypatch
) -> None:
    """Check identity, permissions, titles, unchanged saves, and child snapshots."""
    state, assistant = _state(tmp_path)
    monkeypatch.setattr(
        "idiolect.chat.storage.load_assistant", lambda *_args: assistant
    )
    store = ChatStore(
        tmp_path / "chat", clock=lambda: datetime(2026, 8, 21, tzinfo=UTC)
    )
    state.add_user("A private first message with a default title")
    state.begin_generation()
    state.finish_generation("reply", "stop", 7, TurnTelemetry(12, 2))

    first = store.save(state)
    repeated = store.save(state)
    state.add_user("follow-up")
    child = store.save(state, "Explicit title")

    assert repeated.id == first.id
    assert child.parent_id == first.id
    assert child.title == "Explicit title"
    assert child.path.stat().st_mode & 0o777 == 0o700
    assert (child.path / "manifest.json").stat().st_mode & 0o777 == 0o600
    assert [chat.id for chat in store.leaves()] == [child.id]


def test_snapshot_rejects_changed_turn_content(tmp_path, monkeypatch) -> None:
    """Check that transcript tampering stops resume."""
    state, assistant = _state(tmp_path)
    monkeypatch.setattr(
        "idiolect.chat.storage.load_assistant", lambda *_args: assistant
    )
    store = ChatStore(tmp_path / "chat")
    state.add_user("private")
    saved = store.save(state)
    turns = saved.path / "turns.jsonl"
    row = json.loads(turns.read_text(encoding="utf-8"))
    row["content"] = "changed"
    turns.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ChatStorageError, match="does not match"):
        store.load(saved.id)


def test_base_snapshot_restores_system_persona_and_model_digest(tmp_path) -> None:
    """Check that a base chat resumes with its fixed persona and model."""
    state, _assistant = _base_state(tmp_path)
    store = ChatStore(tmp_path / "chat")
    state.add_user("private")

    saved = store.save(state)
    resumed = store.resume(saved.id)

    assert resumed.assistant.run is None
    assert resumed.assistant.model_digest == "c" * 64
    assert resumed.assistant.data.system_prompt == "Be concise."


def test_erase_removes_only_a_verified_lineage_leaf(tmp_path, monkeypatch) -> None:
    """Check safe leaf erasure and parent reappearance."""
    state, assistant = _state(tmp_path)
    monkeypatch.setattr(
        "idiolect.chat.storage.load_assistant", lambda *_args: assistant
    )
    store = ChatStore(tmp_path / "chat")
    state.add_user("first")
    state.begin_generation()
    state.finish_generation("reply", "stop", 7, TurnTelemetry(2, 1))
    first = store.save(state)
    state.add_user("second")
    child = store.save(state)

    with pytest.raises(ChatStorageError, match="has a child"):
        store.erase(first.id)

    renamed = store.rename(child.id, "Renamed TRACE")

    assert child.path.exists() is False
    assert renamed.title == "Renamed TRACE"
    assert renamed.parent_id == first.id
    assert [chat.id for chat in store.leaves()] == [renamed.id]

    store.erase(renamed.id)

    assert [chat.id for chat in store.leaves()] == [first.id]


def test_snapshot_records_and_restores_backend_versions(tmp_path, monkeypatch) -> None:
    """Check that runtime versions survive the snapshot round trip."""
    state, assistant = _state(tmp_path)
    monkeypatch.setattr(
        "idiolect.chat.storage.load_assistant", lambda *_args: assistant
    )
    store = ChatStore(tmp_path / "chat")
    state.add_user("private")

    saved = store.save(state, backend_versions={"mlx_version": "0.29.0"})
    loaded = store.load(saved.id)

    assert saved.backend_versions == {"mlx_version": "0.29.0"}
    assert loaded.backend_versions == {"mlx_version": "0.29.0"}


def test_snapshot_rejects_an_invalid_recorded_generation_policy(
    tmp_path, monkeypatch
) -> None:
    """Check that a self-consistent snapshot cannot smuggle bad sampling values."""
    state, assistant = _state(tmp_path)
    monkeypatch.setattr(
        "idiolect.chat.storage.load_assistant", lambda *_args: assistant
    )
    store = ChatStore(tmp_path / "chat")
    state.add_user("private")
    saved = store.save(state)

    manifest = json.loads((saved.path / "manifest.json").read_text(encoding="utf-8"))
    manifest["generation_policy"]["temperature"] = -5.0
    new_id = _readdress(saved.path, manifest)

    with pytest.raises(ChatStorageError, match="generation policy is not valid"):
        store.load(new_id)


def _readdress(path: Path, manifest: dict) -> str:
    """Rewrite one snapshot in place under its recomputed content address."""
    identity_keys = (
        "version",
        "assistant",
        "chat_policy",
        "generation_policy",
        "title",
        "parent_chat_id",
        "turn_count",
        "files",
        "backend_versions",
    )
    identity = {key: manifest[key] for key in identity_keys if key in manifest}
    chat_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    manifest["chat_id"] = chat_id
    destination = path.parent / chat_id
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path.rename(destination)
    return chat_id


def _state(tmp_path):
    """Return one synthetic state with complete recorded identity fields."""
    created = datetime(2026, 8, 21, tzinfo=UTC)
    run_id = RunId("a" * 64)
    dataset_id = DatasetId("b" * 64)
    model = ModelSpec("org/Qwen3-14B-4bit", "hub", "fixed", tmp_path / "models", False)
    data = TrainDataConfig(
        format="chat",
        prompt_role="user",
        completion_role="assistant",
    )
    run = LoadedRun(
        RunRef(run_id, dataset_id, tmp_path / "runs" / run_id, created),
        model,
        "c" * 64,
        data,
        tmp_path / "runs" / run_id / "adapter",
        "d" * 64,
        {},
        17,
        2048,
    )
    dataset = BuildResult(
        DatasetRef(
            dataset_id,
            PersonId("person"),
            tmp_path / "data" / dataset_id,
            created,
        ),
        {},
    )
    assistant = Assistant(
        "IDIOLECT // DIXIE@aaaaaaaa [Qwen3-14B-4bit]",
        "DIXIE",
        "Qwen3-14B-4bit",
        run,
        dataset,
        4,
    )
    return _session(tmp_path, assistant), assistant


def _base_state(tmp_path):
    """Return one synthetic configured base assistant state."""
    data = TrainDataConfig(
        format="chat",
        system_prompt="Be concise.",
        prompt_role="user",
        completion_role="assistant",
    )
    assistant = Assistant(
        "IDIOLECT // DIXIE@BASE [Qwen3-14B-4bit]",
        "DIXIE",
        "Qwen3-14B-4bit",
        None,
        None,
        4,
        ModelSpec("org/Qwen3-14B-4bit", "hub", "fixed", tmp_path / "models", False),
        data,
        "c" * 64,
    )
    return _session(tmp_path, assistant), assistant


def _session(tmp_path, assistant):
    """Return one synthetic session for an assistant."""
    chat = ChatConfig(
        output=tmp_path / "chat",
        seed=101,
        participant_name="person_01",
        context_policy="recorded-window-drop-oldest",
        history="explicit-save",
    )
    return ChatSession(assistant, chat, GenerationConfig(max_prompt_tokens=100))
