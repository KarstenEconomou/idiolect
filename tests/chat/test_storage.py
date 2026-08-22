"""Test immutable local chat snapshots."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest

from idiolect.chat.discovery import Assistant
from idiolect.chat.state import ChatSession, TurnTelemetry
from idiolect.chat.storage import ChatStorageError, ChatStore
from idiolect.config import ChatConfig, GenerationConfig, TrainDataConfig


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


def _state(tmp_path):
    """Return one synthetic state with complete recorded identity fields."""
    run_path = tmp_path / "runs" / ("a" * 64)
    dataset_path = tmp_path / "data" / ("b" * 64)
    assistant = SimpleNamespace(
        name="IDIOLECT // Karsten@aaaaaaaa [Qwen3-14B-4bit]",
        target_name="Karsten",
        context_messages=4,
        run=SimpleNamespace(
            ref=SimpleNamespace(id="a" * 64, path=run_path),
            data=TrainDataConfig(
                format="chat", prompt_role="user", completion_role="assistant"
            ),
            model=SimpleNamespace(name="org/Qwen3-14B-4bit", revision="fixed"),
            model_digest="c" * 64,
            adapter_digest="d" * 64,
        ),
        dataset=SimpleNamespace(
            dataset=SimpleNamespace(id="b" * 64, path=dataset_path)
        ),
    )
    chat = ChatConfig(
        output=tmp_path / "chat",
        seed=101,
        participant_name="person_01",
        context_policy="recorded-window-drop-oldest",
        history="explicit-save",
    )
    typed = cast(Assistant, assistant)
    return ChatSession(typed, chat, GenerationConfig(max_prompt_tokens=100)), typed
