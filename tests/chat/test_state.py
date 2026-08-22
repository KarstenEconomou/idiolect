"""Test interactive chat prompt and transcript behavior."""

import pytest

from idiolect.chat.discovery import Assistant
from idiolect.chat.state import (
    ChatSession,
    ChatStateError,
    ChatTurn,
    TurnTelemetry,
    derive_seed,
    prepare_prompt,
)
from idiolect.config import ChatConfig, GenerationConfig, TrainDataConfig
from idiolect.model import ModelSpec


def test_prompt_uses_training_grammar_and_drops_whole_old_messages() -> None:
    """Check participant names, recorded window, token fit, and newest input."""
    state = _state(context=3, prompt_limit=210)
    state.add_user("first")
    state.begin_generation()
    state.finish_generation("second", "stop", 1, TurnTelemetry(10, 1))
    state.add_user("newest")

    prepared = prepare_prompt(
        state, lambda value: sum(len(turn.content) for turn in value.turns), 0
    )

    assert "[person_01]\nnewest" in prepared.prompt
    assert "[DIXIE]\nsecond" in prepared.prompt
    assert prepared.prompt.endswith("[next response]")
    assert prepared.value.turns[0].content == "Construct persona."
    assert prepared.value.turns[-1].content == "<think>\n\n</think>\n\n"
    assert prepared.seed == derive_seed(101, prepared.prompt_digest, 0)


def test_prompt_rejects_newest_message_that_cannot_fit() -> None:
    """Check that fitting never splits or removes the newest user input."""
    state = _state(context=4, prompt_limit=1)
    state.add_user("cannot fit")

    with pytest.raises(ChatStateError, match="newest user message"):
        prepare_prompt(state, lambda _value: 2, 0)


def test_retry_replaces_latest_reply_and_changes_attempt_seed() -> None:
    """Check retry state and deterministic attempt-specific seeds."""
    state = _state()
    state.add_user("hello")
    state.begin_generation()
    state.finish_generation("partial", "cancelled", 4, TurnTelemetry(8, 2))

    attempt = state.retry()
    first = prepare_prompt(state, lambda _value: 10, 0)
    retried = prepare_prompt(state, lambda _value: 10, attempt)

    assert attempt == 1
    assert state.turns[-1].content == "hello"
    assert first.seed != retried.seed


def test_pending_user_message_rejects_another_user_turn() -> None:
    """Check that a failed generation cannot create consecutive user turns."""
    state = _state()
    state.add_user("pending")
    state.begin_generation()
    state.generating = False

    with pytest.raises(ChatStateError, match="requires /retry"):
        state.add_user("must not append")

    assert [turn.content for turn in state.turns] == ["pending"]


def test_session_rejects_invalid_restored_turn_order() -> None:
    """Check that invalid transcript state cannot reach snapshot writing."""
    state = _state()

    with pytest.raises(ChatStateError, match="must alternate"):
        ChatSession(
            state.assistant,
            state.chat,
            state.generation,
            (ChatTurn("user", "one"), ChatTurn("user", "two")),
        )


def _state(context: int = 4, prompt_limit: int = 500) -> ChatSession:
    """Return one synthetic chat state without local artifacts."""
    data = TrainDataConfig(
        format="chat",
        system_prompt="Construct persona.",
        prompt_role="user",
        completion_role="assistant",
        prompt_suffix="\n/no_think",
        completion_prefix="<think>\n\n</think>\n\n",
    )
    assistant = Assistant(
        "IDIOLECT // DIXIE@BASE [Qwen]",
        "DIXIE",
        "Qwen",
        None,
        None,
        context,
        ModelSpec("org/Qwen", "hub", "fixed", None, False),
        data,
        "a" * 64,
    )
    return ChatSession(
        assistant,
        ChatConfig(
            seed=101,
            participant_name="person_01",
            context_policy="recorded-window-drop-oldest",
            history="explicit-save",
        ),
        GenerationConfig(max_prompt_tokens=prompt_limit),
    )
