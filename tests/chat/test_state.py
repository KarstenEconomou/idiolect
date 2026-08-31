"""Test interactive chat prompt and transcript behavior."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from idiolect.chat.discovery import Assistant
from idiolect.chat.state import (
    ChatSession,
    ChatStateError,
    ChatTurn,
    TurnTelemetry,
    derive_seed,
    enumerate_bubbles,
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


def test_prompt_records_stable_references_in_the_fitted_context_window() -> None:
    """Check BUFFER can report only bubbles kept in the active prompt."""
    state = _state(context=2)
    state.add_user("first")
    state.begin_generation()
    state.finish_generation(
        "one\n[new message]\ntwo",
        "stop",
        1,
        TurnTelemetry(10, 2),
    )
    state.add_user("newest")

    prepared = prepare_prompt(
        state,
        lambda value: sum(len(turn.content) for turn in value.turns),
        0,
    )

    assert prepared.dropped_messages == 1
    assert prepared.evicted_tokens > 0
    assert prepared.system_tokens > 0
    assert prepared.history_tokens > 0
    assert prepared.input_tokens > 0
    assert (
        prepared.system_tokens + prepared.history_tokens + prepared.input_tokens
        == prepared.prompt_tokens
    )
    assert prepared.active_turns == 2
    assert [
        (reference.index, reference.role, reference.content)
        for reference in prepared.active_references
    ] == [
        (1, "assistant", "one"),
        (2, "assistant", "two"),
        (3, "user", "newest"),
    ]


def test_env_turns_are_display_only_and_do_not_enter_prompt() -> None:
    """Check local ENV output leaves model context and user ordering intact."""
    state = _state()
    state.add_env("local diagnostic")
    state.add_user("next")

    prepared = prepare_prompt(state, lambda _value: 10, 0)

    assert "local diagnostic" not in prepared.prompt
    assert "next" in prepared.prompt
    assert [bubble.content for bubble in enumerate_bubbles(state.turns)] == ["next"]


def test_reference_numbers_assistant_bubbles_and_rejects_future_targets() -> None:
    """Check stable display numbering and backward-only references."""
    state = _state()
    state.add_user("first")
    state.begin_generation()
    state.finish_generation("one\n[new message]\ntwo", "stop", 1, TurnTelemetry(2, 1))

    assert [
        (bubble.index, bubble.role, bubble.content)
        for bubble in enumerate_bubbles(state.turns)
    ] == [
        (0, "user", "first"),
        (1, "assistant", "one"),
        (2, "assistant", "two"),
    ]

    state.add_user("reply", reference=2)
    assert state.turns[-1].reference == 2

    with pytest.raises(ChatStateError, match="earlier chat bubble"):
        ChatSession(
            state.assistant,
            state.chat,
            state.generation,
            (
                ChatTurn("user", "first"),
                ChatTurn("assistant", "one"),
                ChatTurn("user", "future", reference=99),
            ),
        )


def test_construct_prompt_adds_signal_reply_metadata_for_reference() -> None:
    """Check the selected bubble becomes a quoted Signal-style header."""
    base = _state()
    assistant = replace(
        base.assistant,
        run=SimpleNamespace(data=base.assistant.data),
        base_model=None,
        base_data=None,
    )
    state = ChatSession(assistant, base.chat, base.generation)
    state.add_user("first")
    state.begin_generation()
    state.finish_generation("answer", "stop", 1, TurnTelemetry(2, 1))
    state.add_user("follow-up", reference=1)

    prepared = prepare_prompt(state, lambda _value: 10, 0)

    assert '[person_01 | reply to DIXIE: "answer"]' in prepared.prompt


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

    with pytest.raises(ChatStateError, match="requires a retry"):
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


def test_participant_name_is_frozen_after_the_first_model_turn() -> None:
    """Check an OP override changes prompts only before conversation starts."""
    state = _state()
    state.add_env("Local notice")
    state.set_participant_name(" analyst ")
    state.add_user("hello")

    prepared = prepare_prompt(state, lambda _value: 1, 0)

    assert state.chat.participant_name == "analyst"
    assert "[analyst]\nhello" in prepared.prompt
    assert not state.participant_name_editable
    with pytest.raises(ChatStateError, match="after first turn"):
        state.set_participant_name("other")


@pytest.mark.parametrize("value", ("", "bad[name", "bad|name", "bad\nname"))
def test_participant_name_override_rejects_invalid_text(value: str) -> None:
    """Check an OP override keeps the prompt-header grammar valid."""
    with pytest.raises(ChatStateError, match="must contain|reserved"):
        _state().set_participant_name(value)


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
        "IDIOLECT // DIXIE::BASE [Qwen]",
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
