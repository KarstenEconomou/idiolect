"""Test fixed model text formatting."""

import pytest

from idiolect.config import TrainDataConfig
from idiolect.prompt import (
    MESSAGE_BOUNDARY,
    PromptError,
    format_prompt,
    join_bubbles,
    split_bubbles,
)


@pytest.mark.parametrize(
    "config",
    (
        TrainDataConfig(format="unknown"),
        TrainDataConfig(
            format="chat",
            prompt_role="invalid",
            completion_role="assistant",
        ),
        TrainDataConfig(format="completion", system_prompt="ignored value"),
    ),
)
def test_prompt_rejects_invalid_or_ignored_format_values(
    config: TrainDataConfig,
) -> None:
    """Check that invalid text policies stop before tokenization."""
    with pytest.raises(PromptError):
        format_prompt("context", config)


def test_bubble_serialization_round_trips_exact_text() -> None:
    """Check that joining and splitting preserves every bubble exactly."""
    bubbles = ("first bubble", "", "multi\nline\nbubble", "last")

    serialized = join_bubbles(bubbles)

    assert split_bubbles(serialized) == bubbles


def test_bubble_serialization_rejects_ambiguous_boundary_lines() -> None:
    """Check that source text cannot forge a message boundary."""
    for text in (
        MESSAGE_BOUNDARY,
        f"one\n{MESSAGE_BOUNDARY}\ntwo",
        f" {MESSAGE_BOUNDARY} ",
    ):
        with pytest.raises(PromptError, match="reserved boundary"):
            join_bubbles((text,))

    assert join_bubbles(("one two",)) == "one two"
    assert split_bubbles("one two") == ("one two",)
