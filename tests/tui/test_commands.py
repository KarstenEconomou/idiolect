"""Test interactive chat command parsing."""

import pytest

from idiolect.tui.commands import CommandError, completions, parse_command


@pytest.mark.parametrize("value", ["/exit", "/registry", "/save", "/exit   "])
def test_known_command_is_parsed_without_arguments(value: str) -> None:
    """Check that each command returns its lowercase name."""
    command = parse_command(value)

    assert command is not None
    assert command.name == value.split()[0][1:]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("/quit", "Unknown chat command"),
        ("/retry", "Unknown chat command"),
        ("/exit now", "/exit does not accept an argument"),
        ("/REGISTRY", "Unknown chat command"),
    ],
)
def test_unknown_commands_and_arguments_are_rejected(
    value: str,
    message: str,
) -> None:
    """Check that the command surface stays fixed and argument-free."""
    with pytest.raises(CommandError, match=message):
        parse_command(value)


def test_command_completion_requires_one_command_prefix() -> None:
    """Check command matching for composer prefixes."""
    assert completions("/") == ("/exit", "/registry", "/save")
    assert completions("/re") == ("/registry",)
    assert completions("/sa") == ("/save",)
    assert completions("message") == ()
    assert completions("/exit now") == ()


def test_regular_message_is_not_a_command() -> None:
    """Check that regular transcript text passes command parsing."""
    assert parse_command("plain message") is None
