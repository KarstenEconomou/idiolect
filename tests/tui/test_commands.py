"""Test interactive chat command parsing."""

import pytest

from idiolect.tui.commands import CommandError, completions, parse_command


def test_no_argument_command_is_parsed() -> None:
    """Check a known command without arguments returns its name."""
    command = parse_command("/help")

    assert command is not None
    assert command.name == "help"
    assert command.argument is None


def test_save_command_keeps_one_quoted_title() -> None:
    """Check that a save title can contain spaces."""
    command = parse_command('/save "Night city"')

    assert command is not None
    assert command.argument == "Night city"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("/remove", "Unknown chat command"),
        ('/save "Night city', "Command quotes are not complete"),
        ("/retry now", "/retry does not accept an argument"),
    ],
)
def test_invalid_commands_are_rejected(value: str, message: str) -> None:
    """Check invalid command names, quotes, and arguments."""
    with pytest.raises(CommandError, match=message):
        parse_command(value)


def test_command_completion_requires_one_command_prefix() -> None:
    """Check command completion for composer prefixes."""
    assert completions("/re") == ("/resume", "/retry")
    assert completions("message") == ()
    assert completions("/save title") == ()


def test_regular_message_is_not_a_command() -> None:
    """Check that regular transcript text passes command parsing."""
    assert parse_command("plain message") is None
