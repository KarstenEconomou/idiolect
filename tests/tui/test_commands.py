"""Test interactive chat command parsing."""

import pytest

from idiolect.tui.commands import CommandError, completions, parse_command


@pytest.mark.parametrize(
    "value",
    [
        "/terminate",
        "/disconnect",
        "/trace",
        "/specs",
        "/probe",
        "/buffer",
        "/chroma",
        "/terminate   ",
    ],
)
def test_known_command_is_parsed_without_arguments(value: str) -> None:
    """Check that each command returns its lowercase name."""
    command = parse_command(value)

    assert command is not None
    assert command.name == value.split()[0][1:]


def test_echo_command_keeps_its_argument_text() -> None:
    """Check the one command that consumes composer arguments."""
    command = parse_command("/echo hello world")

    assert command is not None
    assert command.name == "echo"
    assert command.arguments == "hello world"
    assert command.accepts_arguments


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("/quit", "COMMAND unknown"),
        ("/retry", "COMMAND unknown"),
        ("/terminate now", "COMMAND unexpected argument"),
        ("/registry", "COMMAND unknown"),
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
    assert completions("/") == (
        "/terminate",
        "/echo",
        "/disconnect",
        "/trace",
        "/specs",
        "/probe",
        "/buffer",
        "/chroma",
    )
    assert completions("/di") == ("/disconnect",)
    assert completions("/tr") == ("/trace",)
    assert completions("/sp") == ("/specs",)
    assert completions("/pr") == ("/probe",)
    assert completions("/bu") == ("/buffer",)
    assert completions("/ch") == ("/chroma",)
    assert completions("/ec") == ("/echo",)
    assert completions("message") == ()
    assert completions("/terminate now") == ()


def test_regular_message_is_not_a_command() -> None:
    """Check that regular transcript text passes command parsing."""
    assert parse_command("plain message") is None
