"""Parse the fixed interactive chat command set."""

from dataclasses import dataclass

COMMANDS = (
    "/terminate",
    "/echo",
    "/disconnect",
    "/save",
    "/specs",
    "/chroma",
)

COMMAND_DESCRIPTIONS = {
    "/terminate": "TERMINATE IDIOLECT.",
    "/echo": "ENV echo.",
    "/disconnect": "DISCONNECT from CONSTRUCT.",
    "/save": "Save TRACE.",
    "/specs": "View SPECS.",
    "/chroma": "Select CHROMA.",
}

COMMAND_ARGUMENTS = frozenset({"/echo"})


class CommandError(ValueError):
    """Report an invalid slash command."""


@dataclass(frozen=True, slots=True)
class Command:
    """Keep one parsed slash command."""

    name: str
    arguments: str = ""

    @property
    def accepts_arguments(self) -> bool:
        """Return whether this command consumes composer text."""
        return f"/{self.name}" in COMMAND_ARGUMENTS


def parse_command(value: str) -> Command | None:
    """Parse a composer value when it starts with a slash."""
    if not value.startswith("/"):
        return None
    parts = value.split(maxsplit=1)
    if not parts or parts[0] not in COMMANDS:
        raise CommandError("Unknown chat command")
    command_name = parts[0]
    name = command_name[1:]
    arguments = parts[1] if len(parts) == 2 else ""
    if command_name not in COMMAND_ARGUMENTS and arguments.strip():
        raise CommandError(f"/{name} does not accept an argument")
    return Command(name, arguments)


def completions(value: str) -> tuple[str, ...]:
    """Return matching command names for one composer prefix."""
    if not value.startswith("/") or any(character.isspace() for character in value):
        return ()
    return tuple(command for command in COMMANDS if command.startswith(value))
