"""Parse the fixed interactive chat command set."""

from dataclasses import dataclass

COMMANDS = (
    "/exit",
    "/registry",
)

COMMAND_DESCRIPTIONS = {
    "/exit": "Exit IDIOLECT.",
    "/registry": "Return to REGISTRY.",
}


class CommandError(ValueError):
    """Report an invalid slash command."""


@dataclass(frozen=True, slots=True)
class Command:
    """Keep one parsed slash command."""

    name: str


def parse_command(value: str) -> Command | None:
    """Parse a composer value when it starts with a slash."""
    if not value.startswith("/"):
        return None
    parts = value.split()
    if not parts or parts[0] not in COMMANDS:
        raise CommandError("Unknown chat command")
    name = parts[0][1:]
    if len(parts) != 1:
        raise CommandError(f"/{name} does not accept an argument")
    return Command(name)


def completions(value: str) -> tuple[str, ...]:
    """Return matching command names for one composer prefix."""
    if not value.startswith("/") or any(character.isspace() for character in value):
        return ()
    return tuple(command for command in COMMANDS if command.startswith(value))
