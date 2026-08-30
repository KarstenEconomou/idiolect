"""Parse the fixed interactive chat command set."""

from dataclasses import dataclass

COMMAND_DESCRIPTIONS = {
    "/buffer": "View context BUFFER.",
    "/chroma": "Equip CHROMA.",
    "/disconnect": "Disconnect active LINK.",
    "/echo": "SYS echo.",
    "/probe": "View active LINK.",
    "/retry": "Retry cancelled or failed GENERATION.",
    "/specs": "View CONSTRUCT specifications.",
    "/terminate": "Terminate IDIOLECT.",
    "/trace": "Save current TRACE.",
}
COMMANDS = tuple(sorted(COMMAND_DESCRIPTIONS))

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
        raise CommandError("COMMAND unknown")
    command_name = parts[0]
    name = command_name[1:]
    arguments = parts[1] if len(parts) == 2 else ""
    if command_name not in COMMAND_ARGUMENTS and arguments.strip():
        raise CommandError("COMMAND argument unexpected")
    return Command(name, arguments)


def completions(value: str) -> tuple[str, ...]:
    """Return matching command names for one composer prefix."""
    if not value.startswith("/") or any(character.isspace() for character in value):
        return ()
    return tuple(command for command in COMMANDS if command.startswith(value))
