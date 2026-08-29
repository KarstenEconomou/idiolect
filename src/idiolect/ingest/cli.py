"""Register Signal command-line operations."""

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from idiolect.config import ConfigError, load_config, resolve_config_path
from idiolect.ingest.harvest import harvest, reindex
from idiolect.ingest.signal import (
    SignalError,
    SignalFileSource,
    SignalParser,
    SignalSource,
)
from idiolect.store.duck import DuckRepository, StoreError

ERRORS = (ConfigError, SignalError, StoreError)
_LABEL = "com.idiolect.collect"


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register Signal commands."""
    parser = commands.add_parser("signal", help="collect Signal group messages")
    subcommands = parser.add_subparsers(dest="signal_command", required=True)
    for name, help_text in (
        ("groups", "list known Signal groups"),
        ("status", "show the collector service state"),
        ("start", "load the collector service"),
        ("stop", "stop the collector service"),
        ("stats", "show stored record counts"),
        ("reindex", "refresh normalized records"),
    ):
        command = subcommands.add_parser(name, help=help_text)
        command.set_defaults(handler=_signal)
    collect = subcommands.add_parser("collect", help="collect queued Signal messages")
    wait = collect.add_mutually_exclusive_group()
    wait.add_argument("--timeout", type=int, help="receive timeout in seconds")
    wait.add_argument("--follow", action="store_true", help="wait until stopped")
    collect.add_argument("--max-messages", type=int, help="maximum event count")
    collect.set_defaults(handler=_signal)
    imported = subcommands.add_parser("import", help="import signal-cli JSON lines")
    imported.add_argument("path", type=Path)
    imported.set_defaults(handler=_signal)


def _signal(arguments: argparse.Namespace) -> int:
    if arguments.signal_command in {"status", "start", "stop"}:
        return _lifecycle(arguments.signal_command)
    config = load_config(resolve_config_path(arguments.config))
    if arguments.signal_command == "groups":
        for group in SignalSource(config.signal).groups():
            state = "active" if group.active else "inactive"
            print(f"{group.id}\t{state}\t{group.name}")
        return 0
    repository = DuckRepository(config.store.database_path)
    if arguments.signal_command == "stats":
        stats = repository.stats()
        print(
            f"events={stats.events} messages={stats.messages} "
            f"reactions={stats.reactions} database={repository.path}"
        )
        return 0
    if not config.signal.chats:
        raise ConfigError("Set IDIOLECT_SIGNAL_CHATS before Signal message processing")
    parser = SignalParser(config.signal.chats)
    if arguments.signal_command == "reindex":
        result = reindex(parser, repository)
        print(
            f"scanned={result.scanned} updated={result.updated} "
            f"messages={result.messages} reactions={result.reactions} skipped={result.skipped}"
        )
        return 0
    if arguments.signal_command == "import":
        source = SignalFileSource(arguments.path)
    else:
        timeout = -1 if arguments.follow else arguments.timeout
        max_messages = (
            arguments.max_messages
            if arguments.max_messages is not None
            else None if arguments.follow else config.signal.max_messages
        )
        signal = replace(
            config.signal,
            timeout=config.signal.timeout if timeout is None else timeout,
            max_messages=max_messages,
        )
        source = SignalSource(signal)
    result = harvest(source, parser, repository)
    print(
        f"received={result.received} stored={result.stored} messages={result.messages} "
        f"reactions={result.reactions} skipped={result.skipped} duplicates={result.duplicates}"
    )
    return 0


def _lifecycle(action: str) -> int:
    if sys.platform != "darwin":
        raise SignalError("Collector lifecycle commands require macOS")
    domain = f"gui/{os.getuid()}"
    if action == "start":
        command = (
            "launchctl",
            "bootstrap",
            domain,
            str(Path.home() / "Library/LaunchAgents/com.idiolect.collect.plist"),
        )
    elif action == "stop":
        command = ("launchctl", "bootout", f"{domain}/{_LABEL}")
    else:
        command = ("launchctl", "print", f"{domain}/{_LABEL}")
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise SignalError(f"Cannot {action} collector: {error}") from error
    return 0
