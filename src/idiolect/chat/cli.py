"""Register private chat command-line operations."""

import argparse

from idiolect.chat.discovery import (
    ChatDiscoveryError,
    DiscoveryItem,
    default_assistant,
    discover_assistants,
    load_assistant,
)
from idiolect.chat.runtime import ChatError, validate_chat_policy
from idiolect.chat.storage import ChatStorageError, ChatStore
from idiolect.command import artifact_path
from idiolect.config import ConfigError, load_config, resolve_config_path
from idiolect.tui import ChatTuiError, run_chat_app

ERRORS = (ConfigError, ChatDiscoveryError, ChatError, ChatStorageError, ChatTuiError)


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the chat command."""
    parser = commands.add_parser("chat", help="chat with verified local adapters")
    parser.add_argument("--run", help="run content ID or directory")
    parser.add_argument("--dataset", help="dataset content ID or directory")
    parser.add_argument("--resume", help="chat content ID or directory")
    parser.set_defaults(handler=_chat)


def _chat(arguments: argparse.Namespace) -> int:
    direct = arguments.run is not None or arguments.dataset is not None
    if direct and (arguments.run is None or arguments.dataset is None):
        raise ChatError("--run and --dataset must be used together")
    if arguments.resume is not None and direct:
        raise ChatError("--resume cannot be used with --run or --dataset")
    config = load_config(resolve_config_path(arguments.config))
    validate_chat_policy(config.chat, config.inference, config.train)
    if config.chat.output is None:
        raise ChatError("Set chat.output before chat")
    store = ChatStore(config.chat.output)
    default = default_assistant(config.train, config.chat)
    rows = (
        DiscoveryItem(default.name, "BASE", None, default),
        *discover_assistants(config.train.output, config.data.output),
    )
    initial_assistant = None
    initial_chat = None
    if direct:
        run = artifact_path(arguments.run, config.train.output)
        dataset = artifact_path(arguments.dataset, config.data.output)
        initial_assistant = load_assistant(run, dataset)
        selected = next(
            (row for row in rows if row.run_id == initial_assistant.run_id), None
        )
        if selected is not None and not selected.available:
            raise ChatDiscoveryError(selected.error or "Assistant is unavailable")
    elif arguments.resume is not None:
        initial_chat = store.load(
            artifact_path(arguments.resume, config.chat.output)
        )
    run_chat_app(
        config.chat,
        config.inference.generation,
        assistants=rows,
        store=store,
        initial_assistant=initial_assistant,
        initial_chat=initial_chat,
    )
    return 0
