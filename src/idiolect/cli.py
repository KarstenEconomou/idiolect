"""Coordinate the Idiolect command-line interface."""

import argparse
import os
import sys
from collections.abc import Sequence

from idiolect import config_cli
from idiolect.chat import cli as chat_cli
from idiolect.data import cli as data_cli
from idiolect.eval import cli as eval_cli
from idiolect.inference import cli as inference_cli
from idiolect.ingest import cli as ingest_cli
from idiolect.train import cli as train_cli


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _parser()
    try:
        raw_arguments = tuple(sys.argv[1:] if argv is None else argv)
        arguments = parser.parse_args(raw_arguments)
        arguments.raw_arguments = raw_arguments
        return arguments.handler(arguments)
    except KeyboardInterrupt:
        print("Operation stopped.", file=sys.stderr)
        return 130
    except Exception as error:
        handled = (
            config_cli.ERRORS
            + chat_cli.ERRORS
            + data_cli.ERRORS
            + eval_cli.ERRORS
            + inference_cli.ERRORS
            + ingest_cli.ERRORS
            + train_cli.ERRORS
        )
        if not isinstance(error, handled):
            raise
        print(f"error: {error}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    """Create the root parser and register package commands."""
    parser = argparse.ArgumentParser(prog="idiolect")
    parser.add_argument(
        "-c",
        "--config",
        default=os.environ.get("IDIOLECT_CONFIG"),
        help="configuration name or TOML path",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_cli.register(commands)
    data_cli.register(commands)
    train_cli.register(commands)
    inference_cli.register(commands)
    eval_cli.register(commands)
    chat_cli.register(commands)
    config_cli.register(commands)
    return parser
