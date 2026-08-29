"""Register configuration management commands."""

import argparse

from idiolect.config import ConfigError, create_configuration, list_configurations

ERRORS = (ConfigError,)


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register configuration commands."""
    parser = commands.add_parser("config", help="manage experiment configurations")
    subcommands = parser.add_subparsers(dest="config_command", required=True)
    listing = subcommands.add_parser("list", help="list available configurations")
    listing.set_defaults(handler=_list)
    create = subcommands.add_parser("new", help="create an experiment configuration")
    create.add_argument("name")
    create.add_argument("--from", dest="source", help="source name or TOML path")
    create.set_defaults(handler=_new)


def _list(arguments: argparse.Namespace) -> int:
    for name in list_configurations():
        print(name)
    return 0


def _new(arguments: argparse.Namespace) -> int:
    path = create_configuration(arguments.name, arguments.source)
    print(f"created={path}")
    return 0
