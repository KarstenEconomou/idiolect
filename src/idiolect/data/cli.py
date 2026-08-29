"""Register dataset command-line operations."""

import argparse

from idiolect.config import ConfigError, load_config, resolve_config_path
from idiolect.data.local import DataError, LocalBuilder, resolve_self, summarize_people
from idiolect.store.duck import DuckRepository, StoreError
from idiolect.types import PersonId, Split

ERRORS = (ConfigError, DataError, StoreError)


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register dataset commands."""
    parser = commands.add_parser("data", help="inspect data and build datasets")
    subcommands = parser.add_subparsers(dest="data_command", required=True)
    people = subcommands.add_parser("people", help="list normalized message authors")
    people.set_defaults(handler=_people)
    build = subcommands.add_parser("build", help="build one target dataset")
    build.add_argument("name", help="target name in model text")
    build.add_argument("--person", help="normalized target person ID")
    build.set_defaults(handler=_build)


def _people(arguments: argparse.Namespace) -> int:
    config = load_config(resolve_config_path(arguments.config))
    repository = DuckRepository(config.store.database_path)
    for person in summarize_people(repository.messages()):
        state = "self" if person.is_self else "member"
        print(f"{person.id}\t{state}\t{person.messages}\t{person.name or '(unknown)'}")
    return 0


def _build(arguments: argparse.Namespace) -> int:
    config = load_config(resolve_config_path(arguments.config))
    repository = DuckRepository(config.store.database_path)
    people = summarize_people(repository.messages())
    person_id = resolve_self(people) if arguments.person is None else PersonId(arguments.person)
    if config.data.output is None:
        raise ConfigError("Set data.output before dataset construction")
    result = LocalBuilder(repository, config.data.output).build(
        person_id, arguments.name, config.data
    )
    counts = result.counts
    print(
        f"dataset={result.dataset.id} train={counts.get(Split.TRAIN, 0)} "
        f"valid={counts.get(Split.VALID, 0)} test={counts.get(Split.TEST, 0)} "
        f"path={result.dataset.path}"
    )
    return 0
