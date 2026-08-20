"""Run the Idiolect command-line interface."""

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from idiolect.config import ConfigError, load_config
from idiolect.data.local import (
    DataError,
    LocalBuilder,
    load_dataset,
    resolve_self,
    summarize_people,
)
from idiolect.ingest import harvest
from idiolect.ingest.harvest import reindex
from idiolect.ingest.signal import (
    SignalError,
    SignalFileSource,
    SignalParser,
    SignalSource,
)
from idiolect.store.duck import DuckRepository, StoreError
from idiolect.train.mlx import MlxTrainer, TrainError
from idiolect.types import PersonId, Split


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
        if arguments.command == "data":
            repository = DuckRepository(config.store.database_path)
            people = summarize_people(repository.messages())
            if arguments.data_command == "people":
                for person in people:
                    state = "self" if person.is_self else "member"
                    name = person.name or "(unknown)"
                    print(f"{person.id}\t{state}\t{person.messages}\t{name}")
                return 0
            person_id = (
                resolve_self(people)
                if arguments.self_person
                else PersonId(arguments.person)
            )
            if config.data.output is None:
                raise ConfigError("Set data.output before dataset construction")
            result = LocalBuilder(repository, config.data.output).build(
                person_id,
                arguments.name,
                config.data,
            )
            counts = result.counts
            print(
                f"dataset={result.dataset.id} train={counts.get(Split.TRAIN, 0)} "
                f"valid={counts.get(Split.VALID, 0)} test={counts.get(Split.TEST, 0)} "
                f"path={result.dataset.path}"
            )
            return 0
        if arguments.command == "train":
            dataset = load_dataset(arguments.dataset).dataset
            result = MlxTrainer().train(dataset, config.train)
            for run in result.runs:
                print(f"run={run.id} dataset={run.dataset_id} path={run.path}")
            return 0
        if arguments.signal_command == "groups":
            source = SignalSource(config.signal)
            for group in source.groups():
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

        parser_adapter = SignalParser(config.signal.chats)
        if arguments.signal_command == "reindex":
            result = reindex(parser_adapter, repository)
            print(
                f"scanned={result.scanned} updated={result.updated} "
                f"messages={result.messages} reactions={result.reactions} "
                f"skipped={result.skipped}"
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
        result = harvest(source, parser_adapter, repository)
        print(
            f"received={result.received} stored={result.stored} "
            f"messages={result.messages} reactions={result.reactions} "
            f"skipped={result.skipped} duplicates={result.duplicates}"
        )
        return 0
    except (ConfigError, DataError, SignalError, StoreError, TrainError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Operation stopped.", file=sys.stderr)
        return 130


def _parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(prog="idiolect")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("IDIOLECT_CONFIG", "conf/local.toml")),
        help="TOML configuration path",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    signal = commands.add_parser("signal", help="collect Signal group messages")
    signal_commands = signal.add_subparsers(dest="signal_command", required=True)
    signal_commands.add_parser("groups", help="list known Signal groups")
    collect = signal_commands.add_parser("collect", help="collect queued Signal messages")
    wait = collect.add_mutually_exclusive_group()
    wait.add_argument("--timeout", type=int, help="receive timeout in seconds")
    collect.add_argument("--max-messages", type=int, help="maximum event count")
    wait.add_argument(
        "--follow",
        action="store_true",
        help="wait until collection is stopped",
    )
    import_command = signal_commands.add_parser(
        "import",
        help="import signal-cli JSON lines",
    )
    import_command.add_argument("path", type=Path, help="JSON lines path")
    signal_commands.add_parser("stats", help="show stored record counts")
    signal_commands.add_parser(
        "reindex",
        help="refresh normalized records from stored events",
    )
    data = commands.add_parser("data", help="build model datasets")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_commands.add_parser("people", help="list normalized message authors")
    build = data_commands.add_parser("build", help="build one target dataset")
    target = build.add_mutually_exclusive_group(required=True)
    target.add_argument("--self", dest="self_person", action="store_true")
    target.add_argument("--person", help="normalized target person ID")
    build.add_argument("--name", required=True, help="target name in model text")
    train = commands.add_parser("train", help="train configured local adapters")
    train.add_argument("dataset", type=Path, help="immutable dataset directory")
    return parser
