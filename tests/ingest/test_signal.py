"""Test Signal input behavior."""

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from idiolect.config import SignalConfig
from idiolect.ingest.signal import SignalFileSource, SignalParser, SignalSource
from idiolect.types import ChatId, Message, Reaction

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class FakeRunner:
    """Return fixed Signal command output."""

    def __init__(self, output: bytes = b"", lines: tuple[bytes, ...] = ()) -> None:
        """Set the fixed output."""
        self.output = output
        self.output_lines = lines
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: Sequence[str]) -> bytes:
        """Record one command and return fixed output."""
        self.commands.append(tuple(command))
        return self.output

    def lines(self, command: Sequence[str]) -> Iterable[bytes]:
        """Record one command and return fixed lines."""
        self.commands.append(tuple(command))
        return self.output_lines


def test_source_uses_safe_receive_options(signal_events: Path) -> None:
    """Check that collection does not download private media."""
    first_line = signal_events.read_bytes().splitlines(keepends=True)[0]
    runner = FakeRunner(lines=(first_line,))
    source = SignalSource(
        SignalConfig(
            account="+10000000000",
            data_dir=Path("safe-data"),
            chats=(ChatId("group-allowed"),),
            timeout=9,
            max_messages=2,
        ),
        runner=runner,
        clock=lambda: _NOW,
    )

    events = tuple(source.events())

    assert len(events) == 1
    assert events[0].received_at == _NOW
    assert runner.commands == [
        (
            "signal-cli",
            "--output",
            "json",
            "--data-dir",
            "safe-data",
            "--account",
            "+10000000000",
            "receive",
            "--timeout",
            "9",
            "--ignore-attachments",
            "--ignore-stories",
            "--ignore-avatars",
            "--ignore-stickers",
            "--max-messages",
            "2",
        )
    ]


def test_source_lists_group_ids_for_allowlist() -> None:
    """Check the group data that a user needs for configuration."""
    runner = FakeRunner(
        output=(
            b'[{"id":"group-allowed","name":"Writers","isMember":true,'
            b'"isBlocked":false}]'
        )
    )
    source = SignalSource(SignalConfig(account="+10000000000"), runner=runner)

    groups = source.groups()

    assert [(group.id, group.name, group.active) for group in groups] == [
        ("group-allowed", "Writers", True)
    ]
    assert runner.commands == [
        ("signal-cli", "--output", "json", "--account", "+10000000000", "listGroups")
    ]


def test_parser_keeps_allowed_message_context(signal_events: Path) -> None:
    """Check text, media data, replies, edits, and reactions."""
    events = tuple(SignalFileSource(signal_events, clock=lambda: _NOW).events())
    parser = SignalParser((ChatId("group-allowed"),))

    first = tuple(parser.records(events[0]))
    direct = tuple(parser.records(events[1]))
    other_group = tuple(parser.records(events[2]))
    outgoing = tuple(parser.records(events[3]))
    reaction = tuple(parser.records(events[4]))
    edit = tuple(parser.records(events[5]))

    assert len(first) == 1 and isinstance(first[0], Message)
    assert first[0].text == "First message"
    assert first[0].attachments[0].name == "image.jpg"
    assert direct == ()
    assert other_group == ()
    assert len(outgoing) == 1 and isinstance(outgoing[0], Message)
    assert outgoing[0].reply_to == first[0].id
    assert len(reaction) == 1 and isinstance(reaction[0], Reaction)
    assert reaction[0].message_id == first[0].id
    assert len(edit) == 1 and isinstance(edit[0], Message)
    assert edit[0].id == first[0].id
    assert edit[0].text == "Edited message"
    assert edit[0].edited_at is not None


def test_parser_makes_delete_tombstone(signal_events: Path, signal_delete: Path) -> None:
    """Check that a remote delete removes message text."""
    parser = SignalParser((ChatId("group-allowed"),))
    original_event = next(iter(SignalFileSource(signal_events, clock=lambda: _NOW).events()))
    delete_event = next(iter(SignalFileSource(signal_delete, clock=lambda: _NOW).events()))
    original = next(iter(parser.records(original_event)))
    deleted = next(iter(parser.records(delete_event)))

    assert isinstance(original, Message)
    assert isinstance(deleted, Message)
    assert deleted.id == original.id
    assert deleted.text is None
    assert deleted.deleted_at is not None


def test_parser_links_native_mentions_to_people(
    signal_mentions: Path,
) -> None:
    """Check visible names, UTF-16 ranges, and quote snapshots."""
    events = tuple(SignalFileSource(signal_mentions, clock=lambda: _NOW).events())
    parser = SignalParser((ChatId("group-allowed"),))

    target = next(iter(parser.records(events[0])))
    tagged = next(iter(parser.records(events[1])))
    plain = next(iter(parser.records(events[2])))

    assert isinstance(target, Message)
    assert isinstance(tagged, Message)
    assert isinstance(plain, Message)
    assert tagged.text == "😀 ￼ are you coming?"
    assert len(tagged.mentions) == 1
    assert tagged.mentions[0].person_id == target.author_id
    assert (
        tagged.mentions[0].start_utf16,
        tagged.mentions[0].length_utf16,
    ) == (3, 1)
    assert tagged.reply_to == target.id
    assert tagged.quote is not None
    assert tagged.quote.author_id == target.author_id
    assert tagged.quote.text == "Maybe"
    assert plain.text == "Karsten are you coming?"
    assert plain.mentions == ()
