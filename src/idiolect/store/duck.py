"""Store Signal records in DuckDB."""

import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import duckdb

from idiolect.types import (
    Attachment,
    ChatId,
    Event,
    EventId,
    Mention,
    Message,
    MessageId,
    PersonId,
    Quote,
    Reaction,
    Record,
    StoreStats,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,
    source_id VARCHAR NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    payload BLOB NOT NULL,
    stored_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS messages (
    id VARCHAR PRIMARY KEY,
    event_id VARCHAR NOT NULL,
    chat_id VARCHAR NOT NULL,
    author_id VARCHAR NOT NULL,
    sent_at TIMESTAMPTZ NOT NULL,
    text VARCHAR,
    reply_to VARCHAR,
    edited_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,
    revision_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS messages_chat_time ON messages (chat_id, sent_at);
CREATE INDEX IF NOT EXISTS messages_author_time ON messages (author_id, sent_at);

ALTER TABLE messages ADD COLUMN IF NOT EXISTS author_name VARCHAR;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS is_self BOOLEAN DEFAULT false;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS quote_author_id VARCHAR;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS quote_sent_at TIMESTAMPTZ;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS quote_text VARCHAR;

CREATE TABLE IF NOT EXISTS mentions (
    message_id VARCHAR NOT NULL,
    scope VARCHAR NOT NULL CHECK (scope IN ('body', 'quote')),
    ordinal INTEGER NOT NULL,
    person_id VARCHAR NOT NULL,
    start INTEGER NOT NULL,
    length INTEGER NOT NULL,
    name VARCHAR,
    PRIMARY KEY (message_id, scope, ordinal)
);

CREATE TABLE IF NOT EXISTS attachments (
    message_id VARCHAR NOT NULL,
    id VARCHAR NOT NULL,
    media_type VARCHAR,
    name VARCHAR,
    size BIGINT,
    PRIMARY KEY (message_id, id)
);

CREATE TABLE IF NOT EXISTS reactions (
    event_id VARCHAR PRIMARY KEY,
    message_id VARCHAR NOT NULL,
    chat_id VARCHAR NOT NULL,
    author_id VARCHAR NOT NULL,
    value VARCHAR NOT NULL,
    sent_at TIMESTAMPTZ NOT NULL,
    removed BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS reactions_message_time ON reactions (message_id, sent_at);
"""


class StoreError(RuntimeError):
    """Report a local store error."""


class DuckRepository:
    """Store normalized records in one DuckDB file."""

    def __init__(self, path: Path) -> None:
        """Open the local store and create its tables."""
        self._path = path
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            with duckdb.connect(str(self._path)) as connection:
                connection.execute(_SCHEMA)
            self._restrict()
        except (duckdb.Error, OSError) as error:
            raise StoreError(f"Cannot open local store: {self._path}") from error

    def _restrict(self) -> None:
        """Keep the database and its write-ahead log private."""
        os.chmod(self._path, 0o600)
        wal = self._path.with_name(f"{self._path.name}.wal")
        if wal.exists():
            os.chmod(wal, 0o600)

    @property
    def path(self) -> Path:
        """Return the database path."""
        return self._path

    def save(self, event: Event, records: Iterable[Record]) -> bool:
        """Save one event and return true for a new event."""
        values = tuple(records)
        if any(record.event_id != event.id for record in values):
            raise StoreError("Each record must refer to its source event")
        try:
            with duckdb.connect(str(self._path)) as connection:
                connection.begin()
                if connection.execute(
                    "SELECT 1 FROM events WHERE id = ?", [str(event.id)]
                ).fetchone():
                    connection.rollback()
                    return False
                connection.execute(
                    """
                    INSERT INTO events (id, source, source_id, received_at, payload)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        str(event.id),
                        event.source,
                        event.source_id,
                        event.received_at,
                        event.payload,
                    ],
                )
                for record in values:
                    if isinstance(record, Message):
                        self._save_message(connection, record)
                    elif isinstance(record, Reaction):
                        self._save_reaction(connection, record)
                connection.commit()
            self._restrict()
        except duckdb.Error as error:
            raise StoreError(f"Cannot save event: {event.id}") from error
        return True

    def events(self) -> tuple[Event, ...]:
        """Return stored source events in storage order."""
        try:
            with duckdb.connect(str(self._path), read_only=True) as connection:
                rows = connection.execute(
                    """
                    SELECT id, source, source_id, received_at, payload
                    FROM events
                    ORDER BY stored_at, id
                    """
                ).fetchall()
        except duckdb.Error as error:
            raise StoreError(f"Cannot read source events from: {self._path}") from error
        return tuple(
            Event(
                id=EventId(cast(str, row[0])),
                source=cast(str, row[1]),
                source_id=cast(str, row[2]),
                received_at=cast(datetime, row[3]),
                payload=cast(bytes, row[4]),
            )
            for row in rows
        )

    def replace(self, event: Event, records: Iterable[Record]) -> None:
        """Replace normalized records from one stored event."""
        values = tuple(records)
        if any(record.event_id != event.id for record in values):
            raise StoreError("Each record must refer to its source event")
        try:
            with duckdb.connect(str(self._path)) as connection:
                connection.begin()
                exists = connection.execute(
                    "SELECT 1 FROM events WHERE id = ?", [str(event.id)]
                ).fetchone()
                if exists is None:
                    raise StoreError(f"Source event does not exist: {event.id}")
                connection.execute(
                    "DELETE FROM reactions WHERE event_id = ?", [str(event.id)]
                )
                for record in values:
                    if isinstance(record, Message):
                        self._save_message(connection, record)
                    elif isinstance(record, Reaction):
                        self._save_reaction(connection, record)
                connection.commit()
            self._restrict()
        except duckdb.Error as error:
            raise StoreError(f"Cannot replace event records: {event.id}") from error

    def messages(self, person_id: PersonId | None = None) -> tuple[Message, ...]:
        """Return messages in time order."""
        query = """
            SELECT id, event_id, chat_id, author_id, sent_at, author_name, is_self,
                   text, reply_to, edited_at, deleted_at, quote_author_id,
                   quote_sent_at, quote_text
            FROM messages
        """
        parameters: list[str] = []
        if person_id is not None:
            query += " WHERE author_id = ?"
            parameters.append(str(person_id))
        query += " ORDER BY sent_at, id"
        try:
            with duckdb.connect(str(self._path), read_only=True) as connection:
                rows = connection.execute(query, parameters).fetchall()
                attachments = _grouped_attachments(connection)
                reactions = _grouped_reactions(connection)
                mentions = _grouped_mentions(connection)
        except duckdb.Error as error:
            raise StoreError(f"Cannot read messages from: {self._path}") from error
        return tuple(
            self._message(
                row,
                attachments.get(cast(str, row[0])) or [],
                reactions.get(cast(str, row[0])) or [],
                mentions.get((cast(str, row[0]), "body")) or (),
                mentions.get((cast(str, row[0]), "quote")) or (),
            )
            for row in rows
        )

    def stats(self) -> StoreStats:
        """Return record counts from the store."""
        try:
            with duckdb.connect(str(self._path), read_only=True) as connection:
                events = connection.execute("SELECT count(*) FROM events").fetchone()
                messages = connection.execute(
                    "SELECT count(*) FROM messages"
                ).fetchone()
                reactions = connection.execute(
                    "SELECT count(*) FROM reactions"
                ).fetchone()
        except duckdb.Error as error:
            raise StoreError(f"Cannot read counts from: {self._path}") from error
        assert events is not None
        assert messages is not None
        assert reactions is not None
        return StoreStats(
            events=cast(int, events[0]),
            messages=cast(int, messages[0]),
            reactions=cast(int, reactions[0]),
        )

    def _save_message(
        self, connection: duckdb.DuckDBPyConnection, message: Message
    ) -> None:
        revision = message.deleted_at or message.edited_at or message.sent_at
        current = connection.execute(
            "SELECT revision_at FROM messages WHERE id = ?", [str(message.id)]
        ).fetchone()
        if current is not None and cast(datetime, current[0]) > revision:
            return
        connection.execute(
            "DELETE FROM attachments WHERE message_id = ?", [str(message.id)]
        )
        connection.execute(
            "DELETE FROM mentions WHERE message_id = ?", [str(message.id)]
        )
        connection.execute("DELETE FROM messages WHERE id = ?", [str(message.id)])
        connection.execute(
            """
            INSERT INTO messages (
                id, event_id, chat_id, author_id, sent_at, author_name, is_self,
                text, reply_to, edited_at, deleted_at, revision_at, quote_author_id,
                quote_sent_at, quote_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(message.id),
                str(message.event_id),
                str(message.chat_id),
                str(message.author_id),
                message.sent_at,
                message.author_name,
                message.is_self,
                message.text,
                str(message.reply_to) if message.reply_to is not None else None,
                message.edited_at,
                message.deleted_at,
                revision,
                str(message.quote.author_id) if message.quote is not None else None,
                message.quote.sent_at if message.quote is not None else None,
                message.quote.text if message.quote is not None else None,
            ],
        )
        for attachment in message.attachments:
            connection.execute(
                """
                INSERT INTO attachments (message_id, id, media_type, name, size)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    str(message.id),
                    attachment.id,
                    attachment.media_type,
                    attachment.name,
                    attachment.size,
                ],
            )
        self._save_mentions(connection, message.id, "body", message.mentions)
        if message.quote is not None:
            self._save_mentions(connection, message.id, "quote", message.quote.mentions)

    def _save_mentions(
        self,
        connection: duckdb.DuckDBPyConnection,
        message_id: MessageId,
        scope: str,
        mentions: tuple[Mention, ...],
    ) -> None:
        for ordinal, mention in enumerate(mentions):
            connection.execute(
                """
                INSERT INTO mentions (
                    message_id, scope, ordinal, person_id, start, length, name
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(message_id),
                    scope,
                    ordinal,
                    str(mention.person_id),
                    mention.start_utf16,
                    mention.length_utf16,
                    mention.name,
                ],
            )

    def _save_reaction(
        self, connection: duckdb.DuckDBPyConnection, reaction: Reaction
    ) -> None:
        connection.execute(
            """
            INSERT INTO reactions (
                event_id, message_id, chat_id, author_id, value, sent_at, removed
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(reaction.event_id),
                str(reaction.message_id),
                str(reaction.chat_id),
                str(reaction.author_id),
                reaction.value,
                reaction.sent_at,
                reaction.removed,
            ],
        )

    def _message(
        self,
        row: tuple[Any, ...],
        attachment_rows: list[tuple[Any, ...]],
        reaction_rows: list[tuple[Any, ...]],
        body_mentions: tuple[Mention, ...],
        quote_mentions: tuple[Mention, ...],
    ) -> Message:
        message_id = MessageId(cast(str, row[0]))
        quote = None
        if row[11] is not None and row[12] is not None:
            quote = Quote(
                author_id=PersonId(cast(str, row[11])),
                sent_at=cast(datetime, row[12]),
                text=cast(str | None, row[13]),
                mentions=quote_mentions,
            )
        return Message(
            id=message_id,
            event_id=EventId(cast(str, row[1])),
            chat_id=ChatId(cast(str, row[2])),
            author_id=PersonId(cast(str, row[3])),
            sent_at=cast(datetime, row[4]),
            author_name=cast(str | None, row[5]),
            is_self=cast(bool, row[6]),
            text=cast(str | None, row[7]),
            reply_to=MessageId(cast(str, row[8])) if row[8] is not None else None,
            edited_at=cast(datetime | None, row[9]),
            deleted_at=cast(datetime | None, row[10]),
            reactions=tuple(
                Reaction(
                    event_id=EventId(cast(str, item[0])),
                    message_id=message_id,
                    chat_id=ChatId(cast(str, item[1])),
                    author_id=PersonId(cast(str, item[2])),
                    value=cast(str, item[3]),
                    sent_at=cast(datetime, item[4]),
                    removed=cast(bool, item[5]),
                )
                for item in reaction_rows
            ),
            attachments=tuple(
                Attachment(
                    id=cast(str, item[0]),
                    media_type=cast(str | None, item[1]),
                    name=cast(str | None, item[2]),
                    size=cast(int | None, item[3]),
                )
                for item in attachment_rows
            ),
            mentions=body_mentions,
            quote=quote,
        )


def _grouped_attachments(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, list[tuple[Any, ...]]]:
    """Return one attachment row list per message ID."""
    rows = connection.execute(
        """
        SELECT message_id, id, media_type, name, size
        FROM attachments ORDER BY message_id, id
        """
    ).fetchall()
    grouped: dict[str, list[tuple[Any, ...]]] = {}
    for row in rows:
        grouped.setdefault(cast(str, row[0]), []).append(row[1:])
    return grouped


def _grouped_reactions(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, list[tuple[Any, ...]]]:
    """Return one reaction row list per message ID."""
    rows = connection.execute(
        """
        SELECT event_id, message_id, chat_id, author_id, value, sent_at, removed
        FROM reactions ORDER BY message_id, sent_at, event_id
        """
    ).fetchall()
    grouped: dict[str, list[tuple[Any, ...]]] = {}
    for row in rows:
        grouped.setdefault(cast(str, row[1]), []).append((*row[:1], *row[2:]))
    return grouped


def _grouped_mentions(
    connection: duckdb.DuckDBPyConnection,
) -> dict[tuple[str, str], tuple[Mention, ...]]:
    """Return one mention tuple per message ID and scope."""
    rows = connection.execute(
        """
        SELECT message_id, scope, ordinal, person_id, start, length, name
        FROM mentions ORDER BY message_id, scope, ordinal
        """
    ).fetchall()
    grouped: dict[tuple[str, str], list[Mention]] = {}
    for row in rows:
        grouped.setdefault((cast(str, row[0]), cast(str, row[1])), []).append(
            Mention(
                person_id=PersonId(cast(str, row[3])),
                start_utf16=cast(int, row[4]),
                length_utf16=cast(int, row[5]),
                name=cast(str | None, row[6]),
            )
        )
    return {key: tuple(value) for key, value in grouped.items()}
