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
    Message,
    MessageId,
    PersonId,
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
            os.chmod(self._path, 0o600)
        except (duckdb.Error, OSError) as error:
            raise StoreError(f"Cannot open local store: {self._path}") from error

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
                if connection.execute("SELECT 1 FROM events WHERE id = ?", [str(event.id)]).fetchone():
                    connection.rollback()
                    return False
                connection.execute(
                    """
                    INSERT INTO events (id, source, source_id, received_at, payload)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [str(event.id), event.source, event.source_id, event.received_at, event.payload],
                )
                for record in values:
                    if isinstance(record, Message):
                        self._save_message(connection, record)
                    elif isinstance(record, Reaction):
                        self._save_reaction(connection, record)
                connection.commit()
        except duckdb.Error as error:
            raise StoreError(f"Cannot save event: {event.id}") from error
        return True

    def messages(self, person_id: PersonId | None = None) -> tuple[Message, ...]:
        """Return messages in time order."""
        query = """
            SELECT id, event_id, chat_id, author_id, sent_at, text, reply_to, edited_at, deleted_at
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
                return tuple(self._message(connection, row) for row in rows)
        except duckdb.Error as error:
            raise StoreError(f"Cannot read messages from: {self._path}") from error

    def stats(self) -> StoreStats:
        """Return record counts from the store."""
        try:
            with duckdb.connect(str(self._path), read_only=True) as connection:
                events = connection.execute("SELECT count(*) FROM events").fetchone()
                messages = connection.execute("SELECT count(*) FROM messages").fetchone()
                reactions = connection.execute("SELECT count(*) FROM reactions").fetchone()
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

    def _save_message(self, connection: duckdb.DuckDBPyConnection, message: Message) -> None:
        revision = message.deleted_at or message.edited_at or message.sent_at
        current = connection.execute(
            "SELECT revision_at FROM messages WHERE id = ?", [str(message.id)]
        ).fetchone()
        if current is not None and cast(datetime, current[0]) > revision:
            return
        connection.execute("DELETE FROM attachments WHERE message_id = ?", [str(message.id)])
        connection.execute("DELETE FROM messages WHERE id = ?", [str(message.id)])
        connection.execute(
            """
            INSERT INTO messages (
                id, event_id, chat_id, author_id, sent_at, text, reply_to,
                edited_at, deleted_at, revision_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(message.id),
                str(message.event_id),
                str(message.chat_id),
                str(message.author_id),
                message.sent_at,
                message.text,
                str(message.reply_to) if message.reply_to is not None else None,
                message.edited_at,
                message.deleted_at,
                revision,
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

    def _save_reaction(self, connection: duckdb.DuckDBPyConnection, reaction: Reaction) -> None:
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

    def _message(self, connection: duckdb.DuckDBPyConnection, row: tuple[Any, ...]) -> Message:
        message_id = MessageId(cast(str, row[0]))
        attachment_rows = connection.execute(
            """
            SELECT id, media_type, name, size
            FROM attachments WHERE message_id = ? ORDER BY id
            """,
            [str(message_id)],
        ).fetchall()
        reaction_rows = connection.execute(
            """
            SELECT event_id, chat_id, author_id, value, sent_at, removed
            FROM reactions WHERE message_id = ? ORDER BY sent_at, event_id
            """,
            [str(message_id)],
        ).fetchall()
        return Message(
            id=message_id,
            event_id=EventId(cast(str, row[1])),
            chat_id=ChatId(cast(str, row[2])),
            author_id=PersonId(cast(str, row[3])),
            sent_at=cast(datetime, row[4]),
            text=cast(str | None, row[5]),
            reply_to=MessageId(cast(str, row[6])) if row[6] is not None else None,
            edited_at=cast(datetime | None, row[7]),
            deleted_at=cast(datetime | None, row[8]),
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
        )
