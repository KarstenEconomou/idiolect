# Signal Collection

## Requirements

Install these components:

- Python 3.14
- `uv`
- `just` 1.46.0 or later
- A current `signal-cli` release
- A local QR code tool, such as `qrencode`

Run all commands from the repository root.

```console
just setup
mkdir -m 700 -p var/signal
```

Keep `signal-cli` current. Signal service changes can stop an old release.

## Link the device

Link `signal-cli` as a secondary device. Do not register the phone number as a new primary device.

```console
signal-cli --data-dir var/signal link -n Idiolect
```

The command shows a `sgnl://` device link URI. Convert this URI to a QR code on the local computer. Do not send the URI to a website. In the Signal mobile app, select **Settings → Linked devices → Link new device**. Scan the QR code.

The `var/signal` directory contains the linked-device keys and account state. Do not delete it while the device is active.

## Configure Idiolect

The repository contains `conf/idiolect.toml`. This file contains public settings. Create the private environment file:

```console
touch .env
chmod 600 .env
```

Add the Signal account to `.env`. Use the international number format.

```sh
IDIOLECT_SIGNAL_ACCOUNT="+14165550123"
```

Just recipes that launch Idiolect pass `.env` to `uv`. You do not need to load
it in the current shell. The local `launchd` agent also loads it before it
starts Idiolect. A direct `uv run idiolect` command does not load it unless you
include `--env-file .env`.

These environment values are available:

| Value | Function |
|---|---|
| `IDIOLECT_CONFIG` | Select an optional alternate TOML configuration path. |
| `IDIOLECT_SIGNAL_ACCOUNT` | Set the Signal account identifier. |
| `IDIOLECT_SIGNAL_CHATS` | Replace the Signal chat whitelist with a JSON list. |
| `IDIOLECT_SIGNAL_BIN` | Set the absolute `signal-cli` path. |
| `IDIOLECT_SIGNAL_DATA_DIR` | Set the private Signal data directory. |

The default configuration path is `conf/idiolect.toml`. You do not need to set `IDIOLECT_CONFIG` for normal operation. Set it only when you must select a different configuration file.

Signal environment values take priority over the same TOML values. The loader rejects an unknown TOML key. Signal chat IDs are valid only in `IDIOLECT_SIGNAL_CHATS`. The loader rejects invalid or duplicate chat IDs. `timeout` must be `-1` or greater. `max_messages` must be greater than zero.

List the Signal groups:

```console
just idiolect signal groups
```

The output format is:

```text
GROUP_ID=    active    Group name
```

Add the required IDs to `.env` as one JSON list. Use single shell quotes around the JSON text.

```sh
IDIOLECT_SIGNAL_CHATS='["GROUP_ID_ONE=", "GROUP_ID_TWO="]'
```

The next Just command reads the updated `.env`. The chat list is private
metadata. Do not put a real group ID in `conf/idiolect.toml`.
Keep the outer single quotes. Use double quotes around each ID. Do not use a trailing comma.

## Run collection

Run one bounded receive operation:

```console
just idiolect signal collect
```

Run continuous collection:

```console
just idiolect signal collect --follow
```

Set explicit bounds when necessary:

```console
just idiolect signal collect --timeout 30 --max-messages 100
```

Show the stored counts:

```console
just idiolect signal stats
```

Refresh normalized records from the raw events in DuckDB:

```console
just idiolect signal reindex
```

Stop continuous collection before you run `reindex`. Start collection again after the command finishes. This rule prevents two processes from writing the same DuckDB file.

Run `reindex` after an update changes Signal normalization. The command does not contact Signal. It keeps the source events and refreshes their normalized messages and reactions.

Import saved `signal-cli` JSON lines:

```console
just idiolect signal import path/to/events.jsonl
```

## Collector behavior

- The source runs `signal-cli --output json receive`.
- The source skips one malformed output line and keeps the events after it. The harvest result counts such an event as `skipped`.
- The source does not download attachments, stories, avatars, or stickers.
- The parser accepts only group IDs in the configured chat whitelist.
- The parser discards direct messages and messages from other groups.
- The parser reads incoming messages and sent-message sync events.
- The parser records text, identity-linked mentions, reply snapshots, edits, remote deletes, reactions, and attachment metadata.
- The parser keeps original message text and native mention metadata.
- One event that fails normalization is skipped and counted. It does not stop the harvest or discard later drained events.
- Mention ranges stay in the UTF-16 units that Signal supplies.
- The parser does not store attachment bytes.
- The store writes one source event and its normalized records in one transaction.
- The event ID is a SHA-256 hash of the source JSON. A second copy of the same event does not create another record.
- The normalized chat, person, message, and attachment IDs use SHA-256 values. Raw events still contain the original Signal data.
- A newer edit or delete replaces an older message revision. An older revision cannot replace a newer revision.

## Limits

The collector does not import the history that is already on the phone. It stores events that `signal-cli receive` emits after setup.

Signal can queue events while collection is off. Do not depend on an unlimited queue period. Long downtime can cause gaps.

An event from a group that is not on the whitelist is consumed and discarded. If you add that group later, Idiolect cannot restore the discarded event.

Only one collector must use a `signal-cli` data directory at one time.
