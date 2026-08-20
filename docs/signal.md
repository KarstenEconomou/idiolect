# Signal Collection

## Requirements

Install these components:

- Python 3.14
- `uv`
- A current `signal-cli` release
- A local QR code tool, such as `qrencode`

Run all commands from the repository root.

```console
uv sync
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

Create the private files:

```console
cp conf/local.toml.example conf/local.toml
touch .env
chmod 600 .env
```

Add the Signal account to `.env`. Use the international number format.

```sh
IDIOLECT_SIGNAL_ACCOUNT="+14165550123"
```

Load the file for an interactive command:

```console
set -a
source .env
set +a
```

Idiolect does not load `.env` by itself. The local `launchd` agent loads it before it starts Idiolect.

These environment values are available:

| Value | Function |
|---|---|
| `IDIOLECT_CONFIG` | Set another TOML configuration path. |
| `IDIOLECT_SIGNAL_ACCOUNT` | Set the Signal account identifier. |
| `IDIOLECT_SIGNAL_BIN` | Set the absolute `signal-cli` path. |
| `IDIOLECT_SIGNAL_DATA_DIR` | Set the private Signal data directory. |

Signal environment values take priority over the same TOML values. The loader rejects an unknown TOML key. `timeout` must be `-1` or greater. `max_messages` must be greater than zero.

List the Signal groups:

```console
uv run idiolect signal groups
```

The output format is:

```text
GROUP_ID=    active    Group name
```

Copy each required ID to `conf/local.toml`. Quote each ID.

```toml
[signal]
binary = "signal-cli"
data_dir = "var/signal"
chats = [
    "GROUP_ID_ONE=",
    "GROUP_ID_TWO=",
]
timeout = 5
```

## Run collection

Run one bounded receive operation:

```console
uv run idiolect signal collect
```

Run continuous collection:

```console
uv run idiolect signal collect --follow
```

Set explicit bounds when necessary:

```console
uv run idiolect signal collect --timeout 30 --max-messages 100
```

Show the stored counts:

```console
uv run idiolect signal stats
```

Import saved `signal-cli` JSON lines:

```console
uv run idiolect signal import path/to/events.jsonl
```

## Collector behavior

- The source runs `signal-cli --output json receive`.
- The source does not download attachments, stories, avatars, or stickers.
- The parser accepts only group IDs in `signal.chats`.
- The parser discards direct messages and messages from other groups.
- The parser reads incoming messages and sent-message sync events.
- The parser records text, replies, edits, remote deletes, reactions, and attachment metadata.
- The parser does not store attachment bytes.
- The store writes one source event and its normalized records in one transaction.
- The event ID is a SHA-256 hash of the source JSON. A second copy of the same event does not create another record.
- The normalized chat, person, message, and attachment IDs use SHA-256 values. Raw events still contain the original Signal data.
- A newer edit or delete replaces an older message revision. An older revision cannot replace a newer revision.

## Limits

The collector does not import the history that is already on the phone. It stores events that `signal-cli receive` emits after setup.

Signal can queue events while collection is off. Do not depend on an unlimited queue period. Long downtime can cause gaps.

An event from a group that is not on the allowlist is consumed and discarded. If you add that group later, Idiolect cannot restore the discarded event.

Only one collector must use a `signal-cli` data directory at one time.
