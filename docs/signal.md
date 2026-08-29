# Signal Collection

## Requirements

Install the core repository environment, a current `signal-cli` release, and a
local QR-code tool. Run commands from the repository root.

```console
just setup
mkdir -m 700 -p var/signal
```

Keep `signal-cli` current. A Signal service change can stop an old client.

## Link the local device

Link `signal-cli` as a secondary device. Do not register the account as a new
primary device.

```console
signal-cli --data-dir var/signal link -n Idiolect
```

The command returns a `sgnl://` device-link URI. Convert it to a QR code on the
local computer. In the Signal mobile application, open **Settings → Linked
devices → Link new device** and scan the code.

The link URI is a temporary credential. Do not use an online QR service. The
`var/signal` directory contains persistent device credentials. Do not delete it
while the device is active.

## Configure private values

Create the private environment file:

```console
touch .env
chmod 600 .env
```

Add the linked account in international number format:

```sh
IDIOLECT_SIGNAL_ACCOUNT="+14165550123"
```

The number is a placeholder. Do not commit a real account.

Just recipes that launch Idiolect load `.env`. A direct `uv run idiolect`
command must include `--env-file .env` when it needs these values.

Supported launch environment values are:

| Value | Purpose |
|---|---|
| `IDIOLECT_SIGNAL_ACCOUNT` | Select the linked Signal account. |
| `IDIOLECT_SIGNAL_CHATS` | Set the group whitelist as a JSON list. |
| `IDIOLECT_SIGNAL_BIN` | Set the `signal-cli` executable path. |
| `IDIOLECT_SIGNAL_DATA_DIR` | Set the private Signal state directory. |
| `IDIOLECT_CONFIG` | Select a public TOML policy. |

Signal environment values replace the corresponding TOML values. Group IDs are
valid only in `IDIOLECT_SIGNAL_CHATS`.

## Set the group whitelist

List groups that the linked device can see:

```console
just idiolect signal groups
```

The output has this form:

```text
GROUP_ID=    active    Group name
```

Add approved group IDs to `.env`:

```sh
IDIOLECT_SIGNAL_CHATS='["GROUP_ID_ONE=", "GROUP_ID_TWO="]'
```

Keep the outer single quotes and inner double quotes. Do not add a trailing
comma. The loader rejects an empty, invalid, or duplicate ID.

An event from a group outside the whitelist is consumed and discarded. Adding
the group later does not restore that event.

## Collect events

Run one bounded receive operation:

```console
just idiolect signal collect
```

Run until you stop the process:

```console
just idiolect signal collect --follow
```

Set explicit bounds when required:

```console
just idiolect signal collect --timeout 30 --max-messages 100
```

`--follow` and `--timeout` are mutually exclusive. `max_messages` must be
greater than zero. A timeout must be `-1` or greater.

Show stored counts:

```console
just idiolect signal stats
```

Import saved `signal-cli` JSON lines:

```console
just idiolect signal import path/to/events.jsonl
```

## Collection guarantees and limits

- The collector accepts incoming group messages and sent-message sync events.
- The whitelist rejects direct messages and unapproved groups.
- The collector records text, mentions, replies, edits, deletions, reactions,
  and attachment metadata.
- The collector does not download attachment bytes, stories, avatars, or
  stickers.
- One malformed or unprocessable event does not stop later events.
- One accepted event is stored in one DuckDB transaction.
- Duplicate source JSON does not create a duplicate event.
- A newer message revision wins over an older revision.

The collector receives queued events after setup. It does not import existing
phone history. Signal does not guarantee an unlimited queue period.

Use only one process with a `signal-cli` data directory. Use only one DuckDB
writer. Stop continuous collection before `reindex` or dataset construction.

For continuous collection, use the [macOS LaunchAgent procedure](launchd.md).
