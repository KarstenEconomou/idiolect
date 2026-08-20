# Security

## Public files

Tracked files can contain these values:

- Program names
- Relative example paths
- Timeouts and limits
- Model names and training settings
- Empty or false example values

Do not put a real account, group ID, absolute private path, token, or message in a tracked file.

## Private configuration

Use `conf/idiolect.toml` for public and reproducible settings. Git tracks this file. Signal chat IDs are not valid TOML settings.

Use `.env` or a system secret store for these values:

- `IDIOLECT_SIGNAL_ACCOUNT`
- `IDIOLECT_SIGNAL_CHATS`
- `IDIOLECT_SIGNAL_DATA_DIR`
- Model hub tokens
- Experiment service tokens
- Future cloud credentials

Set `.env` mode to `0600`:

```console
chmod 600 .env
```

`IDIOLECT_SIGNAL_CHATS` is a JSON list of group IDs. The account number and group IDs are private metadata. They are not enough to authenticate a Signal device, but they must not be public.

Use the ignored `conf/local.toml` file only when you need a complete local configuration that differs from the canonical file. Set `IDIOLECT_CONFIG` to select it. Keep private values in `.env` or a system secret store.

## Key material and data

The `signal-cli` data directory contains cryptographic keys and account state. Do not put this data in `.env`. Do not commit, copy, or publish this directory.

The device link URI is a temporary credential. Render it locally, scan it, and discard it. Do not put it in a command history, issue, log, or online QR service.

The DuckDB file contains raw messages and original Signal identifiers. Hashed normalized IDs do not make the raw event table anonymous.

The log files can contain command errors and local paths. Keep `var/log/` private.

Model snapshots, model-specific datasets, adapters, run logs, and run manifests are private. An adapter can retain source text. Keep external experiment reporting disabled unless you approve the service and its data flow.

Use disk encryption. Limit local file access. Do not include `var/`, `.env`, or `conf/local.toml` in a public archive.

Git ignore rules reduce accidental commits. They do not remove an item from Git history. If a credential enters Git history, revoke or replace the credential.
