# Security and Privacy

## Data classification

Tracked files can contain public software settings, model names, fixed model
revisions, relative example paths, and synthetic examples.

Do not put these values in a tracked file:

- credentials or access tokens
- a real Signal account or group ID
- a phone number or Signal UUID
- private message text or chat text
- an absolute private path
- a database, dataset, model, adapter, checkpoint, or log
- a prediction, snapshot, evaluation, judgment, or panel artifact

`conf/idiolect.toml` and `conf/exp/` are public policy files. Signal group IDs
are not valid public TOML settings.

## Secrets and local settings

Keep secrets and Signal identifiers in `.env` or a system secret store. The
supported Signal environment values are documented in
[docs/signal.md](signal.md).

Set the file mode before adding values:

```console
touch .env
chmod 600 .env
```

Product commands require the private environment values to be exported. When
running through `uv`, use `uv run --env-file .env idiolect ...`.

Use `IDIOLECT_CONFIG` only to select a public configuration file. Do not use it
to hide private policy values in another TOML file.

## Signal credentials

The `signal-cli` data directory contains device keys and account state. Keep it
under `var/signal/`. Do not copy it into `.env` or a tracked path.

The device-link URI is a temporary credential. Convert it to a QR code on the
local computer. Do not use an online QR service. Do not put the URI in a shell
history, issue, or log.

The DuckDB `events` table contains raw Signal JSON. Hashed normalized IDs do not
make the raw data anonymous.

## Model and evaluation artifacts

A dataset, adapter, generated reply, or evaluation can retain private source
text. Keep external experiment reporting disabled unless the data owner approves
the service and its data flow.

Familiar raters must consent. Each rater must already have permission to view
every sampled conversation. Run the rating on the data owner's computer. Use a
pseudonym that contains no contact data.

## Local protection

- Use disk encryption.
- Restrict access to the local user.
- Keep `var/`, `.env`, and installed LaunchAgent files out of public archives.
- Treat logs and manifests as private. They can contain paths and provenance.
- Back up private state only to an approved encrypted location.

Git ignore rules reduce accidental commits. They do not remove data from Git
history. Revoke or replace a credential if it enters Git history.
