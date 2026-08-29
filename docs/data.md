# Private Data Store

## Runtime paths

The canonical policy keeps private state under `var/`:

```text
var/
├── signal/                 signal-cli keys and account state
├── idiolect.duckdb         raw events and normalized records
├── log/                    LaunchAgent output
├── data/<dataset-id>/      immutable datasets
├── models/                 fixed model snapshots
├── runs/<run-id>/          immutable adapter runs
├── inference/<id>/         immutable predictions
├── chat/<chat-id>/         immutable chat snapshots
└── eval/                   evaluations, judgments, and panels
```

Git ignores `var/`. Create it with mode `0700` on a shared computer. Idiolect
creates private artifact directories with mode `0700` and files with mode
`0600`.

## DuckDB records

`events` is the source record. It contains the received raw JSON and source
identity. Reindexing reads this table.

`messages` contains the current normalized message revision. It records reply,
edit, deletion, and revision times. Dataset construction uses those times to
reconstruct what was available before a target response.

`mentions` keeps native Signal identity and UTF-16 ranges for message and quote
text. `attachments` keeps metadata only. `reactions` keeps reaction and removal
events.

Normalized identifiers are SHA-256 values. Raw events still contain the
original Signal identifiers and text.

## Writer rules

The collector stores each accepted source event in one transaction. A duplicate
event does not create another record. A failed transaction does not keep a
partial event.

Edits and deletions use revision time. An older revision cannot replace a newer
revision. A deletion leaves a tombstone without message text.

Use only one DuckDB writer. Stop continuous collection before `reindex` or
dataset construction. Do not open the database with a write tool while Idiolect
uses it.

## Inspection and repair

Show stored record counts:

```console
idiolect signal stats
```

Rebuild normalized records from stored raw events:

```console
idiolect signal reindex
```

`reindex` does not contact Signal. Stop continuous collection before the
command. Start collection again after the command finishes.

Use a read-only DuckDB connection for manual inspection. Do not change raw or
normalized records by hand.
