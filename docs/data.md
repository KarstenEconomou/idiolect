# Data Store

## Paths

The default local paths are:

```text
var/
├── signal/                 linked-device keys and Signal state
├── idiolect.duckdb         raw events and normalized records
├── log/                    launchd output
├── data/<dataset-id>/      future Parquet datasets
└── run/<run-id>/           future model results
```

Git ignores `var/`. The collector creates the DuckDB file with mode `0600`. Create `var/` with mode `0700` on a multi-user computer.

## Tables

`events` is the source record. It contains the event ID, source name, source ID, receive time, raw JSON bytes, and store time.

`messages` is the current message revision. It contains the source event ID, hashed chat ID, hashed author ID, send time, text, reply ID, edit time, delete time, and revision time.

`attachments` contains metadata only. It contains the message ID, hashed attachment ID, media type, file name, and byte count.

`reactions` contains each reaction event. It contains the source event ID, target message ID, hashed chat ID, hashed author ID, value, time, and remove state.

## Write rules

The store starts one transaction for each accepted source event.

1. Check the event ID.
2. Stop if the event exists.
3. Insert the raw event.
4. Insert or update each normalized record.
5. Commit the transaction.

If DuckDB reports an error, the transaction does not commit. The collector reports the event ID that failed.

Edits and deletes use a revision time. The store keeps the record with the newest revision time. A delete creates a message tombstone with no text.

## Inspection

Use the application for the standard count check:

```console
uv run idiolect signal stats
```

Do not open the DuckDB file with a write tool while collection runs. Use a read-only connection for manual inspection.
