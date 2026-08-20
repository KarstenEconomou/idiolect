# Idiolect Operations

This directory contains the replication record for the current system.

- [Signal collection](signal.md): Install, link, configure, and run `signal-cli`.
- [Data store](data.md): Follow the record flow and inspect DuckDB.
- [Security](security.md): Separate public settings from private data.
- [macOS service](launchd.md): Run collection with `launchd`.
- [Development](development.md): Verify changes without live data or model calls.

## Current state

The Signal collector and the DuckDB store operate now. The dataset, training, evaluation, and inference packages contain contracts only.

The collector uses this flow:

```text
signal-cli JSON line
        │
        v
validate and apply group allowlist
        │
        v
create raw event and normalized record
        │
        v
write one DuckDB transaction
```

The local `launchd` agent runs `idiolect signal collect --follow`. The agent starts at user login. The agent runs only while the Mac is on, awake, and logged in.
