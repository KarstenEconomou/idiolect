# Idiolect Operations

This directory contains the replication record for the current system.

- [Signal collection](signal.md): Install, link, configure, and run `signal-cli`.
- [Data store](data.md): Follow the record flow and inspect DuckDB.
- [Conversation context](context.md): Preserve names, mentions, and replies for training.
- [Dataset build](dataset.md): Build immutable target-specific MLX-LM files.
- [Adapter training](train.md): Configure and run local QLoRA adapters.
- [Local inference](inference.md): Generate fixed base and adapter predictions.
- [Local chat](chat.md): Stream multi-turn replies and save private snapshots.
- [Model evaluation](eval.md): Compare policy fidelity and run private blind ratings.
- [Security](security.md): Separate public settings from private data.
- [macOS service](launchd.md): Run collection with `launchd`.
- [Development](development.md): Verify changes without live data or model calls.

## Current state

The package implements the Signal collector, DuckDB store, context renderer,
dataset builder, MLX-LM trainer, local inference runner, supervised chat TUI,
automatic policy evaluation, and private familiar-panel workflow.

The collector uses this flow:

```text
signal-cli JSON line
        │
        v
validate and apply group whitelist
        │
        v
create raw event and identity-linked record
        │
        v
write one DuckDB transaction
```

The local `launchd` agent runs `idiolect signal collect --follow`. The agent starts at user login. The agent runs only while the Mac is on, awake, and logged in.
