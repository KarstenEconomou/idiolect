# Idiolect Procedures

Use this page to select the correct procedure. Run commands from the repository
root.

## First setup

1. Read [security](security.md) before creating private state.
2. Use [Signal collection](signal.md) to link the local device and configure the
   whitelist.
3. Use the [macOS LaunchAgent](launchd.md) only for continuous collection.

## Data and model pipeline

Follow these procedures in order:

1. [Data store](data.md) explains private paths, stored records, and writer
   restrictions.
2. [Conversation context](context.md) defines identity, mention, reply, and
   response-episode meaning.
3. [Dataset build](dataset.md) defines causal selection, split isolation, and
   artifact contents.
4. [Adapter training](train.md) defines complete experiment policies and local
   MLX-LM runs.
5. [Local inference](inference.md) defines target selection and immutable
   prediction artifacts.
6. [Model evaluation](eval.md) defines the evidence, gates, and familiar-rater
   procedure.

## Local chat

Use [local chat](chat.md) to open the configured base model, a verified adapter,
or a saved snapshot. Chat does not read Signal or DuckDB.

## Development

Use [development and verification](development.md) for package boundaries,
test isolation, and required checks.

## Process restrictions

- Use only one process that writes the Signal data directory.
- Use only one process that writes DuckDB.
- Stop continuous collection before `reindex` or dataset construction.
- Collection can run during training, inference, chat, and evaluation. These
  stages use immutable files.
- Keep the Mac awake for long MLX-LM operations. The training, inference, and
  automatic evaluation recipes use `caffeinate`.
- Keep all files under `var/` private.
