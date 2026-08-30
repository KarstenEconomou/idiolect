# Idiolect Procedures

Use this page to select the correct procedure. Run commands from the repository
root.

## First setup

1. Read [docs/security.md](security.md) before creating private state.
2. Use [docs/signal.md](signal.md) to link the local device and configure the
   whitelist.
3. Use [docs/launchd.md](launchd.md) only for continuous collection.

## Data and model pipeline

Follow these procedures in order:

1. [docs/data.md](data.md) explains private paths, stored records, and writer
   restrictions.
2. [docs/context.md](context.md) defines identity, mention, reply, and
   response-episode meaning.
3. [docs/dataset.md](dataset.md) defines causal selection, split isolation, and
   artifact contents.
4. [docs/train.md](train.md) defines complete experiment policies and local
   MLX-LM runs.
5. [docs/inference.md](inference.md) defines target selection and immutable
   prediction artifacts.
6. [docs/eval.md](eval.md) defines the evidence, gates, and familiar-rater
   procedure.

## Local chat

Use [docs/chat.md](chat.md) to open the configured base model, a verified adapter,
or a saved snapshot. Chat does not read Signal or DuckDB.

## Development

Use [docs/development.md](development.md) for package boundaries,
test isolation, and required checks.

## Process restrictions

- Use only one process that writes the Signal data directory.
- Use only one process that writes DuckDB.
- Stop continuous collection before `reindex` or dataset construction.
- Collection can run during training, inference, chat, and evaluation. These
  stages use immutable files.
- On macOS, Idiolect keeps the Mac awake during training, batch inference, and
  automatic evaluation.
- Keep all files under `var/` private.
