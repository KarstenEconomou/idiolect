# Dataset Build

## Purpose

Dataset construction is a batch operation. Collection does not create target-specific model text.

The builder reads normalized messages and writes MLX-LM completion JSONL. It does not contact Signal or a model service.

## Prepare existing data

Stop continuous collection before a batch operation. This rule prevents two processes from opening the same DuckDB file for writes.

For the documented `launchd` service, stop and later start it with:

```console
just collect stop
just collect start
```

Run the second command only after the batch commands finish.

Refresh records once after an update changes normalization:

```console
set -a
source .env
set +a
just idiolect signal reindex
```

The command rebuilds normalized records from immutable raw events.

## Find a target

List normalized authors:

```console
just idiolect data people
```

The output contains the hashed person ID, `self` or `member`, message count, and latest source display name. The command does not print a phone number or Signal UUID.

Use `--self` for the linked Signal account:

```console
just data build Karsten
```

The recipe builds data for the local Signal account. Use the direct CLI command with `--person` for another consenting person.

Use a normalized person ID for another consenting person:

```console
just idiolect data build --person PERSON_ID --name TARGET_NAME
```

The name becomes model text. The person ID remains the identity key.

## Transformation

The builder uses this order:

1. Select non-deleted text messages from the target.
2. Sort target messages by time and message ID.
3. Make chronological train, validation, and test groups.
4. Purge context at each group boundary.
5. Select prior messages from the same chat.
6. Assign one stable pseudonym to each other person.
7. Render mentions and replies from the target view.
8. Write completion JSONL and a manifest.

Purged boundaries prevent one source message from entering more than one dataset split. The first example in a new split can have less context.

## Output

The `data.output` setting selects the output path. The example uses:

```text
var/data/<dataset-id>/
├── train.jsonl
├── valid.jsonl
├── test.jsonl
└── manifest.json
```

The builder omits an empty split file. Each JSONL row has this form:

```json
{"prompt":"TARGET-RELATIVE CONTEXT","completion":"TARGET MESSAGE"}
```

The dataset ID is a SHA-256 value from the source snapshot and transformation recipe. The manifest records the recipe, split counts, pseudonym mapping, and file hashes. A second equal build returns the same directory. The builder rejects a changed file.

All dataset files are private. Git ignores `var/`. Do not copy a dataset, manifest, or adapter into a tracked path.
