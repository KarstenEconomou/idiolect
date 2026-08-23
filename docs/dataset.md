# Dataset Build

## Purpose

Dataset construction is a batch operation. Collection does not create target-specific model text.

The builder reads normalized messages and writes MLX-LM completion JSONL. It does not contact Signal or a model service.

Each row teaches one conditional task: given only conversation state that was
available before one clean target message, predict the exact text of that
message. It does not teach media generation, message editing, reply selection,
or reaction selection. Consecutive messages from the target remain separate
examples because each Signal message is one observed next-utterance decision.

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
just data build DIXIE
```

The recipe builds data for the local Signal account. Use the direct CLI command with `--person` for another consenting person.

Use a normalized person ID for another consenting person:

```console
just idiolect data build --person PERSON_ID --name TARGET_NAME
```

The name becomes model text. The person ID remains the identity key.

## Transformation

The builder uses this order:

1. Verify unique message IDs, time zones, revisions, and reaction links.
2. Select clean text-only target messages.
3. Sort eligible target messages by time and message ID.
4. Make chronological train, validation, and test groups.
5. Purge context at each group boundary.
6. Select earlier messages from the same chat.
7. Remove context revisions that were not available at target time.
8. Assign one stable pseudonym to each other person.
9. Render messages, mentions, replies, attachments, and causal reactions from the target view.
10. Write completion JSONL, a private source index, and a manifest.

A clean target contains visible text or a native mention. The builder excludes
deleted targets, edited targets, attachment-bearing targets, non-text targets,
and text that contains only whitespace or Signal object-replacement markers.
An attachment caption is not a clean target because its meaning and action are
not reproducible by a text-only model. The manifest reports one count for each
exclusion reason. The completion otherwise preserves the source text exactly.

The normalized store contains the current message revision. For a context
message, the builder includes an edit only when its edit time is before the
target. It includes a deletion tombstone only when its deletion time is before
the target. It omits a message when its stored edit or deletion comes at or
after target time because the earlier content is not available in normalized
storage. A message with the same timestamp as the target is also omitted; its
causal order is not known.

Purged boundaries prevent one source message from entering more than one dataset split. The first example in a new split can have less context.

`data.context` is a message count, not a time interval. The builder uses at most
that many prior same-chat messages after the applicable split boundary. It does
not infer a conversation timeout. Exact repeated replies such as `yes` remain
in the corpus because repetition frequency is part of the target distribution;
the evaluation stage measures training-text overlap separately.

## Output

The `data.output` setting selects the output path. The example uses:

```text
var/data/<dataset-id>/
├── train.jsonl
├── valid.jsonl
├── test.jsonl
├── index.jsonl
└── manifest.json
```

The builder omits an empty split file. Each JSONL row has this form:

```json
{"prompt":"TARGET-RELATIVE CONTEXT","completion":"TARGET MESSAGE"}
```

`index.jsonl` maps each private row to its split, row number, hashed chat ID,
target message ID, target time, context message IDs, and causal reaction event
IDs. It is audit data and is not passed to MLX-LM.

The dataset ID is a SHA-256 value over the source recipe, split counts,
selection audit, pseudonym mapping, canonical JSONL hashes, and source-index
hash. The source digest canonicalizes mention order inside each message and
quote, so equivalent source data hashes equally.
The loader checks that identity, every file hash, every split count, each
row schema, the source-index order, and source disjointness across splits. A
second equal build returns the same directory. The loader rejects missing,
changed, or unrecorded files.

All dataset files are private. Git ignores `var/`. Do not copy a dataset, manifest, or adapter into a tracked path.

The verified dataset metadata reader exposes only the dataset ID, target name,
recorded context-message count, and split counts needed by local chat discovery.
It first verifies the complete immutable dataset and does not read Signal or
DuckDB.
