# Dataset Build

## Purpose

Dataset construction converts normalized Signal records into one immutable
dataset for one target person. It does not contact Signal or a model service.

Each row asks one question: given the conversation state before a clean target
response episode, what complete response episode did the target send?

The dataset does not teach media generation, message editing, reaction
selection, or reply-target selection.

## Source units

- A **message** or **bubble** is one Signal message.
- A **response episode** is one or more consecutive messages from one speaker.
- A **target example** is one prompt and one target response episode.
- **Reply ancestry** is the chain of native reply parents that supplies thread
  context.

The response episode is the supervised unit. One episode always stays in one
split.

## Episode boundaries

Messages share one response episode only when all these conditions are true:

1. They belong to the same chat.
2. They have the same author.
3. No event from another participant occurs between them.
4. Their gap is not greater than `data.burst_gap_seconds`.
5. A reply does not point outside the current episode.

An unusable message ends the current clean run. It does not remove a clean
message before or after it.

A clean target message has visible text or a native mention. Dataset
construction excludes deleted, edited, attachment-bearing, non-text, blank, and
object-replacement-only target messages. An attachment caption is not a clean
text-only completion.

The manifest records exclusion counts and episode-gap diagnostics. Use these
values to review `burst_gap_seconds`; do not infer a suitable threshold from a
small sample.

## Split and purge rules

Dataset construction sorts target episodes by start time and stable ID. It then
creates chronological train, validation, and test splits.

The builder purges context at each split boundary. A validation or test prompt
cannot use a source message from an earlier split's unsafe boundary region. One
multi-message episode cannot cross a split.

The configured ratios apply to target episodes, not to individual messages.
Small datasets can produce an empty test split. Training always requires a
nonempty training and validation split. It also requires a test split when the
training policy enables testing.

## Causal context

Context comes from the same chat and ends before the target episode starts.

The recent window keeps newest whole episodes up to `data.context` messages. It
does not divide an episode. Reply ancestry can extend beyond this message limit.

A context record can enter a prompt only when it was available at target time.
A message with the same timestamp as the target start is not safe because its
order is unknown. A later edit or deletion does not enter an earlier prompt.

Context selection does not read target completion text. It uses only prior
message order, chat membership, reply metadata, mention metadata, and revision
availability.

## Text format

Messages in one target response episode use this reserved delimiter:

```text
I don't know
[new message]
seems overfit to me
```

The delimiter is ordinary text. It is not a tokenizer special token. Dataset
construction rejects source text that contains the delimiter as a standalone
line. This rule makes the serialization reversible.

The canonical dataset stores neutral text:

```json
{"prompt":"TARGET-RELATIVE CONTEXT","completion":"TARGET RESPONSE EPISODE"}
```

Training creates a private model-specific copy. It does not change the
canonical dataset.

## Build

Stop continuous collection. Build a dataset for the linked Signal account:

```console
idiolect data build TARGET_NAME
```

Use the CLI when the target is not the linked account:

```console
idiolect data build TARGET_NAME --person PERSON_ID
```

Start collection again after the build finishes.

## Artifact

The `data.output` setting selects the artifact root:

```text
var/data/<dataset-id>/
├── train.jsonl
├── valid.jsonl
├── test.jsonl
├── index.jsonl
└── manifest.json
```

An empty split file is omitted.

`index.jsonl` maps each row to its target messages, chat, episode times, reply
parent, thread anchors, context messages, and reaction events. It is private
audit data. MLX-LM does not read it.

The dataset ID commits to the source digest, selection policy, split counts,
pseudonym mapping, JSONL files, and source index. An equal build returns the
same directory. The loader verifies the ID, schema version, file hashes, row
schemas, split counts, index order, and split source isolation.

Do not edit an artifact. Create a new dataset when source data or policy
changes. Do not publish a dataset, index, or manifest.
