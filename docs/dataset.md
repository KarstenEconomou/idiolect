# Dataset Build

## Purpose

Dataset construction is a batch operation. Collection does not create target-specific model text.

The builder reads normalized messages and writes MLX-LM completion JSONL. It does not contact Signal or a model service.

Each row teaches one conditional task: given only conversation state that was
available before one clean target response episode, generate that whole
response episode. It does not teach media generation, message editing, reply
selection, or reaction selection.

## Concepts

```text
Signal event
    |
message
    |
response episode
    |
discourse/thread structure
    |
target training example
```

- A **message** (or bubble) is one Signal message as normalized from one raw event.
- A **response episode** is one invocation-worth of behavior by one speaker.
  It contains one or more consecutive messages of that speaker and is the
  supervised conversational unit.
- **Discourse structure** is the temporally ordered conversation plus native
  Signal reply edges from parent message to reply message.
- A **target training example** is one prompt/completion row for one clean
  target response episode.

A Signal message is a serialization action. A response episode is the
conversational behavior that Idiolect learns. The words "message" and "bubble"
always mean one Signal message; "response" and "episode" always mean the
supervised unit.

## Episode rules

Episode construction is structural and causal. It never inspects what message
text means.

Messages share one response episode when all of these hold:

1. They are in the same chat.
2. They have the same author.
3. No message from another participant intervenes. Any other-participant
   message terminates the current episode.
4. The gap since the previous same-author message is at most
   `data.burst_gap_seconds`.
5. The message carries no native reply to a message outside the current
   episode. A reply to an incompatible antecedent starts a new episode even
   between rapid bubbles.

`data.burst_gap_seconds` is a duration in seconds. The default is 120.
Composition bursts are usually seconds apart; two minutes separates bursts
without splitting deliberate multi-message pacing. The manifest records gap
diagnostics for calibration: the sample count, minimum, maximum, median,
p90, and p99 of consecutive same-author gaps in the source, and how many
gaps exceed the configured threshold. It also records the counts of response
episodes and multi-message episodes.

An unusable bubble (deleted, edited, attachment-only, non-text) inside an
otherwise usable burst does not poison its neighbors. It terminates the
current clean run like any intervening observable event, so the surrounding
clean bubbles become separate training episodes. The manifest reports both
per-message exclusion reasons and episode-level accounting.

## Transformation

The builder uses this order:

1. Verify unique message IDs, time zones, revisions, reply targets, quotes,
   and reaction links.
2. Construct response episodes for every chat and author with the rules above.
3. Select clean target episodes: maximal runs of clean messages authored by
   the target person.
4. Sort target episodes by start time and first message ID.
5. Make chronological train, validation, and test groups of episodes.
6. Purge context at each group boundary using whole episode ends.
7. For each target episode, select causal context: recent whole episodes up
   to `data.context` messages, plus anchored reply ancestry beyond that budget.
8. Remove context revisions that were not available at episode start time.
9. Assign one stable pseudonym to each other person.
10. Render context episodes, reactions, mentions, replies, and attachments
    from the target view.
11. Write completion JSONL, a private source index, and a manifest.

A clean bubble contains visible text or a native mention. The builder excludes
deleted bubbles, edited bubbles, attachment-bearing bubbles, non-text bubbles,
and text that contains only whitespace or Signal object-replacement markers.
An attachment caption is not clean because its meaning and action are not
reproducible by a text-only model.

Splitting happens only after episode construction. One training episode is
atomic for train/validation/test assignment, so a three-bubble burst can never
be divided across splits.

## Context selection

Context for one target episode comes from the same chat and ends strictly
before the episode's first message:

- **Recent window:** newest first, the builder keeps whole context episodes
  until the next episode would exceed `data.context` messages. The window
  never divides an episode.
- **Reply anchors:** the native reply ancestry of the target episode bypasses
  the message budget. Each ancestor contributes its whole episode plus that
  ancestor's own direct reply antecedent. A delayed reply therefore keeps its
  parent even when ordinary recency truncation would drop it.

A context episode is usable only when it starts entirely after the applicable
split-purge bound, ends strictly before the target start time, and every
member was available at target time. An edit or deletion at or after target
time hides the message. A message with the same timestamp as the target start
is omitted because causal order is unknown. When a referenced reply parent
cannot be included, the rendered entry still shows the quote snapshot that
Signal recorded.

`data.context` is a maximum message count, not a time interval. Exact repeated
replies such as `yes` remain in the corpus because repetition frequency is
part of the target distribution; the evaluation stage measures training-text
overlap separately.

Context selection never reads target completion text. Only signals available
before generation decide context: native reply metadata, mention metadata,
chat membership, chronological structure, and availability times.

## Serialization

One response episode renders its messages in order separated by the reserved
boundary line `[new message]`:

```text
I don't know
[new message]
seems overfit to me
[new message]
especially on the style evals
```

The marker is a plain textual line rather than a tokenizer special token, so
every tokenizer and the whole MLX-LM stack work unchanged. Source text that
contains `[new message]` as a standalone line is rejected at build time, which
makes the contract deterministic and exactly reversible: splitting the
serialized text on the exact delimiter restores the original bubbles. The
completion preserves the fact that the target sent distinct Signal messages
and teaches how the target fragments one utterance.

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
{"prompt":"TARGET-RELATIVE CONTEXT","completion":"TARGET RESPONSE EPISODE"}
```

`index.jsonl` maps each private row to its split, row number, hashed chat ID,
episode ID, every constituent target message ID, episode start and end times,
the native reply parent ID, thread anchor IDs, context message IDs, and causal
reaction event IDs. It is audit data and is not passed to MLX-LM. Given any
row, the index identifies exactly which raw messages formed the target
episode, which messages supplied context, which reply relationships were used,
and which configuration governed grouping and selection.

The dataset ID is a SHA-256 value over the source recipe, split counts,
selection audit, pseudonym mapping, canonical JSONL hashes, and source-index
hash. The recipe commits to schema version 1, the response-episode unit, the
bubble boundary, and `burst_gap_seconds`, together with the source digest. The
source digest canonicalizes mention order inside each message and quote, so
equivalent source data hashes equally. A second equal build returns the same
directory. The loader verifies identity, every file hash, every split count,
each row schema, the source-index order, and source disjointness across splits,
and rejects any manifest whose recorded schema version is not 1.

All dataset files are private. Git ignores `var/`. Do not copy a dataset,
manifest, or adapter into a tracked path.

The verified dataset metadata reader exposes only the dataset ID, target name,
recorded context-message count, and split counts needed by local chat discovery.
It first verifies the complete immutable dataset and does not read Signal or
DuckDB.
