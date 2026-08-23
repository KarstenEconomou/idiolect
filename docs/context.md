# Conversation Context

## Identity rules

A hashed person ID is the identity source. A display name is model text. Two people can have the same display name.

The renderer receives one target person ID and one target name. For example, it can receive the target name `DIXIE`.

- Plain `DIXIE` stays `DIXIE`.
- A native Signal mention of the target becomes `@DIXIE`.
- The message header adds `mentions @DIXIE` for a native target mention.
- A native mention of another person becomes that person's configured pseudonym.
- A matching display name does not change the mentioned identity.

This rule gives the model similar text for `DIXIE` and `@DIXIE`. The `@` and the message header supply the additional addressing data.

## Mention normalization

Signal supplies mention identity, start, and length data. The source range uses UTF-16 code units. The collector keeps the original text and range. The dataset renderer replaces the source range with readable `@Name` text.

Do not find native mentions with a text search. Text such as `@DIXIE` can be ordinary text. Only Signal mention metadata proves that a person was tagged.

## Reply normalization

Each reply stores the referenced message ID. It also stores the quote author, send time, text, and mentions when Signal supplies them.

The quote is a historical snapshot. Use it before the current referenced message. The referenced message can have a later edit or can be outside the context window.

Native reply edges are first-class discourse relationships. A reply keeps its parent regardless of elapsed wall-clock time: the dataset builder walks reply ancestry beyond the recency window so a delayed reply retains its antecedent. When the parent cannot be selected, the quote snapshot still labels the entry.

The renderer identifies these cases:

- a context episode replies to the target;
- a context episode replies to another person;
- a target episode's native reply selects its context anchor (the edge is used for selection only; see below);
- the reply target is not available.

## Response episodes

A response episode is one invocation-worth of behavior by one speaker. Consecutive messages of one author form one conversational contribution while their bubble boundaries stay visible. Model text distinguishes three separations:

```text
[person_01]                       <- speaker header (speaker change or
one                                episode boundary = new entry)
[new message]                     <- Signal-message boundary inside one
two                                contribution
```

Consecutive entries with the same author label are separate response episodes,
for example because another participant intervened in between.

## Timing

Internal timestamps drive episode grouping, availability, splitting, and
purging. Time gaps are not rendered into model text. Coarse relative timing
would require a serving-side redesign to keep train/serve parity, so it stays
a deliberate non-goal for now.

## Attachments and reactions

An attachment-only context message becomes `[attachment]`. When source text is
a caption, the renderer preserves the text and marks the message header with
the attachment count. Attachment names and media bytes do not enter model text.

The renderer interleaves stored reaction and reaction-removal events with
messages by event time. It includes only reactions strictly before the target
message. The reaction author and referenced message author use the same
target-relative names as message headers. Reactions are context only; they are
not training completions.

## Training form

The target name appears in the instruction, message text, and addressing metadata.

```text
You are DIXIE. Write only DIXIE's next response.

Conversation:

[person_01 | mentions @DIXIE | reply to DIXIE: "maybe"]
Hey @DIXIE, are you coming?

[next response]
```

The completion is the target's next response episode, with internal Signal
message boundaries preserved by the `[new message]` line. The marker carries
no reply metadata: a native reply by the target selects its context anchors at
build time, but the live chat never hands the model a preselected reply
target, so conditioning on one would be an oracle. A future structured model
can make reply selection part of the predicted action; the dataset index
already records `reply_parent_message_id` per episode so that extension does
not need another destructive redesign.

Interactive chat references use the same header grammar for the user's next
context entry. The UI number is local to the chat transcript and is not sent to
the model. Adapter-backed chat renders the referenced author and full quoted
bubble as `reply to AUTHOR: "text"`; BASE chat removes the UI token and sends
no reply metadata.

Supply a stable pseudonym for every other person across one dataset. The renderer rejects a missing pseudonym. Do not put Signal UUIDs, phone numbers, or hashed database IDs into model text.

## Reindex

Raw stored events keep their original mention and quote data. Rebuild normalized records from these events:

```console
just idiolect signal reindex
```

Stop continuous collection before this command. Start collection again after it finishes. The command reads the local DuckDB event table. It does not read the phone and does not contact Signal.
