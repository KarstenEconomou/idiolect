# Conversation Context

This document defines the meaning of model conversation text for dataset audits
and comparisons of training, inference, and chat behavior.

## Identity

A hashed person ID is the identity source. A display name is text. Two people
can have the same display name.

Dataset construction receives one target person ID and one target name. It uses
one stable pseudonym for each other person. Do not use a Signal UUID, phone
number, or hashed database ID as a model name.

For a target named `DIXIE`:

- plain source text `DIXIE` stays `DIXIE`;
- a native mention of the target becomes `@DIXIE`;
- the message header records that native mention;
- a native mention of another person uses that person's pseudonym.

Text alone does not prove a mention. Only Signal mention metadata identifies the
mentioned person.

## Mentions

Signal supplies mention start and length values in UTF-16 code units. The store
keeps the original text and ranges. Dataset construction replaces each verified
range with readable `@Name` text.

Do not find native mentions with a text search. Ordinary message text can
contain an at-sign and a name.

## Replies

A reply stores its parent message ID and the quote snapshot that Signal
supplies. The snapshot can remain valid when the parent is outside the context
window or has a later edit.

Reply ancestry can extend context beyond the recent-message limit. This rule
keeps a delayed reply with its antecedent. If the parent cannot enter context,
the quote snapshot still identifies the reply relationship.

A target reply selects context during dataset construction. Reply selection is
not part of the completion. This avoids an oracle that tells the model which
message its response must answer.

## Response episodes

A response episode is one conversational contribution by one speaker. It can
contain multiple consecutive Signal messages. The messages keep their internal
boundaries:

```text
[person_01]
one
[new message]
two
```

`[person_01]` starts one context entry. `[new message]` separates two Signal
messages in the same response episode. Two entries with the same speaker are
separate episodes when another event or an incompatible reply divides them.

## Time and availability

Internal timestamps control episode grouping, split purging, context selection,
and revision availability. Model text does not contain time gaps.

A target prompt can include only records that were available before the target
episode started. A later edit or deletion must not change an earlier prompt.

## Attachments and reactions

An attachment-only context message becomes `[attachment]`. A caption remains
text, and its header records the attachment count. File names and media bytes do
not enter model text.

Reaction and reaction-removal events enter context in event-time order. Only
events before the target response can enter its prompt. Reactions are never
training completions.

## Conversation form

One prompt has this general form:

```text
You are DIXIE. Write only DIXIE's next response.

Conversation:

[person_01 | mentions @DIXIE | reply to DIXIE: "maybe"]
Hey @DIXIE, are you coming?

[next response]
```

The completion is the target's next response episode. It can contain the
`[new message]` boundary.

Training, inference, evaluation, and adapter chat use the same conversation
meaning. A model-specific policy can add a system prompt, role, prefix, suffix,
or assistant prefill. The tokenizer renderer then defines the exact model-token
sequence.

## Reindexing

Raw events keep the source mention and quote data. Rebuild normalized records
after a normalization change:

```console
idiolect signal reindex
```

Stop continuous collection before this command. `reindex` reads DuckDB only. It
does not contact Signal.
