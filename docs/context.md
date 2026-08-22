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

The renderer identifies these cases:

- a message replies to the target;
- a message replies to another person;
- the target response replies to a selected message;
- the reply target is not available.

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
You are DIXIE. Write only DIXIE's next message.

Conversation:

[person_01 | mentions @DIXIE | reply to DIXIE: "maybe"]
Hey @DIXIE, are you coming?

[next response | reply to person_01: "Hey @DIXIE, are you coming?"]
```

The completion contains only the target message text. A first model must generate reply text. It does not select a Signal reply action. A later structured-output model can select `reply_to` separately.

Supply a stable pseudonym for every other person across one dataset. The renderer rejects a missing pseudonym. Do not put Signal UUIDs, phone numbers, or hashed database IDs into model text.

## Reindex

Raw stored events keep their original mention and quote data. Rebuild normalized records from these events:

```console
just idiolect signal reindex
```

Stop continuous collection before this command. Start collection again after it finishes. The command reads the local DuckDB event table. It does not read the phone and does not contact Signal.
