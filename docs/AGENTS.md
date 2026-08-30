# Documentation Guide

These instructions apply to `docs/`. Read the root `AGENTS.md` first.

## Audience and purpose

Write for a person who operates, audits, or develops the current repository.
Help that person make a correct decision or complete a task safely. Do not turn
`docs/` into a code tour, design notebook, test report, or change log.

Each page has one job:

- `index.md` routes readers to the right page and records only cross-stage
  ordering or process restrictions.
- `signal.md` and `launchd.md` cover Signal setup, collection, and continuous
  operation.
- `data.md` covers private runtime paths, stored-record meaning, writer rules,
  inspection, and repair.
- `context.md` defines the shared meaning of identity, mentions, replies,
  episodes, and model conversation text.
- `dataset.md`, `train.md`, `inference.md`, `chat.md`, and `eval.md` are focused
  stage procedures. They cover inputs, policy choices, safe operation,
  observable behavior, artifacts, stop conditions, and interpretation owned by
  that stage.
- `security.md` is the canonical source for data classification, secrets,
  consent, publication limits, and local protection.
- `development.md` covers contributor setup, package boundaries, test
  isolation, and repository verification. Operator procedures must not absorb
  contributor workflow.

If information does not fit one of these jobs, omit it or improve the one
canonical page whose job it is. Do not create a new page only to hold incidental
details.

## Content rules

- Start from the reader's task. Include only facts needed to prepare, execute,
  verify, recover, or interpret that task.
- State prerequisites before commands, stop and concurrency conditions before
  the risky operation, and expected results after it.
- Explain a stable contract at the level visible in configuration, commands,
  model text, stored records, artifacts, reports, or failures. Omit private
  helper names, call sequences, class inventories, and incidental algorithms.
- Keep commands executable from the repository root. Use public `idiolect`
  commands for operations and `just` recipes for setup, build, and verification.
- Use synthetic placeholders. Never include private identifiers, messages,
  paths, credentials, or generated artifacts.
- Link to the canonical page for shared material. Give only the local context
  needed to make the link useful; do not restate the linked section.
- Write every documentation link with its literal repository path as the visible
  text, for example `[docs/security.md](security.md)`. Do not use a page title or
  descriptive phrase as the link text.
- Keep headings task-oriented and the reading path linear. Prefer short prose,
  small tables for exact mappings, and lists only for genuine sequences or
  sets. Do not append miscellaneous notes or catch-all reference sections.
- Record the current supported state only. Delete or rewrite superseded
  instructions in the same change.

## Update test

Before editing a page, identify the reader-visible contract that changed and
the decision or task the new text supports. If neither is specific, do not add
the text.

After editing, verify that:

1. the information is in the page that owns it;
2. no other page now needs the same explanation;
3. commands, paths, option names, defaults, and artifact shapes match the code
   and public configuration;
4. safety limits and immutable/private artifact rules remain explicit where
   the reader acts on them; and
5. the page contains no work log, future plan, or implementation inventory.
