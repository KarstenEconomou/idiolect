# Chat TUI Guide

These instructions apply to `src/idiolect/tui/`. Read
`src/idiolect/chat/AGENTS.md` when presentation work changes chat behavior.

## Boundary

Keep this package limited to Textual presentation, input handling, and calls into
typed chat services. Do not put model loading, prompt policy, artifact
verification, snapshot identity, or storage policy here. Use fake runtimes in
TUI tests.

## Visual language

- Keep the interface sparse. Use the terminal default for the screen, surfaces,
  primary text, and inactive row names. Use ANSI blue only for accents and active
  primary labels; bright black for metadata, help text, telemetry, inactive
  actions, dividers, and idle borders; ANSI white for table and menu headings;
  and ANSI red for failures and unavailable actions. Use terminal `grey23` as
  the character-width background for transcript code. Do not add RGB colors,
  opaque modal backdrops, tints, or separate surface colors.
- Bold product marks, primary page titles, canonical chat identities, table and
  menu headings, ready status labels, and the primary label of the current
  selection. Do not bold body text outside focused transcript Markdown,
  telemetry, help text, descriptions, or inactive actions. Give supporting
  metadata in a selected item the accent color with dim styling.
- Use uppercase for interface nouns, identities, transcript speaker labels,
  table headings, status values, action labels, telemetry, and keyboard hints.
  Keep command names lowercase. Write command descriptions in sentence case and
  end them with a period. Render transcript content with the focused Markdown
  rules below. Keep stored content unchanged.
- Support Markdown emphasis, ATX headings, inline code, fenced code, lists,
  blockquotes, and explicit HTTP or HTTPS links in every transcript turn. Keep
  heading, list, and quote markers. Add one indentation cell per list or quote
  level. Hide emphasis and code delimiters. Underline link text, show its
  destination in the metadata color, and let pointer activation open validated
  web links. Keep unsupported or unmatched markup literal, and do not interpret
  Rich markup.
- Use a metadata-colored solid border for idle input and an accent-colored solid
  border for focused input. Use terminal-native reversed cursor and selection
  styling. Avoid decorative borders and persistent chrome.

## Selection and input

- Present assistants and saved snapshots in `REGISTRY`, a vertical,
  keyboard-only table without search or pointer activation. Start on the first
  available row, skip unavailable rows, and select with arrows and Enter. Keep
  row identity primary, data and window metadata dim, entry status bold, and
  unavailable rows red and dim.
- Keep slash commands in a vertical menu above the composer. Show at most three
  rows, with command names in a fixed-width column and descriptions in the
  remaining width. Keep composer focus. Wrap enabled commands with arrow keys;
  Tab completes the selection plus one trailing space; Enter runs it; Escape
  closes the menu without clearing the composer. Show disabled commands red and
  dim, and skip them when another enabled command exists.
- Keep unsaved-change actions in one horizontal row ordered `DISCONNECT`,
  `RECORD`, `RESUME`. Select `DISCONNECT` first. Wrap all arrow keys, activate
  with Enter, and make Escape equivalent to `RESUME`. Disable pointer activation.

## Layout and state

- Align registry headings and rows, transcript turns, loading status, and footer
  to the same two-cell gutter. Inset the composer, slash menu, and confirmation
  by one cell and align their internal content. Give the confirmation the
  composer width and place it immediately above the composer. Keep the footer
  directly below the composer with no vertical padding.
- Hide transcript and composer scrollbars. Keep the registry scrollbar in the
  metadata color. Separate turns with one blank line, put the speaker label on
  its own line, and inset every rendered message line by one cell. Use the same
  one-cell offset between menu headings and actions. Preserve stored content and
  its internal whitespace.
- Follow the newest turn only when the transcript is already at the bottom.
  Opening or closing a menu is the exception: scroll the newest turn into view.
  Add bottom space for an overlaid confirmation and remove it when it closes.
- Keep menu headings and actions to one line each. Replace telemetry with the
  applicable keyboard hints while a menu is open, then restore it. Show only
  measured footer values, remove secondary measurements at narrow widths, and
  hide the ready state.
- Update `docs/chat.md` and `README.md` when visible behavior or controls change.
