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
  primary text, and inactive row names. Use the selected ANSI green, yellow,
  blue, magenta-purple, or cyan only for accents and active primary labels;
  bright black for metadata, help text, telemetry, inactive actions, dividers,
  and idle borders; ANSI white for table and menu headings; and ANSI red for
  failures and unavailable actions. Use terminal `grey23` as the character-width
  background for transcript code. Do not add RGB colors, opaque modal backdrops,
  tints, or separate surface colors.
- Bold product marks, primary page titles, canonical chat identities, table and
  menu headings, ready status labels, and the primary label of the current
  selection. Do not bold body text outside focused transcript Markdown,
  telemetry, help text, descriptions, or inactive actions. Give supporting
  metadata in a selected item the accent color with dim styling.
- Keep the watermark blue. Render its mark and product name bold, and render its
  tagline non-bold and dim. Put the blue `· ·` brand motif at the right edge of
  the chat and SPECS identity headers with one blank cell after it.
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
  row identity primary. Treat TYPE and ENTRY like slash-command descriptions:
  use metadata gray while idle and the dimmed selection accent while selected.
  Treat unavailable rows like unavailable `/save`: metadata gray and dimmed.
- Let `T` in REGISTRY cycle the interface accent through ANSI green, yellow,
  blue, magenta-purple, and cyan. Use green by default, advance to yellow on the
  first press, wrap to green after cyan, and do not advertise this branding
  control in the navigation hints.
- For a `TRACE`, keep the canonical model identity first and show its trace name
  to the right in the remaining MODEL width. Ellipsize only the trace name when
  it does not fit. Expand trace names by default. Let Space toggle all trace
  names whenever the registry contains a TRACE. Treat an expanded trace name
  like TYPE and ENTRY while its TRACE is highlighted: use the selection accent
  with dim styling.
- Keep slash commands in a vertical menu above the composer. Show at most three
  rows, with command names in a fixed-width column and descriptions in the
  remaining width. Keep composer focus. Wrap enabled commands with arrow keys;
  Tab completes the selection plus one trailing space; Enter runs it; Escape
  closes the menu without clearing the composer. Show disabled commands red and
  dim, except for the metadata-gray disabled `/save`, and skip them when another
  enabled command exists.
- Offer `/save` only when the current transcript has new unsaved data. Request
  its optional trace name with the existing field, record the checkpoint, and
  keep the chat open. Disable the command after the record succeeds. An explicit
  `/save` in a clean session shows a failure.
- Keep unsaved-change actions in one horizontal row ordered `DISCONNECT`,
  `SAVE`, `RESUME`. Select `DISCONNECT` first. Wrap all arrow keys, activate
  with Enter, and make Escape equivalent to `RESUME`. Disable pointer activation.
  After `SAVE`, request an optional trace name in a single-line field. Enter
  records it, show the generated default as the empty field placeholder, an empty
  or whitespace-only value requests that default, and Escape resumes without
  recording.

## Layout and state

- Align registry headings and rows, transcript turns, loading status, and footer
  to the same two-cell gutter. Inset the composer, slash menu, and confirmation
  by one cell and align their internal content. Give the confirmation the
  composer width and place it immediately above the composer. Keep the footer
  directly below the composer with no vertical padding.
- Keep REGISTRY full-width like chat. Do not add an outer horizontal inset or a
  centered maximum width around its existing two-cell content gutter.
- In SPECS, let Left and Right cycle with wrapping through available registry
  rows, skip FAULT rows, reset the details scroll position, and keep the cycled
  row selected when the user returns to REGISTRY.
- In SPECS, treat field names like command names and values like command
  descriptions: use terminal-default field names and metadata-gray values. Keep
  section headings white and bold. Put every field value, including evaluation
  bars, in a one-cell inset block beneath its label so wrapped continuation lines
  align. Preserve prompt-format lines, but omit trailing blank system-prompt
  lines. Dim the unfilled portion of evaluation bars.
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
- Show transient alerts as right-aligned lines above the applicable control bar.
  Use sentence case, preserve uppercase entity names, match loading-status
  spacing, and terminate each message with punctuation. Use metadata gray for
  informational and success alerts, and reserve the failure color for errors.
  Do not use toasts.
- Show `BACKSPACE MANAGE` only for a highlighted TRACE. Confirm it with a `TRACE`
  heading followed directly by its metadata-colored name and horizontal
  `ERASE`, `RENAME`, `RETAIN` actions. Select `RETAIN` first, and make Escape
  retain the trace. Use the trace name as the rename-field default. Blink the
  subject trace name while either menu is open.
- Update `docs/chat.md` and `README.md` when visible behavior or controls change.
