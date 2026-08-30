# Chat Guide

These instructions apply to `src/idiolect/chat/`. Read
`src/idiolect/tui/AGENTS.md` as well when a change affects presentation or input.

## Ownership

- `discovery.py` builds the configured base persona and discovers verified run
  assistants and saved snapshots.
- `state.py` owns the in-memory transcript, prompt preparation, context policy,
  retries, attempts, seeds, and turn telemetry.
- `storage.py` writes and verifies explicit immutable snapshots.
- `worker.py` owns the isolated MLX process and its typed command/event protocol.
- `runtime.py` coordinates one session with the supervised worker. Keep Textual
  imports and presentation behavior out of this package.

## Contracts

- Keep chat prompts byte-compatible with the shared training conversation
  grammar. Display the local user as `USER`, but serialize the configured
  participant name in prompts. Do not change recorded adapter target-name casing
  inside model prompts.
- Format adapters as `IDIOLECT // NAME::run [BASE]`, where `NAME` is the uppercase
  display of the recorded target name, `run` is a unique eight-character run
  prefix, and `BASE` is the last model repository or path component. Format the
  configured base as `IDIOLECT // NAME::BASE [BASE]`.
- Use uppercase `NAME` for assistant transcript labels. Use the full canonical
  identity in discovery, headers, snapshots, and every non-transcript view.
- Validate chat policy only at the chat boundary so other stages can use a
  configuration without chat settings. Use the recorded context window and
  remove only oldest whole messages when prompt pressure requires it.
- Keep all MLX ownership inside the supervised worker. Model verification and
  loading must run outside the Textual event loop. Keep the worker protocol
  serializable and keep backend imports inside the worker process.
- Preserve the in-memory transcript when loading, generation, cancellation,
  saving, or worker recovery fails. An unsaved chat must not create a temporary
  transcript file.
- Save only on an explicit record action. Snapshots are private, immutable,
  content-addressed, lineage-aware artifacts. Verify assistant identity, policy,
  turns, telemetry, and digests when loading them.
- Erase only a verified saved snapshot with no verified child. Keep lineage
  parents until they become leaves through explicit child erasure.
- Rename a verified leaf by creating its immutable replacement with the same
  lineage parent, then erase the replaced leaf.
- Use `just setup-chat` for optional packages, `idiolect chat` for `REGISTRY`,
  `idiolect chat --run <run> --dataset <dataset>` for a verified assistant, and
  `idiolect chat --resume <chat>` for a verified snapshot. Chat commands read
  private state; do not run them for code verification.
- Review `docs/chat.md` when its operator-visible policy, identity, commands,
  context behavior, snapshots, operational limits, or recovery steps change.
  Edit only the affected procedure or contract; internal worker changes alone
  do not require documentation.
