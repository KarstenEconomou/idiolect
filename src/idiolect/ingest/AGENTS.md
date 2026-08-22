# Ingest Guide

These instructions apply to `src/idiolect/ingest/`.

- Keep `Source` and `Parser` as backend-neutral ports. Put orchestration in
  `harvest.py` and Signal-specific process and JSON behavior in `signal.py`.
- Accept only configured Signal group IDs. Discard direct messages and events
  from groups outside the whitelist.
- Preserve source text and Signal metadata exactly where the data contract
  requires it. Keep mention ranges in Signal's UTF-16 units. Record attachment
  metadata only; do not download or store attachment bytes.
- Derive deterministic IDs from stable source values. Treat a repeated source
  event as a duplicate, and keep event plus normalized-record persistence in one
  repository transaction.
- Preserve revision semantics: a newer edit or remote delete can replace an
  older message revision; an older revision cannot replace a newer one.
- Keep `signal-cli` execution behind the source boundary. Do not invoke a live
  account in tests or as part of ordinary code verification.
- Use `just collect status`, `just collect start`, and `just collect stop` only
  for user-requested collector operations. Follow `docs/signal.md` and
  `docs/launchd.md`; do not install or operate the LaunchAgent during code work.
- Update `docs/signal.md` when collection requirements, parsing, normalization,
  filtering, or operation changes.
