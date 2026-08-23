# Data Guide

These instructions apply to `src/idiolect/data/`.

- Treat one response episode (one invocation-worth of behavior by one speaker)
  as the supervised unit. Construct episodes with structural rules before
  splitting; never split messages into separate examples or across splits.
- Build examples from the configured target person's point of view. Preserve
  chronological context, target attribution, replies, mentions, reactions, and
  the shared conversation grammar.
- Keep rendering deterministic and separate from artifact orchestration.
  `episodes.py` groups messages; `render.py` transforms validated records;
  `local.py` selects, splits, writes, verifies, and loads immutable datasets.
- Preserve time order within splits and prevent context or target leakage across
  split boundaries. Do not randomize examples unless TOML explicitly defines
  that policy.
- Never derive context from target completion text. Select context only from
  causal signals that exist before generation.
- Validate participant names, mentions, context windows, burst-gap settings,
  and required message relationships before writing output. Reject ambiguous or
  incomplete source data instead of silently changing it.
- Make dataset identity depend only on canonical policy and content. Write
  artifacts atomically and never mutate an existing dataset.
- Keep examples and tests synthetic. Never read live Signal data or `var/` during
  verification.
- Use `just idiolect data people` to list normalized authors and
  `just data build <name>` to build the linked user's dataset. These are private
  operational commands; do not run them for code verification.
- Update `docs/dataset.md` when transformation, splitting, identity, or output
  behavior changes.
