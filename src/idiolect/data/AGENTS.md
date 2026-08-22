# Data Guide

These instructions apply to `src/idiolect/data/`.

- Build examples from the configured target person's point of view. Preserve
  chronological context, target attribution, replies, mentions, reactions, and
  the shared conversation grammar.
- Keep rendering deterministic and separate from artifact orchestration.
  `render.py` transforms validated records; `local.py` selects, splits, writes,
  verifies, and loads immutable datasets.
- Preserve time order within splits and prevent context or target leakage across
  split boundaries. Do not randomize examples unless TOML explicitly defines
  that policy.
- Validate participant names, mentions, context windows, and required message
  relationships before writing output. Reject ambiguous or incomplete source
  data instead of silently changing it.
- Make dataset identity depend only on canonical policy and content. Write
  artifacts atomically and never mutate an existing dataset.
- Keep examples and tests synthetic. Never read live Signal data or `var/` during
  verification.
- Use `just idiolect data people` to list normalized authors and
  `just data build <name>` to build the linked user's dataset. These are private
  operational commands; do not run them for code verification.
- Update `docs/dataset.md` when transformation, splitting, identity, or output
  behavior changes.
