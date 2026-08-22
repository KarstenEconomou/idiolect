# Test Guide

These instructions apply to `tests/` and all test subdirectories. Before changing
a mirrored suite, read the matching guide under `src/idiolect/`. For root suites
such as `test_config.py` or `test_prompt.py`, read `src/idiolect/AGENTS.md` and
inspect the production contract directly.

## Quality bar

- Optimize for confidence per test, not test count or coverage percentage. Every
  test must catch a realistic defect, protect an external contract, or serve a
  clear structural purpose.
- Test public behavior, data contracts, invariants, meaningful boundaries,
  invalid inputs, failure behavior, integration boundaries, and known
  regressions. For each bug fix, add a regression test that fails without the
  fix.
- Before editing a suite, inspect its production contract and review existing
  tests critically. Preserve tests for historical bugs, compatibility, subtle
  edge cases, and otherwise-uncovered behavior. Rename or rewrite a legitimate
  test whose purpose is unclear.
- Remove tests that only execute lines, restate implementation details, check
  obvious constructors, getters, or constants, repeat an equivalent case, or
  enumerate branches without adding a semantic property.
- Test schemas only when serialization is nontrivial, validation encodes a
  business rule, compatibility matters, coercion or aliases matter, the schema
  is an external API, or a regression requires it. Assert the behavior rather
  than every declared field or default.
- Keep assertions focused on one behavior or one closely related contract. Use
  parameterization only when all cases establish the same semantic property.
  Avoid arbitrary permutations and combinatorial matrices.
- Name files `test_*.py` and tests `test_<behavior>`. A test name must state the
  protected behavior, such as
  `test_rejects_unknown_metric_when_loading_eval_config`.

## Isolation and fixtures

- Never read live configuration, credentials, environment-specific paths,
  `var/`, user files, real Signal data, chat transcripts, or private artifacts.
  Tests must not depend on `.env` or mutable machine state.
- Use `tmp_path`, synthetic messages, explicit safe settings, and deterministic
  fixtures or factories. A fixture must create and clean up all state it owns and
  must never fall back to live state.
- Keep fixtures minimal, scoped, easy to override, and proportionate to the
  behavior. Use a factory for small variations of one valid object; avoid giant
  fixtures and one-use abstraction layers.
- Never contact Signal, model hubs, tracking systems, cloud services, or other
  networks. Replace each external boundary with a small fake or a mock at the
  defined port.
- Never download a model or run training, fine-tuning, inference, GPU work, or
  another expensive routine. Test the local policy, inputs, outputs, state
  changes, and failures around the expensive boundary.
- Prefer in-memory fakes when behavior across a port matters. Use mocks only at
  defined boundaries. Do not mock the unit under test or assert incidental call
  order.
- Make time, randomness, identifiers, and ordering deterministic. Do not use
  sleeps or timing assumptions when an event, state poll, synchronization point,
  or fake clock can express the contract.

## Required stage coverage

- Ingest and store tests protect duplicate-event handling, transaction behavior,
  revision ordering, and normalized relationships. Use temporary DuckDB files.
- Data tests protect chronological splitting, leakage prevention, target-relative
  rendering, and immutable artifact identity. Use temporary Parquet and artifact
  directories.
- Train, inference, and evaluation tests fake MLX and subprocess boundaries and
  verify complete policies, recorded identities, deterministic seeds, artifacts,
  and failure behavior without loading weights.
- Chat worker tests use fake backends and explicit synchronization. TUI tests use
  a fake runtime and Textual pilot sessions. They must not import MLX, require a
  GPU, or contact a model hub.
- Documentation tests protect commands, public examples, and links without
  reading private configuration.

## Verification and reporting

Run the affected tests first, then the broader relevant suite, then `just check`
before handoff. Do not weaken an assertion, replace behavior coverage with an
import check, or change production behavior only to make a test pass.

After a test cleanup, review every remaining test and ask: "What realistic
regression would this catch?" Remove it unless the answer is convincing or it
has a documented structural purpose. Report tests removed or consolidated,
fixtures introduced, live-state dependencies eliminated, important coverage
retained or added, questionable tests kept and why, and every command and result.
