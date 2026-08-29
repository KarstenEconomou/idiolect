# Test Guide

These instructions apply to `tests/` and all subdirectories. Before changing a
mirrored suite, read the matching guide under `src/idiolect/`. For root suites,
read `src/idiolect/AGENTS.md` and inspect the production contract directly.

## Principle: tests must have alpha

A test has **alpha** when it materially increases confidence that the
implementation is correct by detecting a realistic defect.

Optimize for **defect-detection value per unit of maintenance**, not test count
or coverage percentage.

A good test:

- fails for a plausible incorrect implementation;
- passes for a correct implementation;
- survives contract-preserving refactors;
- provides signal not already supplied by another test; and
- justifies its maintenance cost.

Before adding or retaining a test, ask:

> What realistic defect would this test catch?

If there is no convincing answer, do not add it. Remove or consolidate existing
tests with no distinct signal.

Tests verify implementation correctness. They do not mirror implementation,
enumerate branches, or create mandatory work whenever the repository changes.

## What to test

Test observable behavior, stable contracts, domain invariants, meaningful
boundaries, invalid inputs, failure modes, state transitions, integration
boundaries, and known regressions.

- Prefer the smallest stable interface that exposes the contract.
- For bug fixes, add a regression test only when existing tests do not already
  reproduce the failure and recurrence is plausible.
- Assert the strongest stable property relevant to the contract and no more.
- Prefer properties, invariants, round trips, or metamorphic relations over many
  example cases when they provide stronger signal.
- Use parameterization only when all cases establish the same semantic property.
- Test schemas only when serialization, validation, compatibility, coercion,
  aliases, or external API behavior is material.
- Avoid broad snapshots unless the representation itself is the contract.
- Do not add tests solely for uncovered lines, branches, functions, or files.
  Coverage is diagnostic information, not an objective.

Remove tests that only check constructors, getters, constants, framework
behavior, implementation details, equivalent cases, or arbitrary branch
permutations.

Name files `test_*.py` and tests `test_<behavior>`. Names must describe the
protected behavior.

## Isolation and fixtures

Tests must be hermetic by default.

- Use synthetic fixtures, `tmp_path`, temporary stores, explicit safe settings,
  deterministic state, and small factories.
- Never read live configuration, `.env`, credentials, `var/`, user data, real
  Signal data, private artifacts, home-directory state, or caches.
- Never contact Signal, model hubs, tracking systems, cloud services, or other
  networks.
- Never download models or run training, inference, GPU work, or other expensive
  routines in the normal suite.
- Fake external boundaries such as model providers, subprocesses, storage,
  clocks, and network clients.
- Prefer small functional fakes when boundary behavior matters. Use mocks for
  narrow interactions at defined ports.
- Do not mock the unit under test or private collaborators merely to observe
  implementation structure.
- Keep fixtures minimal and understandable. Avoid production captures, giant
  datasets, and one-use abstraction layers.
- Make time, randomness, identifiers, ordering, and asynchronous behavior
  deterministic. Do not use sleeps when explicit synchronization or fake clocks
  can express the contract.
- Tests must not depend on execution order or leave persistent state.

## Refactor resilience

Tests protect contracts, not implementations.

A contract-preserving refactor should normally leave tests unchanged. Widespread
test edits during an internal refactor indicate excessive implementation
coupling.

Do not test that production code:

- calls a private helper;
- uses a particular internal class or decomposition;
- creates a particular intermediate representation; or
- performs equivalent internal calls in a fixed order or count.

Assert call order or cardinality only when it is part of the external protocol.

If code requires extensive internal mocking to test, prefer improving the
dependency boundary rather than increasing mock choreography.

## High-value contracts

These are priorities, not coverage quotas.

- **Ingest/store:** duplicate handling, transactions, revision ordering, and
  normalized relationships. Use temporary DuckDB files.
- **Data:** chronological splitting, leakage prevention, target-relative
  rendering, and immutable artifact identity. Use synthetic repositories and
  temporary JSONL artifact directories.
- **Train/inference/evaluation:** fake MLX and subprocess boundaries; protect
  policies, identities, seeds, artifacts, and failure behavior without weights.
- **Chat/TUI:** use fake backends, explicit synchronization, fake runtimes, and
  Textual pilot sessions. Never require MLX, GPUs, or model hubs.
- **Documentation:** test executable commands, public examples, or links only
  when automated validation has meaningful alpha.

Prefer one strong invariant test over several tests that enumerate instances.

## Changing tests

A production-code change does not imply that tests should change.

When a test fails, classify it first:

1. **Implementation regression** — fix production code.
2. **Intentional contract change** — update the test.
3. **Implementation-coupled test** — rewrite or remove it.
4. **Invalid, redundant, or flaky test** — repair, consolidate, or remove it.
5. **Environmental failure** — repair isolation or infrastructure.

Never mechanically update expected values or weaken assertions to make the suite
green.

Delete tests when their contract disappears, another test subsumes them, they
protect obsolete implementation structure, or their maintenance burden exceeds
their unique signal.

Deleting a low-alpha test is test-suite maintenance, not loss of quality.

## Agent workflow

When modifying code:

1. Read the applicable guides, production contract, and existing tests.
2. Identify the behavior, invariant, or failure mode affected.
3. Determine whether existing tests already provide sufficient signal.
4. Reproduce a bug before fixing it when practical.
5. Add or modify the minimum tests needed to distinguish correct from incorrect
   behavior.
6. Implement the change.
7. Run the narrowest affected tests, then the broader relevant suite.
8. Run `just check` before handoff.
9. Remove temporary diagnostics and generated state.

Do not add a test for every changed function or rewrite healthy behavioral tests
to accommodate an implementation approach.

## Review and reporting

For each test, ask:

- What realistic defect does it catch?
- Would a plausible incorrect implementation fail it?
- Is that signal already covered?
- Does it test a contract rather than implementation structure?
- Would it survive a legitimate refactor?
- Is there a cheaper or stronger way to establish the same property?
- Is its maintenance burden justified?

If not, simplify, consolidate, or remove it.

After test work, report material tests added, removed, or consolidated; important
fixtures or fakes introduced; live-state or nondeterministic dependencies
removed; and validation commands run with their results.

The objective is not a larger test suite. It is the smallest maintainable suite
that provides strong evidence that the implementation is correct.
