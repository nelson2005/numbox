# Dimension TST — Tests & docs

Audit your assigned **area** (a span of source + its tests + its docs) for coverage gaps, doc/API
drift, and build/CI health risks. You assess by **reading** — do NOT run pytest, sphinx, flake8, or
lychee (reason about what they would catch). Use `Glob`/`Grep` to find the relevant test and doc files
under the paths in your target's `files`/`notes`.

## What to hunt

- **Coverage gaps on risky paths.** For each non-trivial source function in your area, is there a test
  that exercises it — and specifically its **unhappy paths**? Prioritize: error/exception paths
  (`@cfunc` error signalling, raise-on-mid-step), edge values (NaN/±inf, empty, NUL-containing
  TEXT/BLOB, interior NUL, max-length), refcount-leak assertions (a test that the meminfo count is
  balanced after use), align-1 unaligned regression tests, ABI paths (Win64/SysV/AAPCS64, >16B sret,
  16B repack), platform-variable C types, fingerprint/cache collision and cross-process reload,
  segmentation/demotion and recompute cones (diamond/fan-out). Name the untested function and the
  specific scenario.
- **Tests that don't test much.** Asserts that can't fail, tests that exercise the happy path only,
  tests that would pass even with the bug present, over-mocking that hides the real path,
  platform-gated tests that silently skip everywhere in CI.
- **Doc / API drift.** Public API (non-underscore, star-imported) missing from the sphinx docs; an
  `automodule` directive pointing at a renamed/removed module, or a new `_*.py` module not added to
  `docs/numbox.core.bindings.rst`; docstrings describing old behavior; README claims that no longer
  hold; a "Follow-ups"/gotchas note that is now stale.
- **Doc code-block health.** Python in `.rst code-block::` and `.md` fenced blocks must be
  flake8-clean (CI runs `doc-codeblock-flake8`). Spot snippets that would fail (undefined name,
  unused import, bad indentation) or that reference an API that no longer exists.
- **CI parity.** Compare what the workflows check (`numbox_ci.yml` pytest matrix Python 3.10-3.14 ×
  ubuntu/arm/windows/macOS × min/max numba, `--durations=20`; `doc-codeblock-flake8.yml`; `docs.yml`
  sphinx; `link-check.yml` lychee; `security_scan.yml`) against the actual risk surface — call out a
  dimension of risk no CI job covers (e.g. a platform/path only one runner exercises).

## Output

Same findings schema. A TST finding's `file` is the source OR test OR doc file at issue; `lines` may
be a function name + approximate location when a precise line isn't apt; `recommendation` names the
test/doc to add or fix. `severity` reflects the risk of the **untested/undocumented** behavior
(an untested silent-wrong-result path is high; a missing docstring is low).
