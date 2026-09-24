# Dimension DES — Design & simplification

Audit your target for code that works but is **more complex, more duplicated, or less consistent than
it needs to be** — the maintenance-hazard dimension. The bias of this repo's owner is explicitly
toward *simpler* solutions and *surgical* changes, so high-signal simplification findings are welcome;
gratuitous "rewrite everything" suggestions are not.

## What to hunt

- **Duplication.** The same logic copy-pasted across the ~25 binding wrapper files, repeated
  pointer/string idioms that a `utils/lowlevel.py` helper already provides (`array_data_p`,
  `get_str_from_p_as_int`, `get_unicode_data_p`, `load_at`/`store_at`, `load_unaligned`/
  `store_unaligned`), repeated ABI/classification logic, parallel near-identical functions that could
  be one parameterized function. Cite both sites.
- **Reinvention of an existing primitive.** A binding that hand-rolls a byte loop or pointer cast
  instead of composing the canonical helpers; a second fingerprint/digest path that duplicates
  `utils/fingerprint.py`.
- **Dead code / unused surface.** Unreferenced functions, parameters never read, branches that can't
  execute, leftover scaffolding, imports unused. (Report dead code; do not delete it.)
- **Over-complexity.** A 200-line construct that a senior engineer would call overcomplicated; nested
  conditionals that collapse; speculative configurability/abstraction with a single caller; an
  indirection layer that adds no value.
- **API consistency.** Public vs private discipline (leading underscore for internals; star-imported
  public surface in `bindings/__init__.py` — anything non-underscore at top level is public API).
  Naming consistency across sibling wrappers; argument-order consistency; return-type consistency
  (e.g. some wrappers raise, siblings return a code).
- **Module organization.** A file doing too many unrelated things; a helper in the wrong module; a
  grouping that fights the `automodule` doc structure.
- **Comment/code drift.** Comments describing a previous design; planning/task references left in
  comments (the repo forbids task numbers / phase refs in code comments).

## Calibration

- Prefer findings where the simpler form is **clearly** better and the change is **local and low
  risk**. Note the risk/benefit in `recommendation`.
- Match existing style; do not propose reformatting or renaming for taste alone.
- A genuinely clean, well-factored target should produce few or zero findings — say so in `summary`.
