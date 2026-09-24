# Dimension COR — Correctness (logic / edge-case / wrong-result bugs)

Audit your target for code that computes or returns the **wrong answer**, mishandles an edge case, or
breaks an invariant — independent of memory safety (that's MEM's job, though they overlap).

## What to hunt

- **Edge cases & boundaries.** Empty input, single element, zero-length string/array, off-by-one in
  ranges/slices/loops, inclusive-vs-exclusive bound errors, integer overflow/truncation (esp. casts
  between intp/int32/int64/uintp), signed/unsigned confusion, division/modulo by zero, negative sizes.
- **Numeric edge values.** NaN, ±inf, signed zero, denormals where they change a branch; float
  equality used as a control predicate; `sqlite3_result_double` NaN->NULL changing `IS NULL` /
  aggregate NULL-skip semantics.
- **Platform-variable C types.** A `signatures` entry using `int64` for C `long` is **correct on
  POSIX (LP64) and wrong on Windows x64 (LLP64, long=32-bit)** — silently corrupts registers. Same
  risk for `time_t`, `size_t`. Functions historically affected: `fseek`/`ftell`/`fsetpos`/`fgetpos`,
  `time`/`clock`, `strtol`/`strtoul`. Verify each signature against the real C prototype.
- **State-machine / lifecycle correctness.** SQLite errmsg read after the next API call (stale
  pointer -> wrong/garbage message); `sqlite3_expanded_sql` result not freed; aggregate state read in
  `xValue` then wrongly released before `xFinal`; statement reset/clear order.
- **Graph algorithms.** `compile_kernel` / `_kernel_partition`: topological order, external-variable
  discovery, segment minimality (1 + max jit/Python alternations along any dependency path), demotion
  on probe failure, incremental `recompute` cone selection (does the changed-input cone include every
  affected node, and exclude unaffected ones? diamonds are the classic trap), type-change flush/reseed.
- **Caching/fingerprint correctness.** Does the content-addressed fingerprint cover everything that
  changes behavior (code, consts, defaults, closure cells, referenced globals/helpers, module, jit
  flags)? A miss => a stale cached kernel returns wrong results. numba hashes `co_code` but **not**
  `co_consts`, so pure-numeric-literal body changes don't shift `co_code` — does numbox's anchor
  compensate?
- **Generated-code hazards.** Generated identifiers must be keyword- and newline-safe and collision-
  free; format-string/argument mismatches; `%n` must be rejected at typing (incl BSD `q`, MS
  `I32`/`I64`).
- **Concurrency / reentrancy.** Module-level mutable state (e.g. `sqlite3_lib`, anchor orphan sweep
  with a 60s age filter) touched from multiple compiles/threads.
- **Contract drift.** A function's docstring/sig promises X but the body does Y; a wrapper that
  reorders/forgets an argument; a default that doesn't match the C default.

## Known-correct patterns — do NOT report as bugs

- Zeroing the sqlite3_vtab base fields (core owns them). The literal-address presence check in
  `call.py` is intentional and never consumed by codegen (the extern ref does the real work).

Verify the current code's actual behavior before relying on any domain note above.
