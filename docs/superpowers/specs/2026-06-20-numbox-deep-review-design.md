# numbox deep project-wide review — durable, paced

**Status:** design approved 2026-06-20
**Branch:** `review/numbox-2026-06-20` (review artifacts only; no source changes, never pushed without explicit consent)

## Goal

Run an exhaustive, multi-agent review of the entire numbox library (~28k LOC, 140 Python
files) across five dimensions, structured so that **no completed review is ever lost** when
usage limits interrupt the work, and so that the agent throughput is **paced at 10 agents per
hour** to ride under those limits.

Success = a prioritized, adversarially-verified findings report plus a fix-plan `tasks.json`,
produced incrementally with every completed unit durably persisted to disk at the moment it
finishes.

## Core principle: disk is the source of truth

The Workflow tool's built-in `resumeFromRunId` cache is **same-session only** and a usage-limit
kill mid-agent loses that agent's in-flight work. Therefore the durability guarantee does **not**
rely on any in-memory journal. Instead:

- Each review unit is one agent that **writes a self-contained findings file the instant it
  completes**.
- The driver computes its work-list by **subtracting what is already on disk** — completed units
  cost nothing on every resume.
- New files are `git commit`ed after each batch, so work also survives an OS reinstall (consistent
  with the "survives reinstalls" preference).
- Resume is idempotent and mechanism-independent: re-running the same loop command — even days
  later, even in a brand-new session — picks up exactly where it left off. Automated hourly wakes
  are a convenience, not a dependency.

## On-disk layout

```
docs/reviews/2026-06-20-numbox/
  manifest.json                       # full unit list + derived status; regenerated from disk each wake
  findings/<dimension>/<target>.json  # one file per completed review unit
  verified/<dimension>/<target>.json  # one file per completed verification unit
  REPORT.md                           # final synthesis; regenerated from verified findings
  numbox-review.tasks.json            # final deliverable: prioritized fix plan
```

A unit is "done" iff its output file exists on disk. `manifest.json` is a *derived* convenience
index, never the authority — if it disagrees with the `findings/`/`verified/` directories, the
directories win and the manifest is regenerated.

## Units = target × dimension

A **unit** is one (review target × dimension), handled by one specialist agent producing one
findings file. Targets are coherent module groups so a kill costs at most one unit.

### Dimensions (all five)

1. **Correctness** — logic / edge-case / wrong-result bugs; numba-specific footguns already
   documented in memory (`@cfunc` swallowing exceptions and leaking NRT refs across a raising
   call; `sqlite3_result_double` coercing NaN→NULL; refcount handling).
2. **Memory / ABI / refcount safety** *(highest value for this codebase)* — NRT refcount leaks,
   `MemInfo` lifetime, struct-passing ABI (Win64 register vs pointer, AAPCS64), use-after-free,
   BLOB/TEXT NUL handling and fixed-width buffer bridges.
3. **Design & simplification** — duplication across the ~25 binding files, dead code,
   over-complexity, API consistency, reuse opportunities.
4. **Security & input validation** — C-binding bounds/validation, SQL and format-string safety.
5. **Tests & docs** — coverage gaps, doc/API drift, sphinx build health.

### Target × dimension matrix (dimensions applied where relevant, not a full cross-product)

- **Correctness** → all ~20 module targets.
- **Memory/ABI/refcount** → bindings + `utils/lowlevel` + `utils/meminfo` + `variable`/`work`
  refcount paths (~14 targets).
- **Design/simplification** → all ~20 module targets.
- **Security/input-validation** → C-bindings only (sqlite + libc, ~12 targets).
- **Tests & docs** → ~5 area sweeps + one sphinx-build check.

≈ **75 review units.** The exact target list is enumerated in the implementation plan.

### Module targets (indicative grouping)

- sqlite: `vtable`, `tvf`, `udf`+`udf_helpers`, `query`, `conn`/`stmt`/`exec`, `bind`/`column`/`value`/`result`/`blob`, `typemap`/`constants`, `hooks`
- libc: `fmtio`, `math`, `stdio`/`strerror`/`errno`, `_c`
- abi layer: `abi`/`call`/`signatures`/`bindings/utils`
- variable graph: `compile_kernel`, `_kernel_partition`, `variable`, `node`/`utils`
- work graph: `work`, `builder`(+utils), `node`(+base), `combine`/`loader`/`lowlevel_work_utils`, `print_tree`/`explain`
- core misc: `any`, `proxy`, `vector`, `configurations`
- utils: `lowlevel`, `highlevel`, `meminfo`, `fingerprint`, `pysqlite_bridge`, `cstrings`/`digest`/`clock`/`preprocessing`/`standard`/`timer`

## Adversarial verification

Every review unit's findings are verified before they can reach the report. Verification is
**per-unit** (one refute-agent reads a unit's findings file and tries to refute each finding,
defaulting to "refuted" when uncertain) and writes `verified/<dimension>/<target>.json`. A finding
reaches the report only if it survives verification. ≈ **75 verify units.**

## Paced driver loop (one turn per hour)

Each turn:

1. Scan `findings/` and `verified/` → compute `pending` = all units whose output file is absent.
2. Dispatch the **next 10** pending agents (review or verify, in dependency order — a unit's
   verify cannot precede its review, but reviews write to disk so a later batch's verify reads
   them). Every agent's first action: if my output file already exists, no-op and return.
3. `git commit` the newly written files on the review branch.
4. If `pending` is still non-empty → `ScheduleWakeup(3600s)` (1 hour, the tool's max) with the same
   loop prompt, then **end the turn**.
5. If `pending` is empty → run synthesis (below), commit, and **stop** (no further wakes).

**Counting:** every agent dispatch — review *and* verify — counts toward the 10/hour throttle, so
heavy and light work are paced uniformly. Synthesis counts as agent dispatches too.

## Synthesis (final stage)

Once all review + verify units are on disk:

- Regenerate `REPORT.md`: only verified-surviving findings, grouped by dimension and by severity,
  each with file:line, the concrete failure, and a recommended fix. Highest-value section first
  (memory/ABI/refcount).
- Generate `numbox-review.tasks.json`: a prioritized fix plan, one task per actionable finding (or
  coherent cluster), each with a `model` field (per the per-task-model preference) and acceptance
  criteria. This is the handoff artifact for a later fix campaign.
- Synthesis always reads from disk, so it is itself idempotent and cheap to re-run.

## Deliverable & boundaries

- **Output:** verified `REPORT.md` + `numbox-review.tasks.json`. **No source-code changes** in this
  workstream; fixes are a separate, later effort driven by the tasks file.
- **Branch:** `review/numbox-2026-06-20`, **never pushed without explicit per-push consent**
  (fork/no-push workflow). Review artifacts are internal `docs/` content, excluded from any
  upstream PR.
- numbox is currently on `main`, clean, and 0 commits behind `upstream/main`, so the review reads a
  current tree.

## Estimated scope

≈ 75 review + ≈ 75 verify + synthesis ≈ **~155 agent dispatches → ~16 one-hour batches ≈ a full
day-plus of paced wall-clock.** Inherent in 10/hour; interruptible at any batch boundary with zero
loss. Levers if shorter is wanted later: fewer targets, or drop the verify pass (faster, lower
confidence).

## Verification context (CI parity, for the tests/docs dimension)

The repo's CI defines the canonical checks the tests/docs dimension should reason about:
`numbox_ci.yml` (pytest matrix), `doc-codeblock-flake8.yml`, `docs.yml` (sphinx), `link-check.yml`
(lychee), `security_scan.yml`. The review reports gaps against these; it does not run fixes.
