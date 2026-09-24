# numbox Deep Review — Durable Paced Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a disk-checkpointed, 10-agents-per-hour review harness over the numbox library and run it to a verified findings report plus a fix-plan `tasks.json`, losing zero completed work across usage-limit interruptions.

**Architecture:** Disk is the source of truth. Every review/verify unit is one agent that writes its own JSON file the instant it finishes; the driver computes "what's left" by subtracting on-disk files from the enumerated unit matrix (`targets.json`). Pacing lives in the main loop: dispatch the next 10 pending agents, `git commit`, then `ScheduleWakeup(3600s)` to repeat in one hour; synthesize when nothing is pending. Workflow scripts have no filesystem access, so the disk-scan and pacing run in the main loop, not inside a Workflow — each batch is a flat fan-out of up to 10 `Agent` dispatches.

**Tech Stack:** Python 3.12 (venv at `/home/erik/projects/numbox/venv/bin/python`), the `Agent` tool, `ScheduleWakeup`, plain JSON files, git.

**Spec:** `docs/superpowers/specs/2026-06-20-numbox-deep-review-design.md`

**Conventions (apply to every task and every dispatched agent):**
- `set -euo pipefail` on every Bash command; inline absolute values (no shell-var substitution).
- `git -C /home/erik/projects/numbox …` — never `cd`.
- venv python only: `/home/erik/projects/numbox/venv/bin/python`.
- All work on branch `review/numbox-2026-06-20`; **never pushed** without explicit per-push consent.
- Review/verify agents are **read-only on source**: their ONLY write is their one findings/verified JSON file. No source edits, no test runs, no background tasks (`run_in_background`/`Monitor` forbidden in subagents).
- No person names and no AI-provenance anywhere in committed artifacts.

---

## File Structure

```
docs/reviews/2026-06-20-numbox/
  targets.json                 # authoritative unit matrix: targets → files → dimensions
  findings.schema.json         # shape every findings file must follow
  bin/pending.py               # compute the work queue from disk (the resume engine)
  bin/test_pending.py          # standalone test for pending.py (NOT collected by repo CI)
  prompts/MEM.md COR.md DES.md SEC.md TST.md   # per-dimension review templates
  prompts/verify.md            # adversarial refute template
  prompts/synthesis.md         # report + tasks.json generation template
  RUNBOOK.md                   # the exact per-hour driver procedure + resume command
  findings/<DIM>/<target>.json # produced during execution, one per review unit
  verified/<DIM>/<target>.json # produced during execution, one per verify unit
  REPORT.md                    # final synthesis (verified findings, prioritized)
  numbox-review.tasks.json     # final deliverable: prioritized fix plan
```

The harness lives entirely under `docs/reviews/` so the repo's `test/`-scoped CI never collects `bin/test_pending.py`.

---

### Task 0: Scaffold work tree + authoritative `targets.json`

**Goal:** Create the review directory tree and the single authoritative unit matrix that every later step derives from.

**Files:**
- Create: `docs/reviews/2026-06-20-numbox/targets.json`
- Create (dirs): `docs/reviews/2026-06-20-numbox/{findings,verified}/{MEM,COR,DES,SEC,TST}/`, `docs/reviews/2026-06-20-numbox/{bin,prompts}/`

**Acceptance Criteria:**
- [ ] `targets.json` parses as JSON and every file path listed in it exists on disk.
- [ ] The derived review-unit count (sum of `len(dimensions)` over targets) is printed and equals the asserted total.
- [ ] All `findings/<DIM>/` and `verified/<DIM>/` subdirectories exist for the five dimensions.

**Verify:**
```bash
set -euo pipefail
D=/home/erik/projects/numbox
/home/erik/projects/numbox/venv/bin/python - <<'PY'
import json, pathlib
base = pathlib.Path("/home/erik/projects/numbox")
t = json.loads((base / "docs/reviews/2026-06-20-numbox/targets.json").read_text())
units = 0
for tgt in t["targets"]:
    for f in tgt["files"]:
        assert (base / f).exists(), f"missing source file: {f}"
    units += len(tgt["dimensions"])
print("targets:", len(t["targets"]), "review units:", units)
PY
```
Expected: prints `targets: 29 review units: 89` (no AssertionError).

**Steps:**

- [ ] **Step 1: Make the directory tree**

```bash
set -euo pipefail
R=/home/erik/projects/numbox/docs/reviews/2026-06-20-numbox
mkdir -p "$R/bin" "$R/prompts"
for dim in MEM COR DES SEC TST; do mkdir -p "$R/findings/$dim" "$R/verified/$dim"; done
echo "tree ready"
```

- [ ] **Step 2: Write `targets.json`** — the authoritative matrix. Dimension codes: `MEM` memory/ABI/refcount, `COR` correctness, `DES` design/simplification, `SEC` security/input-validation, `TST` tests&docs. Memory is applied broadly because it is the highest-value lens for numba-backed code (this is why the count is 89, above the spec's ~75 estimate — an intentional, documented widening of the MEM lens).

```json
{
  "review": "numbox deep review 2026-06-20",
  "targets": [
    {"id": "sqlite-vtable", "files": ["numbox/core/bindings/_sqlite_vtable.py"], "dimensions": ["MEM", "COR", "SEC", "DES"]},
    {"id": "sqlite-tvf", "files": ["numbox/core/bindings/_sqlite_tvf.py"], "dimensions": ["MEM", "COR", "SEC", "DES"]},
    {"id": "sqlite-udf", "files": ["numbox/core/bindings/_sqlite_udf.py", "numbox/core/bindings/_sqlite_udf_helpers.py"], "dimensions": ["MEM", "COR", "SEC", "DES"]},
    {"id": "sqlite-query", "files": ["numbox/core/bindings/_sqlite_query.py"], "dimensions": ["MEM", "COR", "SEC", "DES"]},
    {"id": "sqlite-conn", "files": ["numbox/core/bindings/_sqlite_conn.py", "numbox/core/bindings/_sqlite_stmt.py", "numbox/core/bindings/_sqlite_exec.py"], "dimensions": ["MEM", "COR", "SEC", "DES"]},
    {"id": "sqlite-data", "files": ["numbox/core/bindings/_sqlite_bind.py", "numbox/core/bindings/_sqlite_column.py", "numbox/core/bindings/_sqlite_value.py", "numbox/core/bindings/_sqlite_result.py", "numbox/core/bindings/_sqlite_blob.py"], "dimensions": ["MEM", "COR", "SEC", "DES"]},
    {"id": "sqlite-meta", "files": ["numbox/core/bindings/_sqlite_typemap.py", "numbox/core/bindings/_sqlite_constants.py", "numbox/core/bindings/_sqlite_hooks.py"], "dimensions": ["MEM", "COR", "SEC", "DES"]},
    {"id": "libc-fmtio", "files": ["numbox/core/bindings/_fmtio.py"], "dimensions": ["MEM", "COR", "SEC", "DES"]},
    {"id": "libc-math", "files": ["numbox/core/bindings/_math.py"], "dimensions": ["MEM", "COR", "SEC", "DES"]},
    {"id": "libc-io", "files": ["numbox/core/bindings/_stdio.py", "numbox/core/bindings/_strerror.py", "numbox/core/bindings/_errno.py"], "dimensions": ["MEM", "COR", "SEC", "DES"]},
    {"id": "bindings-c", "files": ["numbox/core/bindings/_c.py"], "dimensions": ["MEM", "COR", "SEC", "DES"]},
    {"id": "abi-layer", "files": ["numbox/core/bindings/abi.py", "numbox/core/bindings/call.py", "numbox/core/bindings/signatures.py", "numbox/core/bindings/utils.py"], "dimensions": ["MEM", "COR", "DES"]},
    {"id": "compile-kernel", "files": ["numbox/core/variable/compile_kernel.py"], "dimensions": ["MEM", "COR", "DES"]},
    {"id": "kernel-partition", "files": ["numbox/core/variable/_kernel_partition.py"], "dimensions": ["MEM", "COR", "DES"]},
    {"id": "variable", "files": ["numbox/core/variable/variable.py", "numbox/core/variable/node.py", "numbox/core/variable/utils.py"], "dimensions": ["MEM", "COR", "DES"]},
    {"id": "work", "files": ["numbox/core/work/work.py", "numbox/core/work/work_utils.py"], "dimensions": ["MEM", "COR", "DES"]},
    {"id": "work-builder", "files": ["numbox/core/work/builder.py", "numbox/core/work/builder_utils.py", "numbox/core/work/combine_utils.py", "numbox/core/work/loader_utils.py"], "dimensions": ["MEM", "COR", "DES"]},
    {"id": "work-node", "files": ["numbox/core/work/node.py", "numbox/core/work/node_base.py", "numbox/core/work/lowlevel_work_utils.py", "numbox/core/work/print_tree.py", "numbox/core/work/explain.py"], "dimensions": ["MEM", "COR", "DES"]},
    {"id": "core-containers", "files": ["numbox/core/any/any_type.py", "numbox/core/any/content_wrap.py", "numbox/core/any/erased_type.py", "numbox/core/proxy/proxy.py", "numbox/core/vector/vector.py", "numbox/core/configurations.py"], "dimensions": ["MEM", "COR", "DES"]},
    {"id": "utils-lowlevel", "files": ["numbox/utils/lowlevel.py"], "dimensions": ["MEM", "COR", "DES"]},
    {"id": "utils-highlevel", "files": ["numbox/utils/highlevel.py"], "dimensions": ["MEM", "COR", "DES"]},
    {"id": "utils-meminfo", "files": ["numbox/utils/meminfo.py"], "dimensions": ["MEM", "COR", "DES"]},
    {"id": "utils-pysqlite-bridge", "files": ["numbox/utils/pysqlite_bridge.py"], "dimensions": ["MEM", "COR", "SEC", "DES"]},
    {"id": "utils-misc", "files": ["numbox/utils/fingerprint.py", "numbox/utils/digest.py", "numbox/utils/preprocessing.py", "numbox/utils/standard.py", "numbox/utils/cstrings.py", "numbox/utils/clock.py", "numbox/utils/timer.py", "numbox/utils/void_type.py"], "dimensions": ["MEM", "COR", "DES"]},
    {"id": "tst-sqlite", "files": ["test/core"], "dimensions": ["TST"]},
    {"id": "tst-graph", "files": ["test/core"], "dimensions": ["TST"]},
    {"id": "tst-libc", "files": ["test/core"], "dimensions": ["TST"]},
    {"id": "tst-utils", "files": ["test/utils"], "dimensions": ["TST"]},
    {"id": "tst-docs", "files": ["docs"], "dimensions": ["TST"]}
  ]
}
```

- [ ] **Step 3: Run the Verify block above.** Expected `targets: 29 review units: 89`. If a path is missing, fix the path in `targets.json` (source may have moved) before continuing.

- [ ] **Step 4: Commit**

```bash
set -euo pipefail
D=/home/erik/projects/numbox
git -C "$D" add docs/reviews/2026-06-20-numbox/targets.json
git -C "$D" commit -m "review(harness): scaffold tree and authoritative target matrix" --quiet
git -C "$D" log -1 --format='%h %s'
```

---

### Task 1: Findings schema + per-dimension review prompt templates

**Goal:** Lock the exact output shape every review agent must write, and the per-dimension review instructions, so units are uniform and resume-stable.

**Files:**
- Create: `docs/reviews/2026-06-20-numbox/findings.schema.json`
- Create: `docs/reviews/2026-06-20-numbox/prompts/{MEM,COR,DES,SEC,TST}.md`
- Create: `docs/reviews/2026-06-20-numbox/prompts/verify.md`

**Acceptance Criteria:**
- [ ] `findings.schema.json` parses as JSON.
- [ ] All six prompt files exist and are non-empty.
- [ ] Each dimension template instructs the agent to (a) read only the listed files plus what they import, (b) emit findings conforming to the schema, (c) write exactly one file to its output path and nothing else.

**Verify:**
```bash
set -euo pipefail
R=/home/erik/projects/numbox/docs/reviews/2026-06-20-numbox
/home/erik/projects/numbox/venv/bin/python -c "import json; json.load(open('$R/findings.schema.json'))"
for f in MEM COR DES SEC TST verify; do test -s "$R/prompts/$f.md" || { echo "missing $f"; exit 1; }; done
echo "task1 ok"
```
Expected: prints `task1 ok`.

**Steps:**

- [ ] **Step 1: Write `findings.schema.json`**

```json
{
  "type": "object",
  "required": ["unit", "dimension", "target", "files_reviewed", "findings"],
  "properties": {
    "unit": {"type": "string", "description": "DIM/target, e.g. MEM/sqlite-vtable"},
    "dimension": {"enum": ["MEM", "COR", "DES", "SEC", "TST"]},
    "target": {"type": "string"},
    "files_reviewed": {"type": "array", "items": {"type": "string"}},
    "findings": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "severity", "confidence", "file", "line", "title", "detail", "recommendation"],
        "properties": {
          "id": {"type": "string", "description": "DIM-target-N"},
          "severity": {"enum": ["high", "medium", "low"]},
          "confidence": {"enum": ["high", "medium", "low"]},
          "file": {"type": "string"},
          "line": {"type": "integer"},
          "title": {"type": "string"},
          "detail": {"type": "string"},
          "recommendation": {"type": "string"}
        }
      }
    },
    "notes": {"type": "string", "description": "what was checked; anything deferred"}
  }
}
```

- [ ] **Step 2: Write `prompts/MEM.md`** (memory/ABI/refcount — highest-value lens)

```markdown
# Review dimension: Memory / ABI / refcount safety

You are reviewing numbox source for memory-safety defects ONLY. numbox is a numba-based
library; assume the reader knows numba's NRT, structrefs, MemInfo, and C-extension ABI.

Read the listed files and anything they import from numbox. Do NOT run code, edit source,
or run tests. Your ONLY output is one JSON file (path given below).

Hunt specifically for:
- NRT refcount leaks: incref without matching decref, decref-on-error paths missed,
  refs held across a call that can raise (a numba @cfunc swallows exceptions, returns a
  zero default with no unwind, and leaks any NRT refs held across the raising call).
- MemInfo lifetime: borrow vs own confusion, use-after-free, double-free, dangling
  raw pointers derived from a structref whose owner can be collected.
- ABI / struct-passing: wrong by-value vs by-pointer choice (Win64 passes 1/2/4/8-byte
  structs in registers, others by pointer; AAPCS64 differs), wrong ctypes/numba signature
  vs the C declaration, integer width / signedness mismatches at the boundary.
- Buffer & string handling: BLOB/TEXT NUL handling (native sqlite passes exact byte length
  and never NUL-truncates; a fixed-width bridge must trim trailing only and preserve interior
  bytes), off-by-one in length math, unbounded writes.
- result_double NaN/NULL: sqlite3_result_double coerces NaN to SQL NULL; ±inf pass through.

For each defect, give file, exact line, what goes wrong, and the concrete fix. Prefer
fewer high-confidence findings over speculation; mark uncertain ones confidence=low.
```

- [ ] **Step 3: Write `prompts/COR.md`** (correctness)

```markdown
# Review dimension: Correctness

You are reviewing numbox source for logic/correctness defects ONLY (wrong results, bad edge
cases, broken invariants). Read the listed files and what they import. Do NOT run code, edit
source, or run tests. Your ONLY output is one JSON file (path given below).

Hunt for: off-by-one and boundary errors; wrong operator/comparison; incorrect handling of
empty/None/NaN/zero-length inputs; control-flow that silently swallows errors (numba @cfunc
returns a zero default and does NOT unwind to C — a bare try/except is the cure, try/finally
RERAISES on numba 0.65.1); state/caching bugs (stale memoization, fingerprint collisions);
incorrect type coercion; assumptions about ordering or uniqueness that don't hold.

For each defect give file, exact line, the wrong behavior, a triggering input if you can name
one, and the fix. Mark speculation confidence=low.
```

- [ ] **Step 4: Write `prompts/DES.md`** (design/simplification)

```markdown
# Review dimension: Design & simplification

You are reviewing numbox source for design smells and simplification opportunities ONLY (not
bugs). Read the listed files and what they import. Do NOT run code, edit source, or run tests.
Your ONLY output is one JSON file (path given below).

Hunt for: duplicated logic across the ~25 binding files that could share a helper; dead code
(unused functions/vars/imports); over-complex constructs that a senior engineer would call
overbuilt; inconsistent APIs for parallel operations; leaky abstractions; files doing too many
things. Respect numba constraints — some apparent "complexity" is required for JIT; only flag
what is genuinely reducible without losing jitability.

For each item give file, line, why it's a problem, and a concrete simpler form. severity =
impact on maintainability. Mark confidence=low when unsure it's safe under numba.
```

- [ ] **Step 5: Write `prompts/SEC.md`** (security/input-validation)

```markdown
# Review dimension: Security & input validation

You are reviewing numbox C-binding source for input-safety defects ONLY. Read the listed files
and what they import. Do NOT run code, edit source, or run tests. Your ONLY output is one JSON
file (path given below).

Hunt for: missing bounds/length validation before a C call; SQL built by string concatenation
where a bound parameter belongs; format-string injection in fmtio/printf-family wrappers
(user-controlled format string); integer overflow in size/length math reaching malloc/memcpy;
trusting a caller-supplied pointer/length without checks; TOCTOU around handle validity.

This is a JIT/embedding library, not a network service — judge realistically: flag genuine
unsafe-input paths, not theoretical issues on values the library fully controls. Give file,
line, the unsafe path, a triggering input, and the fix. Mark confidence honestly.
```

- [ ] **Step 6: Write `prompts/TST.md`** (tests & docs)

```markdown
# Review dimension: Tests & docs

You are assessing test coverage and doc accuracy for an area of numbox. Read the area's tests
and the source they cover (and for tst-docs, the .rst files vs the public API). Do NOT run the
suite or build docs; assess by reading. Your ONLY output is one JSON file (path given below).

Hunt for: public functions/branches/error paths with no test; edge cases asserted nowhere
(empty input, NaN, NUL bytes, error returns); tests that assert on documented-undefined
behavior; doc/API drift (a documented symbol that moved/renamed/was removed, or a public symbol
absent from docs); sphinx autodoc references that would fail to resolve.

For each gap give the file (test or doc), the uncovered/incorrect item, why it matters, and what
to add. severity = risk of the untested/wrong area. Mark confidence=low for guesses.
```

- [ ] **Step 7: Write `prompts/verify.md`** (adversarial refute)

```markdown
# Adversarial verification

You are given one review unit's findings JSON. For EACH finding, try to REFUTE it: read the
cited file:line and surrounding context and decide whether the defect is real as described.
Do NOT run code, edit source, or run tests. Your ONLY output is one JSON file (path given below).

Default to "refuted" when uncertain — a finding must earn "confirmed". A finding is refuted if:
the cited code doesn't do what the finding claims; the "bug" is actually correct under numba
semantics; the input that would trigger it cannot occur; or the claim is too vague to act on.

Output one verdict per finding id:
{"unit": "...", "verdicts": [{"id": "MEM-sqlite-vtable-1", "status": "confirmed|refuted|uncertain", "reasoning": "..."}]}
Only "confirmed" findings reach the report; "uncertain" are listed separately for human triage.
```

- [ ] **Step 8: Run the Verify block, then commit**

```bash
set -euo pipefail
D=/home/erik/projects/numbox
git -C "$D" add docs/reviews/2026-06-20-numbox/findings.schema.json docs/reviews/2026-06-20-numbox/prompts
git -C "$D" commit -m "review(harness): findings schema and dimension prompt templates" --quiet
git -C "$D" log -1 --format='%h %s'
```

---

### Task 2: `pending.py` — the resume engine

**Goal:** A deterministic helper that reads `targets.json` plus the on-disk `findings/`+`verified/` dirs and prints the ordered work queue. This is what makes resume lossless and mechanism-independent.

**Files:**
- Create: `docs/reviews/2026-06-20-numbox/bin/pending.py`
- Test: `docs/reviews/2026-06-20-numbox/bin/test_pending.py`

**Acceptance Criteria:**
- [ ] On an empty tree, `--summary` reports 89 review pending, 0 verify available, complete=False.
- [ ] A review unit drops out of the queue once its `findings/<DIM>/<target>.json` exists, and its verify unit becomes available.
- [ ] `--next 10` prints at most 10 queue lines, reviews ordered before verifies, MEM-priority first.
- [ ] `test_pending.py` passes.

**Verify:**
```bash
set -euo pipefail
/home/erik/projects/numbox/venv/bin/python /home/erik/projects/numbox/docs/reviews/2026-06-20-numbox/bin/test_pending.py
```
Expected: prints `OK` and exits 0.

**Steps:**

- [ ] **Step 1: Write the failing test `bin/test_pending.py`**

```python
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PENDING = HERE / "pending.py"
PY = "/home/erik/projects/numbox/venv/bin/python"


def run(base, *args):
    out = subprocess.check_output([PY, str(PENDING), "--base", str(base), *args], text=True)
    return out.strip()


def setup_tree(base):
    targets = {"targets": [
        {"id": "alpha", "files": ["x"], "dimensions": ["MEM", "COR"]},
        {"id": "beta", "files": ["y"], "dimensions": ["MEM"]},
    ]}
    (base / "targets.json").write_text(json.dumps(targets))
    for dim in ("MEM", "COR", "DES", "SEC", "TST"):
        (base / "findings" / dim).mkdir(parents=True, exist_ok=True)
        (base / "verified" / dim).mkdir(parents=True, exist_ok=True)


def main():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        setup_tree(base)
        summary = json.loads(run(base, "--summary"))
        assert summary["review_total"] == 3, summary
        assert summary["review_pending"] == 3, summary
        assert summary["verify_available"] == 0, summary
        assert summary["complete"] is False, summary

        (base / "findings" / "MEM" / "alpha.json").write_text("{}")
        summary = json.loads(run(base, "--summary"))
        assert summary["review_pending"] == 2, summary
        assert summary["verify_available"] == 1, summary

        nxt = run(base, "--next", "2").splitlines()
        assert len(nxt) == 2, nxt
        assert nxt[0].startswith("verify MEM alpha") or nxt[0].startswith("review MEM"), nxt
        print("OK")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `/home/erik/projects/numbox/venv/bin/python /home/erik/projects/numbox/docs/reviews/2026-06-20-numbox/bin/test_pending.py`
Expected: FAIL (FileNotFoundError / no such file `pending.py`).

- [ ] **Step 3: Write `bin/pending.py`**

```python
"""Compute the numbox-review work queue from disk. Disk is the source of truth."""
import argparse
import json
from pathlib import Path

DIM_ORDER = {"MEM": 0, "COR": 1, "SEC": 2, "DES": 3, "TST": 4}


def load_units(base):
    targets = json.loads((base / "targets.json").read_text())["targets"]
    review, verify = [], []
    for tgt in targets:
        for dim in tgt["dimensions"]:
            review.append((dim, tgt["id"]))
            verify.append((dim, tgt["id"]))
    return review, verify


def sort_key(item):
    _kind, dim, target = item
    return (DIM_ORDER.get(dim, 9), target)


def queue(base):
    review, verify = load_units(base)
    pending = []
    done_findings = set()
    for dim, target in review:
        if (base / "findings" / dim / f"{target}.json").exists():
            done_findings.add((dim, target))
        else:
            pending.append(("review", dim, target))
    for dim, target in verify:
        if (dim, target) in done_findings and not (base / "verified" / dim / f"{target}.json").exists():
            pending.append(("verify", dim, target))
    kind_rank = {"review": 0, "verify": 1}
    pending.sort(key=lambda it: (kind_rank[it[0]], sort_key(it)))
    return review, verify, done_findings, pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/erik/projects/numbox/docs/reviews/2026-06-20-numbox")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--next", type=int, default=0)
    args = ap.parse_args()
    base = Path(args.base)
    review, verify, done_findings, pending = queue(base)
    if args.summary:
        verified_done = sum(
            1 for dim, t in verify if (base / "verified" / dim / f"{t}.json").exists()
        )
        print(json.dumps({
            "review_total": len(review),
            "review_pending": sum(1 for it in pending if it[0] == "review"),
            "verify_available": sum(1 for it in pending if it[0] == "verify"),
            "verify_done": verified_done,
            "pending_total": len(pending),
            "complete": len(pending) == 0 and len(done_findings) == len(review),
        }))
        return 0
    rows = pending[: args.next] if args.next > 0 else pending
    for kind, dim, target in rows:
        print(f"{kind} {dim} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `/home/erik/projects/numbox/venv/bin/python /home/erik/projects/numbox/docs/reviews/2026-06-20-numbox/bin/test_pending.py`
Expected: PASS — prints `OK`.

- [ ] **Step 5: Sanity-check against the real (empty) tree**

```bash
set -euo pipefail
/home/erik/projects/numbox/venv/bin/python /home/erik/projects/numbox/docs/reviews/2026-06-20-numbox/bin/pending.py --summary
```
Expected: `{"review_total": 89, "review_pending": 89, "verify_available": 0, "verify_done": 0, "pending_total": 89, "complete": false}`.

- [ ] **Step 6: Commit**

```bash
set -euo pipefail
D=/home/erik/projects/numbox
git -C "$D" add docs/reviews/2026-06-20-numbox/bin
git -C "$D" commit -m "review(harness): pending-queue resume engine with test" --quiet
git -C "$D" log -1 --format='%h %s'
```

---

### Task 3: `RUNBOOK.md` — the paced driver procedure

**Goal:** Document the exact per-hour turn so any session (or a `/loop`) drives the review identically and resumes losslessly.

**Files:**
- Create: `docs/reviews/2026-06-20-numbox/RUNBOOK.md`

**Acceptance Criteria:**
- [ ] The runbook gives the exact commands for: scan→pending, per-agent prompt assembly, the 10-dispatch batch, commit, and the `ScheduleWakeup(3600s)` vs synthesize branch.
- [ ] It names the model per agent kind (review = Opus, verify = Sonnet) explicitly.
- [ ] It states the resume command that works from a cold session.

**Verify:** `test -s docs/reviews/2026-06-20-numbox/RUNBOOK.md && echo ok` → `ok`

**Steps:**

- [ ] **Step 1: Write `RUNBOOK.md`** with this content:

```markdown
# Driver runbook — numbox deep review (paced 10 agents/hour)

Disk is the source of truth. One "turn" = one hour. Repeat until complete.

## Each turn

1. Compute the queue:
   `/home/erik/projects/numbox/venv/bin/python docs/reviews/2026-06-20-numbox/bin/pending.py --summary`
   and `… --next 10`.
2. If `--summary` shows `complete: true` AND `verify_done == review_total` → go to Synthesis.
3. Otherwise take the up-to-10 lines from `--next 10`. For each line dispatch ONE agent (all in
   a single message so they run concurrently):
   - `review <DIM> <target>`: model **Opus**. Prompt = contents of `prompts/<DIM>.md`
     + "Files to review: <target's files from targets.json>" + "Write your findings JSON
     (conforming to findings.schema.json) to docs/reviews/2026-06-20-numbox/findings/<DIM>/<target>.json
     and write NOTHING else. If that file already exists, do nothing and stop."
   - `verify <DIM> <target>`: model **Sonnet** (adversarial floor; never Haiku). Prompt =
     contents of `prompts/verify.md` + "Findings to verify: <paste contents of
     findings/<DIM>/<target>.json>" + "Write verdicts JSON to
     docs/reviews/2026-06-20-numbox/verified/<DIM>/<target>.json and nothing else. If it exists, stop."
   - Every agent prompt MUST also carry the shared conventions: read-only on source, no test
     runs, no background tasks, `git -C` / venv-python / no-`cd`, no names / no AI provenance.
4. After all 10 return, commit the new files:
   `git -C /home/erik/projects/numbox add docs/reviews/2026-06-20-numbox/findings docs/reviews/2026-06-20-numbox/verified`
   `git -C /home/erik/projects/numbox commit -m "review(batch): <DIM/target list>" --quiet`
5. Re-run `pending.py --summary`. If `pending_total > 0` → `ScheduleWakeup(delaySeconds=3600,
   reason="next 10 review agents", prompt="<the loop prompt>")` and END the turn.
   If `pending_total == 0` but verify units remain available → continue (do not stop early).

## Counting
Every dispatched agent — review AND verify — counts toward the 10. Never exceed 10 per turn.

## Synthesis (when pending_total == 0 and complete)
Dispatch the synthesis agent(s) per `prompts/synthesis.md` (counts toward the 10/hour like any
batch). Produce REPORT.md + numbox-review.tasks.json, commit, and STOP scheduling wakeups.

## Cold resume (new session, even days later)
1. `git -C /home/erik/projects/numbox checkout review/numbox-2026-06-20`
2. Run `pending.py --summary`. Whatever is on disk is done; pick up at step 1 above.
No in-memory state is needed — the only authority is the findings/ and verified/ dirs.
```

- [ ] **Step 2: Commit**

```bash
set -euo pipefail
D=/home/erik/projects/numbox
git -C "$D" add docs/reviews/2026-06-20-numbox/RUNBOOK.md
git -C "$D" commit -m "review(harness): paced driver runbook" --quiet
git -C "$D" log -1 --format='%h %s'
```

---

### Task 4: Synthesis template (report + fix-plan generation)

**Goal:** Define how the final `REPORT.md` and `numbox-review.tasks.json` are produced from the verified findings, so the deliverable is reproducible from disk.

**Files:**
- Create: `docs/reviews/2026-06-20-numbox/prompts/synthesis.md`

**Acceptance Criteria:**
- [ ] The template specifies: read all `verified/*/*.json` + their `findings/*/*.json`, keep only `confirmed` findings, group by dimension then severity, MEM section first.
- [ ] It specifies the `numbox-review.tasks.json` shape: one task per confirmed finding or coherent cluster, each with `model`, `files`, `acceptanceCriteria`.
- [ ] `uncertain` verdicts are written to a separate "Human triage" section, not dropped silently.

**Verify:** `test -s docs/reviews/2026-06-20-numbox/prompts/synthesis.md && echo ok` → `ok`

**Steps:**

- [ ] **Step 1: Write `prompts/synthesis.md`**

```markdown
# Synthesis: build REPORT.md + numbox-review.tasks.json

Inputs (read all): docs/reviews/2026-06-20-numbox/verified/*/*.json and the matching
findings/*/*.json. Do NOT edit source. Write only the two output files below.

Keep a finding iff its verdict is "confirmed". Collect "uncertain" verdicts separately.
Drop "refuted".

## REPORT.md
- Title + date + one-paragraph summary: counts of confirmed findings by dimension and severity.
- Sections in this order: Memory/ABI/refcount (MEM), Correctness (COR), Security (SEC),
  Design/simplification (DES), Tests & docs (TST). Within each, high → medium → low severity.
- Each finding: `file:line` (clickable), the concrete failure, and the recommended fix.
- Final section "Human triage": every `uncertain` verdict with its reasoning.
- No person names; no AI-provenance statements.

## numbox-review.tasks.json
JSON: {"planPath": "docs/reviews/2026-06-20-numbox/REPORT.md", "tasks": [ ... ]}.
One task per confirmed finding, or per tight cluster of findings in the same file/area.
Each task: {"id", "subject", "status": "pending", "description" (Goal/Files/AC), "files",
"acceptanceCriteria", "model"}. Set "model": "opus" for memory/ABI/correctness fixes,
"sonnet" for mechanical design/doc/test fixes. Order tasks by severity (high first).
This file is a handoff for a LATER fix campaign; it changes no source now.
```

- [ ] **Step 2: Commit**

```bash
set -euo pipefail
D=/home/erik/projects/numbox
git -C "$D" add docs/reviews/2026-06-20-numbox/prompts/synthesis.md
git -C "$D" commit -m "review(harness): synthesis template for report and fix plan" --quiet
git -C "$D" log -1 --format='%h %s'
```

---

### Task 5: Run the paced review to completion

**Goal:** Execute the runbook — ~89 review + ~89 verify + synthesis ≈ 180 agents at 10/hour ≈ ~18 one-hour batches — producing the committed `REPORT.md` and `numbox-review.tasks.json`, with every batch durably committed.

**Files:**
- Create (during run): `docs/reviews/2026-06-20-numbox/findings/**`, `verified/**`, `REPORT.md`, `numbox-review.tasks.json`

**Acceptance Criteria:**
- [ ] After each batch, ≤10 new agents ran and their files are committed before the next wake.
- [ ] `pending.py --summary` eventually reports `complete: true` with `verify_done == review_total`.
- [ ] `REPORT.md` contains only `confirmed` findings (MEM section first) plus a Human-triage section for `uncertain`.
- [ ] `numbox-review.tasks.json` parses and every task has a `model` field.
- [ ] No source file outside `docs/reviews/2026-06-20-numbox/` was modified; nothing was pushed.

**Verify:**
```bash
set -euo pipefail
R=/home/erik/projects/numbox/docs/reviews/2026-06-20-numbox
/home/erik/projects/numbox/venv/bin/python "$R/bin/pending.py" --summary
/home/erik/projects/numbox/venv/bin/python -c "import json; d=json.load(open('$R/numbox-review.tasks.json')); assert all('model' in t for t in d['tasks']); print('tasks:', len(d['tasks']))"
git -C /home/erik/projects/numbox status --short
```
Expected: summary shows `complete: true`; tasks count prints; `git status` shows a clean tree (all committed); diff touches only `docs/reviews/2026-06-20-numbox/`.

**Steps:**

- [ ] **Step 1: Kick off batch 1.** Follow `RUNBOOK.md` "Each turn": run `pending.py --next 10`, dispatch the 10 agents (review = Opus, verify = Sonnet), wait for all, commit.
- [ ] **Step 2: Schedule the next turn.** If `pending.py --summary` shows `pending_total > 0`, call `ScheduleWakeup(delaySeconds=3600, …)` with the loop prompt, then end the turn. On wake, repeat Step 1. (If a usage limit delays the wake, the work resumes from disk whenever the session next runs — no loss.)
- [ ] **Step 3: Synthesis.** When `--summary` is `complete: true`, dispatch the synthesis agent per `prompts/synthesis.md`; commit `REPORT.md` + `numbox-review.tasks.json`; stop scheduling wakeups.
- [ ] **Step 4: Report done.** Show the user the REPORT.md summary counts and the tasks.json path. Do NOT push; ask before any push.

---

## Self-Review

**Spec coverage:** disk-as-truth (Tasks 0/2/3/5), 5-dimension × ~20-target matrix (Task 0), adversarial per-unit verify (Tasks 1/3), 10-agents/hour pacing via ScheduleWakeup (Tasks 3/5), per-batch commits (Tasks 0–5), report + tasks.json deliverable (Task 4/5), no push / fork workflow (conventions + Task 5 AC) — all covered. The spec's "~75 review units" is refined to 89 with a documented reason (MEM lens widened), flagged in Task 0.

**Placeholders:** none — every file's content is given in full.

**Type consistency:** `findings.schema.json` field names (`unit`, `dimension`, `target`, `findings[].id/severity/confidence/file/line/title/detail/recommendation`) match the MEM/COR/DES/SEC/TST prompts and `synthesis.md`. `pending.py` output verb tokens (`review`/`verify`) match the runbook's dispatch switch. Verdict statuses (`confirmed`/`refuted`/`uncertain`) match `verify.md` and `synthesis.md`. Dimension codes (`MEM/COR/DES/SEC/TST`) are identical across `targets.json`, prompts, `pending.py` `DIM_ORDER`, and directory names.
