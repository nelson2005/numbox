# Synthesis — REPORT.md + numbox-review.tasks.json

You assemble the final deliverables **from disk** (idempotent: re-running reproduces them). Inputs:
- `targets.json` — the unit matrix.
- `findings/<DIM>/<target>.json` — full finding detail (claim/evidence/recommendation/confidence).
- `verified/<DIM>/<target>.json` — per-finding verdicts (confirmed/refuted/uncertain + reasoning).

## Join & filter

For every verified file, load the matching findings file and **join verdicts to findings by `id`**.
Then:
- **Keep `confirmed`** findings (apply any `adjusted_severity` from the verdict).
- **Route `uncertain`** findings to a "Human triage" section — never silently drop them.
- **Drop `refuted`** findings from the body, but keep a count (report N refuted in a one-line tally so
  the reader knows the verifier was working).

## Deduplicate / cluster

The same underlying defect may surface under multiple dimensions (e.g. a NUL-truncation bug flagged by
MEM, SEC, and COR). **Cluster** findings that name the same root cause + location into a single entry,
citing all contributing finding ids and the union of their recommendations. Do not double-count
clusters in severity tallies.

## REPORT.md

Markdown, reader-optimized, in this order:
1. **Header** — date, branch, commit SHA under review, unit counts (targets, review units, verify
   units), and a tally: confirmed by severity, uncertain, refuted.
2. **Memory / ABI / refcount (MEM)** FIRST — this is the highest-value dimension. Then Correctness,
   Security, Design, Tests & docs. Within each dimension, order by severity (critical → low).
3. Each entry: a stable heading, `file:line`, the concrete defect + when it triggers, the evidence,
   the recommended fix, confidence, and the contributing finding id(s). Link file references as
   `file:line`. Be precise and terse — no filler, no "worth reviewer attention" editorializing.
4. **Human triage** — the `uncertain` findings, each with why the verifier couldn't decide and what a
   human should check.
5. **Coverage note** — confirm all 119 review units + their verifies are present on disk; if any unit
   is missing its file, list it (so the reader knows coverage was complete or where it wasn't). Never
   imply full coverage if a unit is absent.

## numbox-review.tasks.json

A prioritized fix plan — the handoff artifact for a later, separate fix campaign (this review makes
**no** source changes). Shape:

```json
{
  "planFor": "numbox deep review 2026-06-20",
  "sourceReport": "docs/reviews/2026-06-20-numbox/REPORT.md",
  "lastUpdated": "2026-06-20-numbox-review",
  "tasks": [
    {
      "id": 1,
      "subject": "<imperative fix title>",
      "severity": "critical|high|medium|low",
      "dimension": "MEM|COR|SEC|DES|TST",
      "files": ["numbox/.../x.py"],
      "findingIds": ["MEM-...-1", "SEC-...-2"],
      "description": "<what's wrong and the fix approach>",
      "acceptanceCriteria": ["<verifiable>", "<e.g. add a test that fails before / passes after>"],
      "model": "opus",
      "reasoning": "max"
    }
  ]
}
```

Rules:
- **One task per confirmed cluster**, ordered by severity (critical first), MEM-weighted within ties.
- **Every task has a `model`.** Default `"opus"` with `"reasoning": "max"` for any correctness /
  memory / ABI / security fix. `"sonnet"` is allowed ONLY for purely mechanical tasks (doc text,
  dead-code deletion, a docstring fix) — **never Haiku** for any task. When in doubt, `"opus"`.
- Acceptance criteria must be **verifiable** and, for any behavioral fix, require a regression test
  that fails before and passes after.
- Do not invent fixes for refuted findings. Uncertain findings do NOT become tasks (they're triage);
  if a human later confirms one, it becomes a task then.

## Write & return

Write `REPORT.md` and `numbox-review.tasks.json` to the review root. Make no source changes. Return a
one-line summary: `synthesis: K tasks, <crit>/<high>/<med>/<low>, U uncertain, R refuted`.
