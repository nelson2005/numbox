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
1. **Header** — date, branch, commit under review (capture the FULL SHA — the line links below pin to
   it), unit counts (targets, review units, verify units), and a tally: confirmed by severity,
   uncertain, refuted.
2. **Memory / ABI / refcount (MEM)** FIRST — this is the highest-value dimension. Then Correctness,
   Security, Design, Tests & docs. Within each dimension, order by severity (critical → low).
3. Each entry: a stable heading, the file:line reference, the concrete defect + when it triggers, the
   evidence, the recommended fix, confidence, and the contributing finding id(s). Every file:line MUST
   be a clickable GitHub blob link (see "Hyperlinking file references" below) — plain `file:line` in
   backticks is NOT clickable on GitHub. Be precise and terse — no filler, no "worth reviewer
   attention" editorializing.
4. **Human triage** — the `uncertain` findings, each with why the verifier couldn't decide and what a
   human should check.
5. **Coverage note** — confirm all 119 review units + their verifies are present on disk; if any unit
   is missing its file, list it (so the reader knows coverage was complete or where it wasn't). Never
   imply full coverage if a unit is absent.

### Hyperlinking file references

Every file:line in REPORT.md must be a clickable link to the exact source on GitHub, **pinned to the
commit under review** (an immutable FULL SHA, so the line numbers stay valid even after the branch
moves). The reviewed source equals the tree at that commit (the review adds only `docs/`, no source
changes).

- **Blob base** — derive it, don't hardcode: `git -C <repo> remote get-url origin` (normalize
  `git@github.com:OWNER/REPO.git` or `https://github.com/OWNER/REPO.git` → `https://github.com/OWNER/REPO`)
  and `git -C <repo> rev-parse HEAD` for the full SHA. Base = `https://github.com/OWNER/REPO/blob/<full-sha>/`.
- **Format** — link text is the reference inside a code span: `` [`<path>:<lines>`](<base><path>#L<start>[-L<end>]) ``.
  Example: `` [`numbox/utils/lowlevel.py:249-260`](https://github.com/OWNER/REPO/blob/<sha>/numbox/utils/lowlevel.py#L249-L260) ``.
- **Resolve bare/partial paths** (`abi.py`, `bindings/utils.py`) to the full repo path before building
  the URL (unique-basename or unique-suffix match against the tree); keep the visible link text as written.
- **Compound ranges** (`abi.py:95-96, 120-126`) — anchor the FIRST range (`#L95-L96`); keep the full
  text visible.

**Reliable method (recommended over hand-writing URLs):** writing dozens of correct links by hand is
error-prone. Write REPORT.md first with plain backtick'd `` `path:line` `` refs, then run a small
deterministic post-pass that regex-replaces each `` `path:line` `` token with its blob link, and
**assert an integrity check**: stripping the just-added links must reproduce the pre-link file
byte-for-byte (proves only links were added, no finding text altered). Overwrite REPORT.md with the
linked version and spot-check a few URLs return HTTP 200 at the pinned SHA.

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
