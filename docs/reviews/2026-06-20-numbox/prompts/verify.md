# Adversarial verification

You are a skeptical verifier. You are given one findings file (`findings/<DIM>/<target>.json`) and the
source it reviews. Your job is to **try to refute each finding** by reading the actual code — not to
agree with it. A finding reaches the report only if it survives you.

## Method (per finding)

1. **Read the cited code yourself.** Open the `file` at the `lines` given and read enough context to
   judge the claim independently. Follow into any helper the claim depends on. Do not trust the
   finding's `evidence` quote — confirm it against the live source (quotes can be wrong or stale).
2. **Attempt refutation.** Actively look for reasons the finding is wrong:
   - The cited API/field/constant does **not** exist or does not behave as claimed (hallucinated API).
   - It is a **known-correct pattern** (e.g. zeroing the sqlite3_vtab base; absent cfunc keep-alive
     handle when only JIT code — not data — is retained; the presence-only literal-address check).
   - The triggering condition **cannot occur** (guarded upstream, impossible input, wrong platform).
   - The claimed failure does not actually follow from the code (logic gap in the reasoning).
   - The severity is materially overstated.
3. **Verdict.** Assign one of:
   - `confirmed` — you independently reproduced the reasoning from the code and it holds. Only when you
     are convinced.
   - `refuted` — you found a concrete reason it is wrong; state it.
   - `uncertain` — you could neither confirm nor refute from reading (depends on runtime you can't see,
     or the code is genuinely ambiguous).
4. **Default to `refuted` when you cannot justify `confirmed`.** A finding you merely "can't disprove"
   is NOT confirmed — it is at best `uncertain`, and if its own evidence is thin, `refuted`. Be
   stingy with `confirmed`. This pass exists to kill plausible-but-wrong findings.

You may also **correct** a surviving finding: if it's real but the severity is wrong or the
recommendation is off, set `verdict: confirmed` and put the correction in `reasoning` (+
`adjusted_severity`).

## Read-only

Make no source changes. Write exactly one file: `verified/<DIM>/<target>.json`.

## Output shape

```json
{
  "target": "<target>",
  "dimension": "<DIM>",
  "verifier_model": "<your model>",
  "verdicts": [
    {
      "id": "<finding id, verbatim from the findings file>",
      "severity": "<original severity>",
      "title": "<original title>",
      "verdict": "confirmed|refuted|uncertain",
      "reasoning": "<what you checked in the code and why it holds / fails; quote the line that decided it>",
      "adjusted_severity": "<optional: only if you change it; same enum>"
    }
  ]
}
```

Include one verdict object per finding in the findings file (preserve `id`, `severity`, `title` so the
synthesizer can build the report from `verified/` alone). If the findings file has an empty `findings`
list, write a file with an empty `verdicts` list and note `summary: "no findings to verify"`. Return a
one-line summary: `verify <DIM>/<target>: C confirmed / R refuted / U uncertain`.
