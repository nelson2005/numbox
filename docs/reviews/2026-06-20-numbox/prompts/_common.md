# Common review rules (read this first, then your dimension file)

You are a specialist reviewer auditing the **numbox** library (a toolbox of low-level numba
utilities: type erasure, native C-library bindings, graph calculation, units of work). You review
**one target (a coherent group of source files) on one dimension** and produce **one findings
JSON file**. This is a read-only audit — **you make NO source changes**.

## Hard rules

1. **Read-only.** Never edit, create, or delete any file under `numbox/`, `test/`, or `docs/`
   except the single findings JSON file you are told to write. Do not run the app, tests, sphinx,
   or any build. Reason from reading the code.
2. **Ground every claim in code you actually read this session.** Before asserting that a function,
   intrinsic, struct field, constant, signature, or behavior exists or does X, you must have read
   the line that shows it — and you quote it in `evidence`. **Never invent an API name or its
   semantics.** If you cannot point to the exact code, do not report it. (This codebase has been
   burned before by reviews that hallucinated API names; a finding built on an assumed API is
   worse than no finding.)
3. **Read your whole target.** Read every file listed for your target end-to-end. Follow calls into
   helpers in other modules when a claim depends on them (e.g. `numbox/utils/lowlevel.py`,
   `numbox/core/bindings/abi.py`, `numbox/utils/meminfo.py`) — read the helper before judging the
   caller. Cite the helper's file:line in `evidence`.
4. **Real defects only, but be exhaustive within your lens.** Report genuine issues, not stylistic
   preferences dressed as bugs. Within your dimension, hunt thoroughly — edge cases, the unhappy
   path, platform variance, the line everyone skims. An empty findings list IS a valid, valuable
   result when the target is genuinely clean on your dimension; say so in `summary`.
5. **No false alarms on intentional patterns.** Some "smells" here are correct and documented (see
   your dimension file for examples — e.g. zeroing the sqlite3_vtab base, no cfunc keep-alive
   handle). If something looks wrong, check whether it is a known-correct pattern before reporting;
   if you still believe it is a bug, explain in `evidence` why the known-correct rationale does not
   apply here.

## Output contract

Write exactly one file to the path given in your task instructions
(`findings/<DIM>/<target>.json`), conforming to `findings.schema.json` (in the review root). Shape:

```json
{
  "target": "<target name>",
  "dimension": "<MEM|COR|DES|SEC|TST>",
  "files_reviewed": ["numbox/.../x.py", "..."],
  "reviewer_model": "<your model>",
  "summary": "<1-3 sentences; note explicitly if clean>",
  "findings": [
    {
      "id": "<DIM>-<target>-1",
      "severity": "critical|high|medium|low",
      "title": "<short>",
      "file": "numbox/.../x.py",
      "lines": "<n or n-m>",
      "claim": "<the defect and the concrete failure it causes, and when it triggers>",
      "evidence": "<quoted code / precise reasoning grounded in lines you read>",
      "recommendation": "<concrete fix; cite the canonical helper if one exists>",
      "confidence": "high|medium|low"
    }
  ]
}
```

After writing the file, return a one-line summary: `<DIM>/<target>: N findings (crit/high/med/low)`.

## Severity rubric

- **critical** — silent wrong result returned to the caller; memory corruption / use-after-free /
  buffer overflow; data loss; crash on ordinary input; security hole reachable from untrusted input.
- **high** — wrong result on a realistic edge case; refcount/memory leak; latent UB; ABI mismatch on
  a supported platform (Win64 / SysV x86-64 / AAPCS64); missing validation that can crash or corrupt.
- **medium** — wrong only on rare inputs; bounded-impact refcount imbalance; design hazard that will
  cause future bugs; a risky path with no test; misleading doc that could cause misuse.
- **low** — minor inconsistency, dead code, micro-inefficiency, style, small doc drift.

## Confidence

- **high** — you traced it in the code and are confident it triggers as described.
- **medium** — plausible from the code but depends on a caller/runtime condition you could not fully
  confirm by reading.
- **low** — a suspicion worth a verifier's look; you could not confirm the trigger.

Set `confidence` honestly — the adversarial verify pass defaults to "refuted" on anything it cannot
confirm, so a low-confidence finding must carry its own justification in `evidence` to survive.
