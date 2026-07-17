# compile_kernel cache-correctness remediation (2026-07-17)

Continuation of the #73 cache/hash campaign, on `fix/cache-hash-2026-07-09`.
Triggered by an adversarial Fable audit of `compile_kernel`'s jit-flag folding
(before mirroring that pattern into `builder.py` for #73/H5). The audit found
`compile_kernel` — the campaign's "reference" implementation — is **not**
cache-correct end-to-end: 14 defects reproduced via a two-process shared-cache
protocol, 0 refuted.

Evidence: `~/.claude/projects/-home-erik/numbox-compile-kernel-audit-2026-07-17/`
(`CONFIRMED-FINDINGS.md`, `audit-raw.json`). Audit workflow `wf_79c1d680-540`.

## Root cause

numba's on-disk cache `_index_key = (sig, magic_tuple, sha256(co_code),
sha256(cvarbytes))` (`caching.py:781`) encodes **no jit flags and no global
values**. `compile_kernel` folds `_effective_flags` + formula fingerprints into
the **outer** `_kernel_<digest>` name, which correctly re-keys the outer — but
that name-fold cannot invalidate any independently numba-cached unit the kernel
links against (a user `@njit(cache=True)` Dispatcher/DUFunc), nor does it cover
values the fingerprint itself misses (module-attribute reads, env codegen knobs).

Proven safe: a **plain-Python** formula is `njit`-wrapped with `cache` stripped
(`utils.py:_wrap_formula` + `_effective_flags`), so it has no independent
on-disk entry and re-keys with the flag-folded outer. That path is the model the
`builder.py`/H5 anchor-wrapper fix will copy.

## Decisions

- **CK1 = conservative**: detect a formula carrying its own cache (Dispatcher/
  DUFunc with `cache=True`, or CFunc) → mark the unit uncacheable + warn +
  document. No re-wrap, no semantic change. numbox stops *compounding* numba's
  flag/global-blind inner cache (kills the permanent-poisoning escalation) and is
  honest that it cannot cure a user's own dispatcher cache.
- **builder/H5 (later, T3)**: anchor-wrapper (digest-named `_derive_<digest>`
  wrapper folding derive fingerprint + effective flags; raw derive inlined
  cache-stripped). Not part of this remediation.
- Branch: `fix/cache-hash-2026-07-09` (CK3 *is* #73 H6/T4).

## Defects → fixes

| Task | Defect (findings) | Sev | Fix |
|---|---|---|---|
| CK1 | Pre-jitted `cache=True` inner formula: flag/global-blind inner cache served stale, linked + serialized into fresh outer = permanent poisoning (1,2,8,9,13) | HIGH | detect self-cached formula → unit uncacheable + warn + doc |
| CK2 | DUFunc targetoptions omitted from `_formula_fingerprint` → digest collision (3,4,14) | HIGH | fold `dict(formula.targetoptions or {})` in the DUFunc branch |
| CK3 | Module-attribute reads (`cfg.SCALE`) fingerprint-blind (12) = H6/T4 | HIGH | fold data-valued `module.ATTR` reads into `_fingerprint_function` |
| CK4 | `NUMBA_BOUNDSCHECK` env knob uncovered (7) | MED | fold `numba.core.config.BOUNDSCHECK` into `hash_text` |
| CK5 | `_validated_returns` memo key omits flags (11) | MED | add canonical flags to the memo key |
| CK6 | Un-canonicalizable flags silently disable caching (5) | LOW | warn on the degrade path |
| CK7 | Docstring "a stale binary is never reused" overclaims (6,10) | LOW | scope the guarantee honestly |
| CK8 | Adversarial re-audit against the fixed tree | — | re-run `wf_79c1d680-540`; expect HIGH/MED refuted |
| CK9 | Full CI-equivalent gate | — | cache-cleared pytest+cov, flake8, sphinx, doc-codeblock, lychee |

Each fix: implement → regression test that fails pre-fix / passes post-fix
(two-process where the defect is cross-process) → adversarial verify.

## Correctness ceiling (documented, not a bug)

For a user's own `@njit(cache=True)` dispatcher whose flags/globals vary across
processes, numba serves stale regardless of numbox (bare-numba has the identical
hole). CK1 prevents numbox from compounding it into a poisoned content-addressed
artifact and warns; it cannot cure the user's dispatcher cache. The docstring
(CK7) scopes the guarantee to units numbox itself generates and njit-wraps from
raw Python, and excludes env-level codegen knobs not folded.
