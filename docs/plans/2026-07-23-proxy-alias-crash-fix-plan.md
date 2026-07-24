# Plan: fix the @proxy alias-rename cache crash via load-time validation (s3)

Fork issue [#73](https://github.com/nelson2005/numbox/issues/73). This work stream
emerged from the T6 cross-process regression net of the
[2026-07-09 cache-hash campaign](2026-07-09-cache-hash-fix-plan.md): building
the net surfaced that editing a `@proxy`'d function's body/signature/jit-flags
renames its process-stable cfunc alias, and a warm `cache=True` caller in another
file then loads a cached object referencing an alias this process never
registered — a bare `SIGSEGV` (empty stderr) or `LLVM ERROR: Symbol not found`
`SIGABRT`. It is a release blocker: the next release cut from this branch renames
all 173 shipped aliases at once, so every downstream user with a warm caller
crashes on upgrade.

Full investigation, mechanism, and the design comparison that selected this
approach are in the durable workspace
`~/.claude/projects/-home-erik/numbox-proxy-alias-crash-2026-07-22/`
(`REVIEW.md`, `DIAGNOSIS-SUMMARY.md`, `designs/s3-load-time-validation.md`, and
the three `verdicts/s3-*` adversarial reviews).

Branch: continue on `fix/cache-hash-2026-07-09` (the campaign vehicle; the related
jit-flags alias fix `f50fd87` already lives here). Do not start from `origin/main`.

## Why s3 (load-time validation), stated briefly

The diagnosis proved that **any in-IR guard is unreachable**: one unresolvable
external name makes RuntimeDyld zero *every* external relocation of the whole
object, so the process faults at the cpython wrapper's `PyArg_UnpackTuple` before
any guarded code runs. The fix therefore cannot live in generated code; it must
stop the poisoned object before it reaches the engine.

s3 does exactly that: it wraps numba's cache-load `rebuild` hook, and *before* the
serialized library touches the JIT engine it enumerates the object's undefined
`numbox_pxy_*` symbols and checks each against `ll.address_of_symbol`. A miss
returns `None` — numba's contract for a cache miss — so the stale entry is
discarded and recompiled in place, emitting a named `StaleProxyCacheWarning`.

s3 was chosen over the alternatives (see `REVIEW.md`) because it is the only
design that keeps numba's cache key **intrinsic to the artifact**: it introduces
no cache-key coupling to ambient process state, so — unlike the cache-key-injection
approach — it produces no spurious cache duplication for unchanged code and no
unbounded cache growth, while still self-healing the upgrade transition and
covering live local binding edits. Its price is the one thing this plan is mostly
about: it parses the object's symbol table, and that parser is currently ELF-only.

## The governing constraint: the new parsers can only be validated in CI

The proven, prototyped part of s3 is the **ELF** path, verified end-to-end on
Linux. macOS (Mach-O) and Windows (COFF) need their own undefined-symbol parsers,
and **neither can be exercised on the Linux dev box** — the object formats are
produced only on those platforms. Their real validation is the fork CI matrix
(`macos-latest`, `windows-latest`), where the two-process stale-heal test either
heals (parser correct) or crashes (parser wrong).

Two consequences shape the ordering:

1. Land and prove the **ELF core + full test + docs** first, get the Linux CI jobs
   green, so the platform-agnostic machinery is trusted before the format-specific
   parsers are added.
2. For local iteration on the Mach-O/COFF parsers, check in **small real binary
   fixtures** (one relocatable object per format carrying a `numbox_pxy_*`
   undefined symbol) so the parser has a Linux-runnable unit test — but treat the
   fork CI macOS/Windows jobs as the authoritative gate, and expect a loop back to
   Phase 2 if a job crashes.

## Success criterion (goal-driven)

The two baseline crash shapes, pinned as tests, must flip from crash to
`rc=0` + correct new-body result + a `StaleProxyCacheWarning`, on **all four CI
platforms** (ubuntu, ubuntu-arm, windows, macos), with a warm→edit→rerun
subprocess test — and a second, unedited caller in the same run must still
cache-hit with no warning (selectivity). The full suite stays green with the guard
live on every cache load.

## Phases

### Phase 0 — Harness and baseline pinning (write the failing test first)

- **T1.** Add a two-process stale-heal test module (mirror `_run_probe` /
  `tmp_path` / dedicated `NUMBA_CACHE_DIR` from `test/core/test_cache_crossprocess.py`).
  Process 1 compiles a `cache=True` cross-file caller of a `@proxy` binding and
  warms the cache; the binding body is edited; process 2 reruns against the same
  cache dir. Assert the *target* behaviour (heal + warning + correct new result),
  and — guarded by a marker so it does not fail the suite pre-fix — capture the
  current crash as the baseline. Include a second, unedited caller and assert it
  hits. This test is the verification target for every later phase and runs on the
  whole CI matrix. `verify:` the test file collects and the baseline assertions
  reproduce the crash on an unpatched tree.

### Phase 1 — ELF core (the proven path)

- **T2.** Port the prototyped s3 block into `numbox/core/proxy/proxy.py`: the
  `StaleProxyCacheWarning` class, `_stale_proxy_aliases`, `_guarded_rebuild`,
  `_install_cache_alias_guard`, and the import-time install. Add the two imports
  (`struct`, `warnings`). Structure the parser behind a single
  `_undefined_symbols(object_code)` dispatcher that sniffs the object magic and
  routes to a per-format parser; wire only the ELF parser now, with Mach-O and
  COFF as stubs returning `set()` (so the dispatcher shape is final and Phase 2 is
  pure fill-in). `verify:` T1's heal assertion passes on Linux; full suite green
  with the guard live.
- **T3.** Make the installer **fail-open** (portability verdict finding): wrap
  `_install_cache_alias_guard` so any exception during install degrades to
  "no guard installed" rather than breaking import — matching the fail-open
  philosophy the per-load validator already follows. `verify:` a forced install
  failure leaves `import numbox.core.proxy.proxy` working.
- **T4.** Regression tests for the ELF path: strict-mode escalation
  (`-W error` and the targeted `filterwarnings`), the no-alias fast path (a
  non-proxy `cache=True` function loads with zero validator cost and no warning),
  partial-staleness selectivity (two proxies, one edited — only the edited caller
  heals), and the `CodeLibraryCacheImpl` path via a `@guvectorize(cache=True)`
  caller of a proxy (the code-read-only path in the prototype). `verify:` all pass
  on Linux with the guard live; `test_proxy.py` green.

### Phase 2 — Cross-platform parsers (validated only in CI)

- **T5.** Mach-O undefined-symbol parser. Parse `LC_SYMTAB`, collect external
  undefined symbols (`n_type & N_STAB == 0`, `(n_type & N_TYPE) == N_UNDF`,
  `n_type & N_EXT`). **Strip the leading underscore** Mach-O prepends to C symbols
  *before* the `numbox_pxy_` prefix test and before the `ll.address_of_symbol`
  lookup — a naive ELF-shaped port checks for `numbox_pxy_` , finds only
  `_numbox_pxy_` , matches nothing, and silently passes every stale object
  (the trap the portability verdict flagged). Handle both 32- and 64-bit and both
  endiannesses via the magic (`0xFEEDFACE`/`0xFEEDFACF` and their byte-swaps),
  and fat/universal archives if numba ever emits them. Unit-test against a
  checked-in Mach-O fixture. `verify:` the fixture test extracts the aliased
  symbol; final proof is the fork CI `macos-latest` job.
- **T6.** COFF/PE undefined-symbol parser (Windows): symbol table at
  `f_symptr`, string table immediately after, external symbols with
  `n_scnum == IMAGE_SYM_UNDEFINED (0)` and `n_sclass == IMAGE_SYM_CLASS_EXTERNAL`,
  names either inline (8 bytes) or string-table offset. Note Windows x64 does not
  prefix a leading underscore on `__cdecl` x64 symbols, but confirm against a real
  object rather than assuming. Unit-test against a checked-in COFF fixture.
  `verify:` fixture test; final proof is the fork CI `windows-latest` job.
- **T7.** Guard-fires CI assertion test. On a deliberately seeded stale cache,
  assert a `StaleProxyCacheWarning` is actually raised (escalated to error). This
  is the one test that converts a silently-no-op parser (a broken Mach-O/COFF
  port, or a future numba payload-shape change) into a red job on every platform —
  the analogue of the campaign's "read the key back" guard for s4. `verify:` fails
  on a stubbed-empty parser, passes on a correct one.

### Phase 3 — Diagnostics and hardening

- **T8.** Debug/strict env knob (e.g. `NUMBOX_PROXY_CACHE_STRICT`): when set, make
  the fail-open paths loud (validator exceptions and payload-shape surprises raise
  instead of degrading silently) and escalate stale detections to a hard error.
  Document that strict mode aborts *before* the heal, preserving the stale entry
  for inspection (observed in the correctness verdict). `verify:` knob on → stale
  load raises the named error; knob off → heals silently.
- **T9.** Diagnostic completeness: ensure every stale alias in an object is named
  in the warning (the correctness verdict saw an undercount in a multi-entry
  scenario). `verify:` a multi-stale-entry scenario names all of them.

### Phase 4 — Documentation (mandatory)

- **T10.** Rewrite the migration guidance. Replace the `_stable_cfunc_alias`
  docstring's "clear the numba cache after such a change" and the incorrect
  `~/.cache/numba` remedy location in `docs/numbox.core.proxy.rst` with the new
  behaviour: self-healing on load, the `StaleProxyCacheWarning`, the strict-mode
  recipe, and a plain statement of the first-upgrade transition. `verify:`
  `sphinx-build -W` clean; the doc code-blocks pass the doc-codeblock flake8.

### Phase 5 — Full gate and ship

- **T11.** Full CI-equivalent local gate (Linux): `docs/plans/matrix_check.sh`
  across all 5 CPythons, `flake8 . --max-line-length=127`, `sphinx-build -W`, the
  doc-codeblock lint, and the new subprocess tests. `verify:` all green (the lone
  expected `test_numbox_and_python_use_same_libsqlite3` failure on uv interpreters
  is the documented sqlite-mismatch artifact, not a regression).
- **T12.** Push to the fork feature branch and drive the fork CI matrix to green.
  **This is where the Mach-O and COFF parsers get their real validation** — the
  `macos-latest` and `windows-latest` jobs run T1's stale-heal test. A crash there
  loops back to T5/T6. `verify:` all 27 `numbox_ci` jobs succeed; report the test
  count.
- **T13.** Adversarial review of the final diff (independent reviewer, correctness
  + portability lenses) before the PR: confirm no in-object stale reference can
  reach the engine on any format, the fail-open paths are safe, and the parsers
  reject a malformed object without throwing. `verify:` reviewer sign-off, findings
  addressed.
- **T14.** Update the campaign tracker and open the fork PR referencing #73 (fork
  PR first; upstream only with per-PR approval, per project rules). `verify:` PR
  green; #73 item linked.

## Risks and mitigations

- **Cross-platform parsers only fully validated in CI.** Mitigate with checked-in
  binary fixtures for local unit tests (T5/T6) and the guard-fires assertion (T7)
  so a silent parser no-op is a red job, not a latent crash. Budget a CI round-trip
  or two for parser iteration.
- **Mach-O leading-underscore trap.** Explicit handling + fixture test in T5; the
  guard-fires test (T7) on `macos-latest` is the backstop.
- **Private-API monkeypatch of `CacheImpl.rebuild`.** Verified stable across all 9
  releases in the `numba>=0.60,<0.67` pin; the pin bounds it, the validator and
  installer fail open, and T7 turns any future payload-shape drift into a red test.
- **Fail-open silent regression.** T7 (guard-fires) plus T8 (strict knob) are the
  two mitigations; without T7 a broken parser looks green.
- **Scope creep into the r4 silent-wrong-answer routes.** Those (the swallowed-trap
  return, the opaque-capture fingerprint collision) are orthogonal to the crash and
  out of scope for this plan; note them as follow-ups, do not fold them in.

## Findings from executing T1–T5 (recorded here because the analysis workspace is machine-local)

- **A resolvable alias does not imply a callable body.** `proxy_if_available`'s absent path registers a
  diagnostic trap under the alias, so `ll.address_of_symbol` resolves and the validator would pass the
  object. Reaching the trap is not a safe outcome: its `RuntimeError` is raised inside a `@cfunc`, numba
  swallows it at the C boundary and returns zero, and the caller computes on the zero and exits 0 — a
  silent wrong answer where a cold cache raises a clean `TypingError`. Trap-registered aliases are
  therefore tracked in `_ABSENT_ALIASES` and treated as stale.
- **numba's cached objects on Windows are ELF, not COFF.** LLVM's MCJIT rewrites a COFF target to ELF for
  JIT, so `ll.get_object_format('x86_64-pc-windows-msvc')` reports COFF while the JIT emits ELF. The
  Windows CI jobs pass with the COFF branch stubbed. Confirm with an object-magic assertion before
  writing a COFF parser.
- **A real Mach-O fixture can be produced on Linux** — `ll.initialize_all_targets()` then
  `Target.from_triple("<arch>-apple-darwin").create_target_machine().emit_object(...)` — so the reader has
  a compiler-produced test input on any host, with no binary checked in.
- **A malformed object can hang a reader, and fail-open cannot rescue a hang.** In Mach-O, `cmdsize` is
  what advances the load-command walk, so a zeroed one spins forever. Every reader added here must bound
  its loops on both the declared count and the buffer.
- **`.nbc` bytes are not deterministic** across processes for an identical body (same size, differing
  bytes observed), so object content is not a valid staleness check — only the cache key is.

## Out of scope (follow-ups, recorded not done)

- The four r4 silent-wrong-answer routes (separate `fingerprint.py` / cfunc-ABI
  work).
- The body-blind alias (design s2) as a later optimization that removes the
  recompile-on-body-edit cost; it composes on top of s3 but must not ship alone
  (it detonates the migration and reopens a silent-wrong-body hole).
