# Caching / hashing fix campaign (2026-07-09)

Fix the confirmed caching/hashing hazards in numbox's content-addressed cache
keys. Authoritative problem inventory: **[fork issue #73](https://github.com/nelson2005/numbox/issues/73)**
(23 confirmed — 8 high / 7 medium / 8 low — each adversarially re-verified, many
reproduced live in a two-process or `fork()` + shared-`NUMBA_CACHE_DIR`
experiment). This plan implements the six ranked fix directions from that review.

Frame: [numba discourse t/3260](https://numba.discourse.group/t/why-does-compile-results-mangled-llvm-symbol-name-contain-execution-specific-counter-generated-id/3260)
and the linked [numba/numba#10486](https://github.com/numba/numba/issues/10486).

## The pattern

numba names compiled symbols with a process-local uid (`v<N>`, an
`itertools.count`) that is unstable across processes and duplicated under
`os.fork()`. numbox keeps that uid out of everything it persists and instead
bakes a content hash of the user's code into generated names and cache keys —
which is exactly numba#10486's own proposed fix, and `compile_kernel`
implements it correctly via the deep walker in
[`numbox/utils/fingerprint.py`](../../numbox/utils/fingerprint.py) (bytecode +
consts + defaults + closure-cell values + referenced-global values, recursive).

**The safety is exactly the depth of the baked key, and numbox applies it
inconsistently.** Where the key is shallow — the proxy alias
(`module + qualname + signature`, no body content); `make_graph` derive
functions (`sha256(getsource)`, source text only); `make_structref` methods and
the sqlite UDAF/TVF digests (source text only); kernel `jit_options` (hashed as
the literal string `@njit(**jit_options)`) — two different compiled bodies
collide under one key. Because numba's cache then *hits* and links the wrong
body, #10486's *crash* becomes a *silent wrong result*: correct-looking machine
code returning wrong values for every input, exit 0, no diagnostic. The whole
campaign closes the gap between shallow keys and the machine code they name.

The through-line of almost every fix: route the shallow keys through the deep
walker that already protects `compile_kernel`, fold in the few missing pieces
(resolved jit flags, module-attribute values, a real `__version__`), and make
the one case that still segfaults (a cached caller referencing an alias this
process never registered) fail loudly instead.

## Toolchain (this checkout)

- venv: `/home/erik/projects/numbox/venv/bin/python` — Python 3.12.3, numba
  0.65.1, numpy 2.4.6, llvmlite 0.47.0 (the pin is `numba>=0.60.0,<0.67.0`).
- Fork CI lint: `flake8 . --count --show-source --statistics` (repo `.flake8`,
  `max-line-length = 127`, default rules).
- Fork CI test: `pytest --durations=20 --cov=numbox --cov-report=term-missing`.
- **Versioning (relevant to T3):** CI and `release.yml` overwrite
  `numbox/__init__.py` with `__version__ = '<VERSION>'` before install/build, so
  the checked-in `__version__ = ""` is only inert for dev/editable installs. Any
  `__version__` fix must survive that overwrite (derive from installed metadata,
  and/or fold a numbox-source fingerprint into the digest rather than relying on
  the string).

## Gates (every implementation task)

Per-task, before its commit — caches cleared first so a stale numba cache can't
mask a key change:

1. Clear caches: remove every `__pycache__` (numba's `.nbi`/`.nbc` live inside
   them) via the venv python — never `find -exec rm`.
2. `pytest --durations=20` — the task's targeted tests, then the full suite
   before the task is marked done.
3. `flake8 .` clean (repo `.flake8`, 127, default rules — matches fork CI).
4. Docs build clean when the task changes public surface or docs:
   `cd docs && sphinx-build -b html . _build/html` exit 0 (clear numba cache
   first).
5. `doc-codeblock-flake8` + `link-check` (lychee) when the task touches
   `.md`/`.rst`/URLs.

## Tasks

Severity-ranked (the review's fix-direction order). Each fix task gets an
adversarial verification pass at the Sonnet floor after implementation. Fix
directions 1–4 cover all 8 highs; 5 covers the cache-leak mediums; 6 is the
regression net.

| Task | Fix direction | Closes (#73) | Files |
|------|---------------|--------------|-------|
| T0 | Pre-flight: baseline green | — | read-only |
| T1 | Proxy alias: fold body fingerprint; loud failure for unresolvable aliases | H1, H2, M9, M10 | `core/proxy/proxy.py` |
| T2 | Route builder derive, `make_structref`, `digest()` fallback, sqlite UDAF/TVF through the deep walker; degrade to uncached on `_Unfingerprintable`; enrich `_canon_value` | H3, H4, H7, H8, L19, L21, L22, L23, M11 | `core/work/builder.py`, `utils/highlevel.py`, `utils/digest.py`, `utils/fingerprint.py`, `core/bindings/sqlite/udf_helpers.py`, `core/bindings/sqlite/tvf.py`, `utils/preprocessing.py` |
| T3 | Fold resolved `jit_options` + a real `__version__` into `_kernel_fingerprint` and `digest()` | H5, M14 | `core/work/builder.py`, `utils/digest.py`, `__init__.py` |
| T4 | Walk module-attribute reads in `_canon_value` (else mark unfingerprintable) | H6 | `utils/fingerprint.py` |
| T5 | Dispatcher fingerprint via `prune_type`/`py_func`; `_tvf_xfilter` `cache=False`; digest folds Dispatcher `targetoptions` | M12, M13, M15, L18 | `utils/highlevel.py`, `core/work/builder.py`, `core/bindings/sqlite/tvf.py`, `utils/digest.py` |
| T6 | Cross-process / `fork()` + shared-cache regression tests, seeded from the review's repro scripts | L16, L17 (+ guards all) | `test/` |
| T7 | Full CI-equivalent gate; PR body referencing #73 items; open fork PR | — | — |

### T1 — proxy alias

`_stable_cfunc_alias` keys `module + qualname + str(main_sig)` with no body
content, and `add_symbol` is last-writer-wins, so same-identity different-body
proxies (factory-made same-qualname closures, module re-exec, `fork()` twins
sharing a cache dir) silently reach the wrong body (H2, M9). Fold
`_fingerprint_function(func)` into the alias's raw key so distinct bodies get
distinct aliases. Separately, a `cache=True` caller that inlined an
`inline='always'` proxy cache-hits in a process that never registered the callee
alias (a `proxy_if_available` availability flip, or a signature change that
renamed the alias) and links against an unregistered symbol → bare SIGSEGV
(H1, M10). Make that fail loudly — validate alias resolution at decoration/load,
or register a trapping stub for the absent alias.

### T2 — route shallow keys through the deep walker

- `builder.py` derive hashing: replace `sha256(getsource(derive))` with
  `_fingerprint_function(derive)`; on `_Unfingerprintable` or `OSError` from
  `getsource` (exec/REPL-defined derives, L19) degrade to uncached like
  `compile_kernel`, never crash.
- `make_structref` (`highlevel.py`): replace the method source hash + anchor
  with the walker over the method functions so `ns`-threaded values enter the
  key (H4).
- `digest()` fallback (`digest.py`): the fallback pickles the bare `__code__`,
  dropping the closure/default state that forced the fallback (H7) and leaking
  iteration order for str set/frozenset consts (M11, PYTHONHASHSEED-dependent)
  and a per-process uuid for `__main__` callables (L23). Cloudpickle the
  function *object* (captures bound state), sort where order leaks.
- sqlite UDAF/TVF (`udf_helpers.py`, `tvf.py`): teach the walker the
  proxy-wrapper shape (stable identities) so realistic callbacks calling the
  `@proxy`'d `sqlite3_value_*`/`sqlite3_result_*` wrappers fingerprint cleanly
  instead of degrading to the H7 fallback (H8).
- `_canon_value` enrichment (`fingerprint.py`): structured-dtype ndarrays
  canonicalize by full layout, not `dtype.str` = `|Vn` (L21); StructRef digest
  identity includes module + attached `@overload_method` behavior, not just
  `repr(state_type)` (L22).

### T3 — jit flags + version

`_kernel_fingerprint` hashes the literal text `@njit(**jit_options)`; fold
`repr(sorted(resolved_jit_options.items()))` so a kernel built under one
`error_model`/`fastmath` is not reused under another (H5). Make the digest's
version fold real (M14): derive `__version__` from installed package metadata
with a fallback, coordinating with the CI/release `__init__.py` overwrite, and
fold numbox-source-dependent generated-code content so a numbox upgrade
invalidates stale registrations.

### T4 — module-attribute reads

A formula reading `cfg.SCALE` keeps its fingerprint when `SCALE` changes because
`_canon_value` canonicalizes a referenced module as `module(<name>)` without
descending (H6). Pair `co_names` attribute reads against the module globals in
`_canon_value`/`_fingerprint_function`, or conservatively mark
module-attribute-reading formulas unfingerprintable (→ uncached via the existing
degrade path).

### T5 — dispatcher / tvf cache leaks

`types.Dispatcher.name` embeds an ASLR address, so `hash_type` at
`builder.py:180` differs every process and `_make_<hash>` never cache-hits,
accreting one orphan cache pair per run (M13, L18); fingerprint via
`prune_type`/`py_func` instead of the type string. `_tvf_xfilter`
`@cfunc(cache=True)` never hits (per-process dispatcher uuid in its index key)
and appends an entry per registration per process (M15) → `cache=False`.
`digest()`'s `py_func` shortcut bypasses `_canon_value`'s dispatcher branch and
drops `targetoptions` (M12) → route top-level Dispatchers through the branch
that folds them.

### T6 — regression net

Adapt the review's repro scripts into portable pytest tests: the proxy hazard
set (body edit, availability flip, signature change, same-identity redefinition,
multi-sig cached callers — L16) and the digest fallback path (L17), plus a
`fork()` + shared-cache test mirroring #10486. Skip `fork`-specific cases on
Windows; use `subprocess` where a second interpreter suffices. Written to fail on
the pre-fix code and pass after T1–T5.

## Workflow

One fork feature branch (`fix/cache-hash-2026-07-09`, off fresh `origin/main`)
as the review vehicle → one fork PR (bots review). Upstream contributions staged
into small PRs later, each with explicit per-PR approval. The PR body references
the #73 item numbers and closes #73.

## Out of scope

The two incidentals the review found (exec-created formulas whose `__name__`
is not in `sys.modules` crashing a fresh process's reload; same-file-rewrite
codegen defeating numba's mtime source stamp) are noted as follow-ups, not fixed
here.
