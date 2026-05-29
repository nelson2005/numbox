# SQLite UDAF Registration Helper — Design

**Date:** 2026-05-29
**Status:** Design approved; mechanism re-reviewed in depth across numbox/numbduck/CRE and verified by spikes S1–S4 (B selected, 2026-05-29); pending spec review → implementation plan
**Builds on:** Phase 2 SQLite UDF/UDAF/window bindings (currently on `feat/sqlite-udf`; upstream PR #17 BLOCKED)

## Goal

Provide a thin, lifecycle-only convenience layer that registers structref-backed
**aggregate** and **window** user-defined functions from `@njit` code, so users
no longer hand-write the error-prone state lifecycle that phase 2 requires today.

The helper eliminates the per-callback boilerplate — `sqlite3_aggregate_context`
allocation + NULL/OOM guard, the single-`intp` slot, init-on-first-step,
`borrow_structref`, and the **release-in-`xFinal`-but-not-`xValue`** rule — and the
`cfunc` wrapping + keep-alive + registration call. Users write only their own
state logic (`init` / `step` / `finalize`, plus `inverse` / `value` for windows).

### Non-goals (out of scope for this work item)

- **Scalar UDFs** — stateless, so there is no lifecycle to manage.
- **Typed argument/result marshalling** — the user still reads SQL values
  (`sqlite3_value_*`) and writes results (`sqlite3_result_*`) themselves. A typed
  layer that generates extraction/result-setting from declared types is a possible
  future enhancement layered on top, explicitly deferred.
- **`set_auxdata`/`get_auxdata`** and **`Connection`/`Statement` wrappers** — the
  other two sibling work items, each their own spec/PR.

## Key decisions

| Decision | Choice |
|----------|--------|
| Abstraction level | **Thin (lifecycle-only)** — user does their own value reads / result writes |
| Scope | **Aggregate + window** (window's release-in-`xFinal`-only rule is the highest-value case) |
| Entry-point shape | **Two functions** returning a **keep-alive handle** |
| Wrapper mechanism | **B — content-addressed anchored codegen** (selected by spike 2026-05-29). C (shared lifecycle + dispatcher args) rejected: no stable cross-process cache |
| Caching | **Yes** — consistent with every phase-1/phase-2 binding |
| Empty groups | Helper finalizes a fresh `init()` state; user's `finalize`/`value` always receives a valid state |

## Public API

Two Python-level setup functions (run at registration time, before queries), each
returning a keep-alive handle:

```python
register_aggregate(db, name, n_arg, state_type, init, step, finalize,
                   *, deterministic=False) -> handle
register_window(db, name, n_arg, state_type, init, step, inverse, value, finalize,
                *, deterministic=False) -> handle
```

- `db` — connection pointer (`intp`), as returned by `sqlite3_open`.
- `name` — Python `str`; the helper owns the C-string (`c_string`) lifetime.
- `n_arg` — argument count; `-1` for variadic.
- `state_type` — the numba structref **instance** type
  (e.g. `sum_state_type = SumStateType([("total", int64)])`).
- `deterministic` — OR-in `SQLITE_DETERMINISTIC`; text encoding fixed to `SQLITE_UTF8`.
- **Returns** a handle the caller **must retain**. The handle owns the generated
  `cfunc`s and references to the user functions; dropping it frees the callback
  pointers SQLite holds → segfault. On a non-`SQLITE_OK` rc, the helper **raises**
  (this is the ergonomic layer; the low-level bindings still return rc directly for
  callers who want it).

## User contract (thin / lifecycle-only)

The user supplies `@njit` functions that touch only SQL values/results and their
own state — never `aggregate_context`, `export_meminfo`, `borrow_structref`, or
`release_meminfo`:

```python
def init():                                 # fresh state for a new group
    return SumState(0)

def step(state, ctx, argc, argv_pp):        # mutate state from the row's args
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total += sqlite3_value_int64(args[0])

def finalize(state, ctx):                    # read state, set the result
    sqlite3_result_int64(ctx, state.total)
```

Window adds two more of the same shape:

- `inverse(state, ctx, argc, argv_pp)` — un-applies a row. The helper guarantees the
  state already exists (no init-on-first-step here).
- `value(state, ctx)` — emit the running result **without** finalizing or releasing.

`ctx` is passed to `step`/`inverse` as well (for `sqlite3_user_data` etc.), even
though aggregates usually ignore it — uniform and flexible.

### What the helper owns

For each callback SQLite invokes, the helper generates the lifecycle (generalized
verbatim from the phase-2 hand-written test code). Every callback guards the NULL
`aggregate_context` **before** indexing the slot — `carray` on a NULL pointer
segfaults, and for an empty group `aggregate_context(ctx, 0)` returns NULL (no prior
allocation), so the NULL check *is* the empty-group signal:

- **`xStep`**: `agg = aggregate_context(ctx, 8)`; if `agg == 0` return (OOM);
  `slot = carray(agg, (1,), intp)`; if `slot[0] == 0`: `slot[0] = export_meminfo(init())`;
  `state = borrow_structref(state_type, slot[0])`; `step(state, ctx, argc, argv_pp)`.
- **`xInverse`** (window): `agg = aggregate_context(ctx, 0)`; if `agg == 0` return;
  `slot = carray(agg, (1,), intp)`; if `slot[0] == 0` return;
  `inverse(borrow_structref(state_type, slot[0]), ctx, argc, argv_pp)`.
- **`xValue`** (window): `agg = aggregate_context(ctx, 0)`; if `agg == 0` ⇒
  `value(init(), ctx)`, return; `slot = carray(agg, (1,), intp)`; if `slot[0] == 0` ⇒
  `value(init(), ctx)`, return; else `value(borrow_structref(state_type, slot[0]), ctx)`
  — **no release**.
- **`xFinal`**: `agg = aggregate_context(ctx, 0)`; if `agg == 0` ⇒ `finalize(init(), ctx)`,
  return; `slot = carray(agg, (1,), intp)`; if `slot[0] == 0` ⇒ `finalize(init(), ctx)`,
  return; else `finalize(borrow_structref(state_type, slot[0]), ctx)`;
  `release_meminfo(slot[0])`.

**Empty groups.** SQLite calls `xFinal` (and may call `xValue`) even when `xStep`
never ran — `aggregate_context(ctx, 0)` then returns NULL (or, in the OOM-after-alloc
edge case, a zeroed `slot[0] == 0`). In either case the helper finalizes a fresh
`init()` state — no release, because nothing was exported — so the user's
`finalize`/`value` always receives a valid `state` and never needs a "no rows" branch.

**Lifecycle correctness.** `export_meminfo`'s +1 incref (on first step) is balanced
by exactly one `release_meminfo` in `xFinal`. `borrow_structref`'s incref is balanced
by the local's decref on scope exit (net zero per callback). `xValue` borrows but
never releases, so window state survives across the many `xValue` calls until the
single `xFinal`. This is the exact balance the phase-2 `test_udaf_no_meminfo_leak`
regression guards.

## Mechanism

**Selected: B — content-addressed anchored codegen.** Resolved by the spike on
2026-05-29 (results below); C (a shared lifecycle parameterized by dispatcher args)
was rejected because its overloads have no stable cross-process cache key.

For each registration the helper generates the callback source as text per UDAF,
**baking `state_type` and the user's `init`/`step`/`finalize` (+ `inverse`/`value` for
windows) as constants** — so the user-fn calls **inline** (full per-row speed, finding
2 below) — injects those into the exec namespace, and writes the source to a
**content-addressed anchor file** via `numbox/utils/preprocessing.py`'s
`_anchor_path`/`_materialize_anchor` (new subdir `numbox-sqlite-udaf`). Each distinct
UDAF therefore gets a stable, bounded cross-process cache key. The generated callbacks
are `cfunc(cache=True)`. The lifecycle bodies are exactly the four bullets under "What
the helper owns".

Registration then calls `sqlite3_create_function_v2` (aggregate) or
`sqlite3_create_window_function` (window) with
`flags = SQLITE_UTF8 | (SQLITE_DETERMINISTIC if deterministic)`, `p_app=0`,
`xDestroy=0`; the handle keeps the `cfunc`s and user-fn references alive.

### Correctness requirement: invalidation (validated by spike S3)

The content-address digest **must fold in a co_consts-sensitive serialization of the
user functions — `inspect.getsource(fn)` or `cloudpickle.dumps(fn)`, NOT bare
`fn.__code__.co_code`** — plus the `state_type` repr and the numbox + numba versions.
Two reasons: (1) two distinct UDAFs must not collide; (2) the user functions are
**inlined** into the wrapper but live in the *user's* module, not the anchor file, so
without folding their definition into the digest, editing `step` would not change the
anchor file and numba would serve a stale wrapper that inlined the old `step`.

`co_code` is **not** sufficient and is a real correctness trap (spike S3, confirmed on
numba 0.65.1): a numeric-literal-only edit such as `state.total += x*2` → `x*3` leaves
`co_code` **byte-identical** (the literal moves in `co_consts`, which `LOAD_CONST`
indexes), so a `co_code` digest produces a **false warm cache HIT that runs stale `*2`
machine code** (observed: result 30 where 45 was expected). `cloudpickle.dumps` /
`inspect.getsource` capture `co_consts` and invalidate correctly while still HITting for
an unchanged body. This is the same `co_consts` blind spot that `preprocessing.py`'s
anchor mechanism exists to defeat for the *generated* source; the user fns are an
*external* dependency, so they need the dependency-hash explicitly. (CRE's
`unique_hash_v` / `_py_func_unq` does exactly this — folds `co_code` **and** cloudpickle
bytes **and** `cre.__version__`; it only walks first-level deps, so document that a user
`step` calling other `@njit` helpers is only invalidated to first level unless the hash
recurses.)

### B implementation requirements (validated by spikes S1/S3)

1. **Exec namespace must carry `__name__`** (S1). Exec-ing the generated source into a
   bare `{}` writes the cold `.nbc` but **crashes on warm reload** —
   `Environment._rebuild_env` → `importlib.import_module(None)` → `AttributeError`,
   because `Environment.can_cache()` is False when `'__name__'` is absent from globals.
   Mirror numbox's `make_structref` / `@proxy` exec pattern, which seeds the namespace
   with `inspect.getmodule(func).__dict__` (so `__name__` and the real module globals
   are present).
2. **The user's `state_type` must live in an importable module** with a stable
   `__module__` (not `__main__`). This is a precondition for *any* cached approach —
   numbduck's `irr.py` documents that a `__main__`-defined state type warm-fails type
   inference ("No conversion from …StateType"). Validate or document.
3. **Orphan sweep:** register the new `numbox-sqlite-udaf` anchor subdir with
   `preprocessing.py`'s `_orphan_anchor_sweep`, and consider an age-based sweep of
   *superseded* `_<oldhash>.py` anchors — repeated edits to `step` accrete one stale
   anchor + `.nbc` per edit (unbounded across edits; CRE has the same growth and does
   not sweep, so this is a "nice to have", not a blocker).

### Rejected alternative: C — shared lifecycle + dispatcher args

A single shared `@njit(cache=True)` lifecycle parameterized by `state_type` (a
`TypeRef` arg) and the user fns (dispatcher args), with thin per-UDAF `cfunc`
trampolines. Far less machinery, and it composes and inlines at full speed — but it is
**structurally uncacheable cross-process** (spike S2). `str(typeof(step))` literally
embeds `hex(id(step.py_func))` — the ASLR-randomized heap address of the function — and
numba folds that into the overload **index key** (the signature tuple), so every process
computes a different key → guaranteed warm miss + unbounded `.nbc` growth. No cache
stamp or custom locator can fix this: the broken piece is the index key, not the
validity stamp. (Passing the user fns as `FunctionType` / `.as_func` values *does* give
a stable key, but then the call is **indirect** — see mechanism E. You get stable-cache
XOR inlining, never both, from any argument-passing form.)

### Verification record — deep re-review (2026-05-29, numba 0.65.1)

The mechanism space was mapped across numbox, numbduck, and DannyWeitekamp's
Cognitive-Rule-Engine (CRE), then verified with live spikes:

- **CRE precedent.** `define_CREFunc` (`cre/func.py:2187`) — "user supplies a function,
  wrap+cache it", the exact analog of this helper — *is* mechanism B: it codegens source,
  writes it to a content-hash-named file (`source_to_cache`), `import_from_cached`s it,
  with the hash folding the user fns' bytecode + cloudpickle + `cre.__version__`
  (`unique_hash_v`, `_py_func_unq`). CRE uses the indirect first-class-function path
  **only** for runtime *composition* of CREFunc instances (which can't be codegen'd at
  define time, and which it amortizes over a composition tree); our registration is a
  define-time operation, so B is the right analog.
- **numbduck precedent.** `irr.py` / Welford UDAFs already hand-write exactly the
  per-callback `@njit`-impl + `@cfunc`-trampoline shape that B generates.

Spikes (all on numba 0.65.1, isolated `NUMBA_CACHE_DIR`, cold-then-warm subprocesses
under `NUMBA_DEBUG_CACHE=1`):

| Spike | Hypothesis | Verdict |
|-------|------------|---------|
| **S1** | B (baked-global codegen, anchored, `cfunc(cache=True)`) warm-HITs cross-process on a real SQLite SUM UDAF, user fn inlined | **confirmed** — warm HIT (`.nbc` stable at 1), `sqlite3_value_int64`+NRT inlined into the impl, aggregate `==31`, `mi_alloc==mi_free` |
| **S2** | No argument-passing form (dispatcher / `.as_func` / cfunc-object) gives both warm-HIT and inlining | **confirmed** — dispatcher: inlined 0.43 ns but MISS (`.nbc` 1→2, address in key); `.as_func`/cfunc: HIT but indirect ~1.9 ns (4.4×). stable-cache XOR inlining |
| **S3** | B's invalidation is correct only with a co_consts-sensitive digest; bare `co_code` gives a stale false-HIT after a literal edit | **confirmed** — `co_code` digest: false HIT, stale result 30≠45; `cloudpickle.dumps` digest: recompiles, correct 45, stable HIT unchanged |
| **S4** | Mechanism E (shared cached lifecycle calling user step via `@proxy` extern symbol) caches but is indirect per row | **confirmed** — both warm-HIT, but E=3.02 ns vs B=1.21 ns (2.5×); asm shows `callq *%r13` extern vs B's inlined+unrolled body |

**Rejected non-B mechanisms:** **C** (dispatcher-arg) — uncacheable (S2). **D** (CRE
import-as-module) — same idea as B but heavier for numbox: needs `cloudpickle` (not in
the venv), a writable cache package on `sys.path`, and `config.CACHE_DIR` mutation;
numbox's exec-at-anchor avoids all of it. **E** (proxy extern-symbol) — cacheable but
indirect per row (S4), fatal for a per-row `step`. **F** (`_PreciseCacheLocator`) — a
clean way to auto-fold dep bytecode into the cache stamp, but it monkey-patches
numba-internal cache classes (version-fragile against numbox's `numba<0.66` pin) and
needs cloudpickle; B's manual digest achieves the same using only the public on-disk
anchor contract.

**Decision: implement B** — uniquely keeps **both** inlining (S1/S2/S4) and stable,
bounded cross-process caching (S1), buildable with **zero new infrastructure** (reuses
`preprocessing.py` `_anchor_path`/`_materialize_anchor`, the `highlevel.py`
`make_structref` exec-at-anchor recipe, and the `meminfo.py` bridge), and corroborated
as the right approach by CRE and numbduck.

## Placement

- New module **`numbox/core/bindings/_sqlite_udf_helpers.py`**, alongside the
  `_sqlite_*` family (all sqlite code lives under `core/bindings/`; no new top-level
  subpackage).
- Public `register_aggregate` / `register_window` in `__all__`, star-imported via
  `numbox/core/bindings/__init__.py` (same pattern as `_sqlite_udf`). The keep-alive
  handle class is returned but **not** in `__all__` — users hold it, never construct it.
- **Docs (mandatory):** a new `_*.py` module requires an `automodule` section in
  `docs/numbox.core.bindings.rst` plus the "Bindings module conventions" family entry,
  then `sphinx-build` exit 0 — per the CLAUDE.md "Adding a New Binding" doc rule.

## Error handling

- **Registration failure:** non-`SQLITE_OK` rc ⇒ raise `RuntimeError` with the rc and
  `sqlite3_errmsg(db)` decoded immediately (phase-2 errmsg-lifetime gotcha).
- **Input validation (raised at registration, clear messages):** `state_type` is a
  structref instance type; `register_window` received all five callbacks; the user
  functions are `@njit` and bakeable into the generated wrapper (raise a clear error
  rather than emit a cryptic numba typing error from the generated code).
- **Runtime (inside callbacks):** NULL `aggregate_context` (OOM) ⇒ early return, matching
  the phase-2 tests. Empty group is **not** an error — it is the `init()`-fresh-state path.
- **Keep-alive:** the handle docstring states loudly that dropping it frees the callback
  pointers (segfault). This is the one contract the user must honor.

## Testing & success criteria

The bar is: the helper reproduces the phase-2 hand-written results with the same memory
safety.

1. **Behavioral parity** — port `test_udaf_sum_structref` (→ 15),
   `test_udaf_empty_group` (→ 0), and `test_window_running_sum` (→ `[1,3,5,7,9]`) to the
   helper; assert identical results.
2. **Leak regression** — the helper version of `test_udaf_no_meminfo_leak`: many
   iterations ⇒ `mi_alloc == mi_free`. Proof the helper preserves the `export`/`release`
   balance and the release-in-`xFinal`-only rule.
3. **`deterministic=True`** registers cleanly and computes correctly.
4. **No cross-contamination** — two different UDAFs (different `state_type`s) registered
   in one process give correct independent results (guards C's per-overload specialization).
5. **Cross-process cache** — a subprocess registers + runs a UDAF twice; correct both
   times, and on the warm run the target `.nbc` is **loaded, not re-saved** (assert via
   `NUMBA_DEBUG_CACHE` output or a stable `.nbc` count). Validates B's stable-key claim
   (spike S1) and guards against a regression to the C failure mode (unbounded `.nbc`).
6. **Invalidation** — register a UDAF, then edit `step` so only a numeric literal changes
   (`x*2` → `x*3`); a fresh process must **recompile and return the new result**, not a
   stale cached one. Guards the co_consts-sensitive digest (spike S3 — a `co_code` digest
   silently fails this).
7. **flake8 clean** at `--max-line-length=127`; runs in existing `numbox_ci`
   (which already includes `--durations=20`).

Tests 4–6 are the regression guards for B's per-UDAF specialization, stable
cross-process caching, and correct invalidation (the full mechanism re-review and the
S1–S4 spikes that selected B are recorded under "Mechanism").

## Dependencies & sequencing

This work sits on phase-2 modules that currently exist **only on `feat/sqlite-udf`**
(upstream PR #17 BLOCKED). Implementation therefore branches off `feat/sqlite-udf`, not
`origin/main`. The **wait-for-#17-merge vs. develop-now-and-rebase** call, and the
upstream-PR-branch strategy, are made at the writing-plans → execute boundary per the
standing numbox workflow rules. The spec itself is independent of phase-2's merge status.
