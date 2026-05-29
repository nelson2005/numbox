# SQLite UDAF Registration Helper — Design

**Date:** 2026-05-29
**Status:** Design approved; pending spec review → implementation plan
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
| Wrapper mechanism | **C-primary** (shared cached lifecycle + thin trampolines); **B fallback** (anchored codegen), gated by a spike |
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

### C-primary: shared cached lifecycle + thin trampolines

Four **static** `@njit(cache=True)` functions live in the helper module — the
lifecycle, parameterized by `state_type` (a `TypeRef` argument, exactly as
`borrow_structref` already takes it) and the user functions (passed as values):

```python
_agg_step(state_type, init, step,      ctx, argc, argv_pp)
_agg_inverse(state_type, inverse,      ctx, argc, argv_pp)
_agg_value(state_type, init, value,    ctx)
_agg_final(state_type, init, finalize, ctx)
```

Because `state_type`/`init`/`step` are **arguments**, numba specializes each UDAF
into a **distinct overload** (distinct `TypeRef`, distinct function types) under the
module's own filename, so numba's **normal per-overload cache** handles it — no
content-addressing, no anchor files. Invalidation is **automatic**: the user
functions are genuine compile dependencies, so editing `step` restamps and recompiles.
The expensive compilation (lifecycle + the user's `@njit(cache=True)` logic) is cached
across cold starts.

Per registration, the helper builds the trivial `cfunc`s SQLite needs — each binds
this UDAF's `state_type` + user fns and forwards into the cached lifecycle:

```python
_step_cb = cfunc(types.void(types.intp, types.int32, types.intp))(
    lambda ctx, argc, argv: _agg_step(state_type, init, step, ctx, argc, argv))
```

These trampolines share an identical C signature across UDAFs, so they cannot be
cache-shared — they are **`cache=False`**. That is acceptable: each body is a single
forwarding call and the callee loads from cache, so per-process recompile cost is
negligible. Registration then calls `sqlite3_create_function_v2` (aggregate) or
`sqlite3_create_window_function` (window) with
`flags = SQLITE_UTF8 | (SQLITE_DETERMINISTIC if deterministic)`, `p_app=0`,
`xDestroy=0`; the handle keeps the `cfunc`s and user-fn references alive.

### Why this resolves the caching hazard

Dynamically generated numba callbacks collide in the cache because numba keys an
overload on `(filename, firstlineno, bytecode, signature)`, and **all four callbacks
share the identical C signature** `void(intp, int32, intp)` (or `void(intp)`). A naive
closure makes it worse — the wrapper's *code object is byte-identical* across every
UDAF, only the captured freevars differ — so numba would serve the wrong cached
machine code. C **dissolves** this by lifting the per-UDAF variation into
signature-differentiating arguments. (The alternative, B, *works around* the symptom
with content-addressed filenames.)

### B fallback: anchored codegen

If the spike rules C out, fall back to generating each callback's source as text per
UDAF, injecting `state_type`/user fns into the exec namespace, writing to a
content-addressed anchor file via `numbox/utils/preprocessing.py`'s
`_anchor_path`/`_materialize_anchor` (new subdir `numbox-sqlite-udaf`), and
`cfunc(cache=True)`. **Correctness requirement:** the digest must fold in a hash of
the user functions' bytecode + the `state_type` repr, both so two byte-identical
generated blocks don't collide and so editing `step` invalidates — numba will not see
the user fns as dependencies of generated text on its own.

The **public API and user contract are identical** under C or B; the choice is purely
internal.

### Spike (first implementation task) and decision rule

C rests on numba behavior to be **verified, not assumed**. Build the C skeleton for a
`sum` aggregate and measure:

1. **Composition + caching** — a `cfunc` trampoline calling a cached `@njit` lifecycle
   with `TypeRef` + function-value args composes, runs correctly, and the lifecycle
   *loads from cache* on a second process.
2. **Per-row cost** — whether the user-fn call inlines or is indirect. Passing the user
   fns as **njit dispatchers** (each its own compile-time `Dispatcher` type) may yield
   both clean caching *and* an inlinable direct call; passing them as `.as_func`
   function-values is definitely indirect. Measure the per-row delta on a realistic
   aggregate.

**Decision rule:** ship **C** unless the spike shows it cannot compose, or the
indirect-call cost is material on a realistic aggregate — in which case fall back to **B**.

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
  functions are passable into the lifecycle (if the spike selects the dispatcher path
  and a function lacks the required form, say so explicitly rather than emit a cryptic
  numba typing error).
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
   times, and the numba cache dir is populated on the second run (validates C's caching
   claim end-to-end).
6. **flake8 clean** at `--max-line-length=127`; runs in existing `numbox_ci`
   (which already includes `--durations=20`).

Tests 4–5 plus the spike are the C-vs-B decision evidence.

## Dependencies & sequencing

This work sits on phase-2 modules that currently exist **only on `feat/sqlite-udf`**
(upstream PR #17 BLOCKED). Implementation therefore branches off `feat/sqlite-udf`, not
`origin/main`. The **wait-for-#17-merge vs. develop-now-and-rebase** call, and the
upstream-PR-branch strategy, are made at the writing-plans → execute boundary per the
standing numbox workflow rules. The spec itself is independent of phase-2's merge status.
