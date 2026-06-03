# SQLite `query_to_array` + virtual-table phase 5 — Design

**Date:** 2026-06-03
**Status:** Design (awaiting review → implementation plan)
**Builds on:** phase 4 read-only numpy virtual tables (`register_table`, `_sqlite_vtable.py`)

## Summary

One PR adding four capabilities to numbox's SQLite bindings:

1. **`query_to_array(db, sql, dtype)`** — run a `SELECT` and collect the result rows into a new numpy structured array (the read-out complement to phase-4's read-in-place `register_table`).
2. **`xBestIndex` constraint pushdown** on the read-only vtable — let `WHERE` eq/range constraints prune the scan and report real planner costs.
3. **Table-valued functions** — `register_tvf(db, name, arg_types, out_dtype, fn)`: a vtable whose rows are a numpy array computed lazily from call arguments (`SELECT * FROM f(a, b)`).
4. **`create_module_v2` + `xDestroy`** — SQLite-driven cleanup of the per-table descriptor, replacing the phase-4 "caller must retain the handle forever or risk use-after-free" model.

`insert_array` (numpy → real table) was considered and **dropped**: `register_table` already covers read-in-place, and the write side is only justified for durable/indexable/writable storage from inside an `@njit` pipeline — too narrow for this PR.

## Feasibility verification (done 2026-06-03)

Before this spec was written, the four load-bearing assumptions were verified empirically — each confirmed by an isolated, runnable numba/SQLite spike (evidence retained). **All four: feasible-with-caveats, high confidence.** Their findings are folded into the design below; the appendix records the verdicts.

**Environment note:** the local venv runs **numba 0.65.1**, numpy 2.4.5, Python 3.12.3. The CI matrix exercises numba 0.60.0–0.65.1, so correctness is gated across both pins; the spikes ran on 0.65.1 only and the existing array-UDAF (already CI-green on 0.60.0 + 0.65.1) covers the floor for the shared NRT idiom.

## Module layout

| File | Change | Purpose |
|---|---|---|
| `numbox/core/bindings/_sqlite_typemap.py` | **new** | Shared numpy-dtype-tag ↔ SQLite read/declare mapping + the UTF-32↔UTF-8 helpers, extracted from `_sqlite_vtable.py` (surgical, no behavior change). Consumed by `query_to_array`, the vtable, and the TVF. |
| `numbox/core/bindings/_sqlite_query.py` | **new** | `query_to_array`. |
| `numbox/core/bindings/_sqlite_vtable.py` | extend | pushdown (`xBestIndex`/`xFilter`/`xNext`), `register_tvf`, `create_module_v2`/`xDestroy`. |
| `numbox/core/bindings/_sqlite_constants.py` | extend | `SQLITE_INDEX_CONSTRAINT_*` op codes (EQ=2, GT=4, LE=8, LT=16, GE=32, NE=68, ISNULL=71, IS=72, …). |
| `numbox/core/bindings/signatures.py` | extend | `sqlite3_create_module_v2`. |

All new public symbols reach the package surface via the existing `from ... import *` in `numbox/core/bindings/__init__.py`; private helpers stay `_`-prefixed. `jit_options` cache settings are honored throughout (per the standing maintainer directive); cfunc exception guards use the bare `try/except` → `sqlite3_result_error`/return-code pattern established in phases 2–3.

---

## Component 1 — `query_to_array(db, sql, dtype) -> arr`

Run `sql` on `db`, read each result row into a numpy structured array typed by `dtype`, return an owned, exactly-sized array.

### Interface
- `db` — connection pointer (as elsewhere in the bindings).
- `sql` — query text.
- `dtype` — a numpy structured dtype describing the output columns, **one field per result column, in order**.
- Returns a 1-D structured array of length = number of result rows.

### Mechanism
`prepare_v2(sql)` → `step` loop → per row, read each column by **field position** via the shared type-map (`column_int64`/`column_double`/`column_text`(UTF-8→UTF-32)/`column_blob`) into a geometrically-grown NRT buffer → trim to the exact row count → `finalize` → return.

### Critical design rule (from assumption **A2**)
The dtype **must be passed into the jitted core as a `numba.from_dtype(dtype)` typeref argument** — *not* read from a module-global dtype inside the jitted body.

> The A2 spike proved a silent stale-cache hazard: a module-global dtype that changes while the function's `co_code` stays byte-identical causes numba (whose cache key hashes `co_code`, not the `LOAD_GLOBAL`'d dtype) to load **wrong-layout** machine code from a shared cache dir — returning an array with the *old* itemsize/fields. Passing the dtype as a typeref argument puts the full `Record(...)` layout (field names/types/offsets) into the overload/cache key, so a different or changed dtype correctly recompiles.

The geometric-growth helper `_grow_copy(old, new_cap, dt)` likewise takes `dt` as a typeref arg and must not close over a module global. Trim by allocating a fresh `np.empty(count, dt)` and copying (owned, exactly-`n`; a `buf[:count]` slice would be a non-owning view pinning the over-allocation).

### Semantics
- **Field/column count mismatch** → raise (the field count must equal `sqlite3_column_count`).
- **NULL** cells → `NaN` (float fields) / `0` (int fields) / empty (text/blob), documented.
- **Type mismatch** between a SQL column and its target field → rely on SQLite's `column_*` implicit coercion (documented).
- `np.empty` leaves the over-allocated tail uninitialized; only the first `count` rows are filled and returned.
- Errors from `prepare`/`step` surface via the existing `errcode`/`errmsg` helpers.

---

## Component 2 — `xBestIndex` pushdown (read-only vtable)

Today the vtable always full-scans. Pushdown lets `xBestIndex` claim eq/range `WHERE` constraints, pass their values to `xFilter`, and have the cursor skip non-matching rows.

**Honest benefit ceiling:** a numpy array has no index, so this stays **O(n)** — it does *not* make lookups sublinear. Its real value is (1) correct `estimatedRows`/`estimatedCost` so joins order well, (2) skipping `xColumn` materialization for filtered-out rows, and (3) it is the **exact machinery TVF reuses** for hidden-column args. (A future "declare a column sorted → binary-search in `xFilter`" extension would give true sublinear lookups; deferred.)

### Design (from assumption **A3**, verified against a live vtable + the C ABI)
- Add three element dtypes to `_sqlite_vtable.py` (offsets verified vs the SysV/LP64 C ABI; portable to Win64 since `int` stays 4 bytes):
  - `_CONSTRAINT_DTYPE = np.dtype([('iColumn','i4'),('op','u1'),('usable','u1'),('iTermOffset','i4')], align=True)` — itemsize **12**.
  - `_USAGE_DTYPE = np.dtype([('argvIndex','i4'),('omit','u1')], align=True)` — itemsize **8**.
  - `_ORDERBY_DTYPE = np.dtype([('iColumn','i4'),('desc','u1')], align=True)` — itemsize **8**.
  - Add `itemsize` asserts (12/8/8) mirroring the existing `_IDX_INFO_DTYPE` asserts.
- Keep `_IDX_INFO_DTYPE` unchanged — its input-field offsets (`nConstraint@0`, `aConstraint@8`, `nOrderBy@16`, `aOrderBy@24`, `aConstraintUsage@32`) are already correct and tested.
- **`_xbestindex`:** loop `nConstraint`; for each `usable` constraint whose `op == SQLITE_INDEX_CONSTRAINT_EQ` on a supported column, set `usage[i].argvIndex` (next sequential **1-based** slot); encode the bound set in `idxNum`; set `estimatedRows`/`estimatedCost`. Full-scan fallback when nothing is claimed.
- **`_xfilter`:** view `argv` as `carray(_cast_int_to_void_p(argv),(argc,),dtype=np.intp)`; for each bound slot read the `sqlite3_value*` and decode via the type-appropriate `sqlite3_value_*` accessor; store the bound predicate(s) in cursor state; advance to the first matching row. **`argvIndex` is 1-based, `argv` is 0-based → `argv[argvIndex-1]`** (off-by-one is a silent wrong-value bug).
- **`_xnext`/`_xeof`:** when a filter is active, skip to the next matching row.
- **`omit` conservative:** leave `omit=0` (SQLite re-checks the surfaced rows — all of which pass, so it's cheap) unless `xFilter` provably enforces the exact constraint. Setting `omit=1` without exact enforcement leaks wrong rows (SQLite won't catch it).
- The current `_xfilter` signature already carries `(cur, idx_num, idx_str, argc, argv)` — only the body changes. The cursor dtype gains fields for the bound key(s)/filter state; multi-column/multi-op pushdown uses a `sqlite3_malloc`'d side buffer pointed to from the cursor (same pattern as the existing `scratch_p`), since the cursor dtype is fixed-size.

**First shippable increment:** EQ-only on integer columns, `omit=0`. Expand op/type coverage incrementally.

---

## Component 3 — Table-valued functions: `register_tvf(db, name, arg_types, out_dtype, fn)`

A TVF is an eponymous vtable with hidden columns acting as arguments. `register_tvf` registers a vtable whose result rows are a numpy structured array computed by `fn` from the call arguments — "a `register_table` whose array is computed lazily from the args."

### Interface
- `arg_types` — the hidden argument column types (used to declare the HIDDEN columns and decode `argv`).
- `out_dtype` — the structured dtype of the result rows (used to declare the visible columns and to allocate the per-call array). Passed as a `numba.from_dtype` typeref to the allocator (A2 rule).
- `fn` — a user `@njit` function `fn(*args) -> np.ndarray` (structured, matching `out_dtype`), invoked from `xFilter` via the proxy/cfunc pattern used by the UDAF helpers.

### Schema & arg plumbing
`declare_vtab` emits `out_dtype`'s columns (visible) followed by `arg_types` as **HIDDEN** columns. `xBestIndex` requires EQ constraints on all hidden-arg columns (assigns their `argvIndex`; if any required arg is unbound, mark the plan unusable / very high cost). `xFilter` reads the arg values from `argv` (reusing the Component-2 machinery) and calls `fn`.

### Per-query result lifetime (from assumption **A1**, verified)
Model the cursor's result storage as a `[meminfo_p, data_p]` pair (2×`intp`) — the proven array-backed window-UDAF idiom — added as a **new NRT-backed slot on the TVF cursor dtype**, *parallel to* (not overloading) the existing `scratch_p`:

1. `xFilter` calls the `@njit` allocator → `np.empty(n, dtype=out_dtype)`; takes `structref_meminfo` of the array; `_incref_meminfo` to pin it; stores `meminfo_p` and `data_p` in the slot.
2. `xColumn`/`xNext` read via `carray(data_p, (n,), out_dtype)` + the shared tag dispatch. **Guard `slot[0] == 0`** (NULL / not-yet-allocated) before every `carray` — a NULL `data_p` segfaults.
3. `xClose` calls `release_meminfo(slot[0])` **exactly once** and zeroes the slot.

Required invariants (the spec mandates all of these):
- The release lives in **one** owner (`xClose`) and must be reached **even on the SQLite error path** — wrap the relevant `@njit` body in `try/except` (mirrors the UDAF rule; also lets numba run the local decref the unwind would skip). `xClose` is SQLite-guaranteed for opened cursors.
- Release **only** via `release_meminfo` (`NRT_MemInfo_release`, deliberately outside numba's `removerefctpass` allowlist) — never a bare `context.nrt.decref` intrinsic (could be stripped).
- The `_incref_meminfo` is load-bearing (negative control proved the array is otherwise freed within the producing call); its `removerefctpass`-safety rides on the allocator node being present in the lowered body — the same assumption the shipped array-UDAF already relies on, so the body must **not** be promised allocation-free.
- Reads reconstruct nothing — the producing `@njit` frame is gone by `xColumn` time; only the pinned meminfo keeps the buffer alive.

---

## Component 4 — `create_module_v2` + `xDestroy`

Replace `register_table`/`register_tvf`'s `create_module` with `create_module_v2(db, zName, pModule, pAux, xDestroy)` so SQLite drives per-table descriptor cleanup, fixing phase-4's "caller must retain the handle forever or a later query reads freed memory" hazard.

### What the A4 spikes established (and the complexity this reveals)
- **Per-table firing works.** `xDestroy` is keyed to the `(db, name)` registration entry, not the module pointer. Since `register_table` uses a **distinct name per call** with one shared `THE_MODULE`, each table gets its own `xDestroy` fire with its own `pClientData`. (This was the make-or-break uncertainty — resolved positively.)
- **`xDestroy` must NOT free the descriptor with SQLite's allocator.** The `pClientData` (`built.c`) and all its side buffers (offsets/tags/widths/schema/`arr`) are **numpy/Python-heap** allocations. Calling `sqlite3_free`/C-`free` on them is a cross-allocator free = heap corruption. **`xDestroy`'s only correct job is to release a *Python* reference** (`Py_DECREF`), after which ordinary Python GC reclaims everything.
- **This needs a GIL-safe CPython bridge from the C callback** — a pure numba `@cfunc` that "frees memory" does *not* solve it. Concrete design: a module-level registry dict keyed by an integer token; pass the token (or a `Py_INCREF`'d `py_object`) as `pClientData`; `xDestroy` (a C callback re-entering CPython with the GIL held) does the matching dict-pop / `Py_DECREF`.
- **Teardown ordering is safe.** On a clean `sqlite3_close`: `xColumn → xClose → xDisconnect → xDestroy` — so `xDestroy` dropping the descriptor cannot race in-flight `xColumn` reads.
- **Deferred close.** `sqlite3_close_v2` (zombie close) and a `close` that returns `BUSY` due to an unfinalized statement **defer** `xDestroy`; the descriptor + `arr` must stay alive until the deferred fire — the keep-alive cannot be dropped at the call site.
- **Duplicate-name re-registration** fires the *first* entry's `xDestroy` **synchronously inside** the new `create_module_v2` call → the registry pop must tolerate re-entrancy.
- **`_VTableHandle` still required.** The v2 change does not remove the keep-alive set; it changes *who triggers release* (SQLite-driven instead of caller-driven). The `register_table` docstring's "caller must retain the returned handle" warning is revised: the real keep-alive lives in the module-level registry; any returned handle becomes advisory.

> **Scoping flag (for review):** xDestroy was originally scoped as a "small, mostly mechanical" item. A4 shows it is the **most intricate piece** in this PR — a GIL-safe C→CPython `Py_DECREF` bridge with re-entrancy and deferred-close timing. **Recommendation: keep it, but treat it as its own implementation phase with dedicated tests; if the PR gets too large for review, this is the natural piece to split into a follow-up** (the other three components don't depend on it — they work with the existing `create_module` keep-alive model).

---

## Cross-cutting concerns

- **Type-map reuse:** extract the dtype-tag ↔ SQLite read/declare/value logic into `_sqlite_typemap.py`; `query_to_array`, the vtable column path, and the TVF all consume it. Surgical move, no behavior change to phase 4.
- **Caching:** all jitted wrappers honor `jit_options` (`@proxy(..., jit_options=jit_options)`); cfuncs follow the phase-4 `_CACHE = jit_options.get("cache", True)` precedent. The A2 typeref-dtype rule applies to every jitted body that takes a caller-supplied dtype.
- **Pointer/value helpers:** reuse `lowlevel.py` (`array_data_p`, `get_str_from_p_as_int`, `get_unicode_data_p`, `carray` casts), the `sqlite3_value_*` / `sqlite3_result_*` accessors, and the `meminfo.py` intrinsics (`structref_meminfo`, `_incref_meminfo`, `release_meminfo`). Don't reinvent.
- **Op-code constants:** add `SQLITE_INDEX_CONSTRAINT_*` to `_sqlite_constants.py` + `__all__` so `xBestIndex` branches symbolically.

## Testing

- **`query_to_array`:** round-trips for numeric/text/blob; NULL coercion; growth across the realloc boundary (large `n`); field/column-count mismatch raises; empty result; SQLite type coercion; cross-process cache hit; **A2 guard test** — two different dtypes through the same cache dir must produce correctly-laid-out arrays (no stale-cache cross-contamination).
- **Pushdown:** EQ prunes correctly; each range op; multi-constraint; `EXPLAIN QUERY PLAN` shows the chosen `idxNum`; correctness vs a full-scan baseline; `argvIndex` 1-based→0-based mapping.
- **TVF:** hidden-arg passing; `fn` invoked with the right args; multiple calls with different args; **leak-balance** (NRT `memsys_enable_stats`; `mi_alloc` delta == `mi_free` delta over ≥10 iters) run on **both** the min and max numba pins; missing-arg behavior; NULL `data_p` guard.
- **`xDestroy`:** register→query→`close` fires exactly once; two tables → two distinct fires; duplicate name → first descriptor released at re-register, second at close; unfinalized-statement close returns `BUSY` and does **not** release; no double-free / use-after-free.
- **Gate:** full numba matrix (`numbox_ci`) + `flake8` + `sphinx-build` + `doc-codeblock-flake8` + `lychee`. Docs: extend `docs/numbox.core.bindings.rst` for the new modules/symbols.

## Open questions for review

1. **xDestroy scope** — keep in this PR (with the CPython bridge), or split to a follow-up? (Recommendation: keep but phase it; it's the natural split point if review size becomes a concern.)
2. **TVF `register_tvf` argument shape** — `arg_types` + `out_dtype` as separate parameters (current design), or a single combined spec object?
3. **Pushdown op coverage v1** — EQ-only-on-int (recommended first increment) vs eq+range on all numeric columns in the first cut.

## Appendix — verified assumptions

| # | Assumption | Verdict | Key finding |
|---|---|---|---|
| A1 | `@njit` fn called from an `xFilter` `@cfunc` can allocate + return an NRT structured array, kept alive in the cursor across `xColumn`, released in `xClose`. | feasible-with-caveats (high) | Use the `[meminfo_p, data_p]` slot idiom; `_incref_meminfo` is load-bearing; release via `release_meminfo` in `xClose` only, reached on the error path; NULL-guard `data_p`. Verified on numba 0.65.1; floor covered by the shipped array-UDAF. |
| A2 | `@njit` geometric growth with a caller-supplied dtype caches correctly under `cache=True`. | feasible-with-caveats (high) | **Pass dtype as a `from_dtype` typeref arg, not a module global** — proved a stale-cache hazard otherwise. Trim via fresh `np.empty`+copy. |
| A3 | `sqlite3_index_info` constraint/usage arrays + `xFilter` `argv` can be modeled as numpy dtypes / carray views like the existing `index_info`. | feasible-with-caveats (high) | `_CONSTRAINT_DTYPE` (12B), `_USAGE_DTYPE` (8B), `_ORDERBY_DTYPE` (8B), `align=True`; live spike confirmed `WHERE col0=5` delivers `5` to `xFilter`; `argvIndex` is 1-based; add op-code constants. |
| A4 | `create_module_v2`/`xDestroy` frees the per-table descriptor at the right time without double-free vs `_VTableHandle`. | feasible-with-caveats (high) | Per-`(db,name)` firing works; descriptor is numpy-heap so `xDestroy` must `Py_DECREF` (GIL-safe CPython bridge), not `sqlite3_free`; close_v2 defers; duplicate-name fires re-entrantly; `_VTableHandle` stays. |
