# SQLite virtual tables (numpy-backed, read-only) — design

Date: 2026-05-31. Phase 4 of the SQLite buildout, stacked on phase 3.

**Base.** This work stacks on `feat/sqlite-udaf-helper` (fork [#37](https://github.com/nelson2005/numbox/pull/37) / upstream [#19](https://github.com/Goykhman/numbox/pull/19), the `register_aggregate`/`register_window` helpers). Phase 3 is unmerged, so the phase-4 branch descends from it rather than `origin/main`. Once #19 merges and fork `main` re-syncs, the phase-4 branch rebases onto the new `main` to drop phase 3's commits from its diff.

**Target release.** The minor after phase 3 (#19) lands; Goykhman tags releases (latest is `0.5.15`), so tentatively `0.5.16`/`0.5.17`. Not load-bearing.

## 1. Summary

Add `register_table(db, name, arr, ...)` — a helper that exposes a **numpy array as a read-only SQLite virtual table**, queryable with ordinary SQL (`SELECT`, `WHERE`, `ORDER BY`, `JOIN`, aggregates). The array is exposed **zero-copy**: SQLite reads cells directly out of the array's buffer through generated `@cfunc` callbacks.

Two front doors, one internal model:

- a **2-D ndarray** (homogeneous dtype) → columns are `arr[:, j]`;
- a **1-D structured array** → columns are the dtype fields (mixed types).

Both reduce to one base pointer + per-row stride + a per-column `(byte_offset, dtype_tag, width)` table, so a single `xColumn` code path serves C-order, F-order, non-contiguous slices, reversed/negative-stride views, and structured fields alike.

This mirrors the phase-3 `register_aggregate` idiom (caller supplies data/logic; the helper owns the SQLite-callback plumbing and lifetime discipline) but is **simpler than phase 3**: because the table is read-only and fully generic, the callbacks are one fixed module compiled once — no per-instance codegen, no content-addressed anchors, no NRT/meminfo dance.

The one genuinely new mechanism vs. all prior SQLite phases: a **`sqlite3_module` struct of function pointers** (UDF/UDAF registration passes individual `cfunc.address` values; a virtual table needs a populated struct). We solve it once, at import, for a single shared module.

**Non-goals (→ phase 5+):** writes (`xUpdate`), constraint pushdown (`xBestIndex`/`xFilter` argument binding), table-valued functions with arguments (HIDDEN columns), `>2`-D / object-dtype / ragged arrays, masked-array / NULL semantics, `datetime64`/`timedelta64`, complex dtypes.

## 2. Scope and non-scope

### In scope

| Capability | Surface | Notes |
|---|---|---|
| Public helper | `register_table(db, name, arr, columns=None, *, text_as_blob=False)` | Registers an eponymous module named `name`; returns a `_VTableHandle` the caller must retain |
| 2-D ndarray input | any C/F/strided 2-D array | column names from `columns` (required) |
| Structured 1-D input | `arr.dtype.names` | column names from fields; `columns` optional override (renames in place) |
| Numeric columns | int8/16/32/64, uint8/16/32/64, float32/64, bool | → `sqlite3_result_int64` / `_double` |
| Bytes columns | `'S'` | → `TEXT` (or `BLOB` via `text_as_blob`); trailing NUL padding trimmed, interior NULs preserved |
| Unicode columns | `'U'` (UTF-32LE) | → `TEXT` via a `utf32_to_utf8` njit encoder |
| Read protocol | `xConnect`/`xBestIndex`(full-scan)/`xOpen`/`xClose`/`xFilter`/`xNext`/`xEof`/`xColumn`/`xRowid` + `xDisconnect` | one static `sqlite3_module`, generic over all tables |
| New bindings | `sqlite3_create_module`, `sqlite3_declare_vtab`, `sqlite3_malloc` | `sqlite3_free` already bound (phase 1) |
| New lowlevel helper | `load_unaligned(p, ty)` | `builder.load(..., align=1)` for packed structured fields |

### Out of scope (deferred, with rationale)

| Capability | Why deferred |
|---|---|
| Writes — `xUpdate` (INSERT/UPDATE/DELETE) | Doubles the surface and adds backing-store mutation semantics. The use case is read exposure of existing arrays. Phase 5. |
| `xBestIndex` constraint pushdown / `xFilter` arg binding | Requires reading/writing the nested `sqlite3_index_info` arrays (`aConstraint`/`aOrderBy`/`aConstraintUsage`) and version-sensitive struct offsets. v1 does a trivial full scan; SQLite still applies `WHERE`/`ORDER BY` correctly, just without pushdown. |
| Table-valued functions with arguments (HIDDEN columns) | The only read shape that *forces* real `xBestIndex` (mapping HIDDEN-column constraints to `xFilter` argv). Separate feature. |
| `>2`-D arrays | Would need flattening or multi-index rowids. Reject at registration. |
| object-dtype / ragged / variable-length strings | Not `@njit`-addressable as a flat buffer. |
| masked arrays / NULL / sentinel-NULL | numpy numeric arrays have no NULL; every cell yields a value. NaN passes through as `REAL` (see §6). A `mask=`/NULL story is its own design. |
| `datetime64` / `timedelta64` | Storable as `int64`, but the SQL type + epoch semantics deserve their own decision. |
| complex64/128 | No native SQLite type; would need a 2-column or text encoding convention. |

## 3. Architecture

### 3.1 Key decision: one generic fixed module, not codegen-per-table

Phase 3 generated per-UDAF callback source (baking the state type + user functions in as globals) and content-address-cached it, because each UDAF has different *types*. A numpy table is **read-only and fully generic**: every table uses identical callbacks that differ only in *runtime data* (base pointer, strides, dtype tags, schema). So:

- **Compiled once, at import:** ~10 distinct `@cfunc(cache=True)` callbacks and one module-global `sqlite3_module` ctypes struct.
- **Per `register_table`:** build a small descriptor (plain Python + ctypes) and call `sqlite3_create_module(db, name, THE_MODULE, &descriptor)`. SQLite permits registering the same module under many names with different `pClientData`, so every table shares the one struct.

No anchor/digest/`exec` machinery; `@cfunc(cache=True)` is genuinely supported in numba 0.65.1 (it installs the same `FunctionCache` as `@njit(cache=True)`; phase-4 data lives in the runtime descriptor, not in `co_consts`/globals, so there is no cache-key hazard). Strides and dtypes are *data, not types*, so they belong in the descriptor — baking them would defeat caching (every differently-strided array would recompile). A `@cfunc` body can call the `@proxy`/`@njit` SQLite bindings directly (they inline as extern-decl intrinsics — the same way every libc/sqlite binding is already called inside `@njit`); the phase-3 two-layer pattern existed *only* to cache user-supplied callbacks and is not needed here.

### 3.2 File layout

```
numbox/core/bindings/
├── _sqlite_vtable.py   [NEW]  register_table + _VTableHandle + the static module/cfuncs + utf32_to_utf8
├── signatures.py       [EXTENDED]  +3 entries in signatures_sqlite
├── __init__.py         [EXTENDED]  +1 line:  from numbox.core.bindings._sqlite_vtable import *
├── _sqlite_result.py   [unchanged]  (sqlite3_result_* reused)
├── _sqlite_value.py    [unchanged]
└── ...
numbox/utils/
└── lowlevel.py         [EXTENDED]  + load_unaligned(p, ty)
docs/
└── numbox.core.bindings.rst  [EXTENDED]  new automodule section ONLY — not the _call_lib_func
                                          family-list (follows the _sqlite_udf_helpers precedent:
                                          a codegen/cfunc module, not a thin _call_lib_func wrapper)
test/core/
└── test_sqlite_vtable.py     [NEW]
```

### 3.3 The descriptor and the layout/strides model

`register_table` snapshots the array's layout (in Python) into a `ctypes.Structure` held alive in the handle; its address is the module's `pClientData`:

```python
class _NdarrayTableDescriptor(ctypes.Structure):
    _fields_ = [
        ("nrows",         ctypes.c_int64),   # +0
        ("ncols",         ctypes.c_int32),   # +8
        ("_pad",          ctypes.c_int32),   # +12
        ("row_stride",    ctypes.c_int64),   # +16  arr.strides[0]
        ("data_base",     ctypes.c_int64),   # +24  arr.ctypes.data
        ("col_offsets",   ctypes.c_int64),   # +32  -> int64[ncols] buffer
        ("col_tags",      ctypes.c_int64),   # +40  -> int32[ncols] buffer
        ("col_widths",    ctypes.c_int64),   # +48  -> int64[ncols] buffer
        ("schema_ptr",    ctypes.c_int64),   # +56  -> NUL-terminated CREATE TABLE bytes
        ("scratch_bytes", ctypes.c_int64),   # +64
    ]                                         # sizeof == 72
```

| field | source |
|---|---|
| `nrows` | `arr.shape[0]` |
| `ncols` | `arr.shape[1]` (2-D) or `len(arr.dtype.names)` (structured) |
| `row_stride` | `arr.strides[0]` |
| `data_base` | `arr.ctypes.data` |
| `col_offsets[]` → int64[ncols] | `j*arr.strides[1]` (2-D) or `fields[name].offset` (structured) |
| `col_tags[]` → int32[ncols] | dtype enum per column |
| `col_widths[]` → int64[ncols] | **field byte width (`itemsize`)** — `'S'` reads `width` bytes; `'U'` derives code points = `width // 4`; unused for numeric |
| `schema_ptr` | NUL-terminated `CREATE TABLE …` bytes |
| `scratch_bytes` | `max(width + 1)` over `'U'` columns (= `4·(width//4)+1`), else 0 |

The `col_offsets` / `col_tags` / `col_widths` buffers are retained numpy arrays; their `.ctypes.data` are stored. `schema_ptr` points at a retained `bytes`. njit code reads scalar fields with `load_at(desc + off, ty)` and the per-column arrays with `carray(_cast_int_to_void_p(ptr), (ncols,), dtype)`.

**Offset/width discipline.** The njit offset constants are a hand-mirror of the ctypes layout, so `register_table` asserts each `_NdarrayTableDescriptor.<field>.offset` equals the literal offset the njit code uses (and `ctypes.sizeof(...) == 72`) — a future field change then fails loudly instead of reading garbage. Each `load_at(desc+off, ty)` must use the `ty` whose width matches its ctypes field (`int64` for `nrows`/`row_stride`/`data_base`/`col_*` pointers/`scratch_bytes`, `int32` for `ncols`). The `CREATE TABLE` column list and the `col_offsets`/`col_tags`/`col_widths` arrays are built in **one shared order** — `dtype.names` for a structured array (a `columns` override renames in place, never reorders), the `columns` sequence for a 2-D array — so `xColumn`'s index `j` selects the same column in all of them.

**The one addressing formula** (`xColumn`):

```
addr(rowid, j) = data_base + rowid * row_stride + col_offsets[j]
```

How each layout maps (itemsize `s`, shape `(n, m)`):

| Layout | `arr.strides` | `row_stride` | `col_offsets[j]` |
|---|---|---|---|
| C-contiguous 2-D | `(m·s, s)` | `m·s` | `j·s` |
| F-contiguous 2-D | `(s, n·s)` | `s` | `j·n·s` |
| Arbitrary strided 2-D view (`big[100:200, ::3]`) | `(a, b)` | `a` | `j·b` |
| Reversed rows (`arr[::-1]`) | `(−m·s, s)` | `−m·s` | `j·s` |
| Structured 1-D `(n,)` | `(itemsize,)` | `itemsize` | `fields[name].offset` |
| Strided structured view | `(k,)` | `k` | `fields[name].offset` |

So C-order, F-order, strided slices, reversed/negative-stride views, and structured fields are all served by the same code, zero-copy. numpy already computed the strides; `register_table` lifts `strides[0]`, `j*strides[1]` (2-D) or the field offsets (structured) into the descriptor. For a view, `arr.ctypes.data` already points at element `[0,0]` of the view (slice offset folded in). All quantities are **signed `int64`** so negative strides just work.

`register_table` accepts 2-D arrays and 1-D structured arrays only; everything else is rejected at registration with a clear `TypeError`/`ValueError`.

### 3.4 The static module and callbacks

One module-global `sqlite3_module` ctypes struct, `iVersion = 1`, built at import. `xCreate == xConnect` (eponymous, also CREATE-able) and `xDestroy == xDisconnect`; all write/transaction slots are `NULL`.

**The struct is positional — populate the ctypes fields in exact order, not in the table order below:**

```
iVersion, xCreate, xConnect, xBestIndex, xDisconnect, xDestroy, xOpen, xClose,
xFilter, xNext, xEof, xColumn, xRowid, xUpdate, xBegin, xSync, xCommit,
xRollback, xFindFunction, xRename
```

(everything from `xUpdate` onward = `NULL`). **Note `xNext` precedes `xEof`** in the struct — swapping those two function pointers would make SQLite call eof-test where it expects advance, silently hanging or mis-scanning every query.

**Embedded struct layouts** (64-bit; we write the fields, SQLite owns the base):

- *vtab* (32 bytes): base `sqlite3_vtab` = `{pModule(8), nRef(4)+pad(4), zErrMsg(8)}` = 24; **`descriptor_ptr` at +24**. `malloc(32)` in `xConnect`, `free` in `xDisconnect`.
- *cursor* (24 + `scratch_bytes`): `{pVtab(+0), descriptor_ptr(+8), rowid(+16, int64), scratch@+24}`. `xOpen` reads `descriptor` from `vtab+24`, writes `pVtab`/`descriptor`/`rowid=0`, and the `'U'` scratch lives in the same allocation. `free` in `xClose`.

SQLite does **not** populate `cursor->pVtab`, so `xOpen` sets it explicitly; copying `descriptor_ptr` into the cursor there also lets the hot callbacks reach the data with one indirection.

| callback | numba `@cfunc` signature | behavior |
|---|---|---|
| `xConnect` (= `xCreate`) | `int32(intp db, intp pAux, int32 argc, intp argv, intp ppVtab, intp pzErr)` | `descriptor = pAux`; `sqlite3_declare_vtab(db, schema_ptr)`; `malloc(32)` vtab; write `descriptor` at vtab+24; `*ppVtab = vtab`. On error: leave `*ppVtab` unset, set `*pzErr` to a `sqlite3_malloc`'d message, return the error code |
| `xBestIndex` | `int32(intp pVtab, intp pIdxInfo)` | trivial full scan — `return SQLITE_OK`. Safe because SQLite pre-initializes `estimatedCost`/`estimatedRows`/`aConstraintUsage` before each call, so writing nothing is valid across versions. (Optional polish: set `estimatedCost = estimatedRows = nrows` for join planning.) |
| `xDisconnect` (= `xDestroy`) | `int32(intp pVtab)` | `sqlite3_free(pVtab)` |
| `xOpen` | `int32(intp pVtab, intp ppCursor)` | `malloc(24 + scratch_bytes)`; init `{pVtab, descriptor, rowid=0}`; `*ppCursor = cur` |
| `xClose` | `int32(intp pCursor)` | `sqlite3_free(pCursor)` |
| `xFilter` | `int32(intp pCursor, int32 idxNum, intp idxStr, int32 argc, intp argv)` | `rowid = 0` (constraints ignored) |
| `xNext` | `int32(intp pCursor)` | `rowid += 1` |
| `xEof` | `int32(intp pCursor)` | returns a **boolean**: `1 if rowid >= nrows else 0` (nonzero = EOF; not a result code) |
| `xColumn` | `int32(intp pCursor, intp ctx, int32 j)` | compute `addr`, dispatch `col_tags[j]` → `sqlite3_result_*` |
| `xRowid` | `int32(intp pCursor, intp pRowid)` | `store_at(pRowid, int64(rowid))` (rowid carried as int64; 0-based array index — see §6) |

All cursor state is reached by raw-pointer walks (`load_at`/`store_at`/`carray`/`_cast_int_to_void_p`). No structrefs, no NRT.

### 3.5 Type mapping

`col_tags[j]` is an integer enum (`int8=0 … float64=9, bool=10, bytes('S')=11, unicode('U')=12`). `xColumn` dispatches:

| numpy dtype | SQL type | `xColumn` emits |
|---|---|---|
| int8/16/32/64, uint8/16/32/64, bool | `INTEGER` | widen/reinterpret → `sqlite3_result_int64` (one path for all integer tags; `uint64` ≥ 2⁶³ wraps to negative — doc note only, no special case) |
| float32/64 | `REAL` | `sqlite3_result_double` (NaN passes through, not NULL) |
| `'S'` (bytes) | `TEXT` (or `BLOB` if `text_as_blob`) | trim trailing NUL padding over `[addr, addr+width)` (interior NULs kept) → `sqlite3_result_text/blob(ctx, addr, n, SQLITE_TRANSIENT)` |
| `'U'` (UTF-32LE) | `TEXT` | `utf32_to_utf8(addr, width//4, scratch)` (code points = byte `width`//4; reads via `load_unaligned`) → `sqlite3_result_text(ctx, scratch, n, SQLITE_TRANSIENT)` |

`SQLITE_TRANSIENT` (= −1) tells SQLite to copy, so `addr`/`scratch` need not outlive the call. `'S'` needs no scratch (emit `addr` directly); only `'U'` uses the per-cursor scratch buffer.

### 3.6 Eponymous registration flow

`register_table(db, "points", arr)` → `sqlite3_create_module(db, "points", THE_MODULE, &descriptor)`. On the first `SELECT … FROM points`, SQLite calls `xConnect(db, pAux=&descriptor, …)`, which declares the schema and allocates the vtab. No `CREATE VIRTUAL TABLE` statement is needed; the module name *is* the table. Each `register_table` call registers one eponymous module (one name); they all share `THE_MODULE` and differ only in `pClientData`. `register_table` checks the `sqlite3_create_module` return code and raises on non-`SQLITE_OK` (e.g. a name already registered on this connection), mirroring `_raise_rc` in the phase-3 helpers.

### 3.7 Error handling and lifetime

- **cfunc exception safety (by callback category).** A raising `@cfunc` prints "Exception ignored" and returns the zero default — for a vtable callback that silently corrupts the scan — so any body that calls into C or holds an NRT reference is wrapped in a bare `try/except` (never `try/finally` — numba 0.65.1 reraise assertion, per [[reference_numba_cfunc_swallows_exceptions]]). The `except` action differs by the callback's return convention:
  - **Result-code callbacks** (`xConnect`/`xCreate`, `xBestIndex`, `xDisconnect`, `xOpen`, `xClose`, `xFilter`, `xNext`, `xRowid`): return `SQLITE_ERROR`. `xConnect`/`xCreate` report the message via `*pzErr` (a `sqlite3_malloc`'d string); other callbacks may set `pVtab->zErrMsg`.
  - **`xColumn`** (has `ctx`): call `sqlite3_result_error(ctx, …)` then return `SQLITE_ERROR`.
  - **`xEof`** returns a **boolean**, not a result code, and has no error channel; if wrapped at all, its `except` returns `1` (signal EOF to halt the scan). But `xEof` (`rowid >= nrows`) and `xNext` (`rowid += 1`) are pure scalar arithmetic — no call, no NRT ref — so they cannot raise or leak and need no wrapper. Wrap only callbacks that actually call C / touch NRT.
- **Lifetime.** `THE_MODULE` and the cfuncs are module-global (live forever); the `cfunc.address` values written into `THE_MODULE` are process-local and re-populated at import (numba re-fills the address even on a cache hit), so `THE_MODULE` must never be pickled/serialized across processes — each process builds its own at import. The returned `_VTableHandle` retains `arr`, the descriptor, the offset/tag/width numpy buffers, and the schema `bytes` — that is what keeps the data pointer valid for the connection's use of the table. **The caller MUST retain the handle** (same contract as `_UDAFHandle`); if it is GC'd the descriptor/array free and the next query reads freed memory.
- **Live window.** Because the table reads the array buffer directly, in-place value edits are visible to subsequent queries (a feature). `arr.resize()` / reallocation invalidates the snapshotted `data_base` and is disallowed without re-registering. A future `copy=True` flag could snapshot a frozen C-contiguous array.

## 4. New bindings and helpers

### 4.1 `signatures.py` (`signatures_sqlite`)

```python
"sqlite3_create_module": int32(intp, intp, intp, intp),   # (db, zName, *pModule, *pClientData) -> rc
"sqlite3_declare_vtab":  int32(intp, intp),               # (db, zCreateTableSQL) -> rc
"sqlite3_malloc":        intp(int32),                     # (n) -> ptr  (void *sqlite3_malloc(int))
```

`sqlite3_free : void(intp)` already exists (phase 1). The `sqlite3_result_*` setters and `sqlite3_value_*` accessors already exist (phase 2). `sqlite3_create_module_v2` is **not** needed: `pClientData` is owned by the handle, so no `xDestroy(void*)` is required. The `int` size argument to `sqlite3_malloc` is 32-bit on every supported platform (unlike `long`), so `int32` is correct with no per-platform dispatch, and `sqlite3_malloc64` is unnecessary for the small fixed allocations here.

### 4.2 `lowlevel.py` — `load_unaligned(p, ty)`

A `load_at` variant emitting `builder.load(ptr, align=1)`. Needed because numpy's default structured dtype is **packed**: `np.dtype([('a','i1'),('b','i8')])` places `b` at byte offset 1. A default-aligned LLVM `load` asserts a natural alignment the address does not have — **IR-level UB on every target** (and an actual fault on strict-alignment configs); `align=1` makes the load legal. Both `xColumn`'s field reads **and** `utf32_to_utf8`'s `uint32` code-point reads use `load_unaligned` (a packed `'U'` field is equally misaligned). 2-D arrays are naturally aligned, but using the unaligned load uniformly is harmless and simpler. This is the only place "non-contiguous/structured" costs anything — and it's from struct *packing*, not C-vs-F order.

### 4.3 `utf32_to_utf8(src_ptr, n_codepoints, dst_ptr) -> int32` (njit, internal to `_sqlite_vtable.py`)

`xColumn` passes `n_codepoints = col_widths[j] // 4` (the `'U'` field's byte width // 4). The encoder reads each code point with `load_unaligned(src_ptr + 4*i, uint32)` (not `carray`, which assumes alignment), trims trailing `0` code points (numpy `'U'` NUL-pads) while preserving interior NULs, encodes each as 1–4 UTF-8 bytes into `dst_ptr`, and returns the byte count. Invalid code points (surrogate range `0xD800–0xDFFF` or `> 0x10FFFF`) emit U+FFFD. `dst` is the per-cursor scratch, sized `scratch_bytes = max(width + 1)` over `'U'` columns. Little-endian matches numpy `'U'` on every CI platform. Unit-tested standalone, including a packed dtype whose `'U'` field sits at an odd byte offset.

## 5. Testing (`test/core/test_sqlite_vtable.py`)

All against a numbox `sqlite3_open` connection, comparing query results to direct numpy indexing:

1. **C-contiguous 2-D int64** — `SELECT *`, `COUNT(*)`, `WHERE x > k`, `ORDER BY`, `SUM`.
2. **C-contiguous 2-D float64** — values + `AVG`.
3. **F-contiguous** (`np.asfortranarray`) — identical results to the C case.
4. **Non-contiguous slice** (`big[::2, 1:4]`) — matches the equivalent numpy view.
5. **Reversed view** (`arr[::-1]`) — rows in reversed order.
6. **Mixed-width numeric 2-D** (int32, float32) — widening correctness.
7. **bool 2-D** — 0/1 integers.
8. **Structured array** with int + float + `'S'` + `'U'` fields — each column correct; `'U'` with non-ASCII (`"héllo"`, an emoji) round-trips to UTF-8; `'S'` NUL-trimmed.
9. **Packed/misaligned structured dtype** — including a `'U'` field at an odd byte offset — `load_unaligned` correctness on both numeric and `'U'` reads.
10. **`text_as_blob=True`** — `'S'` column typed `BLOB`, `typeof()` confirms.
11. **Empty table** (`nrows == 0`) — `SELECT` returns nothing; `COUNT(*) == 0`.
12. **`rowid`** — `SELECT rowid FROM t` equals the 0-based array index (`WHERE rowid = 0` selects row 0).
13. **JOIN** — register two arrays under two names; join them.
14. **Handle lifetime** — `gc.collect()` after `register_table`, query still works.
15. **`@cfunc(cache=True)`** — cold compile then warm reuse (cache dir does not grow), mirroring the phase-3 cache test.
16. **Duplicate name** — registering the same `name` twice on one connection raises.
17. **`utf32_to_utf8`** — standalone unit test over ASCII, multi-byte, 4-byte (emoji), and invalid code points.

Commands (per project convention): `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py --durations=20`, with `__pycache__` and numba cache cleared first; `/home/erik/projects/numbox/venv/bin/flake8 --max-line-length=127 .`.

## 6. Load-bearing gotchas

1. **`sqlite3_module` field order is positional** — populate the ctypes struct in the §3.4 field order, with `xNext` before `xEof`; the table in §3.4 is grouped for reading, not for struct population.
2. **`sqlite3_module` must persist for the connection's lifetime.** It is module-global, so this holds automatically. `pClientData` (the descriptor) must persist until the module is unregistered / connection closes — held by `_VTableHandle`.
3. **Caller must retain the handle.** SQLite holds raw pointers into the array buffer and the cfuncs; dropping the handle frees the array/descriptor and the next query reads freed memory.
4. **`col_widths` is field byte width; `'U'` derives code points = `width//4`.** Feeding the byte width as the encoder's `n_codepoints` would read 4× past the field — out-of-bounds. `scratch_bytes` is sized in bytes (`max(width+1)` over `'U'`).
5. **Packed structured fields are unaligned** → `load_unaligned` in both `xColumn` and `utf32_to_utf8` (IR-level UB otherwise, fault on strict-alignment configs). Ordinary C/F/strided 2-D arrays are aligned.
6. **`rowid` is 0-based** (the array index) — an intentional deviation from SQLite's usual 1-based rowids: `WHERE rowid = 0` selects the first row, `ORDER BY rowid` matches array order. Carried as `int64` so `xRowid` writes a full 8-byte `*pRowid`.
7. **NaN is a valid `REAL`, not NULL.** v1 has no NULL concept; document so callers do not expect `IS NULL` to match NaN. masked-array/NULL is a deferred feature.
8. **`uint64 ≥ 2⁶³` wraps** to a negative SQLite integer (SQLite integers are signed 64-bit). Same code path as other ints; documented.
9. **Live window, not a snapshot.** In-place edits show up; `resize`/reallocation must not happen on a registered array.
10. **`'U'` is little-endian UTF-32.** The encoder assumes LE (true on all CI platforms); a big-endian guard is a trivial future add.
11. **No struct-by-value across the cfunc boundary.** Every callback signature is `int32(intp, …)`/scalar, so the `abi.py` struct-classification machinery is not exercised — platform-clean on Windows/macOS/Linux without ABI special-casing.

## 7. Explicit follow-ups (phase 5+)

- **Writable tables** — `xUpdate` dispatch (INSERT/UPDATE/DELETE) over a mutable backing store.
- **Constraint pushdown** — real `xBestIndex`/`xFilter` reading `sqlite3_index_info` (`aConstraint`/`aConstraintUsage`), so `WHERE col = ?` filters in the cursor instead of post-scan. Enables table-valued functions with HIDDEN argument columns.
- **NULL semantics** — expose `numpy.ma` masks or a sentinel as SQL NULL.
- **More dtypes** — `datetime64`/`timedelta64` (with an epoch/type convention), complex, big-endian `'U'`.
- **`copy=True`** — snapshot into a frozen C-contiguous array for callers who want stable values.
- **Higher-level wrapper** — a `Connection`/`Statement` structref integration so `register_table` can take an ergonomic connection object (ties into the existing "higher-level structref wrappers" follow-up).

## 8. Public API reference

```python
def register_table(db, name, arr, columns=None, *, text_as_blob=False):
    """Expose a numpy array as a read-only eponymous SQLite virtual table.

    Parameters
    ----------
    db : int
        Raw sqlite3* connection pointer (the db_p.value filled in by numbox's
        sqlite3_open out-parameter; see the example below).
    name : str
        Table/module name; SELECT * FROM <name> works immediately.
    arr : numpy.ndarray
        A 2-D array (homogeneous dtype) or a 1-D structured array.
    columns : sequence[str] | None
        Column names. Required for a 2-D array; defaults to arr.dtype.names
        for a structured array (and may rename them in place).
    text_as_blob : bool
        Map 'S' (bytes) columns to BLOB instead of TEXT.

    Returns
    -------
    _VTableHandle
        Opaque handle that keeps the array + module state alive. The caller
        MUST retain it for as long as the table is queried. Raises if
        sqlite3_create_module returns non-SQLITE_OK (e.g. duplicate name).
    """
```

```python
import numpy as np
from ctypes import addressof, c_int64
from numbox.utils.cstrings import c_string
from numbox.core.bindings import sqlite3_open, register_table

# numbox's sqlite3_open is the raw 2-arg C binding: (filename_p, db_pp) -> rc.
db_p = c_int64(0)
with c_string(":memory:") as name_p:
    sqlite3_open(name_p, addressof(db_p))
db = db_p.value  # the sqlite3* connection pointer

arr = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int64)
h = register_table(db, "points", arr, columns=["id", "value"])
# SELECT value FROM points WHERE id >= 2 ORDER BY value DESC;  -> 30, 20

dt = np.dtype([("ticker", "U6"), ("qty", "i4"), ("px", "f8")])
trades = np.array([("AAPL", 100, 187.5), ("MSFT", 50, 412.25)], dtype=dt)
h2 = register_table(db, "trades", trades)
# SELECT ticker, qty*px AS notional FROM trades ORDER BY notional DESC;
```
