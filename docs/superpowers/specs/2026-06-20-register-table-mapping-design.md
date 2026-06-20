# Overload `register_table` to expose a dict of 1-D numpy arrays as a read-only SQLite virtual table

**Status:** design — approved 2026-06-20 (keep both query surfaces)
**Date:** 2026-06-20
**Branch:** `feat/register-table-mapping` (off `origin/main` @ `2036fae`; 0 behind `upstream/main` @ `af6bed2`, verified by fresh fetch)
**Scope:** numbox feature; **read-only**; follows the fork feature-branch → fork PR → (later) upstream PR workflow.

## Goal

Let a user expose a **columnar** dataset — a `Mapping[str, np.ndarray]` of equal-length 1-D arrays (column name → column) — as a read-only, zero-copy SQLite virtual table, by **overloading the existing `register_table`** to also accept a mapping. No new public function: the data argument's type selects the layout.

```python
cols = {"id": ids, "px": pxs}            # dict[str, 1-D np.ndarray], equal length
register_table(db, "trades", cols)       # queryable as: SELECT id FROM trades WHERE px > 100
```

## Background: what already exists

`numbox/core/bindings/_sqlite_vtable.py` provides
`register_table(db, name, arr, columns=None, *, text_as_blob=False)` — a read-only, eponymous
(queryable as `name` with no `CREATE VIRTUAL TABLE`), **zero-copy** vtable over a **row-major**
numpy structured array (or 2-D array). It already handles numeric / bool / fixed-width `'U'`/`'S'` /
BLOB via the shared `_sqlite_typemap`, eq/range constraint pushdown in `xBestIndex`, and the
value-semantics edge cases. A `dict` of 1-D arrays is the **column-major** counterpart:

| | today's `register_table` | this feature |
|---|---|---|
| Layout | structured array — **row-major**, one buffer | mapping of 1-D arrays — **column-major**, N buffers |
| Per column | offset within a shared row stride | its own base pointer + dtype + stride |

Row-major is a strict **special case** of column-major: `col_base[j] = data_base + col_offset[j]`,
`col_stride[j] = row_stride`. That is what makes one generalized descriptor serve both.

### Verified facts grounding this design

Established against HEAD `2036fae` by verification workflow `wi233exbx` and adversarial review
`wkbajmbzw` (read-only agents against the real code):

- **3 edit sites only.** The descriptor fields encoding layout (`data_base`, `row_stride`,
  `col_offsets`) are read in exactly three functions — `_cell_value_f64`, `_cell_value_i64`,
  `_xcolumn` — and written only in `_build_descriptor`. No other cfunc/njit consumer. `_xopen`
  reads `scratch_bytes`; `xeof`/`xnext`/`xrowid`/`xfilter`/`xbestindex` read `nrows`/`rowid`/
  `ncols`/`col_tags` only.
- **Descriptor itemsize 72 → 64** (recomputed independently below in Design §1).
- **Strided columns are correct, including `'S'`/`'U'`/BLOB.** Because the per-column stride drives
  the address math and each *cell* is internally contiguous, a non-contiguous 1-D column still hands
  SQLite a correct pointer+length under `SQLITE_STATIC`.
- **No docs `.rst` edit.** `_sqlite_vtable` is covered by an `automodule` directive
  (`docs/numbox.core.bindings.rst`), so the overloaded `register_table` docstring is picked up
  automatically. Rebuild only: `sphinx-build -b html docs/ docs/_build/html` (CI `docs.yml`), or
  locally `cd docs && /home/erik/projects/numbox/venv/bin/sphinx-build -b html . _build/html`.

## Interface (overload by data type)

```python
def register_table(db, name, data, columns=None, *, text_as_blob=False):
    """Expose tabular data as a read-only eponymous SQLite virtual table.

    `data` may be:
      - a 1-D numpy structured array (row-major; column names from the dtype,
        optionally renamed/reordered by `columns`);
      - a 2-D numpy array (row-major; `columns` required, one name per column);
      - a mapping (e.g. a dict) of name -> 1-D numpy array (columnar; all arrays
        one length; the keys name the columns, so passing `columns` raises
        ValueError -- rename/reorder by building the dict with the desired keys
        and order before calling).
    """
```

Dispatch:

```python
if isinstance(data, np.ndarray):
    desc = _build_descriptor(data, columns, text_as_blob)            # existing path, unchanged
elif isinstance(data, collections.abc.Mapping):
    if columns is not None:
        raise ValueError("columns must be omitted when data is a mapping; "
                         "the dict keys name the columns")
    desc = _build_descriptor_columnar(data, text_as_blob)            # new builder
else:
    raise TypeError("data must be a numpy.ndarray or a mapping of name -> 1-D array")
```

- The third parameter is renamed `arr` → `data` (it is no longer always an array). Every current
  caller passes it positionally, so nothing breaks; only a hypothetical `arr=` keyword caller would.
- **What counts as a mapping:** only a `collections.abc.Mapping` subclass (`dict`, `OrderedDict`,
  …). A numpy array — including a structured array, `recarray`, or `MaskedArray` — is an `np.ndarray`
  and takes the array path (dispatch order resolves this). A `pandas.DataFrame` is **not** a
  `Mapping` and is rejected by the `else` branch with `TypeError`; no DataFrame special-casing.
- **No coercion:** mapping *values* must already be `np.ndarray`. Lists, tensors, and other
  array-likes are rejected (see Validation), not converted.

## Design — Approach A: generalize the one module

Both layouts share **100%** of the cfuncs, the constraint pushdown, the typemap, the result
setters, the `sqlite3_create_module_v2` registration, the `_VTableHandle`/`_DATA_ANCHOR`/`xDestroy`
keep-alive, and the eponymous schema generation. Only the descriptor shape and the three
address-math sites change.

1. **Descriptor (`_DESC_DTYPE`).** Replace the three layout fields with two pointer fields, leaving
   the rest in place. Full new field list (declaration order, `align=True`):

   ```python
   _DESC_DTYPE = np.dtype([
       ("nrows", "i8"), ("ncols", "i4"),          # 8 + 4, then 4 pad (align=True)
       ("col_bases", "i8"), ("col_strides", "i8"),  # NEW: ptr -> int64[ncols] each
       ("col_tags", "i8"), ("col_widths", "i8"),
       ("schema_ptr", "i8"), ("scratch_bytes", "i8"),
   ], align=True)
   assert _DESC_DTYPE.itemsize == 64   # was 72: 8 + 4 + 4pad + 6*8 = 64
   ```

   (Removed: `row_stride`, `data_base`, `col_offsets` — 24 B → `col_bases`, `col_strides` — 16 B,
   net −8, so 72 → 64.)
2. **Address math (the 3 sites).** `addr = base + rowid*row_stride + offsets[col]` becomes
   `addr = col_bases[col] + rowid*col_strides[col]` in `_cell_value_f64`, `_cell_value_i64`,
   `_xcolumn`.
3. **`_build_descriptor` (arrays — today's path).** Compute the same per-column offsets it does now,
   then fill two buffers: `col_bases[j] = arr.ctypes.data + offs[j]`,
   `col_strides[j] = arr.strides[0]` (structured branch: offsets from dtype fields; 2-D branch:
   `offs[j] = j*strides[1]`). The address math is identical to today
   (`(arr.data + offs[col]) + rowid*row_stride`), so behavior is byte-for-byte equivalent — the
   56-test row-major suite is the proof.
4. **`_build_descriptor_columnar(mapping, text_as_blob)` (new).** Snapshot the column order once as
   `names = list(data.keys())`, and use that single ordering for both the schema string and the
   buffers. For each column `col_j`: `col_bases[j] = col_j.ctypes.data`;
   `col_strides[j] = col_j.strides[0]` (honors a non-contiguous 1-D column for free); tags/widths via
   `_col_tag(col_j.dtype, text_as_blob)`; `scratch_bytes` computed exactly as the array path. Build
   the eponymous schema from `names` (same quoting as today).
5. **Keep-alive (`_BuiltDescriptor` + `_VTableHandle`).** `register_table` keeps the registration
   alive with `handle = _VTableHandle(built)`, anchored in `_DATA_ANCHOR[client_ptr]` and released by
   SQLite's `xDestroy`. `_VTableHandle(*objs)` is variadic and `built` transitively holds every
   pointer the descriptor references, so no extra anchoring is needed. Generalize `_BuiltDescriptor`
   to a pure keep-alive with `__slots__ = ("c", "bases", "strides", "tags", "widths", "schema",
   "arrays")`, where `c` is the `_DESC_DTYPE` array (its `.ctypes.data` is the client pointer);
   `bases`/`strides`/`tags`/`widths` are the buffers the descriptor points into; `schema` is the
   schema bytes; and `arrays` is a tuple of the underlying data array(s) — `(arr,)` for the array
   paths, `tuple(columns)` for the mapping path. **Drop the current write-only scalar attributes**
   (`nrows`/`ncols`/`row_stride`/`scratch_bytes`/`offsets`/`arr`) — verified assigned-but-never-read
   on the Python side; the C descriptor `c` is the single source of truth. Removing `row_stride`
   touches both its `_DESC_DTYPE` field and its `_BuiltDescriptor` slot/assignment.

**Zero-copy contract (inherited):** queries read each column array's buffer directly; the caller must
not mutate or resize any column array while the table is registered.

## Query surfaces — read-driver-agnostic

Registration installs the module on the `db` handle; *any* SQL executed against that handle invokes
the vtab callbacks, regardless of which driver issues it. The user asked for **both** the stdlib and
the numpy/compiled surfaces; both are first-class and tested below (decision confirmed at review —
see "Resolved decisions").

1. **numbox C connection** — `db = sqlite3_open(":memory:")`, then `register_table(db, name, cols)`,
   then query via numbox's `sqlite3_prepare_v2`/`sqlite3_step`/`sqlite3_column_*` bindings (the
   `_fetchall` test idiom). This is the existing, tested path.

2. **stdlib `sqlite3.Connection`** — ergonomic Python rows:
   ```python
   conn = sqlite3.connect(":memory:")
   register_table(extract_connection_ptr(conn), "trades", cols)
   conn.execute("SELECT id FROM trades WHERE px > 100").fetchall()
   ```
   **New ground.** `numbox/utils/pysqlite_bridge.extract_connection_ptr(conn) -> int` returns the
   stdlib connection's C `sqlite3*` (validated: type check, library-version coordination, non-null,
   errmsg probe). No existing test combines `extract_connection_ptr` + `register_table` +
   `conn.execute`, so **this feature proves it** with a dedicated test (guarded by the
   `_libraries_coordinated()` → `pytest.skip` pattern). Documented caveats:
   - the connection must stay open while the table is registered (the pointer aliases
     connection-owned memory; a closed/GC'd `conn` dangles it);
   - macOS requires coordinated sqlite libraries (`DYLD_INSERT_LIBRARIES`) or the bridge raises and
     the test skips — Linux needs no workaround.

   *Rejected alternative:* opening our own `sqlite3*` instead of borrowing the connection's handle —
   a separate handle would not share the in-memory database, so `conn.execute` could not see the
   registered table, defeating the ergonomic goal.

3. **numpy structured array** via `query_to_array` — compiled materialization:
   ```python
   dtype = np.dtype([("id", "i8"), ("px", "f8")])
   with c_string("SELECT id, px FROM trades WHERE px > 100") as sql_p:
       out = query_to_array(db, sql_p, dtype)   # -> 1-D structured ndarray, one field per column
   ```
   `query_to_array(db, sql_p, dtype)` is a **Python** function whose row-materialization loop is
   `@njit`-compiled. It takes a **NUL-terminated SQL char pointer** (wrap the SQL in the `c_string`
   context manager, not a Python str) and a **required structured result dtype**, and returns a
   **1-D structured numpy array** (numeric *and* text/blob columns supported). It is invoked from
   Python; querying from *within* `@njit` is separately possible by calling the `sqlite3_step` /
   `sqlite3_column_*` njit bindings directly.

## Value semantics (inherited, unchanged)

NaN `REAL` → SQL NULL (SQLite `sqlite3IsNaN`); `uint64` ≥ 2⁶³ stored as signed INTEGER (wraps
negative); fixed-width `'U'`/`'S'` trailing-NUL trimmed on read, interior NUL preserved; `'S'` is
TEXT by default, `text_as_blob=True` stores raw bytes as BLOB. numpy has no missing value, so —
**for the dtypes this feature supports** — every non-NaN cell is non-NULL. **Masked arrays are not
supported:** an `np.ma.MaskedArray` dispatches to the array path as a plain `np.ndarray` and its
mask is silently ignored (masked cells surface their underlying values, not SQL NULL). Same typemap
and result setters as the row-major path.

## Validation (build-time, before any C registration)

For the mapping path:
- **empty mapping** (`len(data) == 0`, no columns) → `ValueError`. A mapping with ≥1 column and 0
  rows is a valid 0-row table and is allowed (consistent with a 0-row structured array). A 1-column
  mapping is valid.
- a value that is **not an `np.ndarray`** → `TypeError` (e.g. `"column 'x' must be a 1-D numpy
  array, got list"`); no coercion.
- a value that **is an `np.ndarray` but `ndim != 1`** → `ValueError` (e.g. `"column 'x' must be 1-D,
  got ndim=2"`).
- columns of **unequal length** → `ValueError`.
- **unsupported dtype** → the existing `_col_tag` rejection path (reused, not reinvented). Supported
  dtypes are exactly those `_col_tag` accepts: int8–int64, uint8–uint64, float32, float64, bool,
  fixed-width unicode (`'U<n>'`), fixed-width bytes (`'S<n>'`). Object/variable-length dtypes are
  rejected.
- `columns` provided alongside a mapping → `ValueError` (keys already name the columns).

## Testing

- **New columnar tests** mirroring the row-major 4-step idiom (`_open_memory` / build dict /
  `register_table` / `_fetchall` assert / `sqlite3_close`), reusing the existing `_fetchall` helper:
  per-dtype round-trips; mixed-dtype multi-column; eq/range pushdown; strings (`'U'`/`'S'`, NUL-trim,
  interior NUL, `text_as_blob`); `uint64` wrap; NaN→NULL; join of two columnar tables; a **strided
  (non-contiguous) column**; and the mapping-specific error cases — non-`np.ndarray` value
  (`TypeError`), `ndim != 1` (`ValueError`), unequal lengths (`ValueError`), empty mapping
  (`ValueError`), `columns`-with-mapping (`ValueError`).
- **Surface tests:** one **stdlib-`Connection`** test (`extract_connection_ptr` + `register_table` +
  `conn.execute(...).fetchall()`, with a `_libraries_coordinated()` skip guard **copied into the new
  test module** — matching the per-file pattern in `test/utils/test_pysqlite_bridge.py`, avoiding a
  cross-cutting conftest change); one **`query_to_array`** test (SQL via the `c_string` context
  manager, a structured result dtype, asserting on the returned 1-D structured ndarray).
- **Regression guard:** the full existing **56-test** row-major suite must stay green — the proof the
  generalization is backward-compatible.
- Clean `__pycache__` and the numba cache (`~/.cache/numba`) before each pytest run, since the
  `_DESC_DTYPE` change invalidates the cfunc cache (clear via
  `venv/bin/python -c "import shutil, pathlib; ..."`).
- Lint: `flake8` at `max-line-length=127` (matches the project `.flake8`; this takes precedence over
  the global 120 default for this repo). Docs: `sphinx-build` exit 0 (no `.rst` edit).
- Run pytest/flake8 with the venv-absolute interpreter (`/home/erik/projects/numbox/venv/bin/...`).

## Boundaries / out of scope

- No writes (INSERT/UPDATE/DELETE), no row insert/delete, no array resizing.
- No object/variable-length-string (`dtype=object`) columns — fixed-width `'U'`/`'S'` only.
- No masked-array support (mask ignored, per Value semantics).
- No change to `query_to_array`'s signature or to the `xBestIndex` pushdown logic.
- No `CLAUDE.md` "2-D float64" correction here — that line's description of `query_to_array` is stale
  (it returns a structured array); fixing it is a separate, flagged follow-up.

## Resolved decisions

- **Keep both query surfaces in this PR** (decided 2026-06-20). A scope review suggested deferring
  surfaces **2 (stdlib `Connection`)** and **3 (`query_to_array`)** to a follow-up, since they
  integrate pre-existing infrastructure and the stdlib test needs the
  `_libraries_coordinated()`/macOS-skip seam. Both are kept: they add *new* coverage (the columnar
  vtable under each driver; the stdlib registration path is untested anywhere today), and both were
  explicitly requested.

## Workflow / integration

Develop on fork feature-branch `feat/register-table-mapping` (off `origin/main`, has CLAUDE.md +
fork CI). Open a **fork PR first** so the fork bots review. On approval, cherry-pick to an
`upstream-pr/*` branch based off `upstream/main` (excludes CLAUDE.md, `docs/superpowers/**`, and the
fork-only CI matrix expansions), then open the upstream PR — only with explicit per-PR user consent.
Per the numbox `CLAUDE.md` conventions.
