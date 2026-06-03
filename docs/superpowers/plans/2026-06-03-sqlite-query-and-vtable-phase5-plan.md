# SQLite `query_to_array` + virtual-table phase 5 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `query_to_array` (SELECT → numpy structured array) plus virtual-table phase 5 (constraint pushdown, table-valued functions, and `create_module_v2`/`xDestroy` cleanup) to numbox's SQLite bindings.

**Architecture:** Extract the shared dtype-tag ↔ SQLite type-map into a new `_sqlite_typemap.py`; add `query_to_array` in a new `_sqlite_query.py` that writes result cells into a caller-dtype'd numpy buffer by offset/tag dispatch (`xColumn` in reverse); extend the existing read-only vtable in `_sqlite_vtable.py` with `xBestIndex`/`xFilter` pushdown, a `register_tvf` mechanism whose rows are a per-query NRT array computed from hidden-column args, and SQLite-driven cleanup via `create_module_v2` + an `xDestroy` callback that pops a module-level registry.

**Tech Stack:** numba 0.65.1 (`@njit`/`@cfunc`/`carray`/`proxy`), numpy 2.4.5 structured dtypes, SQLite C API via `numbox.core.bindings`, ctypes for the module struct + the `xDestroy` callback.

**Design source:** `docs/superpowers/specs/2026-06-03-sqlite-query-and-vtable-phase5-design.md` (four load-bearing assumptions verified by runnable spikes before this plan; their findings are folded in).

**Conventions (all tasks):**
- Run python/pytest/flake8 via the venv: `/home/erik/projects/numbox/venv/bin/{python,pytest,flake8}`.
- Clean caches before every pytest run: `/home/erik/projects/numbox/venv/bin/python -c "import shutil,pathlib; [shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home()/'.cache'/'numba',ignore_errors=True)"`
- flake8 config is `.flake8` (max-line-length=127). New public symbols reach the package via the `import *` lines in `numbox/core/bindings/__init__.py`; keep helpers `_`-prefixed.
- cfunc bodies that call user code or do pointer work wrap the body in a bare `try/except` and report via `sqlite3_result_error` (context callbacks) or a return code (per the phase-2/3 precedent in `_sqlite_udf_helpers.py` and `_xcolumn`).
- Commit after each task. Commit messages describe the change only — attribute to the user, never mention tooling.

---

### Task 0: Extract shared type-map into `_sqlite_typemap.py`

**Goal:** Move the dtype-tag constants, tag/SQL-type maps, and the UTF-32→UTF-8 + NUL-trim helpers out of `_sqlite_vtable.py` into a new shared module, with `_sqlite_vtable.py` re-importing them so nothing else breaks. Pure refactor, no behavior change.

**Files:**
- Create: `numbox/core/bindings/_sqlite_typemap.py`
- Modify: `numbox/core/bindings/_sqlite_vtable.py:45-49,110-175` (remove the moved definitions, import them back)
- Test: existing `test/core/test_sqlite_vtable.py` must still pass unchanged (it imports `utf32_to_utf8`, `_nul_trimmed_len`, `_TAG_I64`, `_TAG_F64`, `_TAG_U` from `_sqlite_vtable`).

**Acceptance Criteria:**
- [ ] `_sqlite_typemap.py` defines `_TAG_*`, `_NUMERIC_TAGS`, `_SQL_TYPE`, `_col_tag`, `utf32_to_utf8`, `_nul_trimmed_len`.
- [ ] `_sqlite_vtable.py` imports those names from `_sqlite_typemap` and is otherwise behavior-identical.
- [ ] `from numbox.core.bindings._sqlite_vtable import utf32_to_utf8, _nul_trimmed_len, _TAG_I64, _TAG_F64, _TAG_U` still works (re-export).
- [ ] Full suite green; `flake8` clean.

**Verify:** `/home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_vtable.py -q` → all pass.

**Steps:**

- [ ] **Step 1: Create `_sqlite_typemap.py`** with the verbatim definitions moved from `_sqlite_vtable.py` (lines 45-49 tags, 110-175 helpers + maps). Header:

```python
"""Shared numpy-dtype ↔ SQLite type mapping + fixed-width string helpers.

Used by the read-only vtable (_sqlite_vtable), the table-valued-function
mechanism, and query_to_array. The dtype tags are the single source of truth
for how a numpy column maps to a SQLite column type and how its bytes are
read/written.
"""
import numpy as np
from numba import carray, njit
from numba.core.types import uint8, uint32

from numbox.core.configurations import jit_options
from numbox.utils.lowlevel import _cast_int_to_void_p, load_unaligned

# dtype tags (col_tags[j])
_TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64 = 0, 1, 2, 3
_TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64 = 4, 5, 6, 7
_TAG_F32, _TAG_F64, _TAG_BOOL = 8, 9, 10
_TAG_S, _TAG_U, _TAG_BLOB = 11, 12, 13

_NUMERIC_TAGS = {
    np.dtype("int8"): _TAG_I8, np.dtype("int16"): _TAG_I16,
    np.dtype("int32"): _TAG_I32, np.dtype("int64"): _TAG_I64,
    np.dtype("uint8"): _TAG_U8, np.dtype("uint16"): _TAG_U16,
    np.dtype("uint32"): _TAG_U32, np.dtype("uint64"): _TAG_U64,
    np.dtype("float32"): _TAG_F32, np.dtype("float64"): _TAG_F64,
    np.dtype("bool"): _TAG_BOOL,
}
_SQL_TYPE = {
    _TAG_I8: "INTEGER", _TAG_I16: "INTEGER", _TAG_I32: "INTEGER", _TAG_I64: "INTEGER",
    _TAG_U8: "INTEGER", _TAG_U16: "INTEGER", _TAG_U32: "INTEGER", _TAG_U64: "INTEGER",
    _TAG_BOOL: "INTEGER", _TAG_F32: "REAL", _TAG_F64: "REAL",
    _TAG_S: "TEXT", _TAG_U: "TEXT", _TAG_BLOB: "BLOB",
}


def _col_tag(dt, text_as_blob):
    if dt.kind == "S":
        return _TAG_BLOB if text_as_blob else _TAG_S
    if dt.kind == "U":
        return _TAG_U
    if dt in _NUMERIC_TAGS:
        return _NUMERIC_TAGS[dt]
    raise TypeError("unsupported column dtype %r" % (dt,))
```

Then paste `_nul_trimmed_len` and `utf32_to_utf8` verbatim from `_sqlite_vtable.py:110-149` (they already only depend on `carray`, `_cast_int_to_void_p`, `load_unaligned`, `uint8`, `uint32`, `np`, `njit`, `jit_options` — all imported above).

- [ ] **Step 2: Strip the moved code from `_sqlite_vtable.py`** (delete lines 45-49, 110-175 and the now-unused `_NUMERIC_TAGS`/`_SQL_TYPE`/`_col_tag`), and add the import near the top (after the existing `from numbox.core.bindings import (...)` block):

```python
from numbox.core.bindings._sqlite_typemap import (
    _TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64, _TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64,
    _TAG_F32, _TAG_F64, _TAG_BOOL, _TAG_S, _TAG_U, _TAG_BLOB,
    _NUMERIC_TAGS, _SQL_TYPE, _col_tag, utf32_to_utf8, _nul_trimmed_len,
)
```

Remove now-unused imports from `_sqlite_vtable.py` only if they became unused (e.g. `load_unaligned` is still used by `_xcolumn`, keep it; `uint8`/`uint32` still used — keep). Run flake8 to confirm no F401.

- [ ] **Step 3: Run the suite to confirm no behavior change.**

Run: `/home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_vtable.py -q`
Expected: same pass count as before the refactor.

- [ ] **Step 4: flake8.**

Run: `/home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_sqlite_typemap.py numbox/core/bindings/_sqlite_vtable.py --max-line-length=127`
Expected: no output (clean).

- [ ] **Step 5: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_typemap.py numbox/core/bindings/_sqlite_vtable.py
git -C /home/erik/projects/numbox commit -m "refactor(sqlite): extract shared dtype/type-map into _sqlite_typemap"
```

---

### Task 1: Add `sqlite3_create_module_v2` signature + `SQLITE_INDEX_CONSTRAINT_*` constants

**Goal:** Bind `sqlite3_create_module_v2` and add the index-constraint op-code constants that pushdown/TVF branch on.

**Files:**
- Modify: `numbox/core/bindings/signatures.py:212` (next to `sqlite3_create_module`)
- Modify: `numbox/core/bindings/_sqlite_constants.py` (`__all__` list + new assignments)
- Test: `test/core/test_sqlite_constants.py` (add op-code presence asserts) or extend `test_sqlite_vtable.py`.

**Acceptance Criteria:**
- [ ] `signatures["sqlite3_create_module_v2"] == int32(intp, intp, intp, intp, intp)`.
- [ ] `SQLITE_INDEX_CONSTRAINT_EQ == 2`, `_GT == 4`, `_LE == 8`, `_LT == 16`, `_GE == 32`, `_NE == 68`, `_ISNULL == 71`, `_IS == 72` are importable from `numbox.core.bindings`.

**Verify:** `/home/erik/projects/numbox/venv/bin/python -c "from numbox.core.bindings import SQLITE_INDEX_CONSTRAINT_EQ; from numbox.core.bindings.signatures import signatures; print(SQLITE_INDEX_CONSTRAINT_EQ, signatures['sqlite3_create_module_v2'])"` → `2 int32(int64, int64, int64, int64, int64)`

**Steps:**

- [ ] **Step 1: Add the signature.** In `signatures.py`, directly after the `"sqlite3_create_module": int32(intp, intp, intp, intp),` line add:

```python
    "sqlite3_create_module_v2": int32(intp, intp, intp, intp, intp),
```

- [ ] **Step 2: Add the constants.** In `_sqlite_constants.py`, append to `__all__` the eight names, then add the assignments (values per <https://www.sqlite.org/c3ref/c_index_constraint_eq.html>):

```python
SQLITE_INDEX_CONSTRAINT_EQ = 2
SQLITE_INDEX_CONSTRAINT_GT = 4
SQLITE_INDEX_CONSTRAINT_LE = 8
SQLITE_INDEX_CONSTRAINT_LT = 16
SQLITE_INDEX_CONSTRAINT_GE = 32
SQLITE_INDEX_CONSTRAINT_NE = 68
SQLITE_INDEX_CONSTRAINT_ISNULL = 71
SQLITE_INDEX_CONSTRAINT_IS = 72
```

- [ ] **Step 3: Test + commit.**

```python
# in test/core/test_sqlite_constants.py
def test_index_constraint_ops():
    from numbox.core.bindings import (
        SQLITE_INDEX_CONSTRAINT_EQ, SQLITE_INDEX_CONSTRAINT_GE, SQLITE_INDEX_CONSTRAINT_LT)
    assert (SQLITE_INDEX_CONSTRAINT_EQ, SQLITE_INDEX_CONSTRAINT_GE, SQLITE_INDEX_CONSTRAINT_LT) == (2, 32, 16)
```

Run: `/home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_constants.py -q` → pass. Then commit.

---

### Task 2: `utf8_to_utf32` decoder in `_sqlite_typemap.py`

**Goal:** Add the inverse of `utf32_to_utf8` — decode a UTF-8 byte buffer into a fixed-width UTF-32 (`'U'`) numpy field — for `query_to_array`'s TEXT→`'U'` columns.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_typemap.py`
- Test: `test/core/test_sqlite_query.py` (new)

**Acceptance Criteria:**
- [ ] `utf8_to_utf32(src_p, nbytes, dst_p, width_cp)` writes up to `width_cp` code points (uint32 LE) at `dst_p`, NUL-padding the remainder, and returns the number of code points written (≤ `width_cp`).
- [ ] ASCII, 2/3/4-byte sequences decode correctly; truncated/over-long input is clamped to `width_cp`; round-trips with `utf32_to_utf8`.

**Verify:** `/home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_query.py -k utf8_to_utf32 -q` → pass.

**Steps:**

- [ ] **Step 1: Write failing tests** in `test/core/test_sqlite_query.py`:

```python
import numpy as np
from numbox.core.bindings._sqlite_typemap import utf8_to_utf32
from numbox.utils.lowlevel import array_data_p


def _decode(s_bytes, width_cp):
    src = np.frombuffer(s_bytes, dtype=np.uint8).copy()
    dst = np.zeros(width_cp, dtype=np.uint32)
    n = utf8_to_utf32(array_data_p(src), len(src), array_data_p(dst), width_cp)
    return n, dst


def test_utf8_to_utf32_ascii():
    n, dst = _decode(b"abc", 6)
    assert n == 3 and list(dst[:3]) == [97, 98, 99] and dst[3] == 0


def test_utf8_to_utf32_multibyte():
    n, dst = _decode("héllo".encode("utf-8"), 6)
    assert list(dst[:n]) == [ord(c) for c in "héllo"]


def test_utf8_to_utf32_4byte():
    n, dst = _decode("a\U0001F600b".encode("utf-8"), 6)
    assert [int(x) for x in dst[:n]] == [ord("a"), 0x1F600, ord("b")]


def test_utf8_to_utf32_clamps_to_width():
    n, dst = _decode(b"abcdefgh", 3)
    assert n == 3 and list(dst[:3]) == [97, 98, 99]
```

Run: expect FAIL (`utf8_to_utf32` undefined).

- [ ] **Step 2: Implement `utf8_to_utf32`** in `_sqlite_typemap.py` (mirror the structure of `utf32_to_utf8`; decode UTF-8, stop at `width_cp` code points or `nbytes`, write uint32 LE via the `out` carray, NUL-pad the tail):

```python
@njit(**jit_options)
def utf8_to_utf32(src, nbytes, dst, width_cp):
    """Decode the UTF-8 bytes at ``src`` (length ``nbytes``) into up to
    ``width_cp`` little-endian uint32 code points at ``dst``; NUL-pad the
    remainder. Returns the number of code points written."""
    inp = carray(_cast_int_to_void_p(src), (nbytes,), dtype=np.uint8)
    out = carray(_cast_int_to_void_p(dst), (width_cp,), dtype=np.uint32)
    for k in range(width_cp):
        out[k] = 0
    i = 0
    k = 0
    while i < nbytes and k < width_cp:
        b0 = uint32(inp[i])
        if b0 < 0x80:
            cp = b0
            i += 1
        elif b0 >> 5 == 0x6 and i + 1 < nbytes:
            cp = ((b0 & 0x1F) << 6) | (uint32(inp[i + 1]) & 0x3F)
            i += 2
        elif b0 >> 4 == 0xE and i + 2 < nbytes:
            cp = ((b0 & 0x0F) << 12) | ((uint32(inp[i + 1]) & 0x3F) << 6) | (uint32(inp[i + 2]) & 0x3F)
            i += 3
        elif b0 >> 3 == 0x1E and i + 3 < nbytes:
            cp = ((b0 & 0x07) << 18) | ((uint32(inp[i + 1]) & 0x3F) << 12) \
                | ((uint32(inp[i + 2]) & 0x3F) << 6) | (uint32(inp[i + 3]) & 0x3F)
            i += 4
        else:
            cp = 0xFFFD
            i += 1
        out[k] = cp
        k += 1
    return k
```

- [ ] **Step 3: Run the tests** → PASS. **Step 4: flake8. Step 5: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_typemap.py test/core/test_sqlite_query.py
git -C /home/erik/projects/numbox commit -m "feat(sqlite): utf8_to_utf32 decoder for fixed-width unicode columns"
```

---

### Task 3: `query_to_array(db, sql, dtype)`

**Goal:** Run a `SELECT` and collect result rows into a new numpy structured array of the caller-supplied `dtype`, by writing each cell at its field offset (`xColumn` in reverse), with geometric NRT growth and a final trim.

**Files:**
- Create: `numbox/core/bindings/_sqlite_query.py`
- Modify: `numbox/core/bindings/__init__.py` (add `from numbox.core.bindings._sqlite_query import *`)
- Test: `test/core/test_sqlite_query.py`

**Acceptance Criteria:**
- [ ] `query_to_array(db, sql, dtype)` returns a 1-D structured array of `dtype` with one row per result row, columns by position.
- [ ] Field count ≠ `sqlite3_column_count` → raises `ValueError`.
- [ ] Numeric/text(`'U'`)/bytes(`'S'`) columns round-trip; SQL `NULL` → `NaN` (float) / `0` (int) / empty (text/blob).
- [ ] Correct across the growth boundary (e.g. 5000 rows from an initial cap of 16); empty result → length-0 array.
- [ ] **A2 cache guard:** two distinct dtypes queried in one process return correctly-laid-out arrays (no stale-cache cross-contamination).

**Verify:** `/home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_query.py -q` → all pass.

**Design notes (carry into the steps):**
- The jitted core takes the output dtype as a numba **typeref argument** (`numba.from_dtype(dtype)`), used in `np.empty(cap, dt)` — this puts the full Record layout into numba's cache key (per the spec's A2 finding); never read a module-global dtype for the allocation.
- Per-column `(offset, tag, width)` arrays are built in Python from `dtype` (reuse `_col_tag` from the type-map) and passed as numpy arrays.
- Writing a cell: `addr = out_data + row*itemsize + offsets[j]`; dispatch on `tag` and `store_at(addr, <typed value>)` for numerics (mirror `_xcolumn`'s tag ladder in reverse, using `store_at` from `lowlevel`); for `'U'` decode via `utf8_to_utf32(text_p, nbytes, addr, width//4)`; for `'S'`/blob copy `min(nbytes, width)` bytes then NUL-pad.
- Read each column with `sqlite3_column_type` (to detect `NULL`) then the typed accessor (`sqlite3_column_int64`/`_double`/`_text`+`_bytes`/`_blob`+`_bytes`).

**Steps:**

- [ ] **Step 1: Write failing tests** in `test/core/test_sqlite_query.py` (append to the file from Task 2). Reuse the open/exec harness style from `test_sqlite_vtable.py` (`sqlite3_open` into a `c_int64` slot, `sqlite3_exec`/prepared statements to populate, then `query_to_array`):

```python
from ctypes import addressof, c_int64
from numbox.core.bindings import sqlite3_open, sqlite3_close, query_to_array, sqlite3_exec
from numbox.utils.cstrings import c_string


def _open_mem():
    db = c_int64(0)
    with c_string(":memory:") as p:
        assert sqlite3_open(p, addressof(db)) == 0
    return db.value


def _exec(db, sql):
    with c_string(sql) as p:
        assert sqlite3_exec(db, p, 0, 0, 0) == 0


def test_query_numeric_roundtrip():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(i INTEGER, x REAL)")
    _exec(db, "INSERT INTO t VALUES (1, 1.5), (2, 2.5), (3, 3.5)")
    dt = np.dtype([("i", "i8"), ("x", "f8")])
    with c_string("SELECT i, x FROM t ORDER BY i") as sql:
        out = query_to_array(db, sql, dt)
    assert out.shape == (3,)
    assert list(out["i"]) == [1, 2, 3]
    assert list(out["x"]) == [1.5, 2.5, 3.5]
    sqlite3_close(db)


def test_query_null_coercion():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(i INTEGER, x REAL)")
    _exec(db, "INSERT INTO t VALUES (NULL, NULL)")
    dt = np.dtype([("i", "i8"), ("x", "f8")])
    with c_string("SELECT i, x FROM t") as sql:
        out = query_to_array(db, sql, dt)
    assert out["i"][0] == 0 and np.isnan(out["x"][0])
    sqlite3_close(db)


def test_query_growth_boundary():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(i INTEGER)")
    _exec(db, "WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n<5000) INSERT INTO t SELECT n FROM c")
    dt = np.dtype([("i", "i8")])
    with c_string("SELECT i FROM t ORDER BY i") as sql:
        out = query_to_array(db, sql, dt)
    assert out.shape == (5000,) and out["i"][0] == 1 and out["i"][4999] == 5000
    sqlite3_close(db)


def test_query_text_roundtrip():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(s TEXT)")
    _exec(db, "INSERT INTO t VALUES ('hi'), ('héllo')")
    dt = np.dtype([("s", "U8")])
    with c_string("SELECT s FROM t") as sql:
        out = query_to_array(db, sql, dt)
    assert out["s"][0] == "hi" and out["s"][1] == "héllo"
    sqlite3_close(db)


def test_query_field_count_mismatch_raises():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(a INTEGER, b INTEGER)")
    with c_string("SELECT a, b FROM t") as sql:
        with pytest.raises(ValueError):
            query_to_array(db, sql, np.dtype([("a", "i8")]))
    sqlite3_close(db)


def test_query_two_dtypes_no_stale_cache():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(a INTEGER, b INTEGER)")
    _exec(db, "INSERT INTO t VALUES (1, 2)")
    with c_string("SELECT a, b FROM t") as sql:
        o1 = query_to_array(db, sql, np.dtype([("a", "i8"), ("b", "i8")]))
        o2 = query_to_array(db, sql, np.dtype([("a", "i4"), ("b", "i4")]))
    assert o1.dtype.itemsize == 16 and o2.dtype.itemsize == 8
    assert tuple(o2[0]) == (1, 2)
    sqlite3_close(db)
```

Run: expect FAIL (`query_to_array`, `sqlite3_exec` import — note `sqlite3_exec` is already bound in `_sqlite_exec`).

- [ ] **Step 2: Implement the jitted core + public wrapper** in `_sqlite_query.py`. The core takes the dtype typeref `dt`, the column metadata arrays, and the prepared-statement helpers. Outline (fill the numeric tag ladder mirroring `_xcolumn` in reverse, using `store_at`):

```python
"""query_to_array: collect SELECT results into a numpy structured array."""
import numpy as np
import numba
from numba import njit, types
from numba.core.types import (
    int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64,
)

from numbox.core.bindings._sqlite_constants import SQLITE_ROW, SQLITE_NULL, SQLITE_OK
from numbox.core.bindings import (
    sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize, sqlite3_column_count,
    sqlite3_column_type, sqlite3_column_int64, sqlite3_column_double,
    sqlite3_column_text, sqlite3_column_blob, sqlite3_column_bytes, sqlite3_errmsg,
)
from numbox.core.bindings._sqlite_typemap import (
    _col_tag, utf8_to_utf32,
    _TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64, _TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64,
    _TAG_F32, _TAG_F64, _TAG_BOOL, _TAG_S, _TAG_U, _TAG_BLOB,
)
from numbox.core.configurations import jit_options
from numbox.utils.lowlevel import _cast_int_to_void_p, array_data_p, store_at
import ctypes

__all__ = ["query_to_array"]


@njit(**jit_options)
def _store_cell(out_data, addr_off, tag, width, stmt, j):
    """Read column j of the current row and write it at out_data+addr_off."""
    ctype = sqlite3_column_type(stmt, j)
    addr = out_data + addr_off
    if ctype == SQLITE_NULL:
        # numeric fields already zeroed by np.empty? No -- np.empty is
        # uninitialized; the core zeroes each fresh row before _store_cell.
        if tag == _TAG_F32:
            store_at(addr, float32(np.nan))
        elif tag == _TAG_F64:
            store_at(addr, float64(np.nan))
        return
    if tag == _TAG_I8:
        store_at(addr, int8(sqlite3_column_int64(stmt, j)))
    elif tag == _TAG_I16:
        store_at(addr, int16(sqlite3_column_int64(stmt, j)))
    elif tag == _TAG_I32:
        store_at(addr, int32(sqlite3_column_int64(stmt, j)))
    elif tag == _TAG_I64:
        store_at(addr, int64(sqlite3_column_int64(stmt, j)))
    elif tag == _TAG_U8:
        store_at(addr, uint8(sqlite3_column_int64(stmt, j)))
    elif tag == _TAG_U16:
        store_at(addr, uint16(sqlite3_column_int64(stmt, j)))
    elif tag == _TAG_U32:
        store_at(addr, uint32(sqlite3_column_int64(stmt, j)))
    elif tag == _TAG_U64:
        store_at(addr, uint64(sqlite3_column_int64(stmt, j)))
    elif tag == _TAG_BOOL:
        store_at(addr, uint8(1) if sqlite3_column_int64(stmt, j) != 0 else uint8(0))
    elif tag == _TAG_F32:
        store_at(addr, float32(sqlite3_column_double(stmt, j)))
    elif tag == _TAG_F64:
        store_at(addr, float64(sqlite3_column_double(stmt, j)))
    elif tag == _TAG_U:
        text_p = sqlite3_column_text(stmt, j)
        nbytes = sqlite3_column_bytes(stmt, j)
        utf8_to_utf32(text_p, nbytes, addr, width // 4)
    else:  # _TAG_S / _TAG_BLOB
        if tag == _TAG_S:
            src = sqlite3_column_text(stmt, j)
        else:
            src = sqlite3_column_blob(stmt, j)
        nbytes = sqlite3_column_bytes(stmt, j)
        dst = carray(_cast_int_to_void_p(addr), (width,), dtype=np.uint8)
        n = nbytes if nbytes < width else width
        srcb = carray(_cast_int_to_void_p(src), (n,), dtype=np.uint8)
        for b in range(n):
            dst[b] = srcb[b]
        for b in range(n, width):
            dst[b] = 0
```

```python
@njit(**jit_options)
def _query_core(stmt, ncols, offsets, tags, widths, itemsize, dt):
    cap = 16
    out = np.empty(cap, dt)
    n = 0
    raw = out.view(np.uint8)            # flat byte view for per-cell offset writes
    while sqlite3_step(stmt) == SQLITE_ROW:
        if n == cap:
            cap = cap * 2
            new = np.empty(cap, dt)
            nraw = new.view(np.uint8)
            for b in range(n * itemsize):
                nraw[b] = raw[b]
            out = new
            raw = nraw
        base = array_data_p(out) + n * itemsize
        for j in range(ncols):
            _store_cell(base, offsets[j], tags[j], widths[j], stmt, j)
        n += 1
    res = np.empty(n, dt)
    rraw = res.view(np.uint8)
    raw = out.view(np.uint8)
    for b in range(n * itemsize):
        rraw[b] = raw[b]
    return res
```

> Note for the implementer: confirm during TDD whether `out.view(np.uint8)` is supported on a structured array inside `@njit` on numba 0.65.1; if not, take the byte view via `carray(_cast_int_to_void_p(array_data_p(out)), (cap*itemsize,), np.uint8)` instead (the spike for A2 used a fresh-`np.empty`+copy trim, which this mirrors). Either way the trim is a fresh `np.empty(n, dt)` + byte copy, returning an owned array.

```python
def query_to_array(db, sql_p, dtype):
    """Run the prepared SQL text at ``sql_p`` on ``db`` and return its rows as a
    1-D numpy structured array of ``dtype`` (one field per result column, by
    position). NULL -> NaN (float) / 0 (int) / empty (text/blob)."""
    if dtype.fields is None or dtype.names is None:
        raise TypeError("dtype must be a structured numpy dtype, got %r" % (dtype,))
    names = list(dtype.names)
    offsets = np.array([dtype.fields[nm][1] for nm in names], dtype=np.int64)
    subs = [dtype.fields[nm][0] for nm in names]
    tags = np.array([_col_tag(s, False) for s in subs], dtype=np.int64)
    widths = np.array([int(s.itemsize) for s in subs], dtype=np.int64)
    stmt = ctypes.c_int64(0)
    rc = sqlite3_prepare_v2(db, sql_p, -1, ctypes.addressof(stmt), 0)
    if rc != SQLITE_OK:
        _raise_rc(db, rc)
    try:
        ncols = sqlite3_column_count(stmt.value)
        if ncols != len(names):
            raise ValueError("dtype has %d fields but query returns %d columns" % (len(names), ncols))
        return _query_core(stmt.value, ncols, offsets, tags, widths,
                           int(dtype.itemsize), numba.from_dtype(dtype))
    finally:
        sqlite3_finalize(stmt.value)
```

Add `_raise_rc(db, rc)` mirroring `_sqlite_vtable._raise_rc` (decode `sqlite3_errmsg`). Zero each fresh row's numeric fields before `_store_cell` so a non-NULL write over uninitialized `np.empty` memory is clean for integer NULL→0: simplest is to `out[...] = 0`-equivalent per new row — instead, in `_query_core`, before the `for j` loop do a byte-zero of the row: `for b in range(itemsize): raw_row[b] = 0` (the implementer adds this; it makes int NULL→0 hold without a per-tag zero in `_store_cell`).

- [ ] **Step 3: Wire the public export.** In `numbox/core/bindings/__init__.py` add `from numbox.core.bindings._sqlite_query import *` next to the other `_sqlite_*` star-imports.

- [ ] **Step 4: Run tests** → PASS. Iterate on the byte-view detail per the implementer note until green. **Step 5: flake8. Step 6: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_query.py numbox/core/bindings/__init__.py test/core/test_sqlite_query.py
git -C /home/erik/projects/numbox commit -m "feat(sqlite): query_to_array reads SELECT results into a numpy structured array"
```

---

### Task 4: `xBestIndex` constraint pushdown on the read-only vtable

**Goal:** Make `register_table`'s vtable claim eq/range `WHERE` constraints in `xBestIndex`, receive their values in `xFilter`, and skip non-matching rows in `xNext`/`xEof` — with correct `estimatedRows`/`estimatedCost`. First increment: `EQ` + the four range ops (`GT/GE/LT/LE`) on numeric columns.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_vtable.py` (add 3 element dtypes; extend `_CUR_DTYPE`; rewrite `_xbestindex`, `_xfilter`, `_xnext`, `_xeof`; extend `_xopen`/`_xclose` for the predicate buffer)
- Test: `test/core/test_sqlite_vtable.py`

**Acceptance Criteria:**
- [ ] `_CONSTRAINT_DTYPE.itemsize == 12`, `_USAGE_DTYPE.itemsize == 8`, `_ORDERBY_DTYPE.itemsize == 8` (asserted at import).
- [ ] `SELECT * FROM t WHERE col0 = k` returns exactly the matching rows; same for `>`, `>=`, `<`, `<=` on a numeric column.
- [ ] Result set is identical to a full-scan baseline (`[r for r in all if pred(r)]`).
- [ ] `EXPLAIN QUERY PLAN` shows the vtable used with the chosen `idxNum`.
- [ ] Multi-constraint (`WHERE a >= x AND b < y`) prunes correctly.

**Verify:** `/home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_vtable.py -k "pushdown or filter or bestindex" -q` → pass.

**Design notes:**
- `xBestIndex` (already verified ABI in the spec): view `aConstraint`/`aConstraintUsage` via `carray(..., _CONSTRAINT_DTYPE/_USAGE_DTYPE)`; for each `usable` constraint whose `op` is supported on a numeric column, assign the next 1-based `argvIndex`, set `idxNum` to the count of bound predicates, and set `estimatedRows`/`estimatedCost`. `omit=0` (SQLite re-checks surfaced rows — safe).
- `xFilter` receives `argv` (C array of `sqlite3_value*`); for slot `k` the value is `argv[k-1]`. Decode each bound predicate's `(column, op, value)` into a per-cursor predicate buffer (`sqlite3_malloc`'d, pointer in the cursor, sized to `argc`), then advance to the first matching row.
- `xNext` advances then skips to the next matching row; `xEof` reports end. A shared `_row_matches(c, rowid)` evaluates the buffered predicates against the row's numeric cell value (read with `load_unaligned` + the column tag, mirroring `_xcolumn`).
- Predicate buffer element dtype: `np.dtype([("col","i4"),("op","i4"),("val","f8")], align=True)` — store the RHS as `f8` (covers int + real comparisons; integer columns compare as float, exact for the int range SQLite uses).

**Steps:**

- [ ] **Step 1: Write failing tests.** Add a query harness helper to `test_sqlite_vtable.py` (open `:memory:`, `register_table`, run a SELECT via prepared statement collecting column 0), then:

```python
def _select_col0(db, sql_text):
    rows = []
    stmt = c_int64(0)
    with c_string(sql_text) as p:
        assert sqlite3_prepare_v2(db, p, -1, addressof(stmt), 0) == 0
    while sqlite3_step(stmt.value) == 100:  # SQLITE_ROW
        rows.append(sqlite3_column_int64(stmt.value, 0))
    sqlite3_finalize(stmt.value)
    return rows


def test_pushdown_eq():
    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    arr = np.array([[3], [5], [7], [5], [9]], dtype=np.int64)
    h = register_table(db.value, "t", arr, ["c"])
    assert sorted(_select_col0(db.value, "SELECT c FROM t WHERE c = 5")) == [5, 5]
    assert _select_col0(db.value, "SELECT c FROM t WHERE c = 7") == [7]
    assert _select_col0(db.value, "SELECT c FROM t WHERE c = 100") == []
    sqlite3_close(db.value)
    del h


def test_pushdown_range_matches_fullscan():
    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    vals = [3, 5, 7, 5, 9]
    arr = np.array([[v] for v in vals], dtype=np.int64)
    h = register_table(db.value, "t", arr, ["c"])
    for op, py in [(">", lambda v: v > 5), (">=", lambda v: v >= 5),
                   ("<", lambda v: v < 5), ("<=", lambda v: v <= 5)]:
        got = sorted(_select_col0(db.value, "SELECT c FROM t WHERE c %s 5" % op))
        assert got == sorted(v for v in vals if py(v)), op
    sqlite3_close(db.value)
    del h
```

Run: expect FAIL (full scan returns all rows / wrong results) until `xFilter`/`xNext` honor predicates.

- [ ] **Step 2: Add the element dtypes** to `_sqlite_vtable.py` (after `_IDX_INFO_DTYPE`):

```python
_CONSTRAINT_DTYPE = np.dtype([("iColumn", "i4"), ("op", "u1"), ("usable", "u1"), ("iTermOffset", "i4")], align=True)
_USAGE_DTYPE = np.dtype([("argvIndex", "i4"), ("omit", "u1")], align=True)
_ORDERBY_DTYPE = np.dtype([("iColumn", "i4"), ("desc", "u1")], align=True)
assert _CONSTRAINT_DTYPE.itemsize == 12 and _USAGE_DTYPE.itemsize == 8 and _ORDERBY_DTYPE.itemsize == 8

_PRED_DTYPE = np.dtype([("col", "i4"), ("op", "i4"), ("val", "f8")], align=True)
_PRED_SIZE = _PRED_DTYPE.itemsize
```

- [ ] **Step 3: Extend the cursor dtype** for filter state (predicate buffer pointer + count). Replace `_CUR_DTYPE`:

```python
_CUR_DTYPE = np.dtype([
    ("base", _SQLITE3_VTAB_CURSOR_DTYPE), ("descriptor", "i8"), ("rowid", "i8"),
    ("scratch_p", "i8"), ("pred_p", "i8"), ("n_pred", "i8"),
], align=True)
_CUR_SIZE = _CUR_DTYPE.itemsize
```

Update `_xopen` to set `c[0].pred_p = 0` and `c[0].n_pred = 0`; update `_xclose` to `sqlite3_free(c[0].pred_p)` before freeing the cursor.

- [ ] **Step 4: Rewrite `_xbestindex`** to claim supported constraints. Add op-code imports at top (`from _sqlite_constants import SQLITE_INDEX_CONSTRAINT_EQ, _GT, _GE, _LT, _LE`). Body:

```python
@cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
def _xbestindex(vtab, idx_info):
    v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
    d = carray(_cast_int_to_void_p(v[0].descriptor), (1,), dtype=_DESC_DTYPE)
    ii = carray(_cast_int_to_void_p(idx_info), (1,), dtype=_IDX_INFO_DTYPE)
    ncon = ii[0].nConstraint
    cons = carray(_cast_int_to_void_p(ii[0].aConstraint), (ncon,), dtype=_CONSTRAINT_DTYPE)
    usage = carray(_cast_int_to_void_p(ii[0].aConstraintUsage), (ncon,), dtype=_USAGE_DTYPE)
    tags = carray(_cast_int_to_void_p(d[0].col_tags), (d[0].ncols,), dtype=np.int32)
    nbound = 0
    for i in range(ncon):
        op = cons[i].op
        col = cons[i].iColumn
        supported = (op == SQLITE_INDEX_CONSTRAINT_EQ or op == SQLITE_INDEX_CONSTRAINT_GT
                     or op == SQLITE_INDEX_CONSTRAINT_GE or op == SQLITE_INDEX_CONSTRAINT_LT
                     or op == SQLITE_INDEX_CONSTRAINT_LE)
        numeric = col >= 0 and col < d[0].ncols and tags[col] <= _TAG_F64
        if cons[i].usable != 0 and supported and numeric:
            nbound += 1
            usage[i].argvIndex = nbound
            usage[i].omit = 0
    ii[0].idxNum = nbound
    sel = d[0].nrows if nbound == 0 else (d[0].nrows // (nbound + 1) + 1)
    ii[0].estimatedRows = sel
    ii[0].estimatedCost = float64(d[0].nrows)
    return SQLITE_OK
```

> `idxNum` only needs to be nonzero to signal "predicates were bound"; `xFilter` reconstructs the `(col, op)` for each `argv` slot by re-walking the constraints — but `xFilter` does not receive `aConstraint`. So instead carry `(col, op)` to `xFilter` through the predicate decode: the simplest robust channel is to have `xBestIndex` store nothing extra and have `xFilter` know only the values — therefore encode `(col, op)` for each bound slot into `idxStr`. **Implementer:** build a small `idxStr` C buffer in `xBestIndex` (`sqlite3_malloc`, set `ii[0].idxStr`, `ii[0].needToFreeIdxStr = 1`) holding `nbound` pairs of `(int32 col, int32 op)`; `xFilter` reads it. This keeps `(col, op)` aligned with `argv` order.

- [ ] **Step 5: Rewrite `_xfilter`** to decode predicates into the cursor buffer and seek to the first match:

```python
@cfunc(types.int32(types.intp, types.int32, types.intp, types.int32, types.intp), cache=_CACHE)
def _xfilter(cur, idx_num, idx_str, argc, argv):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    sqlite3_free(c[0].pred_p)               # free any predicates from a prior xFilter
    c[0].pred_p = 0
    c[0].n_pred = 0
    c[0].rowid = 0
    if idx_num > 0 and argc > 0:
        pred_p = sqlite3_malloc(int32(argc * _PRED_SIZE))
        if pred_p == 0:
            return SQLITE_NOMEM
        preds = carray(_cast_int_to_void_p(pred_p), (argc,), dtype=_PRED_DTYPE)
        spec = carray(_cast_int_to_void_p(idx_str), (2 * argc,), dtype=np.int32)
        vals = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
        for k in range(argc):
            preds[k].col = spec[2 * k]
            preds[k].op = spec[2 * k + 1]
            preds[k].val = sqlite3_value_double(vals[k])
        c[0].pred_p = pred_p
        c[0].n_pred = argc
    _seek_match(cur)
    return SQLITE_OK
```

Add `sqlite3_value_double` to the imports and to `signatures` usage (it is already bound). `_seek_match`/`_row_matches`/`_cell_value` are small `@cfunc`s or `@njit` helpers (see Step 6).

- [ ] **Step 6: Add the matching helpers + rewrite `_xnext`/`_xeof`.** Read a numeric cell value as float (mirror `_xcolumn`'s numeric ladder), evaluate predicates, and advance:

```python
@njit(**jit_options)
def _cell_value(d, rowid, col):
    base = d[0].data_base
    offsets = carray(_cast_int_to_void_p(d[0].col_offsets), (d[0].ncols,), dtype=np.int64)
    tags = carray(_cast_int_to_void_p(d[0].col_tags), (d[0].ncols,), dtype=np.int32)
    addr = base + rowid * d[0].row_stride + offsets[col]
    tag = tags[col]
    if tag == _TAG_I8:
        return float64(load_unaligned(addr, int8))
    # ... full numeric ladder mirroring _xcolumn (I16/I32/I64/U8..U64/BOOL/F32/F64) ...
    elif tag == _TAG_F64:
        return load_unaligned(addr, float64)
    return float64(0)


@njit(**jit_options)
def _row_matches(cur):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    if c[0].n_pred == 0:
        return True
    d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_DESC_DTYPE)
    preds = carray(_cast_int_to_void_p(c[0].pred_p), (c[0].n_pred,), dtype=_PRED_DTYPE)
    for k in range(c[0].n_pred):
        cv = _cell_value(d, c[0].rowid, preds[k].col)
        op = preds[k].op
        rv = preds[k].val
        ok = (cv == rv if op == SQLITE_INDEX_CONSTRAINT_EQ
              else cv > rv if op == SQLITE_INDEX_CONSTRAINT_GT
              else cv >= rv if op == SQLITE_INDEX_CONSTRAINT_GE
              else cv < rv if op == SQLITE_INDEX_CONSTRAINT_LT
              else cv <= rv)
        if not ok:
            return False
    return True


@njit(**jit_options)
def _seek_match(cur):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_DESC_DTYPE)
    while c[0].rowid < d[0].nrows and not _row_matches(cur):
        c[0].rowid = c[0].rowid + 1
```

`_xnext`: `c[0].rowid += 1; _seek_match(cur)`. `_xeof`: unchanged (compares `rowid >= nrows`). `_xfilter` calls `_seek_match` after decoding so the first surfaced row already matches.

- [ ] **Step 7: Run tests** → PASS (full numeric ladder filled in `_cell_value`). **Step 8: flake8. Step 9: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_vtable.py test/core/test_sqlite_vtable.py
git -C /home/erik/projects/numbox commit -m "feat(sqlite-vtable): xBestIndex eq/range constraint pushdown"
```

---

### Task 5: `register_tvf` — table-valued functions

**Goal:** Add `register_tvf(db, name, arg_types, out_dtype, fn)`: a vtable whose rows are a numpy structured array (of `out_dtype`) that the user's `fn(*args)` computes from the hidden-column argument values, served via per-cursor NRT-backed storage.

**Files:**
- Create: `numbox/core/bindings/_sqlite_tvf.py`
- Modify: `numbox/core/bindings/__init__.py` (`from ..._sqlite_tvf import *`)
- Modify: `docs/numbox.core.bindings.rst` (new module automodule — see Task 7)
- Test: `test/core/test_sqlite_tvf.py`

**Acceptance Criteria:**
- [ ] `SELECT * FROM f(a, b)` invokes `fn(a, b)`, materializes its returned array, and yields its rows; the visible columns match `out_dtype`.
- [ ] Two calls with different args in one query plan return the respective computed rows.
- [ ] Per-query result array is released exactly once (leak-balance: `mi_alloc` delta == `mi_free` delta over ≥10 iterations).
- [ ] A query that supplies no value for a required hidden arg returns no rows / a constraint error (not a crash).

**Verify:** `/home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_tvf.py -q` → pass.

**Design notes:**
- Schema (declared in `xConnect`): `out_dtype` columns visible, then `arg_types` as `HIDDEN` columns. `xBestIndex` requires `EQ` on every hidden-arg column (reuse Task 4's constraint walk; if any hidden arg is unbound, return a very high cost so the plan is rejected / yields no rows).
- `xFilter` reads the hidden-arg values from `argv`, calls a generated `@njit` impl that invokes `fn(*args)` and returns `np.empty`-backed rows of `out_dtype`; pin the result via the array's meminfo using the `[meminfo_p, data_p]` slot idiom (spec assumption A1): `mi_p, data_p = structref_meminfo(result); _incref_meminfo(mi_p)`; store both in the cursor; `n_rows = len(result)`.
- `xColumn` reads `data_p + rowid*itemsize + offsets[j]` and dispatches on the column tag (reuse the read ladder — factor `_xcolumn`'s body so both the read-only vtable and the TVF share it, OR duplicate the ladder reading from `data_p` instead of `data_base`).
- `xClose` calls `release_meminfo(slot_mi_p)` exactly once and zeroes the slot; guard `data_p == 0` before any `carray`. The release path is reached on the SQLite error path too (the `xFilter`/`xClose` bodies are wrapped in `try/except`).
- Use the codegen+anchor pattern from `_sqlite_udf_helpers._compile_callbacks` to bake `fn` and `out_dtype` (as a `numba.from_dtype` typeref global) into the `xFilter` impl so it caches cross-process and the allocator specializes on `out_dtype` (A2 rule).

**Steps:**

- [ ] **Step 1: Write failing tests** in `test/core/test_sqlite_tvf.py`:

```python
from ctypes import addressof, c_int64
import numpy as np
from numba import njit
from numbox.core.bindings import sqlite3_open, sqlite3_close, register_tvf
from numbox.core.bindings import (
    sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize, sqlite3_column_int64)
from numbox.utils.cstrings import c_string

_OUT = np.dtype([("n", "i8")])


@njit
def _series(start, stop):
    out = np.empty(stop - start, _OUT)   # _OUT baked as a module global by codegen
    for i in range(stop - start):
        out[i].n = start + i
    return out


def test_tvf_series():
    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    stmt = c_int64(0)
    with c_string("SELECT n FROM series(2, 5)") as p:
        sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    got = []
    while sqlite3_step(stmt.value) == 100:
        got.append(sqlite3_column_int64(stmt.value, 0))
    sqlite3_finalize(stmt.value)
    assert got == [2, 3, 4]
    sqlite3_close(db.value)
    del h
```

Add a leak-balance test mirroring `test_sqlite_udf_helpers.py`'s `test_no_meminfo_leak` (enable NRT stats, run the query in a loop ≥10×, assert alloc delta == free delta).

- [ ] **Step 2: Build the TVF module skeleton** in `_sqlite_tvf.py`. Reuse the patterns: the `_Sqlite3Module` ctypes struct + a second `TVF_MODULE` (its own cfunc set), a TVF descriptor dtype (out-dtype tags/offsets/widths + ncols + visible/hidden split), and a TVF cursor dtype with the NRT slot:

```python
_TVF_CUR_DTYPE = np.dtype([
    ("base", _SQLITE3_VTAB_CURSOR_DTYPE), ("descriptor", "i8"), ("rowid", "i8"),
    ("mi_p", "i8"), ("data_p", "i8"), ("n_rows", "i8"), ("scratch_p", "i8"),
], align=True)
```

(Import `_SQLITE3_VTAB_CURSOR_DTYPE`, `_VTAB_DTYPE`, the read ladder helpers, and the tag constants from `_sqlite_vtable`/`_sqlite_typemap`.)

- [ ] **Step 3: Generate the `xFilter` impl via codegen** (mirror `_sqlite_udf_helpers._compile_callbacks`): bake `_fn` (the user callable, `njit`-prepared via `_prepare_callbacks`-style) and `_out_dtype = numba.from_dtype(out_dtype)` as globals; the generated impl reads the hidden args from `argv`, calls `result = _fn(arg0, arg1, ...)`, pins the meminfo, and stores `mi_p`/`data_p`/`n_rows` in the cursor. Show the generated source template:

```python
_TVF_XFILTER_SRC = '''
@njit(**jit_options)
def _tvf_xfilter_impl(cur, argc, argv):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
    if c[0].mi_p != 0:
        release_meminfo(c[0].mi_p)
        c[0].mi_p = 0
        c[0].data_p = 0
        c[0].n_rows = 0
    c[0].rowid = 0
    vals = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
    a0 = sqlite3_value_int64(vals[0])
    a1 = sqlite3_value_int64(vals[1])
    result = _fn(a0, a1)
    mi_p, data_p = structref_meminfo(result)
    _incref_meminfo(mi_p)
    c[0].mi_p = mi_p
    c[0].data_p = data_p
    c[0].n_rows = result.shape[0]
'''
```

> **Implementer:** the number/decode of hidden args is fixed by `len(arg_types)`; generate the `a0..aN` lines and the `_fn(a0, ..., aN)` call from `arg_types` (int → `sqlite3_value_int64`, float → `sqlite3_value_double`). This is the only part of the template that varies with arity — build the arg lines as a string from `arg_types` before `exec`, exactly as `_sqlite_udf_helpers` bakes per-UDAF source.

- [ ] **Step 4: TVF `xColumn`** reads from `data_p` (not an external base): generate/define it to compute `addr = c[0].data_p + rowid*itemsize + offsets[j]` and run the read ladder (reuse a shared `_emit_cell(ctx, addr, tag, width, scratch_p)` factored from `_xcolumn`). Guard `c[0].data_p == 0` → emit NULL/return. `xEof` compares `rowid >= n_rows`; `xClose` does `release_meminfo(c[0].mi_p)` if nonzero, frees `scratch_p`, frees the cursor.

- [ ] **Step 5: `register_tvf`** builds the TVF descriptor (offsets/tags/widths from `out_dtype` via the type-map; HIDDEN columns from `arg_types`), generates the schema `CREATE TABLE x(<visible> , <hidden> HIDDEN)`, compiles the cfuncs (via the codegen namespace), wires a per-registration `TVF_MODULE` (its function pointers are this registration's cfunc addresses — since `xFilter`/`xColumn` are generated per-call, the module is per-registration, not shared), and calls `sqlite3_create_module` (Task 6 switches this to `_v2`). Returns a handle keeping the module + descriptor + cfuncs alive.

> **Note:** unlike `register_table` (one shared `THE_MODULE`), TVF generates per-registration cfuncs (the user `fn` + `out_dtype` are baked in), so each `register_tvf` builds its own module struct. The handle must retain the ctypes module struct, the cfunc objects, and the descriptor buffers.

- [ ] **Step 6: Run tests** → PASS (iterate on the codegen arg-arity generation). **Step 7: flake8 + the doc-codeblock check if the module docstring contains code blocks. Step 8: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_tvf.py numbox/core/bindings/__init__.py test/core/test_sqlite_tvf.py
git -C /home/erik/projects/numbox commit -m "feat(sqlite): register_tvf table-valued functions backed by a computed numpy array"
```

---

### Task 6: `create_module_v2` + `xDestroy` registry cleanup

**Goal:** Register `register_table` and `register_tvf` modules via `sqlite3_create_module_v2` with an `xDestroy` callback that drops the per-table keep-alive from a module-level registry when SQLite tears the module down — fixing the "caller must retain the handle forever" hazard.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_vtable.py` (registry + ctypes `xDestroy` callback; switch `register_table` to `_v2`)
- Modify: `numbox/core/bindings/_sqlite_tvf.py` (switch `register_tvf` to `_v2` using the shared registry helper)
- Test: `test/core/test_sqlite_vtable.py`

**Acceptance Criteria:**
- [ ] After `register_table` + a query + `sqlite3_close`, the registry no longer holds that table's entry (xDestroy fired exactly once).
- [ ] Two tables → two independent entries, both removed on close.
- [ ] Re-registering the same name removes the first entry at re-registration time; the second is removed on close.
- [ ] `xDestroy` never calls `sqlite3_free`/C-free on the descriptor (numpy-owned) — it only pops the Python registry.
- [ ] No use-after-free: queries before close still read valid data.

**Verify:** `/home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_vtable.py -k "xdestroy or cleanup or registry" -q` → pass.

**Design notes (spec assumption A4):**
- Keep `pClientData` = the descriptor pointer (`built.c.ctypes.data`) so the cfuncs read the descriptor unchanged.
- `_REGISTRY: dict[int, object]` maps that pointer → the `_VTableHandle`. `register_*` sets `_REGISTRY[ptr] = handle` **before** calling `create_module_v2`.
- The `xDestroy` callback is a **ctypes** `CFUNCTYPE(None, c_void_p)` (NOT a numba cfunc — it must run Python and hold the GIL): `_REGISTRY.pop(p_aux, None)`. Pass its address as the 5th `create_module_v2` arg. Store the `CFUNCTYPE` object module-level so it stays alive.
- Re-entrancy: re-registering a name makes `create_module_v2` call the old entry's `xDestroy` synchronously; `dict.pop(..., None)` tolerates it (the new entry was added under a different pointer key beforehand).

**Steps:**

- [ ] **Step 1: Write failing tests:**

```python
def test_xdestroy_pops_registry_on_close():
    from numbox.core.bindings import _sqlite_vtable as v
    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    arr = np.array([[1], [2]], dtype=np.int64)
    h = register_table(db.value, "t", arr, ["c"])
    assert _select_col0(db.value, "SELECT c FROM t") == [1, 2]
    n_before = len(v._REGISTRY)
    sqlite3_close(db.value)
    assert len(v._REGISTRY) == n_before - 1
    del h


def test_xdestroy_two_tables():
    from numbox.core.bindings import _sqlite_vtable as v
    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    h1 = register_table(db.value, "t1", np.array([[1]], np.int64), ["c"])
    h2 = register_table(db.value, "t2", np.array([[2]], np.int64), ["c"])
    n = len(v._REGISTRY)
    sqlite3_close(db.value)
    assert len(v._REGISTRY) == n - 2
    del h1, h2
```

Run: expect FAIL (no `_REGISTRY` / xDestroy not wired).

- [ ] **Step 2: Add the registry + ctypes callback** to `_sqlite_vtable.py`:

```python
_REGISTRY = {}


def _xdestroy_py(p_aux):
    _REGISTRY.pop(p_aux, None)


_XDESTROY_CFUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(_xdestroy_py)
_XDESTROY_ADDR = ctypes.cast(_XDESTROY_CFUNC, ctypes.c_void_p).value
```

Add the `sqlite3_create_module_v2` proxy wrapper next to `sqlite3_create_module`:

```python
@proxy(signatures.get("sqlite3_create_module_v2"), jit_options=jit_options)
def sqlite3_create_module_v2(db, z_name, p_module, p_client_data, x_destroy):
    return _call_lib_func("sqlite3_create_module_v2", (db, z_name, p_module, p_client_data, x_destroy))
```

- [ ] **Step 3: Switch `register_table`** to register the handle then use `_v2`:

```python
def register_table(db, name, arr, columns=None, *, text_as_blob=False):
    built = _build_descriptor(arr, columns, text_as_blob)
    handle = _VTableHandle(built)
    ptr = built.c.ctypes.data
    _REGISTRY[ptr] = handle
    with c_string(name) as name_p:
        rc = sqlite3_create_module_v2(db, name_p, _THE_MODULE_P, ptr, _XDESTROY_ADDR)
    if rc != SQLITE_OK:
        _REGISTRY.pop(ptr, None)
        _raise_rc(db, name, rc)
    return handle
```

Update the `_VTableHandle` docstring: the keep-alive now lives in `_REGISTRY` and is released by SQLite via `xDestroy`; the returned handle is advisory.

- [ ] **Step 4: Switch `register_tvf`** to the same pattern (factor `_register_with_destroy(db, name, module_p, client_ptr, handle)` into `_sqlite_vtable` and call it from both).

- [ ] **Step 5: Run tests** → PASS. **Step 6: flake8. Step 7: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_vtable.py numbox/core/bindings/_sqlite_tvf.py test/core/test_sqlite_vtable.py
git -C /home/erik/projects/numbox commit -m "feat(sqlite-vtable): create_module_v2 + xDestroy registry cleanup"
```

---

### Task 7: Docs + full CI gate

**Goal:** Document the new modules/symbols and run every CI check locally before declaring done.

**Files:**
- Modify: `docs/numbox.core.bindings.rst` (add `automodule` sections for `_sqlite_query` and `_sqlite_tvf`; mention `query_to_array`/`register_tvf` in the conventions/family list)
- Modify: `numbox/core/bindings/__init__.py` (confirm both new star-imports present)

**Acceptance Criteria:**
- [ ] `docs/numbox.core.bindings.rst` lists the two new modules; `sphinx-build` exits 0.
- [ ] Full local gate green: flake8, full pytest, sphinx, doc-codeblock-flake8, lychee.

**Verify:** the five gate commands below all succeed.

**Steps:**

- [ ] **Step 1: Update `docs/numbox.core.bindings.rst`** — add the per-module `automodule` sections under "Modules" for `numbox.core.bindings._sqlite_query` and `numbox.core.bindings._sqlite_tvf`, and add them to the "Bindings module conventions" family list (mirror an existing `_sqlite_*` entry exactly).

- [ ] **Step 2: Run the full local gate** (mirror the CI workflows):

```bash
# clean caches first
/home/erik/projects/numbox/venv/bin/python -c "import shutil,pathlib; [shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home()/'.cache'/'numba',ignore_errors=True)"
/home/erik/projects/numbox/venv/bin/flake8 . --count --show-source --statistics
/home/erik/projects/numbox/venv/bin/pytest --durations=20 -q
( cd /home/erik/projects/numbox/docs && /home/erik/projects/numbox/venv/bin/sphinx-build -b html . _build/html )
/home/erik/projects/numbox/venv/bin/python .github/scripts/extract_codeblocks.py --flake8 /home/erik/projects/numbox/venv/bin/flake8 docs README.md
# lychee on changed .rst/.md/.py vs origin/main, excluding .github/ (see feedback_lychee_local_before_push)
```

Expected: flake8 clean; pytest all pass (no new failures vs baseline); sphinx exit 0; doc-codeblock clean; lychee 0 errors.

- [ ] **Step 3: Commit** docs.

```bash
git -C /home/erik/projects/numbox add docs/numbox.core.bindings.rst
git -C /home/erik/projects/numbox commit -m "docs(sqlite): document query_to_array and register_tvf modules"
```

---

## Self-Review

**Spec coverage:** query_to_array (Tasks 2,3) ✓; pushdown (Tasks 1,4) ✓; TVF (Tasks 1,4,5) ✓; create_module_v2/xDestroy (Tasks 1,6) ✓; shared type-map (Task 0) ✓; docs/gate (Task 7) ✓. The spec's A1 keepalive idiom → Task 5 Step 3; A2 typeref-dtype → Task 3 core + Task 5 codegen; A3 element dtypes → Task 4 Step 2; A4 registry/Py-side release → Task 6.

**Type consistency:** `_CUR_DTYPE` is extended once (Task 4) before TVF defines its own `_TVF_CUR_DTYPE` (Task 5) — they are distinct cursors for distinct modules; confirmed not shared. `register_tvf(db, name, arg_types, out_dtype, fn)` signature is identical in spec, Task 5 tests, and Step 5. `_REGISTRY` keyed by descriptor pointer is used identically in Task 6 Steps 2-4.

**Open follow-the-implementer notes (intentional, not placeholders):** the byte-view mechanism in Task 3 (`out.view(np.uint8)` vs `carray`), the `idxStr` `(col,op)` channel in Task 4, and the per-arity codegen in Task 5 each carry a concrete instruction plus the exact existing pattern to mirror; resolve them during the task's TDD loop.

**Scope:** large but single-PR by the user's explicit choice. Task 5 (TVF) is the heaviest and the natural split point if review size becomes a concern — it depends on Tasks 0/1/4 but nothing depends on it except Task 6's `register_tvf` rewrite (which degrades gracefully if TVF is deferred).
