# Columnar `register_table` (dict of 1-D numpy arrays → SQLite vtable) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overload `numbox.core.bindings.register_table` so its data argument may be a `Mapping[str, np.ndarray]` of equal-length 1-D arrays (columnar/struct-of-arrays), exposed as the same read-only, zero-copy SQLite virtual table it already builds from row-major structured/2-D arrays.

**Architecture:** Generalize the single generic vtable's per-table descriptor from one shared base pointer + row stride + per-column offsets to **per-column base pointers + per-column strides** (row-major becomes the special case `col_base[j]=data_base+offset[j]`, `col_stride[j]=row_stride`). Only the descriptor dtype and three address-computation sites change; pushdown, typemap, result setters, registration, and keep-alive are shared verbatim. Then dispatch on the data argument's type to a new columnar builder.

**Tech Stack:** numba (`@njit`/`@cfunc`, `carray`), numpy structured dtypes, SQLite virtual-table C API via numbox bindings, pytest, flake8 (max-line-length=127), Sphinx.

**Spec:** `docs/superpowers/specs/2026-06-20-register-table-mapping-design.md`
**Branch:** `feat/register-table-mapping` (off `origin/main` @ `2036fae`, 0 behind `upstream/main`).

**Conventions for every task:**
- Use the venv-absolute interpreter: `/home/erik/projects/numbox/venv/bin/python`, `.../venv/bin/flake8`.
- **Clean caches before every pytest run** (the `_DESC_DTYPE` change invalidates cached cfuncs):
  ```bash
  /home/erik/projects/numbox/venv/bin/python -c "import shutil,pathlib; root=pathlib.Path('/home/erik/projects/numbox'); [shutil.rmtree(p,ignore_errors=True) for p in root.rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home()/'.cache'/'numba',ignore_errors=True)"
  ```
- All file paths below are under `/home/erik/projects/numbox/`.
- Do **not** open any PR; the plan ends at a green local gate (PR creation needs explicit user consent).

---

### Task 1: Generalize the descriptor to per-column bases + strides (behavior-preserving refactor)

**Goal:** Replace the row-major descriptor fields (`data_base`/`row_stride`/`col_offsets`) with per-column `col_bases`/`col_strides`, updating the three address sites and the array builder, so the existing row-major behavior is byte-for-byte unchanged.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_vtable.py` (`_DESC_DTYPE` ~56-62; `_BuiltDescriptor` ~151-166; `_build_descriptor` ~169-213; `_cell_value_f64` ~366-400; `_cell_value_i64` ~404-430; `_xcolumn` ~547-600)
- Test: `test/core/test_sqlite_vtable.py` (update white-box assertions at lines 80, 81, 92)

**Acceptance Criteria:**
- [ ] `_DESC_DTYPE.itemsize == 64` (was 72) and the assert reflects it.
- [ ] The three address sites compute `addr = col_bases[col] + rowid*col_strides[col]`.
- [ ] `_build_descriptor` (structured + 2-D paths) produces identical query results as before.
- [ ] The full existing `test/core/test_sqlite_vtable.py` suite passes (same 56 tests) with the three white-box assertions updated to the new descriptor shape.
- [ ] flake8 clean.

**Verify:** `clean caches`, then
`/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py -v --durations=20` → all pass;
`/home/erik/projects/numbox/venv/bin/flake8 --max-line-length=127 numbox/core/bindings/_sqlite_vtable.py test/core/test_sqlite_vtable.py` → no output.

**Steps:**

- [ ] **Step 1: Update `_DESC_DTYPE` and its assert.** Replace the dtype block (currently lines ~56-62) with:

```python
_DESC_DTYPE = np.dtype([
    ("nrows", "i8"), ("ncols", "i4"),
    ("col_bases", "i8"), ("col_strides", "i8"),
    ("col_tags", "i8"), ("col_widths", "i8"),
    ("schema_ptr", "i8"), ("scratch_bytes", "i8"),
], align=True)
assert _DESC_DTYPE.itemsize == 64
```

- [ ] **Step 2: Replace `_BuiltDescriptor`.** It becomes a pure keep-alive that also exposes the layout-independent scalars the white-box tests read (`nrows`/`ncols`/`scratch_bytes`) plus the new `bases`/`strides` buffers and the underlying data arrays. Replace the class (currently lines ~151-166) with:

```python
class _BuiltDescriptor:
    """Keep-alive for the descriptor array, the buffers it points into, and the
    underlying column data array(s). The C descriptor ``c`` is the source of
    truth; nothing here is read by the cfuncs."""
    __slots__ = ("c", "bases", "strides", "tags", "widths", "schema",
                 "nrows", "ncols", "scratch_bytes", "arrays")

    def __init__(self, c, bases, strides, tags, widths, schema, arrays):
        self.c = c
        self.bases = bases
        self.strides = strides
        self.tags = tags
        self.widths = widths
        self.schema = schema
        self.arrays = arrays
        self.nrows = int(c["nrows"][0])
        self.ncols = int(c["ncols"][0])
        self.scratch_bytes = int(c["scratch_bytes"][0])
```

- [ ] **Step 3: Add `_finalize_descriptor` and rewrite `_build_descriptor`.** Replace the whole `_build_descriptor` function (currently lines ~169-213) with the following two functions:

```python
def _finalize_descriptor(nrows, col_names, tags, widths, bases, strides, arrays):
    """Build the _DESC_DTYPE descriptor + keep-alive from per-column layout."""
    if not col_names:
        raise ValueError("table must have at least one column")
    scratch = max([w + 1 for w, t in zip(widths, tags) if t == _TAG_U], default=0)
    bases_buf = np.array(bases, dtype=np.int64)
    strides_buf = np.array(strides, dtype=np.int64)
    tags_buf = np.array(tags, dtype=tags_buf_t)
    widths_buf = np.array(widths, dtype=np.int64)
    cols_sql = ", ".join('"%s" %s' % (n.replace('"', '""'), _SQL_TYPE[t]) for n, t in zip(col_names, tags))
    schema = ("CREATE TABLE x(%s)" % cols_sql).encode("utf-8") + b"\x00"

    c = np.zeros(1, _DESC_DTYPE)
    c["nrows"] = int(nrows)
    c["ncols"] = len(col_names)
    c["col_bases"] = bases_buf.ctypes.data
    c["col_strides"] = strides_buf.ctypes.data
    c["col_tags"] = tags_buf.ctypes.data
    c["col_widths"] = widths_buf.ctypes.data
    c["schema_ptr"] = ctypes.cast(ctypes.c_char_p(schema), ctypes.c_void_p).value
    c["scratch_bytes"] = int(scratch)
    return _BuiltDescriptor(c, bases_buf, strides_buf, tags_buf, widths_buf, schema, tuple(arrays))


def _build_descriptor(arr, columns, text_as_blob):
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a numpy.ndarray, got %r" % (type(arr),))
    fields = arr.dtype.fields
    if fields is not None:
        if arr.ndim != 1:
            raise ValueError("structured array must be 1-D, got ndim=%d" % arr.ndim)
        names = list(arr.dtype.names)
        sub = [arr.dtype.fields[n][0] for n in names]
        offs = [arr.dtype.fields[n][1] for n in names]
        col_names = list(columns) if columns is not None else names
        if len(col_names) != len(names):
            raise ValueError("columns length %d != field count %d" % (len(col_names), len(names)))
    else:
        if arr.ndim != 2:
            raise ValueError("plain array must be 2-D, got ndim=%d" % arr.ndim)
        if columns is None or len(columns) != arr.shape[1]:
            raise ValueError("columns must list all %d column names for a 2-D array" % arr.shape[1])
        col_names = list(columns)
        sub = [arr.dtype] * arr.shape[1]
        offs = [j * arr.strides[1] for j in range(arr.shape[1])]

    tags = [_col_tag(dt, text_as_blob) for dt in sub]
    widths = [int(dt.itemsize) for dt in sub]
    base = arr.ctypes.data
    row_stride = int(arr.strides[0])
    bases = [base + int(o) for o in offs]
    strides = [row_stride] * len(col_names)
    return _finalize_descriptor(arr.shape[0], col_names, tags, widths, bases, strides, (arr,))
```

- [ ] **Step 4: Update the three address sites.** In `_cell_value_f64`, the lines that currently read

```python
    ncols = d[0].ncols
    base = d[0].data_base
    row_stride = d[0].row_stride
    offsets = carray(_cast_int_to_void_p(d[0].col_offsets), (ncols,), dtype=np.int64)
    tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=tags_buf_t)
    addr = base + rowid * row_stride + offsets[col]
```

become

```python
    ncols = d[0].ncols
    bases = carray(_cast_int_to_void_p(d[0].col_bases), (ncols,), dtype=np.int64)
    strides = carray(_cast_int_to_void_p(d[0].col_strides), (ncols,), dtype=np.int64)
    tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=tags_buf_t)
    addr = bases[col] + rowid * strides[col]
```

Apply the identical replacement in `_cell_value_i64` (same five-line block; index is `col`). In `_xcolumn`, the block currently reads

```python
        ncols = d[0].ncols
        base = d[0].data_base
        row_stride = d[0].row_stride
        offsets = carray(_cast_int_to_void_p(d[0].col_offsets), (ncols,), dtype=np.int64)
        tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=tags_buf_t)
        widths = carray(_cast_int_to_void_p(d[0].col_widths), (ncols,), dtype=np.int64)
        addr = base + rowid * row_stride + offsets[j]
```

and becomes (note the index here is `j`):

```python
        ncols = d[0].ncols
        bases = carray(_cast_int_to_void_p(d[0].col_bases), (ncols,), dtype=np.int64)
        strides = carray(_cast_int_to_void_p(d[0].col_strides), (ncols,), dtype=np.int64)
        tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=tags_buf_t)
        widths = carray(_cast_int_to_void_p(d[0].col_widths), (ncols,), dtype=np.int64)
        addr = bases[j] + rowid * strides[j]
```

- [ ] **Step 5: Update the three white-box test assertions.** In `test/core/test_sqlite_vtable.py`:

Line 80 — replace
```python
    assert (d.nrows, d.ncols, d.row_stride) == (3, 2, a.strides[0])
    assert list(d.offsets) == [0, 8]
```
with
```python
    assert (d.nrows, d.ncols) == (3, 2)
    assert list(d.strides) == [a.strides[0], a.strides[0]]
    assert [b - a.ctypes.data for b in d.bases] == [0, 8]
```

Line 92 (inside `test_descriptor_structured_mixed`, where `a` is the structured array) — replace
```python
    assert list(d.offsets) == [dt.fields["t"][1], dt.fields["q"][1], dt.fields["p"][1]]
```
with
```python
    assert [b - a.ctypes.data for b in d.bases] == [dt.fields["t"][1], dt.fields["q"][1], dt.fields["p"][1]]
```

(The `.scratch_bytes` assertions at lines 93 and 116-118 are unchanged — that attribute is retained.)

- [ ] **Step 6: Run the regression suite (this IS the refactor's test).** Clean caches, then:

Run: `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py -v --durations=20`
Expected: all pass (the ~50 black-box query tests prove behavior is preserved; the 3 updated white-box tests prove the new shape).

Run: `/home/erik/projects/numbox/venv/bin/flake8 --max-line-length=127 numbox/core/bindings/_sqlite_vtable.py test/core/test_sqlite_vtable.py`
Expected: no output.

- [ ] **Step 7: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_vtable.py test/core/test_sqlite_vtable.py
git -C /home/erik/projects/numbox commit -m "refactor(vtable): per-column bases+strides descriptor (row-major as special case)"
```

---

### Task 2: Add the mapping overload (dispatch + columnar builder) with columnar tests

**Goal:** Make `register_table` accept a `Mapping[str, np.ndarray]` via type dispatch, building the columnar descriptor from each column's own base pointer + stride, with full columnar test coverage.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_vtable.py` (add `from collections.abc import Mapping`; add `_build_descriptor_columnar`; rewrite `register_table` signature + dispatch + docstring)
- Test: `test/core/test_sqlite_vtable.py` (append columnar tests)

**Acceptance Criteria:**
- [ ] `register_table(db, name, {"a": arr1, "b": arr2})` registers a queryable columnar table.
- [ ] Numeric/bool/`'U'`/`'S'` (+`text_as_blob`) columns round-trip; eq/range pushdown works; uint64 wraps signed; NaN→NULL; two columnar tables join; a non-contiguous (strided) column reads correctly.
- [ ] Error cases raise the specified exceptions: `columns` with a mapping → `ValueError`; non-`ndarray` value → `TypeError`; `ndim != 1` → `ValueError`; unequal lengths → `ValueError`; empty mapping → `ValueError`; object-dtype column → `TypeError`/`ValueError`.
- [ ] Existing row-major suite still green; flake8 clean.

**Verify:** clean caches, then
`/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py -v --durations=20` → all pass;
`/home/erik/projects/numbox/venv/bin/flake8 --max-line-length=127 numbox/core/bindings/_sqlite_vtable.py test/core/test_sqlite_vtable.py` → no output.

**Steps:**

- [ ] **Step 1: Write the failing columnar tests first.** Append to `test/core/test_sqlite_vtable.py`:

```python
# ---- columnar (mapping) overload ----

def test_columnar_int64_roundtrip():
    db = _open_memory()
    cols = {"a": np.array([1, 2, 3], dtype=np.int64),
            "b": np.array([10, 20, 30], dtype=np.int64)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT a, b FROM t ORDER BY a") == [(1, 10), (2, 20), (3, 30)]
    sqlite3_close(db)


def test_columnar_mixed_dtypes():
    db = _open_memory()
    cols = {"i": np.array([1, 2], dtype=np.int32),
            "f": np.array([1.5, 2.5], dtype=np.float64),
            "b": np.array([True, False], dtype=np.bool_)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT i, f, b FROM t ORDER BY i") == [(1, 1.5, 1), (2, 2.5, 0)]
    sqlite3_close(db)


def test_columnar_pushdown_eq_range():
    db = _open_memory()
    cols = {"a": np.array([1, 2, 3, 4], dtype=np.int64),
            "b": np.array([10, 20, 30, 40], dtype=np.int64)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT b FROM t WHERE a = 3") == [(30,)]
    assert _fetchall(db, "SELECT b FROM t WHERE a >= 3 ORDER BY a") == [(30,), (40,)]
    sqlite3_close(db)


def test_columnar_unicode_and_bytes():
    db = _open_memory()
    cols = {"u": np.array(["foo", "héllo"], dtype="U6"),
            "s": np.array([b"ab", b"cde"], dtype="S4")}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT u, s FROM t ORDER BY u") == [("foo", "ab"), ("héllo", "cde")]
    sqlite3_close(db)


def test_columnar_text_as_blob():
    db = _open_memory()
    cols = {"s": np.array([b"\xff\xfe", b"\x01\x02"], dtype="S2")}
    register_table(db, "t", cols, text_as_blob=True)
    assert _fetchall(db, "SELECT s FROM t") == [(b"\xff\xfe",), (b"\x01\x02",)]
    sqlite3_close(db)


def test_columnar_uint64_wraps_signed():
    db = _open_memory()
    cols = {"x": np.array([2 ** 63], dtype=np.uint64)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT x FROM t") == [(-(2 ** 63),)]
    sqlite3_close(db)


def test_columnar_nan_reads_as_null():
    db = _open_memory()
    cols = {"x": np.array([1.0, np.nan, 3.0], dtype=np.float64)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT x FROM t") == [(1.0,), (None,), (3.0,)]
    sqlite3_close(db)


def test_columnar_join_two_tables():
    db = _open_memory()
    left = {"id": np.array([1, 2, 3], dtype=np.int64),
            "v": np.array([10, 20, 30], dtype=np.int64)}
    right = {"id": np.array([2, 3, 4], dtype=np.int64),
             "w": np.array([200, 300, 400], dtype=np.int64)}
    register_table(db, "lhs", left)
    register_table(db, "rhs", right)
    got = _fetchall(db, "SELECT lhs.id, v, w FROM lhs JOIN rhs ON lhs.id = rhs.id ORDER BY lhs.id")
    assert got == [(2, 20, 200), (3, 30, 300)]
    sqlite3_close(db)


def test_columnar_strided_column():
    db = _open_memory()
    backing = np.array([1, 99, 2, 99, 3, 99], dtype=np.int64)
    strided = backing[::2]            # values 1,2,3; stride 16 bytes; non-contiguous
    assert not strided.flags["C_CONTIGUOUS"]
    cols = {"a": strided, "b": np.array([10, 20, 30], dtype=np.int64)}
    register_table(db, "t", cols)
    assert _fetchall(db, "SELECT a, b FROM t ORDER BY a") == [(1, 10), (2, 20), (3, 30)]
    sqlite3_close(db)


def test_columnar_rejects_columns_kwarg():
    with pytest.raises(ValueError):
        register_table(0, "t", {"a": np.array([1], dtype=np.int64)}, columns=["a"])


def test_columnar_rejects_non_array_value():
    with pytest.raises(TypeError):
        register_table(0, "t", {"a": [1, 2, 3]})


def test_columnar_rejects_non_1d_value():
    with pytest.raises(ValueError):
        register_table(0, "t", {"a": np.zeros((2, 2), dtype=np.int64)})


def test_columnar_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        register_table(0, "t", {"a": np.array([1, 2], dtype=np.int64),
                                "b": np.array([1], dtype=np.int64)})


def test_columnar_rejects_empty_mapping():
    with pytest.raises(ValueError):
        register_table(0, "t", {})


def test_columnar_rejects_object_dtype():
    with pytest.raises((TypeError, ValueError)):
        register_table(0, "t", {"a": np.array(["x", "y"], dtype=object)})
```

- [ ] **Step 2: Run to verify they fail.**

Run: `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py -k columnar -v`
Expected: FAIL — `register_table` currently calls `_build_descriptor`, which raises `TypeError("arr must be a numpy.ndarray ...")` on a dict.

- [ ] **Step 3: Add the `Mapping` import.** At the top of `numbox/core/bindings/_sqlite_vtable.py`, after `import ctypes` (line 12), add:

```python
from collections.abc import Mapping
```

- [ ] **Step 4: Add `_build_descriptor_columnar`.** Place it immediately after `_build_descriptor`:

```python
def _build_descriptor_columnar(mapping, text_as_blob):
    names = list(mapping.keys())
    if not names:
        raise ValueError("mapping must have at least one column")
    bases, strides, tags, widths = [], [], [], []
    nrows = None
    for n in names:
        col = mapping[n]
        if not isinstance(col, np.ndarray):
            raise TypeError("column %r must be a 1-D numpy array, got %r" % (n, type(col)))
        if col.ndim != 1:
            raise ValueError("column %r must be 1-D, got ndim=%d" % (n, col.ndim))
        if nrows is None:
            nrows = int(col.shape[0])
        elif int(col.shape[0]) != nrows:
            raise ValueError("column %r length %d != %d" % (n, col.shape[0], nrows))
        tags.append(_col_tag(col.dtype, text_as_blob))
        widths.append(int(col.dtype.itemsize))
        bases.append(col.ctypes.data)
        strides.append(int(col.strides[0]))
    cols = tuple(mapping[n] for n in names)
    return _finalize_descriptor(nrows, names, tags, widths, bases, strides, cols)
```

- [ ] **Step 5: Rewrite `register_table` (signature, dispatch, docstring).** Replace the function definition line and its body. The new `def` line and docstring:

```python
def register_table(db, name, data, columns=None, *, text_as_blob=False):
    """Expose tabular data as a read-only eponymous SQLite virtual table
    (queryable directly as ``name`` with no CREATE VIRTUAL TABLE, since the
    module's xCreate is its xConnect).

    ``data`` may be:

    - a 1-D numpy **structured array** (row-major; column names from the dtype,
      optionally renamed/reordered by ``columns``);
    - a 2-D numpy **array** (row-major; ``columns`` required, one name per column);
    - a **mapping** (e.g. a ``dict``) of column-name -> 1-D numpy array (columnar;
      all arrays must share one length; the keys name the columns, so passing
      ``columns`` raises ``ValueError`` -- rename/reorder by building the mapping
      with the desired keys and order before calling). Mapping values must already
      be ``numpy.ndarray`` (no coercion); a non-array value raises ``TypeError`` and
      a non-1-D array raises ``ValueError``.

    The registration's keep-alive lives in the module-level ``_DATA_ANCHOR`` and is
    released by SQLite via ``xDestroy`` (on connection close or re-registration of
    the same name). The caller must not mutate or resize the array(s) while the
    table is registered -- the view is zero-copy, so queries read each column's
    buffer directly (numeric reads alias it, and ``'S'``/BLOB values are handed to
    SQLite as ``SQLITE_STATIC`` pointers into it).

    Value semantics:

    - ``uint64`` values >= 2**63 are stored as SQLite's signed INTEGER and
      therefore wrap to negative.
    - Floats pass through as REAL, including +/-inf -- EXCEPT NaN: SQLite coerces a
      NaN ``REAL`` to SQL NULL (via ``sqlite3IsNaN``), so a NaN cell reads back as
      NULL. numpy has no missing value, so every non-NaN cell is non-NULL. Masked
      arrays are unsupported: the mask is ignored.

    String columns:

    - ``text_as_blob`` affects only bytes (``'S'``) columns; unicode (``'U'``) is
      always TEXT. By default ``'S'`` becomes TEXT and its raw bytes pass through
      unvalidated -- pass ``text_as_blob=True`` for non-UTF-8 bytes so they are
      stored as BLOB rather than malformed TEXT.
    - Fixed-width ``'S'``/``'U'`` columns are NUL-padded by numpy; trailing NUL
      padding is trimmed on read while interior NULs are preserved.
    - A TEXT value with an interior NUL is stored faithfully (an explicit byte
      length is passed), but C-string readers and most SQL text functions truncate
      at the first NUL; read it via ``sqlite3_column_bytes`` + the text/blob
      pointer, or use ``text_as_blob=True`` for full fidelity.
    """
    if isinstance(data, np.ndarray):
        built = _build_descriptor(data, columns, text_as_blob)
    elif isinstance(data, Mapping):
        if columns is not None:
            raise ValueError("columns must be omitted when data is a mapping; "
                             "the dict keys name the columns")
        built = _build_descriptor_columnar(data, text_as_blob)
    else:
        raise TypeError("data must be a numpy.ndarray or a mapping of name -> 1-D array, "
                        "got %r" % (type(data),))
    handle = _VTableHandle(built)
    _register_with_destroy(db, name, _THE_MODULE_P, built.c.ctypes.data, handle)
```

(Note: `register_table` returns `None` as before; the keep-alive is the module-level `_DATA_ANCHOR`. Do not add a `return`.)

- [ ] **Step 6: Run to verify all pass.** Clean caches, then:

Run: `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py -v --durations=20`
Expected: PASS (existing + new columnar tests).

Run: `/home/erik/projects/numbox/venv/bin/flake8 --max-line-length=127 numbox/core/bindings/_sqlite_vtable.py test/core/test_sqlite_vtable.py`
Expected: no output.

- [ ] **Step 7: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_vtable.py test/core/test_sqlite_vtable.py
git -C /home/erik/projects/numbox commit -m "feat(vtable): register_table accepts a mapping of 1-D arrays (columnar)"
```

---

### Task 3: Query-surface tests (stdlib `sqlite3.Connection` + `query_to_array`)

**Goal:** Prove the columnar table is queryable through both requested surfaces — the Python stdlib connection (`extract_connection_ptr` + `conn.execute`, untested anywhere today) and the compiled `query_to_array` materializer (structured-array result).

**Files:**
- Test: `test/core/test_sqlite_vtable.py` (add imports, a `_libraries_coordinated` guard, two tests)

**Acceptance Criteria:**
- [ ] A columnar dict registered on a stdlib `sqlite3.Connection` via `extract_connection_ptr` is queryable with `conn.execute(...).fetchall()` (skipped when libraries are uncoordinated, e.g. macOS without `DYLD_INSERT_LIBRARIES`).
- [ ] `query_to_array(db, sql_p, dtype)` returns a 1-D structured ndarray with the expected rows over a columnar table.
- [ ] Existing suite still green; flake8 clean.

**Verify:** clean caches, then
`/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py -k "stdlib_connection or query_to_array" -v` → pass (or skip for the stdlib one if uncoordinated);
full `test/core/test_sqlite_vtable.py` green; flake8 clean.

**Steps:**

- [ ] **Step 1: Add imports + the coordination guard.** Add to the imports near the top of `test/core/test_sqlite_vtable.py`:

```python
import sqlite3
from numbox.core.bindings import query_to_array
from numbox.core.bindings._sqlite_conn import sqlite3_libversion
from numbox.utils.pysqlite_bridge import extract_connection_ptr
```

Then add this helper near the top of the test module (after the imports):

```python
def _libraries_coordinated():
    """True if numbox's bindings and Python's sqlite3 use the same libsqlite3."""
    numbox_version = c_char_p(sqlite3_libversion()).value.decode()
    return numbox_version == sqlite3.sqlite_version
```

(`c_char_p` is already imported at the top of this file.)

- [ ] **Step 2: Write the two surface tests.** Append:

```python
def test_columnar_on_stdlib_connection():
    if not _libraries_coordinated():
        pytest.skip("uncoordinated libsqlite3 (see numbox.utils.pysqlite_bridge)")
    conn = sqlite3.connect(":memory:")
    try:
        cols = {"id": np.array([1, 2, 3], dtype=np.int64),
                "px": np.array([100.0, 250.0, 50.0], dtype=np.float64)}
        register_table(extract_connection_ptr(conn), "trades", cols)
        rows = conn.execute("SELECT id FROM trades WHERE px > 100 ORDER BY id").fetchall()
        assert rows == [(2,)]
    finally:
        conn.close()


def test_columnar_query_to_array():
    db = _open_memory()
    cols = {"id": np.array([1, 2, 3], dtype=np.int64),
            "px": np.array([100.0, 250.0, 50.0], dtype=np.float64)}
    register_table(db, "trades", cols)
    out_dtype = np.dtype([("id", "i8"), ("px", "f8")])
    with c_string("SELECT id, px FROM trades WHERE px > 100 ORDER BY id") as sql_p:
        out = query_to_array(db, sql_p, out_dtype)
    assert out.dtype == out_dtype
    assert out["id"].tolist() == [2]
    assert out["px"].tolist() == [250.0]
    sqlite3_close(db)
```

- [ ] **Step 3: Run to verify.** Clean caches, then:

Run: `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py -k "stdlib_connection or query_to_array" -v`
Expected: PASS on Linux (the stdlib test may SKIP if libraries are uncoordinated; the `query_to_array` test must PASS).

Run: `/home/erik/projects/numbox/venv/bin/flake8 --max-line-length=127 test/core/test_sqlite_vtable.py`
Expected: no output.

- [ ] **Step 4: Commit.**

```bash
git -C /home/erik/projects/numbox add test/core/test_sqlite_vtable.py
git -C /home/erik/projects/numbox commit -m "test(vtable): columnar query surfaces (stdlib Connection + query_to_array)"
```

---

### Task 4: Docs build, CLAUDE.md status, full local gate

**Goal:** Confirm the docs build picks up the new docstring, record the change in the fork-only project status, and pass the complete local CI-equivalent gate.

**Files:**
- Modify: `CLAUDE.md` (add a Project Status bullet — fork-only, excluded from upstream PR)

**Acceptance Criteria:**
- [ ] `sphinx-build` exits 0 (the `register_table` docstring update is picked up via the existing `automodule`; no `.rst` edit).
- [ ] flake8 clean across changed files.
- [ ] The **full** test suite passes (clean caches first).
- [ ] CLAUDE.md Project Status has a bullet describing the columnar overload.

**Verify:** the three commands in Step 2 all succeed; `git -C /home/erik/projects/numbox status --short` shows only the intended changes.

**Steps:**

- [ ] **Step 1: Add the CLAUDE.md Project Status bullet.** Under the `## Project Status` section of `CLAUDE.md`, add:

```markdown
- **Columnar `register_table` overload** — `register_table(db, name, data, ...)` now accepts a `Mapping[str, np.ndarray]` of equal-length 1-D arrays (column-major) in addition to row-major structured/2-D arrays, via type dispatch on `data`. The per-table descriptor was generalized from one base + row stride + per-column offsets to per-column `col_bases`/`col_strides` (`_DESC_DTYPE.itemsize` 72→64); row-major is the special case. Zero-copy, read-only, same typemap/pushdown/value-semantics. Honors per-column stride (non-contiguous columns work). Design + plan under `docs/superpowers/{specs,plans}/2026-06-20-register-table-mapping*` (fork-only, not on `main`).
```

- [ ] **Step 2: Run the full local gate.** Clean caches, then run each and confirm:

Run: `/home/erik/projects/numbox/venv/bin/python -m pytest test/ --durations=20`
Expected: all pass.

Run: `/home/erik/projects/numbox/venv/bin/flake8 --max-line-length=127 numbox/core/bindings/_sqlite_vtable.py test/core/test_sqlite_vtable.py`
Expected: no output.

Run: `/home/erik/projects/numbox/venv/bin/sphinx-build -b html /home/erik/projects/numbox/docs /home/erik/projects/numbox/docs/_build/html`
Expected: exit 0 (warning count stable is acceptable).

- [ ] **Step 3: Commit.**

```bash
git -C /home/erik/projects/numbox add CLAUDE.md
git -C /home/erik/projects/numbox commit -m "docs(vtable): record columnar register_table overload in project status"
```

- [ ] **Step 4: Stop.** Report the green gate. Do **not** open a fork or upstream PR — that requires explicit user consent (numbox fork workflow).

---

## Notes / risks (for the implementer)

- **`_sqlite_tvf.py` is independent** — it defines its own `_TVF_DESC_DTYPE` and only imports `sqlite3_declare_vtab` (and a few unrelated helpers) from `_sqlite_vtable`. Renaming `_DESC_DTYPE`'s fields does not touch it. Do not edit `_sqlite_tvf.py`.
- **Cache invalidation is mandatory** between runs — the `_DESC_DTYPE` layout change means stale cached cfuncs would read the old field offsets. The clean-caches command is in the conventions block.
- **The keep-alive is the module-level `_DATA_ANCHOR`**, not the return value of `register_table` (which is `None`). New tests therefore call `register_table(...)` without capturing a handle.
