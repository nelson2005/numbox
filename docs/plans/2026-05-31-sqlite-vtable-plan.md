# SQLite virtual tables (numpy-backed, read-only) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `register_table(db, name, arr, ...)` exposing a numpy array (2-D or 1-D structured) as a read-only, zero-copy SQLite virtual table queryable with ordinary SQL.

**Architecture:** One module-global generic `sqlite3_module` (struct of ~10 `@cfunc(cache=True)` callbacks) built once at import. Per-table state — base pointer, strides, per-column `(offset, dtype-tag, width)`, and the `CREATE TABLE` schema — lives in a `ctypes` descriptor passed as the module's `pClientData`. `xColumn` reads each cell at `data_base + rowid*row_stride + col_offsets[j]` and dispatches on the dtype tag. No codegen/anchors (read-only + generic), unlike the phase-3 UDAF helper.

**Tech stack:** numba (`@cfunc`/`@njit`/`@intrinsic`, 0.60–0.65), llvmlite, numpy, ctypes, libsqlite3. Reuses phase-1/2 bindings (`sqlite3_result_*`, `sqlite3_value_*`, statement/column API) and `numbox.utils.lowlevel` (`load_at`/`store_at`/`_cast_int_to_void_p`).

**Spec:** `docs/plans/2026-05-31-sqlite-vtable-design.md` (read it first).

**Universal command prefix** (every pytest/flake8 step). Clear caches first (numba/structref cache bugs otherwise mask failures), then run from repo root `/home/erik/projects/numbox`:

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
```

Test runner: `/home/erik/projects/numbox/venv/bin/python -m pytest <path> --durations=20`
Lint (whole tree): `/home/erik/projects/numbox/venv/bin/flake8 --max-line-length=127 .`

**Commit rule:** attribute to the user only; no AI/Claude/Anthropic mention, no `Co-Authored-By`. Feature branch is `feat/sqlite-vtable` (already created off the phase-3 branch); never commit to `main`.

---

## File structure

| File | Responsibility | Tasks |
|---|---|---|
| `numbox/utils/lowlevel.py` | `+ load_unaligned(p, ty)` (align=1 load) | 1 |
| `numbox/core/bindings/signatures.py` | `+3` `signatures_sqlite` entries | 2 |
| `numbox/core/bindings/_sqlite_vtable.py` | NEW — wrappers, constants, string helpers, descriptor builder, cfuncs + `THE_MODULE`, `register_table`, `_VTableHandle` | 2,3,4,5 |
| `numbox/core/bindings/__init__.py` | `+1` star-import of `_sqlite_vtable` | 5 |
| `docs/numbox.core.bindings.rst` | `+` automodule section (NOT the `_call_lib_func` family list) | 8 |
| `test/core/test_load_unaligned.py` | NEW — `load_unaligned` unit tests | 1 |
| `test/core/test_sqlite_vtable.py` | NEW — encoder/builder unit tests + integration tests | 3,4,5,6,7,8 |

---

## Task 1: `load_unaligned` intrinsic

**Goal:** Add an `align=1` load helper to `lowlevel.py` so `xColumn`/`utf32_to_utf8` can legally read packed (misaligned) structured fields.

**Files:**
- Modify: `numbox/utils/lowlevel.py` (add after `store_at`, ~line 106)
- Test: `test/core/test_load_unaligned.py`

**Acceptance Criteria:**
- [ ] `load_unaligned(p, ty)` reads the correct value from a deliberately misaligned address inside `@njit` code.
- [ ] Matches `load_at` for aligned addresses across int8/int32/int64/float64.

**Verify:** `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_load_unaligned.py --durations=20` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing test** — `test/core/test_load_unaligned.py`

```python
import numpy as np
from numba import njit
from numba.core.types import int32, int64, float64
from numbox.utils.lowlevel import load_unaligned, array_data_p


def test_load_unaligned_reads_misaligned_int64():
    # int64 stored at byte offset 1 of a byte buffer (deliberately misaligned).
    buf = np.zeros(16, dtype=np.uint8)
    buf[1:9] = np.frombuffer(np.int64(0x0123456789ABCDEF).tobytes(), dtype=np.uint8)

    @njit
    def read(base):
        return load_unaligned(base + 1, int64)

    assert read(array_data_p(buf)) == 0x0123456789ABCDEF


def test_load_unaligned_matches_load_at_aligned():
    a = np.array([7, -3, 11], dtype=np.int32)

    @njit
    def read(base, i):
        return load_unaligned(base + 4 * i, int32)

    base = array_data_p(a)
    assert [read(base, i) for i in range(3)] == [7, -3, 11]


def test_load_unaligned_float64():
    x = np.array([3.5], dtype=np.float64)

    @njit
    def read(base):
        return load_unaligned(base, float64)

    assert read(array_data_p(x)) == 3.5
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: cannot import name 'load_unaligned'`).

- [ ] **Step 3: Implement** — in `numbox/utils/lowlevel.py`, directly after the `store_at` definition (mirrors `_load_at` with `align=1`):

```python
@intrinsic
def _load_unaligned(typingctx: Context, p_ty, ty_ref: TypeRef):
    if unliteral(p_ty) not in (intp, uintp):
        raise TypingError(
            f"load_unaligned: pointer argument must be intp or uintp, got {p_ty}"
        )
    ty = ty_ref.instance_type
    sig = ty(p_ty, ty_ref)

    def codegen(context: BaseContext, builder, signature, args):
        ty_ll = context.get_data_type(ty)
        ptr = builder.inttoptr(args[0], ty_ll.as_pointer())
        return builder.load(ptr, align=1)
    return sig, codegen


@njit(**jit_options)
def load_unaligned(p, ty):
    """Load a value of type ``ty`` from raw pointer ``p`` with byte alignment.

    Like :func:`load_at` but emits an ``align=1`` load, so it is legal on a
    misaligned address (e.g. a packed numpy structured-dtype field). ``load_at``
    asserts the type's natural alignment, which is IR-level UB when the address
    is in fact misaligned.
    """
    return _load_unaligned(p, ty)
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Lint** (`flake8 --max-line-length=127 .`) then **commit**:

```bash
git -C /home/erik/projects/numbox add numbox/utils/lowlevel.py test/core/test_load_unaligned.py
git -C /home/erik/projects/numbox commit -m "feat(lowlevel): add load_unaligned (align=1 load) for packed struct fields"
```

---

## Task 2: SQLite vtable signatures, wrappers, and constants

**Goal:** Bind `sqlite3_create_module` / `sqlite3_declare_vtab` / `sqlite3_malloc`, and create `_sqlite_vtable.py` with its `@proxy` wrappers, result-code constants, and dtype-tag/struct-offset constants.

**Files:**
- Modify: `numbox/core/bindings/signatures.py` (add to `signatures_sqlite`)
- Create: `numbox/core/bindings/_sqlite_vtable.py`
- Test: `test/core/test_sqlite_vtable.py` (smoke import)

**Acceptance Criteria:**
- [ ] The three new signatures are present in `signatures_sqlite`.
- [ ] `_sqlite_vtable.py` imports cleanly; the wrappers are Python-callable (`.py_func` exists).

**Verify:** `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py::test_imports --durations=20` → pass.

**Steps:**

- [ ] **Step 1: Add signatures** — in `signatures.py`, inside the `signatures_sqlite` dict (near the other registration entries). `int32`/`intp` are already imported at top:

```python
    "sqlite3_create_module": int32(intp, intp, intp, intp),
    "sqlite3_declare_vtab": int32(intp, intp),
    "sqlite3_malloc": intp(int32),
```

- [ ] **Step 2: Create `_sqlite_vtable.py` header — wrappers + constants:**

```python
"""Expose a numpy array as a read-only SQLite virtual table (register_table).

A single generic sqlite3_module (built once at import) serves every table; the
per-table base pointer / strides / dtype tags / schema live in a ctypes
descriptor passed as pClientData. See
docs/plans/2026-05-31-sqlite-vtable-design.md.
"""
import ctypes

import numpy as np
from numba import carray, cfunc, njit, types
from numba.core.types import (
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float32, float64, intp,
)

from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.proxy.proxy import proxy
from numbox.core.bindings._sqlite_constants import SQLITE_OK, SQLITE_TRANSIENT
from numbox.core.bindings._sqlite_conn import sqlite3_errmsg
from numbox.core.bindings._sqlite_result import (
    sqlite3_result_int64, sqlite3_result_double,
    sqlite3_result_text, sqlite3_result_blob, sqlite3_result_error,
)
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import (
    _cast_int_to_void_p, load_at, store_at, load_unaligned, get_unicode_data_p,
)

__all__ = ["register_table"]

load_lib("sqlite3")

SQLITE_ERROR = 1
SQLITE_NOMEM = 7

# dtype tags (col_tags[j])
_TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64 = 0, 1, 2, 3
_TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64 = 4, 5, 6, 7
_TAG_F32, _TAG_F64, _TAG_BOOL = 8, 9, 10
_TAG_S, _TAG_U, _TAG_BLOB = 11, 12, 13

# descriptor field byte offsets (mirror of _NdarrayTableDescriptor below)
_D_NROWS, _D_NCOLS, _D_ROW_STRIDE, _D_DATA_BASE = 0, 8, 16, 24
_D_COL_OFFSETS, _D_COL_TAGS, _D_COL_WIDTHS, _D_SCHEMA, _D_SCRATCH = 32, 40, 48, 56, 64

# vtab layout: base sqlite3_vtab is 24 bytes; we append descriptor_ptr at +24
_VTAB_DESC, _VTAB_SIZE = 24, 32
# cursor layout: {pVtab(+0), descriptor(+8), rowid(+16), scratch@+24}
_CUR_PVTAB, _CUR_DESC, _CUR_ROWID, _CUR_SCRATCH = 0, 8, 16, 24


@proxy(signatures.get("sqlite3_create_module"), jit_options={"cache": True})
def sqlite3_create_module(db, z_name, p_module, p_client_data):
    return _call_lib_func("sqlite3_create_module", (db, z_name, p_module, p_client_data))


@proxy(signatures.get("sqlite3_declare_vtab"), jit_options={"cache": True})
def sqlite3_declare_vtab(db, z_sql):
    return _call_lib_func("sqlite3_declare_vtab", (db, z_sql))


@proxy(signatures.get("sqlite3_malloc"), jit_options={"cache": True})
def sqlite3_malloc(n):
    return _call_lib_func("sqlite3_malloc", (n,))
```

- [ ] **Step 3: Add the smoke test** to a new `test/core/test_sqlite_vtable.py`:

```python
def test_imports():
    from numbox.core.bindings import _sqlite_vtable as v
    assert hasattr(v.sqlite3_create_module, "py_func")
    assert hasattr(v.sqlite3_declare_vtab, "py_func")
    assert hasattr(v.sqlite3_malloc, "py_func")
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Lint + commit:**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/signatures.py numbox/core/bindings/_sqlite_vtable.py test/core/test_sqlite_vtable.py
git -C /home/erik/projects/numbox commit -m "feat(sqlite): bind create_module/declare_vtab/malloc + vtable module scaffold"
```

---

## Task 3: String-emit helpers (`utf32_to_utf8`, `_nul_trimmed_len`)

**Goal:** Add the `'U'` UTF-32→UTF-8 encoder and the `'S'` NUL-trim length helper used by `xColumn`.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_vtable.py`
- Test: `test/core/test_sqlite_vtable.py`

**Acceptance Criteria:**
- [ ] `utf32_to_utf8` encodes ASCII, 2-byte, 3-byte, and 4-byte (emoji) code points correctly and stops at a NUL.
- [ ] Invalid code points (surrogates, `>0x10FFFF`) become U+FFFD (`b"\xef\xbf\xbd"`).
- [ ] `_nul_trimmed_len` returns the unpadded byte length.

**Verify:** `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py -k "utf32 or nul_trim" --durations=20` → pass.

**Steps:**

- [ ] **Step 1: Write failing tests** (append to `test/core/test_sqlite_vtable.py`):

```python
import numpy as np
from numbox.core.bindings._sqlite_vtable import utf32_to_utf8, _nul_trimmed_len
from numbox.utils.lowlevel import array_data_p


def _encode(s, width):
    src = np.zeros(width, dtype=np.uint32)
    for i, ch in enumerate(s):
        src[i] = ord(ch)
    dst = np.zeros(4 * width + 1, dtype=np.uint8)
    n = utf32_to_utf8(array_data_p(src), width, array_data_p(dst))
    return bytes(dst[:n])


def test_utf32_ascii():
    assert _encode("abc", 6) == b"abc"


def test_utf32_multibyte():
    assert _encode("héllo", 6) == "héllo".encode("utf-8")


def test_utf32_emoji_4byte():
    assert _encode("a\U0001F600b", 6) == "a\U0001F600b".encode("utf-8")


def test_utf32_stops_at_nul():
    assert _encode("hi", 6) == b"hi"


def test_utf32_invalid_codepoint_replacement():
    src = np.array([0x41, 0xD800, 0x110000, 0], dtype=np.uint32)
    dst = np.zeros(64, dtype=np.uint8)
    n = utf32_to_utf8(array_data_p(src), 4, array_data_p(dst))
    assert bytes(dst[:n]) == b"A" + b"\xef\xbf\xbd" + b"\xef\xbf\xbd"


def test_nul_trimmed_len():
    buf = np.frombuffer(b"hi\x00\x00\x00", dtype=np.uint8).copy()
    assert _nul_trimmed_len(array_data_p(buf), 5) == 2
```

- [ ] **Step 2: Run — expect FAIL** (import error).

- [ ] **Step 3: Implement** (append to `_sqlite_vtable.py`). `'U'` code points are read with `load_unaligned` (a packed `'U'` field is misaligned); `'S'` bytes are a `uint8` view (alignment 1, so `carray` is fine):

```python
@njit(cache=True)
def _nul_trimmed_len(p, width):
    buf = carray(_cast_int_to_void_p(p), (width,), dtype=np.uint8)
    n = 0
    while n < width and buf[n] != 0:
        n += 1
    return n


@njit(cache=True)
def utf32_to_utf8(src, n_codepoints, dst):
    out = carray(_cast_int_to_void_p(dst), (4 * n_codepoints + 1,), dtype=np.uint8)
    k = 0
    for i in range(n_codepoints):
        cp = load_unaligned(src + 4 * i, uint32)
        if cp == 0:
            break
        if cp > 0x10FFFF or (0xD800 <= cp <= 0xDFFF):
            cp = 0xFFFD
        if cp < 0x80:
            out[k] = uint8(cp)
            k += 1
        elif cp < 0x800:
            out[k] = uint8(0xC0 | (cp >> 6))
            out[k + 1] = uint8(0x80 | (cp & 0x3F))
            k += 2
        elif cp < 0x10000:
            out[k] = uint8(0xE0 | (cp >> 12))
            out[k + 1] = uint8(0x80 | ((cp >> 6) & 0x3F))
            out[k + 2] = uint8(0x80 | (cp & 0x3F))
            k += 3
        else:
            out[k] = uint8(0xF0 | (cp >> 18))
            out[k + 1] = uint8(0x80 | ((cp >> 12) & 0x3F))
            out[k + 2] = uint8(0x80 | ((cp >> 6) & 0x3F))
            out[k + 3] = uint8(0x80 | (cp & 0x3F))
            k += 4
    return k
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Lint + commit** (`feat(sqlite): UTF-32→UTF-8 encoder + NUL-trim helper for vtable columns`).

---

## Task 4: Descriptor builder, dtype mapping, schema, validation, handle

**Goal:** Pure-Python `_build_descriptor(arr, columns, text_as_blob)` that turns a 2-D or structured array into the `ctypes` descriptor + retained buffers + `CREATE TABLE` schema, with the offset-mirror assertion and a `_VTableHandle`.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_vtable.py`
- Test: `test/core/test_sqlite_vtable.py`

**Acceptance Criteria:**
- [ ] For a 2-D int64 array with `columns=["a","b"]`: `nrows/ncols/row_stride` and per-column `offsets/tags/widths` match numpy `strides`/`itemsize`; schema is `CREATE TABLE x(a INTEGER, b INTEGER)`.
- [ ] For a structured array with `int+float+'S'+'U'` fields: column names = field names, tags/offsets/widths/types correct; `'U'` → `TEXT`, `'S'` → `TEXT` (or `BLOB` with `text_as_blob`); `scratch_bytes == max(width+1)` over `'U'`.
- [ ] Rejects 1-D non-structured, 3-D, object dtype, and (for 2-D) a missing/mismatched `columns` with a clear error.
- [ ] `_NdarrayTableDescriptor` field offsets equal the `_D_*` constants and `sizeof == 72` (asserted at import).

**Verify:** `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py -k descriptor --durations=20` → pass.

**Steps:**

- [ ] **Step 1: Write failing tests** (append):

```python
import ctypes
from numbox.core.bindings._sqlite_vtable import _build_descriptor, _TAG_I64, _TAG_F64, _TAG_U


def test_descriptor_2d_int64():
    a = np.arange(6, dtype=np.int64).reshape(3, 2)
    d = _build_descriptor(a, ["a", "b"], False)
    assert (d.nrows, d.ncols, d.row_stride) == (3, 2, a.strides[0])
    assert list(d.offsets) == [0, 8]
    assert list(d.tags) == [_TAG_I64, _TAG_I64]
    assert d.schema == b"CREATE TABLE x(a INTEGER, b INTEGER)\x00"


def test_descriptor_structured_mixed():
    dt = np.dtype([("t", "U6"), ("q", "i8"), ("p", "f8")])
    a = np.zeros(2, dtype=dt)
    d = _build_descriptor(a, None, False)
    assert d.ncols == 3
    assert list(d.tags) == [_TAG_U, _TAG_I64, _TAG_F64]
    assert list(d.offsets) == [dt.fields["t"][1], dt.fields["q"][1], dt.fields["p"][1]]
    assert d.scratch_bytes == 6 * 4 + 1  # one 'U6' column
    assert d.schema == b"CREATE TABLE x(t TEXT, q INTEGER, p REAL)\x00"


def test_descriptor_rejects_bad_shapes():
    import pytest
    with pytest.raises((TypeError, ValueError)):
        _build_descriptor(np.zeros((2, 2, 2), dtype=np.int64), ["a", "b"], False)
    with pytest.raises((TypeError, ValueError)):
        _build_descriptor(np.arange(4, dtype=np.int64), ["a"], False)  # 1-D non-structured
    with pytest.raises((TypeError, ValueError)):
        _build_descriptor(np.zeros((2, 3), dtype=np.int64), ["a", "b"], False)  # wrong col count


def test_descriptor_offsets_assertion_holds():
    # importing the module already runs the import-time assert; reaching here means it held.
    from numbox.core.bindings._sqlite_vtable import _NdarrayTableDescriptor
    assert ctypes.sizeof(_NdarrayTableDescriptor) == 72
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** (append to `_sqlite_vtable.py`). Note `_build_descriptor` returns a small holder with the live `ctypes` object plus the retained numpy/bytes buffers (so their pointers stay valid):

```python
class _NdarrayTableDescriptor(ctypes.Structure):
    _fields_ = [
        ("nrows", ctypes.c_int64),
        ("ncols", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
        ("row_stride", ctypes.c_int64),
        ("data_base", ctypes.c_int64),
        ("col_offsets", ctypes.c_int64),
        ("col_tags", ctypes.c_int64),
        ("col_widths", ctypes.c_int64),
        ("schema_ptr", ctypes.c_int64),
        ("scratch_bytes", ctypes.c_int64),
    ]


def _assert_descriptor_layout():
    f = _NdarrayTableDescriptor
    assert ctypes.sizeof(f) == 72
    assert (f.nrows.offset, f.ncols.offset, f.row_stride.offset, f.data_base.offset) == \
        (_D_NROWS, _D_NCOLS, _D_ROW_STRIDE, _D_DATA_BASE)
    assert (f.col_offsets.offset, f.col_tags.offset, f.col_widths.offset) == \
        (_D_COL_OFFSETS, _D_COL_TAGS, _D_COL_WIDTHS)
    assert (f.schema_ptr.offset, f.scratch_bytes.offset) == (_D_SCHEMA, _D_SCRATCH)


_assert_descriptor_layout()

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


class _BuiltDescriptor:
    """The ctypes descriptor plus every buffer whose pointer it holds."""
    __slots__ = ("c", "offsets", "tags", "widths", "schema",
                 "nrows", "ncols", "row_stride", "scratch_bytes", "arr")

    def __init__(self, c, offsets, tags, widths, schema, arr):
        self.c = c
        self.offsets = offsets
        self.tags = tags
        self.widths = widths
        self.schema = schema
        self.arr = arr
        self.nrows = c.nrows
        self.ncols = c.ncols
        self.row_stride = c.row_stride
        self.scratch_bytes = c.scratch_bytes


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
    scratch = max([w + 1 for w, t in zip(widths, tags) if t == _TAG_U], default=0)

    offsets_buf = np.array(offs, dtype=np.int64)
    tags_buf = np.array(tags, dtype=np.int32)
    widths_buf = np.array(widths, dtype=np.int64)
    cols_sql = ", ".join("%s %s" % (n, _SQL_TYPE[t]) for n, t in zip(col_names, tags))
    schema = ("CREATE TABLE x(%s)" % cols_sql).encode("utf-8") + b"\x00"

    c = _NdarrayTableDescriptor()
    c.nrows = int(arr.shape[0])
    c.ncols = len(col_names)
    c.row_stride = int(arr.strides[0])
    c.data_base = arr.ctypes.data
    c.col_offsets = offsets_buf.ctypes.data
    c.col_tags = tags_buf.ctypes.data
    c.col_widths = widths_buf.ctypes.data
    c.schema_ptr = ctypes.cast(ctypes.c_char_p(schema), ctypes.c_void_p).value
    c.scratch_bytes = int(scratch)
    return _BuiltDescriptor(c, offsets_buf, tags_buf, widths_buf, schema, arr)
```

> Note for the implementer: `ctypes.c_char_p(schema)` keeps `schema` (a `bytes`) referenced through the `_BuiltDescriptor.schema` slot, so its pointer stays valid. Confirm `c.schema_ptr` points at the bytes (read it back via `ctypes.string_at`).

- [ ] **Step 4: Run — expect PASS.** Fix any offset/strides mismatch.

- [ ] **Step 5: Lint + commit** (`feat(sqlite): numpy→vtable descriptor builder (layout, tags, schema, validation)`).

---

## Task 5: cfunc callbacks, static module, and `register_table`

**Goal:** Implement the ~10 generic `@cfunc(cache=True)` callbacks, assemble the module-global `THE_MODULE`, write `register_table` + `_VTableHandle`, export from the package, and pass the first end-to-end integration test (a C-contiguous 2-D int64 table).

**Files:**
- Modify: `numbox/core/bindings/_sqlite_vtable.py`
- Modify: `numbox/core/bindings/__init__.py` (`from numbox.core.bindings._sqlite_vtable import *`)
- Test: `test/core/test_sqlite_vtable.py`

**Acceptance Criteria:**
- [ ] `register_table(db, "points", arr2d_int64, columns=["a","b"])` then `SELECT a,b FROM points WHERE a >= k ORDER BY b DESC` returns the numpy-equivalent rows.
- [ ] `COUNT(*)` and `SUM(a)` match numpy.
- [ ] `register_table` is importable as `from numbox.core.bindings import register_table`.
- [ ] A raising/failed registration raises (rc check).

**Verify:** `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py -k "int64_table or count or imports" --durations=20` → pass.

**Steps:**

- [ ] **Step 0: Confirm the result-reading API names** against `numbox/core/bindings/_sqlite_stmt.py` and `_sqlite_column.py` before writing the harness (used below): `sqlite3_prepare_v2(db, sql_p, n_byte, stmt_pp, tail_pp)`, `sqlite3_step(stmt)`, `sqlite3_finalize(stmt)`, `sqlite3_column_count/_type/_int64/_double/_text/_blob/_bytes(stmt, icol)`. These are the confirmed names.

- [ ] **Step 1: Write the failing integration test** (append to `test/core/test_sqlite_vtable.py`). This `_fetchall` helper is reused by Tasks 6–7:

```python
from ctypes import addressof, c_int64, cast, c_char_p, string_at
from numbox.utils.cstrings import c_string
from numbox.core.bindings import (
    sqlite3_open, sqlite3_close, register_table,
    sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize,
    sqlite3_column_count, sqlite3_column_type,
    sqlite3_column_int64, sqlite3_column_double,
    sqlite3_column_text, sqlite3_column_blob, sqlite3_column_bytes,
)

_SQLITE_ROW = 100
_T_INT, _T_FLOAT, _T_TEXT, _T_BLOB, _T_NULL = 1, 2, 3, 4, 5


def _open_memory():
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        rc = sqlite3_open(name_p, addressof(db_p))
    assert rc == 0, rc
    return db_p.value


def _fetchall(db, sql):
    stmt_p = c_int64(0)
    with c_string(sql) as sql_p:
        rc = sqlite3_prepare_v2(db, sql_p, -1, addressof(stmt_p), 0)
    assert rc == 0, (rc, sql)
    stmt = stmt_p.value
    rows = []
    while sqlite3_step(stmt) == _SQLITE_ROW:
        row = []
        for i in range(sqlite3_column_count(stmt)):
            t = sqlite3_column_type(stmt, i)
            if t == _T_INT:
                row.append(sqlite3_column_int64(stmt, i))
            elif t == _T_FLOAT:
                row.append(sqlite3_column_double(stmt, i))
            elif t == _T_TEXT:
                row.append(cast(sqlite3_column_text(stmt, i), c_char_p).value.decode("utf-8"))
            elif t == _T_BLOB:
                n = sqlite3_column_bytes(stmt, i)
                row.append(string_at(sqlite3_column_blob(stmt, i), n) if n else b"")
            else:
                row.append(None)
        rows.append(tuple(row))
    sqlite3_finalize(stmt)
    return rows


def test_int64_table_select_where_order():
    db = _open_memory()
    a = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int64)
    h = register_table(db, "points", a, columns=["a", "b"])  # noqa: F841 (retain)
    assert _fetchall(db, "SELECT a, b FROM points WHERE a >= 2 ORDER BY b DESC") == [(3, 30), (2, 20)]
    sqlite3_close(db)


def test_count_and_sum():
    db = _open_memory()
    a = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    assert _fetchall(db, "SELECT COUNT(*), SUM(a) FROM t") == [(3, 6)]
    sqlite3_close(db)
```

- [ ] **Step 2: Run — expect FAIL** (`register_table` not importable).

- [ ] **Step 3: Implement the cfuncs + module + register_table** (append to `_sqlite_vtable.py`). Result-code callbacks return `SQLITE_ERROR` from `except`; `xColumn` uses `sqlite3_result_error`; `xEof`/`xNext` are pure arithmetic (no wrapper). The struct is populated in **positional** field order (`xNext` before `xEof`).

```python
@cfunc(types.int32(types.intp, types.intp, types.int32, types.intp, types.intp, types.intp), cache=True)
def _xconnect(db, p_aux, argc, argv, pp_vtab, pz_err):
    try:
        schema_p = load_at(p_aux + _D_SCHEMA, int64)
        rc = sqlite3_declare_vtab(db, schema_p)
        if rc != SQLITE_OK:
            return rc
        vtab = sqlite3_malloc(int32(_VTAB_SIZE))
        if vtab == 0:
            return SQLITE_NOMEM
        store_at(vtab + 0, int64(0))
        store_at(vtab + 8, int64(0))
        store_at(vtab + 16, int64(0))
        store_at(vtab + _VTAB_DESC, int64(p_aux))
        slot = carray(_cast_int_to_void_p(pp_vtab), (1,), dtype=np.intp)
        slot[0] = vtab
        return SQLITE_OK
    except Exception:
        return SQLITE_ERROR


@cfunc(types.int32(types.intp, types.intp), cache=True)
def _xbestindex(vtab, idx_info):
    return SQLITE_OK


@cfunc(types.int32(types.intp), cache=True)
def _xdisconnect(vtab):
    try:
        sqlite3_free(vtab)
        return SQLITE_OK
    except Exception:
        return SQLITE_ERROR


@cfunc(types.int32(types.intp, types.intp), cache=True)
def _xopen(vtab, pp_cursor):
    try:
        desc = load_at(vtab + _VTAB_DESC, int64)
        scratch = load_at(desc + _D_SCRATCH, int64)
        cur = sqlite3_malloc(int32(_CUR_SCRATCH + scratch))
        if cur == 0:
            return SQLITE_NOMEM
        store_at(cur + _CUR_PVTAB, int64(vtab))
        store_at(cur + _CUR_DESC, int64(desc))
        store_at(cur + _CUR_ROWID, int64(0))
        slot = carray(_cast_int_to_void_p(pp_cursor), (1,), dtype=np.intp)
        slot[0] = cur
        return SQLITE_OK
    except Exception:
        return SQLITE_ERROR


@cfunc(types.int32(types.intp), cache=True)
def _xclose(cur):
    try:
        sqlite3_free(cur)
        return SQLITE_OK
    except Exception:
        return SQLITE_ERROR


@cfunc(types.int32(types.intp, types.int32, types.intp, types.int32, types.intp), cache=True)
def _xfilter(cur, idx_num, idx_str, argc, argv):
    store_at(cur + _CUR_ROWID, int64(0))
    return SQLITE_OK


@cfunc(types.int32(types.intp), cache=True)
def _xnext(cur):
    store_at(cur + _CUR_ROWID, load_at(cur + _CUR_ROWID, int64) + 1)
    return SQLITE_OK


@cfunc(types.int32(types.intp), cache=True)
def _xeof(cur):
    desc = load_at(cur + _CUR_DESC, int64)
    rowid = load_at(cur + _CUR_ROWID, int64)
    nrows = load_at(desc + _D_NROWS, int64)
    if rowid >= nrows:
        return 1
    return 0


@cfunc(types.int32(types.intp, types.intp), cache=True)
def _xrowid(cur, p_rowid):
    store_at(p_rowid, load_at(cur + _CUR_ROWID, int64))
    return SQLITE_OK


@cfunc(types.int32(types.intp, types.intp, types.int32), cache=True)
def _xcolumn(cur, ctx, j):
    try:
        desc = load_at(cur + _CUR_DESC, int64)
        rowid = load_at(cur + _CUR_ROWID, int64)
        ncols = load_at(desc + _D_NCOLS, int32)
        base = load_at(desc + _D_DATA_BASE, int64)
        row_stride = load_at(desc + _D_ROW_STRIDE, int64)
        offsets = carray(_cast_int_to_void_p(load_at(desc + _D_COL_OFFSETS, int64)), (ncols,), dtype=np.int64)
        tags = carray(_cast_int_to_void_p(load_at(desc + _D_COL_TAGS, int64)), (ncols,), dtype=np.int32)
        widths = carray(_cast_int_to_void_p(load_at(desc + _D_COL_WIDTHS, int64)), (ncols,), dtype=np.int64)
        addr = base + rowid * row_stride + offsets[j]
        tag = tags[j]
        if tag == _TAG_I8:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, int8)))
        elif tag == _TAG_I16:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, int16)))
        elif tag == _TAG_I32:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, int32)))
        elif tag == _TAG_I64:
            sqlite3_result_int64(ctx, load_unaligned(addr, int64))
        elif tag == _TAG_U8:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, uint8)))
        elif tag == _TAG_U16:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, uint16)))
        elif tag == _TAG_U32:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, uint32)))
        elif tag == _TAG_U64:
            sqlite3_result_int64(ctx, int64(load_unaligned(addr, uint64)))
        elif tag == _TAG_BOOL:
            sqlite3_result_int64(ctx, int64(1) if load_unaligned(addr, uint8) != 0 else int64(0))
        elif tag == _TAG_F32:
            sqlite3_result_double(ctx, float64(load_unaligned(addr, float32)))
        elif tag == _TAG_F64:
            sqlite3_result_double(ctx, load_unaligned(addr, float64))
        elif tag == _TAG_S:
            n = _nul_trimmed_len(addr, widths[j])
            sqlite3_result_text(ctx, addr, int32(n), SQLITE_TRANSIENT)
        elif tag == _TAG_BLOB:
            n = _nul_trimmed_len(addr, widths[j])
            sqlite3_result_blob(ctx, addr, int32(n), SQLITE_TRANSIENT)
        elif tag == _TAG_U:
            scratch = cur + _CUR_SCRATCH
            n = utf32_to_utf8(addr, widths[j] // 4, scratch)
            sqlite3_result_text(ctx, scratch, int32(n), SQLITE_TRANSIENT)
        return SQLITE_OK
    except Exception:
        sqlite3_result_error(ctx, get_unicode_data_p("error reading vtable column"), -1)
        return SQLITE_ERROR


class _Sqlite3Module(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in (
        "xCreate", "xConnect", "xBestIndex", "xDisconnect", "xDestroy",
        "xOpen", "xClose", "xFilter", "xNext", "xEof", "xColumn", "xRowid",
        "xUpdate", "xBegin", "xSync", "xCommit", "xRollback",
        "xFindFunction", "xRename")]
    # iVersion precedes the pointers; prepend it explicitly.
    _fields_ = [("iVersion", ctypes.c_int)] + _fields_


THE_MODULE = _Sqlite3Module()
THE_MODULE.iVersion = 1
THE_MODULE.xCreate = _xconnect.address
THE_MODULE.xConnect = _xconnect.address
THE_MODULE.xBestIndex = _xbestindex.address
THE_MODULE.xDisconnect = _xdisconnect.address
THE_MODULE.xDestroy = _xdisconnect.address
THE_MODULE.xOpen = _xopen.address
THE_MODULE.xClose = _xclose.address
THE_MODULE.xFilter = _xfilter.address
THE_MODULE.xNext = _xnext.address
THE_MODULE.xEof = _xeof.address
THE_MODULE.xColumn = _xcolumn.address
THE_MODULE.xRowid = _xrowid.address
_THE_MODULE_P = ctypes.addressof(THE_MODULE)
# keep the cfuncs alive for the process lifetime
_CFUNCS = (_xconnect, _xbestindex, _xdisconnect, _xopen, _xclose,
           _xfilter, _xnext, _xeof, _xcolumn, _xrowid)


class _VTableHandle:
    """Keeps the array + descriptor + buffers alive. SQLite reads the array
    buffer directly; if this handle is GC'd the data frees and the next query
    reads freed memory. The caller MUST retain it."""
    __slots__ = ("_keep",)

    def __init__(self, *objs):
        self._keep = objs


def _raise_rc(db, name, rc):
    msg_p = sqlite3_errmsg(db)
    detail = ""
    if msg_p:
        detail = ": " + ctypes.cast(msg_p, ctypes.c_char_p).value.decode("utf-8", "replace")
    raise RuntimeError("register_table failed for %r (rc=%d)%s" % (name, rc, detail))


def register_table(db, name, arr, columns=None, *, text_as_blob=False):
    """Expose a numpy array as a read-only eponymous SQLite virtual table.

    See docs/plans/2026-05-31-sqlite-vtable-design.md §8 for the full contract.
    The caller MUST retain the returned handle for as long as the table is used.
    """
    built = _build_descriptor(arr, columns, text_as_blob)
    with c_string(name) as name_p:
        rc = sqlite3_create_module(db, name_p, _THE_MODULE_P, ctypes.addressof(built.c))
    if rc != SQLITE_OK:
        _raise_rc(db, name, rc)
    return _VTableHandle(built)
```

- [ ] **Step 4: Add `sqlite3_free` import** — it lives in `_sqlite_exec.py`; add to the imports near the top of `_sqlite_vtable.py`: `from numbox.core.bindings._sqlite_exec import sqlite3_free`.

- [ ] **Step 5: Export from the package** — append to `numbox/core/bindings/__init__.py`: `from numbox.core.bindings._sqlite_vtable import *  # noqa: F401,F403`.

- [ ] **Step 6: Run the integration tests — expect PASS.** Likely first-run issues to debug: `@cfunc(..., cache=True)` kwarg acceptance; the `_Sqlite3Module` double-`_fields_` assignment (if ctypes rejects reassignment, build the field list first then assign once); `sqlite3_malloc(int32(...))` arg typing. Fix inline; re-run.

- [ ] **Step 7: Lint + commit** (`feat(sqlite): register_table — numpy arrays as read-only virtual tables`).

---

## Task 6: Full dtype coverage (float, mixed-width, bool, 'S', 'U', text_as_blob)

**Goal:** Drive every `xColumn` dtype branch through real queries, including a structured array with `'S'`/`'U'` and non-ASCII round-trips. Fix any dispatch bug inline.

**Files:**
- Modify (if bugs found): `numbox/core/bindings/_sqlite_vtable.py`
- Test: `test/core/test_sqlite_vtable.py`

**Acceptance Criteria:**
- [ ] float64 / float32 / int32 / int16 / uint32 / bool 2-D arrays read back exactly (floats within numpy equality).
- [ ] Structured array `[("t","U6"),("q","i4"),("p","f8"),("s","S4")]` reads each column; `'U'` round-trips non-ASCII (`"héllo"`, emoji); `'S'` NUL-trimmed.
- [ ] `text_as_blob=True` makes the `'S'` column read back as `bytes` and `typeof(col)=='blob'`.

**Verify:** `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py -k "dtype or structured or blob" --durations=20` → pass.

**Steps:**

- [ ] **Step 1: Write failing tests** (append; reuse `_open_memory`/`_fetchall`):

```python
import pytest


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int32, np.int16, np.uint32])
def test_numeric_dtype_roundtrip(dt):
    db = _open_memory()
    a = (np.arange(6, dtype=dt).reshape(3, 2) + 1)
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    got = _fetchall(db, "SELECT a, b FROM t ORDER BY a")
    exp = [tuple(row) for row in a.tolist()]
    assert got == exp
    sqlite3_close(db)


def test_bool_dtype():
    db = _open_memory()
    a = np.array([[True, False], [False, True]], dtype=np.bool_)
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    assert _fetchall(db, "SELECT a, b FROM t") == [(1, 0), (0, 1)]
    sqlite3_close(db)


def test_structured_text_and_unicode():
    db = _open_memory()
    dt = np.dtype([("t", "U6"), ("q", "i4"), ("p", "f8"), ("s", "S4")])
    a = np.array([("héllo", 3, 1.5, b"ab"), ("\U0001F600", 7, 2.0, b"cd")], dtype=dt)
    h = register_table(db, "trades", a)  # noqa: F841
    got = _fetchall(db, "SELECT t, q, p, s FROM trades")
    assert got[0] == ("héllo", 3, 1.5, "ab")
    assert got[1][0] == "\U0001F600"
    sqlite3_close(db)


def test_text_as_blob():
    db = _open_memory()
    dt = np.dtype([("s", "S3")])
    a = np.array([(b"xy",)], dtype=dt)
    h = register_table(db, "t", a, text_as_blob=True)  # noqa: F841
    assert _fetchall(db, "SELECT s FROM t") == [(b"xy",)]
    assert _fetchall(db, "SELECT typeof(s) FROM t") == [("blob",)]
    sqlite3_close(db)
```

- [ ] **Step 2: Run — debug any failing branch** (likely candidates: float32 widening, `'U'` width//4, `'S'` BLOB path). Fix in `_xcolumn`/`_build_descriptor`.

- [ ] **Step 3: Run — expect PASS. Lint + commit** (`test(sqlite): full dtype coverage for numpy-backed vtables`).

---

## Task 7: Layout & edge cases (F-order, strided, reversed, packed-'U', empty, rowid, JOIN, duplicate, lifetime, cache)

**Goal:** Validate the strides model and every edge the spec calls out. These mostly exercise existing code; fix bugs inline.

**Files:**
- Modify (if bugs found): `numbox/core/bindings/_sqlite_vtable.py`
- Test: `test/core/test_sqlite_vtable.py`

**Acceptance Criteria:**
- [ ] F-contiguous, non-contiguous slice, and reversed-row views all match their numpy equivalents.
- [ ] A packed structured dtype with a `'U'` field at an odd byte offset reads correctly (`load_unaligned` on both numeric and `'U'`).
- [ ] Empty table returns no rows / `COUNT(*)==0`; `rowid` equals the 0-based index; JOIN of two tables works; duplicate name raises; querying still works after `gc.collect()`.

**Verify:** `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_vtable.py -k "fortran or strided or reversed or packed or empty or rowid or join or duplicate or lifetime" --durations=20` → pass.

**Steps:**

- [ ] **Step 1: Write failing tests** (append):

```python
import gc


def test_fortran_order_matches_c():
    db = _open_memory()
    a = np.asfortranarray(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64))
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    assert _fetchall(db, "SELECT a, b FROM t ORDER BY a") == [(1, 2), (3, 4), (5, 6)]
    sqlite3_close(db)


def test_noncontiguous_slice():
    db = _open_memory()
    big = np.arange(40, dtype=np.int64).reshape(8, 5)
    view = big[::2, 1:4]
    h = register_table(db, "t", view, columns=["a", "b", "c"])  # noqa: F841
    assert _fetchall(db, "SELECT a, b, c FROM t") == [tuple(r) for r in view.tolist()]
    sqlite3_close(db)


def test_reversed_rows():
    db = _open_memory()
    a = np.array([[1], [2], [3]], dtype=np.int64)[::-1]
    h = register_table(db, "t", a, columns=["a"])  # noqa: F841
    assert _fetchall(db, "SELECT a FROM t") == [(3,), (2,), (1,)]
    sqlite3_close(db)


def test_packed_unicode_odd_offset():
    db = _open_memory()
    dt = np.dtype([("a", "i1"), ("u", "U4")])  # 'u' starts at byte offset 1 (packed)
    assert dt.fields["u"][1] == 1
    a = np.array([(1, "wörd")], dtype=dt)
    h = register_table(db, "t", a)  # noqa: F841
    assert _fetchall(db, "SELECT a, u FROM t") == [(1, "wörd")]
    sqlite3_close(db)


def test_empty_table():
    db = _open_memory()
    a = np.empty((0, 2), dtype=np.int64)
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    assert _fetchall(db, "SELECT * FROM t") == []
    assert _fetchall(db, "SELECT COUNT(*) FROM t") == [(0,)]
    sqlite3_close(db)


def test_rowid_is_zero_based():
    db = _open_memory()
    a = np.array([[10], [20], [30]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["a"])  # noqa: F841
    assert _fetchall(db, "SELECT rowid, a FROM t") == [(0, 10), (1, 20), (2, 30)]
    sqlite3_close(db)


def test_join_two_tables():
    db = _open_memory()
    a = np.array([[1, 100], [2, 200]], dtype=np.int64)
    b = np.array([[1, 7], [2, 9]], dtype=np.int64)
    h1 = register_table(db, "lhs", a, columns=["id", "v"])  # noqa: F841
    h2 = register_table(db, "rhs", b, columns=["id", "w"])  # noqa: F841
    got = _fetchall(db, "SELECT lhs.v, rhs.w FROM lhs JOIN rhs ON lhs.id = rhs.id ORDER BY lhs.id")
    assert got == [(100, 7), (200, 9)]
    sqlite3_close(db)


def test_duplicate_name_raises():
    db = _open_memory()
    a = np.array([[1]], dtype=np.int64)
    h = register_table(db, "dup", a, columns=["a"])  # noqa: F841
    with pytest.raises(RuntimeError):
        register_table(db, "dup", a, columns=["a"])
    sqlite3_close(db)


def test_handle_survives_gc():
    db = _open_memory()
    a = np.array([[5, 6]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    gc.collect()
    assert _fetchall(db, "SELECT a, b FROM t") == [(5, 6)]
    sqlite3_close(db)
```

- [ ] **Step 2: Run — debug.** Expect potential issues: negative-stride pointer math, packed-`'U'` width, empty-table `xColumn` never called (fine). Fix inline.

- [ ] **Step 3: `@cfunc(cache=True)` cold/warm subprocess test.** Add a test that runs the int64 table in a fresh subprocess twice and asserts both succeed and the numba cache dir does not grow between runs (mirror the phase-3 `test_xprocess_cache_no_growth` structure in `test/core/test_sqlite_udf_helpers.py` — read it and adapt). If `@cfunc(cache=True)` proves unavailable in the pinned numba, drop `cache=True` from the cfuncs and replace this test with a comment referencing the design note; the module still works (recompiles per process at import).

- [ ] **Step 4: Run — expect PASS. Lint + commit** (`test(sqlite): layout/edge coverage for numpy-backed vtables`).

---

## Task 8: Docs, and full-CI gate

**Goal:** Add the Sphinx automodule entry, and run the complete CI-equivalent gate locally.

**Files:**
- Modify: `docs/numbox.core.bindings.rst`
- Verify only: whole tree

**Acceptance Criteria:**
- [ ] `docs/numbox.core.bindings.rst` has a `_sqlite_vtable` automodule section (NOT added to the `_call_lib_func` family-list prose — it is a cfunc/codegen module, following the `_sqlite_udf_helpers` precedent).
- [ ] `flake8 --max-line-length=127 .` clean.
- [ ] Full `test/` suite passes.
- [ ] `cd docs && sphinx-build` exits 0 (warning count not increased).

**Verify:** the three commands below all succeed.

**Steps:**

- [ ] **Step 1: Add the automodule section** to `docs/numbox.core.bindings.rst`, mirroring the existing per-module blocks (e.g. the `_sqlite_udf_helpers` one):

```rst
.. automodule:: numbox.core.bindings._sqlite_vtable
   :members:
   :undoc-members:
   :show-inheritance:
```

Do NOT add `_sqlite_vtable` to the "Bindings module conventions" family list (that list is for thin `_call_lib_func` wrappers; this module generates cfuncs — same call the spec makes in §3.2).

- [ ] **Step 2: Full lint:** `/home/erik/projects/numbox/venv/bin/flake8 --max-line-length=127 .` → clean.

- [ ] **Step 3: Full suite** (clear caches first via the universal prefix): `/home/erik/projects/numbox/venv/bin/python -m pytest test/ --durations=20` → all pass.

- [ ] **Step 4: Docs build:** `cd /home/erik/projects/numbox/docs && /home/erik/projects/numbox/venv/bin/sphinx-build -b html . _build/html` → exit 0.

- [ ] **Step 5: Commit** (`docs(sqlite): document _sqlite_vtable virtual-table bindings`).

---

## Final state

`feat/sqlite-vtable` carries the phase-3 stack + 8 phase-4 commits. Before any fork PR: rebase onto fork `main` once phase 3 (#19) merges and `main` re-syncs (per the spec's "Base" note), so the phase-4 diff excludes phase-3 commits. The upstream-PR branch (excluding CLAUDE.md, docs/plans/**, fork CI) is built later by file-state reconstruction off `upstream/main`, per the numbox upstream workflow.
