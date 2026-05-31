"""Expose a numpy array as a read-only SQLite virtual table (register_table).

A single generic sqlite3_module (built once at import) serves every table; the
per-table base pointer / strides / dtype tags / schema live in a ctypes
descriptor passed as pClientData. See
docs/plans/2026-05-31-sqlite-vtable-design.md.
"""
import ctypes

import numpy as np
from numba import carray, njit
from numba.core.types import uint8, uint32

from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.proxy.proxy import proxy
from numbox.utils.lowlevel import _cast_int_to_void_p, load_unaligned

load_lib("sqlite3")

# dtype tags (col_tags[j])
_TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64 = 0, 1, 2, 3
_TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64 = 4, 5, 6, 7
_TAG_F32, _TAG_F64, _TAG_BOOL = 8, 9, 10
_TAG_S, _TAG_U, _TAG_BLOB = 11, 12, 13

# descriptor field byte offsets (mirror of _NdarrayTableDescriptor)
_D_NROWS, _D_NCOLS, _D_ROW_STRIDE, _D_DATA_BASE = 0, 8, 16, 24
_D_COL_OFFSETS, _D_COL_TAGS, _D_COL_WIDTHS, _D_SCHEMA, _D_SCRATCH = 32, 40, 48, 56, 64

# vtab layout: base sqlite3_vtab is 24 bytes; the descriptor_ptr is appended at +24
_VTAB_DESC, _VTAB_SIZE = 24, 32
# cursor layout: pVtab(+0), descriptor(+8), rowid(+16), scratch(+24)
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


@njit(cache=True)
def _nul_trimmed_len(p, width):
    buf = carray(_cast_int_to_void_p(p), (width,), dtype=np.uint8)
    n = 0
    while n < width and buf[n] != 0:
        n += 1
    return n


@njit(cache=True)
def utf32_to_utf8(src, n_codepoints, dst):
    """Encode up to ``n_codepoints`` UTF-32 code points at ``src`` into UTF-8 at ``dst``.

    Stops at the first NUL code point. Returns the number of UTF-8 bytes written; the
    output is not NUL-terminated (callers pass this count explicitly, e.g. to
    ``sqlite3_result_text``). ``dst`` must hold at least ``4 * n_codepoints`` bytes.
    """
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
    tags_buf = np.array(tags, dtype=np.int32)  # the xColumn cfunc reads int32 elements; do not widen
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
