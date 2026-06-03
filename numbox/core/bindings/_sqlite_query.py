"""query_to_array: collect SELECT results into a numpy structured array."""
import ctypes

import numpy as np
import numba
from numba import njit, carray
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
from numbox.utils.lowlevel import _cast_int_to_void_p, array_data_p, store_at, get_str_from_p_as_int

__all__ = ["query_to_array"]


@njit(**jit_options)
def _store_cell(out_data, addr_off, tag, width, stmt, j):
    """Write column ``j`` of the current row at ``out_data + addr_off``.

    The destination row is pre-zeroed by the caller, so a SQL NULL leaves an
    integer cell as 0 and a text/blob cell as empty; only float cells are
    overwritten with NaN.
    """
    addr = out_data + addr_off
    ctype = sqlite3_column_type(stmt, j)
    if ctype == SQLITE_NULL:
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
        utf8_to_utf32(sqlite3_column_text(stmt, j), sqlite3_column_bytes(stmt, j), addr, width // 4)
    elif tag == _TAG_S:
        nbytes = sqlite3_column_bytes(stmt, j)
        src = carray(_cast_int_to_void_p(sqlite3_column_text(stmt, j)), (nbytes,), dtype=np.uint8)
        dst = carray(_cast_int_to_void_p(addr), (width,), dtype=np.uint8)
        n = nbytes if nbytes < width else width
        for b in range(n):
            dst[b] = src[b]
        for b in range(n, width):
            dst[b] = 0
    elif tag == _TAG_BLOB:
        nbytes = sqlite3_column_bytes(stmt, j)
        src = carray(_cast_int_to_void_p(sqlite3_column_blob(stmt, j)), (nbytes,), dtype=np.uint8)
        dst = carray(_cast_int_to_void_p(addr), (width,), dtype=np.uint8)
        n = nbytes if nbytes < width else width
        for b in range(n):
            dst[b] = src[b]
        for b in range(n, width):
            dst[b] = 0


@njit(**jit_options)
def _query_core(stmt, ncols, offsets, tags, widths, itemsize, dt):
    """Step ``stmt`` to exhaustion, materialising rows into an NRT buffer that
    grows geometrically, then trim to the exact length and return an owned array.
    """
    cap = 16
    out = np.empty(cap, dt)
    n = 0
    while sqlite3_step(stmt) == SQLITE_ROW:
        if n == cap:
            cap = cap * 2
            new = np.empty(cap, dt)
            old_bytes = carray(_cast_int_to_void_p(array_data_p(out)), (n * itemsize,), dtype=np.uint8)
            new_bytes = carray(_cast_int_to_void_p(array_data_p(new)), (n * itemsize,), dtype=np.uint8)
            for b in range(n * itemsize):
                new_bytes[b] = old_bytes[b]
            out = new
        base = array_data_p(out) + n * itemsize
        row = carray(_cast_int_to_void_p(base), (itemsize,), dtype=np.uint8)
        for b in range(itemsize):
            row[b] = 0
        for j in range(ncols):
            _store_cell(base, offsets[j], tags[j], widths[j], stmt, j)
        n += 1
    res = np.empty(n, dt)
    src_bytes = carray(_cast_int_to_void_p(array_data_p(out)), (n * itemsize,), dtype=np.uint8)
    res_bytes = carray(_cast_int_to_void_p(array_data_p(res)), (n * itemsize,), dtype=np.uint8)
    for b in range(n * itemsize):
        res_bytes[b] = src_bytes[b]
    return res


def _raise_rc(db, rc):
    msg_p = sqlite3_errmsg(db)
    detail = ""
    if msg_p:
        detail = ": " + get_str_from_p_as_int(msg_p)
    raise RuntimeError("query_to_array failed (rc=%d)%s" % (rc, detail))


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
