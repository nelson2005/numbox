"""Collect the results of a SELECT into a numpy structured array (query_to_array)."""
import ctypes

import numpy as np
from numba import njit, carray
from numba.core.types import uint8

from numbox.core.bindings.sqlite.constants import SQLITE_ROW, SQLITE_NULL, SQLITE_OK, SQLITE_DONE
from numbox.core.bindings.sqlite.stmt import sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize
from numbox.core.bindings.sqlite.column import (
    sqlite3_column_count, sqlite3_column_type, sqlite3_column_int64, sqlite3_column_double,
    sqlite3_column_text, sqlite3_column_blob, sqlite3_column_bytes,
)
from numbox.core.bindings.sqlite.conn import sqlite3_errmsg
from numbox.core.bindings.sqlite._typemap import (
    _col_tag,
    _TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64, _TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64,
    _TAG_F32, _TAG_F64, _TAG_BOOL, _TAG_S, _TAG_U, _TAG_BLOB,
    _utf8_decode_one,
)
from numbox.core.configurations import jit_options
from numbox.utils.lowlevel import _cast_int_to_void_p

__all__ = ["query_to_array"]


@njit(**jit_options)
def _copy_bytes(dst, off, src, nbytes):
    """Copy ``nbytes`` bytes from the start of ``src`` into ``dst`` at ``off``."""
    for k in range(nbytes):
        dst[off + k] = src[k]


@njit(**jit_options)
def _put_unicode(buf, off, scratch8, src_p, nbytes, width_cp):
    """Decode the UTF-8 at ``src_p`` into up to ``width_cp`` UTF-32 code points and write them into ``buf`` at ``off``.

    The code points are written in the platform's native byte order, through a uint32 view of ``scratch8``, which
    matches numpy's 'U' dtype. The decoding uses the same ``_typemap._utf8_decode_one`` as ``utf8_to_utf32``. Unlike
    ``utf8_to_utf32``, this function writes into ``buf``, a tracked uint8 array view, instead of through a raw pointer,
    because the macOS-arm64 optimizer eliminates raw-pointer stores as dead code. Malformed input decodes to U+FFFD.
    """
    inp = carray(_cast_int_to_void_p(src_p), (nbytes,), dtype=np.uint8)
    cps = scratch8.view(np.uint32)
    i = 0
    k = 0
    while i < nbytes and k < width_cp:
        cp, i = _utf8_decode_one(inp, i, nbytes)
        cps[0] = cp
        _copy_bytes(buf, off + 4 * k, scratch8, 4)
        k += 1


@njit(**jit_options)
def _store_cell(buf, off, tag, width, stmt, j, scratch8):
    """Write column ``j`` of the current row into the uint8 array view ``buf`` at byte offset ``off``.

    Scalars are serialized through ``scratch8``, a tracked 8-byte uint8 array, with a typed view, so the bytes are in
    the platform's native order before they are copied into ``buf``. These are all numba-tracked array writes, so the
    optimizer cannot remove them on macOS-arm64. The caller pre-zeroes the row (np.zeros). A SQL NULL therefore leaves
    an integer cell at 0 and a text or blob cell empty, and only a float cell is set to NaN. The text or blob accessor
    is called before column_bytes.
    """
    ctype = sqlite3_column_type(stmt, j)
    if ctype == SQLITE_NULL:
        if tag == _TAG_F32:
            scratch8.view(np.float32)[0] = np.float32(np.nan)
            _copy_bytes(buf, off, scratch8, 4)
        elif tag == _TAG_F64:
            scratch8.view(np.float64)[0] = np.float64(np.nan)
            _copy_bytes(buf, off, scratch8, 8)
        return
    if tag == _TAG_I8 or tag == _TAG_U8:
        buf[off] = uint8(sqlite3_column_int64(stmt, j))
    elif tag == _TAG_I16 or tag == _TAG_U16:
        scratch8.view(np.int16)[0] = np.int16(sqlite3_column_int64(stmt, j))
        _copy_bytes(buf, off, scratch8, 2)
    elif tag == _TAG_I32 or tag == _TAG_U32:
        scratch8.view(np.int32)[0] = np.int32(sqlite3_column_int64(stmt, j))
        _copy_bytes(buf, off, scratch8, 4)
    elif tag == _TAG_I64 or tag == _TAG_U64:
        scratch8.view(np.int64)[0] = np.int64(sqlite3_column_int64(stmt, j))
        _copy_bytes(buf, off, scratch8, 8)
    elif tag == _TAG_BOOL:
        buf[off] = uint8(1) if sqlite3_column_int64(stmt, j) != 0 else uint8(0)
    elif tag == _TAG_F32:
        scratch8.view(np.float32)[0] = np.float32(sqlite3_column_double(stmt, j))
        _copy_bytes(buf, off, scratch8, 4)
    elif tag == _TAG_F64:
        scratch8.view(np.float64)[0] = sqlite3_column_double(stmt, j)
        _copy_bytes(buf, off, scratch8, 8)
    elif tag == _TAG_U:
        _put_unicode(buf, off, scratch8, sqlite3_column_text(stmt, j), sqlite3_column_bytes(stmt, j), width // 4)
    elif tag == _TAG_S:
        src_p = sqlite3_column_text(stmt, j)
        nbytes = sqlite3_column_bytes(stmt, j)
        src = carray(_cast_int_to_void_p(src_p), (nbytes,), dtype=np.uint8)
        nn = nbytes if nbytes < width else width
        for b in range(nn):
            buf[off + b] = src[b]
    elif tag == _TAG_BLOB:
        src_p = sqlite3_column_blob(stmt, j)
        nbytes = sqlite3_column_bytes(stmt, j)
        src = carray(_cast_int_to_void_p(src_p), (nbytes,), dtype=np.uint8)
        nn = nbytes if nbytes < width else width
        for b in range(nn):
            buf[off + b] = src[b]


@njit(**jit_options)
def _query_core(stmt, ncols, offsets, tags, widths, itemsize):
    """Step ``stmt`` until it stops returning rows, and return the rows as native-order bytes.

    The rows are materialized into a flat uint8 numpy array that grows geometrically, and the array is then trimmed to
    the exact length. Returns ``(buf, rc)``, where ``buf`` is ``n * itemsize`` native-order bytes and ``rc`` is the
    terminal step return code (SQLITE_DONE on success). The caller views ``buf`` back to the structured dtype.
    """
    cap = 16
    out_u8 = np.zeros(cap * itemsize, np.uint8)
    scratch8 = np.empty(8, np.uint8)
    n = 0
    rc = sqlite3_step(stmt)
    while rc == SQLITE_ROW:
        if n == cap:
            cap = cap * 2
            new_u8 = np.zeros(cap * itemsize, np.uint8)
            new_u8[:n * itemsize] = out_u8[:n * itemsize]
            out_u8 = new_u8
        row_off = n * itemsize
        for j in range(ncols):
            _store_cell(out_u8, row_off + offsets[j], tags[j], widths[j], stmt, j, scratch8)
        n += 1
        rc = sqlite3_step(stmt)
    # The slice alone would be lifetime-safe, because it shares the NRT meminfo of the parent through
    # make_view/impl_ret_borrowed. The .copy() call only trims the unused capacity, whose absolute size is under one
    # doubling. That capacity is less than n rows (n * itemsize bytes) when n > 16, and at most 16 rows when n <= 16,
    # because cap is never below 16.
    return out_u8[:n * itemsize].copy(), rc


def _raise_rc(db, rc):
    msg_p = sqlite3_errmsg(db)
    detail = ""
    if msg_p:
        detail = ": " + ctypes.cast(msg_p, ctypes.c_char_p).value.decode("utf-8", "replace")
    raise RuntimeError("query_to_array failed (rc=%d)%s" % (rc, detail))


def query_to_array(db, sql_p, dtype):
    """Run the SQL text at ``sql_p`` on ``db`` and return its rows as a 1-D numpy structured array of ``dtype``.

    The array has one field per result column, matched by position. ``sql_p`` is a char* pointer to NUL-terminated
    SQL text (e.g. from ``numbox.utils.cstrings.c_string`` or ``get_unicode_data_p``), not a Python str. A SQL NULL
    becomes NaN in a float field, 0 in an integer field and an empty value in a text or blob field.
    """
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
        buf, last_rc = _query_core(stmt.value, ncols, offsets, tags, widths, int(dtype.itemsize))
        if last_rc != SQLITE_DONE:
            _raise_rc(db, last_rc)
        return buf.view(dtype)
    finally:
        sqlite3_finalize(stmt.value)
