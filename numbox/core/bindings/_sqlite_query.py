"""query_to_array: collect SELECT results into a numpy structured array."""
import ctypes

import numpy as np
import numba
from numba import njit, carray
from numba.core.types import uint8, uint32, uint64

from numbox.core.bindings._sqlite_constants import SQLITE_ROW, SQLITE_NULL, SQLITE_OK, SQLITE_DONE
from numbox.core.bindings import (
    sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize, sqlite3_column_count,
    sqlite3_column_type, sqlite3_column_int64, sqlite3_column_double,
    sqlite3_column_text, sqlite3_column_blob, sqlite3_column_bytes, sqlite3_errmsg,
)
from numbox.core.bindings._sqlite_typemap import (
    _col_tag,
    _TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64, _TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64,
    _TAG_F32, _TAG_F64, _TAG_BOOL, _TAG_S, _TAG_U, _TAG_BLOB,
)
from numbox.core.configurations import jit_options
from numbox.utils.lowlevel import _cast_int_to_void_p

__all__ = ["query_to_array"]


@njit(**jit_options)
def _put_int_le(buf, off, v, nbytes):
    """Write the low ``nbytes`` little-endian bytes of integer ``v`` into ``buf``."""
    uv = uint64(v)
    for k in range(nbytes):
        buf[off + k] = uint8((uv >> uint64(8 * k)) & uint64(0xFF))


@njit(**jit_options)
def _put_unicode(buf, off, src_p, nbytes, width_cp):
    """Decode UTF-8 at ``src_p`` into up to ``width_cp`` UTF-32 code points and
    write their little-endian bytes into ``buf`` at ``off``. Mirrors
    ``_sqlite_typemap.utf8_to_utf32`` but writes natively into ``buf`` (a tracked
    uint8 array view) instead of a raw pointer -- raw-pointer stores get
    dead-code-eliminated by the macOS-arm64 optimizer. Malformed input -> U+FFFD."""
    inp = carray(_cast_int_to_void_p(src_p), (nbytes,), dtype=np.uint8)
    i = 0
    k = 0
    while i < nbytes and k < width_cp:
        b0 = uint32(inp[i])
        if b0 < 0x80:
            cp = b0
            i += 1
        elif b0 >> 5 == 0x6 and i + 1 < nbytes and (inp[i + 1] >> 6) == 0x2:
            cp = ((b0 & 0x1F) << 6) | (uint32(inp[i + 1]) & 0x3F)
            if cp < 0x80:
                cp = 0xFFFD
            i += 2
        elif b0 >> 4 == 0xE and i + 2 < nbytes and (inp[i + 1] >> 6) == 0x2 and (inp[i + 2] >> 6) == 0x2:
            cp = ((b0 & 0x0F) << 12) | ((uint32(inp[i + 1]) & 0x3F) << 6) | (uint32(inp[i + 2]) & 0x3F)
            if cp < 0x800 or (0xD800 <= cp <= 0xDFFF):
                cp = 0xFFFD
            i += 3
        elif (b0 >> 3 == 0x1E and i + 3 < nbytes and (inp[i + 1] >> 6) == 0x2
              and (inp[i + 2] >> 6) == 0x2 and (inp[i + 3] >> 6) == 0x2):
            cp = (((b0 & 0x07) << 18) | ((uint32(inp[i + 1]) & 0x3F) << 12)
                  | ((uint32(inp[i + 2]) & 0x3F) << 6) | (uint32(inp[i + 3]) & 0x3F))
            if cp < 0x10000 or cp > 0x10FFFF:
                cp = 0xFFFD
            i += 4
        else:
            cp = 0xFFFD
            i += 1
        bo = off + 4 * k
        buf[bo] = uint8(cp & 0xFF)
        buf[bo + 1] = uint8((cp >> 8) & 0xFF)
        buf[bo + 2] = uint8((cp >> 16) & 0xFF)
        buf[bo + 3] = uint8((cp >> 24) & 0xFF)
        k += 1


@njit(**jit_options)
def _store_cell(buf, off, tag, width, stmt, j, f32buf, f64buf):
    """Write column ``j`` of the current row into the uint8 array view ``buf`` at
    byte offset ``off``. All writes go through ``buf`` (numba-tracked) rather than
    a raw array_data_p pointer, so the optimizer cannot drop them on macOS-arm64.
    The row is pre-zeroed by the caller (np.zeros), so a SQL NULL leaves an integer
    cell 0 and a text/blob cell empty; only float cells are overwritten with NaN.
    The text/blob accessor is read before column_bytes (SQLite contract)."""
    ctype = sqlite3_column_type(stmt, j)
    if ctype == SQLITE_NULL:
        if tag == _TAG_F32:
            f32buf[0] = np.float32(np.nan)
            fb = f32buf.view(np.uint8)
            for k in range(4):
                buf[off + k] = fb[k]
        elif tag == _TAG_F64:
            f64buf[0] = np.float64(np.nan)
            fb = f64buf.view(np.uint8)
            for k in range(8):
                buf[off + k] = fb[k]
        return
    if tag == _TAG_I8 or tag == _TAG_U8:
        _put_int_le(buf, off, sqlite3_column_int64(stmt, j), 1)
    elif tag == _TAG_I16 or tag == _TAG_U16:
        _put_int_le(buf, off, sqlite3_column_int64(stmt, j), 2)
    elif tag == _TAG_I32 or tag == _TAG_U32:
        _put_int_le(buf, off, sqlite3_column_int64(stmt, j), 4)
    elif tag == _TAG_I64 or tag == _TAG_U64:
        _put_int_le(buf, off, sqlite3_column_int64(stmt, j), 8)
    elif tag == _TAG_BOOL:
        buf[off] = uint8(1) if sqlite3_column_int64(stmt, j) != 0 else uint8(0)
    elif tag == _TAG_F32:
        f32buf[0] = np.float32(sqlite3_column_double(stmt, j))
        fb = f32buf.view(np.uint8)
        for k in range(4):
            buf[off + k] = fb[k]
    elif tag == _TAG_F64:
        f64buf[0] = sqlite3_column_double(stmt, j)
        fb = f64buf.view(np.uint8)
        for k in range(8):
            buf[off + k] = fb[k]
    elif tag == _TAG_U:
        _put_unicode(buf, off, sqlite3_column_text(stmt, j), sqlite3_column_bytes(stmt, j), width // 4)
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
def _query_core(stmt, ncols, offsets, tags, widths, itemsize, dt):
    """Step ``stmt`` to exhaustion, materialising rows into an NRT buffer that
    grows geometrically, then trim to the exact length. Returns ``(array, rc)``
    where ``rc`` is the terminal step return code (SQLITE_DONE on success).
    """
    cap = 16
    out = np.zeros(cap, dt)
    out_u8 = out.view(np.uint8)
    f32buf = np.empty(1, np.float32)
    f64buf = np.empty(1, np.float64)
    n = 0
    rc = sqlite3_step(stmt)
    while rc == SQLITE_ROW:
        if n == cap:
            cap = cap * 2
            new = np.zeros(cap, dt)
            new_u8 = new.view(np.uint8)
            new_u8[:n * itemsize] = out_u8[:n * itemsize]
            out = new
            out_u8 = new_u8
        row_off = n * itemsize
        for j in range(ncols):
            _store_cell(out_u8, row_off + offsets[j], tags[j], widths[j], stmt, j, f32buf, f64buf)
        n += 1
        rc = sqlite3_step(stmt)
    return out[:n].copy(), rc


def _raise_rc(db, rc):
    msg_p = sqlite3_errmsg(db)
    detail = ""
    if msg_p:
        detail = ": " + ctypes.cast(msg_p, ctypes.c_char_p).value.decode("utf-8", "replace")
    raise RuntimeError("query_to_array failed (rc=%d)%s" % (rc, detail))


def query_to_array(db, sql_p, dtype):
    """Run the NUL-terminated SQL text at pointer ``sql_p`` on ``db`` and return
    its rows as a 1-D numpy structured array of ``dtype`` (one field per result
    column, by position). ``sql_p`` is a char* pointer (e.g. from
    ``numbox.utils.cstrings.c_string`` or ``get_unicode_data_p``), not a Python
    str. NULL -> NaN (float) / 0 (int) / empty (text/blob)."""
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
        res, last_rc = _query_core(stmt.value, ncols, offsets, tags, widths,
                                   int(dtype.itemsize), numba.from_dtype(dtype))
        if last_rc != SQLITE_DONE:
            _raise_rc(db, last_rc)
        return res
    finally:
        sqlite3_finalize(stmt.value)
