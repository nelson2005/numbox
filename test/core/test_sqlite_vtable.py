import ctypes
from ctypes import addressof, c_int64, cast, c_char_p, string_at

import pytest
import numpy as np
from numbox.utils.cstrings import c_string
from numbox.core.bindings import (
    sqlite3_open, sqlite3_close, register_table,
    sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize,
    sqlite3_column_count, sqlite3_column_type,
    sqlite3_column_int64, sqlite3_column_double,
    sqlite3_column_text, sqlite3_column_blob, sqlite3_column_bytes,
)
from numbox.core.bindings._sqlite_vtable import utf32_to_utf8, _nul_trimmed_len
from numbox.core.bindings._sqlite_vtable import _build_descriptor, _TAG_I64, _TAG_F64, _TAG_U
from numbox.utils.lowlevel import array_data_p


def test_imports():
    from numbox.core.bindings import _sqlite_vtable as v
    assert hasattr(v.sqlite3_create_module, "py_func")
    assert hasattr(v.sqlite3_declare_vtab, "py_func")
    assert hasattr(v.sqlite3_malloc, "py_func")


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
    assert d.scratch_bytes == 6 * 4 + 1
    assert d.schema == b"CREATE TABLE x(t TEXT, q INTEGER, p REAL)\x00"


def test_descriptor_rejects_bad_shapes():
    import pytest
    with pytest.raises((TypeError, ValueError)):
        _build_descriptor(np.zeros((2, 2, 2), dtype=np.int64), ["a", "b"], False)
    with pytest.raises((TypeError, ValueError)):
        _build_descriptor(np.arange(4, dtype=np.int64), ["a"], False)
    with pytest.raises((TypeError, ValueError)):
        _build_descriptor(np.zeros((2, 3), dtype=np.int64), ["a", "b"], False)
    with pytest.raises((TypeError, ValueError)):
        _build_descriptor(np.empty((2, 2), dtype=object), ["a", "b"], False)


def test_descriptor_offsets_assertion_holds():
    from numbox.core.bindings._sqlite_vtable import _NdarrayTableDescriptor
    assert ctypes.sizeof(_NdarrayTableDescriptor) == 72


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
    h = register_table(db, "points", a, columns=["a", "b"])  # noqa: F841
    assert _fetchall(db, "SELECT a, b FROM points WHERE a >= 2 ORDER BY b DESC") == [(3, 30), (2, 20)]
    sqlite3_close(db)


def test_count_and_sum():
    db = _open_memory()
    a = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.int64)
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    assert _fetchall(db, "SELECT COUNT(*), SUM(a) FROM t") == [(3, 6)]
    sqlite3_close(db)


@pytest.mark.parametrize("dt", [np.float64, np.float32, np.int32, np.int16, np.uint32, np.int8, np.uint8, np.uint16])
def test_numeric_dtype_roundtrip(dt):
    db = _open_memory()
    a = (np.arange(6, dtype=dt).reshape(3, 2) + 1)
    h = register_table(db, "t", a, columns=["a", "b"])  # noqa: F841
    got = _fetchall(db, "SELECT a, b FROM t ORDER BY a")
    exp = [tuple(row) for row in a.tolist()]
    assert got == exp
    sqlite3_close(db)


def test_uint64_roundtrip_and_signed_wrap():
    db = _open_memory()
    a = np.array([[1], [2 ** 63], [2 ** 64 - 1]], dtype=np.uint64)
    h = register_table(db, "t", a, columns=["a"])  # noqa: F841
    # SQLite INTEGER is a signed int64; uint64 values >= 2**63 reinterpret as negative.
    assert _fetchall(db, "SELECT a FROM t") == [(1,), (-(2 ** 63),), (-1,)]
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
    assert got == [("héllo", 3, 1.5, "ab"), ("\U0001F600", 7, 2.0, "cd")]
    sqlite3_close(db)


def test_text_as_blob():
    db = _open_memory()
    dt = np.dtype([("s", "S3")])
    a = np.array([(b"xy",)], dtype=dt)
    h = register_table(db, "t", a, text_as_blob=True)  # noqa: F841
    assert _fetchall(db, "SELECT s FROM t") == [(b"xy",)]
    assert _fetchall(db, "SELECT typeof(s) FROM t") == [("blob",)]
    sqlite3_close(db)
