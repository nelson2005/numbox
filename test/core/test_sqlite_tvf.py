from ctypes import addressof, c_int64

import pytest
import numpy as np
from numba import njit

from numbox.utils.cstrings import c_string
from numbox.core.bindings import (
    sqlite3_open, sqlite3_close, register_tvf,
    sqlite3_prepare_v2, sqlite3_step, sqlite3_finalize,
    sqlite3_column_int64, sqlite3_column_double,
)

_OUT = np.dtype([("n", "i8")])
_OUT2 = np.dtype([("n", "i8"), ("v", "f8")])


@njit
def _series(start, stop):
    out = np.empty(stop - start, _OUT)
    for i in range(stop - start):
        out[i].n = start + i
    return out


@njit
def _scaled(start, stop, scale):
    out = np.empty(stop - start, _OUT2)
    for i in range(stop - start):
        out[i].n = start + i
        out[i].v = (start + i) * scale
    return out


def _open():
    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    return db


def _select_int(db, sql, ncol=1):
    stmt = c_int64(0)
    with c_string(sql) as p:
        rc = sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    rows = []
    if rc == 0:
        while sqlite3_step(stmt.value) == 100:
            rows.append(tuple(sqlite3_column_int64(stmt.value, j) for j in range(ncol)))
    sqlite3_finalize(stmt.value)
    return rc, rows


def test_tvf_series():
    db = _open()
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


def test_tvf_two_calls_same_process():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    _, r1 = _select_int(db, "SELECT n FROM series(2, 5)")
    _, r2 = _select_int(db, "SELECT n FROM series(10, 13)")
    assert [x[0] for x in r1] == [2, 3, 4]
    assert [x[0] for x in r2] == [10, 11, 12]
    sqlite3_close(db.value)
    del h


def test_tvf_two_calls_one_query_plan():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    stmt = c_int64(0)
    sql = "SELECT a.n, b.n FROM series(2, 5) a, series(10, 13) b WHERE a.n = 2"
    with c_string(sql) as p:
        sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    got = []
    while sqlite3_step(stmt.value) == 100:
        got.append((sqlite3_column_int64(stmt.value, 0), sqlite3_column_int64(stmt.value, 1)))
    sqlite3_finalize(stmt.value)
    assert got == [(2, 10), (2, 11), (2, 12)]
    sqlite3_close(db.value)
    del h


def test_tvf_multi_column_and_float_arg():
    db = _open()
    h = register_tvf(db.value, "scaled", (np.int64, np.int64, np.float64), _OUT2, _scaled)
    stmt = c_int64(0)
    with c_string("SELECT n, v FROM scaled(0, 3, 2.5)") as p:
        sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    got = []
    while sqlite3_step(stmt.value) == 100:
        got.append((sqlite3_column_int64(stmt.value, 0), sqlite3_column_double(stmt.value, 1)))
    sqlite3_finalize(stmt.value)
    assert got == [(0, 0.0), (1, 2.5), (2, 5.0)]
    sqlite3_close(db.value)
    del h


def test_tvf_missing_hidden_arg():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    stmt = c_int64(0)
    with c_string("SELECT n FROM series WHERE n < 100") as p:
        rc = sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    rows = []
    if rc == 0:
        while sqlite3_step(stmt.value) == 100:
            rows.append(sqlite3_column_int64(stmt.value, 0))
    sqlite3_finalize(stmt.value)
    assert rows == []
    sqlite3_close(db.value)
    del h


def test_tvf_no_meminfo_leak():
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        pytest.skip("NRT allocation stats unavailable")
    _nrt.memsys_enable_stats()
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    _select_int(db, "SELECT n FROM series(2, 5)")
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        _select_int(db, "SELECT n FROM series(2, 5)")
    after = nrt.rtsys.get_allocation_stats()
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    sqlite3_close(db.value)
    del h
    assert allocated == freed, "meminfo imbalance: %d alloc, %d free" % (allocated, freed)


@njit
def _series_sliced(start, stop):
    # returns an offset slice: logical start != allocation base
    out = np.empty((stop - start) + 1, _OUT)
    for i in range((stop - start) + 1):
        out[i].n = (start - 1) + i
    return out[1:]


@njit
def _series_strided(start, stop):
    # returns a strided view: row stride = 2 * itemsize
    out = np.empty(2 * (stop - start), _OUT)
    for i in range(2 * (stop - start)):
        out[i].n = -1
    for i in range(stop - start):
        out[2 * i].n = start + i
    return out[::2]


@njit
def _series_empty(start, stop):
    return np.empty(0, _OUT)


def test_tvf_offset_slice_return():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series_sliced)
    _, rows = _select_int(db, "SELECT n FROM series(2, 5)")
    assert [x[0] for x in rows] == [2, 3, 4]
    sqlite3_close(db.value)
    del h


def test_tvf_strided_return():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series_strided)
    _, rows = _select_int(db, "SELECT n FROM series(2, 5)")
    assert [x[0] for x in rows] == [2, 3, 4]
    sqlite3_close(db.value)
    del h


def test_tvf_empty_return():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series_empty)
    _, rows = _select_int(db, "SELECT n FROM series(2, 5)")
    assert rows == []
    sqlite3_close(db.value)
    del h


def test_tvf_non_numeric_arg_type_raises():
    db = _open()
    with pytest.raises(TypeError):
        register_tvf(db.value, "f", (np.dtype("U4"),), _OUT, _series)
    sqlite3_close(db.value)


@njit
def _raises(start, stop):
    # raise explicitly: numba's default boundscheck is off, so an OOB index would
    # corrupt rather than raise. The @cfunc boundary swallows this -> 0 rows.
    out = np.empty(stop - start, _OUT)
    if start < stop:
        raise ValueError("boom")
    return out


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_tvf_user_fn_raises_yields_no_rows():
    # the @cfunc boundary swallows the user fn's exception (surfaced as an
    # unraisable exception); the observable contract is just "no rows".
    db = _open()
    h = register_tvf(db.value, "boom", (np.int64, np.int64), _OUT, _raises)
    _, rows = _select_int(db, "SELECT n FROM boom(2, 5)")
    assert rows == []
    sqlite3_close(db.value)
    del h


def test_tvf_two_distinct_registrations_same_process():
    db = _open()
    h1 = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    h2 = register_tvf(db.value, "scaled", (np.int64, np.int64, np.float64), _OUT2, _scaled)
    _, r1 = _select_int(db, "SELECT n FROM series(2, 5)")
    stmt = c_int64(0)
    with c_string("SELECT n, v FROM scaled(0, 3, 2.5)") as p:
        sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    r2 = []
    while sqlite3_step(stmt.value) == 100:
        r2.append((sqlite3_column_int64(stmt.value, 0), sqlite3_column_double(stmt.value, 1)))
    sqlite3_finalize(stmt.value)
    assert [x[0] for x in r1] == [2, 3, 4]
    assert r2 == [(0, 0.0), (1, 2.5), (2, 5.0)]
    sqlite3_close(db.value)
    del h1, h2
