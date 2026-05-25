"""Result setter tests for the SQLite UDF buildout.

Each test registers a scalar UDF that calls a specific sqlite3_result_*
setter, then invokes it via sqlite3_exec and verifies the return code.
"""
from ctypes import addressof, c_int64

import numpy as np
from numba import carray, cfunc, njit, types

from numbox.core.bindings._sqlite_conn import sqlite3_close, sqlite3_open
from numbox.core.bindings._sqlite_constants import (
    SQLITE_OK,
    SQLITE_TRANSIENT,
    SQLITE_UTF8,
)
from numbox.core.bindings._sqlite_exec import sqlite3_exec
from numbox.core.bindings._sqlite_result import (
    sqlite3_result_double,
    sqlite3_result_error,
    sqlite3_result_int,
    sqlite3_result_int64,
    sqlite3_result_null,
    sqlite3_result_text,
    sqlite3_result_value,
    sqlite3_result_zeroblob,
)
from numbox.core.bindings._sqlite_udf import sqlite3_create_function_v2
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p, get_unicode_data_p


def _open_memory():
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        rc = sqlite3_open(name_p, addressof(db_p))
    assert rc == SQLITE_OK
    return db_p.value


# --- cfunc callbacks (module-level to outlive registrations) ---

@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_int_cb(ctx, argc, argv_pp):
    sqlite3_result_int(ctx, 42)


@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_int64_cb(ctx, argc, argv_pp):
    sqlite3_result_int64(ctx, 9999999999)


@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_double_cb(ctx, argc, argv_pp):
    sqlite3_result_double(ctx, 2.718)


@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_null_cb(ctx, argc, argv_pp):
    sqlite3_result_null(ctx)


@njit
def _result_text_impl(ctx, argc, argv_pp):
    s = "hello"
    p = get_unicode_data_p(s)
    sqlite3_result_text(ctx, p, 5, SQLITE_TRANSIENT)


_result_text_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_result_text_impl)


@njit
def _result_error_impl(ctx, argc, argv_pp):
    s = "fail"
    p = get_unicode_data_p(s)
    sqlite3_result_error(ctx, p, 4)


_result_error_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_result_error_impl)


@njit
def _result_value_impl(ctx, argc, argv_pp):
    argv_p = _cast_int_to_void_p(argv_pp)
    args = carray(argv_p, (argc,), dtype=np.intp)
    sqlite3_result_value(ctx, args[0])


_result_value_cb = cfunc(
    types.void(types.intp, types.int32, types.intp)
)(_result_value_impl)


@cfunc(types.void(types.intp, types.int32, types.intp))
def _result_zeroblob_cb(ctx, argc, argv_pp):
    sqlite3_result_zeroblob(ctx, 16)


def _register_and_exec(db_p, name, cb, sql, n_arg=0):
    """Register a scalar UDF and execute sql. Returns exec rc."""
    with c_string(name) as name_p:
        rc = sqlite3_create_function_v2(
            db_p, name_p, n_arg, SQLITE_UTF8, 0,
            cb.address, 0, 0, 0)
    assert rc == SQLITE_OK, f"create_function_v2 failed: rc={rc}"
    with c_string(sql) as sql_p:
        return sqlite3_exec(db_p, sql_p, 0, 0, 0)


def test_result_int():
    db_p = _open_memory()
    rc = _register_and_exec(db_p, "ri", _result_int_cb, "SELECT ri()")
    assert rc == SQLITE_OK
    sqlite3_close(db_p)


def test_result_int64():
    db_p = _open_memory()
    rc = _register_and_exec(db_p, "ri64", _result_int64_cb, "SELECT ri64()")
    assert rc == SQLITE_OK
    sqlite3_close(db_p)


def test_result_double():
    db_p = _open_memory()
    rc = _register_and_exec(db_p, "rd", _result_double_cb, "SELECT rd()")
    assert rc == SQLITE_OK
    sqlite3_close(db_p)


def test_result_null():
    db_p = _open_memory()
    rc = _register_and_exec(db_p, "rn", _result_null_cb, "SELECT rn()")
    assert rc == SQLITE_OK
    sqlite3_close(db_p)


def test_result_text_transient():
    db_p = _open_memory()
    rc = _register_and_exec(db_p, "rt", _result_text_cb, "SELECT rt()")
    assert rc == SQLITE_OK
    sqlite3_close(db_p)


def test_result_error_aborts():
    db_p = _open_memory()
    rc = _register_and_exec(db_p, "re", _result_error_cb, "SELECT re()")
    assert rc != SQLITE_OK
    sqlite3_close(db_p)


def test_result_value_passthrough():
    db_p = _open_memory()
    rc = _register_and_exec(
        db_p, "rv", _result_value_cb, "SELECT rv(123)", n_arg=1)
    assert rc == SQLITE_OK
    sqlite3_close(db_p)


def test_result_zeroblob():
    db_p = _open_memory()
    rc = _register_and_exec(db_p, "rz", _result_zeroblob_cb, "SELECT rz()")
    assert rc == SQLITE_OK
    sqlite3_close(db_p)


if __name__ == "__main__":
    from test.auxiliary_utils import collect_and_run_tests
    collect_and_run_tests(__name__)
