"""Tests for the structref-backed SQLite UDAF/window registration helpers."""
import gc
from ctypes import addressof, c_int64

import numpy as np
from numba import carray, cfunc, njit, types
from numba.core import types as nb_types
from numba.experimental import structref

from numbox.core.bindings import (
    SQLITE_OK,
    SQLITE_UTF8,
    sqlite3_close,
    sqlite3_create_function_v2,
    sqlite3_exec,
    sqlite3_open,
    sqlite3_result_int,
    sqlite3_result_int64,
    sqlite3_user_data,
    sqlite3_value_int64,
)
from numbox.core.bindings._sqlite_udf_helpers import register_aggregate
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p


# --- state type (module-level => importable, stable __module__) ---
@structref.register
class SumStateType(nb_types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)


class SumState(structref.StructRefProxy):
    def __new__(cls, total):
        return structref.StructRefProxy.__new__(cls, total)


structref.define_proxy(SumState, SumStateType, ["total"])
sum_state_type = SumStateType([("total", nb_types.int64)])


@njit
def sum_init():
    return SumState(nb_types.int64(0))


@njit
def sum_step(state, ctx, argc, argv_pp):
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total += sqlite3_value_int64(args[0])


@njit
def sum_finalize(state, ctx):
    sqlite3_result_int64(ctx, state.total)


# --- test plumbing ---
def _open_memory():
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    return db_p.value


def _make_table(db, values):
    with c_string("CREATE TABLE t(v INTEGER)") as p:
        assert sqlite3_exec(db, p, 0, 0, 0) == SQLITE_OK
    for v in values:
        with c_string("INSERT INTO t VALUES (%d)" % v) as p:
            assert sqlite3_exec(db, p, 0, 0, 0) == SQLITE_OK


def _read1_int64(db, select_sql):
    """Route a scalar SELECT through a capture UDF that stores arg0 into a
    numpy buffer; returns (value, keepalive)."""
    buf = np.zeros(1, dtype=np.int64)

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def cap_cb(ctx, argc, argv):
        ud = sqlite3_user_data(ctx)
        args = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
        o = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
        o[0] = sqlite3_value_int64(args[0])
        sqlite3_result_int(ctx, 0)

    with c_string("__cap") as cp:
        assert sqlite3_create_function_v2(
            db, cp, 1, SQLITE_UTF8, buf.ctypes.data, cap_cb.address, 0, 0, 0) == SQLITE_OK
    with c_string(select_sql) as sp:
        assert sqlite3_exec(db, sp, 0, 0, 0) == SQLITE_OK
    return int(buf[0]), cap_cb


def test_aggregate_sum():
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    h = register_aggregate(db, "my_sum", 1, sum_state_type,
                           sum_init, sum_step, sum_finalize)
    gc.collect()  # handle must keep callbacks alive
    val, _cap = _read1_int64(db, "SELECT __cap(my_sum(v)) FROM t")
    sqlite3_close(db)
    assert val == 15
    assert h is not None


def test_aggregate_empty_group():
    db = _open_memory()
    _make_table(db, [])
    h = register_aggregate(db, "my_sum", 1, sum_state_type,
                           sum_init, sum_step, sum_finalize)
    val, _cap = _read1_int64(db, "SELECT __cap(my_sum(v)) FROM t")
    sqlite3_close(db)
    assert val == 0
    assert h is not None


def test_aggregate_bad_state_type():
    import pytest
    db = _open_memory()
    with pytest.raises(TypeError):
        register_aggregate(db, "bad", 1, object(), sum_init, sum_step, sum_finalize)
    sqlite3_close(db)
