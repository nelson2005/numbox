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


import os                                            # noqa: E402
import subprocess                                    # noqa: E402
import sys                                           # noqa: E402
import textwrap                                      # noqa: E402


# A self-contained driver: defines a state type + sum UDAF in an importable
# module, registers it, runs SELECT sum(v), prints the result. {MULT} is the
# step multiplier (1 normally; flipped to prove invalidation).
_DRIVER = textwrap.dedent('''
    from ctypes import addressof, c_int64
    import numpy as np
    from numba import carray, cfunc, njit, types
    from numba.core import types as nb_types
    from numba.experimental import structref
    from numbox.core.bindings import (
        SQLITE_OK, SQLITE_UTF8, sqlite3_close, sqlite3_create_function_v2,
        sqlite3_exec, sqlite3_open, sqlite3_result_int, sqlite3_result_int64,
        sqlite3_user_data, sqlite3_value_int64)
    from numbox.core.bindings._sqlite_udf_helpers import register_aggregate
    from numbox.utils.cstrings import c_string
    from numbox.utils.lowlevel import _cast_int_to_void_p

    @structref.register
    class StT(nb_types.StructRef):
        def preprocess_fields(self, fields):
            return tuple((n, nb_types.unliteral(t)) for n, t in fields)
    class St(structref.StructRefProxy):
        def __new__(cls, total):
            return structref.StructRefProxy.__new__(cls, total)
    structref.define_proxy(St, StT, ["total"])
    st = StT([("total", nb_types.int64)])

    @njit
    def s_init():
        return St(nb_types.int64(0))
    @njit
    def s_step(state, ctx, argc, argv_pp):
        a = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
        state.total += {MULT} * sqlite3_value_int64(a[0])
    @njit
    def s_fin(state, ctx):
        sqlite3_result_int64(ctx, state.total)

    db_p = c_int64(0)
    with c_string(":memory:") as n:
        sqlite3_open(n, addressof(db_p))
    db = db_p.value
    with c_string("CREATE TABLE t(v INTEGER)") as p:
        sqlite3_exec(db, p, 0, 0, 0)
    for v in (1, 2, 3, 4, 5):
        with c_string("INSERT INTO t VALUES (%d)" % v) as p:
            sqlite3_exec(db, p, 0, 0, 0)
    h = register_aggregate(db, "f", 1, st, s_init, s_step, s_fin)
    buf = np.zeros(1, dtype=np.int64)
    @cfunc(types.void(types.intp, types.int32, types.intp))
    def cap(ctx, argc, argv):
        ud = sqlite3_user_data(ctx)
        a = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
        o = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
        o[0] = sqlite3_value_int64(a[0]); sqlite3_result_int(ctx, 0)
    with c_string("cap") as cp:
        sqlite3_create_function_v2(db, cp, 1, SQLITE_UTF8, buf.ctypes.data, cap.address, 0, 0, 0)
    with c_string("SELECT cap(f(v)) FROM t") as sp:
        sqlite3_exec(db, sp, 0, 0, 0)
    print("RESULT", int(buf[0]))
''')


def _run_driver(tmp_path, cache_dir, mult):
    script = tmp_path / ("drv_%d.py" % mult)
    script.write_text(_DRIVER.replace("{MULT}", str(mult)))
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache_dir))
    out = subprocess.run([sys.executable, str(script)], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][0]
    return int(line.split()[1])


def _count_nbc(cache_dir):
    # Count only the generated UDAF impl caches (anchor stem "udaf_"/"wudaf_"),
    # not the whole cache: scoping keeps the no-growth assertion immune to
    # unrelated bindings whose compile timing could differ across the matrix.
    return sum(1 for _ in cache_dir.rglob("*udaf*.nbc"))


def test_xprocess_cache_no_growth(tmp_path):
    cache = tmp_path / "nbcache"
    cache.mkdir()
    assert _run_driver(tmp_path, cache, 1) == 15      # cold: compiles + writes cache
    n_cold = _count_nbc(cache)
    assert n_cold > 0
    assert _run_driver(tmp_path, cache, 1) == 15      # warm: must reuse, not append
    assert _count_nbc(cache) == n_cold, "warm run grew the cache (C failure mode)"


def test_invalidation_on_literal_edit(tmp_path):
    cache = tmp_path / "nbcache"
    cache.mkdir()
    assert _run_driver(tmp_path, cache, 1) == 15       # step: += 1*v  => 15
    assert _run_driver(tmp_path, cache, 3) == 45       # step: += 3*v  => 45 (not stale 15)


from numbox.core.bindings._sqlite_udf_helpers import register_window  # noqa: E402


@njit
def w_inverse(state, ctx, argc, argv_pp):
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total -= sqlite3_value_int64(args[0])


@njit
def w_value(state, ctx):
    sqlite3_result_int64(ctx, state.total)


def _read_window(db, select_sql, nrows):
    meta = np.zeros(nrows + 1, dtype=np.int64)  # meta[0]=count, meta[1:]=values

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def wcap_cb(ctx, argc, argv):
        ud = sqlite3_user_data(ctx)
        m = carray(_cast_int_to_void_p(ud), (nrows + 1,), dtype=np.int64)
        i = m[0]
        args = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
        m[1 + i] = sqlite3_value_int64(args[0])
        m[0] = i + 1
        sqlite3_result_int(ctx, 0)

    with c_string("__wcap") as cp:
        assert sqlite3_create_function_v2(
            db, cp, 1, SQLITE_UTF8, meta.ctypes.data, wcap_cb.address, 0, 0, 0) == SQLITE_OK
    with c_string(select_sql) as sp:
        assert sqlite3_exec(db, sp, 0, 0, 0) == SQLITE_OK
    return [int(meta[1 + i]) for i in range(int(meta[0]))], wcap_cb


def test_window_running_sum():
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    h = register_window(db, "my_wsum", 1, sum_state_type,
                        sum_init, sum_step, w_inverse, w_value, sum_finalize)
    sql = ("SELECT __wcap(my_wsum(v) OVER "
           "(ORDER BY v ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)) "
           "FROM t ORDER BY v")
    vals, _cap = _read_window(db, sql, 5)
    sqlite3_close(db)
    assert vals == [1, 3, 5, 7, 9]
    assert h is not None


@njit
def sum2_step(state, ctx, argc, argv_pp):  # distinct body => independent compiled callbacks
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total += 2 * sqlite3_value_int64(args[0])


def test_two_distinct_aggregates_no_collision():
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    h1 = register_aggregate(db, "my_sum", 1, sum_state_type,
                            sum_init, sum_step, sum_finalize)
    h2 = register_aggregate(db, "my_sum2", 1, sum_state_type,
                            sum_init, sum2_step, sum_finalize)
    v1, _c1 = _read1_int64(db, "SELECT __cap(my_sum(v)) FROM t")
    v2, _c2 = _read1_int64(db, "SELECT __cap(my_sum2(v)) FROM t")
    sqlite3_close(db)
    assert (v1, v2) == (15, 30)
    assert h1 is not None and h2 is not None


def test_deterministic_flag():
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    h = register_aggregate(db, "my_sum_det", 1, sum_state_type,
                           sum_init, sum_step, sum_finalize, deterministic=True)
    val, _cap = _read1_int64(db, "SELECT __cap(my_sum_det(v)) FROM t")
    sqlite3_close(db)
    assert val == 15
    assert h is not None


def test_deterministic_flag_ors_bit(monkeypatch):
    """deterministic=True must OR SQLITE_DETERMINISTIC into the flags passed to
    sqlite3_create_function_v2, and the default must not. The flag is a
    query-planner hint with no effect on an aggregate's value, so a result
    assertion alone cannot guard this contract -- spy on the flags instead."""
    import numbox.core.bindings._sqlite_udf_helpers as helpers
    real = helpers.sqlite3_create_function_v2
    seen = []

    def spy(db, name_p, n_arg, flags, *rest):
        seen.append(flags)
        return real(db, name_p, n_arg, flags, *rest)

    monkeypatch.setattr(helpers, "sqlite3_create_function_v2", spy)
    db = _open_memory()
    register_aggregate(db, "det_on", 1, sum_state_type,
                       sum_init, sum_step, sum_finalize, deterministic=True)
    register_aggregate(db, "det_off", 1, sum_state_type,
                       sum_init, sum_step, sum_finalize)
    sqlite3_close(db)
    assert seen[0] & helpers.SQLITE_DETERMINISTIC
    assert not (seen[1] & helpers.SQLITE_DETERMINISTIC)


def test_no_meminfo_leak():
    """The helper-generated lifecycle must preserve export/release balance."""
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")
    _nrt.memsys_enable_stats()
    test_aggregate_sum()          # warm up JIT / one-time allocs
    test_window_running_sum()
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        test_aggregate_sum()
        test_window_running_sum()
    after = nrt.rtsys.get_allocation_stats()
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, "meminfo imbalance: %d alloc, %d free" % (allocated, freed)
