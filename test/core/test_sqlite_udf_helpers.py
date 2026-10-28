"""The tests cover the structref-backed SQLite registration helpers for UDAFs and window functions."""
import gc
from ctypes import addressof, c_char_p, c_int64

import numpy as np
from numba import carray, cfunc, njit, types
from numba.core import types as nb_types
from numba.experimental import structref

from numbox.core.bindings.sqlite.constants import SQLITE_OK, SQLITE_UTF8
from numbox.core.bindings.sqlite.udf_helpers import register_aggregate, register_window
from numbox.core.bindings.sqlite.conn import sqlite3_close, sqlite3_errmsg, sqlite3_open
from numbox.core.bindings.sqlite.udf import sqlite3_create_function_v2, sqlite3_user_data
from numbox.core.bindings.sqlite.exec import sqlite3_exec
from numbox.core.bindings.sqlite.result import sqlite3_result_int, sqlite3_result_int64
from numbox.core.bindings.sqlite.value import sqlite3_value_int64
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p


# --- State type, defined at module level so that it is importable and has a stable __module__ ---
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
    return SumState(np.int64(0))


@njit
def sum_step(state, ctx, argc, argv_pp):
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total += sqlite3_value_int64(args[0])


@njit
def sum_finalize(state, ctx):
    sqlite3_result_int64(ctx, state.total)


# --- Test helpers ---
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


def _errmsg(db):
    """Decode the most recent error string, sqlite3_errmsg(db), to a str."""
    p = sqlite3_errmsg(db)
    return c_char_p(p).value.decode("utf-8", "replace") if p else ""


def _read1_int64(db, select_sql):
    """Route a scalar SELECT through a capture UDF that stores arg0 into a numpy buffer, and return
    (value, keepalive)."""
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
    register_aggregate(db, "my_sum", 1, sum_state_type, sum_init, sum_step, sum_finalize)
    gc.collect()  # No handle is retained. The callbacks survive GC because numba keeps the JIT code alive.
    val, _cap = _read1_int64(db, "SELECT __cap(my_sum(v)) FROM t")
    sqlite3_close(db)
    assert val == 15


def test_aggregate_empty_group():
    db = _open_memory()
    _make_table(db, [])
    register_aggregate(db, "my_sum", 1, sum_state_type, sum_init, sum_step, sum_finalize)
    val, _cap = _read1_int64(db, "SELECT __cap(my_sum(v)) FROM t")
    sqlite3_close(db)
    assert val == 0


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


# _DRIVER is a self-contained driver. It defines a state type and a sum UDAF in an importable module, registers the
# UDAF, runs SELECT sum(v) and prints the result. {MULT} is the step multiplier: it is 1 normally, and a test changes
# it to prove cache invalidation.
_DRIVER = textwrap.dedent('''
    from ctypes import addressof, c_int64
    import numpy as np
    from numba import carray, cfunc, njit, types
    from numba.core import types as nb_types
    from numba.experimental import structref
    from numbox.core.bindings.sqlite.constants import SQLITE_OK, SQLITE_UTF8
    from numbox.core.bindings.sqlite.conn import sqlite3_close, sqlite3_open
    from numbox.core.bindings.sqlite.udf import sqlite3_create_function_v2, sqlite3_user_data
    from numbox.core.bindings.sqlite.exec import sqlite3_exec
    from numbox.core.bindings.sqlite.result import sqlite3_result_int, sqlite3_result_int64
    from numbox.core.bindings.sqlite.value import sqlite3_value_int64
    from numbox.core.bindings.sqlite.udf_helpers import register_aggregate
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
        return St(np.int64(0))
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
    register_aggregate(db, "f", 1, st, s_init, s_step, s_fin)
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
    # The count covers only the caches of the generated UDAF implementations (anchor stem "udaf_" or "wudaf_"), not the
    # whole cache. This limit prevents unrelated bindings from affecting the no-growth assertion, because their compile
    # timing could differ across the matrix.
    return sum(1 for _ in cache_dir.rglob("*udaf*.nbc"))


def test_xprocess_cache_no_growth(tmp_path):
    cache = tmp_path / "nbcache"
    cache.mkdir()
    assert _run_driver(tmp_path, cache, 1) == 15      # The cold run compiles and writes the cache
    n_cold = _count_nbc(cache)
    assert n_cold > 0
    assert _run_driver(tmp_path, cache, 1) == 15      # The warm run must reuse the cache, not append to it
    assert _count_nbc(cache) == n_cold, "warm run grew the cache (C failure mode)"


def test_invalidation_on_literal_edit(tmp_path):
    cache = tmp_path / "nbcache"
    cache.mkdir()
    assert _run_driver(tmp_path, cache, 1) == 15       # The step adds 1*v, so the sum is 15
    assert _run_driver(tmp_path, cache, 3) == 45       # The step adds 3*v, so the sum is 45 and not the stale 15


def test_jit_options_cache_disabled_honored(tmp_path):
    """The generated implementations honor the numbox-wide jit_options. If NUMBOX_JIT_OPTIONS turns the cache off, the
    UDAF still computes correctly and writes no implementation .nbc."""
    cache = tmp_path / "nbcache_nocache"
    cache.mkdir()
    script = tmp_path / "drv_nocache.py"
    script.write_text(_DRIVER.replace("{MULT}", "1"))
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache),
               NUMBOX_JIT_OPTIONS='{"cache": false}')
    out = subprocess.run([sys.executable, str(script)], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    result = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][0]
    assert int(result.split()[1]) == 15                # The result is correct with the cache disabled
    assert _count_nbc(cache) == 0, "cache disabled but a udaf impl .nbc was written"


@njit
def w_inverse(state, ctx, argc, argv_pp):
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total -= sqlite3_value_int64(args[0])


@njit
def w_value(state, ctx):
    sqlite3_result_int64(ctx, state.total)


def _read_window(db, select_sql, nrows):
    meta = np.zeros(nrows + 1, dtype=np.int64)  # meta[0] holds the count and meta[1:] hold the values

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
    register_window(db, "my_wsum", 1, sum_state_type, sum_init, sum_step, w_inverse, w_value, sum_finalize)
    sql = ("SELECT __wcap(my_wsum(v) OVER "
           "(ORDER BY v ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)) "
           "FROM t ORDER BY v")
    vals, _cap = _read_window(db, sql, 5)
    sqlite3_close(db)
    assert vals == [1, 3, 5, 7, 9]


@njit
def sum2_step(state, ctx, argc, argv_pp):  # A distinct body yields independent compiled callbacks
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total += 2 * sqlite3_value_int64(args[0])


def test_two_distinct_aggregates_no_collision():
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_aggregate(db, "my_sum", 1, sum_state_type, sum_init, sum_step, sum_finalize)
    register_aggregate(db, "my_sum2", 1, sum_state_type, sum_init, sum2_step, sum_finalize)
    v1, _c1 = _read1_int64(db, "SELECT __cap(my_sum(v)) FROM t")
    v2, _c2 = _read1_int64(db, "SELECT __cap(my_sum2(v)) FROM t")
    sqlite3_close(db)
    assert (v1, v2) == (15, 30)


def test_deterministic_flag():
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_aggregate(db, "my_sum_det", 1, sum_state_type, sum_init, sum_step, sum_finalize, deterministic=True)
    val, _cap = _read1_int64(db, "SELECT __cap(my_sum_det(v)) FROM t")
    sqlite3_close(db)
    assert val == 15


def test_deterministic_flag_ors_bit(monkeypatch):
    """deterministic=True must OR SQLITE_DETERMINISTIC into the flags passed to sqlite3_create_function_v2, and the
    default must not. The flag is a query-planner hint with no effect on an aggregate's value, so a result assertion
    alone cannot verify this contract. The test therefore records the flags through a spy instead."""
    import numbox.core.bindings.sqlite.udf_helpers as helpers
    real = helpers.sqlite3_create_function_v2
    seen = []

    def spy(db, name_p, n_arg, flags, *rest):
        seen.append(flags)
        return real(db, name_p, n_arg, flags, *rest)

    monkeypatch.setattr(helpers, "sqlite3_create_function_v2", spy)
    db = _open_memory()
    register_aggregate(db, "det_on", 1, sum_state_type, sum_init, sum_step, sum_finalize, deterministic=True)
    register_aggregate(db, "det_off", 1, sum_state_type, sum_init, sum_step, sum_finalize)
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
    test_aggregate_sum()          # This call warms up the JIT and the one-time allocations
    test_window_running_sum()
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        test_aggregate_sum()
        test_window_running_sum()
    after = nrt.rtsys.get_allocation_stats()
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, "meminfo imbalance: %d alloc, %d free" % (allocated, freed)


def test_window_deterministic_flag_ors_bit(monkeypatch):
    """register_window must OR SQLITE_DETERMINISTIC into the flags passed to
    sqlite3_create_window_function when deterministic=True, and not otherwise."""
    import numbox.core.bindings.sqlite.udf_helpers as helpers
    real = helpers.sqlite3_create_window_function
    seen = []

    def spy(db, name_p, n_arg, flags, *rest):
        seen.append(flags)
        return real(db, name_p, n_arg, flags, *rest)

    monkeypatch.setattr(helpers, "sqlite3_create_window_function", spy)
    db = _open_memory()
    register_window(db, "wdet_on", 1, sum_state_type, sum_init, sum_step, w_inverse, w_value, sum_finalize, deterministic=True)
    register_window(db, "wdet_off", 1, sum_state_type, sum_init, sum_step, w_inverse, w_value, sum_finalize)
    sqlite3_close(db)
    assert seen[0] & helpers.SQLITE_DETERMINISTIC
    assert not (seen[1] & helpers.SQLITE_DETERMINISTIC)


def test_registration_error_raises(monkeypatch):
    """A non-OK return from sqlite registration raises a RuntimeError through _raise_rc instead of succeeding silently.
    The test injects the non-OK code, so the test is portable across sqlite builds. An out-of-range n_arg or an
    over-long name would not be portable, because its rejection is build-dependent: SQLITE_MAX_FUNCTION_ARG and name
    limits vary."""
    import pytest
    import numbox.core.bindings.sqlite.udf_helpers as helpers
    monkeypatch.setattr(helpers, "sqlite3_create_function_v2",
                        lambda *a, **k: 1)  # The lambda returns a non-OK rc (SQLITE_ERROR)
    db = _open_memory()
    with pytest.raises(RuntimeError, match="registration failed"):
        register_aggregate(db, "err", 1, sum_state_type, sum_init, sum_step, sum_finalize)
    sqlite3_close(db)


def test_plain_python_callbacks_accepted():
    """The helper wraps plain (undecorated) Python callbacks with njit, and the callbacks work end to end. Callers do
    not need to apply ``@njit`` to them first."""
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])

    def plain_init():
        return SumState(np.int64(0))

    def plain_step(state, ctx, argc, argv_pp):
        args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
        state.total += sqlite3_value_int64(args[0])

    def plain_finalize(state, ctx):
        sqlite3_result_int64(ctx, state.total)

    register_aggregate(db, "plain_sum", 1, sum_state_type, plain_init, plain_step, plain_finalize)
    gc.collect()  # No handle is retained. The njit-wrapped plain callbacks survive GC as well.
    val, _cap = _read1_int64(db, "SELECT __cap(plain_sum(v)) FROM t")
    sqlite3_close(db)
    assert val == 15


def test_non_callable_callback_rejected():
    """A non-callable callback is rejected immediately with a clear TypeError."""
    import pytest
    db = _open_memory()
    with pytest.raises(TypeError, match="step must be a callable"):
        register_aggregate(db, "bad_cb", 1, sum_state_type, sum_init, 42, sum_finalize)
    sqlite3_close(db)


# --- Multi-field state and multi-argument UDAF parity ---
@structref.register
class PairStateType(nb_types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)


class PairState(structref.StructRefProxy):
    def __new__(cls, sa, sb):
        return structref.StructRefProxy.__new__(cls, sa, sb)


structref.define_proxy(PairState, PairStateType, ["sa", "sb"])
pair_state_type = PairStateType([("sa", nb_types.int64), ("sb", nb_types.int64)])


@njit
def pair_init():
    return PairState(np.int64(0), np.int64(0))


@njit
def pair_step(state, ctx, argc, argv_pp):
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.sa += sqlite3_value_int64(args[0])
    state.sb += sqlite3_value_int64(args[1])


@njit
def pair_finalize(state, ctx):
    sqlite3_result_int64(ctx, state.sa + 10 * state.sb)


def test_multiarg_multifield_state():
    db = _open_memory()
    with c_string("CREATE TABLE t2(a INTEGER, b INTEGER)") as p:
        assert sqlite3_exec(db, p, 0, 0, 0) == SQLITE_OK
    for a, b in [(1, 4), (2, 5), (3, 6)]:
        with c_string("INSERT INTO t2 VALUES (%d, %d)" % (a, b)) as p:
            assert sqlite3_exec(db, p, 0, 0, 0) == SQLITE_OK
    register_aggregate(db, "pairsum", 2, pair_state_type, pair_init, pair_step, pair_finalize)
    val, _cap = _read1_int64(db, "SELECT __cap(pairsum(a, b)) FROM t2")
    sqlite3_close(db)
    assert val == 156  # sa=1+2+3=6 and sb=4+5+6=15, so the result is 6 + 10*15


@njit
def raising_finalize(state, ctx):
    raise ValueError("boom from finalize")


def test_finalize_exception_surfaces_error_and_no_leak():
    """A raising finalize must fail the query (xFinal reports SQLITE_ERROR) and must not leak the exported meminfo slot.
    xFinal releases the slot in a try/except."""
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")

    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_aggregate(db, "raise_fin", 1, sum_state_type, sum_init, sum_step, raising_finalize)
    with c_string("SELECT raise_fin(v) FROM t") as sp:
        rc = sqlite3_exec(db, sp, 0, 0, 0)
    assert rc != SQLITE_OK  # sqlite3_exec returns the error instead of a silent wrong result
    assert "finalize callback" in _errmsg(db)  # The message is descriptive, not the generic text for the error code

    _nrt.memsys_enable_stats()
    with c_string("SELECT raise_fin(v) FROM t") as sp:  # This run is the warm-up
        sqlite3_exec(db, sp, 0, 0, 0)
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        with c_string("SELECT raise_fin(v) FROM t") as sp:
            sqlite3_exec(db, sp, 0, 0, 0)
    after = nrt.rtsys.get_allocation_stats()
    sqlite3_close(db)
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, "leak on raising finalize: %d alloc, %d free" % (allocated, freed)


@njit
def raising_step(state, ctx, argc, argv_pp):
    raise ValueError("boom from step")


def test_step_exception_surfaces_error_and_no_leak():
    """A raising step must fail the query (xStep reports SQLITE_ERROR) instead of
    silently dropping rows, and must not leak the borrowed state meminfo."""
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")

    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_aggregate(db, "raise_step", 1, sum_state_type, sum_init, raising_step, sum_finalize)
    with c_string("SELECT raise_step(v) FROM t") as sp:
        rc = sqlite3_exec(db, sp, 0, 0, 0)
    assert rc != SQLITE_OK  # The raising step fails the query instead of producing a silent wrong result
    assert "step callback" in _errmsg(db)  # The message is descriptive, not the generic text for the error code

    _nrt.memsys_enable_stats()
    with c_string("SELECT raise_step(v) FROM t") as sp:  # This run is the warm-up
        sqlite3_exec(db, sp, 0, 0, 0)
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        with c_string("SELECT raise_step(v) FROM t") as sp:
            sqlite3_exec(db, sp, 0, 0, 0)
    after = nrt.rtsys.get_allocation_stats()
    sqlite3_close(db)
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, "leak on raising step: %d alloc, %d free" % (allocated, freed)


_WIN_SQL = ("SELECT %s(v) OVER (ORDER BY v ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) "
            "FROM t ORDER BY v")


@njit
def raising_value(state, ctx):
    raise ValueError("boom from value")


def test_value_exception_surfaces_error_and_no_leak():
    """A raising window value must fail the query (xValue reports SQLITE_ERROR)
    instead of emitting a stale value, and must not leak the borrowed meminfo."""
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")

    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_window(db, "raise_wval", 1, sum_state_type, sum_init, sum_step, w_inverse, raising_value, sum_finalize)
    with c_string(_WIN_SQL % "raise_wval") as sp:
        rc = sqlite3_exec(db, sp, 0, 0, 0)
    assert rc != SQLITE_OK
    assert "value callback" in _errmsg(db)  # The message is descriptive, not the generic text for the error code

    _nrt.memsys_enable_stats()
    with c_string(_WIN_SQL % "raise_wval") as sp:  # This run is the warm-up
        sqlite3_exec(db, sp, 0, 0, 0)
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        with c_string(_WIN_SQL % "raise_wval") as sp:
            sqlite3_exec(db, sp, 0, 0, 0)
    after = nrt.rtsys.get_allocation_stats()
    sqlite3_close(db)
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, "leak on raising value: %d alloc, %d free" % (allocated, freed)


@njit
def raising_inverse(state, ctx, argc, argv_pp):
    raise ValueError("boom from inverse")


def test_inverse_exception_surfaces_error_and_no_leak():
    """A raising window inverse must fail the query and must not leak the borrowed meminfo. SQLite does not document
    whether an error in xInverse propagates, but the propagation is observed on the system sqlite. The essential
    guarantee is the absence of a leak."""
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")

    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    register_window(db, "raise_winv", 1, sum_state_type, sum_init, sum_step, raising_inverse, w_value, sum_finalize)
    with c_string(_WIN_SQL % "raise_winv") as sp:
        rc = sqlite3_exec(db, sp, 0, 0, 0)
    assert rc != SQLITE_OK

    _nrt.memsys_enable_stats()
    with c_string(_WIN_SQL % "raise_winv") as sp:  # This run is the warm-up
        sqlite3_exec(db, sp, 0, 0, 0)
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        with c_string(_WIN_SQL % "raise_winv") as sp:
            sqlite3_exec(db, sp, 0, 0, 0)
    after = nrt.rtsys.get_allocation_stats()
    sqlite3_close(db)
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, "leak on raising inverse: %d alloc, %d free" % (allocated, freed)


def test_finalize_exception_empty_group_surfaces_error():
    """A raising finalize on an empty group must also fail the query, and its exception must not be discarded silently.
    An empty group takes xFinal's no-state branch: no xStep ran, so the aggregate context is NULL. The empty-group
    branch is guarded like the borrow path."""
    db = _open_memory()
    _make_table(db, [])  # With an empty table, xStep never runs and xFinal takes the no-state branch
    register_aggregate(db, "raise_fin_empty", 1, sum_state_type, sum_init, sum_step, raising_finalize)
    with c_string("SELECT raise_fin_empty(v) FROM t") as sp:
        rc = sqlite3_exec(db, sp, 0, 0, 0)
    msg = _errmsg(db)
    sqlite3_close(db)
    assert rc != SQLITE_OK  # A raising finalize on an empty group must fail with an error
    assert "finalize callback" in msg  # The message is descriptive on the empty-group path as well


@njit
def init_raise():
    s = SumState(np.int64(0))
    if s.total >= 0:  # The condition is always true at run time, but it keeps the St return type for typing
        raise ValueError("boom from init")
    return s


def test_init_exception_surfaces_error_no_unraisable():
    """A raising init (called at xStep's first-row export) must be caught and reported as 'error in user init callback'.
    The @cfunc boundary must not discard the exception as an 'Exception ignored' unraisable."""
    db = _open_memory()
    _make_table(db, [1, 2, 3])
    register_aggregate(db, "raise_init", 1, sum_state_type, init_raise, sum_step, sum_finalize)
    captured = []
    old_hook = sys.unraisablehook
    sys.unraisablehook = lambda a: captured.append(a)
    try:
        with c_string("SELECT raise_init(v) FROM t") as sp:
            rc = sqlite3_exec(db, sp, 0, 0, 0)
        msg = _errmsg(db)
    finally:
        sys.unraisablehook = old_hook
    sqlite3_close(db)
    assert not captured, "raising init leaked an unraisable (swallowed): %r" % (captured,)
    assert rc != SQLITE_OK
    assert "init callback" in msg  # The message names the correct role, not a mislabeled finalize error


def test_init_exception_empty_group_surfaces_error():
    """A raising init on an empty group (xFinal's no-state branch) must report
    'error in user init callback', not the mislabeled 'finalize callback'."""
    db = _open_memory()
    _make_table(db, [])
    register_aggregate(db, "raise_init_empty", 1, sum_state_type, init_raise, sum_step, sum_finalize)
    with c_string("SELECT raise_init_empty(v) FROM t") as sp:
        rc = sqlite3_exec(db, sp, 0, 0, 0)
    msg = _errmsg(db)
    sqlite3_close(db)
    assert rc != SQLITE_OK
    assert "init callback" in msg
