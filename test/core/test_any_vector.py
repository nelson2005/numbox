import ctypes
import gc
import os
import subprocess
import sys
import textwrap

import numpy
import pytest

from numba import njit, types as nb_types
from numba.core.errors import NumbaError, TypingError
from numba.core.runtime import rtsys, _nrt_python
from numba.core.types import unicode_type

from numbox.core.any.any_type import make_any
from numbox.core.vector.any_vector import (
    AnyVector, any_vector_clear, any_vector_extend, any_vector_pop,
    any_vector_push, create_any_vector,
)
from numbox.core.vector.vector import make_vector
from numbox.utils.meminfo import get_nrt_refcount, structref_meminfo
from test.auxiliary_utils import collect_and_run_tests
from test.common_structrefs import S1, S1Type

int64 = nb_types.int64
float64 = nb_types.float64
arr_i32_1c = nb_types.Array(nb_types.int32, 1, 'C')


def _ensure_stats():
    """``rtsys`` has no enable method on numba 0.65; the C module does."""
    if not _nrt_python.memsys_stats_enabled():
        _nrt_python.memsys_enable_stats()


def _balance():
    s = rtsys.get_allocation_stats()
    return s.alloc - s.free, s.mi_alloc - s.mi_free


def _settled_balance():
    """Collect until the balance stops moving: earlier tests can leave proxy
    objects in traceback reference cycles whose deferred collection would
    otherwise shift the measurement."""
    prev = _balance()
    for _ in range(6):
        gc.collect()
        cur = _balance()
        if cur == prev:
            return cur
        prev = cur
    return prev


def _refcount_at(mi_p):
    """Read the MemInfo refcount word directly; only valid while the
    allocation is provably alive."""
    return ctypes.c_int64.from_address(mi_p).value


@njit
def _njit_build_use_drop(n):
    v = create_any_vector(2)
    for i in range(n):
        any_vector_push(v, make_any(i))
    v.push(2.5)
    total = 0
    for i in range(n):
        total += v[i].get_as(int64)
    return total + v.get_as(n, float64)


@njit
def _njit_push_leaves_one_ref(v):
    any_vector_push(v, make_any(11))


@njit
def _njit_borrow_is_net_zero(v):
    return v[0].get_as(int64)


def _warmup():
    """Compile every dispatcher used by the balance-asserting tests so their
    one-time allocations don't pollute the measured deltas."""
    _ensure_stats()
    v = create_any_vector(1)
    any_vector_push(v, make_any(1))
    any_vector_push(v, 2.5)
    v.push("warm")
    assert v[0].get_as(int64) == 1
    assert v.get_as(1, float64) == 2.5
    v[1] = make_any(3)
    assert len(v) == 3
    popped = v.pop()
    assert popped.get_as(unicode_type) == "warm"
    del popped
    w = create_any_vector(1)
    any_vector_push(w, make_any(4))
    any_vector_extend(v, w)
    any_vector_clear(v)
    w.clear()
    assert _njit_build_use_drop(3) == 0 + 1 + 2 + 2.5
    u = create_any_vector(1)
    _njit_push_leaves_one_ref(u)
    assert _njit_borrow_is_net_zero(u) == 11
    del v, w, u
    gc.collect()


def test_round_trip_mixed_payloads_from_python():
    a = numpy.array([1, 2], dtype=numpy.int32)
    s1 = S1(81, 67, 2.17)
    v = create_any_vector(2)
    any_vector_push(v, make_any(217))
    any_vector_push(v, make_any(2.17))
    any_vector_push(v, make_any("a few words"))
    any_vector_push(v, make_any(a))
    any_vector_push(v, make_any(s1))
    assert len(v) == 5
    assert v.size == 5
    assert v[0].get_as(int64) == 217
    assert abs(v.get_as(1, float64) - 2.17) < 1e-15
    assert v[2].get_as(unicode_type) == "a few words"
    a_back = v[3].get_as(arr_i32_1c)
    assert a_back.ctypes.data == a.ctypes.data
    assert v.get_as(4, S1Type).x2 == 67
    assert v[0].type_info == "int64"


def test_round_trip_inside_njit():
    assert _njit_build_use_drop(4) == 0 + 1 + 2 + 3 + 2.5


def test_push_auto_wraps_non_any_values():
    v = create_any_vector(1)
    any_vector_push(v, 5.0)
    v.push(7)
    assert v[0].type_info == "float64"
    assert v[0].get_as(float64) == 5.0
    assert v[1].get_as(int64) == 7


def test_refcount_sequence_push_and_borrow():
    s1 = S1(81, 67, 2.17)
    assert get_nrt_refcount(s1) == 1
    a = make_any(s1)
    assert get_nrt_refcount(a) == 1
    assert get_nrt_refcount(s1) == 2
    a_mi = structref_meminfo(a)[0]
    v = create_any_vector(2)
    any_vector_push(v, a)
    assert get_nrt_refcount(a) == 2
    assert get_nrt_refcount(s1) == 2
    del a
    gc.collect()
    assert _refcount_at(a_mi) == 1
    assert get_nrt_refcount(s1) == 2
    b = v[0]
    assert _refcount_at(a_mi) == 2
    got = b.get_as(S1Type)
    assert get_nrt_refcount(s1) == 3
    del got
    gc.collect()
    assert get_nrt_refcount(s1) == 2
    del b
    gc.collect()
    assert _refcount_at(a_mi) == 1
    del v
    gc.collect()
    assert get_nrt_refcount(s1) == 1


def test_njit_push_and_borrow_refcounts():
    v = create_any_vector(1)
    _njit_push_leaves_one_ref(v)
    assert _refcount_at(int(v.buf[0])) == 1
    assert _njit_borrow_is_net_zero(v) == 11
    assert _refcount_at(int(v.buf[0])) == 1


def test_setitem_releases_old_after_securing_new():
    v = create_any_vector(1)
    any_vector_push(v, make_any(10))
    old_mi = int(v.buf[0])
    assert _refcount_at(old_mi) == 1
    keep = v[0]
    assert _refcount_at(old_mi) == 2
    v[0] = make_any(3.14)
    assert _refcount_at(old_mi) == 1
    assert v[0].get_as(float64) == 3.14
    del keep
    gc.collect()


def test_setitem_self_assignment_is_safe():
    v = create_any_vector(1)
    any_vector_push(v, make_any(42))
    mi = int(v.buf[0])
    b = v[0]
    v[0] = b
    del b
    gc.collect()
    assert _refcount_at(mi) == 1
    assert v[0].get_as(int64) == 42


def test_pop_transfers_ownership():
    v = create_any_vector(2)
    any_vector_push(v, make_any(1))
    any_vector_push(v, make_any("last"))
    mi = int(v.buf[1])
    popped = any_vector_pop(v)
    assert len(v) == 1
    assert int(v.buf[1]) == 0
    assert _refcount_at(mi) == 1
    assert popped.get_as(unicode_type) == "last"
    popped2 = v.pop()
    assert popped2.get_as(int64) == 1
    assert len(v) == 0
    with pytest.raises(IndexError):
        any_vector_pop(v)


def test_clear_releases_every_element():
    s1 = S1(1, 2, 3.0)
    v = create_any_vector(1)
    any_vector_push(v, make_any(s1))
    any_vector_push(v, make_any(7))
    assert get_nrt_refcount(s1) == 2
    any_vector_clear(v)
    assert len(v) == 0
    assert int(v.buf[0]) == 0
    assert get_nrt_refcount(s1) == 1


def test_dtor_frees_elements_on_vector_death():
    _warmup()
    before = _settled_balance()
    s1 = S1(9, 8, 7.0)
    v = create_any_vector(2)
    any_vector_push(v, make_any(s1))
    any_vector_push(v, make_any(13))
    any_vector_push(v, make_any("hello"))
    assert get_nrt_refcount(s1) == 2
    del v
    gc.collect()
    assert get_nrt_refcount(s1) == 1
    del s1
    gc.collect()
    assert _settled_balance() == before


def test_dtor_walks_current_buffer_after_regrowth():
    _warmup()
    before = _settled_balance()
    v = create_any_vector(1)
    for i in range(6):
        any_vector_push(v, make_any(i))
    assert v.buf.shape[0] == 8
    del v
    gc.collect()
    assert _settled_balance() == before


def test_njit_scope_death_is_balanced():
    _warmup()
    before = _settled_balance()
    assert _njit_build_use_drop(5) == 10 + 2.5
    gc.collect()
    assert _settled_balance() == before


def test_clear_then_death_is_balanced():
    _warmup()
    before = _settled_balance()
    v = create_any_vector(1)
    for i in range(5):
        any_vector_push(v, make_any(i))
    any_vector_clear(v)
    del v
    gc.collect()
    assert _settled_balance() == before


def test_extend_increfs_each_copied_element():
    _warmup()
    before = _settled_balance()
    dst = create_any_vector(1)
    src = create_any_vector(1)
    any_vector_push(dst, make_any(1))
    any_vector_push(src, make_any(2))
    any_vector_push(src, make_any(3))
    mis = [int(src.buf[0]), int(src.buf[1])]
    any_vector_extend(dst, src)
    assert len(dst) == 3
    assert [_refcount_at(p) for p in mis] == [2, 2]
    assert dst[1].get_as(int64) == 2
    assert dst[2].get_as(int64) == 3
    del src
    gc.collect()
    assert [_refcount_at(p) for p in mis] == [1, 1]
    assert dst[1].get_as(int64) == 2
    del dst
    gc.collect()
    assert _settled_balance() == before


def test_extend_self_doubles():
    v = create_any_vector(1)
    any_vector_push(v, make_any(1))
    any_vector_push(v, make_any(2))
    any_vector_extend(v, v)
    assert len(v) == 4
    assert [v[i].get_as(int64) for i in range(4)] == [1, 2, 1, 2]


def test_extend_method_and_proxy():
    v = create_any_vector(1)
    v.push(1)
    w = create_any_vector(1)
    w.push(2)
    v.extend(w)
    assert [v[i].get_as(int64) for i in range(2)] == [1, 2]

    @njit
    def go(a, b):
        a.extend(b)

    go(v, w)
    assert len(v) == 3
    assert v[2].get_as(int64) == 2


def test_zero_capacity_rejected():
    with pytest.raises(ValueError, match="capacity"):
        create_any_vector(0)


def test_scalar_vector_rejected_by_any_ops():
    create, _ = make_vector(nb_types.float64)
    sv = create(4)
    v = create_any_vector(1)
    with pytest.raises(TypingError):
        any_vector_pop(sv)
    with pytest.raises(TypingError):
        any_vector_clear(sv)
    with pytest.raises(TypingError):
        any_vector_extend(v, sv)
    with pytest.raises(TypingError):
        any_vector_extend(sv, v)


def test_buf_is_a_defensive_copy():
    _warmup()
    before = _settled_balance()
    v = create_any_vector(1)
    any_vector_push(v, make_any(42))
    snap = v.buf
    assert int(snap[0]) != 0
    snap[0] = 0
    assert v[0].get_as(int64) == 42
    del snap, v
    gc.collect()
    assert _settled_balance() == before


def test_growth_preserves_stored_pointers():
    v = create_any_vector(1)
    mis = []
    for i in range(9):
        a = make_any(i)
        mis.append(structref_meminfo(a)[0])
        any_vector_push(v, a)
        del a
    gc.collect()
    assert v.buf.shape[0] == 16
    assert [int(v.buf[i]) for i in range(9)] == mis
    assert [_refcount_at(p) for p in mis] == [1] * 9
    assert [v[i].get_as(int64) for i in range(9)] == list(range(9))


def test_index_bounds_checked():
    v = create_any_vector(2)
    any_vector_push(v, make_any(1))
    with pytest.raises(IndexError):
        v[1]
    with pytest.raises(IndexError):
        v[-1]
    with pytest.raises(IndexError):
        v[1] = make_any(2)


def test_get_as_type_mismatch_raises():
    v = create_any_vector(1)
    any_vector_push(v, make_any(217))
    with pytest.raises(NumbaError, match="cannot decode as"):
        v.get_as(0, float64)


def test_ctor_redirects_to_factory():
    with pytest.raises(NotImplementedError, match="create_any_vector"):
        AnyVector(5)

    def caller():
        return AnyVector(5)

    with pytest.raises(NumbaError, match="create_any_vector"):
        njit()(caller)()


def test_cache_survives_across_processes(tmp_path):
    probe = textwrap.dedent("""
        from numba import types as nb_types
        from numbox.core.any.any_type import make_any
        from numbox.core.vector.any_vector import (
            any_vector_clear, any_vector_extend, any_vector_pop,
            any_vector_push, create_any_vector,
        )
        v = create_any_vector(1)
        any_vector_push(v, make_any(217))
        any_vector_push(v, 2.5)
        any_vector_push(v, make_any("tagged"))
        print(v[0].get_as(nb_types.int64))
        print(v.get_as(1, nb_types.float64))
        print(v[2].get_as(nb_types.unicode_type))
        v[1] = make_any(9.5)
        print(v.get_as(1, nb_types.float64))
        p = any_vector_pop(v)
        print(p.get_as(nb_types.unicode_type))
        w = create_any_vector(1)
        any_vector_push(w, make_any(7))
        any_vector_extend(v, w)
        print(len(v), v[2].get_as(nb_types.int64))
        any_vector_clear(v)
        del v, w, p
        print("ok")
    """)
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path)}
    expected = "217\n2.5\ntagged\n9.5\ntagged\n3 7\nok\n"

    r1 = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True,
    )
    assert r1.returncode == 0, f"run1 (cold) failed:\n{r1.stderr}"
    assert r1.stdout == expected

    r2 = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True,
    )
    assert r2.returncode == 0, f"run2 (warm) failed:\n{r2.stderr}"
    assert r2.stdout == expected


if __name__ == '__main__':
    collect_and_run_tests(__name__)
