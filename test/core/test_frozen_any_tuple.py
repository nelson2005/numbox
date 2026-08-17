import os
import struct
import subprocess
import sys
import textwrap
import zlib

import numpy
import pytest

from numba import float64, int64, njit, typeof
from numba.core import types
from numba.core.errors import NumbaError, TypingError
from numba.core.types import unicode_type
from numba.typed import Dict

from numbox.core.any.any_type import Any, AnyType, make_any
from numbox.core.any.frozen_any_tuple import (
    _fat_type_cache, _slot_code, FrozenAnyTuple, FrozenAnyTupleTypeClass, frozen_any_tuple_type,
    make_frozen_any_tuple,
)
from numbox.utils.meminfo import get_nrt_refcount, structref_meminfo
from test.common_structrefs import S1, S1Type


@njit(cache=True)
def _jit_build_and_checked_reads(x):
    fat = make_frozen_any_tuple((x, 2.5, "abc"))
    return fat.get_as(0, int64), fat.get_as(1, float64), fat.get_as(2, unicode_type), len(fat)


@njit(cache=True)
def _jit_checked_mismatch(fat):
    return fat.get_as(0, float64)


@njit(cache=True)
def _jit_hoisted_iteration_sum(fat):
    t = fat.anys
    acc = 0
    for x in t:
        acc += x.get_as(int64)
    return acc


@njit(cache=True)
def _jit_len_and_getitem(fat):
    return len(fat), fat[0].get_as(int64)


@njit(cache=True)
def _jit_checked_get_as_int64(fat, i):
    return fat.get_as(i, int64)


def test_python_build_and_reads():
    a = numpy.array([1, 2], dtype=numpy.int32)
    fat = make_frozen_any_tuple([7, 2.5, "abc", a])
    assert len(fat) == 4
    assert fat.get_as(0, int64) == 7
    assert abs(fat.get_as(1, float64) - 2.5) < 1e-15
    assert fat.get_as(2, unicode_type) == "abc"
    a1 = fat.get_as(3, typeof(a))
    assert numpy.array_equal(a1, a)
    assert a1.ctypes.data == a.ctypes.data
    assert isinstance(fat[0], Any)
    assert fat[0].get_as(int64) == 7


def test_build_accepts_tuple_and_generator():
    fat1 = make_frozen_any_tuple((1, 2.5))
    fat2 = make_frozen_any_tuple(i + 0.5 for i in range(2))
    assert fat1.get_as(0, int64) == 1
    assert abs(fat2.get_as(1, float64) - 1.5) < 1e-15


def test_jit_build_and_checked_reads():
    assert _jit_build_and_checked_reads(10) == (10, 2.5, "abc", 3)


def test_len_python_and_jit():
    fat = make_frozen_any_tuple([7, 2.5, "s"])
    assert len(fat) == 3
    n, first = _jit_len_and_getitem(fat)
    assert n == 3
    assert first == 7


def test_get_as_mismatch_raises():
    fat = make_frozen_any_tuple([7, 2.5])
    with pytest.raises(TypeError, match="slot type mismatch"):
        fat.get_as(0, float64)
    with pytest.raises(TypeError, match="slot type mismatch"):
        _jit_checked_mismatch(fat)


def test_python_iteration_and_list():
    # The guarded __getitem__ ends the legacy iteration protocol with IndexError at i == n.
    fat = make_frozen_any_tuple([7, 2.5, "abc"])
    seen = []
    for a in fat:
        seen.append(a)
    assert len(seen) == 3
    assert all(isinstance(a, Any) for a in seen)
    assert seen[0].get_as(int64) == 7
    assert abs(seen[1].get_as(float64) - 2.5) < 1e-15
    assert seen[2].get_as(unicode_type) == "abc"
    assert len(list(fat)) == 3


def test_python_oob_raises():
    fat = make_frozen_any_tuple([7, 2.5])
    with pytest.raises(IndexError, match="index out of range"):
        fat[2]
    with pytest.raises(IndexError, match="index out of range"):
        fat[-1]
    with pytest.raises(IndexError, match="index out of range"):
        fat.get_as(2, int64)
    with pytest.raises(IndexError, match="index out of range"):
        fat.get_as(-1, int64)


def test_jit_oob_raises():
    fat = make_frozen_any_tuple([7, 2.5])
    assert _jit_checked_get_as_int64(fat, 0) == 7
    with pytest.raises(IndexError, match="index out of range"):
        _jit_checked_get_as_int64(fat, 2)
    with pytest.raises(IndexError, match="index out of range"):
        _jit_checked_get_as_int64(fat, -1)


def test_unchecked_reinterpret_semantics_pin():
    fat = make_frozen_any_tuple([1.5, 10])
    as_int = fat[0].get_as(int64)
    assert as_int == struct.unpack("<q", struct.pack("<d", 1.5))[0]
    assert as_int == 4609434218613702656
    as_float = fat[1].get_as(float64)
    assert as_float == struct.unpack("<d", struct.pack("<q", 10))[0]
    assert as_float == 5e-323


def test_exact_match_discipline():
    a = numpy.arange(3, dtype=numpy.int64)
    fat = make_frozen_any_tuple([a])
    assert fat.get_as(0, types.Array(types.int64, 1, "C"))[1] == 1
    with pytest.raises(TypeError, match="slot type mismatch"):
        fat.get_as(0, types.Array(types.int64, 1, "A"))


def test_deleted_ctor_python():
    with pytest.raises(NotImplementedError, match="make_frozen_any_tuple"):
        FrozenAnyTuple(1, 2)


def test_deleted_ctor_njit():
    def caller():
        return FrozenAnyTuple(1)

    with pytest.raises((NumbaError, TypingError), match="make_frozen_any_tuple"):
        njit()(caller)()


def test_type_factory_memoization_and_structural_equality():
    t3 = frozen_any_tuple_type(3)
    assert frozen_any_tuple_type(3) is t3
    assert 3 in _fat_type_cache
    # numba interns type instances by structural key, so an independently built
    # same-arity instance is equal (in fact identical): cached code stays valid
    # across processes with no generated modules.
    independent = FrozenAnyTupleTypeClass([
        ("anys", types.UniTuple(AnyType, 3)),
        ("codes", types.Array(types.uint32, 1, "C", readonly=True)),
    ])
    assert independent == t3


def test_type_factory_bounds():
    with pytest.raises(ValueError, match="tuple-arity ceiling"):
        frozen_any_tuple_type(0)
    with pytest.raises(ValueError, match="tuple-arity ceiling"):
        frozen_any_tuple_type(1000)
    assert frozen_any_tuple_type(1) is frozen_any_tuple_type(1)
    assert frozen_any_tuple_type(999) is frozen_any_tuple_type(999)


def test_build_rejects_empty():
    with pytest.raises(TypingError, match="empty"):
        make_frozen_any_tuple([])


def test_build_rejects_any_elements():
    with pytest.raises(TypingError, match="raw values"):
        make_frozen_any_tuple([make_any(7), 8])

    def caller():
        return make_frozen_any_tuple((make_any(7), 8))

    with pytest.raises(TypingError, match="raw values"):
        njit()(caller)()


def test_build_rejects_arity_1000():
    # Heterogeneous on purpose: numba's own arity guard covers UniTuple arguments
    # only, so the library's ValueError must surface at typing time here.
    values = tuple(i if i % 2 == 0 else float(i) for i in range(1000))
    with pytest.raises((TypingError, ValueError), match="tuple-arity ceiling"):
        make_frozen_any_tuple(values)


def test_codes_derivation_pin():
    fat = make_frozen_any_tuple([7, 2.5, "abc"])
    expected = [zlib.crc32(b"int64"), zlib.crc32(b"float64"), zlib.crc32(b"unicode_type")]
    assert fat.codes.dtype == numpy.uint32
    assert list(fat.codes) == expected
    assert [_slot_code(t) for t in (types.int64, types.float64, unicode_type)] == expected


def test_codes_refuse_write_from_python():
    fat = make_frozen_any_tuple([7, 2.5])
    before = list(fat.codes)
    assert not fat.codes.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        fat.codes[0] = 0
    assert list(fat.codes) == before
    assert fat.get_as(0, int64) == 7


def test_codes_refuse_write_from_jit():
    @njit
    def write_code(fat):
        fat.codes[0] = 0

    fat = make_frozen_any_tuple([7, 2.5])
    before = list(fat.codes)
    with pytest.raises(TypingError, match="Cannot modify readonly array"):
        write_code(fat)
    assert list(fat.codes) == before
    assert fat.get_as(0, int64) == 7


def test_codes_setflags_escape_is_the_documented_limit():
    # The refusals above stop an accidental write, not a determined one: the boxed array
    # does not own its data, so numpy permits re-opening it and the guard can then be lied
    # to. Pinned because the docstring states this boundary as fact.
    fat = make_frozen_any_tuple([7, 2.5])
    escaped = fat.codes
    assert not escaped.flags.owndata
    # Two-sided on purpose: without this the test passes just as happily against a codes
    # array that was writable all along, which makes setflags a no-op rather than an escape.
    assert not escaped.flags.writeable
    escaped.setflags(write=True)
    escaped[0] = _slot_code(types.float64)
    assert fat.get_as(0, float64) == struct.unpack("<d", struct.pack("<q", 7))[0]


def test_anys_property_boxes_and_iterates():
    fat = make_frozen_any_tuple([1, 2, 3])
    t = fat.anys
    assert len(t) == 3
    assert all(isinstance(a, Any) for a in t)
    assert [a.get_as(int64) for a in t] == [1, 2, 3]


def test_jit_hoisted_iteration():
    fat = make_frozen_any_tuple(list(range(8)))
    assert _jit_hoisted_iteration_sum(fat) == sum(range(8))


def test_refcount_lifecycle_and_meminfo_identity():
    s1 = S1(81, 67, 2.17)
    assert get_nrt_refcount(s1) == 1
    fat = make_frozen_any_tuple([s1, 5])
    assert get_nrt_refcount(s1) == 2, "the slot's `Any` holds one payload ref via its `_Content`"
    handle = fat[0]
    assert get_nrt_refcount(s1) == 2, "`fat[0]` shares the stored `Any` cell; no new payload ref"
    assert structref_meminfo(handle)[0] == structref_meminfo(fat[0])[0], "same `Any` MemInfo on every read"
    s1_2 = fat.get_as(0, S1Type)
    assert get_nrt_refcount(s1) == 3
    assert structref_meminfo(s1_2)[0] == structref_meminfo(s1)[0], "same MemInfo as the stored payload"
    del s1_2
    assert get_nrt_refcount(s1) == 2
    del handle
    assert get_nrt_refcount(s1) == 2
    del fat
    assert get_nrt_refcount(s1) == 1


def test_reset_aliasing_semantics():
    # Any.reset through an aliased fat[i] handle mutates the shared payload cell
    # without updating the recorded code: the checked read with the original type
    # then passes the guard and reinterprets, and the new type still raises.
    fat = make_frozen_any_tuple([2.5, 7])
    handle = fat[0]
    handle.reset(123)
    assert fat.get_as(0, float64) == struct.unpack("<d", struct.pack("<q", 123))[0]
    assert fat.codes[0] == _slot_code(types.float64)
    with pytest.raises(TypeError, match="slot type mismatch"):
        fat.get_as(0, int64)


def test_n34_build_crosses_display_lowering():
    values = [i if i % 3 == 0 else (i + 0.5 if i % 3 == 1 else "s%02d" % i) for i in range(34)]
    fat = make_frozen_any_tuple(values)
    assert len(fat) == 34
    assert fat.get_as(33, int64) == 33
    assert fat.get_as(32, unicode_type) == "s32"
    assert abs(fat.get_as(31, float64) - 31.5) < 1e-15


def test_typed_dict_value_type():
    d = Dict.empty(key_type=unicode_type, value_type=frozen_any_tuple_type(2))
    d["a"] = make_frozen_any_tuple([1, 2.5])
    d["b"] = make_frozen_any_tuple([3, 4.5])
    assert d["a"].get_as(0, int64) == 1
    assert abs(d["b"].get_as(1, float64) - 4.5) < 1e-15


def test_cache_survives_across_processes(tmp_path):
    # Two processes share one NUMBA_CACHE_DIR. The warm process must load the
    # jit-side build (exec-generated impl compiled into its cache=True caller)
    # and the checked-read raise path from cache, with a stable cache file set.
    probe = tmp_path / "fat_probe.py"
    probe.write_text(textwrap.dedent('''
        from numba import njit, float64, int64
        from numbox.core.any.frozen_any_tuple import make_frozen_any_tuple

        @njit(cache=True)
        def build_and_read(x):
            fat = make_frozen_any_tuple((x, 2.5, "abc"))
            return fat.get_as(0, int64) + int(fat.get_as(1, float64))

        @njit(cache=True)
        def mistyped_read(fat):
            return fat.get_as(0, float64)

        assert build_and_read(7) == 9
        fat = make_frozen_any_tuple([7, 2.5, "abc"])
        try:
            mistyped_read(fat)
            raise SystemExit("guard did not raise")
        except TypeError as e:
            assert "slot type mismatch" in str(e), str(e)
        print("OK")
    '''), encoding="utf-8")
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "nbcache")}

    r1 = subprocess.run([sys.executable, str(probe)], env=env, capture_output=True, text=True)
    assert r1.returncode == 0, f"run1 (cold) failed:\n{r1.stderr}"
    assert r1.stdout.strip() == "OK"
    nbc_after_cold = sorted(p.name for p in (tmp_path / "nbcache").rglob("*.nbc"))
    assert nbc_after_cold, "cold run wrote no cache files"

    r2 = subprocess.run([sys.executable, str(probe)], env=env, capture_output=True, text=True)
    assert r2.returncode == 0, f"run2 (warm) failed:\n{r2.stderr}"
    assert r2.stdout.strip() == "OK"
    nbc_after_warm = sorted(p.name for p in (tmp_path / "nbcache").rglob("*.nbc"))
    assert nbc_after_warm == nbc_after_cold, "warm run changed the cache file set"
