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
    _fat_type_cache, _init_fat, _slot_code, FrozenAnyTuple, FrozenAnyTupleTypeClass,
    frozen_any_tuple_type, make_frozen_any_tuple,
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


def test_codes_not_reachable_from_jit():
    # The codes array never crosses into jit code. Exposing it there would hand out an alias of
    # the guard's own evidence, and a readonly array type does not cover every store that reaches
    # it: `.flat` setitem ignores array mutability, and a raw data pointer bypasses typing.
    @njit
    def read_codes(fat):
        return fat.codes

    @njit
    def element_store(fat):
        fat.codes[0] = 0

    @njit
    def flat_store(fat):
        fat.codes.flat[0] = 0

    fat = make_frozen_any_tuple([7, 2.5])
    before = list(fat.codes)
    for attempt in (read_codes, element_store, flat_store):
        with pytest.raises(TypingError, match="Unknown attribute 'codes'"):
            attempt(fat)
    assert list(fat.codes) == before
    assert fat.get_as(0, int64) == 7


def test_fields_refuse_rebinding_in_jit():
    # No setattr is defined for the type, so neither field can be rebound. Without this, a
    # one-line njit assignment defeats the guard: swapping `codes` forges the recorded types, and
    # swapping `anys` leaves the recorded types describing values that are no longer there.
    @njit
    def rebind_codes(fat, arr):
        fat.codes = arr

    @njit
    def rebind_anys(dst, src):
        dst.anys = src.anys

    fat = make_frozen_any_tuple([7, 2.5])
    other = make_frozen_any_tuple([1.5, 9])
    forged = numpy.full(2, _slot_code(types.float64), dtype=numpy.uint32)

    with pytest.raises(TypingError, match="Cannot resolve setattr"):
        rebind_codes(fat, forged)
    with pytest.raises(NumbaError, match="FrozenAnyTuple is frozen"):
        rebind_anys(fat, other)

    assert list(fat.codes) == [_slot_code(types.int64), _slot_code(types.float64)]
    assert fat.get_as(0, int64) == 7
    with pytest.raises(TypeError, match="slot type mismatch"):
        fat.get_as(0, float64)


def test_get_as_requires_an_integer_index():
    # `_get_code` declares an `intp` index and numba rates intrinsic arguments with unsafe casting
    # allowed, so without an explicit typing guard a float index is truncated rather than refused.
    # Array getitem, which the guard used to go through, refused these.
    @njit
    def jit_get(fat, i):
        return fat.get_as(i, int64)

    fat = make_frozen_any_tuple([7, 2.5, "abc"])
    for bad in (0.0, 0.9, True):
        with pytest.raises(TypingError, match="index must be an integer"):
            jit_get(fat, bad)

    # Both sides of the one documented method must answer the same index the same way. `bool` needs
    # its own clause: `operator.index(True)` is 1, so Python would otherwise read slot 1 for an index
    # the jit side refuses outright.
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        fat.get_as(0.9, int64)
    with pytest.raises(TypeError, match="must be an integer, not a bool"):
        fat.get_as(True, int64)

    assert jit_get(fat, 0) == 7
    assert fat.get_as(0, int64) == 7
    # every integer width and signedness still works
    for good in (numpy.int8(1), numpy.uint8(1), numpy.int32(1), numpy.uint64(1), 1):
        assert fat.get_as(good, float64) == 2.5


def test_init_fat_validates_its_codes_argument():
    # `_init_fat` is private and calling it on a live container is misuse, but it must not be able to
    # make `_get_code`'s manual GEP read outside the allocation. `context.cast` does not consult
    # `can_convert`, so the checks have to be explicit: numba's array-to-array cast asserts only on
    # mutability and layout, and array length is not part of the type at all.
    @njit
    def call_init(dst, anys, codes):
        _init_fat(dst, anys, codes)

    victim = make_frozen_any_tuple([1, 2.5, "z"])
    donor = make_frozen_any_tuple([10, 20.5, "y"])
    expected = list(victim.codes)

    rejected_by_typing = (
        numpy.zeros(3, dtype=numpy.int32),
        numpy.zeros(3, dtype=numpy.float64),
        numpy.zeros((3, 1), dtype=numpy.uint32),
        numpy.arange(6, dtype=numpy.uint32)[::2],
    )
    for arr in rejected_by_typing:
        with pytest.raises(TypingError):
            call_init(victim, donor.anys, arr)

    with pytest.raises(ValueError, match="codes length does not match the arity"):
        call_init(victim, donor.anys, numpy.zeros(1, dtype=numpy.uint32))

    assert list(victim.codes) == expected
    assert victim.get_as(0, int64) == 1


def test_reference_cycle_is_uncollectable(tmp_path):
    # Pins the docstring's claim that a container reachable from one of its own slots leaks, NRT
    # having no cycle detection. Runs in a subprocess because allocation stats are off by default
    # and `rtsys.get_allocation_stats()` raises RuntimeError("NRT stats are disabled.") without the
    # env var, which cannot be set after numba has imported.
    probe = tmp_path / "cycle_probe.py"
    probe.write_text(textwrap.dedent('''
        import gc
        from numba.core.runtime import rtsys
        from numbox.core.any.frozen_any_tuple import make_frozen_any_tuple

        def live():
            s = rtsys.get_allocation_stats()
            return s.alloc - s.free

        def plain():
            fat = make_frozen_any_tuple([1, 2.5])
            del fat

        def cyclic():
            fat = make_frozen_any_tuple([1, 2.5])
            fat[0].reset(fat)
            del fat

        plain(); cyclic(); gc.collect()

        gc.collect(); before = live()
        for _ in range(20):
            plain()
        gc.collect()
        plain_delta = live() - before

        gc.collect(); before = live()
        for _ in range(20):
            cyclic()
        gc.collect()
        cyclic_delta = live() - before

        assert plain_delta == 0, f"a plain build/drop leaked {plain_delta}"
        assert cyclic_delta > 0, f"a self-referential container was collected ({cyclic_delta})"
        print("OK")
    '''), encoding="utf-8")
    # A dedicated cache dir, not the repo's: numba pickles module paths into the `.nbi` index and
    # unpickles them before the freshness check, so a warm entry left by a sibling test that used
    # `test.common_structrefs` fails to load from this subprocess's cwd.
    env = {
        **os.environ,
        "NUMBA_NRT_STATS": "1",
        "NUMBA_CACHE_DIR": str(tmp_path / "nbcache"),
    }
    r = subprocess.run([sys.executable, str(probe)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"probe failed:\n{r.stdout}\n{r.stderr}"
    assert r.stdout.strip().endswith("OK")


def test_codes_property_is_a_defensive_copy():
    # numpy's WRITEABLE flag describes one ndarray wrapper, not the memory behind it, so the
    # property returns a fresh copy per access rather than a view. Each of these writes lands on
    # a throwaway; none can reach the guard.
    fat = make_frozen_any_tuple([7, 2.5])
    expected = [_slot_code(types.int64), _slot_code(types.float64)]

    assert fat.codes is not fat.codes
    assert not fat.codes.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        fat.codes[0] = 0

    reopened = fat.codes
    reopened.setflags(write=True)
    reopened[0] = _slot_code(types.float64)

    class Holder:
        pass

    got = fat.codes
    iface = dict(got.__array_interface__)
    iface["data"] = (iface["data"][0], False)
    holder = Holder()
    holder.__array_interface__ = iface
    numpy.asarray(holder)[0] = _slot_code(types.float64)

    assert list(fat.codes) == expected
    with pytest.raises(TypeError, match="slot type mismatch"):
        fat.get_as(0, float64)
    assert fat.get_as(0, int64) == 7


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

    # The jit routes reach the same shared cell. `fat[i]` and the hoisted `fat.anys[i]` are both
    # aliases, so a reset through either is visible to a later checked read with the stale type.
    @njit
    def reset_via_getitem(f, v):
        f[0].reset(v)

    @njit
    def reset_via_anys(f, v):
        t = f.anys
        t[1].reset(v)

    reset_via_getitem(fat, 456)
    assert fat.get_as(0, float64) == struct.unpack("<d", struct.pack("<q", 456))[0]

    reset_via_anys(fat, 9.5)
    assert fat.codes[1] == _slot_code(types.int64), "the recorded code is unchanged by a reset"
    assert fat.get_as(1, int64) == struct.unpack("<q", struct.pack("<d", 9.5))[0]

    # A payload held by reference stays mutable, which is the other half of the claim.
    arr = numpy.arange(3, dtype=numpy.int64)
    holder = make_frozen_any_tuple([arr])
    arr[0] = 99
    assert holder.get_as(0, types.int64[::1])[0] == 99


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
