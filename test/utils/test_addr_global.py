"""TypingError-path coverage for the private intrinsics that back
``cres_cacheable``. These intrinsics are user-facing only through the
``cres_cacheable`` decorator, but each has a typing-time validator whose
error message is the contract guarantee for "loud failure on misuse" —
pin them with explicit negative-path tests so a refactor can't silently
turn a clean ``TypingError`` into an opaque LLVM lowering error.
"""
import numpy as np
import pytest
from numba import njit, types
from numba.core.errors import TypingError

from numbox.core.bindings import errno_set
from numbox.utils._addr_global import (
    _load_addr_from_named_global,
    _make_icall_for_sig,
    _store_addr_to_named_global,
)


def test_store_addr_rejects_non_literal_name():
    """_store_addr_to_named_global must receive a literal string name; a
    runtime-computed string is rejected at typing time with a clean
    TypingError naming the intrinsic."""
    @njit
    def caller(name_arr):
        # numba's @njit-call site sees a unicode_type, not a Literal
        _store_addr_to_named_global(name_arr, 0)

    with pytest.raises(TypingError, match=r"_store_addr_to_named_global.*literal"):
        caller("dynamic_name_not_literal")


def test_load_addr_rejects_non_literal_name():
    """Symmetric to the store side: _load_addr_from_named_global also
    requires a literal name."""
    @njit
    def caller(name_arr):
        return _load_addr_from_named_global(name_arr)

    with pytest.raises(TypingError, match=r"_load_addr_from_named_global.*literal"):
        caller("dynamic_name_not_literal")


def test_icall_rejects_wrong_arg_count_tuple():
    """_make_icall_for_sig's _icall validates the tuple arity matches sig.args
    at typing time. Passing the wrong number of args via a tuple raises a
    TypingError naming the mismatch."""
    icall = _make_icall_for_sig(types.int32(types.int64, types.int64))  # expects 2 args

    @njit
    def caller():
        # _icall(addr, (a,)) — only 1 arg in tuple, but sig wants 2
        return icall(0, (np.int64(1),))

    with pytest.raises(TypingError, match=r"_icall.*expected 2 arguments"):
        caller()


def test_icall_rejects_wrong_arg_count_scalar():
    """When the sig has 2+ args and the caller passes a non-tuple second arg,
    _icall reports an arity mismatch (n_args==1 path) rather than silently
    miscompiling."""
    icall = _make_icall_for_sig(types.int32(types.int64, types.int64))  # expects 2 args

    @njit
    def caller():
        # _icall(addr, scalar) — non-tuple, but sig wants 2 args
        return icall(0, np.int64(5))

    with pytest.raises(TypingError, match=r"_icall.*expected 1 argument"):
        caller()


def test_make_icall_for_sig_is_memoized():
    """Identical signatures must return the same _Intrinsic instance so
    numba's specialization cache hits across cres_cacheable decorations.
    """
    sig = types.int32(types.int64)
    a = _make_icall_for_sig(sig)
    b = _make_icall_for_sig(sig)
    assert a is b, "expected memoization; got distinct intrinsic instances"


def test_store_int32_at_rejects_int64_value():
    """_store_int32_at is private and currently only called via errno_set with
    an explicit int32 cast. If a future caller bypasses the cast, the
    intrinsic raises TypingError at typing time rather than producing
    cryptic IR lowering errors."""
    from numbox.core.bindings._errno import _errno_ptr, _store_int32_at

    @njit
    def caller(v):
        _store_int32_at(_errno_ptr(), v)  # v is int64 (default), no cast

    with pytest.raises(TypingError, match=r"_store_int32_at.*int32"):
        caller(np.int64(7))


def test_errno_set_still_works_after_guard():
    """Smoke test: the int32 guard on _store_int32_at must not break the
    normal errno_set(v) path which already casts v to int32 internally."""
    @njit(cache=True)
    def rt(v):
        errno_set(v)
        return 0
    # If the guard mistakenly fired here, this would raise TypingError.
    assert rt(np.int64(42)) == 0
