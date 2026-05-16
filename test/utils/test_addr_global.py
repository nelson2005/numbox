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
    miscompiling. The error message reports the sig's expected arity AND
    the actual arg count in the right direction (an earlier version had
    them swapped)."""
    icall = _make_icall_for_sig(types.int32(types.int64, types.int64))  # expects 2 args

    @njit
    def caller():
        # _icall(addr, scalar) — non-tuple, but sig wants 2 args
        return icall(0, np.int64(5))

    with pytest.raises(
        TypingError, match=r"_icall.*expected 2 arguments, got 1",
    ):
        caller()


def test_store_addr_rejects_non_intp_value():
    """_store_addr_to_named_global stores into an intp-sized global; if a
    caller passes an int32 (or other-width integer) the IR's builder.store
    would fail with a type-mismatch error during lowering rather than the
    clean TypingError this typing-time guard provides."""
    @njit
    def caller():
        # Passing an int32 value where intp is required
        _store_addr_to_named_global("test_global", np.int32(42))

    with pytest.raises(TypingError, match=r"_store_addr_to_named_global.*intp"):
        caller()


def test_cres_cacheable_global_name_distinguishes_nested_functions():
    """Two functions with the same __module__ + __name__ but distinct
    __qualname__ (e.g. nested helpers from different outer scopes) must
    get distinct LLVM global symbols. Without __qualname__ in the FQN,
    the second cres_cacheable setup would overwrite the first's address
    slot and indirect callers would invoke the wrong function pointer."""
    from numbox.utils.highlevel import _cres_cacheable_global_name

    def outer_a():
        def impl():
            pass
        return impl

    def outer_b():
        def impl():
            pass
        return impl

    impl_a = outer_a()
    impl_b = outer_b()
    # Both impl_a and impl_b have __name__ == "impl" and the same
    # __module__ — only __qualname__ distinguishes them.
    assert impl_a.__name__ == impl_b.__name__ == "impl"
    assert impl_a.__module__ == impl_b.__module__
    assert impl_a.__qualname__ != impl_b.__qualname__
    name_a = _cres_cacheable_global_name(impl_a)
    name_b = _cres_cacheable_global_name(impl_b)
    assert name_a != name_b, (
        f"distinct nested functions got identical global symbol {name_a!r}"
    )


def test_cres_cacheable_rejects_non_signature():
    """cres_cacheable mirrors cres's upfront Signature validation, so a
    typo like signatures.get('does_not_exist') (which returns None)
    fails with a clean ValueError naming the binding instead of an
    AttributeError on `sig.args` at import time."""
    from numbox.utils.highlevel import cres_cacheable

    with pytest.raises(ValueError, match=r"cres_cacheable.*Signature"):
        cres_cacheable(None)

    with pytest.raises(ValueError, match=r"cres_cacheable.*Signature"):
        cres_cacheable("not a signature")


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
