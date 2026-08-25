"""Tests for the content-fingerprint walker (numbox.utils.fingerprint).

Two concerns run through these: canonicalizing a *value* stably (so that two artifacts
differing only in captured data get different digests), and telling a genuine global read
(``LOAD_GLOBAL``) apart from an attribute access (``LOAD_ATTR``) whose name merely collides
with a module global. ``co_names`` pools the latter two, so folding globals off it either
over-invalidates the fingerprint (a colliding global that canonicalizes) or makes the whole
function un-fingerprintable (a colliding global that does not) -- which, on the builder's
derive path, forces an address-bearing fallback and unbounded cache growth.
``_loaded_global_names`` reads the opcodes instead.
"""
import textwrap

import numpy as np

from numbox.utils.fingerprint import (
    _Unfingerprintable, _canon_value, _fingerprint_function, _loaded_global_names,
    _referenced_global_names,
)


def _make(src, **globs):
    """Compile ``src`` (defining ``f``) with ``globs`` as its module globals."""
    ns = dict(globs)
    exec(textwrap.dedent(src), ns)  # nosec B102 - test helper compiling a fixed snippet
    return ns["f"]


def test_canon_value_structured_dtype_preserves_layout():
    # dtype.str is |V<n> for structured dtypes (field layout erased), so same-byte
    # different-layout arrays would collide. The full descriptor discriminates them.
    raw = (1).to_bytes(4, "little") + (2).to_bytes(4, "little")
    two_i4 = np.frombuffer(raw, dtype=np.dtype([("a", "<i4"), ("b", "<i4")])).copy()
    two_f4 = np.frombuffer(raw, dtype=np.dtype([("a", "<f4"), ("b", "<f4")])).copy()
    one_i8 = np.frombuffer(raw, dtype=np.dtype([("x", "<i8")])).copy()

    assert two_i4.tobytes() == two_f4.tobytes() == one_i8.tobytes()
    canons = {_canon_value(a, set()) for a in (two_i4, two_f4, one_i8)}
    assert len(canons) == 3, "structured-dtype arrays with identical bytes collided"


def test_canon_value_plain_dtype_unchanged():
    canon = _canon_value(np.arange(3, dtype=np.float64), set())
    assert canon.startswith("ndarray(<f8;")


def test_proxy_binding_callback_fingerprints_cleanly_and_distinctly():
    # A callback that calls a @proxy'd binding used to abort the walker (the
    # wrapper recurses into its @intrinsic, which has no canonical form), so the
    # cache digest degraded to a coarse fallback. The dispatcher is now identified
    # by its stable alias, so the callback fingerprints cleanly and two callbacks
    # calling different bindings differ.
    from numbox.core.bindings.sqlite.value import sqlite3_value_int64, sqlite3_value_double

    def cb_int(v):
        return sqlite3_value_int64(v)

    def cb_dbl(v):
        return sqlite3_value_double(v)

    fp_int = _fingerprint_function(cb_int, set())   # must not raise
    fp_dbl = _fingerprint_function(cb_dbl, set())
    assert "proxy(numbox_pxy_" in fp_int
    assert fp_int != fp_dbl


def test_proxy_binding_canon_does_not_recurse_into_intrinsic():
    from numbox.core.bindings.sqlite.value import sqlite3_value_int64

    canon = _canon_value(sqlite3_value_int64, set())
    assert canon.startswith("proxy(numbox_pxy_sqlite3_value_int64_")


def test_unfingerprintable_value_still_raises_for_strict_walker():
    # The strict walker must keep raising on a genuinely un-canonicalizable value
    # (so degrade-to-uncached consumers still trigger); only the best-effort walker
    # swallows it.
    class Opaque:
        pass

    def cb():
        return Opaque()

    # Opaque() is a const-less local; reference it via a closure to force the walk.
    marker = Opaque()

    def cb2():
        return marker

    try:
        _fingerprint_function(cb2, set())
    except _Unfingerprintable:
        pass
    else:
        raise AssertionError("expected _Unfingerprintable for an opaque closure value")


def test_loaded_global_names_distinguishes_globals_from_attributes():
    """An attribute access is a ``LOAD_ATTR``; only ``LOAD_GLOBAL`` / ``LOAD_NAME`` name globals."""
    attr = _make("def f(a):\n    return a.c1\n")
    assert _referenced_global_names(attr.__code__) == {"c1"}, "co_names should pool the attribute name"
    assert _loaded_global_names(attr.__code__) == set(), "an attribute access names no global"

    glob = _make("def f(x):\n    return x + K\n", K=5)
    assert "K" in _loaded_global_names(glob.__code__)

    nested = _make("def f(xs):\n    return [y + K for y in xs]\n", K=5)
    assert "K" in _loaded_global_names(nested.__code__), "a global read inside a comprehension counts"


def test_fingerprint_ignores_a_global_shadowed_by_an_attribute_name():
    """A record-field / chained attribute whose name matches a module global must not fold that global.

    With a canonicalizable colliding global the old behaviour over-invalidated (folded a value the body
    never reads); with an un-canonicalizable one it raised, discarding the whole fingerprint. Neither may
    happen now: ``a.c1`` reads an attribute, not the global ``c1``.
    """
    canon = _make("def f(a):\n    return a.c1\n", c1=3)
    assert "c1=" not in _fingerprint_function(canon, set()), "folded a global the body only accesses as an attribute"

    uncanon = _make("def f(a):\n    return a.c1\n", c1=np.array([object()], dtype=object))
    # Must not raise despite the colliding global having no canonical form.
    fp = _fingerprint_function(uncanon, set())
    assert "c1=" not in fp

    # The fingerprint depends on the attribute name itself (it is still in co_names), so a different
    # field is a different body -- the fix drops only the spurious global-value fold, not the attribute.
    other = _make("def f(a):\n    return a.c2\n", c1=3)
    assert _fingerprint_function(canon, set()) != _fingerprint_function(other, set())


def test_fingerprint_still_folds_a_genuine_global_read():
    """A real ``LOAD_GLOBAL`` read is folded, and changing its value re-keys the fingerprint."""
    f = _make("def f(x):\n    return x * K\n", K=2)
    assert "K=" in _fingerprint_function(f, set())
    g = _make("def f(x):\n    return x * K\n", K=9)
    assert _fingerprint_function(f, set()) != _fingerprint_function(g, set())

    # An un-canonicalizable *genuine* global read still raises -- that degrade path is intended and unchanged.
    bad = _make("def f(x):\n    return x * K\n", K=np.array([object()], dtype=object))
    try:
        _fingerprint_function(bad, set())
    except _Unfingerprintable:
        pass
    else:
        raise AssertionError("an un-canonicalizable genuine global read should still be _Unfingerprintable")


def test_module_attribute_read_still_rekeys(tmp_path):
    """Non-regression: a value read through a module (``cfg.SCALE``) is still folded, so changing it
    re-keys; an attribute present on the module but never referenced (``cfg.OFFSET``) does not. The fix keeps
    passing the full ``co_names`` set to the module-attribute walk, so this path is unaffected."""
    import types

    cfg = types.ModuleType("cfgfake")
    cfg.SCALE = 2.0
    cfg.OFFSET = 9.0
    f = _make("def f(x):\n    return x * cfg.SCALE\n", cfg=cfg)

    before = _fingerprint_function(f, set())
    cfg.OFFSET = 99.0
    assert _fingerprint_function(f, set()) == before, "an unread module attribute must not re-key"
    cfg.SCALE = 3.0
    assert _fingerprint_function(f, set()) != before, "a read module attribute must re-key"


def test_an_external_function_is_canonicalized_by_symbol_and_signature():
    """Two bindings a factory made over different C symbols must not share a fingerprint.

    ``ExternalFunction`` has no ``_canon_value`` branch of its own until one is written for it,
    so it used to reach the terminal ``raise`` and, on the best-effort path that the ``@proxy``
    alias runs on, collapse to ``<opaque:ExternalFunction>``: the same placeholder for every
    symbol. numba's own key for the type is ``(symbol, signature)``, both of which mean the same
    thing in the next process, so folding those separates the two without costing stability.
    """
    from numba.core.types import ExternalFunction, float64

    sin = _canon_value(ExternalFunction("sin", float64(float64)), set())
    cos = _canon_value(ExternalFunction("cos", float64(float64)), set())
    assert sin != cos, "two C symbols collapsed onto one canonical form"
    assert "sin" in sin and "cos" in cos
    assert _canon_value(ExternalFunction("sin", float64(float64)), set()) == sin, (
        "the canonical form has to be the same for an equal value, or it is not a fingerprint"
    )


def test_a_library_bound_ctypes_pointer_is_canonicalized_by_library_and_symbol():
    """The same for a pointer reached by attribute access on a loaded library."""
    import ctypes

    libm = ctypes.CDLL("libm.so.6")
    floor = _canon_value(libm.floor, set())
    ceil = _canon_value(libm.ceil, set())
    assert floor != ceil, "two C functions from one library collapsed onto one canonical form"
    assert _canon_value(ctypes.CDLL("libm.so.6").floor, set()) == floor, (
        "the canonical form must not depend on which handle the library was opened through"
    )


def test_a_ctypes_pointer_built_from_an_address_stays_unfingerprintable():
    """The residual class, and why it is deliberate.

    A pointer built from a raw address carries no library and no symbol name. The only thing
    separating two of them is the address, which ASLR moves, so folding it would buy
    discrimination at the cost of the process-stability the fingerprint exists for. Refusing is
    the honest answer, and it leaves the caller to detect the collision instead of hiding it.
    """
    import ctypes
    import pytest

    libm = ctypes.CDLL("libm.so.6")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
    fp = proto(ctypes.cast(libm.floor, ctypes.c_void_p).value)
    with pytest.raises(_Unfingerprintable):
        _canon_value(fp, set())
