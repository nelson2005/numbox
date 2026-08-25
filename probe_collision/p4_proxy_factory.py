"""Does the SAME collision reach the @proxy cfunc alias, which has no registry at all?

`_stable_cfunc_alias` folds the same `_body_fingerprint`, and `proxy` publishes with a
bare `ll.add_symbol`: no registry, no first-writer check, no refusal. Two factory-made
`@proxy` bodies over different C function pointers therefore mint one cfunc alias and
the second decoration rebinds it.
"""
import ctypes

from numba import cfunc, njit
from numba.core.types import float64

from numbox.core.proxy.proxy import _stable_cfunc_alias, proxy

PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)


@cfunc(float64(float64))
def c_times_hundred(x):
    return x * 100.0


@cfunc(float64(float64))
def c_plus_one(x):
    return x + 1.0


def make_binding(c_function):
    @proxy(float64(float64))
    def bound(x):
        return c_function(x) + 3.0
    return bound


hundred = make_binding(PROTO(c_times_hundred.address))
plus_one = make_binding(PROTO(c_plus_one.address))

print("cfunc aliases equal:",
      _stable_cfunc_alias(hundred.py_func, float64(float64))
      == _stable_cfunc_alias(plus_one.py_func, float64(float64)))
print("cc alias hundred:", hundred.as_func.jit_alias)
print("cc alias plus_one:", plus_one.as_func.jit_alias)


@njit(float64(float64))
def dispatch_hundred(x):
    return hundred(x)


@njit(float64(float64))
def dispatch_plus_one(x):
    return plus_one(x)


AF_HUNDRED = hundred.as_func
AF_PLUS_ONE = plus_one.as_func


@njit(float64(float64))
def const_hundred(x):
    return AF_HUNDRED(x)


@njit(float64(float64))
def const_plus_one(x):
    return AF_PLUS_ONE(x)


@njit
def as_argument(f, x):
    return f(x)


print()
print(f"{'route':<24} {'hundred (want 403.0)':<24} {'plus_one (want 8.0)'}")
print(f"{'dispatcher from python':<24} {hundred(4.0)!r:<24} {plus_one(4.0)!r}")
print(f"{'dispatcher from jit':<24} {dispatch_hundred(4.0)!r:<24} {dispatch_plus_one(4.0)!r}")
print(f"{'as_func constant':<24} {const_hundred(4.0)!r:<24} {const_plus_one(4.0)!r}")
print(f"{'as_func argument':<24} {as_argument(hundred.as_func, 4.0)!r:<24} "
      f"{as_argument(plus_one.as_func, 4.0)!r}")
