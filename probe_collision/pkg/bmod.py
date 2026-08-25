"""Shared factory: one body, one qualname, one code object, per-binding C pointer."""
import ctypes

from numba import cfunc
from numba.core.types import float64

from numbox.utils.highlevel import cres

PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)


@cfunc(float64(float64))
def c_times_hundred(x):
    return x * 100.0


@cfunc(float64(float64))
def c_plus_one(x):
    return x + 1.0


def bind(c_function):
    @cres(float64(float64))
    def body(x):
        return c_function(x) + 3.0
    return body
