"""The C-bindings idiom: a factory closing over a ctypes function pointer."""
import ctypes
import ctypes.util
from numba import float64
from numbox.utils.highlevel import cres

_libm = ctypes.CDLL(ctypes.util.find_library("m"))
_PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
sin_p = _PROTO(("sin", _libm))
cos_p = _PROTO(("cos", _libm))


def make(cfn):
    @cres(float64(float64))
    def wrapped(x):
        return cfn(x)
    return wrapped
