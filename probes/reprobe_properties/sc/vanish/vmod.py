"""A proxy_if_available binding that is present or absent by environment.

PRESENT=1 gives a real library object, otherwise one with no attributes at all, so
the binding takes the stub path exactly as it would against an older shared library.
BODY=c uses a real C symbol through numbox's own binding machinery; anything else
uses a pure-Python body.
"""
import ctypes
import os

from numba.core.types import float64

from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.utils import load_lib
from numbox.core.proxy.proxy import proxy_if_available


class _Empty:
    pass


load_lib("m")
LIB = ctypes.CDLL(None) if os.environ.get("PRESENT") == "1" else _Empty()

if os.environ.get("BODY") == "c":
    @proxy_if_available(LIB, float64(float64))
    def cos(x):
        return _call_lib_func("cos", (x,))
else:
    @proxy_if_available(LIB, float64(float64))
    def cos(x):
        return x * 2.0

BINDING = cos
