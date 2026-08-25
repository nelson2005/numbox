import ctypes
import ctypes.util
import math
from numbox.core.work.builder import End, Derived, make_graph

_libm = ctypes.CDLL(ctypes.util.find_library("m"))
_PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
sin_p = _PROTO(("sin", _libm))
cos_p = _PROTO(("cos", _libm))


def make(cfn):
    def d(x):
        return cfn(x)
    return d


reg = {}
x = End(name="x", init_value=0.5, registry=reg)
s = Derived(name="s", init_value=0.0, derive=make(sin_p), sources=(x,), registry=reg)
c = Derived(name="c", init_value=0.0, derive=make(cos_p), sources=(s,), registry=reg)
access = make_graph(c, registry=reg)
access.c.calculate()
try:
    from numbox.utils.derive_wap import _JIT_ALIASES
    print("aliases:", sorted(_JIT_ALIASES))
except Exception as e:
    print("aliases n/a", e)
print("c =", access.c.data, "want", math.cos(math.sin(0.5)), "collide_would_give", math.sin(math.sin(0.5)))
