"""A Work graph with two Derived nodes whose derives are factory-made closures
over different ctypes function pointers (the C-bindings idiom)."""
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
x = End(name="x", init_value=0.0, registry=reg)
s = Derived(name="s", init_value=0.0, derive=make(sin_p), sources=(x,), registry=reg)
c = Derived(name="c", init_value=0.0, derive=make(cos_p), sources=(x,), registry=reg)
t = Derived(name="t", init_value=0.0, derive=lambda a, b: a + 10.0 * b, sources=(s, c), registry=reg)
access = make_graph(t, registry=reg)
access.t.calculate()
print("t =", access.t.data, "want", math.sin(0.0) + 10.0 * math.cos(0.0))
