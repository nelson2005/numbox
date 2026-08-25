"""In-process battery: for each opaque-closure shape, check both call routes agree."""
import ctypes
import math
import numpy as np
from numba import cfunc, njit, types
from numba.core.types import float64
from numbox.utils.fingerprint import _body_fingerprint
from numbox.utils.highlevel import cres
import numbox.utils.derive_wap as dw

PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
results = []


@njit
def call_as_argument(f, x):
    return f(x)


def fp(d):
    return _body_fingerprint(d.cres.type_annotation.func_id.func)


def report(name, first, second, cf, cs, af, as_, want_f, want_s):
    ok = (cf == want_f and cs == want_s and af == want_f and as_ == want_s)
    results.append((name, fp(first) == fp(second), first.jit_alias is not None,
                    second.jit_alias is not None, cf, cs, af, as_, want_f, want_s, ok))


_sin = types.ExternalFunction("sin", float64(float64))
_cos = types.ExternalFunction("cos", float64(float64))


def bind_external(ext):
    @cres(float64(float64))
    def body(x):
        return ext(x) + 3.0
    return body


e1 = bind_external(_sin)
e2 = bind_external(_cos)


@njit(float64(float64))
def const_e1(x):
    return e1(x)


@njit(float64(float64))
def const_e2(x):
    return e2(x)


report("ExternalFunction sin/cos", e1, e2,
       const_e1(0.5), const_e2(0.5), call_as_argument(e1, 0.5), call_as_argument(e2, 0.5),
       math.sin(0.5) + 3.0, math.cos(0.5) + 3.0)


@cfunc(float64(float64))
def _c_times_hundred(x):
    return x * 100.0


@cfunc(float64(float64))
def _c_plus_one(x):
    return x + 1.0


def bind_ptr(fn):
    @cres(float64(float64))
    def body(x):
        return fn(x) + 3.0
    return body


p1 = bind_ptr(PROTO(_c_times_hundred.address))
p2 = bind_ptr(PROTO(_c_plus_one.address))


@njit(float64(float64))
def const_p1(x):
    return p1(x)


@njit(float64(float64))
def const_p2(x):
    return p2(x)


report("ctypes pointer", p1, p2,
       const_p1(4.0), const_p2(4.0), call_as_argument(p1, 4.0), call_as_argument(p2, 4.0),
       403.0, 8.0)


def bind_scalar_type(dt):
    @cres(float64(float64))
    def body(x):
        return float(dt(x))
    return body


d1 = bind_scalar_type(np.float32)
d2 = bind_scalar_type(np.float64)


@njit(float64(float64))
def const_d1(x):
    return d1(x)


@njit(float64(float64))
def const_d2(x):
    return d2(x)


report("numpy scalar type", d1, d2,
       const_d1(0.1), const_d2(0.1), call_as_argument(d1, 0.1), call_as_argument(d2, 0.1),
       float(np.float32(0.1)), 0.1)


def bind_dtype_obj(dt):
    @cres(float64(float64))
    def body(x):
        a = np.zeros(1, dtype=dt)
        a[0] = x
        return float(a[0])
    return body


q1 = bind_dtype_obj(np.dtype("float32"))
q2 = bind_dtype_obj(np.dtype("float64"))


@njit(float64(float64))
def const_q1(x):
    return q1(x)


@njit(float64(float64))
def const_q2(x):
    return q2(x)


report("np.dtype instance", q1, q2,
       const_q1(0.1), const_q2(0.1), call_as_argument(q1, 0.1), call_as_argument(q2, 0.1),
       float(np.float32(0.1)), 0.1)

hdr = (f"{'shape':<26} {'fpsame':<7} {'a1':<6} {'a2':<6} {'constA':<20} {'constB':<20} "
       f"{'argA':<20} {'argB':<20} {'wantA':<20} {'wantB':<20} ok")
print()
print(hdr)
for name, fpsame, a1, a2, cf, cs, af, as_, wf, ws, ok in results:
    print(f"{name:<26} {str(fpsame):<7} {str(a1):<6} {str(a2):<6} {cf!r:<20} {cs!r:<20} "
          f"{af!r:<20} {as_!r:<20} {wf!r:<20} {ws!r:<20} {'OK' if ok else 'FAIL'}")
print()
print("aliases in registry:", len(dw._JIT_ALIASES))
