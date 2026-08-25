"""Is a refused derive's machine code kept alive?

A published derive is pinned for the process lifetime by _JIT_ALIASES. A refused
one is not, and its constant callers bake the raw entry address. If the only
reference to the derive is dropped, the compile result and its library go with it
while the baked address stays in already-compiled code.
"""
import ctypes
import gc
import json
import os
import sys

from numba import cfunc, njit
from numba.core.types import float64

from numbox.utils import derive_wap as dw
from numbox.utils.highlevel import cres

PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)


@cfunc(float64(float64))
def _c_hundred(x):
    return x * 100.0


@cfunc(float64(float64))
def _c_one(x):
    return x + 1.0


def bind(p):
    @cres(float64(float64))
    def body(x):
        return p(x) + 3.0
    return body


A = bind(PROTO(_c_hundred.address))
B = bind(PROTO(_c_one.address))

out = {}
out["tree"] = os.environ.get("PROBE_TREE", "?")
out["a_alias"] = getattr(A, "jit_alias", "NOATTR")
out["b_alias"] = getattr(B, "jit_alias", "NOATTR")
out["b_pinned"] = any(v is B.cres for v in dw._JIT_ALIASES.values())


@njit(float64(float64))
def const_b(x):
    return B(x)


out["before_drop"] = const_b(4.0)
print("PROBEJSON " + json.dumps(out))
sys.stdout.flush()

del B
gc.collect()
gc.collect()
print("AFTERDROP " + repr(const_b(4.0)))
sys.stdout.flush()
