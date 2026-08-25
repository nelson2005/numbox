"""Cross-process collision probe with a variable construction order.

ORDER=b   -> only the "+1" body exists, so it publishes the alias
ORDER=ab  -> the "*100" body is built first and publishes, "+1" comes second
ORDER=ba  -> the "+1" body is built first and publishes

The cache=True caller `const_b` always calls the "+1" body and must always answer
5.0 for 4.0. numba's cache key is the caller's own code, so the object written in
one run is loaded in the others regardless of which body holds the alias.
"""
import ctypes
import json
import os
import warnings

from numba import cfunc, njit
from numba.core.types import float64

from numbox.utils.highlevel import cres

PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)


@cfunc(float64(float64))
def _c_times_hundred(x):
    return x * 100.0


@cfunc(float64(float64))
def _c_plus_one(x):
    return x + 1.0


def bind(c_function):
    @cres(float64(float64))
    def body(x):
        return c_function(x) + 3.0
    return body


ORDER = os.environ["ORDER"]
A = None
B = None
if ORDER == "b":
    B = bind(PROTO(_c_plus_one.address))
elif ORDER == "ab":
    A = bind(PROTO(_c_times_hundred.address))
    B = bind(PROTO(_c_plus_one.address))
elif ORDER == "ba":
    B = bind(PROTO(_c_plus_one.address))
    A = bind(PROTO(_c_times_hundred.address))
else:
    raise SystemExit("bad ORDER")


@njit(cache=True)
def const_b(x):
    return B(x)


@njit(cache=True)
def arg_call(f, x):
    return f(x)


out = {}
out["order"] = ORDER
out["tree"] = os.environ.get("PROBE_TREE", "?")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out["const_b"] = const_b(4.0)
    out["arg_b"] = arg_call(B, 4.0)
out["expected"] = 8.0
out["b_alias"] = getattr(B, "jit_alias", "NOATTR")
out["a_alias"] = getattr(A, "jit_alias", "NOATTR") if A is not None else "NOTBUILT"
s = const_b.stats
out["const_b_hits"] = sum(s.cache_hits.values())
out["const_b_misses"] = sum(s.cache_misses.values())
out["warnings"] = sorted(set(w.category.__name__ + ": " + str(w.message)[:120] for w in caught))
print("PROBEJSON " + json.dumps(out))
