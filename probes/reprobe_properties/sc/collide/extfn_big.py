"""Same collision shape with a body too large to inline, to check whether the PR
branch's link-the-body-in lowering is itself order-dependent (its weak-definition
hazard) rather than merely different."""
import json
import os
import warnings

from numba import njit
from numba.core.types import ExternalFunction, float64

from numbox.utils.highlevel import cres

COS = ExternalFunction("cos", float64(float64))
SIN = ExternalFunction("sin", float64(float64))


def bind(c_function):
    @cres(float64(float64))
    def body(x):
        s = 0.0
        for i in range(200):
            s += c_function(x + i * 0.001)
            s += c_function(x - i * 0.002)
            s *= 0.999
        return s + 3.0
    return body


ORDER = os.environ["ORDER"]
A = None
if ORDER == "b":
    B = bind(SIN)
elif ORDER == "ab":
    A = bind(COS)
    B = bind(SIN)
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
out["b_alias"] = getattr(B, "jit_alias", "NOATTR")
out["a_alias"] = getattr(A, "jit_alias", "NOATTR") if A is not None else "NOTBUILT"
if A is not None:
    out["arg_a"] = arg_call(A, 4.0)
s = const_b.stats
out["const_b_hits"] = sum(s.cache_hits.values())
out["const_b_misses"] = sum(s.cache_misses.values())
out["warnings"] = sorted(set(w.category.__name__ + ": " + str(w.message)[:120] for w in caught))
print("PROBEJSON " + json.dumps(out))
