"""The FOLLOWUP's own collision shape: a factory over two ExternalFunction values.

Unlike a ctypes pointer, an ExternalFunction lowers to a plain symbol reference, so
the body carries no dynamic global and a caller that links it in stays cacheable.
That makes this the shape where the PR branch is both cacheable and correct.

ORDER=b   -> only the sin body exists
ORDER=ab  -> cos body first, sin body second
"""
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
        return c_function(x) + 3.0
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
out["expected_sin"] = 2.243197504692072
out["cos_value"] = 2.3463737622865774
out["b_alias"] = getattr(B, "jit_alias", "NOATTR")
out["a_alias"] = getattr(A, "jit_alias", "NOATTR") if A is not None else "NOTBUILT"
s = const_b.stats
out["const_b_hits"] = sum(s.cache_hits.values())
out["const_b_misses"] = sum(s.cache_misses.values())
out["warnings"] = sorted(set(w.category.__name__ + ": " + str(w.message)[:120] for w in caught))
print("PROBEJSON " + json.dumps(out))
