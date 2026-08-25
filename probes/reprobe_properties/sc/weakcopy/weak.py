"""Two same-qualname bodies that the fingerprint DOES tell apart, each too large
to inline, each reached as a compile-time constant.

On a tree that links the callee's library into the caller, both callers carry a
definition of the same mangled name; on a tree that references an alias, each
caller names its own body.
"""
import json
import os
import warnings

from numba import njit
from numba.core.types import float64

from numbox.utils.highlevel import cres


def bind(k):
    @cres(float64(float64))
    def body(x):
        s = 0.0
        for i in range(200):
            s += (x + i * 0.001) * k
            s -= (x - i * 0.002) / k
            s *= 0.999
        return s + 3.0
    return body


A = bind(2.0)
B = bind(7.0)


@njit(cache=True)
def const_a(x):
    return A(x)


@njit(cache=True)
def const_b(x):
    return B(x)


@njit(cache=True)
def arg_call(f, x):
    return f(x)


out = {}
out["tree"] = os.environ.get("PROBE_TREE", "?")
out["phase"] = os.environ.get("PROBE_PHASE", "?")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out["const_a"] = const_a(4.0)
    out["const_b"] = const_b(4.0)
    out["arg_a"] = arg_call(A, 4.0)
    out["arg_b"] = arg_call(B, 4.0)
out["const_a_matches_arg_a"] = out["const_a"] == out["arg_a"]
out["const_b_matches_arg_b"] = out["const_b"] == out["arg_b"]
out["a_alias"] = getattr(A, "jit_alias", "NOATTR")
out["b_alias"] = getattr(B, "jit_alias", "NOATTR")
out["aliases_differ"] = out["a_alias"] != out["b_alias"]
out["warnings"] = sorted(set(w.category.__name__ + ": " + str(w.message)[:120] for w in caught))
print("PROBEJSON " + json.dumps(out))
