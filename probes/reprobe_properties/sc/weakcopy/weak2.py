"""The same two distinguishable bodies, with a construction order that varies.

The mangled name numba gives a body carries a per-process compile counter, so the
body compiled first in one run is not the body compiled first in the next. A caller
whose cached object names the callee by that mangled name is therefore keyed to a
name that does not identify the body either.
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


ORDER = os.environ["ORDER"]
A = None
if ORDER == "b":
    B = bind(7.0)
elif ORDER == "ab":
    A = bind(2.0)
    B = bind(7.0)
else:
    raise SystemExit("bad ORDER")


@njit(cache=True)
def const_b(x):
    return B(x)


@njit(cache=True)
def arg_call(f, x):
    return f(x)


out = {}
out["tree"] = os.environ.get("PROBE_TREE", "?")
out["order"] = ORDER
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out["const_b"] = const_b(4.0)
    out["arg_b"] = arg_call(B, 4.0)
out["agree"] = out["const_b"] == out["arg_b"]
out["b_llvm_name"] = B.cres.fndesc.llvm_func_name
out["b_alias"] = getattr(B, "jit_alias", "NOATTR")
s = const_b.stats
out["hits"] = sum(s.cache_hits.values())
out["misses"] = sum(s.cache_misses.values())
out["warnings"] = sorted(set(w.category.__name__ + ": " + str(w.message)[:120] for w in caught))
print("PROBEJSON " + json.dumps(out))
