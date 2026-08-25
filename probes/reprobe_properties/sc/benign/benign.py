"""What refusing a benign duplicate costs a const caller.

Two derives with identical bodies (same source, same closure, same globals) mint
one alias. The alias does identify the body in this case, so sharing it is safe;
the fix refuses it anyway because the compile results differ. This measures what
that costs: whether the second derive's cache=True const caller still caches, and
whether it still answers correctly.
"""
import json
import os
import warnings

from numba import njit
from numba.core.types import float64

from numbox.utils.highlevel import cres


def make_doubler():
    @cres(float64(float64))
    def body(x):
        return x * 2.0 + 1.0
    return body


FIRST = make_doubler()
SECOND = make_doubler()


@njit(cache=True)
def use_first(x):
    return FIRST(x)


@njit(cache=True)
def use_second(x):
    return SECOND(x)


out = {}
out["tree"] = os.environ.get("PROBE_TREE", "?")
out["phase"] = os.environ.get("PROBE_PHASE", "?")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out["first"] = use_first(4.0)
    out["second"] = use_second(4.0)
out["expected"] = 9.0
out["first_alias"] = getattr(FIRST, "jit_alias", "NOATTR")
out["second_alias"] = getattr(SECOND, "jit_alias", "NOATTR")
out["same_alias"] = out["first_alias"] == out["second_alias"]
for label, fn in (("use_first", use_first), ("use_second", use_second)):
    s = fn.stats
    out[label + "_hits"] = sum(s.cache_hits.values())
    out[label + "_misses"] = sum(s.cache_misses.values())
out["warnings"] = sorted(set(w.category.__name__ + ": " + str(w.message)[:130] for w in caught))
print("PROBEJSON " + json.dumps(out))
