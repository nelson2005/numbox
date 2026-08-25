import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from numba import njit  # noqa: E402

import rbodies  # noqa: E402


@njit
def const_proxy(x):
    return rbodies.PROXY_AS(x)


@njit
def const_cres(x):
    return rbodies.CRES_AS(x)


@njit
def const_rewrap(x):
    return rbodies.REWRAP_AS(x)


@njit
def arg_call(f, x):
    return f(x)


def attempt(fn, *args):
    try:
        return "returned " + repr(fn(*args))
    except Exception as exc:
        return "raised " + type(exc).__name__


out = {}
out["tree"] = os.environ.get("PROBE_TREE", "?")
out["const_proxy"] = attempt(const_proxy, 1.0)
out["const_cres"] = attempt(const_cres, 1.0)
out["const_rewrap"] = attempt(const_rewrap, 1.0)
out["arg_proxy"] = attempt(arg_call, rbodies.PROXY_AS, 1.0)
out["arg_cres"] = attempt(arg_call, rbodies.CRES_AS, 1.0)
out["arg_rewrap"] = attempt(arg_call, rbodies.REWRAP_AS, 1.0)
out["dispatcher_proxy"] = attempt(rbodies.pboom, 1.0)
print("PROBEJSON " + json.dumps(out))
