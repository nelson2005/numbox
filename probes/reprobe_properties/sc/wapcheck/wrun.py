"""Does the new py_func check refuse anything legitimate, and what does its
cache-restored fallback accept?"""
import functools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from numba import njit  # noqa: E402
from numba.core.types import float64  # noqa: E402

from numbox.utils.derive_wap import DeriveWAP  # noqa: E402

import wbody  # noqa: E402

SIG = float64(float64)
out = {}
out["tree"] = os.environ.get("PROBE_TREE", "?")
out["phase"] = os.environ.get("PROBE_PHASE", "?")


def attempt(label, fn):
    try:
        fn()
        out[label] = "ok"
    except Exception as exc:
        out[label] = type(exc).__name__ + ": " + str(exc)[:120]


cres_cached = wbody.cached_body.get_compile_result(SIG)
out["cached_type_annotation"] = type(cres_cached.type_annotation).__name__
out["cached_has_func_id"] = hasattr(cres_cached.type_annotation, "func_id")

attempt("A_no_py_func", lambda: DeriveWAP(cres_cached))
attempt("B_correct_py_func", lambda: DeriveWAP(cres_cached, py_func=wbody.cached_body.py_func))


@njit(SIG)
def other(x):
    return x + 1.0


attempt("C_wrong_py_func", lambda: DeriveWAP(cres_cached, py_func=other.py_func))


def make_same_name():
    def cached_body(x):
        return x * 999.0
    cached_body.__module__ = wbody.cached_body.py_func.__module__
    cached_body.__qualname__ = wbody.cached_body.py_func.__qualname__
    return cached_body


attempt("D_impostor_of_the_same_name", lambda: DeriveWAP(cres_cached, py_func=make_same_name()))


print("PROBEJSON " + json.dumps(out))
