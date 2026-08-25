"""How many derives out of a realistic set are refused an alias.

Counts publishes and refusals per shape. The shapes span what a bindings library
and its users actually build, not what the repository suite happens to do.
"""
import ctypes
import importlib
import json
import os

import numba
from numba import cfunc, njit
from numba.core.types import ExternalFunction, float64, int64

from numbox.utils import derive_wap as dw
from numbox.utils.highlevel import cres

CURRENT = []
_orig = dw._publish_jit_alias


def _publish(cres_, py_func, jit_options, jit_address):
    out = _orig(cres_, py_func, jit_options, jit_address)
    CURRENT.append(out)
    return out


dw._publish_jit_alias = _publish

TALLY = {}


def measure(name, fn):
    CURRENT.clear()
    value = fn()
    TALLY[name] = {"derives": len(CURRENT), "refused": sum(1 for x in CURRENT if x is None)}
    CURRENT.clear()
    return value


def shipped():
    import numbox.core.bindings.libm  # noqa: F401
    import numbox.core.bindings.libc  # noqa: F401
    import numbox.core.bindings.errno  # noqa: F401
    import numbox.core.bindings.stdio  # noqa: F401
    import numbox.core.bindings.strerror  # noqa: F401
    import numbox.core.bindings.sqlite.conn  # noqa: F401
    import numbox.core.bindings.sqlite.stmt  # noqa: F401
    import numbox.core.bindings.sqlite.value  # noqa: F401
    import numbox.core.bindings.sqlite.result  # noqa: F401
    import numbox.core.bindings.sqlite.vtable  # noqa: F401


def ten_cres_mod():
    import ten_cres
    return ten_cres


def ten_proxy_mod():
    import ten_proxy
    return ten_proxy


def make_scaled(k):
    @cres(float64(float64))
    def body(x):
        return x * k + 1.0
    return body


def make_shift(n):
    @cres(int64(int64))
    def body(x):
        return x + n
    return body


def make_named(tag):
    @cres(float64(float64))
    def body(x):
        return x + len(tag)
    return body


def make_via_proxy(binding):
    @cres(float64(float64))
    def body(x):
        return binding(x) + 1.0
    return body


def make_over_cptr(p):
    @cres(float64(float64))
    def body(x):
        return p(x) + 1.0
    return body


def make_over_ext(f):
    @cres(float64(float64))
    def body(x):
        return f(x) + 1.0
    return body


def make_over_derive(inner):
    @cres(float64(float64))
    def body(x):
        return inner(x) + 1.0
    return body


measure("shipped_numbox_bindings", shipped)
TEN_CRES = measure("ten_distinct_cres_functions", ten_cres_mod)
TEN_PROXY = measure("ten_distinct_proxy_bindings", ten_proxy_mod)
measure("factory_over_float_parameter", lambda: [make_scaled(float(i)) for i in range(5)])
measure("factory_over_int_parameter", lambda: [make_shift(i) for i in range(5)])
measure("factory_over_string_parameter", lambda: [make_named("a" * i) for i in range(1, 6)])
measure("factory_over_a_proxy_binding",
        lambda: [make_via_proxy(b) for b in TEN_PROXY.BINDINGS[:5]])

PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
CFUNCS = []
for _k in (1.0, 2.0, 3.0, 4.0, 5.0):
    def _mk(k):
        @cfunc(float64(float64))
        def f(x):
            return x * k
        return f
    CFUNCS.append(_mk(_k))
CPTRS = [PROTO(f.address) for f in CFUNCS]
measure("factory_over_a_ctypes_pointer", lambda: [make_over_cptr(p) for p in CPTRS])

EXTS = [ExternalFunction(n, float64(float64)) for n in ("cos", "sin", "tan", "exp", "log")]
measure("factory_over_an_external_function", lambda: [make_over_ext(f) for f in EXTS])

BASES = [make_scaled(float(100 + i)) for i in range(5)]
measure("factory_over_another_derive", lambda: [make_over_derive(d) for d in BASES])

measure("reimport_of_one_module", lambda: importlib.reload(TEN_CRES))


def build_graph():
    import graph_mod
    return graph_mod


measure("numbox_work_graph", build_graph)

out = {}
out["tree"] = os.environ.get("PROBE_TREE", "?")
out["numba"] = numba.__version__
out["tally"] = TALLY
out["total_derives"] = sum(v["derives"] for v in TALLY.values())
out["total_refused"] = sum(v["refused"] for v in TALLY.values())
print("PROBEJSON " + json.dumps(out))
