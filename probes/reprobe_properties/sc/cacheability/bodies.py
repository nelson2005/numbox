"""Three producers of a first-class derive value, for the cacheability probe.

Padded so that the @proxy-decorated function sits well below the cache-anchor
minimum line the decorator enforces.
"""
from numba import float64, njit
from numba.core.types.function_type import CompileResultWAP

from numbox.core.proxy.proxy import proxy
from numbox.utils.highlevel import cres
from numbox.utils.derive_wap import rewrap_derive


SIG = float64(float64)


@proxy(SIG)
def pbody(x):
    return x * 2.0 + 1.0


@cres(SIG)
def cbody(x):
    return x * 3.0 + 1.0


@njit(SIG)
def rawbody(x):
    return x * 4.0 + 1.0


PROXY_AS = pbody.as_func
CRES_AS = cbody
REWRAP_AS = rewrap_derive(CompileResultWAP(rawbody.get_compile_result(SIG)))
