"""Three producers whose bodies raise, for the exception-propagation property.

Padded so the @proxy-decorated function clears the cache-anchor minimum line.
"""
from numba import float64, njit
from numba.core.types.function_type import CompileResultWAP

from numbox.core.proxy.proxy import proxy
from numbox.utils.derive_wap import rewrap_derive
from numbox.utils.highlevel import cres

SIG = float64(float64)


@proxy(SIG)
def pboom(x):
    if x > 0.0:
        raise ValueError("proxy boom")
    return x


@cres(SIG)
def cboom(x):
    if x > 0.0:
        raise ValueError("cres boom")
    return x


@njit(SIG)
def rawboom(x):
    if x > 0.0:
        raise ValueError("raw boom")
    return x


PROXY_AS = pboom.as_func
CRES_AS = cboom
REWRAP_AS = rewrap_derive(CompileResultWAP(rawboom.get_compile_result(SIG)))
