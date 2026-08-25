"""Body module whose two derives get edited between runs.

Padded so the @proxy-decorated function clears the cache-anchor minimum line, and
kept at a fixed line count so an edit changes only the constant.
"""
from numba import float64

from numbox.core.proxy.proxy import proxy
from numbox.utils.highlevel import cres

SIG = float64(float64)
FACTOR = 2.0


@proxy(SIG)
def ebody(x):
    return x * 5.0


@cres(SIG)
def cbody(x):
    return x * 5.0


PROXY_AS = ebody.as_func
CRES_AS = cbody
