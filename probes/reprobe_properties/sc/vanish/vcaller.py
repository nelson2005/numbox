from numba import njit
from numba.core.types import float64

from numbox.utils.highlevel import cres

import vmod


@cres(float64(float64))
def fallback(x):
    return x * 100.0


AS = vmod.BINDING.as_func if hasattr(vmod.BINDING, "as_func") else fallback


@njit(cache=True)
def call_it(x):
    return AS(x)
