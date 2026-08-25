from numba import njit, float64
from numbox.utils.highlevel import cres
from absmod import scale


@cres(float64(float64))
def _fallback(x):
    return -1.0


AS = scale.as_func if hasattr(scale, "as_func") else _fallback


@njit(cache=True)
def const_scale(x):
    return AS(x)


@njit(cache=True)
def disp_scale(x):
    return scale(x)
