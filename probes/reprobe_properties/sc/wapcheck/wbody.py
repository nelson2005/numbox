from numba import njit
from numba.core.types import float64

SIG = float64(float64)


@njit(SIG, cache=True)
def cached_body(x):
    return x * 7.0
