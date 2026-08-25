from numba import njit
from derives_order import o2


@njit(cache=True)
def caller_o2(x):
    return o2(x)
