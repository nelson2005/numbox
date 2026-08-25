from numba import njit
from gsmod import scaled


@njit(cache=True)
def const_scaled(x):
    return scaled(x)
