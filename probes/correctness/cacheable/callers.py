from numba import njit
from bind import pderive, cbody, fderive


@njit(cache=True)
def const_proxy(x):
    return pderive(x)


@njit(cache=True)
def const_cres(x):
    return cbody(x)


@njit(cache=True)
def const_foreign(x):
    return fderive(x)
