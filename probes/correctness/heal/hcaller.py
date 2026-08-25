from numba import njit
from hbind import hpd, hc, stable


@njit(cache=True)
def cp(x):
    return hpd(x)


@njit(cache=True)
def cc(x):
    return hc(x)


@njit(cache=True)
def cstable(x):
    return stable(x)
