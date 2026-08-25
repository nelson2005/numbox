from numba import njit

from body_mod import CRES_AS, PROXY_AS


@njit(cache=True)
def const_proxy(x):
    return PROXY_AS(x)


@njit(cache=True)
def const_cres(x):
    return CRES_AS(x)
