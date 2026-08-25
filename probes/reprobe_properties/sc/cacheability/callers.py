"""cache=True callers, one per producer per route. Lazy so compilation happens
inside the probe's warning capture."""
from numba import njit

from bodies import PROXY_AS, CRES_AS, REWRAP_AS


@njit(cache=True)
def const_proxy(x):
    return PROXY_AS(x)


@njit(cache=True)
def const_cres(x):
    return CRES_AS(x)


@njit(cache=True)
def const_rewrap(x):
    return REWRAP_AS(x)


@njit(cache=True)
def arg_any(x, f):
    return f(x)
