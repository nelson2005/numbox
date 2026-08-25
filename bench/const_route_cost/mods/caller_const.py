"""Cached callers of the benchmark bindings, one per route, compiled lazily so each compile can be timed."""
from numba import njit

import binding_big
import binding_small
import cres_small

SMALL = binding_small.small.as_func
BIG = binding_big.big.as_func
CSMALL = cres_small.csmall


@njit(cache=True)
def const_small(x):
    return SMALL(x) + 1.0


@njit(cache=True)
def const_big(x):
    return BIG(x) + 1.0


@njit(cache=True)
def const_cres(x):
    return CSMALL(x) + 1.0


@njit(cache=True)
def disp_small(x):
    return binding_small.small(x) + 1.0


@njit(cache=True)
def plain(x):
    return x * 2.0 + 2.0
