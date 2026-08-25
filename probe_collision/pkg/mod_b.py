from numba import njit
from numba.core.types import float64

from . import bmod

plus_one = bmod.bind(bmod.PROTO(bmod.c_plus_one.address))


@njit(float64(float64), cache=True)
def call_b(x):
    return plus_one(x)
