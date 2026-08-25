from numba import njit
from numba.core.types import float64

from . import bmod

hundred = bmod.bind(bmod.PROTO(bmod.c_times_hundred.address))


@njit(float64(float64), cache=True)
def call_a(x):
    return hundred(x)
