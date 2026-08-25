from numba import njit
from derives_ok import da, db


@njit(cache=True)
def ca(x):
    return da(x)


@njit(cache=True)
def cb(x):
    return db(x)
