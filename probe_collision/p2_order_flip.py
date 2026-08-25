"""Cross-process order flip: does a warm const caller still run its own body?

Two factory-made bodies whose fingerprints collide, so they mint one alias name.
Which of them OWNS that alias is decided by construction order, which ORDER picks.
"""
import ctypes
import os

from numba import cfunc, njit
from numba.core.types import float64

from numbox.utils.highlevel import cres

PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)


@cfunc(float64(float64))
def _c_times_hundred(x):
    return x * 100.0


@cfunc(float64(float64))
def _c_plus_one(x):
    return x + 1.0


def bind(c_function):
    @cres(float64(float64))
    def body(x):
        return c_function(x) + 3.0
    return body


if os.environ["ORDER"] == "AB":
    hundred = bind(PROTO(_c_times_hundred.address))
    plus_one = bind(PROTO(_c_plus_one.address))
else:
    plus_one = bind(PROTO(_c_plus_one.address))
    hundred = bind(PROTO(_c_times_hundred.address))


@njit(float64(float64), cache=True)
def const_hundred(x):
    return hundred(x)


@njit(float64(float64), cache=True)
def const_plus_one(x):
    return plus_one(x)


def served(f):
    hits = sum(f.stats.cache_hits.values())
    misses = sum(f.stats.cache_misses.values())
    return "served" if hits and not misses else "compiled" if misses else f"{hits}/{misses}"


print(f"ORDER={os.environ['ORDER']} "
      f"alias_hundred={'yes' if hundred.jit_alias else 'NO'} "
      f"alias_plus_one={'yes' if plus_one.jit_alias else 'NO'} "
      f"const_hundred={const_hundred(4.0)} (want 403.0) "
      f"const_plus_one={const_plus_one(4.0)} (want 8.0) "
      f"{served(const_hundred)} {served(const_plus_one)}")
