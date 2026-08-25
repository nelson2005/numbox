"""The originally reported shape, cross process: a factory over ExternalFunction values.

Unlike a ctypes pointer, an ``ExternalFunction`` carries no address, so the two bodies
are byte-for-byte identical from one process to the next and the only variable is which
one is built first.
"""
import math
import os

from numba import njit, types
from numba.core.types import float64

from numbox.utils.highlevel import cres

SIN = types.ExternalFunction("sin", float64(float64))
COS = types.ExternalFunction("cos", float64(float64))


def bind(ext):
    @cres(float64(float64))
    def body(x):
        return ext(x) + 3.0
    return body


if os.environ["ORDER"] == "AB":
    sin_bound = bind(SIN)
    cos_bound = bind(COS)
else:
    cos_bound = bind(COS)
    sin_bound = bind(SIN)


@njit(float64(float64), cache=True)
def const_sin(x):
    return sin_bound(x)


@njit(float64(float64), cache=True)
def const_cos(x):
    return cos_bound(x)


def served(f):
    hits = sum(f.stats.cache_hits.values())
    misses = sum(f.stats.cache_misses.values())
    return "served" if hits and not misses else "compiled" if misses else f"{hits}/{misses}"


want_sin = math.sin(0.5) + 3.0
want_cos = math.cos(0.5) + 3.0
got_sin = const_sin(0.5)
got_cos = const_cos(0.5)
print(f"ORDER={os.environ['ORDER']} "
      f"const_sin={got_sin:.6f} want {want_sin:.6f} {'ok' if got_sin == want_sin else 'WRONG'} "
      f"[{served(const_sin)}]  "
      f"const_cos={got_cos:.6f} want {want_cos:.6f} {'ok' if got_cos == want_cos else 'WRONG'} "
      f"[{served(const_cos)}]")
