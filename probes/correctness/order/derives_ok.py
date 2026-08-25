"""Two NON-colliding derives; ORDER flips which is compiled first."""
import os
from numba import float64
from numbox.utils.highlevel import cres


def mk_a():
    @cres(float64(float64))
    def a(x):
        return x * 2.0
    return a


def mk_b():
    @cres(float64(float64))
    def b(x):
        return x * 5.0
    return b


if os.environ.get("ORDER", "ab") == "ab":
    da = mk_a()
    db = mk_b()
else:
    db = mk_b()
    da = mk_a()
