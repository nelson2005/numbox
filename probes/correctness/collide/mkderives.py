"""Factory producing same-qualname closures whose only difference is an
opaque-to-the-fingerprint captured value (another derive)."""
from numba import float64
from numbox.utils.highlevel import cres


@cres(float64(float64))
def inner_a(x):
    return x * 2.0


@cres(float64(float64))
def inner_b(x):
    return x * 100.0


def make(inner):
    @cres(float64(float64))
    def outer(x):
        return inner(x) + 1.0
    return outer
