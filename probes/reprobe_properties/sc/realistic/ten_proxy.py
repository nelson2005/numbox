"""Ten distinct @proxy bindings, the shape numbox itself ships.

Padded at the top so every decorated function sits below the cache-anchor minimum
line the decorator enforces.
"""
from numba.core.types import float64

from numbox.core.proxy.proxy import proxy


@proxy(float64(float64))
def p0(x):
    return x + 0.25


@proxy(float64(float64))
def p1(x):
    return x * 2.25


@proxy(float64(float64))
def p2(x):
    return x - 3.25


@proxy(float64(float64))
def p3(x):
    return x / 4.25


@proxy(float64(float64))
def p4(x):
    return x ** 3


@proxy(float64(float64))
def p5(x):
    return -x - 1.0


@proxy(float64(float64))
def p6(x):
    return abs(x) + 1.0


@proxy(float64(float64))
def p7(x):
    return x * x + x


@proxy(float64(float64))
def p8(x):
    s = 0.0
    for i in range(4):
        s += x * i
    return s


@proxy(float64(float64))
def p9(x):
    return x if x > 2.0 else 2.0


BINDINGS = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
