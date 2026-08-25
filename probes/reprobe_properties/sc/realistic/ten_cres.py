"""Ten distinct cres-compiled functions, the ordinary shape."""
from numba.core.types import float64

from numbox.utils.highlevel import cres


@cres(float64(float64))
def c0(x):
    return x + 0.5


@cres(float64(float64))
def c1(x):
    return x * 1.5


@cres(float64(float64))
def c2(x):
    return x - 2.5


@cres(float64(float64))
def c3(x):
    return x / 3.5


@cres(float64(float64))
def c4(x):
    return x ** 2


@cres(float64(float64))
def c5(x):
    return -x


@cres(float64(float64))
def c6(x):
    return abs(x)


@cres(float64(float64))
def c7(x):
    if x < 0:
        raise ValueError("negative")
    return x


@cres(float64(float64))
def c8(x):
    s = 0.0
    for i in range(3):
        s += x + i
    return s


@cres(float64(float64))
def c9(x):
    return x if x > 1.0 else 1.0


ALL = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
