import math
from numba import njit
from ctmod import make, sin_p, cos_p

s = make(sin_p)
c = make(cos_p)
print("alias_sin =", getattr(s, "jit_alias", "<none>"))
print("alias_cos =", getattr(c, "jit_alias", "<none>"))
print("collide   =", getattr(s, "jit_alias", 1) == getattr(c, "jit_alias", 2))


@njit(cache=False)
def call_sin(x):
    return s(x)


@njit(cache=False)
def call_cos(x):
    return c(x)


print("CONST sin(0.0) ->", call_sin(0.0), "want", math.sin(0.0))
print("CONST cos(0.0) ->", call_cos(0.0), "want", math.cos(0.0))
