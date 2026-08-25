from numba import njit
from dtmod import make, DT1, DT2

o1 = make(DT1)
o2 = make(DT2)
print("alias1 =", getattr(o1, "jit_alias", "<none>"))
print("alias2 =", getattr(o2, "jit_alias", "<none>"))
print("collide =", getattr(o1, "jit_alias", 1) == getattr(o2, "jit_alias", 2))


@njit(cache=False)
def c1(x):
    return o1(x)


@njit(cache=False)
def c2(x):
    return o2(x)


print("CONST o1 ->", c1(0.0), "(want 16.0)")
print("CONST o2 ->", c2(0.0), "(want 24.0)")
