import sys
from numba import njit, float64
from mkderives import make, inner_a, inner_b

oa = make(inner_a)
ob = make(inner_b)

print("alias_a =", getattr(oa, "jit_alias", "<none>"))
print("alias_b =", getattr(ob, "jit_alias", "<none>"))
print("collide =", getattr(oa, "jit_alias", 1) == getattr(ob, "jit_alias", 2))
print("addr_a  =", hex(oa.jit_address), " addr_b =", hex(ob.jit_address))

print("python  oa(3.0) =", 3.0 * 2.0 + 1.0, "expected")
print("python  ob(3.0) =", 3.0 * 100.0 + 1.0, "expected")


@njit(cache=False)
def const_a(x):
    return oa(x)


@njit(cache=False)
def const_b(x):
    return ob(x)


print("CONST oa ->", const_a(3.0), "(want 7.0)")
print("CONST ob ->", const_b(3.0), "(want 301.0)")


@njit(cache=False)
def arg_call(f, x):
    return f(x)


print("ARG   oa ->", arg_call(oa, 3.0), "(want 7.0)")
print("ARG   ob ->", arg_call(ob, 3.0), "(want 301.0)")
