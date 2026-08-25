from numba import njit
from sbind import numpy_model, python_model, dbl, trp, boom, pboom_as, fboom


def report(label, fn, *a):
    try:
        print("%-26s -> %r" % (label, fn(*a)))
    except Exception as e:
        print("%-26s RAISED %s: %s" % (label, type(e).__name__, str(e)[:60]))


@njit(cache=False)
def c_numpy(x):
    return numpy_model(x)


@njit(cache=False)
def c_python(x):
    return python_model(x)


report("flags numpy 1/0", c_numpy, 0.0)
report("flags python 1/0", c_python, 0.0)
print("alias numpy ", getattr(numpy_model, "jit_alias", "-"))
print("alias python", getattr(python_model, "jit_alias", "-"))

TUP = (dbl, trp)


@njit(cache=False)
def tuple_pick(x):
    return TUP[0](x) + 100.0 * TUP[1](x)


report("tuple of derives", tuple_pick, 1.0)


@njit(cache=False)
def const_boom_cres(x):
    return boom(x)


@njit(cache=False)
def const_boom_proxy(x):
    return pboom_as(x)


@njit(cache=False)
def const_boom_foreign(x):
    return fboom(x)


report("CONST cres raise", const_boom_cres, 1.0)
report("CONST proxy raise", const_boom_proxy, 1.0)
report("CONST foreign raise", const_boom_foreign, 1.0)


@njit(cache=False)
def arg_call(f, x):
    return f(x)


report("ARG proxy raise", arg_call, pboom_as, 1.0)
report("ARG cres raise", arg_call, boom, 1.0)
