"""Is the extern declared by the fix the same LLVM type declare_function built?

The fix replaces ``context.declare_function(module, fndesc)`` with
``cgutils.get_or_insert_function(module, context.call_conv.get_function_type(restype, argtypes), alias)``.
A mismatch is an ABI break that no test would notice until the wrong bytes came back.
"""
import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.core.registry import cpu_target

cpu_target.target_context.refresh()
ctx = cpu_target.target_context

f8 = types.float64
i8 = types.int64


def body_scalar(x):
    return x * 2.0


def body_void(x):
    pass


def body_tuple(x):
    return (x, x * 2.0)


def body_array(a):
    return a.sum()


def body_unicode(x):
    return "abc" if x > 0 else "de"


def body_two(x, y):
    return x + y


cases = [
    ("float64(float64)", f8(f8), body_scalar),
    ("void(float64)", types.void(f8), body_void),
    ("UniTuple(float64 x 2)(float64)", types.UniTuple(f8, 2)(f8), body_tuple),
    ("float64(float64[:])", f8(types.float64[:]), body_array),
    ("unicode_type(float64)", types.unicode_type(f8), body_unicode),
    ("float64(float64, int64)", f8(f8, i8), body_two),
]

for label, sig, fn in cases:
    d = njit(sig)(fn)
    cres = d.get_compile_result(d.nopython_signatures[0])
    fndesc = cres.fndesc
    mod_a = ir.Module(name="a")
    mod_b = ir.Module(name="b")
    declared = ctx.declare_function(mod_a, fndesc)
    fnty = ctx.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
    aliased = cgutils.get_or_insert_function(mod_b, fnty, "numbox_pxy_cc_probe")
    same = str(declared.type) == str(aliased.type)
    print(f"{label:34s} match={same}")
    if not same:
        print(f"   declare_function : {declared.type}")
        print(f"   call_conv        : {aliased.type}")
    print(f"   attrs declared={sorted(str(a) for a in declared.attributes)} "
          f"aliased={sorted(str(a) for a in aliased.attributes)}")
    print(f"   linkage declared={declared.linkage!r} aliased={aliased.linkage!r} "
          f"cconv declared={declared.calling_convention!r} aliased={aliased.calling_convention!r}")
