"""Lifetime of the constant-lowered entry point when the derive is dropped.

rewrap_derive over a foreign CompileResultWAP mints no alias, so the fixed tree
bakes cres.library.get_pointer_to_function(...) as a raw address with nothing
keeping that library alive. The baseline linked the library into the caller, so
the caller was self-contained.
"""
import ctypes
import gc
import sys
from numba import njit, float64
from numba.core.types.function_type import CompileResultWAP
from numbox.utils.derive_wap import rewrap_derive


def _body(x):
    return x * 7.0


_jit = njit(float64(float64), cache=False)(_body)
foreign = CompileResultWAP(_jit.get_compile_result(float64(float64)))
d = rewrap_derive(foreign)
print("type(d) =", type(d).__name__, "alias =", getattr(d, "jit_alias", "<none>"))


@njit(cache=False)
def caller(x):
    return d(x)


print("before drop:", caller(2.0), flush=True)

lib = d.cres.library
print("refcount(cres.library) before drop:", sys.getrefcount(lib), flush=True)
del lib
del d
del foreign
del _jit
del globals()["_body"]
gc.collect()
gc.collect()
print("dropped and collected", flush=True)

blob = [bytearray(1 << 20) for _ in range(64)]
for b in blob:
    ctypes.memset((ctypes.c_char * len(b)).from_buffer(b), 0xAB, len(b))
print("heap churned", flush=True)

print("after drop:", caller(2.0), flush=True)
print("DONE", flush=True)
