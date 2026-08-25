"""Address-level evidence that the second @proxy decoration rebinds the first's alias."""
import ctypes

from llvmlite import binding as ll
from numba import cfunc, njit
from numba.core.types import float64

from numbox.core.proxy.proxy import _stable_cfunc_alias, proxy

PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)


@cfunc(float64(float64))
def c_times_hundred(x):
    return x * 100.0


@cfunc(float64(float64))
def c_plus_one(x):
    return x + 1.0


def make_binding(c_function):
    @proxy(float64(float64))
    def bound(x):
        return c_function(x) + 3.0
    return bound


def body_of(dispatcher):
    return dispatcher.as_func.cres.type_annotation.func_id.func


def cfunc_addr(dispatcher):
    cres = dispatcher.as_func.cres
    return cres.library.get_pointer_to_function(cres.fndesc.llvm_cfunc_wrapper_name)


hundred = make_binding(PROTO(c_times_hundred.address))
alias = _stable_cfunc_alias(body_of(hundred), float64(float64))
addr_after_first = ll.address_of_symbol(alias)

plus_one = make_binding(PROTO(c_plus_one.address))
alias2 = _stable_cfunc_alias(body_of(plus_one), float64(float64))
addr_after_second = ll.address_of_symbol(alias)

print("alias(hundred body) :", alias)
print("alias(plus_one body):", alias2)
print("one alias for both bodies:", alias == alias2)
print("alias resolves after first decoration :", hex(addr_after_first or 0))
print("alias resolves after second decoration:", hex(addr_after_second or 0))
print("hundred  cfunc wrapper addr:", hex(cfunc_addr(hundred)))
print("plus_one cfunc wrapper addr:", hex(cfunc_addr(plus_one)))


@njit(float64(float64))
def dispatch_hundred(x):
    return hundred(x)


@njit(float64(float64))
def dispatch_plus_one(x):
    return plus_one(x)


print(f"hundred(4.0) from python  = {hundred(4.0)!r}  want 403.0")
print(f"dispatch_hundred(4.0)     = {dispatch_hundred(4.0)!r}  want 403.0")
print(f"dispatch_plus_one(4.0)    = {dispatch_plus_one(4.0)!r}  want 8.0")
