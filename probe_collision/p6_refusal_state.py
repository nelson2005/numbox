"""Does the refusal leave anything inconsistent, and can it reach a derive that is fine?

Three colliding bodies, a re-wrap of the winner, the same function through two
producers, and a redefinition in place.
"""
import ctypes

from llvmlite import binding as ll
from numba import cfunc, njit
from numba.core.types import float64

import numbox.utils.derive_wap as dw
from numbox.core.proxy.proxy import proxy
from numbox.utils.derive_wap import DeriveWAP
from numbox.utils.highlevel import cres

PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
fails = []


def check(label, got, want):
    ok = got == want
    if not ok:
        fails.append(f"{label}: got {got!r} want {want!r}")
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}" + ("" if ok else f" (want {want!r})"))


@cfunc(float64(float64))
def c_hundred(x):
    return x * 100.0


@cfunc(float64(float64))
def c_plus_one(x):
    return x + 1.0


@cfunc(float64(float64))
def c_minus_ten(x):
    return x - 10.0


def bind(c_function):
    @cres(float64(float64))
    def body(x):
        return c_function(x) + 3.0
    return body


@njit
def as_arg(f, x):
    return f(x)


print("A. three colliding bodies")
a = bind(PROTO(c_hundred.address))
b = bind(PROTO(c_plus_one.address))
c = bind(PROTO(c_minus_ten.address))
alias = a.jit_alias
check("first publishes", alias is not None, True)
check("second refused", b.jit_alias, None)
check("third refused", c.jit_alias, None)
check("registry holds the first cres", dw._JIT_ALIASES[alias] is a.cres, True)
check("symbol resolves to the first entry point", ll.address_of_symbol(alias), a.jit_address)


@njit(float64(float64))
def const_a(x):
    return a(x)


@njit(float64(float64))
def const_b(x):
    return b(x)


@njit(float64(float64))
def const_c(x):
    return c(x)


check("const a", const_a(4.0), 403.0)
check("const b", const_b(4.0), 8.0)
check("const c", const_c(4.0), -3.0)
check("arg a", as_arg(a, 4.0), 403.0)
check("arg b", as_arg(b, 4.0), 8.0)
check("arg c", as_arg(c, 4.0), -3.0)

print("B. re-wrapping the winner is idempotent")
again = DeriveWAP(a.cres, py_func=a.cres.type_annotation.func_id.func, jit_options={})
check("same alias handed back", again.jit_alias, alias)
check("registry unchanged", dw._JIT_ALIASES[alias] is a.cres, True)
check("symbol unchanged", ll.address_of_symbol(alias), a.jit_address)


@njit(float64(float64))
def const_again(x):
    return again(x)


check("const through the re-wrap", const_again(4.0), 403.0)

print("C. the same function compiled twice, identical py_func object")


def twice_body(x):
    return x * 5.0 + 1.0


first_twice = cres(float64(float64))(twice_body)
second_twice = cres(float64(float64))(twice_body)
check("first publishes", first_twice.jit_alias is not None, True)
check("second refused though the body is literally the same function",
      second_twice.jit_alias, None)


@njit(float64(float64))
def const_first_twice(x):
    return first_twice(x)


@njit(float64(float64))
def const_second_twice(x):
    return second_twice(x)


check("const first", const_first_twice(2.0), 11.0)
check("const second", const_second_twice(2.0), 11.0)

print("D. one function through two producers, @proxy then cres")


def two_producer_body(x):
    return x * 7.0


proxied = proxy(float64(float64))(two_producer_body)
via_cres = cres(float64(float64))(two_producer_body)
check("proxy's as_func published", proxied.as_func.jit_alias is not None, True)
check("the cres over the same function is refused", via_cres.jit_alias, None)

AS_FUNC = proxied.as_func


@njit(float64(float64))
def const_proxied(x):
    return AS_FUNC(x)


@njit(float64(float64))
def const_via_cres(x):
    return via_cres(x)


check("const through @proxy.as_func", const_proxied(3.0), 21.0)
check("const through cres", const_via_cres(3.0), 21.0)
check("arg through cres", as_arg(via_cres, 3.0), 21.0)

print("E. redefined in place, differing only in a value the walker cannot see")
redef_ptr = PROTO(c_hundred.address)


def redef_body(x):
    return redef_ptr(x) + 3.0


redef_first = cres(float64(float64))(redef_body)
redef_ptr = PROTO(c_plus_one.address)
redef_second = cres(float64(float64))(redef_body)
check("first publishes", redef_first.jit_alias is not None, True)
check("second refused", redef_second.jit_alias, None)


@njit(float64(float64))
def const_redef_first(x):
    return redef_first(x)


@njit(float64(float64))
def const_redef_second(x):
    return redef_second(x)


check("const before the rebind", const_redef_first(4.0), 403.0)
check("const after the rebind", const_redef_second(4.0), 8.0)
check("arg after the rebind", as_arg(redef_second, 4.0), 8.0)

print()
print("FAILURES:", len(fails))
for f in fails:
    print("  ", f)
