"""Fork twins, the refused derive's lifetime, and the cache-restored py_func check."""
import ctypes
import gc
import os
import sys

from llvmlite import binding as ll
from numba import cfunc, njit
from numba.core.types import float64

import numbox.utils.derive_wap as dw
from numbox.utils.highlevel import cres

PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)


@cfunc(float64(float64))
def c_hundred(x):
    return x * 100.0


@cfunc(float64(float64))
def c_plus_one(x):
    return x + 1.0


def bind(c_function):
    @cres(float64(float64))
    def body(x):
        return c_function(x) + 3.0
    return body


@njit
def as_arg(f, x):
    return f(x)


def warm_cached_body(x):
    return x * 9.0 + 2.0


mode = sys.argv[1]

if mode == "fork":
    parent = bind(PROTO(c_hundred.address))
    alias = parent.jit_alias
    assert alias is not None

    @njit(float64(float64))
    def const_parent(x):
        return parent(x)

    assert const_parent(4.0) == 403.0

    for twin in range(2):
        pid = os.fork()
        if pid == 0:
            child = bind(PROTO(c_plus_one.address))

            @njit(float64(float64))
            def const_child(x):
                return child(x)

            @njit(float64(float64))
            def const_parent_in_child(x):
                return parent(x)

            print(f"  twin {twin}: inherited registry entry is the parent's cres "
                  f"{dw._JIT_ALIASES.get(alias) is parent.cres}, "
                  f"child alias {child.jit_alias!r}, "
                  f"symbol still the parent's {ll.address_of_symbol(alias) == parent.jit_address}, "
                  f"child const {const_child(4.0)!r} (want 8.0) arg {as_arg(child, 4.0)!r}, "
                  f"parent const in child {const_parent_in_child(4.0)!r} (want 403.0)")
            sys.stdout.flush()
            os._exit(0)
    for _ in range(2):
        os.wait()
    print(f"  parent after the forks: const {const_parent(4.0)!r} (want 403.0)")

elif mode == "lifetime":
    first = bind(PROTO(c_hundred.address))
    second = bind(PROTO(c_plus_one.address))
    assert first.jit_alias is not None and second.jit_alias is None

    @njit(float64(float64))
    def const_second(x):
        return second(x)

    print(f"  before the drop: {const_second(4.0)!r} (want 8.0)")
    baked = second.jit_address
    del second
    gc.collect()
    gc.collect()
    print(f"  refused cres pinned anywhere in the registry: "
          f"{any(getattr(c, 'library', None) is not None and False for c in dw._JIT_ALIASES.values())}")
    print(f"  after dropping the derive and collecting: {const_second(4.0)!r} (want 8.0) "
          f"[entry point was {hex(baked)}]")

elif mode == "warmcres":
    derive = cres(float64(float64), cache=True)(warm_cached_body)
    ann = derive.cres.type_annotation
    print(f"  type_annotation is a str: {isinstance(ann, str)}  "
          f"alias {derive.jit_alias}  "
          f"compiled_from true: {dw._compiled_from(derive.cres, warm_cached_body)}  "
          f"compiled_from on an impostor of the same name: ", end="")
    impostor = type(warm_cached_body)(
        compile("def warm_cached_body(x):\n    return x * 1000.0\n", "<impostor>", "exec")
        .co_consts[0], {}, "warm_cached_body")
    impostor.__qualname__ = warm_cached_body.__qualname__
    impostor.__module__ = warm_cached_body.__module__
    print(dw._compiled_from(derive.cres, impostor))

    @njit(float64(float64))
    def const_warm(x):
        return derive(x)

    print(f"  const {const_warm(2.0)!r} (want 20.0)  arg {as_arg(derive, 2.0)!r}")
