"""Is the publish/refuse read-modify-write safe under threads?

`_publish_jit_alias` reads `_JIT_ALIASES.get(alias)` and, on a miss, assigns and calls
`ll.add_symbol`. Nothing serialises the three steps, and `DeriveWAP` is public.

The compile results are made once, serially. Each round re-arms the race by dropping the
alias from the registry, which is the state a fresh process starts in, and then has every
thread construct its `DeriveWAP` at the same moment. A round with more than one winner is
two different bodies holding one alias, which is what the fix exists to prevent. When one
turns up, the two winners are const-lowered and asked for their own answer.
"""
import ctypes
import sys
import threading

from llvmlite import binding as ll
from numba import cfunc, njit
from numba.core.types import float64

import numbox.utils.derive_wap as dw
from numbox.utils.derive_wap import DeriveWAP
from numbox.utils.highlevel import cres

PROTO = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
ROUNDS = 3000
WIDTH = 4
WANT = {0: 403.0, 1: 8.0, 2: -3.0, 3: 11.0}


@cfunc(float64(float64))
def c0(x):
    return x * 100.0


@cfunc(float64(float64))
def c1(x):
    return x + 1.0


@cfunc(float64(float64))
def c2(x):
    return x - 10.0


@cfunc(float64(float64))
def c3(x):
    return x * 2.0


def bind(c_function):
    @cres(float64(float64))
    def body(x):
        return c_function(x) + 3.0
    return body


derives = [bind(PROTO(c.address)) for c in (c0, c1, c2, c3)]
bodies = [d.cres.type_annotation.func_id.func for d in derives]
alias = derives[0].jit_alias
assert alias is not None
assert all(d.jit_alias is None for d in derives[1:]), "the four bodies do not collide"

sys.setswitchinterval(1e-6)
barrier = threading.Barrier(WIDTH)
raced = 0
first_race = None

for rnd in range(ROUNDS):
    dw._JIT_ALIASES.pop(alias, None)
    won = []
    lock = threading.Lock()

    def run(i):
        barrier.wait()
        wrapper = DeriveWAP(derives[i].cres, py_func=bodies[i], jit_options={})
        if wrapper.jit_alias is not None:
            with lock:
                won.append((i, wrapper))

    threads = [threading.Thread(target=run, args=(i,)) for i in range(WIDTH)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if len(won) > 1:
        raced += 1
        if first_race is None:
            first_race = (rnd, sorted(won, key=lambda p: p[0]), ll.address_of_symbol(alias))

print(f"rounds={ROUNDS} width={WIDTH}")
print(f"rounds where more than one body was handed the alias: {raced}")
if first_race is None:
    print("  no round produced two winners")
    raise SystemExit(0)

rnd, won, resolved = first_race
print(f"first at round {rnd}: winners {[i for i, _ in won]}")
for i, wrapper in won:
    print(f"  body {i} was handed the alias for entry point {hex(wrapper.jit_address)}")
print(f"  the alias resolves to {hex(resolved or 0)}")

lo, hi = won[0], won[1]
LO, HI = lo[1], hi[1]


@njit(float64(float64))
def const_lo(x):
    return LO(x)


@njit(float64(float64))
def const_hi(x):
    return HI(x)


@njit
def as_arg(f, x):
    return f(x)


for label, fn, wrapper, idx in (("lo", const_lo, LO, lo[0]), ("hi", const_hi, HI, hi[0])):
    got_const = fn(4.0)
    got_arg = as_arg(wrapper, 4.0)
    want = WANT[idx]
    print(f"  body {idx}: const={got_const!r} arg={got_arg!r} want={want!r} "
          f"{'OK' if got_const == want else 'WRONG ON THE CONSTANT ROUTE'}")
