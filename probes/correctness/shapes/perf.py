"""Cost of referencing the body by extern alias instead of linking it in."""
import time
from numba import njit, float64
from numbox.utils.highlevel import cres


@cres(float64(float64))
def tiny(x):
    return x * 2.0 + 1.0


@njit(cache=False)
def loop(n):
    acc = 0.0
    for i in range(n):
        acc += tiny(acc + 1.0)
    return acc


loop(10)
best = min(_t for _t in (
    (lambda: (lambda t0: (loop(2_000_000), time.perf_counter() - t0)[1])(time.perf_counter()))()
    for _ in range(5)))
print("const-derive loop 2e6 iters: %.4f s" % best)
