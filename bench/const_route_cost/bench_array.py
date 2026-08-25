"""Worst case for losing inlining: an elementwise array map, plain and under fastmath."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mods"))

from numba import njit, types  # noqa: E402

import binding_small  # noqa: E402

SMALL = binding_small.small.as_func
F64A = types.float64[::1]

N = int(os.environ.get("ARR_N", "4000000"))
REPS = int(os.environ.get("ARR_REPS", "9"))


@njit(types.void(F64A, F64A), cache=False)
def map_const(a, out):
    for i in range(a.shape[0]):
        out[i] = SMALL(a[i])


@njit(types.void(F64A, F64A), cache=False, fastmath=True)
def map_const_fm(a, out):
    for i in range(a.shape[0]):
        out[i] = SMALL(a[i])


@njit(cache=False)
def map_arg(f, a, out):
    for i in range(a.shape[0]):
        out[i] = f(a[i])


@njit(types.void(F64A, F64A), cache=False, fastmath=True)
def map_inline_fm(a, out):
    for i in range(a.shape[0]):
        out[i] = a[i] * 2.0 + 1.0


@njit(types.void(F64A, F64A), cache=False)
def map_inline(a, out):
    for i in range(a.shape[0]):
        out[i] = a[i] * 2.0 + 1.0


a = np.arange(N, dtype=np.float64)
out = np.empty_like(a)


def bench(label, fn, args):
    fn(*args)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return {"label": label, "ns_per_elem": min(ts) / N * 1e9, "best_s": min(ts)}


rows = [
    bench("map_const", map_const, (a, out)),
    bench("map_const_fastmath", map_const_fm, (a, out)),
    bench("map_arg", map_arg, (SMALL, a, out)),
    bench("map_inline", map_inline, (a, out)),
    bench("map_inline_fastmath", map_inline_fm, (a, out)),
]
print(json.dumps({"tree": os.environ.get("BENCH_TREE", "?"), "n": N, "rows": rows}))
