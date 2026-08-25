"""The representative shipped-binding shape: a real libm proxy reached as a compile-time constant."""
import json
import os
import time

from numba import njit, types

from numbox.core.bindings.libm import cos, sqrt

COS = cos.as_func
SQRT = sqrt.as_func

N = int(os.environ.get("BENCH_N", "10000000"))
REPS = int(os.environ.get("BENCH_REPS", "9"))
F64 = types.float64
I64 = types.int64


@njit(F64(I64, F64), cache=False)
def const_cos(n, x):
    acc = 0.0
    for i in range(n):
        acc += COS(x + i)
    return acc


@njit(F64(I64, F64), cache=False)
def const_sqrt(n, x):
    acc = 0.0
    for i in range(n):
        acc += SQRT(x + i)
    return acc


@njit(F64(I64, F64), cache=False)
def disp_cos(n, x):
    acc = 0.0
    for i in range(n):
        acc += cos(x + i)
    return acc


@njit(cache=False)
def arg_cos(f, n, x):
    acc = 0.0
    for i in range(n):
        acc += f(x + i)
    return acc


def bench(label, fn, args):
    fn(*args)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return {"label": label, "ns_per_call": min(ts) / N * 1e9}


rows = [
    bench("const_cos", const_cos, (N, 1.0)),
    bench("const_sqrt", const_sqrt, (N, 1.0)),
    bench("disp_cos", disp_cos, (N, 1.0)),
    bench("arg_cos", arg_cos, (COS, N, 1.0)),
]
print(json.dumps({"tree": os.environ.get("BENCH_TREE", "?"), "n": N, "rows": rows}))
