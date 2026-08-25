"""Runtime call overhead through .as_func: constant route vs argument route, small body vs big body.

Each measurement is min-of-REPS wall time over a jitted loop of N calls, reported as ns per call.
Compilation is forced before timing so no compile time leaks into a measurement.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mods"))

from numba import njit, types  # noqa: E402

import binding_big  # noqa: E402
import binding_small  # noqa: E402
import cres_small  # noqa: E402

N = int(os.environ.get("BENCH_N", "3000000"))
REPS = int(os.environ.get("BENCH_REPS", "9"))

SMALL = binding_small.small.as_func
BIG = binding_big.big.as_func
CSMALL = cres_small.csmall

F64 = types.float64
I64 = types.int64


@njit(F64(I64, F64), cache=False)
def const_small_sum(n, x):
    acc = 0.0
    for i in range(n):
        acc += SMALL(x + i)
    return acc


@njit(F64(I64, F64), cache=False)
def const_small_chain(n, x):
    acc = x
    for _ in range(n):
        acc = SMALL(acc) * 0.5 - 0.5
    return acc


@njit(F64(I64, F64), cache=False)
def const_big_sum(n, x):
    acc = 0.0
    for i in range(n):
        acc += BIG(x + i)
    return acc


@njit(F64(I64, F64), cache=False)
def const_cres_sum(n, x):
    acc = 0.0
    for i in range(n):
        acc += CSMALL(x + i)
    return acc


@njit(cache=False)
def arg_sum(f, n, x):
    acc = 0.0
    for i in range(n):
        acc += f(x + i)
    return acc


@njit(cache=False)
def arg_chain(f, n, x):
    acc = x
    for _ in range(n):
        acc = f(acc) * 0.5 - 0.5
    return acc


@njit(F64(I64, F64), cache=False)
def baseline_sum(n, x):
    """Same loop with the body written inline. Floor for the sum shape."""
    acc = 0.0
    for i in range(n):
        acc += (x + i) * 2.0 + 1.0
    return acc


def bench(label, fn, args, n):
    fn(*args)
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        val = fn(*args)
        times.append(time.perf_counter() - t0)
    best = min(times)
    return {
        "label": label,
        "ns_per_call": best / n * 1e9,
        "best_s": best,
        "median_s": sorted(times)[len(times) // 2],
        "worst_s": max(times),
        "value": val,
        "n": n,
    }


NBIG = max(1, N // 10)
rows = [
    bench("const_small_sum", const_small_sum, (N, 1.0), N),
    bench("const_small_chain", const_small_chain, (N, 1.0), N),
    bench("const_cres_sum", const_cres_sum, (N, 1.0), N),
    bench("arg_small_sum", arg_sum, (SMALL, N, 1.0), N),
    bench("arg_small_chain", arg_chain, (SMALL, N, 1.0), N),
    bench("inline_baseline_sum", baseline_sum, (N, 1.0), N),
    bench("const_big_sum", const_big_sum, (NBIG, 1.0), NBIG),
    bench("arg_big_sum", arg_sum, (BIG, NBIG, 1.0), NBIG),
]

asm = const_small_sum.inspect_asm()
asm_txt = next(iter(asm.values()))
llvm = const_small_sum.inspect_llvm()
llvm_txt = next(iter(llvm.values()))

meta = {
    "tree": os.environ.get("BENCH_TREE", "?"),
    "n": N,
    "reps": REPS,
    "const_small_sum_asm_calls": asm_txt.count("\tcall"),
    "const_small_sum_asm_lines": len(asm_txt.splitlines()),
    "const_small_sum_llvm_has_alias": "numbox_pxy_cc_" in llvm_txt,
    "const_small_sum_llvm_lines": len(llvm_txt.splitlines()),
}
print(json.dumps({"meta": meta, "rows": rows}))
