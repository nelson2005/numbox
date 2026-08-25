"""The PR's motivating shape: a Work node whose derive is a proxied binding's .as_func, built in jitted scope."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mods"))

from numba import njit  # noqa: E402

from numbox.core.work.work import make_work  # noqa: E402

import binding_small  # noqa: E402

SMALL = binding_small.small.as_func
N = int(os.environ.get("WORK_N", "200000"))
REPS = int(os.environ.get("WORK_REPS", "9"))


@njit(cache=False)
def run_const(n):
    acc = 0.0
    for i in range(n):
        src = make_work("src", float(i))
        node = make_work("node", 0.0, sources=(src,), derive=SMALL)
        node.calculate()
        acc += node.data
    return acc


run_const(3)
ts = []
for _ in range(REPS):
    t0 = time.perf_counter()
    v = run_const(N)
    ts.append(time.perf_counter() - t0)
print(json.dumps({
    "tree": os.environ.get("BENCH_TREE", "?"),
    "n": N,
    "ns_per_node": min(ts) / N * 1e9,
    "value": v,
}))
