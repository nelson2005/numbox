"""Compile time and cache-file size for a const-referencing caller, cold and warm.

Run with a cache dir this process owns. Pass "cold" or "warm" as argv[1].
"""
import json
import os
import pathlib
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mods"))

from numba.core.types import float64  # noqa: E402

import caller_const  # noqa: E402

PHASE = sys.argv[1]
SIG = float64(float64)
NAMES = ("const_small", "const_big", "const_cres", "disp_small", "plain")

out = {"phase": PHASE, "tree": os.environ.get("BENCH_TREE", "?"), "compile_s": {}, "warnings": []}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for name in NAMES:
        fn = getattr(caller_const, name)
        t0 = time.perf_counter()
        fn.compile(SIG)
        out["compile_s"][name] = time.perf_counter() - t0
    out["warnings"] = sorted({f"{w.category.__name__}: {str(w.message).splitlines()[0]}" for w in caught})

cache_dir = pathlib.Path(os.environ["NUMBA_CACHE_DIR"])
out["files"] = {}
for name in NAMES:
    fn = getattr(caller_const, name)
    out["files"][name] = {
        kind: sorted((p.name, p.stat().st_size)
                     for p in cache_dir.rglob(f"caller_const.{name}-*.{kind}"))
        for kind in ("nbi", "nbc")
    }
    out.setdefault("stats", {})[name] = {
        "hits": sum(fn.stats.cache_hits.values()),
        "misses": sum(fn.stats.cache_misses.values()),
    }
out["binding_files"] = sorted(
    (p.name, p.stat().st_size) for p in cache_dir.rglob("*.nbc")
    if not p.name.startswith("caller_const.")
)
print(json.dumps(out))
