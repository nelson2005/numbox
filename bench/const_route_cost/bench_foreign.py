"""Cacheability and per-process compile cost of a const caller of a rewrap_derive-upgraded wrapper."""
import json
import os
import pathlib
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mods"))

t0 = time.perf_counter()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import foreign_mod  # noqa: E402
    val = foreign_mod.const_foreign(3.0)
    msgs = sorted({f"{w.category.__name__}: {str(w.message).splitlines()[0]}" for w in caught})
elapsed = time.perf_counter() - t0

cache_dir = pathlib.Path(os.environ["NUMBA_CACHE_DIR"])
fn = foreign_mod.const_foreign
print(json.dumps({
    "tree": os.environ.get("BENCH_TREE", "?"),
    "import_and_call_s": elapsed,
    "value": val,
    "hits": sum(fn.stats.cache_hits.values()),
    "misses": sum(fn.stats.cache_misses.values()),
    "nbi": sum(1 for _ in cache_dir.rglob("foreign_mod.const_foreign-*.nbi")),
    "nbc": sum(1 for _ in cache_dir.rglob("foreign_mod.const_foreign-*.nbc")),
    "warnings": msgs,
}))
