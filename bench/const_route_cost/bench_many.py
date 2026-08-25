"""Cold compile time and cache size for a caller reaching twenty bindings as constants."""
import json
import os
import pathlib
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mods"))

import numbox.core.bindings.libm  # noqa: E402,F401

t0 = time.perf_counter()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import many_const  # noqa: E402
    v = many_const.many(0.5)
    msgs = sorted({f"{w.category.__name__}: {str(w.message).splitlines()[0]}" for w in caught})
elapsed = time.perf_counter() - t0
cache_dir = pathlib.Path(os.environ["NUMBA_CACHE_DIR"])
fn = many_const.many
print(json.dumps({
    "tree": os.environ.get("BENCH_TREE", "?"),
    "phase": sys.argv[1],
    "compile_or_load_s": elapsed,
    "value": v,
    "hits": sum(fn.stats.cache_hits.values()),
    "misses": sum(fn.stats.cache_misses.values()),
    "nbc_bytes": sorted(p.stat().st_size for p in cache_dir.rglob("many_const.many-*.nbc")),
    "warnings": msgs,
}))
