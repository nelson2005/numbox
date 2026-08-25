"""Cost of the stale-alias guard on a const-route cache load, and what it sees, on either tree."""
import json
import os
import pathlib
import pickle
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mods"))

from numbox.core.proxy.proxy import _ALIAS_PREFIX, _stale_proxy_aliases, _undefined_symbols  # noqa: E402

from numba.core.types import float64  # noqa: E402

import caller_const  # noqa: E402

SIG = float64(float64)
for name in ("const_small", "const_big", "const_cres", "disp_small", "plain"):
    getattr(caller_const, name).compile(SIG)

cache_dir = pathlib.Path(os.environ["NUMBA_CACHE_DIR"])
REPS = 500
rows = []
for name in ("const_small", "const_big", "const_cres", "disp_small", "plain"):
    paths = sorted(cache_dir.rglob(f"caller_const.{name}-*.nbc"))
    if not paths:
        rows.append({"name": name, "note": "no nbc written"})
        continue
    raw = paths[0].read_bytes()
    payload = pickle.loads(raw)
    _n, kind, data = payload[0]
    obj = data[0]
    has_prefix = _ALIAS_PREFIX.encode() in obj

    def timed(fn):
        fn()
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return min(ts), statistics.median(ts)

    g_min, g_med = timed(lambda: _stale_proxy_aliases(payload, lambda p: p[0]))
    e_min, e_med = timed(lambda: _undefined_symbols(obj)) if has_prefix else (0.0, 0.0)
    undef = sorted(s for s in _undefined_symbols(obj) if s.startswith(_ALIAS_PREFIX))
    rows.append({
        "name": name,
        "nbc_bytes": len(raw),
        "object_bytes": len(obj),
        "kind": kind,
        "carries_alias_prefix": has_prefix,
        "guard_us_min": g_min * 1e6,
        "guard_us_median": g_med * 1e6,
        "elf_reader_us_min": e_min * 1e6,
        "numbox_undefined_symbols": undef,
    })
print(json.dumps({"tree": os.environ.get("BENCH_TREE", "?"), "rows": rows}, indent=1))
