"""Does minting a cres derive pull numbox.core.proxy.proxy into a process that never asked for it?"""
import json
import os
import sys
import time

t0 = time.perf_counter()
from numbox.utils.highlevel import cres  # noqa: E402
t_import = time.perf_counter() - t0
before = sorted(m for m in sys.modules if m.startswith("numbox"))

from numba import types  # noqa: E402

t1 = time.perf_counter()


@cres(types.float64(types.float64))
def f(x):
    return x * 3.0 + 1.0


t_decorate = time.perf_counter() - t1
after = sorted(m for m in sys.modules if m.startswith("numbox"))
print(json.dumps({
    "tree": os.environ.get("BENCH_TREE", "?"),
    "import_highlevel_s": t_import,
    "decorate_s": t_decorate,
    "numbox_modules_before": len(before),
    "numbox_modules_after": len(after),
    "added_by_decoration": [m for m in after if m not in before],
    "proxy_loaded_before": "numbox.core.proxy.proxy" in before,
    "proxy_loaded_after": "numbox.core.proxy.proxy" in after,
    "total_modules": len(sys.modules),
}))
