"""Does _JIT_ALIASES pin compile results a dropped derive used to release?

Mint K derives with distinct bodies through numbox.utils.highlevel.cres, drop every reference,
collect, and report RSS. Base has no registry; fix keeps one entry per distinct fingerprint.
"""
import gc
import json
import os
import resource
import sys

from numba import types

from numbox.utils.highlevel import cres

K = int(os.environ.get("PIN_K", "40"))
SIG = types.float64(types.float64)


def rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def rss_now_kb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


gc.collect()
base_rss = rss_now_kb()

ns = {}
for i in range(K):
    src = f"def body_{i}(x):\n    return x * {1.0 + i} + {3.0 + i}\n"
    exec(compile(src, f"<pin{i}>", "exec"), ns)
    w = cres(SIG)(ns[f"body_{i}"])
    del w
    del ns[f"body_{i}"]

gc.collect()
after_rss = rss_now_kb()

registry = 0
try:
    from numbox.utils.derive_wap import _JIT_ALIASES
    registry = len(_JIT_ALIASES)
except Exception:
    registry = -1

print(json.dumps({
    "tree": os.environ.get("BENCH_TREE", "?"),
    "k": K,
    "rss_before_kb": base_rss,
    "rss_after_kb": after_rss,
    "rss_delta_kb": after_rss - base_rss,
    "rss_delta_kb_per_derive": (after_rss - base_rss) / K,
    "peak_rss_kb": rss_kb(),
    "registry_entries": registry,
}))
assert sys.argv is not None
