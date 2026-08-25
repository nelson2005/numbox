import gc
import os
import resource
import weakref
from numba import float64
from numbox.utils.highlevel import cres

N = int(os.environ.get("N", "40"))


def make(k):
    src = "def body(x):\n    return x * %r\n" % (float(k),)
    ns = {}
    exec(compile(src, "<leakprobe>", "exec"), ns)
    f = ns["body"]
    f.__module__ = "leakprobe"
    f.__qualname__ = "body"
    return cres(float64(float64), cache=False)(f)


refs = []
for i in range(N):
    d = make(i + 1)
    refs.append(weakref.ref(d.cres.library))
    del d
gc.collect()
alive = sum(1 for r in refs if r() is not None)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
try:
    from numbox.utils.derive_wap import _JIT_ALIASES
    n_alias = len(_JIT_ALIASES)
except Exception:
    n_alias = -1
print("N=%d libraries_still_alive_after_gc=%d n_jit_aliases=%d maxrss_kb=%d" % (N, alive, n_alias, rss))
