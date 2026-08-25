import glob
import os
import sys
import warnings

_caught = []
_orig = warnings.showwarning


def _show(message, category, filename, lineno, file=None, line=None):
    _caught.append("%s: %s" % (category.__name__, str(message)[:160]))


warnings.simplefilter("always")
warnings.showwarning = _show

from callers import const_proxy, const_cres, const_foreign

for name, fn, arg, want in (("const_proxy", const_proxy, 4.0, 8.0),
                            ("const_cres", const_cres, 4.0, 12.0),
                            ("const_foreign", const_foreign, 4.0, 20.0)):
    got = fn(arg)
    st = fn.stats
    hits = sum(st.cache_hits.values())
    misses = sum(st.cache_misses.values())
    verdict = "served" if hits else ("compiled" if misses else "?")
    print("%-14s %-8s result=%s want=%s hits=%d misses=%d" % (name, verdict, got, want, hits, misses))

cdir = os.environ["NUMBA_CACHE_DIR"]
nbi = len(glob.glob(os.path.join(cdir, "**", "*.nbi"), recursive=True))
nbc = len(glob.glob(os.path.join(cdir, "**", "*.nbc"), recursive=True))
print("cachefiles nbi=%d nbc=%d" % (nbi, nbc))
for w in _caught:
    print("WARN", w)
