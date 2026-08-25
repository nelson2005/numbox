import warnings
_caught = []


def _show(message, category, filename, lineno, file=None, line=None):
    _caught.append("%s: %s" % (category.__name__, str(message)[:200]))


warnings.simplefilter("always")
warnings.showwarning = _show

from hcaller import cp, cc, cstable

for name, fn in (("cp", cp), ("cc", cc), ("cstable", cstable)):
    got = fn(5.0)
    st = fn.stats
    hits = sum(st.cache_hits.values())
    misses = sum(st.cache_misses.values())
    print("%-8s result=%s %s" % (name, got, "served" if hits else "compiled"))
print("N_WARN=%d" % len(_caught))
for w in _caught:
    print("WARN", w)
