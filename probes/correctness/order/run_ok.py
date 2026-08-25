import os
import warnings
_caught = []


def _show(message, category, filename, lineno, file=None, line=None):
    _caught.append("%s: %s" % (category.__name__, str(message)[:100]))


warnings.simplefilter("always")
warnings.showwarning = _show

from caller_ok import ca, cb
from derives_ok import da, db

out = []
for name, fn, want in (("ca", ca, 6.0), ("cb", cb, 15.0)):
    got = fn(3.0)
    served = "served" if sum(fn.stats.cache_hits.values()) else "compiled"
    out.append("%s=%s(%s,want %s)" % (name, got, served, want))
print("ORDER=%s %s alias_a=%s alias_b=%s N_WARN=%d" % (
    os.environ.get("ORDER", "ab"), " ".join(out),
    getattr(da, "jit_alias", "-")[-8:], getattr(db, "jit_alias", "-")[-8:], len(_caught)))
for w in _caught:
    print("WARN", w)
