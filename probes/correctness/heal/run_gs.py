import os
import warnings
_caught = []


def _show(message, category, filename, lineno, file=None, line=None):
    _caught.append("%s: %s" % (category.__name__, str(message)[:110]))


warnings.simplefilter("always")
warnings.showwarning = _show

from gscaller import const_scaled
from gsmod import scaled

got = const_scaled(1.0)
st = const_scaled.stats
served = "served" if sum(st.cache_hits.values()) else "compiled"
print("G=%s const_scaled(1.0)=%s want=%s [%s] pycall=%s" % (
    os.environ.get("G", "2.0"), got, os.environ.get("G", "2.0"), served, scaled(1.0)))
print("N_WARN", len(_caught))
for w in _caught:
    print("WARN", w)
