import os
import warnings
_caught = []


def _show(message, category, filename, lineno, file=None, line=None):
    _caught.append("%s: %s" % (category.__name__, str(message)[:130]))


warnings.simplefilter("always")
warnings.showwarning = _show

print("PRESENT=%s CALLER_IMPORTING" % os.environ.get("PRESENT", "1"), flush=True)
from abscaller import const_scale, disp_scale
print("CALLER_IMPORTED", flush=True)

for name, fn in (("const_scale", const_scale), ("disp_scale", disp_scale)):
    try:
        r = fn(1.0)
        st = fn.stats
        served = "served" if sum(st.cache_hits.values()) else "compiled"
        print("%-12s RESULT=%s %s" % (name, r, served), flush=True)
    except Exception as e:
        print("%-12s RAISED %s" % (name, type(e).__name__), flush=True)
print("N_WARN", len(_caught))
for w in _caught:
    print("WARN", w)
