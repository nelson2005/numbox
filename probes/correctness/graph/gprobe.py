import os
import time
import warnings
_caught = []


def _show(message, category, filename, lineno, file=None, line=None):
    _caught.append("%s: %s" % (category.__name__, str(message)[:120]))


warnings.simplefilter("always")
warnings.showwarning = _show

t0 = time.time()
from numbox.core.work.builder import End, Derived, make_graph

K = float(os.environ.get("K", "2.0"))
reg = {}
x = End(name="x", init_value=3.0, registry=reg)


def derive_y(x):
    return x * 2.0


def derive_z(y):
    return y + K


y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x,), registry=reg)
z = Derived(name="z", init_value=0.0, derive=derive_z, sources=(y,), registry=reg)
access = make_graph(z, registry=reg)
access.z.calculate()
print("RESULT", access.z.data, "elapsed %.2f" % (time.time() - t0))
try:
    from numbox.utils.derive_wap import _JIT_ALIASES
    print("N_JIT_ALIASES", len(_JIT_ALIASES))
except Exception as e:
    print("N_JIT_ALIASES n/a", e)
print("N_WARN", len(_caught))
for w in _caught:
    print("WARN", w)
