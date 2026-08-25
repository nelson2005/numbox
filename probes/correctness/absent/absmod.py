import os
from numba import float64
from numbox.core.proxy.proxy import proxy_if_available


class _Lib:
    pass


lib = _Lib()
if os.environ.get("PRESENT", "1") == "1":
    lib.scale = 1


@proxy_if_available(lib, float64(float64))
def scale(x):
    return x * 11.0
