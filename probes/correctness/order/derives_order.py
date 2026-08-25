"""Two colliding derives; ORDER env var flips which one is minted first."""
import os
import numpy as np
from numba import float64
from numbox.utils.highlevel import cres

DT1 = np.dtype([("a", "f8"), ("b", "f8")])
DT2 = np.dtype([("a", "f8"), ("b", "f8"), ("c", "f8")])


def make(dt):
    @cres(float64(float64))
    def outer(x):
        rec = np.zeros(1, dtype=dt)
        return x + rec.itemsize
    return outer


if os.environ.get("ORDER", "12") == "12":
    o1 = make(DT1)
    o2 = make(DT2)
else:
    o2 = make(DT2)
    o1 = make(DT1)
