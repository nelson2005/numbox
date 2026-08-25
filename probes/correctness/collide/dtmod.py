"""Same-qualname closures capturing different structured numpy dtypes.

type(np.dtype([...])).__name__ is 'dtype[void]' for every structured dtype, so
_fingerprint_function_best_effort collapses them all to <opaque:dtype[void]>.
"""
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
