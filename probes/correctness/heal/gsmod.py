import os
from numba import float64
from numbox.utils.highlevel import cres

G = float(os.environ.get("G", "2.0"))


@cres(float64(float64), cache=True)
def scaled(x):
    return x * G
