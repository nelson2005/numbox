from numba import float64
from numbox.core.proxy.proxy import proxy
from numbox.utils.highlevel import cres

# padding line 5
# padding line 6
# padding line 7
# padding line 8
# padding line 9


@proxy(float64(float64))
def hp(x):
    return x * 2.0


@cres(float64(float64))
def hc(x):
    return x * 2.0


@proxy(float64(float64))
def stable(x):
    return x + 1.0


hpd = hp.as_func
