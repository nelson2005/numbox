from numba import float64, njit
from numba.core.types.function_type import CompileResultWAP
from numbox.core.proxy.proxy import proxy
from numbox.utils.highlevel import cres
from numbox.utils.derive_wap import rewrap_derive


def make_flagged(opts):
    def recip(x):
        return 1.0 / x
    return cres(float64(float64), **opts)(recip)


numpy_model = make_flagged({"error_model": "numpy"})
python_model = make_flagged({"error_model": "python"})


@cres(float64(float64))
def dbl(x):
    return x * 2.0


@cres(float64(float64))
def trp(x):
    return x * 3.0


@cres(float64(float64))
def boom(x):
    if x > 0:
        raise ValueError("cres boom")
    return x


@proxy(float64(float64))
def pboom(x):
    if x > 0:
        raise ValueError("proxy boom")
    return x


def _fboom(x):
    if x > 0:
        raise ValueError("foreign boom")
    return x


_fjit = njit(float64(float64), cache=False)(_fboom)
fboom = rewrap_derive(CompileResultWAP(_fjit.get_compile_result(float64(float64))))
pboom_as = pboom.as_func
