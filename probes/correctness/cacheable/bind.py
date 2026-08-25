import ctypes
import ctypes.util
from numba import float64, njit
from numba.core.types.function_type import CompileResultWAP
from numbox.core.proxy.proxy import proxy
from numbox.utils.highlevel import cres
from numbox.utils.derive_wap import rewrap_derive

libm = ctypes.CDLL(ctypes.util.find_library("m"), use_errno=True)


@proxy(float64(float64))
def pbody(x):
    return x * 2.0


@cres(float64(float64))
def cbody(x):
    return x * 3.0


def _foreign_body(x):
    return x * 5.0


_foreign_jit = njit(float64(float64), cache=False)(_foreign_body)
foreign = CompileResultWAP(_foreign_jit.get_compile_result(float64(float64)))
fderive = rewrap_derive(foreign)

pderive = pbody.as_func
