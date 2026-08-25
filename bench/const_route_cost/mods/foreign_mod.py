"""A foreign CompileResultWAP upgraded by rewrap_derive, and a const caller of it."""
from numba import njit
from numba.core.types import float64
from numba.core.types.function_type import CompileResultWAP

from numbox.utils.derive_wap import rewrap_derive

SIG = float64(float64)


@njit(SIG, cache=True)
def foreign_body(x):
    return x * 4.0 + 7.0


FOREIGN = CompileResultWAP(foreign_body.get_compile_result(SIG))
UPGRADED = rewrap_derive(FOREIGN)


@njit(SIG, cache=True)
def const_foreign(x):
    return UPGRADED(x) + 1.0
