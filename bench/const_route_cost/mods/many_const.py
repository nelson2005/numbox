"""A caller that reaches twenty shipped libm bindings as compile-time constants."""
from numba import njit
from numba.core.types import float64

import numbox.core.bindings.libm as L

NAMES = ("cos", "sin", "tan", "acos", "asin", "atan", "cosh", "sinh", "tanh",
         "exp", "log", "log10", "sqrt", "cbrt", "ceil", "floor", "fabs",
         "expm1", "log1p", "trunc")
F = tuple(getattr(L, n).as_func for n in NAMES)
(F0, F1, F2, F3, F4, F5, F6, F7, F8, F9,
 F10, F11, F12, F13, F14, F15, F16, F17, F18, F19) = F


@njit(float64(float64), cache=True)
def many(x):
    return (F0(x) + F1(x) + F2(x) + F3(x) + F4(x) + F5(x) + F6(x) + F7(x) + F8(x) + F9(x)
            + F10(x) + F11(x) + F12(x) + F13(x) + F14(x) + F15(x) + F16(x) + F17(x)
            + F18(x) + F19(x))
