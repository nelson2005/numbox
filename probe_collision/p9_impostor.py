"""Does the new py_func check hold on a cache-restored compile result?

`_compiled_from` prefers `cres.type_annotation.func_id.func is py_func`. numba replaces
`type_annotation` with a plain string on a result it restored from its own cache, which
is the ordinary state of `cres(sig, cache=True)` in a warm process, and the check then
falls back to module plus qualified name. Run this twice against one cache directory.
"""
import warnings

from numba import njit
from numba.core.types import float64

from numbox.utils.derive_wap import DeriveWAP, _stable_jit_alias
from numbox.utils.highlevel import cres

OPTS = {"cache": True}


def warm_cached_body(x):
    return x * 9.0 + 2.0


def make_impostor():
    code = compile("def warm_cached_body(x):\n    return x * 1000.0\n", "<impostor>", "exec")
    fn = type(warm_cached_body)(code.co_consts[0], {}, "warm_cached_body")
    fn.__qualname__ = warm_cached_body.__qualname__
    fn.__module__ = warm_cached_body.__module__
    return fn


derive = cres(float64(float64), **OPTS)(warm_cached_body)
print(f"  compile result came back from numba's cache: "
      f"{isinstance(derive.cres.type_annotation, str)}")

impostor = make_impostor()
try:
    keyed_wrong = DeriveWAP(derive.cres, py_func=impostor, jit_options=OPTS)
except ValueError as exc:
    print(f"  refused: {exc}")
    raise SystemExit(0)

honest = _stable_jit_alias(warm_cached_body, float64(float64), OPTS)
print(f"  accepted; alias {keyed_wrong.jit_alias}")
print(f"  which is the impostor's alias: "
      f"{keyed_wrong.jit_alias == _stable_jit_alias(impostor, float64(float64), OPTS)}")
print(f"  the real body's alias is        {honest}")

KEYED_WRONG = keyed_wrong

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")

    @njit(float64(float64), cache=True)
    def const_keyed_wrong(x):
        return KEYED_WRONG(x)

    value = const_keyed_wrong(2.0)

dynamic = [str(w.message) for w in caught if "dynamic globals" in str(w.message)]
print(f"  caller returns {value!r} (the real body gives 20.0); cacheable: {not dynamic}")
