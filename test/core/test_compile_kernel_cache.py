"""compile_kernel cache-correctness regression tests.

Folding the jit flags into the outer ``_kernel_<digest>`` name cannot by itself invalidate a unit that numba caches
independently and that the kernel links against. It also cannot cover the values that the digest fingerprint misses.
Each test checks one such hazard. A test checks the fingerprint when the mechanism is deterministic, and it runs a
probe in two processes that share one ``NUMBA_CACHE_DIR`` when the failure crosses processes.
"""
import os
import subprocess
import sys
import textwrap
import types

import pytest
from numba import njit, typeof, vectorize
from numba.core import config as numba_config
from numba.core.caching import NullCache

from numbox.core.variable.compile_kernel import (
    _compile, _formula_fingerprint, _is_self_cached, _references_self_cached,
)
from numbox.utils.fingerprint import _fingerprint_function


# A module-level cache=True dispatcher has a real locator (this file), so numba enables its FunctionCache. This
# dispatcher is the self-cached formula that the guard must detect.
_cached_formula = njit(cache=True)(lambda y: y + 1.0)
_plain_formula = njit(lambda y: y + 1.0)                     # cache=False, so it is safe
_KERNEL_SRC = "def _kernel(a0):\n    return (f_x(a0),)\n"


# The formula below is a plain formula that calls a module-level cache=True dispatcher, rather than being one. It is
# the transitive case that the top-level check, _is_self_cached, would miss.
def _formula_calls_cached(x):
    return _cached_formula(x)


# The formulas below reach the cached dispatcher only through a default argument or through a container global. The
# digest folds both channels, so the guard must cover them as well.
def _formula_default_arg(x, h=_cached_formula):
    return h(x)


_HELPERS = (_cached_formula,)


def _formula_container_global(x):
    return _HELPERS[0](x)


# The formula below reaches the cached dispatcher as a module attribute (helpers.helper(x)).
_helpers_mod = types.ModuleType("test_ck_helpers_mod")
_helpers_mod.helper = _cached_formula


def _formula_module_attr(x):
    return _helpers_mod.helper(x)


def _run_probe(probe, env):
    r = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert r.returncode == 0, f"probe failed:\n{r.stderr}"
    return r.stdout.strip()


def _shared_cache_env(tmp_path):
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    return env


def _body(x):
    # fastmath's nnan and reassociation flags simplify (x + 1e308) - 1e308 to x instead of 0.0. The body is one line,
    # and its result visibly depends on the flag.
    return (x + 1e308) - 1e308


# DUFunc (@vectorize) formula targetoptions must enter the digest
# ---------------------------------------------------------------------------

def test_dufunc_targetoptions_distinguish_fingerprint():
    """The targetoptions of a @vectorize formula must enter the fingerprint. Otherwise two kernels that differ only in a
    DUFunc flag share one digest, and a cache hit loads a stale binary for the outer cached kernel."""
    f_plain = vectorize(["float64(float64)"], nopython=True)(_body)
    f_fast = vectorize(["float64(float64)"], nopython=True, fastmath=True)(_body)
    fp_plain, cacheable_plain = _formula_fingerprint(f_plain)
    fp_fast, _ = _formula_fingerprint(f_fast)
    assert cacheable_plain is True
    assert fp_plain != fp_fast, "DUFunc fastmath flag absent from fingerprint (digest collision)"


def test_dufunc_fastmath_flip_not_stale_across_processes(tmp_path):
    """Two processes share one cache directory, and the fastmath flag of a @vectorize formula changes between the runs.
    The second process must recompile instead of loading the first process's stale binary. If the targetoptions were
    not folded into the digest, both runs would share one digest."""
    probe = tmp_path / "dufunc_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        import numpy as np
        from numba import vectorize
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        fm = bool(int(os.environ["FASTMATH"]))
        f = vectorize(["float64(float64)"], nopython=True, fastmath=fm)(
            lambda x: (x + 1e308) - 1e308)
        g = Graph(
            variables_lists={"variables": [
                {"name": "out", "inputs": {"y": "basket"}, "formula": f}]},
            external_source_names=["basket"],
        )
        ck = compile_kernel(g, "variables.out")
        print(float(np.asarray(ck.kernel(1.0)[0]).ravel()[0]))
    '''), encoding="utf-8")

    env = _shared_cache_env(tmp_path)
    env["FASTMATH"] = "0"
    assert _run_probe(probe, env) == "0.0"             # cold: caches under its digest
    env["FASTMATH"] = "1"
    got = _run_probe(probe, env)                        # shared cache, flag changed
    assert got == "1.0", f"DUFunc fastmath flip served a stale cached kernel: got {got}, expected 1.0"


# A value read through module indirection (cfg.SCALE) must enter the fingerprint
# ---------------------------------------------------------------------------

def _func_reading_module_attr(scale_value):
    """Build a fresh function ``f(x): return x * cfg.SCALE`` whose globals hold a module ``cfg`` with
    ``SCALE = scale_value``. The function reads the value through one level of module indirection, which a fingerprint
    of the globals alone would not see."""
    cfg = types.ModuleType("cfg_probe_mod")
    cfg.SCALE = scale_value
    code = compile("def f(x):\n    return x * cfg.SCALE\n", "<probe>", "exec").co_consts[0]
    return types.FunctionType(code, {"cfg": cfg})


def test_module_attribute_value_change_rekeys_fingerprint():
    """numba freezes the value of cfg.SCALE into the compiled binary, but the value is one module attribute away from a
    global. The value must therefore enter the fingerprint, or the cached binary is stale after the value changes."""
    assert _fingerprint_function(_func_reading_module_attr(2.0), set()) \
        != _fingerprint_function(_func_reading_module_attr(3.0), set())


def test_module_function_attr_stays_a_stable_reference():
    """A module attribute that is a function or ufunc (np.sqrt) is resolved by identity and is not frozen as data. The
    fingerprint must not fold it, because folding it would recurse into numpy, and the attribute must not make the
    formula uncacheable."""
    import numpy as np

    def mk():
        code = compile("def f(x):\n    return np.sqrt(x) + 1.0\n", "<p>", "exec").co_consts[0]
        return types.FunctionType(code, {"np": np})

    fp = _fingerprint_function(mk(), set())
    assert fp == _fingerprint_function(mk(), set())     # stable
    assert "sqrt=function" not in fp                    # ufunc not folded as a value


def test_module_constant_change_not_stale_across_processes(tmp_path):
    """Two processes share one cache directory, and a formula reads a module constant whose value changes between the
    runs. The second process must recompile instead of returning the value frozen into the first process's binary."""
    probe = tmp_path / "modattr_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        import sys
        import types
        import numpy as np
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        cfg = types.ModuleType("ck_cfg_probe")
        cfg.SCALE = float(os.environ["SCALE"])
        sys.modules["ck_cfg_probe"] = cfg
        import ck_cfg_probe  # noqa: E402,F401  -- now a module global of this script

        def scale(y):
            return y * ck_cfg_probe.SCALE

        g = Graph(
            variables_lists={"variables": [
                {"name": "out", "inputs": {"y": "basket"}, "formula": scale}]},
            external_source_names=["basket"],
        )
        ck = compile_kernel(g, "variables.out")
        print(float(np.asarray(ck.kernel(10.0)[0]).ravel()[0]))
    '''), encoding="utf-8")

    env = _shared_cache_env(tmp_path)
    env["SCALE"] = "2.0"
    assert _run_probe(probe, env) == "20.0"            # cold: caches with SCALE=2
    env["SCALE"] = "3.0"
    got = _run_probe(probe, env)                        # shared cache, constant changed
    assert got == "30.0", f"module constant change served a stale kernel: got {got}, expected 30.0"


# A self-cached formula makes the unit uncacheable (no poisoned artifact)
# ---------------------------------------------------------------------------

def test_is_self_cached_detects_only_cache_true_dispatchers():
    assert _is_self_cached(_cached_formula) is True
    assert _is_self_cached(_plain_formula) is False


def test_self_cached_dispatcher_formula_makes_unit_uncacheable():
    """A user @njit(cache=True) formula has its own numba cache, whose key includes neither the flags nor the globals.
    Folding the flags into the outer name cannot invalidate that cache. A fresh outer kernel would link the stale inner
    binary and serialize it, which would poison the cache permanently. The unit must therefore compile uncached, with a
    warning."""
    assert not isinstance(_cached_formula._cache, NullCache)   # precondition
    with pytest.warns(UserWarning, match="a @njit.cache=True. dispatcher"):
        disp = _compile(_KERNEL_SRC, {"f_x": _cached_formula}, None, None)
    assert isinstance(disp._cache, NullCache), "outer kernel must not be disk-cached"


def test_plain_formula_unit_stays_cacheable():
    """A cache=False formula has no independent on-disk cache and is re-keyed with the outer kernel, whose name folds
    the flags. The unit must therefore still cache, and the guard must not turn caching off as a false positive."""
    disp = _compile(_KERNEL_SRC, {"f_x": _plain_formula}, None, None)
    assert not isinstance(disp._cache, NullCache), "plain-formula kernel should cache"


def test_references_self_cached_detects_a_cached_callee():
    assert _references_self_cached(_formula_calls_cached, set()) is True
    assert _references_self_cached(_plain_formula, set()) is False


def test_formula_calling_cache_true_dispatcher_makes_unit_uncacheable():
    """A plain formula that calls a @njit(cache=True) dispatcher links the callee's binary, whose key includes neither
    the flags nor the globals. The unit must be uncacheable as well. Otherwise a stale inner binary is serialized into
    an artifact under a fresh digest, which poisons the cache one call deeper."""
    with pytest.warns(UserWarning, match="reference, a @njit"):
        disp = _compile(_KERNEL_SRC, {"f_x": njit(_formula_calls_cached)}, None, None)
    assert isinstance(disp._cache, NullCache), "unit linking a cache=True callee must not cache"


def test_formula_calling_cache_true_dispatcher_not_poisoned(tmp_path):
    """The test runs end-to-end. A plain formula calls a cache=True dispatcher whose flags change between processes, and
    the formula must leave no outer cache artifact owned by numbox, so that no artifact can be poisoned. numba still
    loads the user dispatcher's own stale binary in-process. That behavior is an accepted numba limitation, not a numbox
    defect.
    """
    probe = tmp_path / "calls_cached_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        import numpy as np
        from numba import njit
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        helper = njit(cache=True, error_model=os.environ["EM"])(lambda y: 1.0 / y)

        def f(x):
            return helper(x)

        g = Graph(
            variables_lists={"variables": [
                {"name": "out", "inputs": {"y": "basket"}, "formula": f}]},
            external_source_names=["basket"],
        )
        ck = compile_kernel(g, "variables.out")
        try:
            print(float(np.asarray(ck.kernel(0.0)[0]).ravel()[0]))
        except ZeroDivisionError:
            print("ZeroDivisionError")
    '''), encoding="utf-8")

    env = _shared_cache_env(tmp_path)
    env["EM"] = "numpy"
    _run_probe(probe, env)
    env["EM"] = "python"
    _run_probe(probe, env)
    outer = list((tmp_path / "nbcache").rglob("_kernel_*"))
    assert not outer, f"numbox cached an outer kernel despite a cache=True callee (poisoning): {outer}"


def test_references_self_cached_through_default_argument():
    assert _references_self_cached(_formula_default_arg, set()) is True


def test_references_self_cached_through_container_global():
    assert _references_self_cached(_formula_container_global, set()) is True


def test_references_self_cached_through_module_attribute():
    assert _references_self_cached(_formula_module_attr, set()) is True


def test_default_arg_cached_dispatcher_makes_unit_uncacheable():
    with pytest.warns(UserWarning, match="reference, a @njit"):
        disp = _compile(_KERNEL_SRC, {"f_x": njit(_formula_default_arg)}, None, None)
    assert isinstance(disp._cache, NullCache)


def test_dispatcher_typed_declared_sig_makes_unit_uncacheable():
    """A declared signature that carries a Dispatcher type (a function-typed external) makes the unit uncacheable.

    If ``declared_sigs`` folded the raw ``repr`` of each type, a Dispatcher's
    ``type(CPUDispatcher(<function f at 0x...>))`` would carry an ASLR address into the digest and orphan one
    ``_kernel_`` cache pair per run. numba also cannot cache a Dispatcher signature across processes. The declared
    types therefore go through ``_type_identity``, which reports a Dispatcher as uncacheable, so ``_compile`` drops the
    on-disk cache.
    """
    disp = _compile(
        _KERNEL_SRC, {"f_x": _plain_formula}, None, None,
        declared_sigs=((typeof(_plain_formula),),),
    )
    assert isinstance(disp._cache, NullCache), (
        "a Dispatcher-typed declared signature must make the unit uncacheable")


# Chained (cfg.sub.SCALE) and closure-captured module attributes must be folded too
# ---------------------------------------------------------------------------

def _func_reading_chained_module_attr(scale):
    cfg = types.ModuleType("cfg_chain_probe")
    sub = types.ModuleType("cfg_chain_probe.sub")
    cfg.sub = sub
    sub.SCALE = scale
    code = compile("def f(x):\n    return x * cfg.sub.SCALE\n", "<p>", "exec").co_consts[0]
    return types.FunctionType(code, {"cfg": cfg})


def test_chained_module_attribute_rekeys_fingerprint():
    assert _fingerprint_function(_func_reading_chained_module_attr(2.0), set()) \
        != _fingerprint_function(_func_reading_chained_module_attr(3.0), set())


def _closure_module_formula(scale):
    mod = types.ModuleType("cmod_closure_probe")
    mod.SCALE = scale

    def make(m):
        def f(x):
            return x * m.SCALE
        return f
    return make(mod)


def test_closure_captured_module_attribute_rekeys_fingerprint():
    assert _fingerprint_function(_closure_module_formula(2.0), set()) \
        != _fingerprint_function(_closure_module_formula(3.0), set())


def _module_attr_dispatcher_formula(other_val):
    """Return f(x) = cfg.compute(x), where cfg.compute is a Dispatcher that captures the module global OTHER.

    numba links the Dispatcher and freezes OTHER, so the digest must fold the fingerprint of the module-attribute
    dispatcher, as it does for a direct global.
    """
    disp_code = compile("def compute(z):\n    return z + OTHER\n", "<d>", "exec").co_consts[0]
    compute = njit(cache=False)(types.FunctionType(disp_code, {"OTHER": other_val}))
    cfg = types.ModuleType("cfg_disp_fp_probe")
    cfg.compute = compute
    f_code = compile("def f(x):\n    return cfg.compute(x)\n", "<f>", "exec").co_consts[0]
    return types.FunctionType(f_code, {"cfg": cfg})


def test_module_attr_dispatcher_captured_value_rekeys_fingerprint():
    assert _fingerprint_function(_module_attr_dispatcher_formula(10.0), set()) \
        != _fingerprint_function(_module_attr_dispatcher_formula(999.0), set())


def _module_attr_dufunc_formula(other_val):
    """Return f(x) = cfg.uf(x), where cfg.uf is a @vectorize DUFunc that captures OTHER.

    numba links the DUFunc's loop and freezes OTHER, so the digest must fold the DUFunc as well.
    """
    wrapped = types.FunctionType(
        compile("def uf(z):\n    return z + OTHER\n", "<d>", "exec").co_consts[0], {"OTHER": other_val})
    uf = vectorize(["float64(float64)"], nopython=True)(wrapped)
    cfg = types.ModuleType("cfg_uf_fp_probe")
    cfg.uf = uf
    f_code = compile("def f(x):\n    return cfg.uf(x)\n", "<f>", "exec").co_consts[0]
    return types.FunctionType(f_code, {"cfg": cfg})


def test_module_attr_dufunc_captured_value_rekeys_fingerprint():
    assert _fingerprint_function(_module_attr_dufunc_formula(10.0), set()) \
        != _fingerprint_function(_module_attr_dufunc_formula(999.0), set())


# A value that a module exposes only through a PEP 562 __getattr__ is deliberately not folded. If the fingerprint
# resolved referenced names through getattr, a collision with a name in co_names would invoke the __getattr__ of an
# unrelated module, such as numpy. That module would then import submodules and emit deprecation warnings on every
# compile. The limitation is documented, and no test asserts that the lazy-attribute case is covered.


# Result-affecting numba codegen settings beyond BOUNDSCHECK enter the digest
# ---------------------------------------------------------------------------

def test_vectorize_env_knob_enters_digest(monkeypatch):
    src = "def _kernel(a0):\n    return (a0,)\n"
    monkeypatch.setattr(numba_config, "LOOP_VECTORIZE", 0, raising=False)
    name0 = _compile(src, {}, None, None).py_func.__name__
    monkeypatch.setattr(numba_config, "LOOP_VECTORIZE", 1, raising=False)
    name1 = _compile(src, {}, None, None).py_func.__name__
    assert name0 != name1, "LOOP_VECTORIZE absent from the cache digest"


# NUMBA_BOUNDSCHECK is a codegen environment variable outside the jit flags and numba's key
# ---------------------------------------------------------------------------

def test_boundscheck_env_enters_digest(monkeypatch):
    """NUMBA_BOUNDSCHECK overrides even an explicit boundscheck flag at lowering, but it is in neither the jit flags nor
    numba's cache key. It must enter the digest, so that a change of the bounds-check setting does not produce a cache
    hit on the unchecked binary."""
    src = "def _kernel(a0):\n    return (a0,)\n"
    monkeypatch.setattr(numba_config, "BOUNDSCHECK", None)
    name_off = _compile(src, {}, None, None).py_func.__name__
    monkeypatch.setattr(numba_config, "BOUNDSCHECK", 1)
    name_on = _compile(src, {}, None, None).py_func.__name__
    assert name_off != name_on, "BOUNDSCHECK absent from the cache digest"


def test_svml_env_enters_digest(monkeypatch):
    """NUMBA_DISABLE_INTEL_SVML decides whether numba calls Intel's SVML for
    vectorized math, and numba's cache key does not hold it."""
    src = "def _kernel(a0):\n    return (a0,)\n"
    monkeypatch.setattr(numba_config, "DISABLE_INTEL_SVML", 0)
    name_svml = _compile(src, {}, None, None).py_func.__name__
    monkeypatch.setattr(numba_config, "DISABLE_INTEL_SVML", 1)
    name_plain = _compile(src, {}, None, None).py_func.__name__
    assert name_svml != name_plain, "DISABLE_INTEL_SVML absent from the cache digest"


# Un-canonicalizable jit flags must warn, not silently disable caching
# ---------------------------------------------------------------------------

def test_uncanonicalizable_flags_warn_and_disable_cache():
    from numba.core.compiler import Compiler
    src = "def _kernel(a0):\n    return (a0,)\n"
    with pytest.warns(UserWarning, match="no canonical"):
        disp = _compile(src, {}, {"pipeline_class": Compiler}, None)
    assert isinstance(disp._cache, NullCache)


# The declared-return probe memo must key on flags, not skip a flag-dependent probe
# ---------------------------------------------------------------------------

def _declared_locals_graph():
    """Return a graph with a declared float64 node whose formula returns a local `m`.

    The natural return type becomes float32 under ``locals={'m': float32}``, which violates the declared float64. The
    build-time probe catches the violation.
    """
    from numba.core.types import float64
    from numbox.core.variable.variable import Graph, Params

    def f(x):
        m = x
        return m
    g = Graph(variables_lists={"variables": [
        {"name": "a", "inputs": {"x": "e"}, "formula": f, "params": Params(type=float64)},
    ]}, external_source_names=["e"])
    g.external["e"].declare("x", Params(type=float64))
    return g


def test_validated_returns_memo_keys_on_flags():
    """A prior build with the default flags must not let a later build under a flag that affects the return type skip
    the declared-return probe. If the memo key left out the flags, the later build would skip the probe and accept a
    float32 return under a declared float64."""
    from numba.core.types import float32
    from numbox.core.variable.compile_kernel import compile_kernel, _validated_returns

    saved = set(_validated_returns)
    _validated_returns.clear()
    try:
        compile_kernel(_declared_locals_graph(), "variables.a")   # default flags: the build passes and is memoized
        with pytest.raises(ValueError, match="float32"):
            compile_kernel(_declared_locals_graph(), "variables.a",
                           jit_options={"locals": {"m": float32}})
    finally:
        _validated_returns.clear()
        _validated_returns.update(saved)
