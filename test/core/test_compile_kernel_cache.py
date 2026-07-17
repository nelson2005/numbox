"""compile_kernel cache-correctness regression tests.

Folding jit flags into the outer ``_kernel_<digest>`` name cannot, by itself,
invalidate an independently numba-cached unit the kernel links against, nor
cover values the digest fingerprint misses. Each test pins one such hazard --
fingerprint-level where the mechanism is deterministic, and a two-process
shared-``NUMBA_CACHE_DIR`` probe where the failure is cross-process.
"""
import os
import subprocess
import sys
import textwrap
import types

from numba import vectorize

from numbox.core.variable.compile_kernel import _formula_fingerprint
from numbox.utils.fingerprint import _fingerprint_function


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
    # fastmath's nnan/reassociation makes (x + 1e308) - 1e308 collapse to x
    # instead of 0.0 -- a visible, one-line flag-sensitive result.
    return (x + 1e308) - 1e308


# DUFunc (@vectorize) formula targetoptions must enter the digest
# ---------------------------------------------------------------------------

def test_dufunc_targetoptions_distinguish_fingerprint():
    """A @vectorize formula's targetoptions must enter the fingerprint, else two
    kernels differing only in a DUFunc flag share one digest and the outer cached
    kernel is served stale on a cache hit."""
    f_plain = vectorize(["float64(float64)"], nopython=True)(_body)
    f_fast = vectorize(["float64(float64)"], nopython=True, fastmath=True)(_body)
    fp_plain, cacheable_plain = _formula_fingerprint(f_plain)
    fp_fast, _ = _formula_fingerprint(f_fast)
    assert cacheable_plain is True
    assert fp_plain != fp_fast, "DUFunc fastmath flag absent from fingerprint (digest collision)"


def test_dufunc_fastmath_flip_not_stale_across_processes(tmp_path):
    """Two processes sharing one cache dir, a @vectorize formula whose fastmath
    flips between runs: the second process must recompile, not serve the first's
    stale binary. Without the targetoptions fold both runs share one digest."""
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
    got = _run_probe(probe, env)                        # shared cache, flag flipped
    assert got == "1.0", f"DUFunc fastmath flip served a stale cached kernel: got {got}, expected 1.0"


# A value read through module indirection (cfg.SCALE) must enter the fingerprint
# ---------------------------------------------------------------------------

def _func_reading_module_attr(scale_value):
    """Build a fresh function ``f(x): return x * cfg.SCALE`` whose globals hold a
    module ``cfg`` with ``SCALE = scale_value`` -- the one-level indirection that
    was invisible to the fingerprint."""
    cfg = types.ModuleType("cfg_probe_mod")
    cfg.SCALE = scale_value
    code = compile("def f(x):\n    return x * cfg.SCALE\n", "<probe>", "exec").co_consts[0]
    return types.FunctionType(code, {"cfg": cfg})


def test_module_attribute_value_change_rekeys_fingerprint():
    """cfg.SCALE is baked into the compiled binary by numba but is one module hop
    from a global, so its value must enter the fingerprint or a change is stale."""
    assert _fingerprint_function(_func_reading_module_attr(2.0), set()) \
        != _fingerprint_function(_func_reading_module_attr(3.0), set())


def test_module_function_attr_stays_a_stable_reference():
    """A module attribute that is a function/ufunc (np.sqrt) is resolved by
    identity, not frozen as data -- it must not be folded (which would recurse
    into numpy) nor make the formula uncacheable."""
    import numpy as np

    def mk():
        code = compile("def f(x):\n    return np.sqrt(x) + 1.0\n", "<p>", "exec").co_consts[0]
        return types.FunctionType(code, {"np": np})

    fp = _fingerprint_function(mk(), set())
    assert fp == _fingerprint_function(mk(), set())     # stable
    assert "sqrt=function" not in fp                    # ufunc not folded as a value


def test_module_constant_change_not_stale_across_processes(tmp_path):
    """Two processes sharing one cache dir, a formula reading a module constant
    whose value changes between runs: the second must recompile, not serve the
    first's baked-in value."""
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
