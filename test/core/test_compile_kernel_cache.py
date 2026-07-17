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

from numba import vectorize

from numbox.core.variable.compile_kernel import _formula_fingerprint


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
