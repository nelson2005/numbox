"""Cross-process cache-correctness regression tests.

Each test runs a probe in two separate interpreter processes against one shared
``NUMBA_CACHE_DIR``, changing an input the numba cache key does not cover between
runs, and asserts the second process is not served a stale cached result. These
guard the caching/hashing fixes from the 2026-07-09 review (fork issue #73).
"""
import os
import subprocess
import sys
import textwrap


def _run_probe(probe, env):
    r = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert r.returncode == 0, f"probe failed:\n{r.stderr}"
    return r.stdout.strip()


def test_make_graph_derive_reading_global_is_not_stale_across_processes(tmp_path):
    """A make_graph derive that reads a module global must not serve a stale
    cached result when the global changes in a later process (issue #73 H3).

    The derive compiles as a standalone dispatcher whose numba cache key covers
    only co_code + closure -- blind to the global -- so before the fix a second
    process with a changed global was silently served the first's baked-in value
    (2.0 instead of 3.0). State-reading derives are now compiled uncached.
    """
    probe = tmp_path / "graph_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        from numbox.core.work.builder import End, Derived, make_graph

        G = float(os.environ["G_VAL"])
        reg = {}
        x = End(name="x", init_value=1.0, registry=reg)

        def derive_y(x):
            return x * G

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x,), registry=reg)
        access = make_graph(y, registry=reg)
        access.y.calculate()
        print(access.y.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    env["G_VAL"] = "2.0"
    assert _run_probe(probe, env) == "2.0"           # process 1: cold, writes cache

    env["G_VAL"] = "3.0"
    got = _run_probe(probe, env)                       # process 2: shared cache, changed global
    assert got == "3.0", f"derive served a stale cached global (H3): got {got}, expected 3.0"


def test_make_structref_method_ns_value_is_not_stale_across_processes(tmp_path):
    """A make_structref method that captures a value through the ``ns`` argument
    must not serve a stale cached result when that value changes in a later
    process (issue #73 H4).

    The method source is identical across runs; only the ns-threaded value
    differs. Before the fix the source-only method hash matched, numba loaded
    the frozen binary, and a second process was served the first's baked-in
    value (6.0 instead of 10.0). The method hash now folds the referenced ns
    values, so its content-addressed anchor changes and numba recompiles.
    """
    probe = tmp_path / "struct_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        from numba.experimental.structref import register
        from numba.core.types import StructRef
        from numbox.utils.highlevel import make_structref

        @register
        class GStructTypeClass(StructRef):
            pass

        G = float(os.environ["G_VAL"])

        def scale(self):
            return self.x * G

        make_g = make_structref(
            "GStruct", ("x",), GStructTypeClass,
            struct_methods={"scale": scale}, ns={"G": G},
        )
        s = make_g(2.0)
        print(s.scale())
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    env["G_VAL"] = "3.0"
    assert _run_probe(probe, env) == "6.0"             # process 1: cold, writes cache

    env["G_VAL"] = "5.0"
    got = _run_probe(probe, env)                        # process 2: shared cache, changed ns value
    assert got == "10.0", f"method served a stale ns value (H4): got {got}, expected 10.0"
