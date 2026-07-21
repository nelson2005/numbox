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


def test_digest_fallback_is_pythonhashseed_stable(tmp_path):
    """The digest fallback for an un-fingerprintable function with a str-set
    constant must be PYTHONHASHSEED-independent (issue #73 M11).

    A bare code-object hash embedded the frozenset in its (hash-seeded)
    iteration order, so the cache key varied per process -> cross-process cache
    misses and unbounded cache-dir growth. The best-effort walker canonicalizes
    the set sorted, so the digest is identical across seeds.
    """
    probe = tmp_path / "digest_probe.py"
    probe.write_text(textwrap.dedent('''
        from numbox.core.bindings.call import _call_lib_func
        from numbox.utils.digest import digest

        def cb(x):
            s = {"alpha", "beta", "gamma", "delta", "epsilon"}
            # references the @intrinsic _call_lib_func -> strict walker aborts ->
            # the set-bearing body goes through the best-effort fallback.
            return _call_lib_func("cos", (x,)) if "alpha" in s else 0.0

        print(digest("StateType", (cb,)))
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    env["PYTHONHASHSEED"] = "0"
    d0 = _run_probe(probe, env)
    env["PYTHONHASHSEED"] = "123456789"
    d1 = _run_probe(probe, env)
    assert d0 == d1, f"digest fallback leaked set iteration order (M11): {d0} != {d1}"


def test_make_graph_kernel_is_not_stale_across_jit_flags(tmp_path):
    """A graph built under one ``error_model`` must not be served the binary
    compiled under another in a later process (issue #73 H5).

    ``error_model="numpy"`` makes ``x / 0`` evaluate to ``inf``; the default
    "python" model raises ZeroDivisionError, which ``calculate()`` surfaces by
    leaving the node at its init value. The jit flags reach the generated kernel
    source only as the literal ``@njit(**jit_options)``, and the derive compiles
    as its own cres whose numba cache file is named after the derive's own source
    and qualname -- so before the fix both the kernel name and the derive cache
    entry were identical across the two models, and the second process was served
    the first's binary: ``inf`` where the default model must not produce a value.

    The kernel fingerprint now folds the effective flags, and the derive is
    compiled through a digest-named anchor that folds them too.
    """
    probe = tmp_path / "flags_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        from numba.core.types import float64
        from numbox.core.work.builder import End, Derived, make_graph

        opts = {"cache": True}
        if os.environ["ERROR_MODEL"] == "numpy":
            opts["error_model"] = "numpy"

        reg = {}
        num = End(name="num", init_value=1.0, ty=float64, registry=reg)
        den = End(name="den", init_value=0.0, ty=float64, registry=reg)

        def derive_ratio(num, den):
            return num / den

        ratio = Derived(name="ratio", init_value=0.0, derive=derive_ratio,
                        sources=(num, den), ty=float64, registry=reg)
        access = make_graph(ratio, registry=reg, jit_options=opts)
        try:
            access.ratio.calculate()
        except ZeroDivisionError:
            pass
        print(access.ratio.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    env["ERROR_MODEL"] = "numpy"
    assert _run_probe(probe, env) == "inf"             # process 1: cold, writes cache

    env["ERROR_MODEL"] = "python"
    got = _run_probe(probe, env)                       # process 2: shared cache, other model
    assert got == "0.0", (
        f"kernel/derive served a binary compiled under another error_model (H5): "
        f"got {got}, expected 0.0"
    )

    # And the cache must still be doing its job: the same configuration re-run
    # must not mint a new entry. A "fix" that merely disabled caching would pass
    # the assertion above.
    cache_root = tmp_path / "nbcache"
    before = sorted(p.name for p in cache_root.rglob("*.nbi"))
    assert _run_probe(probe, env) == "0.0"
    after = sorted(p.name for p in cache_root.rglob("*.nbi"))
    assert before == after, f"identical rebuild recompiled instead of reusing: {set(after) - set(before)}"


def test_dispatcher_typed_node_hash_is_process_stable(tmp_path):
    """hash_type of a Dispatcher type must be the same in every process
    (issue #73 M13/L18).

    A Dispatcher type mangles through its repr --
    `type(CPUDispatcher(<function f at 0x7f...>))` -- which carries an ASLR
    address. Folding that into the content-addressed `_make_<hash>` made the
    kernel name change every run, so `cache=True` never hit and each run left a
    fresh orphan cache pair behind. Dispatchers are now fingerprinted by the
    wrapped function's content.

    Both the bare hash and a whole graph's kernel fingerprint are checked, and a
    changed body must still re-key -- process-stability is worthless if it is
    bought by ignoring the content.
    """
    probe = tmp_path / "disp_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        import numbox.core.work.builder as builder_mod
        from numba import njit
        from numba.core.types import float64
        from numbox.core.work.builder import End, Derived, make_graph
        from numbox.utils.highlevel import hash_type

        captured = []
        orig = builder_mod._kernel_fingerprint

        def spy(*a, **k):
            h = orig(*a, **k)
            captured.append(h)
            return h

        builder_mod._kernel_fingerprint = spy

        BUMP = float(os.environ["BUMP"])

        @njit(float64(float64))
        def helper(x):
            return x + BUMP

        reg = {}
        x = End(name="x", init_value=1.0, ty=float64, registry=reg)
        h = End(name="h", init_value=helper, registry=reg)

        def derive_y(x, h):
            return h(x)

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x, h),
                    ty=float64, registry=reg)
        make_graph(y, registry=reg, jit_options={"cache": True})
        print(hash_type(helper._numba_type_), captured[0])
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    env["BUMP"] = "1.0"
    first = _run_probe(probe, env)
    second = _run_probe(probe, env)
    assert first == second, (
        f"Dispatcher-typed hash is process-unstable: {first} != {second}; the "
        "kernel name would change every run and never cache-hit"
    )

    env["BUMP"] = "2.0"
    changed = _run_probe(probe, env)
    assert changed != first, (
        "a changed dispatcher body did not re-key; process-stability must not be "
        "bought by ignoring the content"
    )
