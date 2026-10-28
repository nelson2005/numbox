"""Cross-process cache-correctness regression tests.

Each test runs a probe in two separate interpreter processes that share one ``NUMBA_CACHE_DIR``. Between the runs, the
test changes an input that the numba cache key does not cover, and it asserts that the second process does not return
a stale result from the cache. The tests cover the caching and hashing fixes from the 2026-07-09 review.
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
    """A make_graph derive that reads a module global must not return a stale cached result when the global changes in a
    later process.

    The derive compiles as a standalone dispatcher, and numba's cache key for it covers only co_code and the closure,
    not the global. If the derive were cached, a second process with a changed global would silently return the value
    frozen into the first process's binary (2.0 instead of 3.0). A derive that reads module state is therefore
    compiled uncached.
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
    assert got == "3.0", f"derive served a stale cached global: got {got}, expected 3.0"


def test_make_structref_method_ns_value_is_not_stale_across_processes(tmp_path):
    """A make_structref method that captures a value through the ``ns`` argument must not return a stale cached result
    when that value changes in a later process.

    The method source is identical across runs, and only the value passed through ``ns`` differs. A method hash of the
    source alone would match in both runs. numba would then load the frozen binary, and the second process would
    return the first process's value (6.0 instead of 10.0). The method hash therefore folds the referenced ns values,
    so its content-addressed anchor changes and numba recompiles.
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
    assert got == "10.0", f"method served a stale ns value: got {got}, expected 10.0"


def test_digest_fallback_is_pythonhashseed_stable(tmp_path):
    """The digest fallback for an un-fingerprintable function with a str-set constant must be
    PYTHONHASHSEED-independent.

    A bare code-object hash would embed the frozenset in its iteration order, which depends on the hash seed. The cache
    key would then vary per process, causing cache misses across processes and unbounded growth of the cache
    directory. The best-effort walker canonicalizes the set in sorted order, so the digest is identical across seeds.
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
    assert d0 == d1, f"digest fallback leaked set iteration order: {d0} != {d1}"


def test_make_graph_kernel_is_not_stale_across_jit_flags(tmp_path):
    """A later process must not load the binary compiled under one ``error_model`` for a graph built under another.

    ``error_model="numpy"`` makes ``x / 0`` evaluate to ``inf``. The default "python" model raises ZeroDivisionError,
    and ``calculate()`` shows that error by leaving the node at its init value. The jit flags reach the generated
    kernel source only as the literal ``@njit(**jit_options)``. The derive compiles as its own cres, and numba names
    its cache file after the derive's own source and qualname. If the kernel fingerprint and the derive's anchor left
    out the flags, the kernel name and the derive cache entry would both be identical across the two models. The second
    process would then load the first process's binary and return ``inf``, although the default model must not produce
    a value.

    The kernel fingerprint therefore folds the effective flags, and the derive is compiled through a digest-named
    anchor that folds them as well.
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
        f"kernel/derive served a binary compiled under another error_model: "
        f"got {got}, expected 0.0"
    )

    # The cache must still work: a re-run of the same configuration must not write a new entry. A change that merely
    # disabled caching would also pass the assertion above.
    cache_root = tmp_path / "nbcache"
    before = sorted(p.name for p in cache_root.rglob("*.nbi"))
    assert _run_probe(probe, env) == "0.0"
    after = sorted(p.name for p in cache_root.rglob("*.nbi"))
    assert before == after, f"identical rebuild recompiled instead of reusing: {set(after) - set(before)}"


def test_dispatcher_typed_node_hash_is_process_stable(tmp_path):
    """hash_type of a Dispatcher type must be the same in every process.

    numba mangles a Dispatcher type through its repr, `type(CPUDispatcher(<function f at 0x7f...>))`, which carries an
    ASLR address. If that repr were folded into the content-addressed `_make_<hash>`, the kernel name would change on
    every run. `cache=True` would then never produce a cache hit, and each run would leave a fresh orphan cache pair
    behind. A Dispatcher is therefore fingerprinted by the content of the wrapped function.

    The test checks both the bare hash and a whole graph's kernel fingerprint. A changed body must still re-key,
    because process stability has no value if it comes from ignoring the content.
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
            h, ok = orig(*a, **k)
            captured.append(h)
            return h, ok

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


def test_unfingerprintable_derive_makes_kernel_uncached_no_growth(tmp_path):
    """A make_graph derive that references an un-canonicalizable global (here a numba Type used as a cast) cannot be
    fingerprinted. The kernel that folds its hash must compile without an on-disk cache, instead of writing a fresh
    id()-named ``_make_<hash>`` entry on every run.

    If ``_derive_fingerprint`` fell back to ``<repr> @{id(derive)}`` while the kernel kept ``cache=True``, the count of
    ``builder._make*.nbc`` files would grow from 1 to 2 to 3 across processes. The derive therefore has a best-effort
    fingerprint, which is process-stable, and the enclosing kernel drops its cache, so no cache files accumulate.
    """
    probe = tmp_path / "unfp_derive_probe.py"
    probe.write_text(textwrap.dedent('''
        from numba import float64 as nb_float64
        from numbox.core.work.builder import End, Derived, make_graph

        reg = {}
        x = End(name="x", init_value=2.0, ty=nb_float64, registry=reg)

        def derive_y(x):
            return nb_float64(x) * 2.0   # LOAD_GLOBAL of a numba Type -> unfingerprintable

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x,),
                    ty=nb_float64, registry=reg)
        access = make_graph(y, registry=reg, jit_options={"cache": True})
        access.y.calculate()
        print(access.y.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    cache = tmp_path / "nbcache"
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    counts = []
    for _ in range(3):
        assert _run_probe(probe, env) == "4.0"                 # correct value, compiled uncached
        counts.append(sum(1 for _ in cache.rglob("builder._make*.nbc")))
    assert counts[1] == counts[0] and counts[2] == counts[0], (
        f"an un-fingerprintable derive reminted the kernel per process: "
        f"builder._make*.nbc counts {counts}")
    assert counts[0] == 0, (
        f"an un-fingerprintable-derive kernel must be uncached (best-effort hash is "
        f"content-blind), but {counts[0]} _make .nbc were written")


def test_intrinsic_referencing_node_makes_kernel_uncached_no_growth(tmp_path):
    """A graph node typed as a Dispatcher whose wrapped body references an un-canonicalizable global has only a
    best-effort type identity. The kernel that folds this identity must drop its on-disk cache.

    If ``hash_type`` fell back to ``mangle_type_or_value``, whose Dispatcher form embeds an ASLR address, the count of
    ``builder._make*.nbc`` files would grow with every process. ``_type_identity`` therefore uses a process-stable
    best-effort fingerprint and reports the type as uncacheable, so ``make_graph`` compiles the kernel without a cache.
    """
    probe = tmp_path / "intrin_node_probe.py"
    probe.write_text(textwrap.dedent('''
        from numba import njit, float64 as nb_float64
        from numbox.core.work.builder import End, Derived, make_graph

        @njit(nb_float64(nb_float64))
        def helper(x):
            return nb_float64(x) * 2.0    # numba Type cast in the body -> unfingerprintable

        reg = {}
        x = End(name="x", init_value=3.0, ty=nb_float64, registry=reg)
        h = End(name="h", init_value=helper, registry=reg)

        def derive_y(x, h):
            return h(x)

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x, h),
                    ty=nb_float64, registry=reg)
        access = make_graph(y, registry=reg, jit_options={"cache": True})
        access.y.calculate()
        print(access.y.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    cache = tmp_path / "nbcache"
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    counts = []
    for _ in range(3):
        assert _run_probe(probe, env) == "6.0"
        counts.append(sum(1 for _ in cache.rglob("builder._make*.nbc")))
    assert counts[1] == counts[0] and counts[2] == counts[0], (
        f"a Dispatcher-typed node with an un-fingerprintable body reminted the "
        f"kernel per process: builder._make*.nbc counts {counts}")
    assert counts[0] == 0, (
        f"the kernel must be uncached when a node type has only a best-effort "
        f"identity, but {counts[0]} _make .nbc were written")


def test_dispatcher_typed_component_is_uncached_no_growth(tmp_path):
    """A graph with a Dispatcher-typed component must not accumulate a cache pair per process, for both the fused kernel
    and the derive anchor. Here the component is a source node that holds an njit helper and feeds a cache=True
    derive.

    numba cannot cache a Dispatcher-typed signature across processes. Even with a process-stable, content-addressed
    name, numba's on-disk index misses on every run and appends a fresh ``.nbc``, as measured. If the units were
    cached, the counts of ``builder._make*.nbc`` and ``_derive_*.nbc`` files would both grow from 1 to 2 to 3. The
    numbox-named kernel and derive units are therefore compiled uncached, so neither is written and neither
    accumulates. The two existing Dispatcher tests assert only the *name* (the fingerprint or the ``.nbi``), so they
    cannot detect such growth. The helper body here is fully fingerprintable, so this test covers the ordinary case,
    not a degenerate one.

    Scope: the test covers the numbox-named kernel and derive units, whose caching numbox controls. numba's own Work
    structref-method cache (``work.Work.calculate``, ``_calculate_``, ...) still grows by one entry per process for a
    Dispatcher-typed node. Those ``@njit(cache=True)`` methods specialize on the Dispatcher-carrying Work type, which
    numba cannot cache across processes either. That growth is a pre-existing numba limitation, outside the scope of
    this test, and that cache never loads a stale binary.
    """
    probe = tmp_path / "disp_component_probe.py"
    probe.write_text(textwrap.dedent('''
        from numba import njit, float64 as nb_float64
        from numbox.core.work.builder import End, Derived, make_graph

        @njit(nb_float64(nb_float64))
        def helper(x):
            return x + 1.0             # fully fingerprintable body

        reg = {}
        x = End(name="x", init_value=1.0, ty=nb_float64, registry=reg)
        h = End(name="h", init_value=helper, registry=reg)

        def derive_y(x, h):
            return h(x)

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x, h),
                    ty=nb_float64, registry=reg)
        access = make_graph(y, registry=reg, jit_options={"cache": True})
        access.y.calculate()
        print(access.y.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    cache = tmp_path / "nbcache"
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    kernel_counts, derive_counts = [], []
    for _ in range(3):
        assert _run_probe(probe, env) == "2.0"
        kernel_counts.append(sum(1 for _ in cache.rglob("builder._make*.nbc")))
        derive_counts.append(sum(1 for _ in cache.rglob("_derive_*.nbc")))
    assert kernel_counts == [0, 0, 0], (
        f"a Dispatcher-typed node left kernel cache pairs behind (numba cannot "
        f"cross-process-cache it): builder._make*.nbc counts {kernel_counts}")
    assert derive_counts == [0, 0, 0], (
        f"a Dispatcher-typed source left derive cache pairs behind: "
        f"_derive_*.nbc counts {derive_counts}")


def test_nested_dispatcher_node_kernel_and_derive_uncached_no_growth(tmp_path):
    """A node typed as a heterogeneous tuple that *contains* an njit dispatcher must not accumulate the numbox-named
    kernel or derive cache units either. The top-level isinstance check does not catch a nested Dispatcher, so the type
    falls through to the address-bearing mangle. Without the guard, both units would keep cache=True, and the counts
    of builder._make*.nbc and _derive_*.nbc files would grow from 1 to 2 to 3. numba's own Work structref-method cache
    still grows for a Dispatcher-typed node. That growth is a separate, pre-existing numba limitation that the test
    does not address, so the test covers only the numbox-named units, whose caching numbox controls.
    """
    probe = tmp_path / "nested_disp_probe.py"
    probe.write_text(textwrap.dedent('''
        from numba import njit, float64 as nb_float64
        from numbox.core.work.builder import End, Derived, make_graph

        @njit(nb_float64(nb_float64))
        def helper(x):
            return x + 1.0

        reg = {}
        # a heterogeneous tuple node: Tuple(float64, Dispatcher) -- nested dispatcher
        p = End(name="p", init_value=(2.0, helper), registry=reg)

        def derive_y(p):
            return p[0] + 1.0

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(p,),
                    ty=nb_float64, registry=reg)
        access = make_graph(y, registry=reg, jit_options={"cache": True})
        access.y.calculate()
        print(access.y.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    cache = tmp_path / "nbcache"
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    kernel_counts, derive_counts = [], []
    for _ in range(3):
        assert _run_probe(probe, env) == "3.0"
        kernel_counts.append(sum(1 for _ in cache.rglob("builder._make*.nbc")))
        derive_counts.append(sum(1 for _ in cache.rglob("_derive_*.nbc")))
    assert kernel_counts == [0, 0, 0], (
        f"a nested-Dispatcher node left kernel cache pairs behind: {kernel_counts}")
    assert derive_counts == [0, 0, 0], (
        f"a nested-Dispatcher node left derive cache pairs behind: {derive_counts}")


def test_module_state_derive_keeps_kernel_cacheable(tmp_path):
    """A derive that reads a module global is compiled uncached, but the enclosing kernel stays cacheable, because its
    name folds the global's value. An unchanged re-run therefore hits the cache instead of writing a new entry. The
    test checks that dropping the kernel cache for the *un-fingerprintable* path does not also drop it for the
    module-state path: kernel_safe stays True when the derive fingerprints.
    """
    probe = tmp_path / "modstate_probe.py"
    probe.write_text(textwrap.dedent('''
        from numba import float64 as nb_float64
        from numbox.core.work.builder import End, Derived, make_graph

        G = 2.0    # plain module global, canonicalizable -> derive fingerprints

        reg = {}
        x = End(name="x", init_value=3.0, ty=nb_float64, registry=reg)

        def derive_y(x):
            return x * G

        y = Derived(name="y", init_value=0.0, derive=derive_y, sources=(x,),
                    ty=nb_float64, registry=reg)
        access = make_graph(y, registry=reg, jit_options={"cache": True})
        access.y.calculate()
        print(access.y.data)
    '''), encoding="utf-8")

    env = dict(os.environ)
    cache = tmp_path / "nbcache"
    env["NUMBA_CACHE_DIR"] = str(cache)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    counts = []
    for _ in range(3):
        assert _run_probe(probe, env) == "6.0"
        counts.append(sum(1 for _ in cache.rglob("builder._make*.nbc")))
    assert counts[0] > 0, "a module-state-reading graph's kernel must still cache"
    assert counts[1] == counts[0] and counts[2] == counts[0], (
        f"a module-state kernel reminted across processes with an unchanged global: {counts}")
