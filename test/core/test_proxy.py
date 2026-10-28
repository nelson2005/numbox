import importlib
import math
import os
import re
import subprocess
import sys
import textwrap
import threading
import warnings

import numpy as np
import pytest
from numba import float64, njit, typeof
from numba.core.errors import TypingError
from numba.core.types import FunctionType, Omitted
from numba.core.types.function_type import CompileResultWAP

from numbox.core.bindings.errno import errno_get
from numbox.core.bindings.libc import getenv, memcpy
from numbox.core.bindings.call import _call_lib_func
from numbox.core.configurations import numba_version
from numbox.core.proxy.proxy import proxy, proxy_if_available, make_proxy_name
from numbox.utils.derive_wap import DeriveFunctionType, DeriveWAP
from numbox.utils.lowlevel import array_data_p, get_unicode_data_p
from test.auxiliary_utils import (
    assert_njit_cache_survives_subprocess_roundtrip,
    collect_and_run_tests,
    open_libm,
)


aux_1_sig = float64(float64)


@proxy(aux_1_sig, jit_options={'cache': True})
def aux_1(x):
    return 3.14 * x


def test_1():
    assert abs(aux_1(2.2) - 3.14 * 2.2) < 1e-15
    llvm_ir = next(iter(aux_1.inspect_llvm().values()))
    assert aux_1.__name__ == make_proxy_name('aux_1')
    if 'numbox_pxy_aux_1' in llvm_ir:
        alias_name = r"double @numbox_pxy_aux_1_\w+\(double"  # noqa: W605
        assert len(re.findall(f"declare {alias_name}", llvm_ir)) == 1
        assert len(re.findall(f"call {alias_name}", llvm_ir)) == 1
    else:
        print(f"LLVM inspection disabled for cached code, {aux_1.__name__}")


aux_2_sig = [float64(float64, float64), float64(float64, Omitted(1.3))]


@proxy(aux_2_sig, jit_options={'cache': True})
def aux_2(x, *, y=1.3):
    return 3.14 * x + y


def test_2():
    assert abs(aux_2(2.2) - (3.14 * 2.2 + 1.3)) < 1e-15
    assert abs(aux_2(2.2, 1.4) - (3.14 * 2.2 + 1.4)) < 1e-15


def _sole_compile_result(dispatcher):
    """Return the single compiled result on a numba dispatcher."""
    sigs = dispatcher.nopython_signatures
    assert len(sigs) == 1, sigs
    return dispatcher.get_compile_result(sigs[0])


def test_proxy_zero_arg_caller_is_cacheable():
    @njit(cache=True)
    def caller():
        return errno_get()
    caller()
    assert not _sole_compile_result(caller).library.has_dynamic_globals


def test_proxy_single_arg_caller_is_cacheable():
    @njit(cache=True)
    def caller(name_p):
        return getenv(name_p)
    caller(get_unicode_data_p("NUMBOX_NONEXISTENT_XYZZY"))
    assert not _sole_compile_result(caller).library.has_dynamic_globals


def test_proxy_multi_arg_caller_is_cacheable():
    @njit(cache=True)
    def caller(dst, src):
        memcpy(array_data_p(dst), array_data_p(src), src.nbytes)
    caller(np.zeros(4, dtype=np.uint8), np.arange(4, dtype=np.uint8))
    assert not _sole_compile_result(caller).library.has_dynamic_globals


def test_proxy_caller_survives_subprocess_round_trip(tmp_path):
    """A caller of an ``@proxy`` binding survives a real cross-process round trip through the numba cache.

    The heuristic tests above (``has_dynamic_globals is False``) only prove cache *eligibility*. This test proves
    cache *correctness*. A caller compiled with ``@njit(cache=True)`` against an ``@proxy`` binding round-trips
    through the on-disk cache (.nbi/.nbc files). The second process loads the cached IR and produces output identical
    to that of the first process, which starts with a cold cache. Neither file is rewritten on the warm run (mtimes
    preserved).

    ``proxy`` declares a process-stable alias for the callee's cfunc wrapper as an extern in the caller's IR module,
    and ``llvmlite.binding.add_symbol`` registers that alias. llvmlite's JIT linker resolves the alias in each
    process when the cache is reloaded. The cached IR therefore survives ASLR across processes without embedding any
    runtime address. See the ``assert_njit_cache_survives_subprocess_roundtrip`` helper in ``test/auxiliary_utils.py``
    for the full assertion contract. See ``test_proxy_referenced_symbol_is_process_stable`` for the reason why the
    alias, and not numba's process-local ``v<uid>`` wrapper name, keeps concurrently built caches consistent.
    """
    assert_njit_cache_survives_subprocess_roundtrip(
        tmp_path,
        probe_source="""
            from numba import njit
            from numbox.core.bindings.errno import errno_get, errno_set

            @njit(cache=True)
            def caller():
                errno_set(42)
                return errno_get()

            v = caller()
            assert v == 42, f"got {v!r}"
            print(v)
        """,
        expected_stdout_lines=["42"],
    )


def test_proxy_referenced_symbol_is_process_stable(tmp_path):
    """A caller must embed the *same* callee symbol regardless of compile order.

    This is a regression test for the concurrent-cache hazard. ``proxy`` references the cfunc wrapper of each body by
    a deterministic alias registered with ``llvmlite.binding.add_symbol``, not by numba's process-local ``v<uid>``
    wrapper name. If ``proxy`` regressed to the uid name, two processes that first compiled a different number of
    functions would embed different symbols in otherwise-equal cached objects. A concurrently built shared cache could
    then pair a body that defines ``v<Na>`` with a caller that references ``v<Nb>``, and the load would abort with
    ``LLVM ERROR: Symbol not found: cfunc...``. The test runs a probe twice, with a different number of warm-up
    compiles, and asserts that the embedded callee symbol is identical in both runs and is the stable alias.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent('''
        import sys
        from numba import njit
        from numba.core import types
        from numbox.core.proxy.proxy import proxy

        def d0(x): return x
        def d1(x): return x
        def d2(x): return x
        def d3(x): return x
        def d4(x): return x

        for _f in (d0, d1, d2, d3, d4)[:int(sys.argv[1])]:
            njit(types.int64(types.int64))(_f)(0)

        @proxy(types.int64(types.int64))
        def binding(x):
            return x + 1

        @njit
        def caller(x):
            return binding(x)

        caller(0)
        ir = "\\n".join(caller.inspect_llvm().values())
        toks = set()
        for tok in ir.replace("(", " ").replace(")", " ").replace("*", " ").split():
            if tok.startswith("@") and "numbox_pxy_" in tok:
                toks.add(tok.strip('@"'))
        print("|".join(sorted(toks)))
    '''), encoding="utf-8")

    def _run(prior):
        r = subprocess.run(
            [sys.executable, str(probe), str(prior)],
            capture_output=True, text=True, encoding="utf-8",
        )
        assert r.returncode == 0, f"probe failed (prior={prior}):\n{r.stderr}"
        return r.stdout.strip()

    baseline = _run(0)
    shifted = _run(5)
    assert baseline, "no callee symbol found in caller IR"
    assert baseline == shifted, (
        "@proxy baked a process-dependent callee symbol (concurrent-cache hazard):\n"
        f"  prior=0: {baseline!r}\n  prior=5: {shifted!r}"
    )
    assert "numbox_pxy_" in baseline, f"expected a stable add_symbol alias, got {baseline!r}"


def test_proxy_if_available_present_symbol_returns_real_proxy():
    """When the C symbol is present, ``proxy_if_available`` returns a real ``@proxy``-wrapped dispatcher with
    ``.as_func`` attached. ``.as_func`` has the type that ``proxy`` produces: a ``DeriveWAP``."""
    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    @proxy_if_available(lib, float64(float64), jit_options={"cache": True})
    def cos(x):
        return _call_lib_func("cos", (x,))

    assert hasattr(cos, "as_func")
    assert isinstance(cos.as_func, DeriveWAP)
    assert abs(cos(0.5) - math.cos(0.5)) < 1e-15


def test_proxy_if_available_missing_symbol_returns_stub():
    """When the C symbol is absent, ``proxy_if_available`` returns a Python stub that raises ``NotImplementedError``
    on call. The stub intentionally lacks ``.as_func`` (see helper docstring).

    The stub metadata matches the real ``@proxy`` dispatcher where applicable. :func:`make_proxy_name` prefixes
    ``__name__``, so callers that ``repr()`` or log the binding see the same name whether or not the symbol was
    available. ``__qualname__`` and ``__doc__`` preserve the user-side function for debugging.
    """
    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    @proxy_if_available(lib, float64(float64))
    def nonexistent_fn(x):
        """Docstring on the stub-target for metadata-preservation check."""
        return x

    assert nonexistent_fn.__name__ == make_proxy_name("nonexistent_fn")
    assert nonexistent_fn.__qualname__.endswith("nonexistent_fn")
    assert nonexistent_fn.__doc__ == "Docstring on the stub-target for metadata-preservation check."
    assert not hasattr(nonexistent_fn, "as_func")
    with pytest.raises(NotImplementedError, match="nonexistent_fn is not available"):
        nonexistent_fn(1.0)


def test_proxy_if_available_missing_symbol_njit_raises_clear_error():
    """When the C symbol is absent, calling the stub from ``@njit`` raises a
    clear ``TypingError`` naming the binding and the missing-symbol cause at
    typing time, not an opaque numba typing failure."""
    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    @proxy_if_available(lib, float64(float64))
    def nonexistent_njit_fn(x):
        return x

    @njit
    def use_it(x):
        return nonexistent_njit_fn(x)

    with pytest.raises(TypingError) as excinfo:
        use_it(1.0)
    message = str(excinfo.value)
    assert "nonexistent_njit_fn is not available" in message
    assert "C symbol missing" in message


def test_proxy_factory_closures_dispatch_to_their_own_body():
    """Same-qualname @proxy closures over different values must not collide.

    A factory ``mk(c)`` produces two ``@proxy`` closures with an identical module+qualname+signature but a different
    closure value ``c``. Without a body fingerprint in the alias, both closures would map to the same alias.
    llvmlite's last-writer-wins ``add_symbol`` would then silently rebind both jitted callers to the second body, and
    call_p1(10) would return 30 instead of 20. The folded fingerprint captures the closure cell, so each body receives
    its own alias, and each caller calls its own body.
    """
    def mk(c):
        @proxy(float64(float64))
        def tmpl(x):
            return c * x
        return tmpl

    p1 = mk(2.0)
    p2 = mk(3.0)

    @njit
    def call_p1(x):
        return p1(x)

    @njit
    def call_p2(x):
        return p2(x)

    assert p1(10.0) == 20.0 and p2(10.0) == 30.0
    assert call_p1(10.0) == 20.0, "jitted caller of p1 reached the wrong body (alias collision)"
    assert call_p2(10.0) == 30.0, "jitted caller of p2 reached the wrong body (alias collision)"


def test_proxy_factory_over_c_symbols_dispatch_distinctly():
    """Factory-made proxies over per-instance C symbol names must not collide.

    The common numbox binding body references the ``@intrinsic`` ``_call_lib_func``, which the strict fingerprint
    cannot canonicalize. The best-effort fallback still captures the per-instance closure (the C symbol name), so two
    same-qualname bindings over different symbols each call their own body. Without that fallback, the alias would be
    computed from the bytecode alone, which is identical for both, and a caller of ``cos`` would return ``sin``'s
    result.
    """
    import numbox.core.bindings.libm  # noqa: F401 - loads libm so cos/sin resolve

    def mk(fname):
        @proxy(float64(float64))
        def libfn(x):
            return _call_lib_func(fname, (x,))
        return libfn

    p_cos = mk("cos")
    p_sin = mk("sin")

    @njit(float64(float64))
    def call_cos(x):
        return p_cos(x)

    @njit(float64(float64))
    def call_sin(x):
        return p_sin(x)

    assert abs(call_cos(1.0) - math.cos(1.0)) < 1e-12, "jitted caller of cos reached the wrong body"
    assert abs(call_sin(1.0) - math.sin(1.0)) < 1e-12, "jitted caller of sin reached the wrong body"


def test_proxy_redefinition_keeps_callers_on_their_own_body():
    """Re-decorating a same module+qualname+sig function with a different body must not rebind a later-compiled
    caller of the stale dispatcher to the new body.

    The test reproduces the in-process redefinition hazard. Without the body fingerprint in the alias, a caller that
    is compiled after the redefinition but calls the first dispatcher would run the second body, and
    caller_after(1.0) would return 101.0. The folded body fingerprint gives the two bodies distinct aliases.
    """
    @proxy(float64(float64))
    def g(x):
        return x + 1.0

    old_g = g

    @proxy(float64(float64))
    def g(x):  # noqa: F811 - deliberate same-name redefinition
        return x + 100.0

    @njit
    def caller_after(x):
        return old_g(x)

    assert old_g(1.0) == 2.0
    assert g(1.0) == 101.0
    assert caller_after(1.0) == 2.0, "stale dispatcher's caller reached the redefined body"


def test_proxy_absent_binding_caller_gets_loud_diagnostic_not_segfault(tmp_path):
    """A cache=True caller of a proxy_if_available binding that is present when the cache is written but absent on
    reload must fail with a diagnostic. It must not segfault, and it must not return a value.

    numba's cache key for the caller does not distinguish callees. A warm caller therefore hits the cache even in a
    process in which ``proxy_if_available`` took the absent (stub) path and registered no real body. Without the
    trap, the unresolved extern would link to null, and the call would be a bare SIGSEGV (exit 139, zero
    diagnostics). The trap keeps the alias resolvable, but reaching the trap does not suffice. The trap raises its
    ``RuntimeError`` inside a ``@cfunc``, where numba discards the exception and returns zero, so a caller that
    reached the trap would print a wrong answer and exit 0. The cache guard therefore discards an entry whose alias
    resolves only to a trap. The recompilation then reaches the typing error of the stub, which is the same outcome
    that a cold cache gives.
    """
    pkg = tmp_path / "flip_pkg"
    pkg.mkdir()
    (pkg / "binding_flip.py").write_text(textwrap.dedent('''
        import os
        from numba.core.types import float64
        from numbox.core.proxy.proxy import proxy_if_available

        class FakeLib:
            pass

        lib = FakeLib()
        if os.environ.get("HAVE_SYM") == "1":
            lib.myfn = True

        @proxy_if_available(lib, float64(float64), jit_options={"cache": True})
        def myfn(x):
            return x * 2.0
    '''), encoding="utf-8")
    (pkg / "caller_flip.py").write_text(textwrap.dedent('''
        from numba import njit
        from numba.core.types import float64
        from binding_flip import myfn

        @njit(float64(float64), cache=True)
        def caller(x):
            return myfn(x) + 1.0
    '''), encoding="utf-8")
    (pkg / "run_flip.py").write_text(textwrap.dedent('''
        from caller_flip import caller
        print("RESULT", caller(2.5), flush=True)
    '''), encoding="utf-8")

    cache_dir = tmp_path / "nbcache"
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    env["PYTHONPATH"] = os.pathsep.join([str(pkg), *sys.path])

    # Process 1: the symbol is available, so the alias is registered and the caller cache is written.
    env["HAVE_SYM"] = "1"
    r1 = subprocess.run(
        [sys.executable, str(pkg / "run_flip.py")],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert r1.returncode == 0, f"warm-up failed:\n{r1.stderr}"
    assert r1.stdout.strip() == "RESULT 6.0", r1.stdout

    # Process 2: the symbol is absent, so the stub path runs and the alias is never registered. The warm caller hits
    # the cache and references the unregistered alias.
    env["HAVE_SYM"] = "0"
    r2 = subprocess.run(
        [sys.executable, str(pkg / "run_flip.py")],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert r2.returncode not in (-11, 139), (
        f"segfault instead of a loud diagnostic (returncode={r2.returncode})\n{r2.stderr}"
    )
    assert "RESULT" not in r2.stdout, (
        f"the absent binding was called and produced a value: {r2.stdout!r}"
    )
    assert "TypingError" in r2.stderr and "myfn" in r2.stderr, (
        f"expected a typing error naming the binding on stderr, got:\n{r2.stderr}"
    )
    assert "not available in the loaded library" in r2.stderr, r2.stderr


def test_proxy_function_above_anchor_line_raises_clear_error(tmp_path):
    # The cache anchor prepends blank lines, so that the generated @njit is placed at the function's co_firstlineno.
    # A function defined above that line cannot be anchored, because a negative prepend would be silently clamped to
    # 0 and would anchor the function at the wrong line. Decorating such a function must raise a clear error instead.
    mod = tmp_path / "top_proxy_mod.py"
    mod.write_text(
        "from numba import float64\n"
        "from numbox.core.proxy.proxy import proxy\n"
        "@proxy(float64(float64))\n"
        "def top_fn(x):\n"
        "    return 3.14 * x\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(ValueError, match="anchor"):
            importlib.import_module("top_proxy_mod")
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("top_proxy_mod", None)


# A proxied body that raises, shared by the exception-semantics tests below.
aux_raises_sig = float64(float64)


@proxy(aux_raises_sig, jit_options={'cache': True})
def aux_raises(x):
    if x > 0.0:
        raise ValueError("proxy boom")
    return x + 1.0


def test_proxy_as_func_is_a_derive_wap():
    """``.as_func`` is the handle that carries the propagating calling convention.

    The binding's ``.as_func`` is a ``DeriveWAP`` typed as ``DeriveFunctionType``. That type lets a first-class call
    select numba's own calling convention and unwind an exception out of the proxied body. ``isinstance(...,
    CompileResultWAP)`` holds for the value too, because ``DeriveWAP`` subclasses it. The test therefore also asserts
    the ``DeriveWAP`` type, the ``DeriveFunctionType`` typing and a populated ``jit_addr``.
    """
    @proxy(float64(float64))
    def scaled(x):
        return 2.0 * x

    as_func = scaled.as_func
    assert isinstance(as_func, CompileResultWAP)
    assert isinstance(as_func, DeriveWAP)
    assert isinstance(typeof(as_func), DeriveFunctionType)
    assert as_func.jit_address != 0, "the callconv entry point is what fills `jit_addr`"


def test_proxy_as_func_as_an_njit_argument_propagates_a_raising_body():
    """An exception raised inside a proxied body reaches the caller through ``.as_func``.

    The caller's signature is inferred, so numba types the parameter from the value passed in and sees
    ``DeriveFunctionType``. A plain ``CompileResultWAP`` in the same position carries no entry point that could
    unwind. With it, the call would return a zero-filled ``0.0`` and report the failure only as an unraisable
    exception on stderr.
    """
    @njit
    def call_through(f, x):
        return f(x)

    assert call_through(aux_raises.as_func, -3.0) == -2.0
    with pytest.raises(ValueError, match="proxy boom"):
        call_through(aux_raises.as_func, 1.0)


def test_proxy_as_func_as_a_jitted_constant_propagates_a_raising_body():
    """The same contract holds on the constant path, which is lowered separately.

    An argument is typed when it is passed in, and then unboxed. A constant is resolved at compile time and lowered
    by ``lower_constant``. The two paths therefore reach ``jit_addr`` through different code, and one can regress
    while the other still passes. The caching behavior also depends on this path. Declaring the entry point
    symbolically rather than as a literal address lets a ``cache=True`` caller of a constant-reached ``.as_func`` be
    cached.
    """
    as_func = aux_raises.as_func

    @njit
    def call_constant(x):
        return as_func(x)

    assert call_constant(-3.0) == -2.0
    with pytest.raises(ValueError, match="proxy boom"):
        call_constant(1.0)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_proxy_dispatcher_call_still_discards_a_raising_body():
    """The other handle of the binding, the dispatcher, still discards an exception on every numba version.

    Calling the dispatcher goes through the cfunc wrapper of the proxied body, which numba documents as not
    supporting exceptions. The wrapper discards the exception, zero-fills the return value and reports the failure
    only as an unraisable exception on stderr. One binding therefore carries two handles with different exception
    semantics, and the test asserts the dispatcher's side of that difference.

    The unraisable exception reaches pytest as a ``PytestUnraisableExceptionWarning``. The marker above filters it,
    as ``test/core/test_sqlite_tvf.py`` does for its own raising callback. The test asserts the returned value and
    not the warning text, because the ``repr`` of the context object that numba names in the text varies by numba
    version.
    """
    assert aux_raises(-3.0) == -2.0
    assert aux_raises(1.0) == 0.0, "the cfunc wrapper stopped zero-filling a discarded raise"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_proxy_as_func_declared_as_a_plain_function_type_still_discards():
    """A value that crosses the Python boundary into a *declared* plain ``FunctionType`` degrades to the C convention.

    ``DeriveFunctionType.can_convert_to`` permits the conversion, so an explicitly typed ``njit`` written before the
    derive type existed keeps accepting a binding. ``can_convert_to`` also documents that the value arrives on the C
    convention, so the exception is discarded exactly as it would be without the derive type. The sibling suite
    covers that degradation for a ``cres`` value. A ``@proxy`` binding reaches the same declaration through
    ``.as_func``, and this test covers that case.

    The test asserts the type of the value passed in first, because both assertions below also hold for a plain
    ``CompileResultWAP``. Without that check, the test would keep passing if ``.as_func`` silently stopped being a
    ``DeriveWAP``, and ``can_convert_to`` would never be entered.

    As in the dispatcher-call test above, the discarded exception is reported as an unraisable exception, and the
    test asserts only the returned value.
    """
    as_func = aux_raises.as_func
    assert isinstance(as_func, DeriveWAP), (
        f"the value handed in is a {type(as_func).__name__}, so the declared plain "
        f"`FunctionType` never reaches `DeriveFunctionType.can_convert_to`"
    )

    @njit(float64(FunctionType(float64(float64)), float64))
    def call_through_declared_plain(f, x):
        return f(x)

    assert call_through_declared_plain(as_func, -3.0) == -2.0
    assert call_through_declared_plain(as_func, 1.0) == 0.0


def test_proxy_as_func_mixed_with_a_numba_native_wrapper_fails_to_unify_for_numba_lt_68_and_works_for_numba_gte_68():
    """Characterization of a narrowing: a heterogeneous tuple of function values is refused
    for numba<0.68.

    numba<0.68.0 unifies a tuple's element types in ``unified_function_type`` with a bare
    class-identity comparison. That comparison runs before any of numbox's conversions apply, so a
    ``DeriveFunctionType`` element beside a plain ``FunctionType`` one raises an ``AssertionError`` without a message
    in ``numba/core/utils.py``. The same tuple would return a value if ``.as_func`` were a plain
    ``CompileResultWAP``, so the narrowing is a real loss of capability. The test records it, so that it does not
    surprise a caller. Homogeneous tuples are unaffected.

    ``test/utils/test_derive_wap.py::test_the_derive_type_stays_distinct_from_the_plain_function_type`` explains why
    the two types cannot simply be made to compare equal.

    numba<0.68.0's assert carries no message, so the frame it was raised from is checked instead of any
    text. Matching on ``AssertionError`` alone would be satisfied by an unrelated one raised
    anywhere in the same call.
    """
    @njit(float64(float64))
    def native(x):
        return x + 10.0

    foreign = CompileResultWAP(native.get_compile_result(native.nopython_signatures[0]))

    @njit
    def use_pair(pair, x):
        return pair[0](x) + pair[1](x)

    if numba_version >= 68:
        assert use_pair((aux_raises.as_func, foreign), -3.0) == 5
    else:
        with pytest.raises(AssertionError) as raised:
            use_pair((aux_raises.as_func, foreign), -3.0)
        assert raised.traceback[-1].name == "unified_function_type", (
            f"the refusal moved out of numba's function-type unification: {raised.traceback[-1].name}"
        )


if __name__ == "__main__":
    collect_and_run_tests(__name__)


def test_proxy_jit_flags_dispatch_to_their_own_body():
    """Two decorations of one body differing only in jit flags must not collide.

    The flags govern the machine code that ``njit(sig, **jit_options)`` emits. They appear in neither the
    module+qualname+signature identity nor the body fingerprint, and the body source is byte-identical here. Without
    the flags in the alias, both decorations would map to a single alias. llvmlite's last-writer-wins ``add_symbol``
    would then rebind every caller to whichever decoration compiled last. A jitted caller of the ``numpy``
    error-model binding would run the ``python`` one, so ``1.0 / 0.0`` would return 0.0 instead of inf with exit 0.
    The ``ZeroDivisionError`` would be discarded at the cfunc boundary.
    """
    def mk(error_model):
        @proxy(float64(float64), jit_options={"error_model": error_model})
        def recip(x):
            return 1.0 / x
        return recip

    p_np = mk("numpy")
    p_py = mk("python")

    assert p_np._numbox_proxy_alias != p_py._numbox_proxy_alias, (
        "jit flags absent from the alias -- the two bodies collide"
    )

    @njit
    def call_np(x):
        return p_np(x)

    assert math.isinf(call_np(0.0)), (
        "jitted caller of the numpy error-model binding reached the python one"
    )


def _make_extfn_binding(symbol):
    """Return a factory binding over ``symbol``, a shape whose captured value the fingerprint must include."""
    from numba.core.types import ExternalFunction, float64 as nb_f8

    fn = ExternalFunction(symbol, nb_f8(nb_f8))

    @proxy(float64(float64))
    def body(x):
        return fn(x)
    return body


def test_two_factory_bindings_over_different_symbols_get_different_aliases():
    """Two bodies that a factory made over different C symbols must not share one alias.

    The bodies share a code object and a qualname, so the alias can distinguish them only by the captured value. If
    neither the alias nor the collision check could distinguish them, both bodies would receive one alias, and
    ``ll.add_symbol`` would keep the second registration. Every caller compiled against the first body would then be
    silently rebound to the second body. The result would be a wrong number, decided by construction order, with no
    diagnostic.
    """
    floor_b = _make_extfn_binding("floor")
    ceil_b = _make_extfn_binding("ceil")
    assert floor_b._numbox_proxy_alias != ceil_b._numbox_proxy_alias, (
        "two bodies over different C symbols collapsed onto one alias"
    )

    @njit(float64(float64))
    def call_floor(x):
        return floor_b(x)

    @njit(float64(float64))
    def call_ceil(x):
        return ceil_b(x)

    assert call_floor(2.5) == 2.0, "the floor binding ran another body"
    assert call_ceil(2.5) == 3.0, "the ceil binding ran another body"


def test_bodies_the_fingerprint_cannot_separate_are_detected_rather_than_rebound():
    """The residual class is a captured value with no process-stable identity.

    A ctypes pointer built from a raw address has no library and no symbol name. Two such pointers therefore differ
    only by an address that changes between processes. No fingerprint that has to mean the same thing in the next run
    can separate them. The alias is therefore allowed to collide, and the collision is detected instead. The second
    body is published under a process-local name of its own, so both bodies stay callable and correct. The shared
    name is retired, so that no warm caller is bound to whichever body holds the name at the time.
    """
    import ctypes
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    def make(addr):
        fp = proto(addr)

        @proxy(float64(float64))
        def body(x):
            return fp(x)
        return body

    first = make(ctypes.cast(libm.floor, ctypes.c_void_p).value)
    with pytest.warns(AliasCollisionWarning):
        second = make(ctypes.cast(libm.ceil, ctypes.c_void_p).value)

    assert first._numbox_proxy_alias != second._numbox_proxy_alias, (
        "the second body was handed a name the first already holds"
    )
    assert first._numbox_proxy_alias in _ABSENT_ALIASES, (
        "the shared name still identifies a body, so a warm caller can be served the wrong one"
    )

    @njit(float64(float64))
    def call_first(x):
        return first(x)

    @njit(float64(float64))
    def call_second(x):
        return second(x)

    assert call_first(2.5) == 2.0
    assert call_second(2.5) == 3.0


def test_a_retired_alias_stays_retired_when_its_body_is_registered_again():
    """A retired alias must stay retired after the body that lost the name registers again, not only at the loss.

    The collision above retires the shared name because another process may build the two bodies in the other
    order, and a later registration of either body does not change that. If that registration removed the alias from
    ``_ABSENT_ALIASES``, a warm ``cache=True`` caller would again receive the entry that the retirement exists to
    discard. Which of the two bodies that entry was compiled against depends on a construction order that this
    process cannot observe. Every other retirement assertion in this file reads the set at the moment of
    publication. This test reads it after the same body is registered a second time.

    The two assertions before the last one prevent this test from passing by construction. A re-registration
    reaches the branch under test only when it is read as the same body. It must receive the shared name again,
    rather than a ``_c<N>`` name of its own, and it must issue no warning. An implementation that read it as a fresh
    collision would also leave the alias retired. Without those two assertions, the test would pass without ever
    reaching the branch that it covers. The test fails against an implementation that clears ``_ABSENT_ALIASES``
    unconditionally on that branch.
    """
    import ctypes
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    def make(addr):
        fp = proto(addr)

        @proxy(float64(float64))
        def body(x):
            return fp(x)
        return body

    floor_addr = ctypes.cast(libm.floor, ctypes.c_void_p).value
    first = make(floor_addr)
    alias = first._numbox_proxy_alias
    with pytest.warns(AliasCollisionWarning):
        second = make(ctypes.cast(libm.ceil, ctypes.c_void_p).value)
    assert alias in _ABSENT_ALIASES, (
        "the collision left the shared name identifying a body, so there is nothing here to keep retired"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", AliasCollisionWarning)
        again = make(floor_addr)

    assert again._numbox_proxy_alias == alias, (
        "the body that already holds the shared name was handed a second name, so this never "
        "reached the branch that reads a registration as the same body"
    )
    assert alias in _ABSENT_ALIASES, (
        "registering the first body again took the shared name back out of _ABSENT_ALIASES; the "
        "name still does not say which of the two bodies a caller cached in another process was "
        "compiled against, and that caller now loads instead of recompiling"
    )

    @njit(float64(float64))
    def call_again(x):
        return again(x)

    @njit(float64(float64))
    def call_the_other(x):
        return second(x)

    assert call_again(2.5) == 2.0, "the re-registered body ran another body"
    assert call_the_other(2.5) == 3.0, (
        "re-pointing the shared name rebound the body that took a name of its own"
    )


def _make_scaling_binding(scale):
    """Return a binding whose body captures only values that the fingerprint canonicalizes."""
    @proxy(float64(float64))
    def scaled(x):
        return scale * x
    return scaled


def _make_intrinsic_binding():
    """Return a binding whose body reads a global that the fingerprint cannot canonicalize.

    ``_call_lib_func`` is an ``@intrinsic``, which has no canonical form. This body is therefore fingerprinted
    through the best-effort walker, and its digest carries a placeholder where the intrinsic was. Two different
    bodies can still receive one alias only through that placeholder, so the re-registration check must handle this
    path correctly.
    """
    @proxy(float64(float64))
    def flr(x):
        return _call_lib_func("floor", (x,))
    return flr


def test_one_body_registered_twice_keeps_its_alias():
    """Registering the same body again is not a collision, and must not be read as one.

    A factory called twice with equal arguments builds one body twice, and so does reloading a module. Both
    registrations reach the alias that the first registration already holds, but both name the same code.
    Re-pointing the alias is therefore correct, and it is also what would happen without the collision check. If the
    check read the registration as a collision, it would retire the alias. That would needlessly discard every warm
    caller of the alias in this and every later process.
    """
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    with warnings.catch_warnings():
        warnings.simplefilter("error", AliasCollisionWarning)
        first = _make_scaling_binding(2.0)
        second = _make_scaling_binding(2.0)

    assert first._numbox_proxy_alias == second._numbox_proxy_alias, (
        "one body registered twice was handed two different aliases"
    )
    assert first._numbox_proxy_alias not in _ABSENT_ALIASES, (
        "one body registered twice retired its own alias, so every warm caller recompiles"
    )

    @njit(float64(float64))
    def call(x):
        return second(x)

    assert call(3.0) == 6.0


def test_a_body_over_an_opaque_value_registered_twice_keeps_its_alias():
    """The same holds for the fingerprint path that carries a placeholder.

    Every shipped libc, libm and sqlite binding takes this path, because each of them reads the ``@intrinsic``
    ``_call_lib_func``; errno, stdio and strerror read intrinsics of their own. If the check compared the two
    registrations by the address they compiled to, it would report a collision for all of them. One ``importlib.reload``
    of a bindings module would then retire all 42 libm aliases. The check therefore compares the values behind the
    placeholders instead.
    """
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    with warnings.catch_warnings():
        warnings.simplefilter("error", AliasCollisionWarning)
        first = _make_intrinsic_binding()
        second = _make_intrinsic_binding()

    assert first._numbox_proxy_alias == second._numbox_proxy_alias, (
        "one body registered twice was handed two different aliases"
    )
    assert first._numbox_proxy_alias not in _ABSENT_ALIASES, (
        "one body registered twice retired its own alias, so every warm caller recompiles"
    )

    @njit(float64(float64))
    def call(x):
        return second(x)

    assert call(2.5) == 2.0


def test_a_trap_keeps_its_hands_off_an_alias_a_present_binding_holds():
    """The absent-symbol trap must not take an alias that a live body already holds.

    Suppose one definition is bound over two candidate library handles, and the first handle carries the symbol while
    the second does not. Both bindings reach one alias, because ``_stable_cfunc_alias`` keys on the body, signature
    and jit_options, and not on the handle. If the trap were registered there anyway, it would replace a body that
    returns the right answer with a cfunc that raises inside its own wrapper. The wrapper discards the exception and
    returns a zeroed value to the caller, measured as ``6.2e-310`` where ``1.0`` is correct, with no warning of any
    kind. That is the silent wrong answer that this registry exists to prevent, reached from the other direction.

    The body keeps the alias, because a caller compiled in this process calls a binding that is present. A process
    that only ever sees the absent handle registers the trap under a name that no body holds, and the trap there
    provides the diagnostic that it was written for.
    """
    from llvmlite import binding as ll
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    class _WithoutTheSymbol:
        """A second candidate handle, exporting nothing."""

    def bind(handle):
        @proxy_if_available(handle, float64(float64))
        def cos(x):
            return _call_lib_func("cos", (x,))
        return cos

    present = bind(lib)
    alias = present._numbox_proxy_alias
    address_before = ll.address_of_symbol(alias)

    with pytest.warns(AliasCollisionWarning, match="no trap was installed"):
        bind(_WithoutTheSymbol())

    assert ll.address_of_symbol(alias) == address_before, (
        "the trap took an alias a present binding already answers to"
    )
    assert alias not in _ABSENT_ALIASES, (
        "the alias was retired even though the body holding it is callable here, so every "
        "warm caller of a working binding recompiles for nothing"
    )

    @njit(float64(float64))
    def call_present(x):
        return present(x)

    assert abs(call_present(0.5) - math.cos(0.5)) < 1e-15, (
        "a caller compiled after the absent handle was bound reached the trap, not the body"
    )


def test_a_body_reaching_a_traps_alias_is_told_it_found_a_trap():
    """The collision message must name the object that holds the alias.

    A trap records a witness that no body can match, so a body that reaches the alias of a trap always takes the
    collision branch. If the warning reported that case as "two @proxy bodies fingerprint alike", it would describe a
    second body that does not exist. Anyone diagnosing a cold cache would then look for a factory that is not there.

    The action itself is correct and does not change. ``_stale_proxy_aliases`` treats membership of
    ``_ABSENT_ALIASES`` as stale whether or not the symbol still resolves. Keeping the alias retired therefore
    prevents a warm ``cache=True`` caller from receiving the zero that the trap returns after its wrapper discards
    the exception, in a process in which the binding really is missing.
    """
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    class _WithoutTheSymbol:
        """A first candidate handle, exporting nothing."""

    def bind(handle):
        @proxy_if_available(handle, float64(float64))
        def cos(x):
            return _call_lib_func("cos", (x,))
        return cos

    bind(_WithoutTheSymbol())
    with pytest.warns(AliasCollisionWarning, match="absent-binding trap") as caught:
        present = bind(lib)

    assert "fingerprint alike" not in str(caught[0].message), (
        "the trap case reports a second body that does not exist"
    )
    assert present._numbox_proxy_alias.rsplit("_c", 1)[0] in _ABSENT_ALIASES, (
        "the alias a trap answers to was handed to a body, so a warm caller in a process "
        "without the binding is served the trap's swallowed return instead of its error"
    )
    assert abs(present(0.5) - math.cos(0.5)) < 1e-15


def test_open_libm_hands_back_a_usable_handle():
    """The test checks the contract of the helper on the platform that runs it.

    Twelve tests skip when ``open_libm`` returns ``None``, and four of them are the only coverage of the
    ctypes-pointer fingerprint path. If the helper silently stopped finding a math library on some platform, the
    other tests would lose that coverage there and still pass. The assertion here turns that case into a failure. On
    Windows, it detects the case in which neither ``msvcrt`` nor ``ucrtbase`` can supply the symbols.

    This test checks the platform, not the gate. Any handle that reaches these assertions has already satisfied the
    check in ``open_libm``. A gate narrowed to a subset would therefore still pass here on any runtime that exports
    all three symbols. ``test_open_libm_refuses_a_handle_a_caller_would_reach_past`` below checks what the gate
    admits.

    This test is in this file rather than beside the helper in ``test/auxiliary_utils.py``, because that filename
    does not match pytest's ``python_files``, and a plain run collects nothing in it.
    """
    lib = open_libm()
    assert lib is not None, "no usable math library on this platform, so twelve tests skip"
    assert hasattr(lib, "ceil") and hasattr(lib, "cos") and hasattr(lib, "floor")
    assert open_libm() is not lib, "handles are shared, so a two-handle comparison is vacuous"


def test_open_libm_refuses_a_handle_a_caller_would_reach_past(monkeypatch):
    """A gate narrower than the set of symbols that the callers bind admits a library on which they then fail.

    ``open_libm`` returns ``None`` so that a platform without a usable C runtime skips rather than fails. That holds
    only while the gate covers each symbol that the callers need. Four of the callers bind ``cos``. A runtime that
    carries ``ceil`` and ``floor`` but not ``cos`` must therefore yield ``None`` and skip those tests, rather than
    yield a handle and fail them.

    Every runtime in the matrix exports all three symbols, so a narrowed gate would go unnoticed by the rest of the
    suite. The test hides each symbol in turn from a handle that is otherwise real, so the check exercises the gate
    rather than the platform.
    """
    import ctypes

    from numbox.core.bindings import utils

    class _Hiding:
        def __init__(self, lib, hidden):
            self._lib = lib
            self._hidden = hidden

        def __getattr__(self, name):
            if name == self._hidden:
                raise AttributeError(name)
            return getattr(self._lib, name)

    real = open_libm()
    if real is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    for hidden in ("ceil", "cos", "floor"):
        crippled = _Hiding(real, hidden)
        if utils.platform_ == "Windows":
            monkeypatch.setattr(ctypes, "CDLL", lambda name, _h=crippled: _h)
        else:
            monkeypatch.setattr(utils, "load_lib_path", lambda path, _h=crippled: _h)
        assert open_libm() is None, f"a handle without {hidden} was admitted"


def test_one_symbol_reached_through_two_prototypes_is_not_one_body():
    """A ctypes pointer's address does not identify the body, and a check that treats it as the body fails silently.

    Two prototypes over one symbol are two different calls. numba types and lowers a call from the ``restype`` and
    ``argtypes`` of the pointer. The ``c_double`` prototype therefore compiles a call that returns a double, the
    ``c_float`` one compiles a call that returns a float, and the two bodies are different machine code. They
    fingerprint alike, because a pointer built from a raw address has no process-stable identity to fold. Both
    bodies therefore map to one placeholder and reach one alias. A check that distinguished the two registrations by
    the address alone would read them as one body registered twice. The branch for that case re-points the alias and
    issues no warning, so the second body would silently take over the symbol that every caller of the first was
    compiled against.

    The last assertion is the one that fails against that implementation. There, a caller compiled after the second
    registration returns 0.0 where 2.0 is right, with exit 0 and no warning of any kind. The assertions before it
    establish the premise that it needs. The caller compiled *before* the second registration returns the right
    answer either way, because it was linked before any rebinding. That makes the pair of callers a measurement
    rather than a coincidence. The registered address separates a second name from a re-pointed symbol, so a fix
    that assigns a new name but still re-points the old one does not pass.
    """
    import ctypes

    from llvmlite import binding as ll
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take a symbol address from")
    addr = ctypes.cast(libm.floor, ctypes.c_void_p).value

    def make(fp):
        @proxy(float64(float64))
        def body(x):
            return fp(x)
        return body

    first = make(ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(addr))
    alias = first._numbox_proxy_alias
    address_before = ll.address_of_symbol(alias)

    @njit(float64(float64))
    def call_before(x):
        return first(x)

    assert call_before(2.5) == 2.0, "the first body was not calling floor to begin with"

    with pytest.warns(AliasCollisionWarning, match="fingerprint alike"):
        second = make(ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float)(addr))

    assert second._numbox_proxy_alias != alias, (
        "the second prototype was handed the name the first is published under"
    )
    assert alias in _ABSENT_ALIASES, (
        "the shared name still identifies a body, so a warm caller can be served the wrong one"
    )
    assert ll.address_of_symbol(alias) == address_before, (
        "the second body re-pointed the symbol the first is published under"
    )

    @njit(float64(float64))
    def call_after(x):
        return first(x)

    assert call_after(2.5) == 2.0, (
        "a caller compiled after the second registration reached the other prototype's body"
    )


def test_a_body_that_lost_a_collision_does_not_serve_a_warm_caller(tmp_path):
    """The name given to the body that loses a collision does not identify it, so no warm caller may load on it.

    The name is built from a count of the aliases that this process has published. The same name therefore belongs
    to the other body in a process that builds the colliding pair in the other order. The two processes here
    share one numba cache and build the pair in opposite orders. The first process caches a caller of the ceil
    binding, which loses the collision there and is published under the process-local name. The second process
    builds ceil first, so ceil keeps the shared alias, and floor receives the process-local name. numba's cache key
    does not distinguish callees, so the warm caller hits the cache. If the process-local name were not retired, the
    caller would compute ``floor(2.5)`` and print 2.0 where 3.0 is right, with exit 0 and no warning. Retiring the
    process-local name together with the shared one discards that entry.

    The ``STALE`` count prevents this test from passing by construction. A second process that never found the
    cached entry would also print 3.0, because numba would recompile the caller for some reason of its own. The
    second process must therefore report that it found a cached entry and discarded it. The first process must
    report no discarded entry, which is the other half of the check. If the cache were already stale when the first
    process started, the discard in the second run would show nothing about the order.
    """
    if open_libm() is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")

    pkg = tmp_path / "order_pkg"
    pkg.mkdir()
    (pkg / "colliding_pair.py").write_text(textwrap.dedent('''
        import ctypes
        import os

        from numba.core.types import float64

        from numbox.core.proxy.proxy import proxy
        from test.auxiliary_utils import open_libm

        libm = open_libm()
        proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

        def make(symbol):
            fp = proto(ctypes.cast(getattr(libm, symbol), ctypes.c_void_p).value)

            @proxy(float64(float64))
            def body(x):
                return fp(x)
            return body

        if os.environ["ORDER"] == "floor_first":
            floor_binding = make("floor")
            ceil_binding = make("ceil")
        else:
            ceil_binding = make("ceil")
            floor_binding = make("floor")
    '''), encoding="utf-8")
    (pkg / "caller_pair.py").write_text(textwrap.dedent('''
        from numba import njit
        from numba.core.types import float64

        from colliding_pair import ceil_binding

        @njit(float64(float64), cache=True)
        def caller(x):
            return ceil_binding(x)
    '''), encoding="utf-8")
    (pkg / "run_pair.py").write_text(textwrap.dedent('''
        import warnings

        from numbox.core.proxy.proxy import StaleProxyCacheWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import caller_pair
        print("STALE", sum(issubclass(w.category, StaleProxyCacheWarning) for w in caught), flush=True)
        print("RESULT", caller_pair.caller(2.5), flush=True)
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join([str(pkg), *sys.path])

    def run(order):
        env["ORDER"] = order
        done = subprocess.run(
            [sys.executable, str(pkg / "run_pair.py")],
            capture_output=True, text=True, encoding="utf-8", env=env,
        )
        assert done.returncode == 0, f"{order} failed:\n{done.stderr}"
        return done.stdout

    cold = run("floor_first")
    assert "STALE 0" in cold, f"the cold run discarded a cache entry it should not have:\n{cold}"
    assert "RESULT 3.0" in cold, f"the caller of the ceil binding did not run ceil:\n{cold}"

    warm = run("ceil_first")
    assert "RESULT 3.0" in warm, (
        f"a warm caller of the ceil binding was served against the body the minted name "
        f"happens to point at in this process:\n{warm}"
    )
    assert "STALE 0" not in warm, (
        f"the warm run reported no discarded entry, so the right answer here says nothing "
        f"about the order it was cached in -- it can be had by never writing the cache at "
        f"all:\n{warm}"
    )


def test_a_body_deriving_from_a_collision_does_not_serve_a_warm_caller(tmp_path):
    """Retiring the two colliding names is not enough, because a body that captures one of them inherits the order.

    The sibling test above covers a caller of a colliding body. This test covers the same failure one composition
    step further, where retiring the two names alone does not protect the caller. A second ``@proxy`` body that
    merely *calls* a colliding binding captures its dispatcher. If the fingerprint folded that dispatcher by the
    alias that the dispatcher holds at the time, the alias of the derived body would depend on the order in which
    the process built the pair. The wrapper over ceil would fold the process-local name in a process that built
    floor first, and the wrapper over floor would fold it in a process that built ceil first. The two wrappers would
    therefore exchange names between the two runs. Neither wrapper name would be retired, because each is newly
    built and resolves. Unlike the colliding pair, the derived body would therefore keep its cross-process cache.
    numba's cache key does not distinguish callees, so the warm caller would hit the cache and compute
    ``floor(2.5)`` where ``ceil(2.5)`` is right. It would print 2.0 with exit 0, and the only warning would be the
    collision warning, which says that callers of both bodies recompile in every process.

    The fingerprint therefore refuses to fold a retired alias, and that closes the gap. The two wrappers become
    indistinguishable to the fingerprint, rather than distinguishable by the wrong property. They therefore collide
    with each other, and that collision retires their names, as the first collision retired the binding names.

    The ``STALE`` counts serve the same purpose as in the sibling test. The warm run must report that it found a
    cached entry and discarded it. Otherwise, the run could produce the right answer because the cache was never
    written.
    """
    if open_libm() is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")

    pkg = tmp_path / "derived_pkg"
    pkg.mkdir()
    (pkg / "colliding_pair.py").write_text(textwrap.dedent('''
        import ctypes
        import os

        from numba.core.types import float64

        from numbox.core.proxy.proxy import proxy
        from test.auxiliary_utils import open_libm

        libm = open_libm()
        proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

        def make(symbol):
            fp = proto(ctypes.cast(getattr(libm, symbol), ctypes.c_void_p).value)

            @proxy(float64(float64))
            def body(x):
                return fp(x)
            return body

        if os.environ["ORDER"] == "floor_first":
            floor_binding = make("floor")
            ceil_binding = make("ceil")
        else:
            ceil_binding = make("ceil")
            floor_binding = make("floor")
    '''), encoding="utf-8")
    (pkg / "wrapped_pair.py").write_text(textwrap.dedent('''
        from numba.core.types import float64

        from numbox.core.proxy.proxy import proxy

        from colliding_pair import ceil_binding, floor_binding

        def make_caller(binding):
            @proxy(float64(float64))
            def caller(x):
                return binding(x)
            return caller

        wrapped_floor = make_caller(floor_binding)
        wrapped_ceil = make_caller(ceil_binding)
    '''), encoding="utf-8")
    (pkg / "caller_pair.py").write_text(textwrap.dedent('''
        from numba import njit
        from numba.core.types import float64

        from wrapped_pair import wrapped_ceil

        @njit(float64(float64), cache=True)
        def caller(x):
            return wrapped_ceil(x)
    '''), encoding="utf-8")
    (pkg / "run_pair.py").write_text(textwrap.dedent('''
        import warnings

        from numbox.core.proxy.proxy import StaleProxyCacheWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import caller_pair
        print("STALE", sum(issubclass(w.category, StaleProxyCacheWarning) for w in caught), flush=True)
        print("RESULT", caller_pair.caller(2.5), flush=True)
    '''), encoding="utf-8")

    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "nbcache")
    env["PYTHONPATH"] = os.pathsep.join([str(pkg), *sys.path])

    def run(order):
        env["ORDER"] = order
        done = subprocess.run(
            [sys.executable, str(pkg / "run_pair.py")],
            capture_output=True, text=True, encoding="utf-8", env=env,
        )
        assert done.returncode == 0, f"{order} failed:\n{done.stderr}"
        return done.stdout

    cold = run("floor_first")
    assert "STALE 0" in cold, f"the cold run discarded a cache entry it should not have:\n{cold}"
    assert "RESULT 3.0" in cold, f"the caller of the ceil wrapper did not run ceil:\n{cold}"

    warm = run("ceil_first")
    assert "RESULT 3.0" in warm, (
        f"a warm caller of the wrapper around the ceil binding was served against the wrapper "
        f"around floor, because the wrapper's own alias followed the order its dependency was "
        f"built in:\n{warm}"
    )
    assert "STALE 0" not in warm, (
        f"the warm run reported no discarded entry, so the right answer here says nothing "
        f"about the order it was cached in -- it can be had by never writing the cache at "
        f"all:\n{warm}"
    )


class _AliasLockProbe:
    """Asks, from a second thread, whether the alias registry lock is free at this instant.

    A ``threading.Lock`` is not reentrant, and the thread being probed is the one that holds it, so that thread
    cannot ask the question itself. The timeout turns the answer into an observation rather than a deadlock. Every
    probe waits for the timeout once when the lock is held, as it should be. The probe is disarmed until the
    publication under test, so that the registrations before it do not wait for the timeout.
    """

    def __init__(self, lock):
        self.lock = lock
        self.armed = False
        self.excluded = {}

    def at(self, where):
        if not self.armed:
            return
        taken = []
        waiter = threading.Thread(target=lambda: taken.append(self.lock.acquire(timeout=0.25)))
        waiter.start()
        waiter.join()
        if taken[0]:
            self.lock.release()
        self.excluded[where] = not taken[0]


class _EntryCountingLock:
    """The alias lock, counting how many times the region it guards was entered.

    A lock found held at two instants does not show that the two instants are inside one critical section. A lock
    taken and released separately around each instant gives the same answer to a probe. It also leaves a gap between
    the instants that is wide enough for another thread to build the same name. The count distinguishes the two
    cases, because a publication that releases the lock between the instants enters the lock more than once.
    """

    def __init__(self, lock):
        self.lock = lock
        self.entries = 0

    def __enter__(self):
        self.lock.acquire()
        self.entries += 1
        return self

    def __exit__(self, *exc_info):
        self.lock.release()
        return False


class _ProbingLl:
    """``llvmlite.binding`` as seen by ``proxy.py``, with a probe at each symbol registration."""

    def __init__(self, real, probe, where):
        self.real = real
        self.probe = probe
        self.where = where

    def __getattr__(self, name):
        return getattr(self.real, name)

    def add_symbol(self, name, address):
        self.probe.at(self.where)
        return self.real.add_symbol(name, address)


def test_a_colliding_body_takes_its_name_under_the_registry_lock(monkeypatch):
    """The name given to a body that loses a collision is read from the registry and written back into it.

    ``len(_PUBLISHED_ALIASES)`` decides the name, and the store that follows claims it. Two threads that both read
    the count before either stores would therefore build one name for two bodies. ``ll.add_symbol`` keeps the
    registration written last, and the name of the other body then resolves to it. That is the silent rebinding that
    the registry exists to prevent. Plain Python reaches ``_publish_cfunc_alias`` at decoration time, after numba's
    compiler lock has been released, so nothing else serializes it.

    The window is narrow, and timing decides which thread loses the race. The test therefore checks the exclusion
    rather than a race: while one publisher is inside the region, no other thread can enter it. That takes two
    measurements. First, the test probes the region at both ends: before the comparison that decides the branch, and
    at the ``add_symbol`` that registers the new name. A lock around only the read or only the write leaves the count
    and the store on opposite sides of the lock, and duplicate names still result. Second, the test counts entries
    into the lock. Both ends are inside *a* critical section even if the lock is released and taken again between
    them, and that reopens the window that the lock exists to close. The test rules out those two implementations,
    and an implementation without any lock. The probe runs on a second thread with a timeout, because the lock is
    not reentrant and the publishing thread holds it.
    """
    import ctypes

    from llvmlite import binding as ll

    import numbox.core.proxy.proxy as proxy_module
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ALIAS_LOCK

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    probe = _AliasLockProbe(_ALIAS_LOCK)
    real_match = proxy_module._opaque_values_match

    def spy_match(old, new):
        probe.at("while the branch was being decided")
        return real_match(old, new)

    monkeypatch.setattr(proxy_module, "_opaque_values_match", spy_match)
    monkeypatch.setattr(
        proxy_module, "ll", _ProbingLl(ll, probe, "while the minted name was being registered")
    )

    def make(addr):
        fp = proto(addr)

        @proxy(float64(float64))
        def body(x):
            return fp(x)
        return body

    first = make(ctypes.cast(libm.floor, ctypes.c_void_p).value)
    counted = _EntryCountingLock(_ALIAS_LOCK)
    monkeypatch.setattr(proxy_module, "_ALIAS_LOCK", counted)
    probe.armed = True
    try:
        with pytest.warns(AliasCollisionWarning):
            second = make(ctypes.cast(libm.ceil, ctypes.c_void_p).value)
    finally:
        probe.armed = False

    assert first._numbox_proxy_alias != second._numbox_proxy_alias, (
        "the second body did not take a name of its own, so it never minted one"
    )
    assert sorted(probe.excluded) == [
        "while the branch was being decided", "while the minted name was being registered"
    ], f"the publish region was not reached at both ends: {sorted(probe.excluded)}"
    for where, held in probe.excluded.items():
        assert held, f"another thread could enter the publish region {where}"
    assert counted.entries == 1, (
        f"the publication entered the alias lock {counted.entries} times, so the read that "
        f"decides the name and the store that takes it are not held apart by one critical "
        f"section; another thread can hold the lock in the gap and mint the same name"
    )


class _TrapCompileHeld:
    """Replaces ``cfunc`` and holds the absent side inside the compile of its trap.

    ``proxy.py`` compiles a cfunc only for the trap. The replacement is therefore enough to stop an absent
    registration between the read that decides it and the write that acts on it. It also lets a body publish at that
    point deliberately rather than by chance. The replacement still checks the function name, so that a future use
    of ``cfunc`` on the present side cannot deadlock the test.
    """

    def __init__(self, real_cfunc):
        self.real_cfunc = real_cfunc
        self.compiling = threading.Event()
        self.published = threading.Event()

    def __call__(self, sig):
        def compile_it(fn):
            if fn.__name__ == "_trap":
                self.compiling.set()
                assert self.published.wait(60), "the present side never published"
            return self.real_cfunc(sig)(fn)
        return compile_it


def test_a_trap_does_not_take_an_alias_a_body_won_while_the_trap_was_compiling(monkeypatch):
    """The trap decides that no body holds the alias, and that decision must still be true when the trap writes.

    ``_register_absent_alias_trap`` reads ``_PUBLISHED_ALIASES`` to decide whether a body already holds the alias,
    and then compiles a cfunc before it writes. That compile is the widest check-then-act window in the module:
    around 100 ms, against around 40 ms for a publication. A body that publishes on another thread therefore
    publishes during the compile routinely rather than rarely, after the decision that found no body. The trap raises
    inside a cfunc wrapper, which discards the exception and returns a zeroed value. The caller then gets
    ``6.2e-310`` where the answer of the library was right, with no warning of any kind. That is the same silent
    wrong answer that :func:`_publish_cfunc_alias` exists to prevent, reached from the other direction.

    The test blocks the absent side rather than racing the two sides, so it demonstrates the interleaving with
    certainty rather than with some probability. The absent side is held inside its compile until the present side
    has published, which is the ordering that the difference in timing produces without the block.
    """
    from llvmlite import binding as ll

    import numbox.core.proxy.proxy as proxy_module
    from numbox.core.proxy.proxy import AliasCollisionWarning, _ABSENT_ALIASES

    lib = open_libm()
    if lib is None:
        pytest.skip("No suitable math/C runtime library discoverable")

    class _WithoutTheSymbol:
        """A second candidate handle, exporting nothing."""

    def bind(handle):
        @proxy_if_available(handle, float64(float64))
        def floor(x):
            return _call_lib_func("floor", (x,))
        return floor

    held = _TrapCompileHeld(proxy_module.cfunc)
    monkeypatch.setattr(proxy_module, "cfunc", held)
    present = {}
    absent_warnings = []

    def register_present():
        assert held.compiling.wait(60), "the absent side never reached its compile"
        present["binding"] = bind(lib)
        held.published.set()

    def register_absent():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bind(_WithoutTheSymbol())
        absent_warnings.extend(w.category for w in caught)

    absent = threading.Thread(target=register_absent)
    live = threading.Thread(target=register_present)
    absent.start()
    live.start()
    live.join(120)
    held.published.set()
    absent.join(120)
    assert not (absent.is_alive() or live.is_alive()), "a registration thread did not finish"
    assert "binding" in present, "the present side did not finish its registration"
    binding = present["binding"]
    alias = binding._numbox_proxy_alias

    @njit(float64(float64))
    def call_present(x):
        return binding(x)

    assert abs(call_present(2.5) - 2.0) < 1e-15, (
        f"a caller compiled after the race reached the trap, not the body: "
        f"{call_present(2.5)!r}"
    )
    assert alias not in _ABSENT_ALIASES, (
        "the alias of a body that is callable here was retired by a trap that lost to it, so "
        "every warm caller of a working binding recompiles for nothing"
    )
    assert ll.address_of_symbol(alias) == proxy_module._PUBLISHED_ALIASES[alias][0], (
        "the symbol and the registry disagree about what answers to the alias"
    )
    assert AliasCollisionWarning in absent_warnings, (
        f"the trap gave up its alias without saying so: {absent_warnings}"
    )


def test_each_body_that_loses_a_collision_takes_a_name_no_other_body_holds():
    """A process-local name identifies a body only if the next body that loses a collision does not receive it too.

    The name comes from ``len(_PUBLISHED_ALIASES)`` and is registered with ``ll.add_symbol``, which is
    last-writer-wins and silent. If two losing bodies received one name, the result would be the same wrong answer
    as two bodies on one alias, one level down. The count grows on every publication, and that growth prevents it. A
    constant, or a per-alias counter that did not advance, would give the same name at the one collision that the
    other tests exercise. Three bodies on one alias are the smallest arrangement in which the difference shows.
    """
    import ctypes

    from llvmlite import binding as ll

    from numbox.core.proxy.proxy import AliasCollisionWarning

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take three distinct symbol addresses from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    def make(symbol):
        fp = proto(ctypes.cast(getattr(libm, symbol), ctypes.c_void_p).value)

        @proxy(float64(float64))
        def three_way(x):
            return fp(x)
        return three_way

    first = make("floor")
    with pytest.warns(AliasCollisionWarning):
        second = make("ceil")
    with pytest.warns(AliasCollisionWarning):
        third = make("cos")

    aliases = [b._numbox_proxy_alias for b in (first, second, third)]
    assert len(set(aliases)) == 3, (
        f"two bodies were published under one name, so whichever registered last holds it and "
        f"the other's callers reach the wrong body: {aliases}"
    )

    @njit(float64(float64))
    def call_second(x):
        return second(x)

    @njit(float64(float64))
    def call_third(x):
        return third(x)

    assert abs(call_second(2.5) - 3.0) < 1e-15, (
        f"the second body's name reached another body: {call_second(2.5)!r}"
    )
    assert abs(call_third(2.5) - math.cos(2.5)) < 1e-15, (
        f"the third body's name reached another body: {call_third(2.5)!r}"
    )
    assert len({ll.address_of_symbol(a) for a in aliases}) == 3, (
        "three names, fewer than three addresses: one of them was overwritten by add_symbol"
    )
