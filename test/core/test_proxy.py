import importlib
import math
import os
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from numba import float64, njit, typeof
from numba.core.errors import TypingError
from numba.core.types import FunctionType, Omitted
from numba.core.types.function_type import CompileResultWAP

from numbox.core.bindings.errno import errno_get
from numbox.core.bindings.libc import getenv, memcpy
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.utils import load_lib_path, platform_
from numbox.core.configurations import numba_version
from numbox.core.proxy.proxy import proxy, proxy_if_available, make_proxy_name
from numbox.utils.derive_wap import DeriveFunctionType, DeriveWAP, jit_addr_supported
from numbox.utils.lowlevel import array_data_p, get_unicode_data_p
from test.auxiliary_utils import (
    assert_njit_cache_survives_subprocess_roundtrip,
    collect_and_run_tests,
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
    """Real cross-process cache survival test for @proxy-decorated bindings.

    The heuristic tests above (``has_dynamic_globals is False``) only prove
    cache *eligibility*. This test proves cache *correctness*: a caller
    compiled with ``@njit(cache=True)`` against an ``@proxy`` binding
    actually round-trips through the on-disk cache (.nbi/.nbc files), with
    the second process loading the cached IR and producing identical output
    to the cold-cache first process — and neither file is rewritten on the
    warm run (mtimes preserved).

    ``proxy`` declares a process-stable alias for the callee's cfunc wrapper
    (registered via ``llvmlite.binding.add_symbol``) as an extern in the caller's IR module;
    llvmlite's JIT linker resolves the alias per process at cache reload, so
    cached IR survives ASLR across processes without baking in any runtime
    address. See the ``assert_njit_cache_survives_subprocess_roundtrip`` helper
    in ``test/auxiliary_utils.py`` for the full assertion contract, and
    ``test_proxy_referenced_symbol_is_process_stable`` for why the alias (not
    numba's process-local ``v<uid>`` wrapper name) is what keeps concurrently
    built caches consistent.
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
    """A caller must bake the *same* callee symbol regardless of compile order.

    Regression for the concurrent-cache hazard: ``proxy`` references each body's
    cfunc wrapper by a deterministic alias registered via ``llvmlite.binding.add_symbol``,
    not numba's process-local ``v<uid>`` wrapper name. If it regressed to the uid
    name, two processes that compiled a different number of functions first would
    bake different symbols into otherwise-equal cached objects, so a
    concurrently-built shared cache could pair a body defining ``v<Na>`` with a
    caller referencing ``v<Nb>`` and abort on load with
    ``LLVM ERROR: Symbol not found: cfunc...``. We run a probe twice with a
    different number of warm-up compiles and assert the baked callee symbol is
    identical (and is the stable alias).
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


def _locate_libm():
    """Find a math/libc library with at least the ``cos`` symbol."""
    if platform_ == "Windows":
        from ctypes.util import find_msvcrt
        return find_msvcrt()
    from ctypes.util import find_library
    return find_library("m")


def test_proxy_if_available_present_symbol_returns_real_proxy():
    """When the C symbol is present, ``proxy_if_available`` returns a
    real ``@proxy``-wrapped dispatcher with ``.as_func`` attached, of the
    same numba-version-dependent type ``proxy`` hands out: a ``DeriveWAP``
    where the ``jit_addr`` slot exists, a plain ``CompileResultWAP`` where
    it does not."""
    lib_path = _locate_libm()
    if lib_path is None:
        pytest.skip("No suitable math/C runtime library discoverable")
    lib = load_lib_path(lib_path)

    @proxy_if_available(lib, float64(float64), jit_options={"cache": True})
    def cos(x):
        return _call_lib_func("cos", (x,))

    assert hasattr(cos, "as_func")
    if jit_addr_supported():
        assert isinstance(cos.as_func, DeriveWAP)
    else:
        # `isinstance` would pass either way, since `DeriveWAP` subclasses this.
        assert type(cos.as_func) is CompileResultWAP
    assert abs(cos(0.5) - math.cos(0.5)) < 1e-15


def test_proxy_if_available_missing_symbol_returns_stub():
    """When the C symbol is absent, ``proxy_if_available`` returns a
    Python stub that raises ``NotImplementedError`` on call. The stub
    intentionally lacks ``.as_func`` (see helper docstring).

    Stub metadata matches the real ``@proxy`` dispatcher where applicable:
    ``__name__`` is prefixed via :func:`make_proxy_name` (so callers
    that ``repr()`` or log the binding see the same shape regardless
    of whether the symbol was available); ``__qualname__`` and
    ``__doc__`` preserve the user-side function for debugging.
    """
    lib_path = _locate_libm()
    if lib_path is None:
        pytest.skip("No suitable math/C runtime library discoverable")
    lib = load_lib_path(lib_path)

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
    lib_path = _locate_libm()
    if lib_path is None:
        pytest.skip("No suitable math/C runtime library discoverable")
    lib = load_lib_path(lib_path)

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

    A factory ``mk(c)`` produces two ``@proxy`` closures with an identical
    module+qualname+signature but different closure value ``c``. Before the
    alias folded a body fingerprint, both mapped to the same alias and
    llvmlite's last-writer-wins ``add_symbol`` silently rebound BOTH jitted
    callers to the second body (the review reproduced call_p1(10)=30 instead of
    20). The folded fingerprint captures the closure cell, so each body gets its
    own alias and each caller reaches its own body.
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

    The common numbox binding body references the ``@intrinsic``
    ``_call_lib_func``, which the strict fingerprint cannot canonicalize; the
    best-effort fallback still captures the per-instance closure (the C symbol
    name), so two same-qualname bindings over different symbols reach their own
    body. Before that fallback the alias collapsed to bytecode alone (identical
    for both) and a caller of ``cos`` returned ``sin``'s result.
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
    """Re-decorating a same module+qualname+sig function with a different body
    must not rebind a stale dispatcher's later-compiled caller to the new body.

    Reproduces the in-process redefinition hazard: with the pre-fix alias a
    caller compiled after the redefinition but calling the FIRST dispatcher got
    the SECOND body (the review reproduced caller_after(1.0)=101.0). The folded
    body fingerprint gives the two bodies distinct aliases.
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
    """A cache=True caller of a proxy_if_available binding that is present when
    the cache is written but absent on reload must fail loudly -- not segfault,
    and not hand back a value.

    numba's caller cache key is callee-blind, so a warm-cached caller cache-hits
    even in a process where ``proxy_if_available`` took the absent (stub) path
    and registered no real body. Before the trap the unresolved extern linked to
    null and the call was a bare SIGSEGV (exit 139, zero diagnostics). The trap
    keeps the alias resolvable, but reaching it is not enough on its own: the
    ``RuntimeError`` it raises is raised inside a ``@cfunc``, where numba swallows
    it and returns zero, so the caller went on to print a wrong answer and exit 0.
    The cache guard now discards an entry whose alias resolves only to a trap, so
    the recompile reaches the stub's typing error instead -- the same outcome a
    cold cache gives.
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

    # Process 1: symbol available -> alias registered, caller cache written warm.
    env["HAVE_SYM"] = "1"
    r1 = subprocess.run(
        [sys.executable, str(pkg / "run_flip.py")],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    assert r1.returncode == 0, f"warm-up failed:\n{r1.stderr}"
    assert r1.stdout.strip() == "RESULT 6.0", r1.stdout

    # Process 2: symbol absent -> stub path, alias never registered. The warm
    # caller cache-hits and references the unregistered alias.
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
    # The cache anchor prepends blank lines so the generated @njit lands at the
    # function's co_firstlineno; a function defined above that line can't be
    # anchored (a negative prepend was silently clamped to 0, mis-anchoring it).
    # Decorating such a function must raise a clear error, not mis-anchor.
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


_JIT_ADDR_REASON = "`jit_addr` slot added to `FunctionModel` in numba 0.61"


def test_proxy_as_func_is_a_derive_where_the_jit_addr_slot_exists():
    """``.as_func`` is the handle that carries the propagating calling convention.

    Where ``FunctionModel`` has the ``jit_addr`` slot the binding hands out a ``DeriveWAP``
    typed as ``DeriveFunctionType``, which is what lets a first-class call select numba's own
    calling convention and unwind an exception out of the proxied body. Where the slot is
    absent there is nothing to populate, so the value stays the plain ``CompileResultWAP`` it
    has always been. Both arms are written out rather than skipped, because every numba in the
    supported range runs one of them.

    ``isinstance(..., CompileResultWAP)`` holds either way, which is why the second arm has to
    compare the exact type: ``DeriveWAP`` subclasses it.
    """
    @proxy(float64(float64))
    def scaled(x):
        return 2.0 * x

    as_func = scaled.as_func
    assert isinstance(as_func, CompileResultWAP)
    if jit_addr_supported():
        assert isinstance(as_func, DeriveWAP)
        assert isinstance(typeof(as_func), DeriveFunctionType)
        assert as_func.jit_address != 0, "the callconv entry point is what fills `jit_addr`"
    else:
        assert type(as_func) is CompileResultWAP
        assert type(typeof(as_func)) is FunctionType


@pytest.mark.skipif(numba_version < 61, reason=_JIT_ADDR_REASON)
def test_proxy_as_func_as_an_njit_argument_propagates_a_raising_body():
    """An exception raised inside a proxied body reaches the caller through ``.as_func``.

    The caller's signature is inferred, so numba types the parameter from the value handed in
    and sees ``DeriveFunctionType``. A plain ``CompileResultWAP`` in the same position carries
    no entry point that could unwind, so the call returns a zero-filled ``0.0`` and reports the
    failure only as an unraisable on stderr.
    """
    @njit
    def call_through(f, x):
        return f(x)

    assert call_through(aux_raises.as_func, -3.0) == -2.0
    with pytest.raises(ValueError, match="proxy boom"):
        call_through(aux_raises.as_func, 1.0)


@pytest.mark.skipif(numba_version < 61, reason=_JIT_ADDR_REASON)
def test_proxy_as_func_as_a_jitted_constant_propagates_a_raising_body():
    """The same contract on the constant path, which is lowered separately.

    An argument is typed on the way in and unboxed; a constant is resolved at compile time and
    lowered by ``lower_constant``, so the two reach ``jit_addr`` through different code and one
    can regress while the other stays green. This path is also what the caching behaviour rides
    on: declaring the entry point symbolically rather than baking an address is what lets a
    ``cache=True`` caller of a constant-reached ``.as_func`` be cached at all.
    """
    as_func = aux_raises.as_func

    @njit
    def call_constant(x):
        return as_func(x)

    assert call_constant(-3.0) == -2.0
    with pytest.raises(ValueError, match="proxy boom"):
        call_constant(1.0)


def test_proxy_dispatcher_call_still_discards_a_raising_body():
    """The binding's other handle keeps its old semantics, on every numba version.

    Calling the dispatcher goes through the proxied body's cfunc wrapper, which numba documents
    as not supporting exceptions: it discards the exception, zero-fills the return value and
    reports the failure only as an unraisable on stderr. So one binding carries two handles that
    disagree, and the asymmetry is worth pinning rather than assuming away.

    The unraisable arrives as a ``PytestUnraisableExceptionWarning``, handled the same way as for
    ``test_a_foreign_wrapper_reached_from_jitted_scope_still_discards`` in
    ``test/utils/test_derive_wap.py``: the returned value is what is asserted. Its text is not,
    because the ``repr`` of the context object numba names in it varies by numba version.
    """
    assert aux_raises(-3.0) == -2.0
    assert aux_raises(1.0) == 0.0, "the cfunc wrapper stopped zero-filling a discarded raise"


def test_proxy_as_func_declared_as_a_plain_function_type_still_discards():
    """Crossing the Python boundary into a *declared* plain ``FunctionType`` degrades.

    ``DeriveFunctionType.can_convert_to`` permits the conversion, so an explicitly typed ``njit``
    written before the derive type existed keeps accepting a binding, and it documents that the
    value arrives on the C convention: the exception is discarded exactly as it was. The sibling
    suite covers that degradation for a ``cres`` value; a ``@proxy`` binding reaches the same
    declaration through ``.as_func`` and is covered here.

    Like the dispatcher-call test above, the discarded exception surfaces as an unraisable and
    only the returned value is asserted.
    """
    @njit(float64(FunctionType(float64(float64)), float64))
    def call_through_declared_plain(f, x):
        return f(x)

    assert call_through_declared_plain(aux_raises.as_func, -3.0) == -2.0
    assert call_through_declared_plain(aux_raises.as_func, 1.0) == 0.0


@pytest.mark.skipif(numba_version < 61, reason=_JIT_ADDR_REASON)
def test_proxy_as_func_mixed_with_a_numba_native_wrapper_fails_to_unify():
    """Characterization of a narrowing: a heterogeneous tuple of function values is refused.

    numba unifies a tuple's element types in ``unified_function_type`` with a bare
    class-identity comparison, which runs before any of numbox's conversions get a say, so a
    ``DeriveFunctionType`` element beside a plain ``FunctionType`` one trips a message-less
    ``AssertionError`` out of ``numba/core/utils.py``. The same tuple returned a value while
    ``.as_func`` was a plain ``CompileResultWAP``, so this is a real loss of reach and it is
    pinned here rather than left to surprise a caller. Homogeneous tuples are unaffected.

    Why the two types cannot simply be made to compare equal is worked through in
    ``test/utils/test_derive_wap.py::test_the_derive_type_stays_distinct_from_the_plain_function_type``.

    numba's assert carries no message, so the frame it was raised from is checked instead of any
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

    with pytest.raises(AssertionError) as raised:
        use_pair((aux_raises.as_func, foreign), -3.0)
    assert raised.traceback[-1].name == "unified_function_type", (
        f"the refusal moved out of numba's function-type unification: {raised.traceback[-1].name}"
    )


if __name__ == "__main__":
    collect_and_run_tests(__name__)


def test_proxy_jit_flags_dispatch_to_their_own_body():
    """Two decorations of one body differing only in jit flags must not collide.

    The flags govern the machine code ``njit(sig, **jit_options)`` emits, but
    they appear in neither the module+qualname+signature identity nor the body
    fingerprint -- the body source is byte-identical here. Both decorations
    therefore mapped to a single alias and llvmlite's last-writer-wins
    ``add_symbol`` rebound every caller to whichever compiled last: a jitted
    caller of the ``numpy`` error-model binding ran the ``python`` one, so
    ``1.0 / 0.0`` returned 0.0 instead of inf, exit 0, with the
    ``ZeroDivisionError`` swallowed at the cfunc boundary.
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
