import hashlib
import inspect
from llvmlite import binding as ll
from llvmlite import ir
from numba import cfunc, njit
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.core.types.function_type import CompileResultWAP
from numba.core.typing.templates import Signature
from numba.extending import intrinsic, overload
from types import FunctionType as PyFunctionType
from typing import List, Optional, Tuple

from numbox.utils.fingerprint import (
    _Unfingerprintable, _fingerprint_function, _fingerprint_function_best_effort,
)
from numbox.utils.standard import make_params_strings


def make_proxy_name(name):
    return f'__{name}'


def _body_fingerprint(func):
    """Content fingerprint of ``func``'s body, for alias disambiguation.

    Reuses the deep walker so bytecode, constants, default arguments, closure
    cell values and referenced-global values all count. The common numbox
    binding body references the ``@intrinsic`` ``_call_lib_func``, which has no
    canonical form, so the strict walker raises; the best-effort walker then
    still captures the constants/closure/defaults/globals it *can* canonicalize
    (substituting an opaque type placeholder for the rest). Two bodies that
    differ only in a captured value -- a factory over per-instance C symbol
    names, or a literal-only redefinition -- therefore get distinct aliases,
    not one collapsed to bytecode alone.
    """
    try:
        return _fingerprint_function(func, set())
    except (_Unfingerprintable, RecursionError):
        return _fingerprint_function_best_effort(func)


def _stable_cfunc_alias(func, main_sig):
    """Deterministic, process-stable LLVM symbol name for ``func``'s cfunc wrapper.

    numba mangles the wrapper name (``fndesc.llvm_cfunc_wrapper_name``) with a
    process-local unique-id abi-tag (``v<N>`` from ``FunctionIdentity.unique_id``,
    an ``itertools.count``). That tag is not part of the numba cache key, so two
    processes give the same function different wrapper names. Persisting a
    reference to that name — in this proxy, or in any ``cache=True`` caller that
    inlines the ``inline='always'`` proxy — lets concurrently-built caches pair a
    body object defining ``v<Na>`` with a caller referencing ``v<Nb>``, which
    aborts on load with ``LLVM ERROR: Symbol not found: cfunc...``. Referencing
    this deterministic alias instead (resolved per-process via
    ``llvmlite.binding.add_symbol``) keeps cached references valid
    across processes.

    The alias folds a body fingerprint alongside ``module + qualname + sig``:
    without it, two different bodies sharing one identity (factory-made
    same-qualname closures, in-process redefinition, or fork twins on a shared
    cache) collapse onto one alias, and the last-writer-wins ``add_symbol``
    silently rebinds callers to the wrong body. Folding the body gives each body
    its own alias. A consequence is that a body/signature change renames the
    alias, so a warm ``cache=True`` caller in another file (numba's cache key is
    callee-blind) references a symbol this process never registered -- clear the
    numba cache after such a change. The one variant that leaves the alias
    unchanged, an absent ``proxy_if_available`` binding, is covered by a
    diagnostic trap (see ``_register_absent_alias_trap``) so it surfaces a named
    error rather than a null-pointer call.
    """
    raw = (
        f"{func.__module__ or ''}.{func.__qualname__}.{main_sig}."
        f"{_body_fingerprint(func)}"
    ).encode("utf-8")
    safe_name = "".join(c if c.isascii() and c.isalnum() else "_" for c in func.__name__)
    return f"numbox_pxy_{safe_name}_{hashlib.sha256(raw).hexdigest()[:16]}"


def _call_proxied_alias(context, builder, main_sig, alias_name, args):
    """Emit a call to the proxied body's process-stable cfunc alias.

    Factored out of the generated codegen so the exec'd template stays short:
    every line above its ``@njit`` raises the minimum source line a ``@proxy``
    function may occupy (the ``co_firstlineno`` cache anchor).
    """
    func_ty_ll = ir.FunctionType(
        context.get_data_type(main_sig.return_type),
        [context.get_data_type(arg) for arg in main_sig.args],
    )
    f = cgutils.get_or_insert_function(builder.module, func_ty_ll, alias_name)
    return builder.call(f, args)


# The cfunc traps below must outlive the add_symbol registrations that publish
# their addresses, so keep a process-lifetime reference to each.
_PROXY_TRAP_KEEPALIVE = []


def _register_absent_alias_trap(func, main_sig):
    """Register a diagnostic trap under the alias ``proxy(sig)(func)`` would use.

    A ``cache=True`` caller compiled when the binding was present bakes that
    alias into its cached object; numba's cache key is callee-blind, so the
    caller cache-hits even in a process where ``proxy_if_available`` took the
    absent (stub) path and registered no real body. With no symbol there the
    baked extern call is a null-pointer jump -- a bare SIGSEGV with zero
    diagnostics. Registering a cfunc that raises a named ``RuntimeError`` under
    the same alias turns that into a clear message on stderr instead. The trap
    matches ``main_sig``'s arity with plain positional parameters (the cfunc
    wrapper the caller invokes is positional, even for ``Omitted`` bindings).
    """
    alias = _stable_cfunc_alias(func, main_sig)
    msg = (
        f"numbox @proxy binding {func.__name__!r} is not available in the loaded library "
        f"(C symbol missing), but a cache=True caller compiled when it was present is "
        f"calling it -- clear the numba cache (NUMBA_CACHE_DIR) and rebuild."
    )
    trap_params = ", ".join(f"_a{i}" for i in range(len(main_sig.args)))
    trap_ns = {"cfunc": cfunc, "main_sig": main_sig, "_trap_msg": msg}
    exec(  # nosec B102 - internal codegen of a fixed-shape trap body
        f"def _trap({trap_params}):\n"
        f"    raise RuntimeError(_trap_msg)\n"
        f"_trap_cfunc = cfunc(main_sig)(_trap)\n",
        trap_ns,
    )
    _PROXY_TRAP_KEEPALIVE.append(trap_ns["_trap_cfunc"])
    ll.add_symbol(alias, trap_ns["_trap_cfunc"].address)


def proxy(sig, jit_options: Optional[dict] = None):
    """ Create a proxy for the decorated function `func` with the given signature(s) `sig`.

    The original function `func` will be eagerly JIT-compiled with the given signature(s).
    A proxy with the name `func_proxy_name` will be created to call `func` in the LLVM scope.
    The original function's variable will be bound to the proxy, i.e., calling the decorated
    function will call the proxy.

    The proxy is a JIT-compiled wrap that invokes the intrinsic that *declares* the `func`
    and calls it with the original arguments. Declaration instructions are relatively cheap
    to statically link into (potential) caller's LLVM code, which is the main motivation behind
    this decorator.

    Machine code for `func` can be cached when so specified in `jit_options`, in which case its
    JIT-compilation will load the `func` into the LLVM scope. Caching option is the other major
    motivation for this decorator, without the need to cache one can avoid static linking
    of the callee's LLVM code into the caller's by simply ignoring the former.

    In case when more than one signature is provided as the `sig` parameter, it is assumed
    that the first signature is the 'main' one while the other ones are supplied to
    allow for the `Omitted` types with default values for (some of) the parameters.

    The returned dispatcher also exposes ``.as_func``: a ``CompileResultWAP``
    for the main signature. Cacheable as a called jitted function (via the
    dispatcher); passable as a function-type argument (via ``.as_func``).

    See tests for some examples of the use cases.
    """
    main_sig = isinstance(sig, Signature) and sig or isinstance(sig, (List, Tuple)) and sig[0]
    jit_options = isinstance(jit_options, dict) and jit_options or {}
    jit_opts = jit_options.copy()
    jit_opts.update(jit_opts, inline='always')

    def wrap(func):
        assert isinstance(func, PyFunctionType)
        func_jit = njit(sig, **jit_options)(func)
        cres = func_jit.get_compile_result(main_sig)
        # Register a process-stable alias for the body's cfunc wrapper and reference
        # that instead of numba's process-local ``v<uid>`` name (see _stable_cfunc_alias).
        cfunc_alias = _stable_cfunc_alias(func, main_sig)
        ll.add_symbol(cfunc_alias, cres.library.get_pointer_to_function(cres.fndesc.llvm_cfunc_wrapper_name))
        func_args_str, func_names_args_str = make_params_strings(func)
        func_proxy_name = make_proxy_name(func.__name__)
        # The alias resolution lives in _call_proxied_alias so this generated
        # source stays short: every line above the @njit raises the minimum source
        # line a @proxy function may occupy (the co_firstlineno cache anchor below).
        code_txt = f"""
@intrinsic
def _{func_proxy_name}(typingctx, {func_names_args_str}):
    def codegen(context, builder, signature, args):
        return _call_proxied_alias(context, builder, main_sig, "{cfunc_alias}", args)
    return main_sig, codegen

@njit(sig, **jit_opts)
def {func_proxy_name}({func_args_str}):
    return _{func_proxy_name}({func_names_args_str})
"""
        ns = {
            **inspect.getmodule(func).__dict__,
            **{
                'cgutils': cgutils, 'intrinsic': intrinsic, 'ir': ir, 'jit_opts': jit_opts, 'njit': njit,
                'sig': sig, 'main_sig': main_sig, '_call_proxied_alias': _call_proxied_alias
            }
        }
        if ns.get(func_proxy_name) is not None:
            raise ValueError(f"Name {func_proxy_name} in module {inspect.getmodule(func)} is reserved")
        # Anchor the wrapper at func's source file: prepend blank lines so the
        # wrapper's @njit decorator lands at func.__code__.co_firstlineno (the
        # user's @proxy decorator line). See docs/numbox.core.proxy.rst —
        # section "Cache-anchor mechanism" — for the design rationale + the
        # findsource-finds-@-in-docstring hazard this avoids.
        code_lines = code_txt.split('\n')
        njit_lineno_in_txt = next(
            i + 1 for i, line in enumerate(code_lines) if line.startswith('@njit(')
        )
        co_firstlineno = func.__code__.co_firstlineno
        if co_firstlineno < njit_lineno_in_txt:
            raise ValueError(
                f"@proxy function {func.__name__!r} is defined at line {co_firstlineno} of "
                f"{inspect.getfile(func)}, above the cache anchor's minimum line "
                f"{njit_lineno_in_txt}; the generated @njit cannot be anchored to its "
                f"co_firstlineno (a negative prepend would mis-anchor it). Move the "
                f"function further down in the file."
            )
        prepend = co_firstlineno - njit_lineno_in_txt
        prefixed = '\n' * prepend + code_txt
        code = compile(prefixed, inspect.getfile(func), mode='exec')
        exec(code, ns)  # nosec B102 - JIT codegen of internal source
        dispatcher = ns[func_proxy_name]
        dispatcher.as_func = CompileResultWAP(cres)
        return dispatcher
    return wrap


def proxy_if_available(lib, sig, jit_options: Optional[dict] = None):
    """Like ``proxy(sig, jit_options=...)``, but stubs out the wrapper if
    the C symbol matching ``func.__name__`` is absent from ``lib``.

    Use for binding sets that target multiple library versions where some
    symbols only exist in newer releases. Python callers get a stub that raises
    ``NotImplementedError`` instead of a confusing LLVM link error at call
    time; ``@njit`` callers get a ``TypingError`` naming the binding and the
    missing-symbol cause at typing time (the untyped stub would otherwise
    surface an untyped-global failure that names the binding but not why it
    is unusable). Parallel to ``cres_if_available`` in
    :mod:`numbox.utils.highlevel`.

    The stub does NOT expose ``.as_func`` — a function-value handle is
    meaningless without an underlying jitted body, and a stub one would
    have to either raise on attribute access (ugly) or pretend to be a
    function value (worse). Callers that pass ``.as_func`` to function-
    type arguments must guard the access::

        if hasattr(my_binding, "as_func"):
            use(my_binding.as_func)
    """
    def _(func):
        if hasattr(lib, func.__name__):
            return proxy(sig, jit_options=jit_options)(func)

        name = func.__name__
        main_sig = sig if isinstance(sig, Signature) else sig[0]
        # A warm cache=True caller from a process where the binding WAS present
        # will cache-hit and call the alias; register a loud trap under it so that
        # is a named error rather than a null-pointer SIGSEGV. See the helper.
        _register_absent_alias_trap(func, main_sig)

        def stub(*args, **_kwargs):
            raise NotImplementedError(f"{name} is not available")
        stub.__name__ = make_proxy_name(name)
        stub.__qualname__ = func.__qualname__
        stub.__doc__ = func.__doc__

        # The stub is untyped, so a bare @njit call to it fails typing with an
        # untyped-global error that names the binding but not why it is unusable.
        # Register an @overload that raises a clear, named TypingError instead.
        @overload(stub)
        def _unavailable(*args, **kwargs):
            raise TypingError(
                f"{name} is not available in the loaded library "
                f"(C symbol missing)"
            )
        return stub
    return _
