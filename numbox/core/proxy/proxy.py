import hashlib
import inspect
import struct
import threading
import warnings
from llvmlite import binding as ll
from llvmlite import ir
from numba import cfunc, njit
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.core.typing.templates import Signature
from numba.extending import intrinsic, overload
from types import FunctionType as PyFunctionType
from typing import List, Optional, Tuple

from numbox.core.cache_key import _keyed, numbox_njit
from numbox.core.configurations import _PROXY_CACHE_STRICT_ENV, _strict_cache_mode
from numbox.utils.derive_wap import DeriveWAP
from numbox.utils.fingerprint import (
    _ABSENT_ALIASES, _Unfingerprintable, _fingerprint_function, _fingerprint_function_best_effort,
    _opaque_values_match,
)
from numbox.utils.standard import make_params_strings


_ALIAS_PREFIX = "numbox_pxy_"


def make_proxy_name(name):
    return f'__{name}'


def _body_fingerprint(func, opaque=None):
    """Content fingerprint of ``func``'s body, for alias disambiguation.

    The function reuses the deep walker, so the bytecode, the constants, the default arguments, the closure cell
    values and the values of referenced globals all enter the fingerprint. The common numbox binding body references
    the ``@intrinsic`` ``_call_lib_func``, which has no canonical form, so the strict walker raises. The best-effort
    walker then still captures the constants, closure cells, defaults and globals that it *can* canonicalize, and
    substitutes an opaque type placeholder for the rest. Two bodies that differ only in a captured value therefore
    receive distinct aliases, rather than one alias computed from the bytecode alone. Examples are a factory over
    per-instance C symbol names, and a redefinition that changes only a literal.

    ``opaque`` collects the values that the best-effort walker could not canonicalize. Two different bodies can
    still produce one fingerprint only if they differ in those values, so ``_publish_cfunc_alias`` compares them to
    distinguish a second body from the same body registered twice.
    """
    try:
        return _fingerprint_function(func, set())
    except (_Unfingerprintable, RecursionError):
        return _fingerprint_function_best_effort(func, opaque)


def _stable_cfunc_alias(func, main_sig, jit_options=None, opaque=None):
    """Deterministic, process-stable LLVM symbol name for ``func``'s cfunc wrapper.

    numba mangles the wrapper name (``fndesc.llvm_cfunc_wrapper_name``) with a process-local unique-id abi tag:
    ``v<N>`` from ``FunctionIdentity.unique_id``, an ``itertools.count``. That tag is not part of the numba cache key,
    so two processes give the same function different wrapper names. A reference to that name could be persisted in
    this proxy, or in any ``cache=True`` caller that inlines the ``inline='always'`` proxy. Concurrently built caches
    could then pair a body object that defines ``v<Na>`` with a caller that references ``v<Nb>``, and the load would
    abort with ``LLVM ERROR: Symbol not found: cfunc...``. The proxy references this deterministic alias instead, and
    ``llvmlite.binding.add_symbol`` resolves the alias in each process, so cached references remain valid across
    processes.

    The alias folds a body fingerprint together with ``module + qualname + sig``. Without the fingerprint, two different
    bodies that share one identity would collapse onto one alias: closures of the same qualname made by a factory, an
    in-process redefinition, or processes related by ``fork()`` on a shared cache. ``add_symbol`` is last-writer-wins,
    so it would rebind callers to the wrong body and issue no warning. Folding the body gives each body its own alias.

    As a consequence, a change to the body or the signature renames the alias. numba's cache key does not
    distinguish callees, so a warm ``cache=True`` caller in another file then references a symbol that this process
    never registered. The load-time guard repairs this automatically: it discards that one cache entry, recompiles
    it, and warns once with :class:`StaleProxyCacheWarning`. No manual cache clearing is needed. Set
    ``NUMBOX_PROXY_CACHE_STRICT`` to make the guard raise :class:`StaleProxyCacheError` before the repair instead.

    The one variant that leaves the alias unchanged is an absent ``proxy_if_available`` binding. A diagnostic trap
    covers it (see ``_register_absent_alias_trap``). The guard treats such an alias as stale too, so a warm caller gets
    the same typing error that a cold cache gives, rather than the zero that the trap returns once its cfunc wrapper
    discards the exception.

    The resolved ``jit_options`` are folded too. They govern the machine code that ``njit(sig, **jit_options)``
    emits, but they appear nowhere in the identity above. Without them, two decorations of one body that differ only
    in a flag, such as ``error_model``, ``fastmath`` or ``boundscheck``, would collapse onto a single alias.
    ``add_symbol`` would then send every caller to whichever decoration compiled last, and no warning would be
    issued. A caller of the ``numpy`` error-model binding would run the ``python`` one, so a division by zero would
    return 0.0 instead of inf, and the exception would be discarded at the cfunc boundary.
    """
    raw = (
        f"{func.__module__ or ''}.{func.__qualname__}.{main_sig}."
        f"{_body_fingerprint(func, opaque)}."
        f"{sorted((jit_options or {}).items(), key=repr)!r}"
    ).encode("utf-8")
    safe_name = "".join(c if c.isascii() and c.isalnum() else "_" for c in func.__name__)
    return f"{_ALIAS_PREFIX}{safe_name}_{hashlib.sha256(raw).hexdigest()[:16]}"


def _call_proxied_alias(context, builder, main_sig, alias_name, args):
    """Emit a call to the proxied body's process-stable cfunc alias.

    The call is emitted here, outside the generated codegen, so that the exec'd template remains short: every line
    above its ``@njit`` raises the minimum source line that a ``@proxy`` function may occupy (the ``co_firstlineno``
    cache anchor).
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

# _ABSENT_ALIASES holds the aliases that currently represent an ABSENT binding. A trap registration makes an
# alias resolve although no body is published under it, so the presence of the symbol alone no longer means
# "safe to call"; see _stale_proxy_aliases. The set is defined in numbox.utils.fingerprint and imported above,
# because the walker there has to refuse to fold a retired alias into the fingerprint of a body that captured
# it, and this module already imports that module. The set is only ever mutated in place, so both modules see
# the same set.

# Maps each alias to the address published under it and to its witness: the opaque values of its fingerprint.
# The witness lets _publish_cfunc_alias detect a second body that reaches the same alias, instead of letting
# that body rebind every caller compiled against the first with no warning. A witness of None marks an alias
# that a trap published rather than a proxied body, and no body can match it.
_PUBLISHED_ALIASES = {}


# _ALIAS_LOCK serializes the read-modify-write sequences over the two registries above, _ABSENT_ALIASES and
# _PUBLISHED_ALIASES. _publish_cfunc_alias reads the count that names a colliding body from _PUBLISHED_ALIASES
# and writes the name back into it. Two threads that both read before either writes would give one name to two
# bodies. _register_absent_alias_trap reads _PUBLISHED_ALIASES to decide, and writes the alias several
# statements later. A body that publishes in between would be overwritten by a trap, with no warning. Both
# functions run from plain Python at decoration time, after numba has released its compiler lock, so nothing
# else serializes them.
#
# The trap compiles a cfunc between its decision and its write. It therefore reads the registry again under the
# lock, instead of holding the lock across the compile: the read that decides and the writes that act on it must
# be atomic, and the compile need not be.
_ALIAS_LOCK = threading.Lock()


def _a_body_answers(alias):
    """Return whether a proxied body is published under ``alias``, rather than a trap or nothing."""
    published, witness = _PUBLISHED_ALIASES.get(alias, (None, None))
    return published is not None and witness is not None


class AliasCollisionWarning(RuntimeWarning):
    """Two different bodies reached one alias, so the alias stopped identifying a body."""


def _publish_cfunc_alias(alias, address, opaque):
    """Publish ``address`` under ``alias``, and return the alias that callers are to reference.

    ``ll.add_symbol`` is a silent, process-global, last-writer-wins assignment. A second body that reaches an alias
    the first body already holds would therefore rebind every caller compiled against the first. The result would be
    a wrong numeric answer with no diagnostic, determined by which body was constructed first.

    Reaching an alias twice is usually not a collision. The same body is registered again whenever a module is
    reloaded or a factory is called twice with equal arguments. Both registrations then name the same code, so
    last-writer-wins is exactly right. ``opaque`` distinguishes the two cases: it holds the values that the
    fingerprint could not canonicalize. Two bodies that fingerprint alike are the same body except in those values.
    If the values match, one body was compiled twice, and the alias is re-pointed.

    The values differ only if the fingerprint could not distinguish two bodies, which happens for values that have
    no process-stable identity to fold (see ``_ctypes_func_key``). Two conditions must then hold at once. First, the
    new body needs a name of its own, so it receives a process-local name, and both bodies stay callable in this
    process. Second, the shared name no longer identifies a body, because the next process may construct the two
    bodies in the other order. The shared name is therefore retired into ``_ABSENT_ALIASES``. Warm callers of either
    body are then discarded and recompiled, rather than bound to whichever body holds the name at the time. Both
    bodies lose cross-process caching, which is the cost of a name that does not identify what it names.

    The name given to the new body is retired too, and it must be, for both bodies to lose cross-process caching.
    The name is built from a count of the aliases this process has published, so it carries no information about
    the body it names. The same body receives a different name in a process that published a different number of
    aliases before reaching it. The same name belongs to the other body in a process that builds the pair in the
    other order. A name derived from the content is not available here, by construction: these two bodies share one
    alias because no process-stable fingerprint separates them. The name is therefore process-local in the strict
    sense, valid for calls compiled in this process and for nothing else. A warm caller that references it was
    compiled in another process, and this process cannot tell which body the name identified there. Retiring the
    name discards that caller. The name it collided with receives the same treatment, for the same reason.

    The retirement is permanent for the life of the process. The name was retired because of a property of the two
    bodies, not because of the order in which this process built them. A second registration of either body
    therefore does not make the name identify a body again. If the body that holds the name registers again, it
    re-points the symbol; if the other body registers again, it only receives another name. In either case, another
    process may still build the two bodies in the opposite order, and the retirement protects the warm caller cached
    against the name in that process. The re-registration branch above therefore re-points ``_PUBLISHED_ALIASES``
    and the symbol, and leaves ``_ABSENT_ALIASES`` unchanged: no alias is ever removed from that set. A first
    publication has nothing to remove either. Every member of the set already has a ``_PUBLISHED_ALIASES`` entry, so
    an alias that nobody has published cannot already be retired.

    The other possible holder of the alias is a trap. The warning then names the trap, rather than reporting a second
    body that does not exist. A trap records a witness that no body can match, so the body receives a name of its
    own, and the alias stays retired. The process that cached a warm caller of the alias may be one in which the
    binding is missing, and that caller must reach the trap's named error there.
    """
    with _ALIAS_LOCK:
        published, witness = _PUBLISHED_ALIASES.get(alias, (None, None))
        if published is None or (witness is not None and _opaque_values_match(witness, opaque)):
            _PUBLISHED_ALIASES[alias] = (address, tuple(opaque))
            ll.add_symbol(alias, address)
            return alias
        _ABSENT_ALIASES.add(alias)
        distinct = f"{alias}_c{len(_PUBLISHED_ALIASES)}"
        _ABSENT_ALIASES.add(distinct)
        _PUBLISHED_ALIASES[distinct] = (address, tuple(opaque))
        ll.add_symbol(distinct, address)
    if witness is None:
        reason = (
            f"an absent-binding trap already answers to the alias {alias}, so this body is "
            f"published as {distinct} instead and the alias stays retired. A warm caller that "
            f"reaches that name has to find the trap and its named error, because the process "
            f"that cached it may be one where the binding is missing."
        )
    else:
        reason = (
            f"two @proxy bodies fingerprint alike and both reached the alias {alias}; it names "
            f"neither of them now, so callers of both recompile in every process. This one is "
            f"published as {distinct} instead. A body whose only difference is a value with no "
            f"process-stable identity, such as a ctypes pointer built from a raw address, "
            f"cannot be told apart by a fingerprint that has to mean the same thing next run."
        )
    warnings.warn(reason, AliasCollisionWarning, stacklevel=3)
    return distinct


def _register_absent_alias_trap(func, main_sig, jit_options=None):
    """Register a diagnostic trap under the alias that ``proxy(sig)(func)`` would use.

    A ``cache=True`` caller compiled while the binding was present embeds that alias in its cached object. numba's
    cache key does not distinguish callees, so the caller hits the cache even in a process in which
    ``proxy_if_available`` took the absent (stub) path and registered no real body. With no symbol under the alias,
    the embedded extern call is a jump to a null pointer: a bare SIGSEGV with no diagnostic at all. A cfunc that
    raises a named ``RuntimeError``, registered under the same alias, turns that crash into a clear message on
    stderr. The trap matches the arity of ``main_sig`` with plain positional parameters, because the cfunc wrapper
    that the caller invokes is positional, even for ``Omitted`` bindings.

    A body that is already published under this alias in this process keeps the alias. The trap raises inside a
    cfunc wrapper, which discards the exception and returns a zeroed value. A trap installed over a working body
    would therefore replace a correct answer with a wrong one, and no warning would be issued. That is the failure
    this registry exists to prevent, reached from the opposite direction. A caller that reaches the alias here calls
    a binding that is present. A process in which the binding is absent registers the trap under an alias that no
    body holds, so this rule does not affect it.

    The check for a body under the alias runs twice, and the second answer decides. Compiling the trap takes long
    enough that a body publishing on another thread publishes during the compile routinely, not rarely, and the
    first answer was read before the compile began. The trap is therefore compiled speculatively, and the registry is
    read again under ``_ALIAS_LOCK``, together with the writes that act on that reading. A trap whose alias was taken
    while it compiled is dropped.
    """
    alias = _stable_cfunc_alias(func, main_sig, jit_options)
    # The first, unlocked check is cheap; it exists only so that the common case does not compile a trap it discards.
    yielded_to_body = _a_body_answers(alias)
    if not yielded_to_body:
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
        with _ALIAS_LOCK:
            yielded_to_body = _a_body_answers(alias)
            if not yielded_to_body:
                _PROXY_TRAP_KEEPALIVE.append(trap_ns["_trap_cfunc"])
                _ABSENT_ALIASES.add(alias)
                ll.add_symbol(alias, trap_ns["_trap_cfunc"].address)
                # Record the trap under a witness that no body can match, so that a real body that later reaches
                # this alias is detected, rather than silently taking a name that a trap holds.
                _PUBLISHED_ALIASES[alias] = (trap_ns["_trap_cfunc"].address, None)
    if yielded_to_body:
        warnings.warn(
            f"the @proxy binding {func.__name__!r} is absent from this library but a body is already "
            f"published under the alias {alias} in this process, so the alias keeps naming that body "
            f"and no trap was installed. A caller compiled here calls the body that is present.",
            AliasCollisionWarning,
            stacklevel=3,
        )


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

    The returned dispatcher also exposes ``.as_func``: a first-class function value
    for the main signature. The binding is cacheable as a called jitted function through the
    dispatcher, and it can be passed as a function-type argument through ``.as_func``.

    ``.as_func`` is a ``DeriveWAP`` typed as ``DeriveFunctionType``,
    so an exception raised inside the proxied body propagates out of a first-class call made
    through ``.as_func`` instead of being discarded; see :mod:`numbox.utils.derive_wap`.
    Calling the dispatcher itself still discards the exception: that path goes through the
    body's cfunc wrapper, which discards it and zero-fills the return value. Passing
    ``.as_func`` from Python into a parameter *declared* as a plain ``FunctionType`` also
    discards the exception, as ``DeriveFunctionType.can_convert_to`` documents.

    Reaching ``.as_func`` as a compile-time constant from a ``cache=True`` jitted caller makes
    that caller cacheable. Its cached binary then binds the proxied body's machine code. After
    an edit to the body, such a caller does not reliably run the edited body. Which body it runs
    is not single-valued. The numba cache therefore has to be cleared after such an edit, because
    numba cannot be relied on to detect it.
    The stale-alias guard below does not cover that caller, because a constant reference emits
    no alias for it to check; for the same reason it does not catch a ``proxy_if_available``
    binding that has since gone absent, where such a caller returns the vanished binding's value
    or segfaults.

    See tests for some examples of the use cases.
    """
    main_sig = isinstance(sig, Signature) and sig or isinstance(sig, (List, Tuple)) and sig[0]
    jit_options = isinstance(jit_options, dict) and jit_options or {}
    jit_opts = jit_options.copy()
    jit_opts.update(jit_opts, inline='always')

    def wrap(func):
        assert isinstance(func, PyFunctionType)
        with _keyed(func):
            func_jit = njit(sig, **jit_options)(func)
        cres = func_jit.get_compile_result(main_sig)
        # Register a process-stable alias for the body's cfunc wrapper and reference
        # that instead of numba's process-local ``v<uid>`` name (see _stable_cfunc_alias).
        opaque = []
        cfunc_alias = _publish_cfunc_alias(
            _stable_cfunc_alias(func, main_sig, jit_options, opaque),
            cres.library.get_pointer_to_function(cres.fndesc.llvm_cfunc_wrapper_name),
            opaque,
        )
        func_args_str, func_names_args_str = make_params_strings(func)
        func_proxy_name = make_proxy_name(func.__name__)
        # _call_proxied_alias resolves the alias, so that this generated source remains short: every line above the
        # @njit raises the minimum source line that a @proxy function may occupy (the co_firstlineno cache anchor
        # below).
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
        # The namespace binds the template's name njit to numbox_njit, so numbox keys the wrapper's cache as it keys
        # the body's. The template keeps its @njit( line, which the anchor below measures.
        ns = {
            **inspect.getmodule(func).__dict__,
            **{
                'cgutils': cgutils, 'intrinsic': intrinsic, 'ir': ir, 'jit_opts': jit_opts, 'njit': numbox_njit,
                'sig': sig, 'main_sig': main_sig, '_call_proxied_alias': _call_proxied_alias
            }
        }
        if ns.get(func_proxy_name) is not None:
            raise ValueError(f"Name {func_proxy_name} in module {inspect.getmodule(func)} is reserved")
        # Anchor the wrapper at func's source file: prepend blank lines, so that the wrapper's @njit decorator is
        # placed at func.__code__.co_firstlineno, the line of the user's @proxy decorator. The section "Cache-anchor
        # mechanism" of docs/numbox.core.proxy.rst gives the design rationale. It also describes the hazard that this
        # avoids, in which findsource matches an @ inside a docstring.
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
        dispatcher.as_func = DeriveWAP(cres)
        # Tag the dispatcher with its process-stable alias, so that the fingerprint walker can identify a @proxy
        # binding by that alias instead of recursing into its wrapper's @intrinsic, which has no canonical form.
        # Without the tag, a callback that calls a proxied binding cannot be fingerprinted, and its cache digest
        # degrades to a coarse fallback with no warning.
        dispatcher._numbox_proxy_alias = cfunc_alias
        # Tag the function value with the alias too. A body that captures `.as_func` rather than the dispatcher
        # reaches a wrapper that carries only a compiled address and a mangled name with a per-process counter in it.
        # Without the tag, the walker has nothing process-stable to fold, and falls back on the type name, which every
        # first-class function value in the process shares.
        dispatcher.as_func._numbox_proxy_alias = cfunc_alias
        # Store the body's dispatcher as well, because explain_cache_key reads its cache.
        dispatcher.as_func._numbox_dispatcher = func_jit
        return dispatcher
    return wrap


def proxy_if_available(lib, sig, jit_options: Optional[dict] = None):
    """Like ``proxy(sig, jit_options=...)``, but stubs out the wrapper if ``lib`` lacks the C symbol ``func.__name__``.

    Use it for a binding set that targets several versions of a library, when some symbols exist only in newer
    releases. If the symbol is absent, a Python caller receives a stub that raises ``NotImplementedError`` at call
    time, instead of an LLVM link error that is hard to diagnose. At typing time, an ``@njit`` caller receives a
    ``TypingError`` that names the binding and the missing symbol as the cause. Otherwise the untyped stub would
    produce an untyped-global failure that names the binding but not the reason it is unusable. The function
    parallels ``cres_if_available`` in :mod:`numbox.utils.highlevel`.

    The stub does *not* expose ``.as_func``, because a function-value handle is meaningless without an underlying
    jitted body. A stub handle would have to raise on attribute access or pretend to be a function value. The first
    is awkward, and the second is worse. A caller that passes ``.as_func`` to a function-type argument must guard the
    access::

        if hasattr(my_binding, "as_func"):
            use(my_binding.as_func)

    A present binding's ``.as_func`` follows the contract of ``proxy`` above, including its type and exception
    semantics.
    """
    def _(func):
        if hasattr(lib, func.__name__):
            return proxy(sig, jit_options=jit_options)(func)

        name = func.__name__
        main_sig = sig if isinstance(sig, Signature) else sig[0]
        # A warm cache=True caller from a process in which the binding was present hits the cache and calls the
        # alias. Register a trap under the alias, so that the call produces a named error rather than a null-pointer
        # SIGSEGV. See the helper.
        _register_absent_alias_trap(func, main_sig, jit_options)

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


class StaleProxyCacheWarning(RuntimeWarning):
    """A cached numba function referenced a ``@proxy`` alias this process never registered.

    numbox discarded the entry and recompiled it. To make the warning fatal, filter it programmatically with
    ``warnings.filterwarnings("error", category=StaleProxyCacheWarning)``, or, under pytest, with
    ``-W error::numbox.core.proxy.proxy.StaleProxyCacheWarning``. The interpreter's own ``-W`` cannot name this
    class: CLI warning filters are resolved before the import system can reach numbox, so the category is rejected
    as an invalid module name. The blanket ``-W error::RuntimeWarning`` works.
    """


class UnvalidatedProxyCacheWarning(RuntimeWarning):
    """A cached payload could not be read, so it was loaded without being checked for stale ``@proxy`` aliases.

    The warning is issued once per process, because the condition is a property of the environment, not of one entry.
    """


class StaleProxyCacheError(RuntimeError):
    """Strict mode found a stale ``@proxy`` alias and refused to repair it.

    Raised in place of :class:`StaleProxyCacheWarning` when ``NUMBOX_PROXY_CACHE_STRICT`` is set. The guard aborts
    the load *before* the discard and recompile, so the stale entry stays on disk for inspection instead of being
    replaced. The default repair does the opposite: it overwrites the entry silently. To recover, clear the numba
    cache (``NUMBA_CACHE_DIR``) or unset ``NUMBOX_PROXY_CACHE_STRICT``.
    """


class UnvalidatedProxyCacheError(RuntimeError):
    """Strict mode could not read a cache payload and refused to load it unchecked.

    Raised in place of :class:`UnvalidatedProxyCacheWarning` when ``NUMBOX_PROXY_CACHE_STRICT`` is set and
    unpacking the payload raises. The cause may be a payload shape that a future numba introduces, or a malformed
    object. The guard loads such a payload unchecked by default. Strict mode raises instead, so that no validation
    gap goes unreported. A well-formed object that the reader parses without error, and in which it finds no stale
    alias, is *not* a gap. It is loaded normally, in strict mode as by default, because it cannot be distinguished
    from a valid object that merely contains the alias prefix in a string constant.
    """


_ELF_MAGIC = b"\x7fELF"
# 32- and 64-bit, both byte orders, and both widths of the fat/universal container.
_MACHO_MAGICS = frozenset((0xFEEDFACE, 0xFEEDFACF, 0xCEFAEDFE, 0xCFFAEDFE,
                           0xCAFEBABE, 0xBEBAFECA, 0xCAFEBABF, 0xBFBAFECA))
_unvalidated_reported = False


def _elf_undefined_symbols(obj):
    """Names of the undefined (imported) symbols of an ELF64 little-endian relocatable object.

    A 32-bit or big-endian ELF is left unread and yields nothing, the same contract ``_undefined_symbols``
    documents for a container format it cannot parse.
    """
    und = set()
    if len(obj) < 0x40 or obj[4] != 2 or obj[5] != 1:
        return und
    (e_shoff,) = struct.unpack_from("<Q", obj, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", obj, 0x3A)
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        (sh_type,) = struct.unpack_from("<I", obj, sh + 4)
        if sh_type != 2:  # SHT_SYMTAB
            continue
        sym_off, sym_size = struct.unpack_from("<QQ", obj, sh + 24)
        (sh_link,) = struct.unpack_from("<I", obj, sh + 40)
        (sh_entsize,) = struct.unpack_from("<Q", obj, sh + 56)
        (str_off,) = struct.unpack_from("<Q", obj, e_shoff + sh_link * e_shentsize + 24)
        for j in range(sym_size // sh_entsize):
            (st_name,) = struct.unpack_from("<I", obj, sym_off + j * sh_entsize)
            (st_shndx,) = struct.unpack_from("<H", obj, sym_off + j * sh_entsize + 6)
            if st_shndx == 0 and st_name:  # SHN_UNDEF and named
                end = obj.index(b"\x00", str_off + st_name)
                und.add(obj[str_off + st_name:end].decode("utf-8", "replace"))
    return und


def _macho_undefined_symbols(obj):
    """Names of the undefined external symbols of a 64-bit little-endian Mach-O relocatable object.

    Mach-O prefixes C symbols with an underscore. The reader strips it, so the names are returned as numbox
    registered them. A reader that skipped that step would test ``_numbox_pxy_...`` against the prefix, match
    nothing, and pass every stale object without a warning. That is the failure this guard exists to prevent, and in
    this one place it would look like success. Other widths, byte orders and the fat container are left unread, as
    in the ELF reader.
    """
    und = set()
    if len(obj) < 32 or struct.unpack_from("<I", obj, 0)[0] != 0xFEEDFACF:
        return und
    (ncmds,) = struct.unpack_from("<I", obj, 16)
    off = 32
    for _ in range(ncmds):
        if off + 8 > len(obj):
            break
        cmd, cmdsize = struct.unpack_from("<II", obj, off)
        # Both bounds guard against a corrupt or foreign object, not a real one. cmdsize advances the walk, so a
        # zero would loop here forever, and this code runs on every cache load. Failing open recovers from an
        # exception but not from a hang.
        if cmdsize < 8 or off + cmdsize > len(obj):
            break
        if cmd == 0x2:  # LC_SYMTAB
            symoff, nsyms, stroff = struct.unpack_from("<III", obj, off + 8)
            nsyms = min(nsyms, max(0, len(obj) - symoff) // 16)
            for i in range(nsyms):
                n_strx, n_type = struct.unpack_from("<IB", obj, symoff + i * 16)
                if n_type & 0xE0:  # N_STAB: a debugging entry, never a link-time import
                    continue
                if (n_type & 0x0E) or not (n_type & 0x01):  # not N_UNDF, or not N_EXT
                    continue
                end = obj.index(b"\x00", stroff + n_strx)
                name = obj[stroff + n_strx:end].decode("utf-8", "replace")
                und.add(name[1:] if name.startswith("_") else name)
        off += cmdsize
    return und


def _coff_undefined_symbols(obj):
    """COFF/PE (Windows) objects are currently not read; see ``_undefined_symbols`` for what an empty result means."""
    return set()


def _undefined_symbols(object_code):
    """Undefined (imported) symbol names of one relocatable object, dispatched on its container format.

    A format that this function cannot read yields an empty set, so the caller degrades to "nothing to check" rather
    than to a false verdict. An unreadable object is therefore loaded exactly as it would be without this guard. COFF
    is the fallback, because a COFF object begins with a machine-type word rather than a distinctive magic number.
    """
    if object_code[:4] == _ELF_MAGIC:
        return _elf_undefined_symbols(object_code)
    if len(object_code) >= 4 and struct.unpack_from("<I", object_code, 0)[0] in _MACHO_MAGICS:
        return _macho_undefined_symbols(object_code)
    return _coff_undefined_symbols(object_code)


def _stale_proxy_aliases(payload, libdata_of_payload):
    """``@proxy`` aliases a serialized code library references that this process cannot correctly call.

    A non-empty result means that the payload must not be loaded. The result holds *every* stale alias that the
    object imports, not only the first. An object that calls two edited proxies references two unregistered names, and
    the caller joins the whole list into one warning, so the diagnostic never undercounts within an object. The
    guard cannot enumerate the stale entries that are never loaded. A proxied body whose own source changed is
    recompiled, overwrites its cache file and never reaches ``rebuild``, so it correctly produces no warning.

    ``ll.address_of_symbol`` is nearly an exact oracle. These names exist only in llvmlite's explicit-symbol map. The
    alias encodes the body, the signature and the jit options, so an alias that resolves normally names the correct
    body to call. The exception is an alias that represents an absent ``proxy_if_available`` binding: it resolves to
    a trap that holds no body. Loading that object is worse than discarding it. The trap's error is raised inside a
    ``@cfunc``, where numba discards the error and returns zero, so the caller computes with a wrong value, and no
    warning is issued. Discarding the object instead yields the same typing error that a cold cache gives.

    The payload is unpacked inside the handler rather than by the caller, because the payload's shape is exactly
    what a future numba might change.

    If ``NUMBOX_PROXY_CACHE_STRICT`` is set, a payload that cannot be *read* raises
    :class:`UnvalidatedProxyCacheError` instead of being loaded unchecked. Strict mode does not raise for a
    well-formed object that the reader parses without error, even when the reader finds no stale alias in it. An
    object that contains the prefix only in a string constant parses exactly as an object with the prefix in an
    unresolved import would parse if the reader failed to extract its symbols. Strict mode therefore cannot escalate
    the second case without also aborting the first, which is a valid cached function that involves no proxy at all.
    A reader that fails to extract symbols is detected where the two cases can be distinguished: in CI, a test
    asserts that the reader parses the objects that numba actually emits. The guard does not guess at such a failure
    on live loads.
    """
    strict = _strict_cache_mode()
    try:
        _name, kind, data = libdata_of_payload(payload)
        if kind != "object":
            return []
        object_code = data[0]
        if _ALIAS_PREFIX.encode() not in object_code:
            return []  # fast path: the alias string appears nowhere in the object
        return sorted(
            s for s in _undefined_symbols(object_code)
            if s.startswith(_ALIAS_PREFIX) and (s in _ABSENT_ALIASES or not ll.address_of_symbol(s))
        )
    except Exception as exc:
        # Fail open: this code runs on every numba cache load in the process. An unexpected failure, such as a future
        # numba changing the payload shape or a malformed object, must degrade to "no validation" and never break
        # unrelated caching. A caller who sets NUMBOX_PROXY_CACHE_STRICT asks for the opposite, so raise the error.
        if strict:
            raise UnvalidatedProxyCacheError(
                f"numbox: could not check a numba cache payload for stale @proxy aliases ({exc!r}); "
                f"{_PROXY_CACHE_STRICT_ENV} is set, so the load was aborted rather than proceeding unchecked"
            ) from exc
        # Report the failure once, though. A check abandoned without a report is indistinguishable from a check that
        # passed, and the unchecked load can cause the crash that this guard exists to prevent.
        global _unvalidated_reported
        if not _unvalidated_reported:
            _unvalidated_reported = True
            warnings.warn(
                f"numbox: could not check a numba cache payload for stale @proxy aliases ({exc!r}); it was "
                "loaded unchecked. Later payloads in this process are checked, but not reported again.",
                UnvalidatedProxyCacheWarning, stacklevel=3,
            )
        return []


def _guarded_rebuild(orig_rebuild, libdata_of_payload):
    """Wrap a numba ``CacheImpl.rebuild`` so that a payload is validated while it is still plain bytes.

    This is the last point upstream of the execution engine. Once ``unserialize_library`` runs, the object's
    externals are resolved in one batch, and a single unregistered name zeroes every relocation in the object. That
    failure is unrecoverable and terminates the process without a diagnostic. Returning ``None`` is numba's own
    contract for a cache miss, so the stale entry is discarded and recompiled in place. If
    ``NUMBOX_PROXY_CACHE_STRICT`` is set, the wrapper refuses the discard and raises :class:`StaleProxyCacheError`
    before the ``None`` return, so the stale entry stays on disk for inspection.
    """
    def rebuild(self, target_context, payload):
        stale = _stale_proxy_aliases(payload, libdata_of_payload)
        if stale:
            filename_base = getattr(self, 'filename_base', '?')
            if _strict_cache_mode():
                raise StaleProxyCacheError(
                    f"numbox: {filename_base} references @proxy cfunc alias(es) this process cannot call "
                    f"({', '.join(stale)}); {_PROXY_CACHE_STRICT_ENV} is set, so the stale entry was left on disk for "
                    "inspection instead of being discarded and recompiled"
                )
            warnings.warn(
                "numbox: discarding a stale numba cache entry and recompiling: "
                f"{filename_base} references @proxy cfunc alias(es) this process "
                "cannot call (the proxied body, signature or jit_options changed since the caller was "
                "cached, or the binding is no longer available): "
                + ", ".join(stale),
                StaleProxyCacheWarning,
                stacklevel=3,
            )
            return None
        return orig_rebuild(self, target_context, payload)
    return rebuild


def _install_cache_alias_guard():
    """Validate every numba cache load in this process against the registered ``@proxy`` aliases.

    The ordering needs no coordination. A caller can only have been compiled against a ``@proxy`` binding whose
    module it imports, and importing that module imports this one. A cached function that references
    no alias may load before numbox is imported, and needs no validation.

    Every lookup happens before the first assignment, so a numba that lacks one of these classes leaves the process
    with no guard rather than with a partial one.
    """
    from numba.core import caching
    if getattr(caching, "_numbox_proxy_alias_guard", False):
        return
    wrapped = [
        (caching.CompileResultCacheImpl, _guarded_rebuild(caching.CompileResultCacheImpl.rebuild,
                                                          lambda payload: payload[0])),
        (caching.CodeLibraryCacheImpl, _guarded_rebuild(caching.CodeLibraryCacheImpl.rebuild,
                                                        lambda payload: payload)),
    ]
    caching._numbox_proxy_alias_guard = True
    for impl, rebuild in wrapped:
        impl.rebuild = rebuild


try:
    _install_cache_alias_guard()
except Exception:
    # Fail open, as the per-load validator does. Without the guard, the process loses its protection against a
    # stale cache entry; an install-time error that propagated would make the whole library unimportable.
    pass
