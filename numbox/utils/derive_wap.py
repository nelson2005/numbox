"""First-class function values whose exceptions propagate out of the call.

The mechanism
-------------

numba lowers a first-class ``FunctionType`` call two ways and picks between them on
the function model's ``jit_addr`` slot. A populated slot selects the numba calling
convention plus ``return_status_propagate``, so an exception unwinds into the caller.
An empty slot selects the C wrapper, which numba documents as not supporting
exceptions: it discards the exception, zero-fills the return value and reports the
failure only as an unraisable on stderr.

numba fills the slot for a ``Dispatcher`` and leaves it empty for everything else,
``_get_jit_address`` returning 0. A bare compile result therefore arrives at a call
site with no entry point that could carry an exception out.

:class:`DeriveWAP` captures the compile result's numba-callconv entry point, and
:class:`DeriveFunctionType` populates ``jit_addr`` from it on both unboxing and
constant lowering. A value of this type carries a usable entry point wherever it
goes.

Using it
--------

:func:`numbox.utils.highlevel.cres` mints these values, so a `cres`-compiled function
is already one::

    @cres(float64(float64))
    def f(x):
        if x < 0:
            raise ValueError("negative")
        return x

Filling the slot is necessary but not sufficient: the *call site* decides which
convention it emits, and one that reads the wrapper address unconditionally gets the
C wrapper however the slot is filled. A consumer opts in by reading ``jit_addr`` and
emitting ``call_conv.call_function`` plus ``return_status_propagate`` when it is
non-null; numba's own version of that is
``numba.core.lowering.Lower.__call_first_class_function_pointer``.

``numbox.core.work.work._call_derive`` is the worked example in this repository, and
:doc:`numbox.core.work` walks through it. It selects the propagating convention at
compile time for a :class:`DeriveFunctionType`, and tests the slot at runtime for a
plain ``FunctionType``, which numba does populate for an njit dispatcher passed as a
``FunctionType``-typed argument.

Limits
------

:meth:`DeriveFunctionType.can_convert_to` documents the one direction that does not
survive: a value handed from Python into a parameter *declared* as a plain
``FunctionType`` degrades to the C convention on the way in.

Where numba has no such slot :func:`jit_addr_supported` is false, the whole mechanism
is inert and ``cres`` returns a plain ``CompileResultWAP``.

Standing on numba
-----------------

The decorators below are numba's public extension API, save for ``lower_constant``,
which ``numba.extending`` does not re-export. What they register against is not:
``FunctionModel``, ``CompileResultWAP``, ``Conversion``, ``box_function_type`` and
``lower_get_wrapper_address`` all sit outside ``numba.extending``, and the constant
lowering reads ``context.call_conv`` and ``context.add_dynamic_addr`` directly. What
does hold is that no numba internals are patched: nothing here replaces numba
behaviour, it only registers against it.
"""
import hashlib

from llvmlite import binding as ll
from numba.core import cgutils, types
from numba.core.imputils import lower_cast, lower_constant
from numba.core.typeconv import Conversion
from numba.core.types.function_type import CompileResultWAP, FunctionType
from numba.core.typing.typeof import typeof_impl
from numba.experimental.function_type import (
    FunctionModel, box_function_type, lower_get_wrapper_address,
)
from numba.extending import NativeValue, box, register_model, unbox

from numbox.core.configurations import _ALIAS_PREFIX, function_struct_has
from numbox.utils.fingerprint import _body_fingerprint


__all__ = ["DeriveFunctionType", "DeriveWAP", "jit_addr_supported", "rewrap_derive"]


#: Where :func:`rewrap_derive` parks the upgraded wrapper on the object it upgrades,
#: so that the address stored in ``py_addr`` stays backed for as long as the caller
#: holds the original.
_UPGRADED_ATTR = "_numbox_derive_wap"


def jit_addr_supported() -> bool:
    """Whether the running numba exposes the ``jit_addr`` slot.

    The slot was added to ``FunctionModel`` in numba 0.61. Where it is absent the
    mechanism has nothing to populate, so numbox leaves first-class calls to numba
    rather than shipping a half-installed variant.
    """
    return function_struct_has("jit_addr")


class DeriveFunctionType(FunctionType):
    """First-class function type whose values always carry a populated ``jit_addr``.

    Kept distinct from ``FunctionType`` so that ``_call_derive`` can select the
    propagating calling convention at compile time, with no runtime branch, for
    every derive numbox itself compiled.
    """

    def can_convert_to(self, typingctx, other):
        """Permit passing a :class:`DeriveWAP` where a plain ``FunctionType`` of
        the same signature is declared.

        Such a call site keeps working unchanged, which is the point, but it
        keeps its *old* behaviour too: crossing the Python boundary into a
        parameter declared as a plain ``FunctionType`` degrades the value to the
        C convention, so a `derive` supplied that way still discards its
        exception. Reaching the same declared type by a cast within jitted scope
        does not degrade it, because :func:`lower_cast_derive_to_function_type`
        is an identity on a shared ``FunctionModel`` and the populated
        ``jit_addr`` survives. One source-level call site therefore has two
        different exception semantics depending on where the value came from.
        """
        if type(other) is FunctionType and other.signature == self.signature:
            return Conversion.safe
        return None


@lower_cast(DeriveFunctionType, FunctionType)
def lower_cast_derive_to_function_type(context, builder, fromty, toty, val):
    """Identity cast. Both types use ``FunctionModel``, so the struct is shared."""
    return val


#: Alias name -> the compile result it was published from. Three duties, all required.
#: ``llvmlite.binding.add_symbol`` records an address, not a reference, so the compiled
#: body has to be kept alive by something for as long as any caller can jump to that
#: address, and a `DeriveWAP` is an ordinary object a caller may well drop; a symbol a
#: caller was lowered against must keep resolving to the same code for the rest of the
#: process, so the registration is never rebound underneath code already compiled; and
#: the compile result recorded here is what a later `DeriveWAP` minting the same alias
#: is checked against, so that an alias which does not identify the body is refused
#: rather than shared -- see :func:`_publish_jit_alias`.
_JIT_ALIASES = {}


def _stable_jit_alias(func, sig, jit_options=None):
    """Deterministic, process-stable LLVM symbol name for ``func``'s callconv entry point.

    The sibling of ``numbox.core.proxy.proxy._stable_cfunc_alias``, and deliberately
    the same construction, for the same reason: ``fndesc.llvm_func_name`` carries a
    process-local ``v<N>`` abi tag that is no part of numba's cache key, so a cached
    caller must not reference it by that name. What differs is *which* entry point is
    published. The cfunc wrapper takes its arguments positionally and cannot carry an
    exception out; this one is the numba calling convention, returning a status code
    through a return-value pointer, which is the entry ``jit_addr`` has to hold. They
    are different functions with different LLVM types, so they need different names --
    hence the ``cc`` tag, which is folded into the digest as well so that the two can
    never collide.

    Folding the body fingerprint is what fixes staleness rather than merely relocating
    it: edit the derive's body and the alias is renamed, so a warm ``cache=True`` caller
    that baked in the old name references a symbol this process never registered, and
    the guard in `numbox.core.proxy.proxy` discards that entry and recompiles it. The
    resolved ``jit_options`` are folded for the same reason they are on the cfunc side:
    they govern the machine code without appearing in the identity above.
    """
    raw = (
        f"cc.{func.__module__ or ''}.{func.__qualname__}.{sig}."
        f"{_body_fingerprint(func)}."
        f"{sorted((jit_options or {}).items(), key=repr)!r}"
    ).encode("utf-8")
    safe_name = "".join(c if c.isascii() and c.isalnum() else "_" for c in func.__name__)
    return f"{_ALIAS_PREFIX}cc_{safe_name}_{hashlib.sha256(raw).hexdigest()[:16]}"


def _cache_guard_installed():
    """Install `numbox.core.proxy.proxy`'s numba cache guard, and say whether it is there.

    An alias is only safe to bake into a cached caller if something notices when it stops
    resolving. That watcher is the guard `numbox.core.proxy.proxy` installs over numba's
    cache loads, and until now every alias came from that module, so importing the minter
    installed the watcher. A derive minted by :func:`numbox.utils.highlevel.cres` breaks
    that: a process can reach an alias with no ``@proxy`` binding anywhere in it, and an
    unwatched alias is the worst of both worlds -- a caller cached against a name that,
    once the body is edited, nothing registers and nothing checks, which is a segfault.

    The import is deferred rather than module-level because `numbox.core.proxy.proxy`
    imports this module; by the time a wrapper is minted that import has long settled.
    A failure degrades to ``False``, so the caller falls back to a baked address and an
    uncacheable caller rather than minting an alias nobody is watching.
    """
    try:
        from numba.core import caching
        from numbox.core.proxy.proxy import _install_cache_alias_guard
        _install_cache_alias_guard()
        return bool(getattr(caching, "_numbox_proxy_alias_guard", False))
    except Exception:
        return False


def _publish_jit_alias(cres, py_func, jit_options, jit_address):
    """Publish ``cres``'s callconv entry point under a process-stable alias.

    Returns the alias, or ``None`` when no alias may be published -- see
    :class:`DeriveWAP` for what the constant lowering does then. That is the outcome
    when :func:`rewrap_derive` supplies no `py_func`, when the cache guard could not be
    installed, and when the alias this body mints is already bound to a *different*
    compile result.

    That last one is the case where the name does not identify the body.
    :func:`~numbox.utils.fingerprint._body_fingerprint` degrades to its best-effort
    walker for any body it cannot canonicalise -- which is every numbox binding body,
    since they all reach the ``@intrinsic`` ``_call_lib_func`` -- and that walker
    substitutes one placeholder per *type*, so two bodies from one factory closing over
    different C function pointers are indistinguishable to it and mint one alias between
    them. A symbol a caller was lowered against has to keep resolving to the same code,
    so the second body cannot be published under it; and handing the alias back anyway
    lowers the second derive's constant callers against the *first* derive's machine
    code, which is a wrong answer that only the constant route gives -- the argument
    route reads ``jit_address`` off the object it was handed and stays correct. So the
    second derive is refused an alias, its constant callers bake the address, numba
    declines to cache them, and they run the current body in every process. Uncacheable
    is the honest reading of "this name does not identify this body".

    The comparison is compile-result identity because nothing weaker separates a genuine
    collision from a harmless duplicate. Across a full run of this repository's suite,
    every publish that duplicated an already-registered alias differed from it in both
    the compile result and the resolved entry address -- exactly as a genuine collision
    does -- and in ``fndesc.llvm_func_name`` on half of them as well, that name carrying
    a per-process compile counter. What identity costs is therefore the constant-route
    cacheability of a second derive over an already-published body, and nothing at all
    unless something reaches that second derive as a compile-time constant. Over the
    same run nothing does: the duplicates are `numbox.core.work.builder`'s graph
    derives, which reach jitted scope as arguments rather than as constants, repeat
    compilations of one test helper, and two textually identical lambdas in one test.

    Registration happens here, when the wrapper is minted, which is strictly before any
    caller can be lowered against it: a caller reaches the alias only through this
    value, and it does not exist yet.
    """
    if py_func is None or not _cache_guard_installed():
        return None
    alias = _stable_jit_alias(py_func, cres.signature, jit_options)
    published = _JIT_ALIASES.get(alias)
    if published is None:
        _JIT_ALIASES[alias] = cres
        ll.add_symbol(alias, jit_address)
    elif published is not cres:
        return None
    return alias


def _compiled_from(cres, py_func):
    """Whether ``py_func`` is the Python function ``cres`` was compiled from.

    The alias is content-addressed on `py_func`, so supplying one the compile result
    knows nothing about mints a name that moves when the wrong body is edited and
    stands still when the right one is: a caller cacheable against a key that never
    re-keys. :class:`DeriveWAP` is public, and it is the one argument that cannot be
    got wrong safely.

    A freshly compiled result carries the function itself, on
    ``type_annotation.func_id``, and that identity is what every minter in numbox
    supplies -- measured over all of them. One numba restored from its own cache
    replaces ``type_annotation`` with a string, so nothing is left to compare by
    identity; there the check falls back to the module and qualified name, which
    ``fndesc`` records in every compile result. That is weaker -- it accepts a
    different function of the same name -- but the alternative is to accept anything
    at all on a warm compile result, and a same-name body is the case
    :func:`_publish_jit_alias` already refuses to share an alias with.
    """
    func_id = getattr(cres.type_annotation, "func_id", None)
    if func_id is not None:
        return func_id.func is py_func
    return (cres.fndesc.modname == py_func.__module__
            and cres.fndesc.qualname == py_func.__qualname__)


class DeriveWAP(CompileResultWAP):
    """``CompileResultWAP`` that also captures the numba-callconv entry point.

    ``CompileResultWAP`` records only the cfunc wrapper address, which is the
    entry point that cannot carry an exception out.

    Where the Python function behind the compile result is available it is passed
    in as `py_func`, together with the `jit_options` it was compiled under, and the
    callconv entry point is published under a content-addressed alias (see
    :func:`_stable_jit_alias`). That alias is what a constant reference from jitted
    code emits, which is what makes such a caller both cacheable and safe to cache.

    Without it -- :func:`rewrap_derive` upgrading a foreign wrapper, which carries
    the compile result and nothing else -- no alias is minted, the constant lowering
    bakes the address instead, and numba declines to cache the caller. That is the
    outcome to want: such a caller is recompiled in every process and can never serve
    a stale body. The choice is made here rather than at lowering, and it comes out
    the same way in a cold process and in a warm one, so a caller cannot be cached
    against an alias in one run and lowered without one in the next.

    A function *of the right name* is in fact reachable from a bare compile result,
    warm as well as cold: ``fndesc.lookup_module()`` and ``fndesc.qualname`` between
    them find one, and fingerprinting it mints the same alias in a cold process and a
    warm one. Reaching it that way is nevertheless a guess rather than a fact. The
    name may since have been rebound, wrapped or deleted, and the guess cannot be
    checked against anything the compile result carries. A wrong guess is worse than
    no alias, not better: the caller becomes cacheable, and the alias it is keyed to
    is content-addressed on the *other* function, so it never moves when the body the
    caller actually calls is edited. That is permanent silent staleness, which is the
    hazard the alias exists to close. `py_func` is therefore supplied by whoever
    compiled the body and by nobody else.

    `py_func` is checked against the compile result for that reason: a value keyed to
    the wrong body is exactly as stale as one recovered by guesswork, and the
    constructor is public.
    """

    def __init__(self, cres, py_func=None, jit_options=None):
        super().__init__(cres)
        if py_func is not None and not _compiled_from(cres, py_func):
            raise ValueError(
                f"py_func {py_func.__module__}.{py_func.__qualname__} is not the "
                f"function behind the compile result for {cres.fndesc.modname}."
                f"{cres.fndesc.qualname}; the alias would be keyed to a body the "
                f"caller never calls and would never re-key when that body changed")
        self.jit_address = cres.library.get_pointer_to_function(
            cres.fndesc.llvm_func_name)
        if self.jit_address <= 0:
            raise ValueError(
                f"no callconv entry point for {cres.fndesc.llvm_func_name}")
        self.jit_alias = _publish_jit_alias(cres, py_func, jit_options, self.jit_address)


@typeof_impl.register(DeriveWAP)
def typeof_derive_wap(val, c):
    return DeriveFunctionType(val.signature())


register_model(DeriveFunctionType)(FunctionModel)


def _get_derive_jit_address(func, sig):
    """Resolve the callconv entry address during unboxing.

    Called from jitted unboxing code with the GIL held, mirroring how numba's own
    first-class unboxing reaches back into Python. Raising here surfaces at the
    unbox boundary rather than corrupting the struct, so a value that is not a
    :class:`DeriveWAP` fails loudly instead of producing a null ``jit_addr`` that
    would silently reinstate the swallow.
    """
    if isinstance(func, DeriveWAP):
        return func.jit_address
    raise TypeError(
        f"DeriveFunctionType value must be a DeriveWAP, got {type(func)}")


def _lower_get_derive_jit_address(context, builder, func, sig):
    """Emit the call to :func:`_get_derive_jit_address`.

    Follows numba's ``_lower_get_address`` with ``failure_mode='return_null'``:
    a null result returns NULL from the unboxing function, which propagates the
    Python exception rather than swallowing it.
    """
    pyapi = context.get_python_api(builder)
    modname = context.insert_const_string(builder.module, __name__)
    mod = pyapi.import_module(modname)
    fn = pyapi.object_getattr_string(mod, "_get_derive_jit_address")
    pyapi.decref(mod)
    sig_obj = pyapi.unserialize(pyapi.serialize_object(sig))
    addr = pyapi.call_function_objargs(fn, (func, sig_obj))
    # `fn` and `sig_obj` are new references and `addr` does not borrow from either,
    # so release them before the branch below: the null path returns early, and
    # unboxing runs on every call for a derive passed as an argument.
    pyapi.decref(fn)
    pyapi.decref(sig_obj)
    with builder.if_then(cgutils.is_null(builder, addr), likely=False):
        builder.ret(pyapi.get_null_object())
    return addr


@unbox(DeriveFunctionType)
def unbox_derive_function_type(typ, obj, c):
    typ = typ.get_precise()
    sfunc = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    addr = lower_get_wrapper_address(
        c.context, c.builder, obj, typ.signature, failure_mode="return_null")
    sfunc.c_addr = c.pyapi.long_as_voidptr(addr)
    c.pyapi.decref(addr)

    llty = c.context.get_value_type(types.voidptr)
    sfunc.py_addr = c.builder.ptrtoint(obj, llty)

    addr = _lower_get_derive_jit_address(c.context, c.builder, obj, typ.signature)
    sfunc.jit_addr = c.pyapi.long_as_voidptr(addr)
    c.pyapi.decref(addr)

    return NativeValue(sfunc._getvalue())


@box(DeriveFunctionType)
def box_derive_function_type(typ, val, c):
    return box_function_type(typ, val, c)


@lower_constant(DeriveFunctionType)
def lower_constant_derive_function_type(context, builder, typ, pyval):
    """Lower a derive reached as a compile-time constant from jitted code.

    ``c_addr`` and ``py_addr`` follow numba's own lowering for a
    ``WrapperAddressProtocol`` value. ``jit_addr`` holds the callconv entry
    point, and how it gets there decides everything downstream.

    Where the value carries an alias (:func:`_stable_jit_alias`), the entry point
    is declared as an *external* function of that name and nothing else about the
    derive enters the caller's module. Three things follow from that one choice.
    The caller is cacheable, because an extern reference is not a dynamic global,
    so the dead ``c_addr``/``py_addr`` globals are eliminated before numba scans
    the final module and the caller reports none. It is safe to cache, because the
    alias folds a fingerprint of the derive's body: edit the body and the name
    moves, so the object a warm caller loads imports a symbol nobody registered and
    the cache guard in `numbox.core.proxy.proxy` discards it and recompiles rather
    than serving the old numbers -- the same treatment, through the same guard,
    that a caller of the dispatcher already gets. And which body runs is
    single-valued, because there is no longer a copy of it in the caller to
    compete with the definition the alias names.

    Numba's own ``Dispatcher -> FunctionType`` cast instead declares the mangled
    name and links the callee's library into the caller. That is what this used to
    do, and it fails all three: the mangled name folds a per-process compile
    counter that is no part of numba's cache key, the caller's object carries the
    body rather than a reference to it, so the guard has no name to check and an
    edit is served stale indefinitely, and a body too large to inline is embedded
    as a *weak* definition, leaving the caller running its own copy or whichever
    definition of that name got there first. Linking it in also gives up exactly
    what ``@proxy`` exists to avoid, static linking of the callee's LLVM into the
    caller.

    Without an alias -- :func:`rewrap_derive` over a foreign wrapper, whose body is
    identified by a guess rather than by the caller that compiled it, or a second
    derive whose alias is already bound to another body (see
    :func:`_publish_jit_alias`) -- the address is baked instead. That leaves a live
    dynamic global, so numba declines to cache the caller and says so; such a caller
    is recompiled in every process and therefore always runs the current body.
    Uncacheable is the honest reading of "no name here identifies this body", and it
    is the safe half of the trade.

    There is deliberately no fallback to the entry point's mangled name. A value
    of this type always takes the propagating call, so a `jit_addr` that failed to
    resolve would be called unconditionally, and failing the compilation is the
    only honest outcome.
    """
    typ = typ.get_precise()
    assert typ.check_signature(pyval.signature())
    sfunc = cgutils.create_struct_proxy(typ)(context, builder)
    sfunc.c_addr = context.add_dynamic_addr(
        builder, pyval.__wrapper_address__(), info=str(typ))
    sfunc.py_addr = context.add_dynamic_addr(
        builder, id(pyval), info=type(pyval).__name__)
    alias = getattr(pyval, "jit_alias", None)
    if alias is None:
        sfunc.jit_addr = context.add_dynamic_addr(
            builder, pyval.jit_address, info=f"{typ} callconv entry point")
    else:
        fndesc = pyval.cres.fndesc
        fnty = context.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
        fn = cgutils.get_or_insert_function(builder.module, fnty, alias)
        sfunc.jit_addr = builder.bitcast(fn, context.get_value_type(types.voidptr))
    return sfunc._getvalue()


def rewrap_derive(derive):
    """Upgrade a foreign ``CompileResultWAP`` so its exceptions propagate.

    A derive compiled by numbox's own :func:`~numbox.utils.highlevel.cres` is
    already a :class:`DeriveWAP`. One built directly against numba is not, and
    would keep the swallowing convention. The compile result it carries is all
    that is needed to upgrade it.

    Anything else, including ``None``, is returned unchanged so callers can apply
    this unconditionally.

    On a numba without the ``jit_addr`` slot there is nothing to upgrade into: the
    struct has no field to hold the entry point, so producing a
    :class:`DeriveWAP` there would only yield a value that cannot be unboxed.

    The upgraded wrapper is memoized onto the object it upgrades, and that is
    required rather than an optimization. ``py_addr`` holds the derive's address
    without taking a reference, so a wrapper minted fresh per call would be freed
    as soon as the caller returned, leaving every `Work` built from it pointing at
    released memory. Hanging it off the original ties its lifetime to the object
    the caller already holds, which is the lifetime the address assumed all along.
    """
    if not jit_addr_supported():
        return derive
    if not isinstance(derive, CompileResultWAP) or isinstance(derive, DeriveWAP):
        return derive
    upgraded = getattr(derive, _UPGRADED_ATTR, None)
    if upgraded is None:
        upgraded = DeriveWAP(derive.cres)
        setattr(derive, _UPGRADED_ATTR, upgraded)
    return upgraded
