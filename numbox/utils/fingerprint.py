"""Content fingerprinting of Python values and functions for cache keys.

The module provides a value canonicalizer (`_canon_value`) and a deep function fingerprint (`_fingerprint_function`).
They capture everything numba freezes into a compiled artifact: the bytecode, consts and names of the code object, the
default arguments, the closure-cell values, and the values of the module-level globals the code references. The walk
recurses into helper functions and dispatchers, with protection against cycles. The fingerprint is stronger than a
hash of the bare code object, because two functions with identical source but different captured closure or global
values fingerprint differently. `numbox.core.variable.compile_kernel` uses it for the kernel cache digest, and
`numbox.utils.digest` uses it for the SQLite UDAF cache key.

Both functions raise `_Unfingerprintable` for any value with no canonical form, and each caller decides how to degrade:
compile_kernel marks the kernel uncached, and digest uses the best-effort walker.
"""
import ctypes
import dis
import hashlib

from types import CodeType, FunctionType, ModuleType
from typing import Any, Optional, Sequence

import numpy as np

from numba.core.dispatcher import Dispatcher
from numba.core.types import ExternalFunction
from numba.core.types.function_type import CompileResultWAP
from numba.np.ufunc.dufunc import DUFunc

from numbox.core.configurations import jit_options as _default_jit_options


class _Unfingerprintable(Exception):
    """A value with no canonical fingerprint; the caller decides how to degrade."""


def _safe_repr(obj: object) -> str:
    """Return ``repr(obj)`` without ever raising.

    The fingerprint fallback must always yield a string, so that an un-fingerprintable formula degrades to uncached
    rather than crashing when its ``__repr__`` itself raises."""
    try:
        return repr(obj)
    except Exception:  # noqa: BLE001 - fallback must not crash on a raising __repr__
        return f"<{type(obj).__name__} repr-failed>"


def _dtype_key(dtype: np.dtype) -> str:
    """Canonical string for a numpy dtype that preserves structured layout.

    ``dtype.str`` collapses any structured dtype to ``|V<itemsize>`` and erases the field names, types, order and
    offsets, so two arrays with the same bytes and different layouts would collide. For a structured dtype, the
    function folds the full descriptor and the itemsize instead."""
    if dtype.names is not None:
        return f"struct({dtype!s};descr={dtype.descr};itemsize={dtype.itemsize})"
    return dtype.str


# This set holds the aliases that no longer identify the body they were created for. `numbox.core.proxy.proxy`
# writes the set and imports this module, so the set is defined on this side: the walker can then read it without
# importing that module back. The set holds both names of a collision, the shared name and the process-local name
# that the new body is given. It also holds the alias under which a trap for an absent binding is registered.
#
# `_canon_value` refuses to fold an alias in this set, so that a collision does not spread. The order in which this
# process built two colliding bodies decides the name that each body holds. If a third body that captures one of them
# folded `proxy(<alias>)`, its fingerprint would name the other body in a process that built the pair in the
# opposite order. The colliding bodies themselves are protected, because their own names are retired and the cache
# guard discards a warm caller of a retired name. A body derived from one of them is not protected in that way. It is
# published under a new name that resolves and is in no retirement set, so it would keep a cross-process cache that
# is now wrong, and the collision would propagate one composition step. Because the alias is not folded, the derived
# body falls through to the ordinary walk, where the dispatcher it captured has no canonical form.
# `_opaque_values_match` then checks it like any other pair that the fingerprint cannot separate, and its own alias
# is retired in turn.
#
# A derived body fingerprinted before the collision that retires its dependency is not protected. It folded a name
# that still identified a body, and it is already published under a name of its own. Protecting it would require
# recording which aliases each fingerprint consumed and retiring the dependent bodies transitively.
_ABSENT_ALIASES: set[str] = set()


def _canon_value(value: Any, seen: set[int]) -> str:
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return repr(value)
    if isinstance(value, np.generic):
        return f"npscalar({_dtype_key(value.dtype)};{value.tobytes().hex()})"
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise _Unfingerprintable(f"object-dtype ndarray {value.dtype.str}")
        data = np.ascontiguousarray(value)
        try:
            raw = hashlib.sha256(data.tobytes()).hexdigest()
        except (ValueError, TypeError) as e:
            raise _Unfingerprintable(f"unhashable ndarray {value.dtype.str}") from e
        return f"ndarray({_dtype_key(data.dtype)};{value.shape};{raw})"
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__}[" + ",".join(_canon_value(v, seen) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}[" + ",".join(sorted(_canon_value(v, seen) for v in value)) + "]"
    if isinstance(value, dict):
        items = sorted((_canon_value(k, seen), _canon_value(v, seen)) for k, v in value.items())
        return "dict[" + ",".join(f"{k}:{v}" for k, v in items) + "]"
    if isinstance(value, ModuleType):
        return f"module({value.__name__})"
    if isinstance(value, Dispatcher):
        proxy_alias = getattr(value, "_numbox_proxy_alias", None)
        if proxy_alias is not None and proxy_alias not in _ABSENT_ALIASES:
            # For a @proxy binding, the content-addressed alias fully identifies the body and the signature. Walking
            # its wrapper would recurse into the @intrinsic it calls, which has no canonical form. A retired alias
            # identifies nothing, so a binding with a retired alias falls through to that walk instead.
            return f"proxy({proxy_alias})"
        topts = _canon_value(dict(getattr(value, "targetoptions", {}) or {}), seen)
        return f"dispatcher({_fingerprint_function(value.py_func, seen)};{topts})"
    if isinstance(value, DUFunc):
        # For a @vectorize DUFunc, numba links its compiled loop and freezes the captured globals and closure of the
        # scalar body, so the fingerprint folds that body and its targetoptions.
        wrapped = getattr(value, "__wrapped__", None)
        topts = _canon_value(dict(getattr(value, "targetoptions", {}) or {}), seen)
        body = _fingerprint_function(wrapped, seen) if isinstance(wrapped, FunctionType) else _safe_repr(wrapped)
        return f"dufunc({body};{topts})"
    if isinstance(value, FunctionType):
        return f"function({_fingerprint_function(value, seen)})"
    if isinstance(value, ExternalFunction):
        # numba's own key for this type is (symbol, signature). Its parts are strings and types, so the key is the
        # same in the next process, and it separates two bindings that a factory made over different C symbols.
        return f"externalfunction({value.symbol};{value.sig})"
    if isinstance(value, CompileResultWAP):
        # A CompileResultWAP is a first-class function value. numbox tags the two kinds it builds with what identifies
        # them across processes, because the wrapper itself carries only a compiled address and a mangled name that
        # holds a per-process counter. An untagged value, built directly against numba, falls through to the
        # placeholder below.
        proxy_alias = getattr(value, "_numbox_proxy_alias", None)
        if proxy_alias is not None and proxy_alias not in _ABSENT_ALIASES:
            return f"proxyfunc({proxy_alias})"
        py_func = getattr(value, "_numbox_py_func", None)
        if py_func is not None:
            return f"cresfunc({_fingerprint_function(py_func, seen)};{value.cres.signature})"
    if isinstance(value, ctypes._CFuncPtr):
        return f"cfuncptr({_ctypes_func_key(value)})"
    raise _Unfingerprintable(type(value).__name__)


def _ctypes_func_key(value: Any) -> str:
    """Identity of a ctypes function pointer: the library it was opened under, and the symbol.

    A pointer reached by attribute access on a loaded library carries both, and both mean the same thing in the next
    process. The key folds the *string* the library was opened with, not the library the loader resolved it to, so
    two handles on one library, opened under a soname and under an absolute path, canonicalize differently. That is
    over-discrimination: it costs a recompile that the load-time guard repairs, and never a wrong answer.

    A pointer with no symbol name to fold carries no identity that means the same thing in the next run. The common
    case is a pointer built from a raw address, and two of those differ only by an address that ASLR moves. A
    prototype built over a library, ``proto(("floor", lib))``, and a Windows lookup by ordinal have the same problem
    for a different reason, because ctypes names neither. Folding an address would gain discrimination at the cost of
    the process stability that the whole fingerprint exists for. All three therefore raise, and the caller decides
    what to do with a value that it cannot distinguish from another.

    The key also folds the prototype over the symbol, through :func:`_ctypes_call_shape`.
    """
    name = getattr(value, "__name__", None)
    objects = getattr(value, "_objects", None)
    lib = objects.get("0") if isinstance(objects, dict) else None
    # Only a library carries ``_name``. A Python callback's ``_objects["0"]`` is its CThunkObject, which does not, and
    # folding the None would map every callback that shares a ``__name__`` to one key.
    lib_name = getattr(lib, "_name", None)
    if name is None or not isinstance(lib_name, str) or not lib_name:
        raise _Unfingerprintable("ctypes function pointer with no library and symbol name")
    return f"{lib_name}:{name};{_ctypes_call_shape(value)}"


def _ctypes_call_shape(value: Any) -> str:
    """The call a ctypes function pointer describes, apart from what it points at.

    numba types the call from ``restype``, ``argtypes`` and the calling convention, and lowers it against them. Two
    pointers that agree on the symbol or the address but not on these describe two different calls, and they compile
    to two different bodies. Every part of the shape is a type name or a flag word, so the shape means the same thing
    in the next process, and :func:`_ctypes_func_key` can fold it. If that key refuses a pointer for lack of a
    process-stable identity, :func:`_opaque_values_match` compares the pointer on its shape instead. That comparison
    admits an address, and the shape determines whether the address is called the same way.

    An unset ``argtypes`` and an empty one are different: numba rejects a call through the first and accepts the
    second, so they must not fold together.

    The calling convention is folded because numba selects the call's ABI from it on win32. A symbol reached through
    ``CDLL`` and the same symbol reached through ``WinDLL`` are two different calls, and sharing one alias would pair a
    caller with a body at the wrong convention.
    """
    restype = getattr(getattr(value, "restype", None), "__name__", None)
    argtypes = getattr(value, "argtypes", None)
    args = "<unset>" if argtypes is None else ",".join(getattr(a, "__name__", "?") for a in argtypes)
    return f"{restype}({args});flags={getattr(type(value), '_flags_', None)}"


def _fingerprint_codeobj(code: CodeType, seen: set[int]) -> str:
    consts = ",".join(
        _fingerprint_codeobj(c, seen) if isinstance(c, CodeType) else _canon_value(c, seen)
        for c in code.co_consts
    )
    return (
        f"code({code.co_code.hex()};flags={code.co_flags};argc={code.co_argcount};"
        f"kwonly={code.co_kwonlyargcount};names={','.join(code.co_names)};consts=[{consts}])"
    )


def _referenced_global_names(code: CodeType) -> set[str]:
    names = set(code.co_names)
    for c in code.co_consts:
        if isinstance(c, CodeType):
            names |= _referenced_global_names(c)
    return names


# The bytecode ops that name the *global/module namespace*, as opposed to an
# attribute (``LOAD_ATTR``/``LOAD_METHOD``/``STORE_ATTR``) or an import target.
_GLOBAL_NAME_OPS = frozenset({
    "LOAD_GLOBAL", "STORE_GLOBAL", "DELETE_GLOBAL",
    "LOAD_NAME", "STORE_NAME", "DELETE_NAME",
})


def _loaded_global_names(code: CodeType) -> set[str]:
    """Names this code loads, stores or deletes in the *global* namespace, from the instruction stream.

    ``co_names``, which :func:`_referenced_global_names` uses, pools global reads with attribute names: ``a.c1`` and
    ``cfg.SCALE`` both put ``c1`` and ``SCALE`` in ``co_names`` through ``LOAD_ATTR``, though neither is a global.
    Choosing the globals whose *values* are folded from ``co_names`` would therefore fold a module global that merely
    shares a name with a record field or a chained attribute. If that global's value has no canonical form, the walk
    would fail on it instead. Either way, the walk would over-invalidate the fingerprint, or, on the builder's derive
    path, force the address-bearing fallback and unbounded cache growth. The opcodes distinguish a ``LOAD_GLOBAL c1``
    from a ``LOAD_ATTR c1``. The function recurses into nested code objects, such as comprehensions and lambdas. The
    walker still folds a module attribute such as ``cfg.SCALE`` by the full ``co_names`` set, so a module attribute
    still re-keys the fingerprint. Only the base ``cfg`` is a global here, and this function finds it.
    """
    names: set[str] = set()
    for instr in dis.get_instructions(code):
        if instr.opname in _GLOBAL_NAME_OPS and isinstance(instr.argval, str):
            names.add(instr.argval)
    for c in code.co_consts:
        if isinstance(c, CodeType):
            names |= _loaded_global_names(c)
    return names


# These are the value types that numba freezes into a compiled artifact as constants. A module attribute of one of
# these types is data, and its value must enter the fingerprint. A Dispatcher module attribute is folded too, because
# numba links it and freezes its own captured globals and closure, exactly as for a direct global dispatcher. A ufunc,
# a plain function or a type is a stable reference that numba resolves by identity, or by its own builtin lowering, so
# it is skipped.
_FROZEN_DATA_TYPES = (bool, int, float, complex, str, bytes,
                      np.generic, np.ndarray, tuple, list, set, frozenset, dict)


def _fingerprint_module_attrs(referenced: set[str], namespace: dict, seen: set[int]) -> str:
    """Fold the values of ``<module>.<attr>`` reads that numba freezes as constants.

    ``_fingerprint_function`` folds the referenced globals present in ``__globals__``, but a value read through module
    indirection is invisible to it: ``import cfg; cfg.SCALE``, the chained ``cfg.sub.SCALE``, or a closure-captured
    module. The module canonicalizes to ``module(<name>)``, and the leaf ``SCALE`` is not itself a global. numba
    freezes the leaf value into the binary, so after a later change numba would load a stale binary from the cache.

    ``namespace`` maps the referenced globals AND the closure free variables to their values. For each module
    reachable there, the function folds every referenced name in the module's ``__dict__`` that is frozen data or a
    Dispatcher. numba freezes a Dispatcher's own captured globals and closure when it links the call, so
    ``_canon_value`` folds the Dispatcher exactly as it folds a direct global dispatcher. The function recurses into
    referenced submodules, with cycle protection by module id. Ufuncs, plain functions and types are stable
    references that numba resolves by identity, so they are skipped. The formula then stays cacheable, and the walk
    does not recurse into a module such as numpy.

    The scan tests ``__dict__`` membership and never calls ``getattr``, so it triggers no module ``__getattr__``.
    Resolving the referenced names through ``getattr`` would call an unrelated module's PEP 562 ``__getattr__``, such
    as numpy's, on every co-name collision, and that call imports submodules and emits deprecation warnings. The cost
    is a documented limitation: a value that a module exposes ONLY through PEP 562 ``__getattr__`` is not folded.
    numba freezes such a value, so clear the cache after changing it."""
    attrs: list[str] = []
    mods_seen: set[int] = set()

    def _walk(prefix: str, mod: ModuleType) -> None:
        if id(mod) in mods_seen:
            return
        mods_seen.add(id(mod))
        mod_dict = getattr(mod, "__dict__", {})
        for attr in sorted(referenced):
            if attr not in mod_dict:
                continue
            value = mod_dict[attr]
            if isinstance(value, ModuleType):
                _walk(f"{prefix}.{attr}", value)
                continue
            if not (value is None or isinstance(value, _FROZEN_DATA_TYPES)
                    or isinstance(value, (Dispatcher, DUFunc))):
                continue
            try:
                canon = _canon_value(value, seen)
            except (_Unfingerprintable, RecursionError):
                continue
            attrs.append(f"{prefix}.{attr}={canon}")

    for name in sorted(namespace):
        val = namespace[name]
        if isinstance(val, ModuleType):
            _walk(name, val)
    return f";module_attrs=[{';'.join(sorted(attrs))}]" if attrs else ""


def _fingerprint_function(func: FunctionType, seen: set[int]) -> str:
    if id(func) in seen:
        return f"recursive({func.__qualname__})"
    seen = seen | {id(func)}
    code = func.__code__
    cells = []
    closure_vals = {}
    for name, cell in zip(code.co_freevars, func.__closure__ or ()):
        try:
            contents = cell.cell_contents
        except ValueError as e:
            raise _Unfingerprintable("empty closure cell") from e
        cells.append(f"{name}={_canon_value(contents, seen)}")
        closure_vals[name] = contents
    referenced = _referenced_global_names(code)
    global_names = _loaded_global_names(code)
    hashed_globals = []
    for name in sorted(global_names):
        if name in func.__globals__:
            hashed_globals.append(f"{name}={_canon_value(func.__globals__[name], seen)}")
    modattr_ns = {n: func.__globals__[n] for n in referenced if n in func.__globals__}
    modattr_ns.update(closure_vals)
    module_attrs = _fingerprint_module_attrs(referenced, modattr_ns, seen)
    return (
        f"func({func.__module__}:{func.__qualname__};{_fingerprint_codeobj(code, seen)};"
        f"defaults={_canon_value(func.__defaults__ or (), seen)};"
        f"kwdefaults={_canon_value(func.__kwdefaults__ or {}, seen)};"
        f"closure=[{';'.join(cells)}];globals=[{';'.join(hashed_globals)}]{module_attrs})"
    )


def _opaque_values_match(old: Sequence[Any], new: Sequence[Any]) -> bool:
    """Whether two runs of the best-effort walker collected the same un-canonical values.

    Two bodies that fingerprint alike are the same body except in whatever the walker could not canonicalize, so
    comparing exactly those values determines whether they are the same body. The comparison only has to hold within
    one process, so it may read an address, which the fingerprint may not.

    Identity decides the ordinary case: reloading a module rebinds the same ``@intrinsic`` object, and a factory
    called twice closes over the same one. The function tries equality next, for values that define it. A ctypes
    function pointer defines neither, so the function compares the address it holds together with the call that the
    prototype over it describes. The address alone is not the body. One symbol reached through two prototypes is two
    calls that numba types and lowers differently. Reading those as one value would give the second body the alias
    under which the first was published, silently and with no collision to report. The fingerprint itself folds
    :func:`_ctypes_call_shape` for a pointer it can name, so the two paths agree on what makes two pointers two calls.
    """
    if len(old) != len(new):
        return False
    for a, b in zip(old, new):
        if a is b:
            continue
        if isinstance(a, ctypes._CFuncPtr) and isinstance(b, ctypes._CFuncPtr):
            if (ctypes.cast(a, ctypes.c_void_p).value == ctypes.cast(b, ctypes.c_void_p).value
                    and _ctypes_call_shape(a) == _ctypes_call_shape(b)):
                continue
            return False
        try:
            if bool(a == b):
                continue
        except Exception:  # nosec B112 - an object that cannot answer == is not a match
            pass
        return False
    return True


def _fingerprint_function_best_effort(func: FunctionType, opaque: Optional[list] = None) -> str:
    """Return a best-effort fingerprint of ``func`` without ever raising.

    ``_fingerprint_function`` raises ``_Unfingerprintable`` on the *first* value with no canonical form and discards
    everything computed so far. A body that merely *references* an un-canonicalizable global (e.g. an ``@intrinsic``)
    therefore yields no usable fingerprint. This variant substitutes a stable ``<opaque:type>`` placeholder for any such
    value instead, so bodies differing only in a captured constant, closure cell, default, or a canonicalizable global
    still fingerprint distinctly. It serves consumers that need a discriminating identifier even for un-fingerprintable
    bodies, such as the ``@proxy`` cfunc alias and ``digest``. ``compile_kernel`` degrades to uncached through the
    raising contract of ``_fingerprint_function``. The result is process-stable: every placeholder is derived from a
    type name, never from an address.

    A placeholder can still give two different bodies one fingerprint. Pass ``opaque`` to collect the values that
    received a placeholder, in walk order. A caller that holds two equal fingerprints can then ask
    :func:`_opaque_values_match` whether they came from one body or from two that the fingerprint could not separate.
    """
    seen: set[int] = set()

    def _canon(value: Any) -> str:
        try:
            return _canon_value(value, seen)
        except (_Unfingerprintable, RecursionError):
            if opaque is not None:
                opaque.append(value)
            return f"<opaque:{type(value).__name__}>"

    def _canon_const(const: object) -> str:
        if isinstance(const, CodeType):
            inner = ",".join(_canon_const(c) for c in const.co_consts)
            return f"code({const.co_code.hex()};consts=[{inner}])"
        return _canon(const)

    code = func.__code__
    consts = ",".join(_canon_const(c) for c in code.co_consts)
    cells = []
    for name, cell in zip(code.co_freevars, func.__closure__ or ()):
        try:
            contents = cell.cell_contents
        except ValueError:
            cells.append(f"{name}=<empty-cell>")
            continue
        cells.append(f"{name}={_canon(contents)}")
    hashed_globals = [
        f"{name}={_canon(func.__globals__[name])}"
        for name in sorted(_loaded_global_names(code))
        if name in func.__globals__
    ]
    return (
        f"besteffort({func.__module__}:{func.__qualname__};"
        f"code({code.co_code.hex()};flags={code.co_flags};argc={code.co_argcount};"
        f"kwonly={code.co_kwonlyargcount};names={','.join(code.co_names)};consts=[{consts}]);"
        f"defaults={_canon(func.__defaults__ or ())};"
        f"kwdefaults={_canon(func.__kwdefaults__ or {})};"
        f"closure=[{';'.join(cells)}];globals=[{';'.join(hashed_globals)}])"
    )


# ---- Effective jit flags ----
#
# Every content-addressed cache key that must re-key when the resolved jit flags change shares these helpers: the
# `compile_kernel` digest and the `make_graph` kernel fingerprint. They are defined here rather than in
# `compile_kernel` so that `builder` can reuse them without importing that module, whose import runs an orphan-anchor
# sweep. The NUMBA_* settings those digests fold come from `numbox.core.cache_key`.

def _effective_flags(jit_options: dict | None) -> dict:
    """The non-`cache` jit flags, resolved against the numbox defaults.

    `cache` is excluded deliberately: whether an artifact is written to disk does
    not change the emitted binary, so folding it would split the key needlessly.
    Every other flag (`fastmath`, `error_model`, `boundscheck`, ...) does change
    the binary and must re-key.
    """
    opts = {**_default_jit_options, **(jit_options or {})}
    return {k: v for k, v in opts.items() if k != "cache"}


def _flags_canon(flags: dict) -> tuple[str, bool]:
    """Canonical string for the effective jit flags, and whether it is a true canonicalization.

    The second item is ``False`` if a flag value had no canonical form, e.g. a ``pipeline_class`` or numba-typed
    ``locals``. A ``False`` here means the caller must degrade to uncached: a key that does not include a flag cannot
    distinguish the binaries that different values of the flag produce."""
    try:
        return _canon_value(flags, set()), True
    except (_Unfingerprintable, RecursionError):
        return repr(sorted(flags.items(), key=repr)), False
