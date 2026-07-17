"""Content fingerprinting of Python values and functions for cache keys.

A value canonicalizer (`_canon_value`) and a deep function fingerprint
(`_fingerprint_function`) that capture everything numba freezes into a compiled
artifact: code-object bytecode/consts/names, default arguments, closure-cell
values, and the values of referenced module-level globals (recursing into helper
functions and dispatchers, with cycle protection). Stronger than hashing the
bare code object -- two functions with identical source but different captured
closure/global values fingerprint differently. Shared by
`numbox.core.variable.compile_kernel` (kernel cache digest) and
`numbox.utils.digest` (SQLite UDAF cache key).

`_Unfingerprintable` is raised for any value with no canonical form; callers
decide how to degrade (compile_kernel marks the kernel uncached, digest falls
back to cloudpickle of the code object).
"""
import hashlib

from types import CodeType, FunctionType, ModuleType
from typing import Any

import numpy as np

from numba.core.dispatcher import Dispatcher


class _Unfingerprintable(Exception):
    """A value with no canonical fingerprint; the caller decides how to degrade."""


def _safe_repr(obj: object) -> str:
    """``repr(obj)`` that never raises -- the fingerprint fallback must always
    yield a string so an un-fingerprintable formula degrades to uncached rather
    than crashing when its ``__repr__`` itself raises."""
    try:
        return repr(obj)
    except Exception:  # noqa: BLE001 - fallback must not crash on a raising __repr__
        return f"<{type(obj).__name__} repr-failed>"


def _dtype_key(dtype: np.dtype) -> str:
    """Canonical string for a numpy dtype that preserves structured layout.

    ``dtype.str`` collapses any structured dtype to ``|V<itemsize>`` -- field
    names, types, order and offsets are all erased, so two same-byte different-
    layout arrays would collide. For a structured dtype fold its full
    descriptor and itemsize instead."""
    if dtype.names is not None:
        return f"struct({dtype!s};descr={dtype.descr};itemsize={dtype.itemsize})"
    return dtype.str


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
        if proxy_alias is not None:
            # A @proxy binding: its content-addressed alias fully identifies the
            # body and signature. Walking its wrapper would recurse into the
            # @intrinsic it calls, which has no canonical form.
            return f"proxy({proxy_alias})"
        topts = _canon_value(dict(getattr(value, "targetoptions", {}) or {}), seen)
        return f"dispatcher({_fingerprint_function(value.py_func, seen)};{topts})"
    if isinstance(value, FunctionType):
        return f"function({_fingerprint_function(value, seen)})"
    raise _Unfingerprintable(type(value).__name__)


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


# The value types numba freezes into a compiled artifact as constants -- the
# data module attributes whose value must enter the fingerprint. A Dispatcher
# module attribute is folded too (numba links it and freezes its own captured
# globals/closure, exactly as for a direct global dispatcher). A ufunc, plain
# function, or type is a stable reference numba resolves by identity (or by its
# own builtin lowering), so it is skipped.
_FROZEN_DATA_TYPES = (bool, int, float, complex, str, bytes,
                      np.generic, np.ndarray, tuple, list, set, frozenset, dict)


def _fingerprint_module_attrs(referenced: set[str], namespace: dict, seen: set[int]) -> str:
    """Fold the values of ``<module>.<attr>`` reads that numba freezes as constants.

    ``_fingerprint_function`` folds referenced globals present in ``__globals__``,
    but a value read through module indirection (``import cfg; cfg.SCALE``, or the
    chained ``cfg.sub.SCALE``, or a closure-captured module) is invisible: the
    module canonicalizes to ``module(<name>)`` and the leaf ``SCALE`` is not itself
    a global. numba bakes the leaf value into the binary, so a later change would be
    served stale. ``namespace`` maps referenced globals AND closure free-variables
    to their values; for each module reachable there, fold every referenced name in
    its ``__dict__`` that is frozen data or a Dispatcher (whose own captured
    globals/closure numba freezes when it links the call -- folded via
    ``_canon_value`` exactly as a direct global dispatcher is), recursing into
    referenced submodules (with module-id cycle protection). Ufuncs, plain
    functions and types are stable references numba resolves by identity, so they
    are skipped (the formula stays cacheable, no recursion into e.g. numpy).
    ``__dict__`` membership (not ``getattr``) avoids triggering a module
    ``__getattr__`` (lazy import)."""
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
                    or isinstance(value, Dispatcher)):
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
    hashed_globals = []
    for name in sorted(referenced):
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


def _fingerprint_function_best_effort(func: FunctionType) -> str:
    """Best-effort function fingerprint that never raises.

    ``_fingerprint_function`` raises ``_Unfingerprintable`` on the *first* value
    with no canonical form and discards everything computed so far, so a body
    that merely *references* an un-canonicalizable global (e.g. an
    ``@intrinsic``) collapses to nothing usable. This variant substitutes a
    stable ``<opaque:type>`` placeholder for any such value instead, so bodies
    differing only in a captured constant, closure cell, default, or a
    canonicalizable global still fingerprint distinctly. For consumers that need
    a discriminating identifier even for un-fingerprintable bodies -- the
    ``@proxy`` cfunc alias -- rather than the degrade-to-uncached path that
    ``_fingerprint_function``'s raising contract serves (``compile_kernel``,
    ``digest``). Process-stable: every placeholder is derived from a type name,
    never an address.
    """
    seen: set[int] = set()

    def _canon(value: Any) -> str:
        try:
            return _canon_value(value, seen)
        except (_Unfingerprintable, RecursionError):
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
        for name in sorted(_referenced_global_names(code))
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
