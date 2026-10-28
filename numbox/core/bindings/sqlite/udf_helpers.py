"""Higher-level registration helpers for structref-backed SQLite UDAFs.

``register_aggregate`` and ``register_window`` generate the SQLite callback functions (xStep, xInverse, xValue and
xFinal) that perform the per-group state lifecycle. The callbacks allocate a single intp slot with
``sqlite3_aggregate_context`` and guard against a NULL result. They create the state on the first step through
``export_meminfo``, keep it in that slot and borrow it with ``borrow_structref``. xFinal releases the state, and xValue
does not. Callers therefore write only the state logic of ``init``, ``step`` and ``finalize`` (and ``inverse`` and
``value`` for windows), as plain Python or as already-jitted (``@njit`` or ``@proxy``) functions.

**Exception handling.** Each generated callback wraps the call of the user step, inverse, value or finalize callback
in a ``try``/``except``. When the user callback raises, the handler reports a descriptive error through
``sqlite3_result_error``, for example "error in user step callback" with the code ``SQLITE_ERROR``. Without the
handler, a numba ``@cfunc`` discards the exception: it prints "Exception ignored" and returns the zero default without
unwinding into SQLite, so the query returns a wrong result without an error. The catch inside the body also lets
numba decrement the reference count of the borrowed state. The unwind would skip that decrement and leak one meminfo
per group. Only ``xFinal`` releases the slot.

**Mechanism.** numbox generates the callback source of each UDAF with the state type and the user functions as module
globals, so the calls are inlined. It writes the source to a content-addressed anchor file under numba's cache
directory, with the helpers of numbox.utils.preprocessing. The impls (``@njit(**jit_options)``, the numbox-wide
configuration with the default ``cache=True``) cache across processes. The content hash of the anchor includes a
fingerprint of each user function's body, closure cells, defaults and referenced globals (``numbox.utils.digest``). If a
body changes, even only in a numeric literal, the cache therefore invalidates correctly.

**Caller requirements.** Callbacks may be plain Python functions or already-jitted callables (``@njit`` or ``@proxy``),
and ``njit`` compiles the plain ones (see ``_prepare_callbacks``). The generated impls (``@njit(**jit_options)``,
default ``cache=True``) cache and invalidate correctly even when the state-type class and the callbacks are defined in
``__main__``. The anchor key is content-addressed on the digest of each callback, which fingerprints its body, closure
cells, defaults and referenced globals, together with ``repr(state_type)``, the resolved ``jit_options`` and the numba
and numbox versions, and never on ``__module__`` (see ``numbox.utils.digest.digest``). The subprocess tests
``test_xprocess_cache_no_growth`` and ``test_invalidation_on_literal_edit`` exercise this ``__main__`` path. Only a
*caller-side* ``@njit(cache=True)`` callback requires a stable ``__module__``, because numba does not cache functions
defined in ``__main__``. If the callbacks are themselves cached, define the state-type class and the callbacks in an
importable module.
"""
import ctypes

from numba import cfunc, types
from numba.core.types import StructRef
from numba.extending import is_jitted

from numbox.core.bindings.sqlite.conn import sqlite3_errmsg
from numbox.core.bindings.sqlite.constants import SQLITE_DETERMINISTIC, SQLITE_OK, SQLITE_UTF8
from numbox.core.bindings.sqlite.udf import sqlite3_create_function_v2, sqlite3_create_window_function
from numbox.utils.cstrings import c_string
from numbox.utils.digest import digest
from numbox.utils.preprocessing import (
    _anchor_path,
    _anchored_or_uncached,
    _orphan_anchor_sweep,
)

# The generated anchor source references these names. Importing them here adds them to this module's __dict__, and
# _compile_callbacks copies that __dict__ into the exec namespace.
import numpy as np  # noqa: F401
from numba import carray, njit  # noqa: F401
from numbox.core.configurations import jit_options  # noqa: F401
from numbox.core.bindings.sqlite.result import sqlite3_result_error  # noqa: F401
from numbox.core.bindings.sqlite.udf import sqlite3_aggregate_context  # noqa: F401
from numbox.utils.lowlevel import _cast_int_to_void_p, get_unicode_data_p  # noqa: F401
from numbox.utils.meminfo import (  # noqa: F401
    borrow_structref,
    export_meminfo,
    release_meminfo,
)

__all__ = ["register_aggregate", "register_window"]

_ANCHOR_SUBDIR = "numbox-sqlite-udaf"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)


# Generated-source templates. The generated source has these module globals: _state_type, _init, _step, _finalize,
# _inverse, _value and jit_options. Each impl is @njit(**jit_options), the numbox-wide configuration with the default
# cache=True, so it caches across processes. The calls of the user functions are inlined, because the functions are
# module globals here.
_XSTEP_SRC = '''
@njit(**jit_options)
def _xstep_impl(ctx, argc, argv_pp):
    agg = sqlite3_aggregate_context(ctx, 8)
    if agg == 0:
        return
    slot = carray(_cast_int_to_void_p(agg), (1,), dtype=np.intp)
    if slot[0] == 0:
        try:
            slot[0] = export_meminfo(_init())
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user init callback"), -1)
            return
    try:
        _step(borrow_structref(_state_type, slot[0]), ctx, argc, argv_pp)
    except Exception:
        sqlite3_result_error(ctx, get_unicode_data_p("error in user step callback"), -1)
'''

_XFINAL_SRC = '''
@njit(**jit_options)
def _xfinal_impl(ctx):
    agg = sqlite3_aggregate_context(ctx, 0)
    if agg == 0:
        try:
            _state = _init()
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user init callback"), -1)
            return
        try:
            _finalize(_state, ctx)
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user finalize callback"), -1)
        return
    slot = carray(_cast_int_to_void_p(agg), (1,), dtype=np.intp)
    if slot[0] == 0:
        try:
            _state = _init()
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user init callback"), -1)
            return
        try:
            _finalize(_state, ctx)
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user finalize callback"), -1)
        return
    try:
        _finalize(borrow_structref(_state_type, slot[0]), ctx)
    except Exception:
        sqlite3_result_error(ctx, get_unicode_data_p("error in user finalize callback"), -1)
    release_meminfo(slot[0])
'''

_XINVERSE_SRC = '''
@njit(**jit_options)
def _xinverse_impl(ctx, argc, argv_pp):
    agg = sqlite3_aggregate_context(ctx, 0)
    if agg == 0:
        return
    slot = carray(_cast_int_to_void_p(agg), (1,), dtype=np.intp)
    if slot[0] == 0:
        return
    try:
        _inverse(borrow_structref(_state_type, slot[0]), ctx, argc, argv_pp)
    except Exception:
        sqlite3_result_error(ctx, get_unicode_data_p("error in user inverse callback"), -1)
'''

_XVALUE_SRC = '''
@njit(**jit_options)
def _xvalue_impl(ctx):
    agg = sqlite3_aggregate_context(ctx, 0)
    if agg == 0:
        try:
            _state = _init()
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user init callback"), -1)
            return
        try:
            _value(_state, ctx)
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user value callback"), -1)
        return
    slot = carray(_cast_int_to_void_p(agg), (1,), dtype=np.intp)
    if slot[0] == 0:
        try:
            _state = _init()
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user init callback"), -1)
            return
        try:
            _value(_state, ctx)
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error in user value callback"), -1)
        return
    try:
        _value(borrow_structref(_state_type, slot[0]), ctx)
    except Exception:
        sqlite3_result_error(ctx, get_unicode_data_p("error in user value callback"), -1)
'''


def _validate_state_type(state_type):
    if not isinstance(state_type, StructRef):
        raise TypeError(
            "state_type must be a numba StructRef instance "
            "(e.g. MyStateType([('x', int64)])), got %r" % (state_type,))


def _prepare_callbacks(**fns):
    """Return ``fns`` with each callback as a numba-compiled callable.

    An already jitted callable (``@njit``, ``@proxy``, ...) passes through, and a plain Python function is wrapped
    with ``njit``. The callbacks are inlined into the generated ``@njit(cache=True)`` impls, so a lazy ``njit`` with no
    explicit signature suffices: the impl determines the specialization. The content-addressed anchor, keyed on
    ``py_func.__code__``, provides the cross-process caching, so the callbacks need no ``cache=True`` of their own.
    """
    prepared = {}
    for role, fn in fns.items():
        if is_jitted(fn):
            prepared[role] = fn
        elif callable(fn):
            prepared[role] = njit(fn)
        else:
            raise TypeError(
                "%s must be a callable (plain Python or @njit), got %r" % (role, fn))
    return prepared


def _compile_callbacks(stem, srcs, state_type, fns):
    """Generate the @njit(cache=True) impls, anchor them at a content-addressed file and exec them.

    ``fns`` maps the generated global names (``_init``, ``_step``, ...) to the user callables. Returns the exec
    namespace, which contains ``_xstep_impl`` and the other impls.
    """
    # repr(state_type) gives only the class qualname and the field list, so StructRef classes with the same qualname
    # in different modules would collide. The digest subject therefore includes the defining module and the qualname.
    # An @overload_method attached to the type outside this registration is still not captured here. For a type that
    # make_structref builds, the method hash includes the body of such a method.
    state_id = "%s.%s;%r" % (type(state_type).__module__, type(state_type).__qualname__, state_type)
    udaf_digest = digest(state_id, list(fns.values()))
    code_txt = "# udaf-digest: %s\n%s" % (udaf_digest, "".join(srcs))
    # globals() is this module's __dict__, and it carries __name__. numba needs __name__ to rebuild the Environment
    # when it reloads the cached impls from a warm cache.
    anchor = _anchor_path(_ANCHOR_SUBDIR, stem, code_txt)
    ns = {**globals(), "_state_type": state_type, **fns,
          "jit_options": _anchored_or_uncached(anchor, code_txt, jit_options)}
    code = compile(code_txt, str(anchor), mode="exec")
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns


def _raise_rc(db, name, rc):
    msg_p = sqlite3_errmsg(db)
    detail = ""
    if msg_p:
        detail = ": " + ctypes.cast(
            msg_p, ctypes.c_char_p).value.decode("utf-8", "replace")
    raise RuntimeError(
        "sqlite3 UDAF registration failed for %r (rc=%d)%s" % (name, rc, detail))


def _stem(prefix, name):
    return prefix + "".join(c if c.isalnum() else "_" for c in name)


def register_aggregate(db, name, n_arg, state_type, init, step, finalize,
                       *, deterministic=False):
    """Register a structref-backed aggregate UDAF.

    Callbacks may be plain Python or already-jitted (``@njit`` or ``@proxy``), and ``njit`` compiles the plain
    functions.

    :param db: connection pointer (intp), as returned by ``sqlite3_open``.
    :param name: SQL function name (str); this function handles the lifetime of the C string.
    :param n_arg: argument count, or -1 for variadic.
    :param state_type: the numba structref *instance* type for per-group state.
    :param init: ``() -> state`` returning a fresh state.
    :param step: ``(state, ctx, argc, argv_pp)`` updating state.
    :param finalize: ``(state, ctx)`` writing the result.
    :param deterministic: if true, OR ``SQLITE_DETERMINISTIC`` into the flags.

    Returns ``None``. The caller need not retain the generated callbacks, because numba keeps the compiled cfunc code
    and the dispatchers resident for the lifetime of the process.
    """
    _validate_state_type(state_type)
    fns = _prepare_callbacks(init=init, step=step, finalize=finalize)
    ns = _compile_callbacks(
        _stem("udaf_", name), [_XSTEP_SRC, _XFINAL_SRC], state_type,
        {"_init": fns["init"], "_step": fns["step"], "_finalize": fns["finalize"]})
    xstep_impl = ns["_xstep_impl"]
    xfinal_impl = ns["_xfinal_impl"]

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def step_cb(ctx, argc, argv):
        xstep_impl(ctx, argc, argv)

    @cfunc(types.void(types.intp))
    def final_cb(ctx):
        xfinal_impl(ctx)

    flags = SQLITE_UTF8 | (SQLITE_DETERMINISTIC if deterministic else 0)
    with c_string(name) as name_p:
        rc = sqlite3_create_function_v2(
            db, name_p, n_arg, flags, 0, 0,
            step_cb.address, final_cb.address, 0)
    if rc != SQLITE_OK:
        _raise_rc(db, name, rc)


def register_window(db, name, n_arg, state_type, init, step, inverse, value,
                    finalize, *, deterministic=False):
    """Register a structref-backed window UDAF.

    The parameters are those of :func:`register_aggregate`, plus ``inverse(state, ctx, argc, argv_pp)`` and
    ``value(state, ctx)``. ``inverse`` removes the contribution of a row from a state that already exists. ``value``
    emits the running result without releasing the state. Only ``xFinal`` releases the meminfo.
    """
    _validate_state_type(state_type)
    fns = _prepare_callbacks(init=init, step=step, inverse=inverse, value=value,
                             finalize=finalize)
    ns = _compile_callbacks(
        _stem("wudaf_", name),
        [_XSTEP_SRC, _XINVERSE_SRC, _XVALUE_SRC, _XFINAL_SRC], state_type,
        {"_init": fns["init"], "_step": fns["step"], "_inverse": fns["inverse"],
         "_value": fns["value"], "_finalize": fns["finalize"]})
    xstep_impl = ns["_xstep_impl"]
    xinverse_impl = ns["_xinverse_impl"]
    xvalue_impl = ns["_xvalue_impl"]
    xfinal_impl = ns["_xfinal_impl"]

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def step_cb(ctx, argc, argv):
        xstep_impl(ctx, argc, argv)

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def inverse_cb(ctx, argc, argv):
        xinverse_impl(ctx, argc, argv)

    @cfunc(types.void(types.intp))
    def value_cb(ctx):
        xvalue_impl(ctx)

    @cfunc(types.void(types.intp))
    def final_cb(ctx):
        xfinal_impl(ctx)

    flags = SQLITE_UTF8 | (SQLITE_DETERMINISTIC if deterministic else 0)
    with c_string(name) as name_p:
        rc = sqlite3_create_window_function(
            db, name_p, n_arg, flags, 0,
            step_cb.address, final_cb.address,
            value_cb.address, inverse_cb.address, 0)
    if rc != SQLITE_OK:
        _raise_rc(db, name, rc)
