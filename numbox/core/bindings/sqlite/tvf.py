"""Expose a numpy structured array, computed per query, as a SQLite table-valued function (register_tvf).

``register_tvf(db, name, arg_types, out_dtype, fn)`` registers an eponymous virtual table whose rows are not static.
The user's ``fn(*args)`` produces them for each query and returns them as a 1-D numpy structured array of
``out_dtype``. The function arguments are exposed as HIDDEN columns. In ``SELECT * FROM f(2, 5)``, SQLite passes ``2``
and ``5`` as EQ constraints on the hidden columns. ``xBestIndex`` binds these constraints into ``argv`` in declaration
order, and ``xFilter`` decodes them, passes them to ``fn`` and keeps them in the cursor. A query that names a hidden
column therefore receives the argument of that column.

The cursor holds the returned array for its own lifetime through an NRT-backed ``[meminfo_p, data_p]`` slot, and
``xClose`` releases the array exactly once. The generated ``xFilter`` takes that reference with the inlined
``_incref_meminfo`` intrinsic, so numba's refcount pass cannot remove it.

Unlike the read-only ``register_table``, which shares one module, each ``register_tvf`` call generates its own
``xFilter`` and ``xColumn`` and builds its own ``sqlite3_module`` from the cfunc addresses of this registration. The
user ``fn`` is a global of the generated code, and ``fn`` and ``out_dtype`` enter the anchor's digest, so each dtype
gets its own cached ``xFilter`` across processes. The handle holds the module struct, every cfunc object (SQLite stores
their addresses), the descriptor buffers and ``fn``. The module-level ``_DATA_ANCHOR`` holds the keep-alive of the
handle, and SQLite releases it through ``xDestroy``.
"""
import ctypes

import numpy as np
from numba import cfunc, types

from numbox.core.bindings.sqlite.constants import (
    SQLITE_OK, SQLITE_ERROR, SQLITE_NOMEM, SQLITE_CONSTRAINT,
    SQLITE_INDEX_CONSTRAINT_EQ,
)
from numbox.core.bindings.sqlite._typemap import _col_tag, _SQL_TYPE, tags_buf_t
from numbox.core.bindings.sqlite.vtable import (
    _Sqlite3Module, _SQLITE3_VTAB_CURSOR_DTYPE, _VTAB_DTYPE, _VTAB_SIZE,
    _IDX_INFO_DTYPE, _CONSTRAINT_DTYPE, _USAGE_DTYPE,
    _emit_cell, _register_with_destroy,
)
from numbox.utils.digest import digest
from numbox.utils.preprocessing import (
    _anchor_path, _anchored_or_uncached, _orphan_anchor_sweep,
)

# The generated source references these names. Importing them here adds them to this module's __dict__, and
# _compile_xfilter copies that __dict__ into the exec namespace.
from numba import carray, njit  # noqa: F401
from numba.core.types import (  # noqa: F401
    int32, int64, float64,
)
from numbox.core.bindings.sqlite.vtable import sqlite3_declare_vtab  # noqa: F401
from numbox.core.bindings.sqlite.exec import sqlite3_malloc, sqlite3_free  # noqa: F401
from numbox.core.bindings.sqlite.value import (  # noqa: F401
    sqlite3_value_int64, sqlite3_value_double,
)
from numbox.core.bindings.sqlite.result import (  # noqa: F401
    sqlite3_result_int64, sqlite3_result_double,
    sqlite3_result_text, sqlite3_result_blob, sqlite3_result_error,
)
from numbox.core.bindings.sqlite._typemap import (  # noqa: F401
    _TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64, _TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64,
    _TAG_F32, _TAG_F64, _TAG_BOOL, _TAG_S, _TAG_U, _TAG_BLOB,
    _nul_trimmed_len, utf32_to_utf8,
)
from numbox.core.configurations import jit_options  # noqa: F401
from numbox.utils.lowlevel import (  # noqa: F401
    _cast_int_to_void_p, get_unicode_data_p, load_unaligned, store_at, array_data_p,
)
from numbox.utils.meminfo import (  # noqa: F401
    structref_meminfo, _incref_meminfo, release_meminfo,
)

__all__ = ["register_tvf"]

_CACHE = jit_options.get("cache", True)
_SQLITE_TRANSIENT = -1

_ANCHOR_SUBDIR = "numbox-sqlite-tvf"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)


# The layout of the per-registration descriptor, which SQLite receives as pClientData. xConnect, xBestIndex and
# xColumn read its fields by name through carray(ptr, (1,), dtype=_TVF_DESC_DTYPE). The schema declares the visible
# (out_dtype) columns first and the n_hidden argument columns after them. col_tags has an entry for every schema
# column, the hidden arguments included, while col_offsets and col_widths cover only the visible columns.
_TVF_DESC_DTYPE = np.dtype([
    ("ncols", "i4"), ("n_hidden", "i4"), ("itemsize", "i8"),
    ("col_offsets", "i8"), ("col_tags", "i8"), ("col_widths", "i8"),
    ("schema_ptr", "i8"), ("scratch_bytes", "i8"),
], align=True)

_TVF_CUR_DTYPE = np.dtype([
    ("base", _SQLITE3_VTAB_CURSOR_DTYPE), ("descriptor", "i8"), ("rowid", "i8"),
    ("mi_p", "i8"), ("data_p", "i8"), ("n_rows", "i8"), ("row_stride", "i8"), ("scratch_p", "i8"),
    ("args_p", "i8"),
], align=True)
_TVF_CUR_SIZE = _TVF_CUR_DTYPE.itemsize
# Bytes per hidden argument in the cursor's args_p buffer: xFilter stores each argument as the int64 or float64
# that it decoded.
_ARG_SLOT = 8

_INT_TAGS = frozenset((_TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64,
                       _TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64, _TAG_BOOL))
_FLOAT_TAGS = frozenset((_TAG_F32, _TAG_F64))


# --- generated-source templates -------------------------------------------
# _compile_xfilter execs the template in a namespace that holds _fn (the user callable), _N_HIDDEN, jit_options and
# every helper imported above. The {arg_decode} and {fn_call} substitutions are the only parts that vary with the
# arity. _compile_xfilter generates them from arg_types before exec, exactly as the UDAF helpers generate per-UDAF
# source.
#
# Every registration execs this one template, but caching stays correct across out_dtypes. Each registration gets a
# distinct anchor and cache key from digest((out_dtype, tuple(arg_tags)), [fn]). The fixed return type of _fn, which
# out_dtype sets, also makes numba specialize the implementation, so two out_dtypes never collide.
_XFILTER_SRC = '''
@njit(**jit_options)
def _tvf_xfilter_impl(cur, argc, argv):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
    if c[0].mi_p != 0:
        release_meminfo(c[0].mi_p)
        c[0].mi_p = 0
        c[0].data_p = 0
        c[0].n_rows = 0
        c[0].row_stride = 0
    c[0].rowid = 0
    if argc < _N_HIDDEN:
        return
    vals = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
{arg_decode}
    result = {fn_call}
    mi_p, _base = structref_meminfo(result)
    _incref_meminfo(mi_p)
    c[0].mi_p = mi_p
    c[0].data_p = array_data_p(result)
    c[0].n_rows = result.shape[0]
    c[0].row_stride = result.strides[0]
'''


def _gen_arg_decode(arg_tags):
    lines = []
    for i, tag in enumerate(arg_tags):
        if tag in _INT_TAGS:
            lines.append("    a%d = sqlite3_value_int64(vals[%d])" % (i, i))
        else:
            lines.append("    a%d = sqlite3_value_double(vals[%d])" % (i, i))
        # The generated code stores the argument in the cursor, and xColumn returns it as the hidden column's value.
        lines.append("    store_at(c[0].args_p + %d, a%d)" % (_ARG_SLOT * i, i))
    return "\n".join(lines)


# --- per-registration cfunc factories -------------------------------------
def _make_xconnect():
    @cfunc(types.int32(types.intp, types.intp, types.int32, types.intp, types.intp, types.intp),
           cache=_CACHE)
    def _tvf_xconnect(db, p_aux, argc, argv, pp_vtab, pz_err):
        vtab = 0
        try:
            d = carray(_cast_int_to_void_p(p_aux), (1,), dtype=_TVF_DESC_DTYPE)
            rc = sqlite3_declare_vtab(db, d[0].schema_ptr)
            if rc != SQLITE_OK:
                return rc
            vtab = sqlite3_malloc(int32(_VTAB_SIZE))
            if vtab == 0:
                return SQLITE_NOMEM
            # The vtab needs no zeroing, because SQLite memsets the sqlite3_vtab base after this callback returns.
            # The descriptor field is the only member that this module owns.
            v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
            v[0].descriptor = p_aux
            slot = carray(_cast_int_to_void_p(pp_vtab), (1,), dtype=np.intp)
            slot[0] = vtab
            return SQLITE_OK
        except Exception:
            sqlite3_free(vtab)
            return SQLITE_ERROR
    return _tvf_xconnect


def _make_xbestindex():
    @cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
    def _tvf_xbestindex(vtab, idx_info):
        # Every hidden argument column requires a usable EQ constraint, and each constraint binds to argv in
        # declaration order. A binding sets argvIndex to the hidden argument's index plus 1 and omit to 1, so SQLite
        # passes the value and drops the synthetic constraint. If any hidden argument is unbound, the callback returns
        # SQLITE_CONSTRAINT, so SQLite rejects the plan cleanly and never calls xFilter with too few arguments.
        v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
        d = carray(_cast_int_to_void_p(v[0].descriptor), (1,), dtype=_TVF_DESC_DTYPE)
        ii = carray(_cast_int_to_void_p(idx_info), (1,), dtype=_IDX_INFO_DTYPE)
        ncols = d[0].ncols
        n_hidden = d[0].n_hidden
        n_constraint = ii[0].nConstraint
        cons = carray(_cast_int_to_void_p(ii[0].aConstraint), (n_constraint,), dtype=_CONSTRAINT_DTYPE)
        usage = carray(_cast_int_to_void_p(ii[0].aConstraintUsage), (n_constraint,), dtype=_USAGE_DTYPE)

        # The loop binds each hidden argument separately, to the first usable EQ on its column. It keeps no running
        # count, because a duplicate usable EQ on one argument must not mask another argument that is unbound. It
        # keeps no bitmask either, because a 64-bit mask would limit the arguments to 63. A second EQ on the same
        # argument, as in f(1) WHERE arg0 = 1, gets no argvIndex, because SQLite refuses a plan in which two
        # constraints share one argvIndex. That EQ is not omitted, so SQLite checks it against the hidden column.
        # The scan is O(n_hidden * n_constraint). It takes microseconds for a few arguments and about a millisecond
        # at SQLite's default cap of 2000 columns, and it runs only when SQLite plans a query.
        for h in range(n_hidden):
            bound = False
            for i in range(n_constraint):
                if (cons[i].usable != 0 and cons[i].op == SQLITE_INDEX_CONSTRAINT_EQ
                        and cons[i].iColumn - ncols == h):
                    usage[i].argvIndex = int32(h + 1)
                    usage[i].omit = 1
                    bound = True
                    break
            if not bound:
                return SQLITE_CONSTRAINT
        ii[0].idxNum = int32(1)
        ii[0].estimatedCost = float64(1)
        ii[0].estimatedRows = 16
        return SQLITE_OK
    return _tvf_xbestindex


def _make_static_cfuncs():
    @cfunc(types.int32(types.intp), cache=_CACHE)
    def _tvf_xdisconnect(vtab):
        try:
            sqlite3_free(vtab)
            return SQLITE_OK
        except Exception:
            return SQLITE_ERROR

    @cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
    def _tvf_xopen(vtab, pp_cursor):
        cur = 0
        scratch_p = 0
        args_p = 0
        try:
            v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
            desc = v[0].descriptor
            d = carray(_cast_int_to_void_p(desc), (1,), dtype=_TVF_DESC_DTYPE)
            scratch = d[0].scratch_bytes
            n_hidden = d[0].n_hidden
            cur = sqlite3_malloc(int32(_TVF_CUR_SIZE))
            if cur == 0:
                return SQLITE_NOMEM
            if scratch > 0:
                scratch_p = sqlite3_malloc(int32(scratch))
                if scratch_p == 0:
                    sqlite3_free(cur)
                    return SQLITE_NOMEM
            if n_hidden > 0:
                args_p = sqlite3_malloc(int32(_ARG_SLOT * n_hidden))
                if args_p == 0:
                    sqlite3_free(scratch_p)
                    sqlite3_free(cur)
                    return SQLITE_NOMEM
            c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
            c[0].base.pVtab = vtab
            c[0].descriptor = desc
            c[0].rowid = 0
            c[0].mi_p = 0
            c[0].data_p = 0
            c[0].n_rows = 0
            c[0].row_stride = 0
            c[0].scratch_p = scratch_p
            c[0].args_p = args_p
            slot = carray(_cast_int_to_void_p(pp_cursor), (1,), dtype=np.intp)
            slot[0] = cur
            return SQLITE_OK
        except Exception:
            sqlite3_free(args_p)
            sqlite3_free(scratch_p)
            sqlite3_free(cur)
            return SQLITE_ERROR

    @cfunc(types.int32(types.intp), cache=_CACHE)
    def _tvf_xclose(cur):
        try:
            c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
            if c[0].mi_p != 0:
                release_meminfo(c[0].mi_p)
                c[0].mi_p = 0
                c[0].data_p = 0
                c[0].n_rows = 0
                c[0].row_stride = 0
            sqlite3_free(c[0].args_p)
            sqlite3_free(c[0].scratch_p)
            sqlite3_free(cur)
            return SQLITE_OK
        except Exception:
            return SQLITE_ERROR

    @cfunc(types.int32(types.intp), cache=_CACHE)
    def _tvf_xnext(cur):
        c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
        c[0].rowid = c[0].rowid + 1
        return SQLITE_OK

    @cfunc(types.int32(types.intp), cache=_CACHE)
    def _tvf_xeof(cur):
        c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
        if c[0].data_p == 0 or c[0].rowid >= c[0].n_rows:
            return 1
        return 0

    @cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
    def _tvf_xrowid(cur, p_rowid):
        c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
        store_at(p_rowid, c[0].rowid)
        return SQLITE_OK

    return _tvf_xdisconnect, _tvf_xopen, _tvf_xclose, _tvf_xnext, _tvf_xeof, _tvf_xrowid


def _make_xcolumn():
    @cfunc(types.int32(types.intp, types.intp, types.int32), cache=_CACHE)
    def _tvf_xcolumn(cur, ctx, j):
        try:
            c = carray(_cast_int_to_void_p(cur), (1,), dtype=_TVF_CUR_DTYPE)
            data_p = c[0].data_p
            if data_p == 0:
                return SQLITE_OK
            rowid = c[0].rowid
            d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_TVF_DESC_DTYPE)
            ncols = d[0].ncols
            tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols + d[0].n_hidden,), dtype=tags_buf_t)
            if j >= ncols:
                # For a hidden argument column, xColumn returns the argument that xFilter decoded for fn, as
                # SQLite's own table-valued functions do.
                arg_p = c[0].args_p + _ARG_SLOT * (j - ncols)
                if tags[j] == _TAG_F32 or tags[j] == _TAG_F64:
                    sqlite3_result_double(ctx, load_unaligned(arg_p, float64))
                else:
                    sqlite3_result_int64(ctx, load_unaligned(arg_p, int64))
                return SQLITE_OK
            offsets = carray(_cast_int_to_void_p(d[0].col_offsets), (ncols,), dtype=np.int64)
            widths = carray(_cast_int_to_void_p(d[0].col_widths), (ncols,), dtype=np.int64)
            addr = data_p + rowid * c[0].row_stride + offsets[j]
            # The vtable passes S and BLOB cells as STATIC, but every tvf result stays TRANSIENT. A tvf result points
            # into the per-cursor NRT array or into the scratch buffer. The next xFilter releases and replaces that
            # array, so a STATIC pointer here would dangle.
            return _emit_cell(ctx, addr, tags[j], widths[j], c[0].scratch_p, _SQLITE_TRANSIENT)
        except Exception:
            sqlite3_result_error(ctx, get_unicode_data_p("error reading tvf column"), -1)
            return SQLITE_ERROR
    return _tvf_xcolumn


def _prepare_fn(fn):
    from numba.extending import is_jitted
    if is_jitted(fn):
        return fn
    if callable(fn):
        return njit(fn)
    raise TypeError("fn must be a callable (plain Python or @njit), got %r" % (fn,))


def _compile_xfilter(stem, arg_tags, out_dtype, fn):
    n_hidden = len(arg_tags)
    arg_decode = _gen_arg_decode(arg_tags)
    fn_call = "_fn(%s)" % ", ".join("a%d" % i for i in range(n_hidden))
    src = _XFILTER_SRC.format(arg_decode=arg_decode, fn_call=fn_call)
    tvf_digest = digest((out_dtype, tuple(arg_tags)), [fn])
    code_txt = "# tvf-digest: %s\n%s" % (tvf_digest, src)
    anchor = _anchor_path(_ANCHOR_SUBDIR, stem, code_txt)
    ns = {**globals(), "_fn": fn, "_N_HIDDEN": n_hidden,
          "jit_options": _anchored_or_uncached(anchor, code_txt, jit_options)}
    code = compile(code_txt, str(anchor), mode="exec")
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns["_tvf_xfilter_impl"]


def _build_tvf_descriptor(name, arg_types, out_dtype):
    if out_dtype.fields is None or out_dtype.names is None:
        raise TypeError("out_dtype must be a structured numpy dtype, got %r" % (out_dtype,))
    names = list(out_dtype.names)
    subs = [out_dtype.fields[nm][0] for nm in names]
    offs = [out_dtype.fields[nm][1] for nm in names]
    vis_tags = [_col_tag(s, False) for s in subs]
    vis_widths = [int(s.itemsize) for s in subs]

    arg_dtypes = [np.dtype(t) for t in arg_types]
    arg_tags = [_col_tag(dt, False) for dt in arg_dtypes]
    if any(t not in _INT_TAGS and t not in _FLOAT_TAGS for t in arg_tags):
        raise TypeError(
            "register_tvf arg_types must be integer or float scalar dtypes; "
            "string/bytes hidden args are not supported")

    offsets_buf = np.array(offs, dtype=np.int64)
    # tags_buf holds one tag per schema column, in the order that the schema below declares them: the visible columns
    # first, then the hidden arguments.
    tags_buf = np.array(vis_tags + arg_tags, dtype=tags_buf_t)
    widths_buf = np.array(vis_widths, dtype=np.int64)
    scratch = max([w + 1 for w, t in zip(vis_widths, vis_tags) if t == _TAG_U], default=0)

    vis_sql = ['"%s" %s' % (n.replace('"', '""'), _SQL_TYPE[t]) for n, t in zip(names, vis_tags)]
    hid_sql = ['"arg%d" %s HIDDEN' % (i, _SQL_TYPE[t]) for i, t in enumerate(arg_tags)]
    schema = ("CREATE TABLE x(%s)" % ", ".join(vis_sql + hid_sql)).encode("utf-8") + b"\x00"

    c = np.zeros(1, _TVF_DESC_DTYPE)
    c["ncols"] = len(names)
    c["n_hidden"] = len(arg_tags)
    c["itemsize"] = int(out_dtype.itemsize)
    c["col_offsets"] = offsets_buf.ctypes.data
    c["col_tags"] = tags_buf.ctypes.data
    c["col_widths"] = widths_buf.ctypes.data
    c["schema_ptr"] = ctypes.cast(ctypes.c_char_p(schema), ctypes.c_void_p).value
    c["scratch_bytes"] = int(scratch)
    return c, offsets_buf, tags_buf, widths_buf, schema, arg_tags


class _TvfHandle:
    """Hold references to the module struct, every cfunc object, the descriptor buffers and fn of one registration.

    SQLite stores the addresses of the cfuncs and calls them by address. If this handle is garbage-collected, the
    cfuncs are freed and SQLite calls freed code. The module-level ``_DATA_ANCHOR`` holds the handle, and SQLite
    releases it through ``xDestroy``.
    """
    __slots__ = ("_keep",)

    def __init__(self, *objs):
        self._keep = objs


def _stem(name):
    return "tvf_" + "".join(c if c.isalnum() else "_" for c in name)


def register_tvf(db, name, arg_types, out_dtype, fn):
    """Register an eponymous table-valued function backed by a computed array.

    ``SELECT * FROM name(<args>)`` calls ``fn(*args)``, a plain Python or ``@njit`` callable, and produces the rows
    of the array that ``fn`` returns. ``arg_types`` lists numpy scalar dtypes. numbox reads an argument of an integer
    kind with ``sqlite3_value_int64`` and one of a floating kind with ``sqlite3_value_double``. The arguments are
    exposed as trailing HIDDEN columns ``arg0``, ``arg1``, ..., and the query must supply an equality value for each
    of them. A call form that leaves a hidden argument unbound fails with a SQLite constraint error and returns no
    rows. A query that names a hidden column receives that argument as ``fn`` received it.

    ``fn`` must return a 1-D numpy structured array whose dtype is ``out_dtype``. A slice, a strided view or an offset
    view is handled, because the row stride is honored. A returned array whose *dtype* differs from ``out_dtype`` is
    read through the layout of ``out_dtype`` and yields undefined values. A NaN ``float`` cell reads as SQL NULL, as in
    ``register_table``, because SQLite coerces a NaN REAL to NULL.

    The module-level ``_DATA_ANCHOR`` holds the keep-alive of the registration. SQLite releases it through ``xDestroy``
    when the connection closes or when the same name is registered again. ``out_dtype`` enters the anchor's digest, so
    numba caches the generated ``xFilter`` across processes, one per dtype.
    """
    c, offsets_buf, tags_buf, widths_buf, schema, arg_tags = _build_tvf_descriptor(
        name, arg_types, out_dtype)

    xfilter_impl = _compile_xfilter(_stem(name), arg_tags, out_dtype, _prepare_fn(fn))

    # This cfunc sets cache=False, unlike every other cfunc here. It closes over xfilter_impl, a dispatcher that each
    # registration creates, so its cache key never repeats. With cache=True, numba would write an index entry per
    # registration per process and read none of them back. A measurement with cache=True shows 1, 2 and 3
    # overloads over three identical processes, and 4 over four registrations in one process, while the sibling
    # cfuncs stay at 1. The content-addressed xfilter_impl anchor (_compile_xfilter) caches this path instead.
    @cfunc(types.int32(types.intp, types.int32, types.intp, types.int32, types.intp), cache=False)
    def _tvf_xfilter(cur, idx_num, idx_str, argc, argv):
        # The try/except converts an exception from the user fn into SQLITE_ERROR, through xFilter's own return code.
        # A try/finally would not, because it reraises on numba 0.65.1. Without the except clause, the @cfunc
        # boundary discards the exception and returns the zero default (SQLITE_OK), so SQLite reports a successful
        # query with no rows. xfilter_impl zeroes the cursor before it calls the user fn, so the cursor is left in a
        # consistent empty state when fn raises.
        try:
            xfilter_impl(cur, argc, argv)
        except Exception:
            return SQLITE_ERROR
        return SQLITE_OK

    xconnect = _make_xconnect()
    xbestindex = _make_xbestindex()
    xdisconnect, xopen, xclose, xnext, xeof, xrowid = _make_static_cfuncs()
    xcolumn = _make_xcolumn()

    module = _Sqlite3Module()
    module.iVersion = 1
    module.xCreate = xconnect.address
    module.xConnect = xconnect.address
    module.xBestIndex = xbestindex.address
    module.xDisconnect = xdisconnect.address
    module.xDestroy = xdisconnect.address
    module.xOpen = xopen.address
    module.xClose = xclose.address
    module.xFilter = _tvf_xfilter.address
    module.xNext = xnext.address
    module.xEof = xeof.address
    module.xColumn = xcolumn.address
    module.xRowid = xrowid.address
    module_p = ctypes.addressof(module)

    handle = _TvfHandle(
        module, c, offsets_buf, tags_buf, widths_buf, schema, fn,
        xfilter_impl, _tvf_xfilter, xconnect, xbestindex, xdisconnect,
        xopen, xclose, xnext, xeof, xrowid, xcolumn)
    _register_with_destroy(db, name, module_p, c.ctypes.data, handle)
