"""Bindings for sqlite3_exec and sqlite3_free.

sqlite3_exec runs arbitrary SQL in a single call: it parses, prepares, steps and finalizes a SQL string, which may
hold several statements. The third argument is a function pointer to a per-row callback, or 0 for no callback.

sqlite3_free releases memory that SQLite allocated and returned to the caller, in particular the errmsg buffer from
the out-param of sqlite3_exec and the result of sqlite3_expanded_sql. sqlite3_malloc is the corresponding allocator,
and the virtual-table code uses it to allocate the vtab and cursor structs.

Callback shape (informational)::

    int (*sqlite3_exec_callback)(void *ctx, int ncol,
                                 char **col_values, char **col_names)

A return value of 0 continues the execution, and a nonzero value aborts it with SQLITE_ABORT.

Produce the callback address from Python as follows::

    @cfunc(int32(voidptr, int32, intp, intp))
    def my_row_cb(ctx, n, values_pp, names_pp): ...
    sqlite3_exec(db, sql, my_row_cb.address, ctx_p, errmsg_pp)
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = [
    "sqlite3_exec", "sqlite3_free", "sqlite3_malloc",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_exec"), jit_options=jit_options)
def sqlite3_exec(db_p, sql_p, cb_p, ctx_p, errmsg_pp):
    return _call_lib_func(
        "sqlite3_exec", (db_p, sql_p, cb_p, ctx_p, errmsg_pp)
    )


@proxy(signatures.get("sqlite3_free"), jit_options=jit_options)
def sqlite3_free(mem_p):
    return _call_lib_func("sqlite3_free", (mem_p,))


@proxy(signatures.get("sqlite3_malloc"), jit_options=jit_options)
def sqlite3_malloc(n):
    return _call_lib_func("sqlite3_malloc", (n,))
