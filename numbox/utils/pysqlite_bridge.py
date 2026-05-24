"""Extract raw SQLite C API pointers from Python ``sqlite3`` objects.

Bridges CPython's stdlib ``sqlite3`` module to the numba-callable
bindings layer by exposing the underlying ``sqlite3 *db`` handle that
sits inside a Python ``sqlite3.Connection``. Mirrors the pattern in
`numbduck.pybridge <https://github.com/Goykhman/numbduck/blob/main/numbduck/pybridge.py>`_,
which does the same for DuckDB.

**macOS library coordination.** On macOS, python.org framework Python
*statically* links libsqlite3 into ``_sqlite3.cpython-3X-darwin.so`` --
the actions/python-versions builder runs the configure step with
``-L$(brew --prefix sqlite3)/lib``, so the resulting extension carries
its own copy of sqlite3 (typically a newer version than ``/usr/lib``).
Meanwhile ``ctypes.util.find_library("sqlite3")`` resolves to the
system ``/usr/lib/libsqlite3.dylib``, often an older release. A
``sqlite3*`` allocated by Python's bundled sqlite has a struct layout
incompatible with the system sqlite's functions; calling
``sqlite3_errmsg(db_p)`` from the wrong library crashes.

Naively preloading Python's libsqlite3 with ``RTLD_GLOBAL`` doesn't
help -- macOS dyld's shared cache resolves system libraries through
``dlsym(RTLD_DEFAULT)`` regardless of user load order.

Importing this module on macOS extracts Python's ``sqlite3_*`` symbol
*addresses* from ``_sqlite3.so`` (its 275 sqlite3_* symbols are
exported globally; ``nm -gU`` confirms) and registers them with
:func:`llvmlite.binding.add_symbol`. LLVM's JIT linker consults its
add_symbol registry *before* falling back to ``dlsym(RTLD_DEFAULT)``,
so numbox's eager-compiled ``@proxy`` bindings link to Python's
addresses regardless of dyld's shared-cache resolution.

**Order constraint.** ``@proxy`` compiles eagerly at decoration time
(see ``numbox.core.proxy.proxy``: *"The original function func will be
eagerly JIT-compiled with the given signature(s)"*). The patch must
run *before* any ``numbox.core.bindings._sqlite_*`` import. This
module guarantees that by registering symbols at module load before
its own ``from numbox.core.bindings._sqlite_conn import ...`` line.
Callers must import :mod:`numbox.utils.pysqlite_bridge` before any
direct or transitive import of ``numbox.core.bindings._sqlite_*``.
"""
import ctypes
import sqlite3  # noqa: F401  -- triggers _sqlite3 + libsqlite3 dependency load
from platform import system


_NUMBOX_SQLITE_SYMBOLS = (
    # Connection (numbox/core/bindings/_sqlite_conn.py)
    "sqlite3_open", "sqlite3_open_v2", "sqlite3_close",
    "sqlite3_libversion", "sqlite3_libversion_number",
    "sqlite3_errmsg", "sqlite3_errcode", "sqlite3_extended_errcode",
    "sqlite3_threadsafe", "sqlite3_db_handle", "sqlite3_db_filename",
    "sqlite3_db_readonly",
    "sqlite3_changes", "sqlite3_total_changes",
    "sqlite3_changes64", "sqlite3_total_changes64",
    "sqlite3_last_insert_rowid",
    # Statement
    "sqlite3_prepare_v2", "sqlite3_finalize", "sqlite3_reset",
    "sqlite3_step", "sqlite3_sql", "sqlite3_expanded_sql",
    "sqlite3_stmt_busy",
    # Bind
    "sqlite3_bind_int", "sqlite3_bind_int64", "sqlite3_bind_double",
    "sqlite3_bind_text", "sqlite3_bind_blob", "sqlite3_bind_null",
    "sqlite3_bind_parameter_count", "sqlite3_bind_parameter_index",
    "sqlite3_bind_parameter_name",
    # Column
    "sqlite3_column_int", "sqlite3_column_int64", "sqlite3_column_double",
    "sqlite3_column_text", "sqlite3_column_blob", "sqlite3_column_bytes",
    "sqlite3_column_type", "sqlite3_column_count",
    "sqlite3_column_name", "sqlite3_column_decltype",
    "sqlite3_column_database_name", "sqlite3_column_table_name",
    "sqlite3_column_origin_name",
    # Exec
    "sqlite3_exec", "sqlite3_free",
    # Blob
    "sqlite3_blob_open", "sqlite3_blob_close", "sqlite3_blob_bytes",
    "sqlite3_blob_read", "sqlite3_blob_write", "sqlite3_blob_reopen",
    # Hooks
    "sqlite3_update_hook", "sqlite3_progress_handler",
    "sqlite3_busy_handler", "sqlite3_commit_hook",
    "sqlite3_rollback_hook", "sqlite3_trace_v2",
)


def _patch_numbox_sqlite_for_python_libsqlite3():
    """Redirect numbox's @njit sqlite3_* bindings to Python's bundled
    libsqlite3 via :func:`llvmlite.binding.add_symbol`.

    macOS only. See module docstring for the full rationale and the
    order constraint.
    """
    if system() != "Darwin":
        return

    import _sqlite3
    from llvmlite import binding as llvm_binding

    py_sqlite = ctypes.CDLL(_sqlite3.__file__)
    for sym in _NUMBOX_SQLITE_SYMBOLS:
        func = getattr(py_sqlite, sym, None)
        if func is None:
            continue
        addr = ctypes.cast(func, ctypes.c_void_p).value
        if addr:
            llvm_binding.add_symbol(sym, addr)


_patch_numbox_sqlite_for_python_libsqlite3()


# CRITICAL: must come AFTER the patch above. @proxy decoration in
# _sqlite_conn eagerly JIT-compiles each binding, baking the resolved
# symbol address into the compiled code. Without the prior add_symbol
# registration, sqlite3_* externs would resolve via dlsym(RTLD_DEFAULT)
# to the system /usr/lib/libsqlite3.dylib on macOS.
from numbox.core.bindings._sqlite_conn import sqlite3_errmsg  # noqa: E402


def extract_connection_ptr(conn):
    """Return the raw ``sqlite3*`` underlying a Python ``sqlite3.Connection``.

    Reads the ``pysqlite_Connection`` PyObject struct layout defined in
    CPython's `Modules/_sqlite/connection.h
    <https://github.com/python/cpython/blob/main/Modules/_sqlite/connection.h>`_::

        [0]  PyObject_HEAD   (16 bytes on 64-bit release builds)
        [16] sqlite3 *db     <-- returned

    Assumes a release (non-debug) Python build. ``Py_DEBUG`` builds
    prepend a 16-byte ``_PyObject_HEAD_EXTRA`` for refcount tracing,
    shifting the ``db`` field to offset 32; this function does not
    handle that case.

    The extracted pointer is validated by calling
    :func:`~numbox.core.bindings._sqlite_conn.sqlite3_errmsg` against
    it: a healthy connection returns ``"not an error"``. See the
    module docstring for how macOS coordinates Python's libsqlite3
    with numbox's bindings.

    Parameters
    ----------
    conn : sqlite3.Connection

    Returns
    -------
    int
        ``sqlite3*`` as a Python int (``intp``-compatible).

    Raises
    ------
    TypeError
        If *conn* is not a ``sqlite3.Connection``.
    RuntimeError
        If the extracted pointer fails the validation call.
    """
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError(
            f"expected sqlite3.Connection, got {type(conn).__name__}"
        )
    db_ptr = ctypes.c_void_p.from_address(id(conn) + 16).value
    if db_ptr is None:
        raise RuntimeError("extracted null sqlite3* from sqlite3.Connection")
    errmsg = ctypes.c_char_p(sqlite3_errmsg(db_ptr)).value
    if errmsg != b"not an error":
        raise RuntimeError(
            f"extracted connection pointer failed validation: {errmsg!r}"
        )
    return db_ptr
