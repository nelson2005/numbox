"""SQLite connection + metadata bindings.

Initializes the module-level ``sqlite3_lib`` CDLL handle via
``load_lib_with_handle("sqlite3")``. Subsequent ``_sqlite_*.py`` modules
import ``sqlite3_lib`` from here for their own ``proxy_if_available`` calls,
ensuring a single load of the shared library across the suite.

Two functions are decorated with ``proxy_if_available``:
``sqlite3_changes64`` and ``sqlite3_total_changes64``, both added in
SQLite 3.37 (Nov 2021). On Python 3.10 (which ships SQLite 3.34), they
stub to ``NotImplementedError`` so callers can ``hasattr(...,"as_func")``
to decide whether to use them or fall back to the int32 variants.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib_with_handle
from numbox.core.proxy.proxy import proxy, proxy_if_available


sqlite3_lib = load_lib_with_handle("sqlite3")


@proxy(signatures.get("sqlite3_open"), jit_options={"cache": True})
def sqlite3_open(filename_p, db_pp):
    return _call_lib_func("sqlite3_open", (filename_p, db_pp))


@proxy(signatures.get("sqlite3_open_v2"), jit_options={"cache": True})
def sqlite3_open_v2(filename_p, db_pp, flags, vfs_p):
    return _call_lib_func("sqlite3_open_v2", (filename_p, db_pp, flags, vfs_p))


@proxy(signatures.get("sqlite3_close"), jit_options={"cache": True})
def sqlite3_close(db_p):
    return _call_lib_func("sqlite3_close", (db_p,))


@proxy(signatures.get("sqlite3_libversion"), jit_options={"cache": True})
def sqlite3_libversion():
    return _call_lib_func("sqlite3_libversion")


@proxy(signatures.get("sqlite3_libversion_number"), jit_options={"cache": True})
def sqlite3_libversion_number():
    return _call_lib_func("sqlite3_libversion_number")


@proxy(signatures.get("sqlite3_errmsg"), jit_options={"cache": True})
def sqlite3_errmsg(db_p):
    return _call_lib_func("sqlite3_errmsg", (db_p,))


@proxy(signatures.get("sqlite3_errcode"), jit_options={"cache": True})
def sqlite3_errcode(db_p):
    return _call_lib_func("sqlite3_errcode", (db_p,))


@proxy(signatures.get("sqlite3_extended_errcode"), jit_options={"cache": True})
def sqlite3_extended_errcode(db_p):
    return _call_lib_func("sqlite3_extended_errcode", (db_p,))


@proxy(signatures.get("sqlite3_threadsafe"), jit_options={"cache": True})
def sqlite3_threadsafe():
    return _call_lib_func("sqlite3_threadsafe")


@proxy(signatures.get("sqlite3_db_handle"), jit_options={"cache": True})
def sqlite3_db_handle(stmt_p):
    return _call_lib_func("sqlite3_db_handle", (stmt_p,))


@proxy(signatures.get("sqlite3_db_filename"), jit_options={"cache": True})
def sqlite3_db_filename(db_p, name_p):
    return _call_lib_func("sqlite3_db_filename", (db_p, name_p))


@proxy(signatures.get("sqlite3_db_readonly"), jit_options={"cache": True})
def sqlite3_db_readonly(db_p, name_p):
    return _call_lib_func("sqlite3_db_readonly", (db_p, name_p))


@proxy(signatures.get("sqlite3_changes"), jit_options={"cache": True})
def sqlite3_changes(db_p):
    return _call_lib_func("sqlite3_changes", (db_p,))


@proxy(signatures.get("sqlite3_last_insert_rowid"), jit_options={"cache": True})
def sqlite3_last_insert_rowid(db_p):
    return _call_lib_func("sqlite3_last_insert_rowid", (db_p,))


@proxy(signatures.get("sqlite3_total_changes"), jit_options={"cache": True})
def sqlite3_total_changes(db_p):
    return _call_lib_func("sqlite3_total_changes", (db_p,))


# SQLite 3.37+; stubbed via proxy_if_available on older library versions.
@proxy_if_available(sqlite3_lib, signatures.get("sqlite3_changes64"), jit_options={"cache": True})
def sqlite3_changes64(db_p):
    return _call_lib_func("sqlite3_changes64", (db_p,))


@proxy_if_available(sqlite3_lib, signatures.get("sqlite3_total_changes64"), jit_options={"cache": True})
def sqlite3_total_changes64(db_p):
    return _call_lib_func("sqlite3_total_changes64", (db_p,))
