"""SQLite callback hooks: update_hook / progress_handler / busy_handler /
commit_hook / rollback_hook / trace_v2.

Each takes a function pointer (intp) the caller produces via @cfunc(...).address.
The cfunc instance MUST outlive the hook registration — keep it at module scope
in the caller.

Callback shapes (informational; signatures are caller's responsibility):
- update_hook:       void(void*, int op, const char* db, const char* tbl, sqlite3_int64 rowid)
- progress_handler:  int(void*) -- nonzero aborts
- busy_handler:      int(void*, int) -- 0 to abort, nonzero to retry
- commit_hook:       int(void*) -- nonzero vetoes commit
- rollback_hook:     void(void*)
- trace_v2:          int(unsigned, void*, void*, void*)

Exception handling: a numba @cfunc swallows any exception raised by its body (it
prints "Exception ignored" and returns the zero default -- 0 for the int hooks,
nothing for the void hooks) without unwinding into SQLite. None of these hooks
receive an sqlite3_context, so sqlite3_result_error is NOT available -- the only
channel is the return code (or nothing, for the void hooks). Guard the body with
a bare try/except (never try/finally) and choose the except's return per hook:

- commit_hook: return NONZERO to veto (else a raising veto silently COMMITS).
- progress_handler: return NONZERO to interrupt (else a raising cancel is lost).
- busy_handler: return 0 to give up (SQLITE_BUSY) or nonzero to retry.
- trace_v2: return 0 (SQLite currently ignores the return value).
- update_hook / rollback_hook: void, so no error can be signalled to SQLite -- catch internally to avoid the leak below.

The try/except also prevents an NRT meminfo leak when the body holds an
array / list / structref live across a call that raises.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.proxy.proxy import proxy

__all__ = [
    "sqlite3_update_hook", "sqlite3_progress_handler", "sqlite3_busy_handler",
    "sqlite3_commit_hook", "sqlite3_rollback_hook", "sqlite3_trace_v2",
]


load_lib("sqlite3")


@proxy(signatures.get("sqlite3_update_hook"), jit_options={"cache": True})
def sqlite3_update_hook(db_p, cb_p, ctx_p):
    return _call_lib_func("sqlite3_update_hook", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_progress_handler"), jit_options={"cache": True})
def sqlite3_progress_handler(db_p, n_ops, cb_p, ctx_p):
    return _call_lib_func("sqlite3_progress_handler", (db_p, n_ops, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_busy_handler"), jit_options={"cache": True})
def sqlite3_busy_handler(db_p, cb_p, ctx_p):
    return _call_lib_func("sqlite3_busy_handler", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_commit_hook"), jit_options={"cache": True})
def sqlite3_commit_hook(db_p, cb_p, ctx_p):
    return _call_lib_func("sqlite3_commit_hook", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_rollback_hook"), jit_options={"cache": True})
def sqlite3_rollback_hook(db_p, cb_p, ctx_p):
    return _call_lib_func("sqlite3_rollback_hook", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_trace_v2"), jit_options={"cache": True})
def sqlite3_trace_v2(db_p, mask, cb_p, ctx_p):
    return _call_lib_func("sqlite3_trace_v2", (db_p, mask, cb_p, ctx_p))
