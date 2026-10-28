"""The tests cover the SQLite callback hook bindings.

The tests use numpy int64 arrays as ctx. If `arr.ctypes.data` is passed as the ctx pointer, the @cfunc callbacks can
read and write the bytes of the array directly. This pattern is the canonical way to capture state across the boundary
from C to Python without ctypes-level references to Python objects.
"""
from ctypes import addressof, c_int64

import numpy as np
import pytest
from numba import carray, cfunc, types

from numbox.core.bindings.sqlite.constants import SQLITE_INTERRUPT, SQLITE_OK, SQLITE_TRACE_STMT
from numbox.core.bindings.sqlite.hooks import (
    sqlite3_busy_handler, sqlite3_commit_hook, sqlite3_progress_handler, sqlite3_rollback_hook,
    sqlite3_trace_v2, sqlite3_update_hook,
)
from numbox.core.bindings.sqlite.conn import sqlite3_close, sqlite3_open
from numbox.core.bindings.sqlite.exec import sqlite3_exec
from numbox.utils.cstrings import c_string


# These cfuncs are defined at module level because they must outlive any hook registration.

@cfunc(types.void(types.voidptr, types.int32, types.intp, types.intp, types.int64))
def _update_cb(ctx, op, db_p, tbl_p, rowid):
    """Write op into the next entry of the op log in ctx, starting at ctx[0]. The index of that entry is ctx[64], which
    the callback then increments."""
    arr = carray(ctx, 128, dtype=np.int64)
    idx = arr[64]
    arr[idx] = op
    arr[64] = idx + 1


@cfunc(types.int32(types.voidptr))
def _progress_abort_cb(ctx):
    return 1  # A nonzero return aborts the operation


@cfunc(types.int32(types.voidptr, types.int32))
def _busy_abort_cb(ctx, n):
    return 0  # A return of 0 aborts instead of retrying


@cfunc(types.int32(types.voidptr))
def _commit_veto_cb(ctx):
    return 1  # A nonzero return vetoes the commit


@cfunc(types.void(types.voidptr))
def _rollback_count_cb(ctx):
    arr = carray(ctx, 1, dtype=np.int64)
    arr[0] += 1


@cfunc(types.int32(types.uint32, types.voidptr, types.voidptr, types.voidptr))
def _trace_count_cb(mask, ctx, p, x):
    arr = carray(ctx, 1, dtype=np.int64)
    arr[0] += 1
    return 0


@pytest.fixture
def populated_db(tmp_path):
    import sqlite3 as stdlib_sqlite3
    db_file = tmp_path / "hooks.sqlite"
    conn = stdlib_sqlite3.connect(str(db_file))
    conn.executescript(
        "CREATE TABLE t(a INTEGER);"
        "INSERT INTO t VALUES (1), (2), (3);"
    )
    conn.commit()
    conn.close()
    db_p = c_int64(0)
    with c_string(str(db_file)) as name_p:
        assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    yield db_p.value
    sqlite3_close(db_p.value)


def test_update_hook_records_ops(populated_db):
    # In the ctx layout, arr[0..63] holds the op log and arr[64] holds the next-write index.
    ctx = np.zeros(128, dtype=np.int64)
    sqlite3_update_hook(populated_db, _update_cb.address, ctx.ctypes.data)
    with c_string("INSERT INTO t VALUES (99); DELETE FROM t WHERE a=99;") as sql_p:
        rc = sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    # SQLITE_INSERT = 18, SQLITE_DELETE = 9
    assert ctx[64] == 2  # SQLite invokes the callback twice
    assert ctx[0] == 18
    assert ctx[1] == 9


def test_progress_handler_aborts(populated_db):
    sqlite3_progress_handler(populated_db, 1, _progress_abort_cb.address, 0)
    with c_string("SELECT * FROM t") as sql_p:
        rc = sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    assert rc == SQLITE_INTERRUPT


def test_busy_handler_registration_returns_ok(populated_db):
    # A busy condition is difficult to trigger reliably in a unit test, because it requires two connections and file
    # locking. The test verifies only that the registration succeeds.
    rc = sqlite3_busy_handler(populated_db, _busy_abort_cb.address, 0)
    assert rc == SQLITE_OK


def test_commit_hook_vetoes(populated_db):
    sqlite3_commit_hook(populated_db, _commit_veto_cb.address, 0)
    with c_string("INSERT INTO t VALUES (42)") as sql_p:
        rc = sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    # SQLite rolls back the vetoed commit. sqlite3_exec then returns a code other than SQLITE_OK (typically
    # SQLITE_CONSTRAINT_TRIGGER or SQLITE_CONSTRAINT in some versions).
    assert rc != SQLITE_OK


def test_rollback_hook_fires(populated_db):
    ctx = np.zeros(1, dtype=np.int64)
    sqlite3_rollback_hook(populated_db, _rollback_count_cb.address, ctx.ctypes.data)
    with c_string("BEGIN; INSERT INTO t VALUES (77); ROLLBACK;") as sql_p:
        rc = sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert ctx[0] == 1


def test_trace_v2_fires_for_stmt(populated_db):
    ctx = np.zeros(1, dtype=np.int64)
    rc = sqlite3_trace_v2(populated_db, SQLITE_TRACE_STMT,
                          _trace_count_cb.address, ctx.ctypes.data)
    assert rc == SQLITE_OK
    with c_string("SELECT 1") as sql_p:
        sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    assert ctx[0] >= 1


if __name__ == "__main__":
    from test.auxiliary_utils import collect_and_run_tests
    collect_and_run_tests(__name__)
