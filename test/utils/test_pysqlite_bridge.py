import ctypes
import sqlite3

import pytest

from numbox.utils.pysqlite_bridge import extract_connection_ptr
from numbox.core.bindings._sqlite_conn import sqlite3_changes, sqlite3_libversion


def test_extract_connection_ptr_memory_db():
    conn = sqlite3.connect(":memory:")
    try:
        p = extract_connection_ptr(conn)
        assert isinstance(p, int)
        assert p != 0
    finally:
        conn.close()


def test_extract_connection_ptr_file_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    try:
        p = extract_connection_ptr(conn)
        assert isinstance(p, int)
        assert p != 0
    finally:
        conn.close()


def test_extract_connection_ptr_rejects_non_connection():
    with pytest.raises(TypeError, match="expected sqlite3.Connection"):
        extract_connection_ptr("not a connection")


def test_numbox_and_python_use_same_libsqlite3():
    """Library-coordination sanity check.

    numbox's ``sqlite3_libversion()`` and Python's ``sqlite3.sqlite_version``
    must report the same version string. If they disagree, the two libraries
    are different instances (likely the macOS-Homebrew vs. system divergence
    documented in ``pysqlite_bridge``), and any ``sqlite3*`` extracted via
    ``extract_connection_ptr`` would be unsafe to pass to numbox's bindings.
    """
    numbox_version = ctypes.c_char_p(sqlite3_libversion()).value.decode()
    assert numbox_version == sqlite3.sqlite_version, (
        f"numbox sees libsqlite3 {numbox_version!r}, Python sees "
        f"{sqlite3.sqlite_version!r}: libraries are not coordinated"
    )


def test_extracted_pointer_usable_with_numbox_bindings(tmp_path):
    """The pointer returned by ``extract_connection_ptr`` works with numbox's
    @njit-callable sqlite bindings.

    This is the load-bearing test for the whole helper: it exercises the
    cross-library safety. INSERT 3 rows via Python's sqlite3, then ask
    numbox's ``sqlite3_changes`` (which calls into libsqlite3 via JIT-linked
    extern) what the last statement changed. Mismatch or crash here means
    Python and numbox are not using the same libsqlite3.
    """
    db_path = tmp_path / "shared.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE t(x INTEGER)")
        conn.execute("INSERT INTO t VALUES (1), (2), (3)")
        conn.commit()
        p = extract_connection_ptr(conn)
        assert sqlite3_changes(p) == 3
    finally:
        conn.close()
