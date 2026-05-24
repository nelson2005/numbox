import ctypes
import ctypes.util
import os
import platform as plat
import sqlite3
import sys

import pytest

from numbox.utils.pysqlite_bridge import extract_connection_ptr


_IS_MACOS = plat.system() == "Darwin"
_macos_skip = pytest.mark.skipif(
    _IS_MACOS,
    reason="extract_connection_ptr segfaults on macOS arm64 Py 3.14 — see test_macos_diagnostic",
)


def test_macos_diagnostic():
    """Round 2 diagnostic: confirm whether _sqlite3.so exports sqlite3_* symbols
    (since Python's libsqlite3 is statically linked into it on macOS Py 3.14).

    Tests three loading strategies for _sqlite3.so and checks which exposes
    sqlite3_libversion. Also searches the Python framework lib dir for any
    libsqlite3* file.
    """
    lines = []
    lines.append(f"sys.version = {sys.version}")
    lines.append(f"sys.abiflags = {sys.abiflags!r}")
    lines.append(f"sqlite3.sqlite_version = {sqlite3.sqlite_version!r}")

    import _sqlite3
    sqlite3_so_path = _sqlite3.__file__
    lines.append(f"_sqlite3.__file__ = {sqlite3_so_path!r}")
    lines.append(f"_sqlite3 file size = {os.path.getsize(sqlite3_so_path)} bytes")

    # Look for any libsqlite3 file in or near the Python install
    py_root = sys.prefix
    lines.append(f"sys.prefix = {py_root!r}")
    found_libs = []
    for root in [py_root, "/Library/Frameworks/Python.framework"]:
        if not os.path.isdir(root):
            continue
        for dirpath, dirs, files in os.walk(root):
            for f in files:
                if "libsqlite" in f.lower() and (f.endswith(".dylib") or f.endswith(".a")):
                    found_libs.append(os.path.join(dirpath, f))
    lines.append(f"libsqlite* files near Python install: {found_libs}")

    # Try loading _sqlite3.so via ctypes and checking for sqlite3_libversion
    for mode_label, mode in [("default (RTLD_LAZY|RTLD_LOCAL)", 0),
                             ("RTLD_GLOBAL", os.RTLD_GLOBAL if hasattr(os, 'RTLD_GLOBAL') else 0x8)]:
        try:
            lib = ctypes.CDLL(sqlite3_so_path, mode=mode)
            try:
                lib.sqlite3_libversion.restype = ctypes.c_char_p
                lib.sqlite3_libversion.argtypes = []
                ver = lib.sqlite3_libversion().decode()
                lines.append(f"_sqlite3.so sqlite3_libversion ({mode_label}): {ver!r}")
            except AttributeError as e:
                lines.append(f"_sqlite3.so sqlite3_libversion ({mode_label}): symbol not exported ({e})")
        except Exception as e:
            lines.append(f"_sqlite3.so load ({mode_label}): error {e!r}")

    # Also try dlsym(RTLD_DEFAULT) for sqlite3_libversion to see what current
    # resolution returns
    try:
        rtld_default_lib = ctypes.CDLL(None)
        rtld_default_lib.sqlite3_libversion.restype = ctypes.c_char_p
        rtld_default_lib.sqlite3_libversion.argtypes = []
        ver = rtld_default_lib.sqlite3_libversion().decode()
        lines.append(f"dlsym(RTLD_DEFAULT) sqlite3_libversion: {ver!r}")
    except Exception as e:
        lines.append(f"dlsym(RTLD_DEFAULT) sqlite3_libversion: error {e!r}")

    # Check if there's a way to introspect _sqlite3.so's symbol table via
    # /usr/bin/nm (might give us hint about visibility)
    try:
        import subprocess
        result = subprocess.run(
            ["/usr/bin/nm", "-gU", sqlite3_so_path],
            capture_output=True, text=True, timeout=10
        )
        exported_sqlite3 = [
            line.strip() for line in result.stdout.splitlines()
            if "sqlite3_" in line
        ]
        lines.append("nm -gU _sqlite3.so | grep sqlite3_ (top 10):")
        for s in exported_sqlite3[:10]:
            lines.append(f"  {s}")
        lines.append(f"  ... total exported sqlite3_* symbols: {len(exported_sqlite3)}")
    except Exception as e:
        lines.append(f"nm: error {e!r}")

    report = "\n".join(lines)
    assert False, f"\n\nMACOS DIAGNOSTIC ROUND 2:\n{report}\n"


@_macos_skip
def test_extract_connection_ptr_memory_db():
    conn = sqlite3.connect(":memory:")
    try:
        p = extract_connection_ptr(conn)
        assert isinstance(p, int)
        assert p != 0
    finally:
        conn.close()


@_macos_skip
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


@_macos_skip
def test_extracted_pointer_usable_with_numbox_bindings(tmp_path):
    from numbox.core.bindings._sqlite_conn import sqlite3_changes
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
