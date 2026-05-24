import ctypes
import ctypes.util
import platform as plat
import sqlite3
import sys
import sysconfig

import pytest

from numbox.utils.pysqlite_bridge import extract_connection_ptr


_IS_MACOS = plat.system() == "Darwin"
_macos_skip = pytest.mark.skipif(
    _IS_MACOS,
    reason="extract_connection_ptr segfaults on macOS arm64 Py 3.14 — see test_macos_diagnostic",
)


def test_macos_diagnostic():
    """Captures everything we need to diagnose the macOS arm64 Py 3.14 segfault.

    Performs no sqlite3 function calls on the extracted ``db_ptr`` — only
    reads raw bytes at offsets and calls libversion (no-arg, safe). Always
    fails so the captured diagnostic shows in pytest output.
    """
    lines = []
    lines.append(f"sys.version = {sys.version}")
    lines.append(f"platform.platform() = {plat.platform()}")
    lines.append(f"platform.machine() = {plat.machine()}")
    lines.append(f"Py_GIL_DISABLED = {sysconfig.get_config_var('Py_GIL_DISABLED')!r}")
    lines.append(f"Py_DEBUG = {sysconfig.get_config_var('Py_DEBUG')!r}")
    lines.append(f"sys.abiflags = {sys.abiflags!r}")
    lines.append(f"sqlite3.Connection.__basicsize__ = {sqlite3.Connection.__basicsize__}")
    lines.append(f"sqlite3.sqlite_version = {sqlite3.sqlite_version!r}")

    sqlite_images = []
    if _IS_MACOS:
        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        libsystem._dyld_image_count.restype = ctypes.c_uint32
        libsystem._dyld_image_count.argtypes = []
        libsystem._dyld_get_image_name.restype = ctypes.c_char_p
        libsystem._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        for i in range(libsystem._dyld_image_count()):
            n = libsystem._dyld_get_image_name(i)
            if n and b"sqlite" in n.lower():
                sqlite_images.append(n.decode())
    lines.append(f"loaded sqlite dylibs (macOS): {sqlite_images}")

    find_lib_path = ctypes.util.find_library("sqlite3")
    lines.append(f"ctypes.util.find_library('sqlite3') = {find_lib_path!r}")

    for label, path in [("first dyld sqlite image", sqlite_images[0] if sqlite_images else None),
                        ("find_library result", find_lib_path)]:
        if path is None:
            lines.append(f"libversion via {label}: (no path)")
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.sqlite3_libversion.restype = ctypes.c_char_p
            lib.sqlite3_libversion.argtypes = []
            ver = lib.sqlite3_libversion().decode()
            lines.append(f"libversion via {label} ({path}): {ver!r}")
        except Exception as e:
            lines.append(f"libversion via {label} ({path}): error {e!r}")

    try:
        from numbox.core.bindings._sqlite_conn import sqlite3_libversion as numbox_libversion
        ver_p = numbox_libversion()
        ver = ctypes.c_char_p(ver_p).value.decode() if ver_p else "(null)"
        lines.append(f"libversion via numbox @njit binding: {ver!r}")
    except Exception as e:
        lines.append(f"libversion via numbox @njit binding: error {e!r}")

    conn = sqlite3.connect(":memory:")
    lines.append(f"id(conn) = 0x{id(conn):016x}")
    lines.append(f"conn.__sizeof__() = {conn.__sizeof__()}")
    for off in range(0, 65, 8):
        v = ctypes.c_void_p.from_address(id(conn) + off).value or 0
        lines.append(f"  offset +{off:3d}: 0x{v:016x}")
    conn.close()

    report = "\n".join(lines)
    assert False, f"\n\nMACOS PYSQLITE_BRIDGE DIAGNOSTIC:\n{report}\n"


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
