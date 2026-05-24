"""Extract raw SQLite C API pointers from Python ``sqlite3`` objects.

Bridges CPython's stdlib ``sqlite3`` module to the numba-callable
bindings layer by exposing the underlying ``sqlite3 *db`` handle that
sits inside a Python ``sqlite3.Connection``. Mirrors the pattern in
`numbduck.pybridge <https://github.com/Goykhman/numbduck/blob/main/numbduck/pybridge.py>`_,
which does the same for DuckDB.

**macOS library coordination.** macOS python.org framework builds
link the stdlib ``_sqlite3`` extension against Homebrew's
``libsqlite3.dylib``, not the system ``/usr/lib/libsqlite3.dylib``.
See `actions/python-versions' macos-python-builder.psm1
<https://github.com/actions/python-versions/blob/main/builders/macos-python-builder.psm1>`_
-- it adds ``-L$(brew --prefix sqlite3)/lib`` to ``LDFLAGS``.
Calling sqlite3 C API functions from one libsqlite3 instance on a
``sqlite3*`` pointer allocated by another is undefined behavior --
the two libraries can have different versions with incompatible
internal struct layouts, causing segfaults.

To avoid this hazard, importing this module on macOS preloads
Python's ``libsqlite3.dylib`` with ``RTLD_GLOBAL`` *before*
``numbox.core.bindings._sqlite_conn`` loads its own copy. macOS dyld
resolves symbols in load order under ``RTLD_GLOBAL``, so subsequent
numba JIT compiles resolve sqlite3 symbols to the same shared library
Python uses.

Linux and Windows already coordinate naturally: glibc's
``find_library("sqlite3")`` resolves to the system ``libsqlite3.so``
that Python also links against, and Windows uses CPython's bundled
``sqlite3.dll`` for both.

**Import order.** Import this module before
``numbox.core.bindings._sqlite_*`` or anything that triggers them, so
the preload lands before the system libsqlite3 is loaded by
``load_lib("sqlite3")``.
"""
import ctypes
import sqlite3  # noqa: F401  -- triggers _sqlite3 + libsqlite3 dependency load
from platform import system

from numbox.core.bindings.utils import load_lib_path


def _find_loaded_libsqlite3():
    """Return the absolute path of the libsqlite3 currently loaded by
    Python's sqlite3 module, or None if not determinable.

    Uses macOS's `dyld image enumeration
    <https://developer.apple.com/documentation/kernel/dyld>`_
    (``_dyld_image_count`` / ``_dyld_get_image_name``) to inspect
    currently-loaded dylibs. Must be called after ``import sqlite3``
    has triggered ``_sqlite3.so`` to load its libsqlite3 dependency.
    """
    libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    libsystem._dyld_image_count.restype = ctypes.c_uint32
    libsystem._dyld_image_count.argtypes = []
    libsystem._dyld_get_image_name.restype = ctypes.c_char_p
    libsystem._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    for i in range(libsystem._dyld_image_count()):
        name = libsystem._dyld_get_image_name(i)
        if name and b"libsqlite3" in name:
            return name.decode()
    return None


_preloaded_libsqlite3_handle = None
if system() == "Darwin":
    _python_libsqlite3 = _find_loaded_libsqlite3()
    if _python_libsqlite3 is not None:
        _preloaded_libsqlite3_handle = load_lib_path(_python_libsqlite3)


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
    module docstring for how the macOS library-coordination hazard is
    handled.

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
