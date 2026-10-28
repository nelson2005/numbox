"""Extract raw SQLite C API pointers from Python ``sqlite3`` objects.

The module bridges CPython's standard-library ``sqlite3`` module to the numba-callable bindings layer. It exposes the
underlying ``sqlite3 *db`` handle that a Python ``sqlite3.Connection`` holds. It follows the pattern of
`numbduck.pybridge <https://github.com/Goykhman/numbduck/blob/main/numbduck/pybridge.py>`_.

macOS caveat
~~~~~~~~~~~~

On macOS, the system sqlite is in the `dyld shared cache
<https://keith.github.io/xcode-man-pages/dyld.1.html>`_ (structure defined in Apple's
`DyldSharedCache.h <https://github.com/apple-oss-distributions/dyld/blob/main/common/DyldSharedCache.h>`_),
which is mapped into every process at launch. LLVM's JIT linker resolves ``sqlite3_*`` symbols through
``dlsym(RTLD_DEFAULT)``, which returns the first match in load order: the system copy. Python's ``_sqlite3.so`` may
use a *different* sqlite, statically linked on python.org framework builds or dynamically linked to Homebrew's copy.

If the system sqlite and Python's sqlite differ enough in version or internal layout, passing the pointer from
:func:`extract_connection_ptr` to numbox's ``@njit`` bindings can produce wrong results or segfaults.

**Workaround:** force the process to load the sqlite that Python uses first::

    DYLD_INSERT_LIBRARIES=/path/to/your/libsqlite3.dylib python my_script.py

Common paths:

- Homebrew (Apple Silicon): ``/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib``
- Homebrew (Intel): ``/usr/local/opt/sqlite/lib/libsqlite3.dylib``
- Conda-forge: ``$CONDA_PREFIX/lib/libsqlite3.dylib``
- python.org framework builds link sqlite statically and provide no external dylib; install Homebrew sqlite and use
  its path.

A Linux system typically has only one sqlite, so no workaround is needed there.
"""
import ctypes
import sqlite3

from numbox.core.bindings.sqlite.conn import sqlite3_errmsg, sqlite3_libversion


def _pyobject_head_fields():
    """Return ctypes field list for CPython's ``PyObject_HEAD``.

    The function detects debug and free-threaded builds at run time, so the struct offset of ``db`` is correct in all
    configurations.
    """
    import sys
    import sysconfig

    fields = []

    if hasattr(sys, "gettotalrefcount"):
        fields += [
            ("_ob_next", ctypes.c_void_p),
            ("_ob_prev", ctypes.c_void_p),
        ]

    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        fields += [
            ("ob_tid", ctypes.c_size_t),
            ("_padding", ctypes.c_uint16),
            ("ob_mutex", ctypes.c_uint8),
            ("ob_gc_bits", ctypes.c_uint8),
            ("ob_ref_local", ctypes.c_uint32),
            ("ob_ref_shared", ctypes.c_ssize_t),
            ("ob_type", ctypes.c_void_p),
        ]
    else:
        fields += [
            ("ob_refcnt", ctypes.c_ssize_t),
            ("ob_type", ctypes.c_void_p),
        ]

    return fields


class _PysqliteConnection(ctypes.Structure):
    """Model the first fields of CPython's ``pysqlite_Connection``.

    The model follows the layout in `Modules/_sqlite/connection.h
    <https://github.com/python/cpython/blob/main/Modules/_sqlite/connection.h>`_. ``ctypes.Structure`` computes the
    ``db`` offset from the declared field types, so the layout is correct across 32/64-bit, debug, and free-threaded
    builds without hardcoded constants.

    The pattern comes from `numbsql <https://github.com/cpcloud/numbsql/blob/main/numbsql/sqlite.py#L282>`_.
    """

    _fields_ = _pyobject_head_fields() + [("db", ctypes.c_void_p)]


def libraries_coordinated():
    """True if numbox's ``@njit`` bindings and Python's ``sqlite3`` resolve to the same libsqlite3.

    The function compares :func:`~numbox.core.bindings.sqlite.conn.sqlite3_libversion` (what numbox's bindings link
    against) with ``sqlite3.sqlite_version`` (what Python's ``sqlite3`` module links against). When they differ, as in
    the macOS shared-cache situation that the module docstring describes, passing a connection pointer from one
    library to the other is unsafe. :func:`extract_connection_ptr` then raises instead of proceeding.

    Returns
    -------
    bool
    """
    numbox_version = ctypes.c_char_p(sqlite3_libversion()).value.decode()
    return numbox_version == sqlite3.sqlite_version


def extract_connection_ptr(conn):
    """Return the raw ``sqlite3*`` underlying a Python ``sqlite3.Connection``.

    The function reads the ``db`` field at its platform-correct offset in the PyObject struct through
    :class:`_PysqliteConnection`. It then validates the pointer by calling
    :func:`~numbox.core.bindings.sqlite.conn.sqlite3_errmsg`, which returns ``"not an error"`` for a healthy
    connection.

    Before it reads the pointer, the function checks through :func:`libraries_coordinated` that numbox's bindings and
    Python's ``sqlite3`` module resolve to the same libsqlite3. If they differ, as in the macOS shared-cache situation
    of the module docstring, the function raises instead of passing the pointer to a mismatched library, which could
    segfault.

    Parameters
    ----------
    conn : sqlite3.Connection

    Returns
    -------
    int
        ``sqlite3*`` as a Python int (``intp``-compatible). The pointer is
        **borrowed** from *conn*: it is owned by the Python ``Connection`` and
        stays valid only while *conn* is alive and open. An ``int`` cannot keep
        *conn* referenced, so the keep-alive is the caller's responsibility.

    Warning
    -------
    The caller must retain *conn* for the entire lifetime of any ``@njit`` use of the returned pointer, and must not
    use the pointer after ``conn.close()`` or after *conn* is garbage-collected. Such a use is a use-after-free, which
    dereferences a dangling ``sqlite3*``. This matches the ``SQLITE_STATIC`` ownership contract on ``bind_text`` /
    ``bind_blob``: the borrowed memory must outlive every use.

    Raises
    ------
    TypeError
        If *conn* is not a ``sqlite3.Connection``.
    RuntimeError
        If numbox's bindings and Python's ``sqlite3`` use different
        libsqlite3 instances, or if the extracted pointer fails the
        validation call.
    """
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError(
            f"expected sqlite3.Connection, got {type(conn).__name__}"
        )
    if not libraries_coordinated():
        numbox_version = ctypes.c_char_p(sqlite3_libversion()).value.decode()
        raise RuntimeError(
            f"numbox's @njit bindings resolve libsqlite3 {numbox_version!r}, but Python's "
            f"sqlite3 module uses {sqlite3.sqlite_version!r}.  Passing this connection "
            f"pointer to numbox's bindings would be unsafe (possible segfault).  On macOS, "
            f"set DYLD_INSERT_LIBRARIES to your Python's libsqlite3.dylib — see the "
            f"numbox.utils.pysqlite_bridge module docstring."
        )
    db_ptr = _PysqliteConnection.from_address(id(conn)).db
    if db_ptr is None:
        raise RuntimeError("extracted null sqlite3* from sqlite3.Connection")
    errmsg = ctypes.c_char_p(sqlite3_errmsg(db_ptr)).value
    if errmsg != b"not an error":
        raise RuntimeError(
            f"extracted connection pointer failed validation: {errmsg!r}"
        )
    return db_ptr
