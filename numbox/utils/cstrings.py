"""Python-side helpers for passing C strings into the bindings layer.

The bindings layer (``numbox.core.bindings.sqlite.*``, ``numbox.core.bindings.libc`` and the others) takes ``intp``
pointers for every text argument. Producing a valid NUL-terminated UTF-8 C string from a Python ``str`` is
non-trivial. ``get_unicode_data_p`` in ``lowlevel.py`` returns a pointer to the Python string's internal data payload,
which CPython stores as UCS-1, UCS-2 or UCS-4 depending on the contents, so that pointer is safe for ASCII input only.

This module's :func:`c_string` allocates a separate C buffer holding the UTF-8 encoding of the input and yields the
pointer. The ``with`` statement manages the lifetime of the buffer safely.

**Python only.** ``c_string`` is a context manager, and numba does not support arbitrary context managers inside
``@njit`` (it raises ``UnsupportedBytecodeError``). ``ctypes`` objects cannot be manipulated under JIT either. An
``@njit`` caller that needs a C string should pre-allocate a numpy ``uint8`` buffer holding the UTF-8 bytes and a
trailing NUL outside the JIT scope, and pass ``array_data_p(buf)`` from ``numbox.utils.lowlevel`` into the JIT kernel.
"""
from contextlib import contextmanager
from ctypes import addressof, create_string_buffer


@contextmanager
def c_string(s):
    """Yield an ``intp`` pointer to a freshly allocated NUL-terminated UTF-8 C string for ``s``.

    Usage::

        from numbox.utils.cstrings import c_string
        from numbox.core.bindings.sqlite.exec import sqlite3_exec

        with c_string("CREATE TABLE t(x INTEGER)") as sql_p:
            sqlite3_exec(db_p, sql_p, 0, 0, 0)
        # buffer freed here

    The underlying ``ctypes`` buffer stays allocated for the duration of the ``with`` block. When the block exits, the
    Python reference is dropped and ctypes frees the memory; the pointer is then dangling and must not be used.

    An embedded NUL in ``s`` is rejected with ``ValueError``. A string containing ``"\\x00"`` would otherwise truncate
    the C string at the first NUL without any error. CPython's own ``sqlite3.connect`` rejects such a filename in the
    same way.

    A call that needs several strings at once can nest ``with`` statements or use ``contextlib.ExitStack``::

        from contextlib import ExitStack
        with ExitStack() as stack:
            a_p = stack.enter_context(c_string("a"))
            b_p = stack.enter_context(c_string("b"))
            # both pointers valid here

    ``c_string`` is Python-only and cannot be called inside ``@njit``. See the module docstring for the JIT
    alternative.
    """
    encoded = s.encode("utf-8")
    if b"\x00" in encoded:
        raise ValueError(
            "c_string: input contains an embedded NUL byte; would truncate "
            "the C string at the first NUL"
        )
    buf = create_string_buffer(encoded)
    yield addressof(buf)
