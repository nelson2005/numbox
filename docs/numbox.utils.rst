numbox.utils
============

numbox.utils.highlevel
----------------------

Dynamically defining StructRef
''''''''''''''''''''''''''''''

Defining numba `StructRef` requires writing a lot of boilerplate code.
A utility for concise definition of `StructRef` types that supports caching
is provided in :func:`numbox.utils.highlevel.make_structref`. To use it,
define a *separate module* such as type_classes.py such as::

    from numba.experimental.structref import register
    from numba.core.types import StructRef


    @register
    class DataStructTypeClass(StructRef):
        pass

Then in a *different module* main.py define::

    from numba.core.types import float32, unicode_type
    from numpy import isclose
    from numbox.utils.highlevel import make_structref
    from type_classes import DataStructTypeClass


    def derive_output(struct_):
        if struct_.control == "double":
            return struct_.value * 2
        else:
            return struct_.value


    data_struct = make_structref(
        "DataStruct",
        {"value": float32, "control": unicode_type},
        DataStructTypeClass,
        struct_methods={
            "derive_output": derive_output
        }
    )


    if __name__ == "__main__":
        data_1 = data_struct(3.14, "double")
        data_2 = data_struct(2.17, "something else")
        assert isclose(data_1.derive_output(), 6.28)
        assert isclose(data_2.derive_output(), 2.17)


.. automodule:: numbox.utils.highlevel
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.derive_wap
-----------------------

A first-class ``FunctionType`` value carries three addresses in its data model, and the address that numba calls
decides whether an exception can leave the call. The ``jit_addr`` slot holds the entry point of the numba calling
convention, which unwinds. When the slot is empty, numba calls the C wrapper instead, which numba documents as not
supporting exceptions.

numba fills that slot for a ``Dispatcher`` and leaves it empty for everything else, so the compile result that
:func:`numbox.utils.highlevel.cres` wraps would arrive with the slot unset.
:class:`~numbox.utils.derive_wap.DeriveWAP`
captures the entry point from the compile result, and :class:`~numbox.utils.derive_wap.DeriveFunctionType` fills
the slot from it, on both unboxing and constant lowering.

The three slots can be read through :func:`numbox.utils.lowlevel.get_func_tuple`, which shows what the type adds.
``jit_addr`` is filled and matches the entry point the wrapper captured, while numba's own ``_get_jit_address``
still returns 0 for the same value, because it resolves an address only for a ``Dispatcher``.

.. code-block:: python

    from numba import float64
    from numba.experimental.function_type import _get_jit_address

    from numbox.utils.derive_wap import DeriveWAP
    from numbox.utils.highlevel import cres
    from numbox.utils.lowlevel import get_func_tuple

    sig = float64(float64, float64)

    @cres(sig, cache=True)
    def add(x, y):
        return x + y

    assert isinstance(add, DeriveWAP)

    c_addr, py_addr, jit_addr = get_func_tuple(add)
    assert jit_addr != 0
    assert jit_addr == add.jit_address
    assert _get_jit_address(add, sig) == 0

:func:`numbox.utils.highlevel.cres` is not the only source of one.
A ``@proxy`` binding's ``.as_func`` is a :class:`~numbox.utils.derive_wap.DeriveWAP` too, built from the proxied
body's compile result, with the same exception semantics and the same effect on a jitted caller that reaches it as
a constant. See :doc:`numbox.core.proxy`.

``test/utils/test_lowlevel.py::test_get_func_tuple`` checks every assertion above except the first; the wrapper's
type is checked separately by ``test/utils/test_derive_wap.py::test_cres_mints_a_derive_wap``.

.. automodule:: numbox.utils.derive_wap
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.preprocessing
--------------------------

Cache-anchor mechanism
''''''''''''''''''''''

``make_structref`` writes the generated ``code_txt`` to a content-addressed file under numba's cache directory and
uses that file, not ``highlevel.py``, as the ``compile()`` anchor. The content addressing keeps numba's per-overload
cache correct when two versions of the generated code differ only in ``co_consts``.

numba's per-overload cache key (``numba.core.caching.Cache._index_key``) is::

    (sig, codegen.magic_tuple(), hash(co_code), hash(closure_cells))

It hashes ``co_code`` only, not ``co_consts``. Python's `LOAD_CONST
<https://docs.python.org/3/library/dis.html#opcode-LOAD_CONST>`_ opcode encodes an *index* into ``co_consts`` rather
than the value itself, so two methods that differ only in a numeric literal (``return self.x + 1`` and ``return
self.x + 1000``) have identical ``co_code``. With a shared anchor file, both would resolve to the same cache files.
numba names a cache file after the anchor file's stem and the function's qualified name
(``CacheImpl.get_filename_base``), and puts every such file in one directory. That directory is the ``__pycache__``
beside the anchors where numba's in-tree locator takes them. Under ``NUMBA_CACHE_DIR`` or the user's cache directory,
it is a subdirectory derived from a hash of the anchors' *directory* (``_CacheLocator.get_suitable_cache_subpath``).
The second version to compile would then load the first version's binary, and numba would issue no warning. A
per-content anchor gives each version its own stem, which numba puts in the cache files' names, so the collision
never arises; every anchor shares that one directory.

A structural change to a body (a different operator, an additional statement, a renamed variable) produces
different ``co_code`` and therefore a different ``_index_key`` under any anchoring scheme, and invalidates the cache
correctly even under a shared anchor.
The content addressing protects against a narrow failure mode: an edit to constants only (numeric or string literals,
default argument values), where ``co_code`` is identical across versions.

Python 3.14's `LOAD_SMALL_INT <https://docs.python.org/3.14/whatsnew/3.14.html>`_ opcode inlines small integers
into ``co_code``, which narrows that case on that version to constants outside the inline range. The earlier
supported versions (3.10 to 3.13) collide on any constant edit.

numbox writes the anchor whenever it can.
numba caches from the anchor and quotes the source from it in
its messages, a typing error's message among them. With caching off, a failed write has no effect, and the path
serves as the code's name, as it did before.
With caching on, numbox then asks numba whether it can cache a function of that file, the same question the package
asks for its own modules.
The write can fail for any reason, and the answer is no when, for example, the user's cache directory or
``NUMBA_CACHE_DIR`` is unwritable and no other location is left.
The generated code then compiles
without a cache in either case, after one warning that names the remedy, instead of failing at the write or at
numba's set-up.

A path too long for the file system is one such failure, and the warning reports it where the file system reports it.
Windows reports a component that is too long as a syntax error, and the warning then offers ``NUMBA_CACHE_DIR`` at a
writable directory and quotes the error.
With the file names bounded, as described below, the directory is the only part that can make the path
too long, ``NUMBA_CACHE_DIR`` or the user's cache directory, and ``NUMBA_CACHE_DIR`` set to a shorter path moves
the anchor out of either. A cache directory whose path is not UTF-8 is another failure: numba hashes the anchor's
directory path as UTF-8 to name its cache directory, and the warning asks for ``NUMBA_CACHE_DIR`` at a UTF-8 path.

The anchor's check writes a file whose name is as long as the longest name numba writes for the functions it
anchors.
It reads those functions from the generated code: the decorated functions themselves and the functions defined within
a decorated one, an overload's constructor among the latter.
The name consists of the stem, the function's qualified name, its line, Python's version, three digits for the data
file's number, and the 21 characters of the temporary name through which numba writes each file.
numba counts the data files from one without
bound. A user cache directory or a ``NUMBA_CACHE_DIR`` within that length of the path limit, 4096 on Linux, fails
the check, and the generated code compiles without a cache after the warning.
A short probe name would pass there, as numba's own check does, and numba's first save would then fail while the
struct is made, without a warning.
A fixed bound for every struct, such as 251 bytes, would turn caching off on a file system whose name limit is
lower, for a struct whose names are a quarter of that.
The package's own check fails first only nearer the limit, within about 175 bytes of it.
The package's location under ``NUMBA_CACHE_DIR`` is a directory named for the
package's own directory, by its name and a hash of its path. That check reserves the length of numba's files for the
package's functions, so it fails only where those no longer fit.
The generated code then compiles according to the package's answer.

The anchor's name carries the struct's or the function's name. numba names its cache files after the anchor and the
qualified name of the function it caches, and that qualified name carries the struct's name again, through the class
whose body defines the jitted getters and method thunks. A struct name of about 93 characters would therefore push
numba's own file names past the 255 bytes a file system allows for a name, beyond the reach of the anchor's check.
Those names are bounded (``bounded_stem``).
A name of 40 bytes or fewer is used unchanged, so nearly every struct keeps the file names it had.
A longer name becomes the whole characters of it that fit in 31 bytes plus a digest of
the whole name. The measure is the name's UTF-8 length, which is the file system's: a character of another script
takes up to four bytes.
The generated class is defined under the bounded name and is assigned the struct's full name once its body is
compiled, so ``__name__``, ``__qualname__`` and ``repr`` show the name the caller gave, whatever its length, and the
struct caches.
A field's jitted getter is defined under the bounded name the same way, and the property is assigned the field's
name, so a field's name of any length caches too.
A method's name is bounded in
its thunk and its overload. ``compile_kernel``, the work builder's derives and the sqlite aggregate, window and
table-valued function registrations anchor their generated code the same way and fall back the same way, the derive
without a warning. The kernel and the derive write no anchor when not caching, as before. ``make_graph``'s kernel is
anchored to the builder's own file and asks for that file under the options it was given. See
:doc:`numbox.core.configurations` for the package-wide rule the anchors follow.

See also ``numba.core.caching.Cache._index_key`` and
``numba.core.caching._SourceFileBackedLocatorMixin.get_source_stamp`` in numba's source for the cache key
construction and the source-stamp validity check.


.. automodule:: numbox.utils.preprocessing
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.digest
-------------------

.. automodule:: numbox.utils.digest
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.fingerprint
------------------------

.. automodule:: numbox.utils.fingerprint
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.lowlevel
---------------------

.. automodule:: numbox.utils.lowlevel
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.meminfo
--------------------

Passing a reference through a pointer
'''''''''''''''''''''''''''''''''''''

:func:`~numbox.utils.meminfo.export_meminfo` returns a structref's (or an array's) MemInfo pointer as an ``intp``
and takes a reference for it, :func:`~numbox.utils.meminfo.borrow_structref` rebuilds a live structref from that
pointer, and :func:`~numbox.utils.meminfo.release_meminfo` releases the reference. The value stays alive for as
long as the reference is held.

On numba 0.65 and earlier, numba's ``removerefctpass`` deletes every ``NRT_incref`` and ``NRT_decref`` call in a
function it judges not to need them: roughly, one whose arguments are plain values or arrays, that returns a plain
value, and that allocates nothing. The reference ``export_meminfo`` takes is one of those calls, so numbox marks
the function that makes it with numba's ``numba_args_may_always_need_nrt`` metadata, which keeps the pass from
running on it. numba 0.66 removed the pass and every mention of that metadata, so from 0.66 on nothing reads it.

**numba's cache key does not include the mark.**
numbox releases without the mark lose the reference when ``export_meminfo`` is given an array on numba 0.65 and
earlier, so the pointer ``export_meminfo`` returns does not hold the array.
The MemInfo the pointer refers to can be freed while the pointer is still in use.
A ``cache=True`` function that makes such a
call, and that numba compiled against one of those releases, keeps the lost reference after numbox is upgraded.
numba's cache key covers the function's own source file, bytecode and closure, never the code of the functions it
calls (`numba's caching limitations <https://numba.readthedocs.io/en/stable/user/jit.html#cache>`__).
Delete that function's cache files (``*.nbi`` and ``*.nbc``) so that numba compiles it again.

.. automodule:: numbox.utils.meminfo
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.cstrings
---------------------

Allocating C strings for the bindings layer
'''''''''''''''''''''''''''''''''''''''''''

The bindings family (``numbox.core.bindings.libc``, ``numbox.core.bindings.sqlite.*`` and the others) takes
``intp`` pointers for every text argument. A valid NUL-terminated UTF-8 C string is not obtained from a Python
``str`` directly. :func:`~numbox.utils.lowlevel.get_unicode_data_p` returns a pointer to the Python string's
internal data, which CPython stores as UCS-1, UCS-2 or UCS-4 depending on the contents, so that pointer is safe for
ASCII input only.

:func:`~numbox.utils.cstrings.c_string` is a Python-side context manager. It allocates a C buffer holding the
UTF-8 encoding of the input and yields the pointer, whose lifetime is the ``with`` block::

    from numbox.utils.cstrings import c_string
    from numbox.core.bindings.sqlite.exec import sqlite3_exec

    with c_string("CREATE TABLE t(x INTEGER)") as sql_p:
        sqlite3_exec(db_p, sql_p, 0, 0, 0)
    # buffer freed here automatically

A call that takes several strings at once uses a single ``with`` with several context managers, or
:class:`contextlib.ExitStack` when their number is dynamic::

    with c_string("main") as schema_p, c_string("t") as table_p, c_string("b") as col_p:
        sqlite3_blob_open(db_p, schema_p, table_p, col_p, rowid, flags,
                          addressof(blob_p))

**Python only.** ``c_string`` cannot be used inside ``@njit``: numba does not support arbitrary context managers
(it raises ``UnsupportedBytecodeError``), and ``ctypes`` objects cannot be manipulated under JIT.
The Python side of an ``@njit`` caller that needs a C string allocates a numpy ``uint8`` buffer holding the UTF-8
bytes and a trailing NUL, and passes :func:`~numbox.utils.lowlevel.array_data_p` of it into the JIT kernel.

.. automodule:: numbox.utils.cstrings
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.pysqlite_bridge
----------------------------

Bridging Python ``sqlite3`` connections to the bindings layer
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

CPython's standard-library ``sqlite3.Connection`` holds the underlying ``sqlite3 *db`` C handle as a private field
of its PyObject struct. :func:`~numbox.utils.pysqlite_bridge.extract_connection_ptr` reads that field through
``ctypes``, so that a caller can pass the handle to numbox's JIT-callable SQLite bindings::

    import numbox.utils.pysqlite_bridge as pysqlite_bridge
    import sqlite3
    from numbox.core.bindings.sqlite.conn import sqlite3_changes

    conn = sqlite3.connect("app.db")
    conn.execute("CREATE TABLE t(x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1), (2), (3)")
    db_p = pysqlite_bridge.extract_connection_ptr(conn)
    assert sqlite3_changes(db_p) == 3

This follows the pattern of `numbduck.pybridge
<https://github.com/Goykhman/numbduck/blob/main/numbduck/pybridge.py>`_.

**Build configurations are handled at run time.** ``_pyobject_head_fields()`` computes the offset of the
``sqlite3 *db`` field for the running interpreter. A ``Py_DEBUG`` build (detected through
``sys.gettotalrefcount``) prepends the ``_ob_next`` and ``_ob_prev`` trace pointers, and a free-threaded build
(``Py_GIL_DISABLED``) uses the no-GIL object header, so release, debug and free-threaded builds all read the field
at the correct offset.

The macOS shared-cache caveat and its ``DYLD_INSERT_LIBRARIES`` workaround are described in the module docstring
below.
:func:`extract_connection_ptr` also checks, through :func:`libraries_coordinated`, that numbox's bindings and
Python's ``sqlite3`` resolve to the same libsqlite3 before it returns a pointer, and raises an exception rather than
returning a pointer to a mismatched library.

.. automodule:: numbox.utils.pysqlite_bridge
   :members:
   :show-inheritance:
   :undoc-members:

numbox.utils.timer
------------------

.. automodule:: numbox.utils.timer
   :members:
   :show-inheritance:
   :undoc-members:
