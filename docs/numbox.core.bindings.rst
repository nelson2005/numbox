numbox.core.bindings
====================

Overview
++++++++

Loads dynamic libraries available in the Python environment, such as, `libc`, `libm`, and `libsqlite3`
in global symbol mode (`RTLD_GLOBAL`) via `ctypes`.
This adds global symbols (including native API) exported from those libraries to the LLVM symbol table.
These functions can then be invoked from the numba jitted code [#f1]_, complementing the suite
of numba-supported functionality.

Analogous technique can be expanded as needed for the user custom code.

.. rubric:: References

.. [#f1] See `numbsql <https://github.com/cpcloud/numbsql>`_ for previous work on jit-wrapping FFI imported functions.

Bindings module conventions
+++++++++++++++++++++++++++

Every binding in the ``numbox.core.bindings`` family (``libc``, ``libm``, ``stdio``, ``errno``, ``strerror``,
``fmtio``, and the ``numbox.core.bindings.sqlite`` subpackage, see :doc:`numbox.core.bindings.sqlite`) uses
extern-symbol references, so the ABI dispatch is correct under ASLR. ``libc``, ``libm`` and the sqlite subpackage
reach their symbols by way of :func:`~numbox.core.bindings.call._call_lib_func`; ``stdio``, ``errno``, ``strerror``
and ``fmtio`` do so through intrinsics of their own.
Pointer arguments are typed as ``intp``. The caller is responsible for the liveness,
alignment and ownership of the memory they point to.

Every wrapper except the formatted I/O family is decorated with :func:`~numbox.core.proxy.proxy.proxy`, so a user's
``@njit(cache=True)`` function can reference it safely.
``proxy`` declares the callee's ``llvm_cfunc_wrapper_name`` as an extern in the caller's module, and
llvmlite's JIT linker resolves the name in each process, so a cached caller resolves the function pointer correctly
under ASLR.

References for the semantics of the individual bindings:

- POSIX / Linux glibc: `man7.org <https://man7.org/linux/man-pages/man3/>`_
- macOS Darwin: `Apple Open Source Libc
  <https://github.com/apple-oss-distributions/Libc>`_
- Windows UCRT: `Microsoft Learn
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/c-run-time-library-reference>`_

ABI dispatch
++++++++++++

LLVM's JIT treats ABI lowering as the frontend's responsibility, so it does not insert the calling convention for
struct arguments and struct returns. ``numbox.core.bindings.call._call_lib_func`` dispatches per platform and per
struct shape, with the primitives of ``numbox.core.bindings.abi``: ``_current_platform`` identifies the platform,
``_classify`` classifies the struct shape, and ``_struct_bytes`` measures the struct size. Two ABI families are
relevant:

- **Windows x64** passes an aggregate larger than 8 bytes through a pointer that the caller allocates, and returns
  one through ``sret``. An aggregate of 1, 2, 4 or 8 bytes goes directly in a register.
- **SysV x86-64 and AAPCS64** pass and return an aggregate of up to 16 bytes directly in general-purpose registers.
  On SysV x86-64, a by-value argument larger than 16 bytes uses ``byval`` with ``optnone`` and ``noinline``, so that
  the LLVM optimizer does not remove the caller's stack copy before the callee reads it.

References:

- `llvmlite#300 <https://github.com/numba/llvmlite/issues/300#issuecomment-327235846>`_
- `llvm-project#85417 <https://github.com/llvm/llvm-project/issues/85417>`_
- `Windows x64 calling convention <https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention>`_
- `AAPCS64 <https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst>`_

Stdio handles, errno, and thread-safe strerror
++++++++++++++++++++++++++++++++++++++++++++++

**Stdio handles.** ``stdout()``, ``stderr()`` and ``stdin()`` are JIT-callable functions rather than module-level
Python constants, because the C library exposes its stdio ``FILE *`` values either as data symbols or through
accessor functions:

- **Linux (glibc and musl)**: data symbols, the global variables ``stdout``, ``stderr`` and ``stdin`` declared in
  `<stdio.h> <https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/stdio.h.html>`_
- **macOS Darwin**: data symbols, ``__stdoutp``, ``__stderrp`` and ``__stdinp``, which the ``stdout``, ``stderr``
  and ``stdin`` macros of the libc headers expand to, per
  `Apple's _stdio.h <https://github.com/apple-oss-distributions/Libc/blob/main/include/_stdio.h#L218-L220>`_
- **Windows**: an accessor function, `__acrt_iob_func(0|1|2)
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/internal-crt-globals-and-functions>`_, present in the
  UCRT only (Windows 10 and later, VS 2015 and later)

Both shapes are exposed through one ``() -> intp`` interface.
The wrapper refers to the symbol by an extern reference in the LLVM IR, never by a literal address, so ``cache=True``
remains correct under ASLR.
The JIT linker resolves the
address on each run, and the cached object holds no address.

Example: writing to stderr from inside ``@njit``:

.. code-block:: python

    from numba import njit
    from numbox.core.bindings.stdio import stderr
    from numbox.core.bindings.libc import fputs, fflush

    @njit(cache=True)
    def log_to_stderr(msg_p):
        fputs(msg_p, stderr())
        fflush(stderr())

**errno.** ``errno_get()`` and ``errno_set(v)`` reach the thread's errno location on every call through the
platform's accessor function: `__errno_location
<https://man7.org/linux/man-pages/man3/errno.3.html>`_ on Linux glibc and musl,
`__error <https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/intro.2.html>`_
on Darwin, and `_errno <https://learn.microsoft.com/en-us/cpp/c-runtime-library/errno-doserrno-sys-errlist-and-sys-nerr>`_
on Windows UCRT. The wrappers are therefore correct under ``@njit(parallel=True)``: each ``prange`` worker reads its
own thread's errno. An ordinary ``@njit`` function runs on the caller's OS thread, so a Python caller observes an errno
set inside it. A ``prange`` worker runs on another thread, so the caller does not observe an errno set there.

Example: reading and reporting errno after a binding in the style of a system call:

.. code-block:: python

    from numba import njit
    from numbox.core.bindings.errno import errno_get, errno_set

    @njit(cache=True)
    def clear_then_call_and_report(do_work):
        errno_set(0)
        result = do_work()
        return result, errno_get()

**Thread-safe strerror.** ``strerror_safe(errnum, buf, buflen)`` writes the error message into a buffer that the
caller supplies. It returns 0 on success and a positive errno code on failure. The symbol is selected at lowering
time:

- **glibc Linux**: `__xpg_strerror_r
  <https://codebrowser.dev/glibc/glibc/string/xpg-strerror.c.html>`_, the POSIX XSI form, present since glibc 2.3.4
  (2004)
- **musl Linux**: ``__xpg_strerror_r`` as well, exported as a `weak alias
  <https://git.musl-libc.org/cgit/musl/tree/src/string/strerror_r.c>`_ of musl's own ``strerror_r``, which is itself
  the POSIX form; musl never shipped the GNU form that returns ``char *``
- **macOS Darwin**: `strerror_r
  <https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/strerror_r.3.html>`_,
  the POSIX form
- **Windows**: `strerror_s
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/strerror-s-strerror-s-wcserror-s-wcserror-s>`_,
  whose arguments are ordered (buffer, size, errnum)

Other Linux libcs are not supported. On glibc, the plain ``strerror_r`` symbol is the
`GNU form <https://man7.org/linux/man-pages/man3/strerror_r.3.html>`_, which returns ``char *`` and does not match
the POSIX-shaped IR this module generates. The Linux selector always picks ``__xpg_strerror_r``, which resolves on
glibc and on musl, where the weak alias supplies it. The selector keeps a ``strerror_r`` fallback for a future libc
that drops ``__xpg_strerror_r``; on every supported libc the fallback is unreachable.

A Linux-only test inspects the IR to verify the selector, and monkeypatches ``ll.address_of_symbol`` to drive the
fallback branch. The binding depends on musl's symbol layout at run time: ``strerror_r`` and ``__xpg_strerror_r``
are both exported and resolve to the same address through the weak alias.
A CI job in an Alpine container that checks this with ``nm -D`` would verify that assumption.

Example: rendering the message for ``ENOENT`` (errno 2 on POSIX) into a buffer:

.. code-block:: python

    import errno
    import numpy as np
    from numba import njit
    from numbox.core.bindings.strerror import strerror_safe
    from numbox.utils.lowlevel import array_data_p

    @njit(cache=True)
    def explain(errnum, buf):
        rc = strerror_safe(errnum, array_data_p(buf), buf.size)
        return rc

    buf = np.zeros(128, dtype=np.uint8)
    rc = explain(errno.ENOENT, buf)
    msg = bytes(buf[:buf.tolist().index(0)]).decode()
    # rc == 0; msg is the (locale-dependent) string for ENOENT

Variadic formatted I/O: printf, fprintf, snprintf and sscanf
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``printf``, ``fprintf`` and ``snprintf`` are dual-mode bindings, and ``sscanf`` is their counterpart for parsing.
Each of the four is an ordinary Python function with an ``@overload`` registration that routes a call made inside
``@njit`` to a private ``@intrinsic``. The three writers run from the same source in plain Python and under
``@njit``, which is numba's own convention for builtins such as ``print`` and ``range``. ``sscanf`` runs under
``@njit`` only, as its section below says.

**Call convention.** The arguments follow the format string as C-style ``*args``, with no tuple at the call site::

    printf("x = %d, ratio = %.3f\n", n, ratio)
    fprintf(stderr(), "warning: %s\n", msg)
    snprintf(array_data_p(buf), buf.size, "[%d:%d]", lo, hi)
    sscanf(buf_p, "%d %lf", array_data_p(n_out), array_data_p(x_out))
    printf("no args here\n")

The internal ``@intrinsic`` still takes its arguments as a tuple, because numba's ``@intrinsic`` typing function does
not accept Python-level ``*args``. The ``@overload`` implementation packs the caller's ``*args`` into a tuple before
it calls the intrinsic.

**Dual mode:**

.. code-block:: python

    from numba import njit
    from numbox.core.bindings.fmtio import printf
    from numbox.core.bindings.libc import fflush
    from numbox.core.bindings.stdio import stdout

    def debug_kernel(x, label):
        printf("step %d: %s\n", x, label)
        fflush(stdout())
        return x * 2

    debug_kernel(7, "before")        # pure Python: writes via sys.stdout
    njit(debug_kernel)(7, "before")  # jitted: writes via libc printf
    # NUMBA_DISABLE_JIT=1 mode: also works (decorator becomes no-op)

**The format string must be a literal under @njit.** The intrinsic embeds it as a global constant in the IR at
typing time.
A ``unicode`` format string built at run time raises ``TypingError``.
In plain Python the format string can be any
``str``.

**The format string is encoded as UTF-8.** A non-ASCII code point in the literal is encoded as its UTF-8 bytes and
embedded in the IR global. printf passes every byte other than ``%`` through unchanged, so the bytes reach stdout,
the ``FILE *`` or the snprintf buffer as they were written. Modern terminals, files and the consoles of Windows 10
and later expect UTF-8.

  .. note::
     printf counts the width of ``%-Ns`` in bytes in every libc, so non-ASCII output is not padded to a count of
     code points.
     That is printf's contract. If a width in code points is wanted, pad the string with numba's own formatting
     (``f"{s:<10}"``) before passing it through ``%s``.

**String arguments are converted.** When the ``@overload`` receives an argument of type ``unicode_type`` or
``Literal[str]``, it wraps the argument in ``get_unicode_data_p``, so that libc's ``%s`` receives a NUL-terminated C
string.
User code passes a Python ``str`` and does not call ``get_unicode_data_p`` at the call site::

    printf("hello %s\n", "world")  # works in both modes; no get_unicode_data_p

The converted form, ``printf("hello %s\n", get_unicode_data_p(s))``, is still accepted, for compatibility and for a
caller that already holds a pointer.

**Integers are promoted to 64 bits.** The ``@njit`` implementation widens every integer variadic argument to 64 bits
before the libc call: ``sext`` for a signed integer, ``zext`` for an unsigned one and for ``bool`` to ``int64``, and
``fpext`` for ``float32`` to ``float64``. This differs from the C ABI, which promotes only values narrower than
``int``. It makes ``%lld`` with an ``int32`` argument work under ``@njit``, as it does in plain Python, where ``%``
ignores length modifiers and formats the value at its own width.
The widening adds one LLVM instruction per argument, which is free at run time.

.. list-table::
   :header-rows: 1

   * - Numba type
     - Widened to (in varargs slot)
   * - ``float32``
     - ``float64`` (``fpext``)
   * - ``float64``
     - pass through
   * - ``bool``
     - ``int64`` (``zext``)
   * - ``int8`` / ``int16`` / ``int32`` (signed)
     - ``int64`` (``sext``)
   * - ``uint8`` / ``uint16`` / ``uint32`` (unsigned)
     - ``int64`` (``zext``)
   * - ``int64`` / ``uint64`` / ``intp``
     - pass through
   * - ``unicode_type`` / ``Literal[str]``
     - converted through ``get_unicode_data_p`` to ``intp``

**An unsupported type raises ``TypingError`` at compile time.** That covers numpy arrays (``Array``), complex numbers
(``Complex``), tuples (``UniTuple`` and ``Tuple``), records, and every other type outside the table above.
Without the check, an aggregate LLVM value would be passed to libc's variadic call unchanged, and printf would read
bytes from the neighboring stack slots, which produces wrong output with no error.
Pass the field you want instead (``arr[0]``,
``c.real``, ``t[0]``).

**Format specifications in plain Python.**
Python's ``str.__mod__`` derives from printf but ignores a single C length modifier (``%ld``, ``%lf``, ``%hd``) and
rejects a doubled one (``%lld``, ``%hhd``) with ``ValueError``.
The plain-Python implementation
removes the length modifiers with a regular expression before it formats (``%lld`` becomes ``%d``, ``%.3lf``
becomes ``%.3f``), so one format string serves both modes.

  .. warning::
     ``%ld`` is the most common source of cross-platform errors.
     ``long`` is 8 bytes on LP64 (Linux, macOS) and 4 bytes on
     Win64 (LLP64). Under ``@njit`` on Win64, ``printf("%ld", int64_val)`` truncates the high 32 bits. Plain Python
     does not show the difference, because its ``%`` ignores ``l``. Use ``%lld`` with ``int64`` for a portable
     8-byte width, and test on Windows any format string that contains ``%ld``.

**The return value of ``snprintf`` on truncation differs on Windows** from C99 and POSIX. Under ``@njit`` on Linux
and macOS, and in plain Python everywhere, ``snprintf`` returns the number of bytes it would have written, without
the NUL. Under ``@njit`` on Windows, the binding calls MSVCRT's ``_snprintf``, which returns ``-1`` on truncation and
does not guarantee a NUL terminator. The check ``(rc < 0) or (rc >= size)`` detects truncation on every platform and
in both modes. The UCRT's C99 ``snprintf`` is an inline function in the header over ``__stdio_common_vsnprintf``
and cannot be linked in the plain C99 shape. Declaring ``i32 @snprintf(...)`` in the LLVM IR and letting the JIT
linker resolve it crashes with an access violation on Windows.

**``fprintf`` to a ``FILE *`` other than the standard streams, in plain Python.** The Python implementation records
the addresses of ``stdout()``, ``stderr()`` and ``stdin()`` at first use and routes those handles to the
corresponding ``sys.*`` streams. A ``FILE *`` returned by ``fopen`` cannot be dereferenced from Python without
ctypes, so in plain Python it raises ``RuntimeError``. Use ``open()`` and ``f.write()`` for file output from Python,
or call ``fprintf`` from ``@njit``.

**``sscanf`` runs under @njit only.**
In plain Python, ``int()``, ``float()`` and ``re`` parse input instead.
A call of
``sscanf`` from plain Python raises ``NotImplementedError``, and the message names those alternatives.

**Stdout buffering.** ``stdout`` is line-buffered on a terminal and block-buffered when redirected.
The plain-Python implementation flushes ``sys.stdout`` after each ``printf``, in imitation of a terminal's line
buffering.
Under ``@njit``, call ``fflush(stdout())`` when the output must appear immediately, for example before a long
computation or under pytest's ``capfd``.

**Caching.** A ``@njit(cache=True)`` caller caches across processes: each call site emits an extern reference to the
libc symbol and a global constant for the format string, both deterministic. The JIT linker resolves the libc symbol
in each process, so the cached IR is correct under ASLR. No ``@proxy`` indirection is needed, since the variadic
intrinsics inline the call into the caller.

Example: logging to stderr with `fprintf(3) <https://man7.org/linux/man-pages/man3/fprintf.3.html>`_, in both
modes:

.. code-block:: python

    from numba import njit
    from numbox.core.bindings.fmtio import fprintf
    from numbox.core.bindings.libc import fflush
    from numbox.core.bindings.stdio import stderr

    def warn(code, msg):
        fprintf(stderr(), "warning code=%d: %s\n", code, msg)
        fflush(stderr())

    warn(7, "disk getting full")        # plain Python
    njit(warn)(7, "disk getting full")  # jitted — identical output

Example: formatting into a buffer with `snprintf(3) <https://man7.org/linux/man-pages/man3/snprintf.3.html>`_,
detecting truncation and decoding:

.. code-block:: python

    import numpy as np
    from numba import njit
    from numbox.core.bindings.fmtio import snprintf
    from numbox.utils.lowlevel import array_data_p

    def fmt_range(lo, hi, buf):
        return snprintf(array_data_p(buf), buf.size, "[%d:%d]", lo, hi)

    buf = np.zeros(64, dtype=np.uint8)
    n = njit(fmt_range)(7, 11, buf)
    # Portable truncation check (works on Linux/macOS C99 snprintf
    # *and* Windows MSVCRT _snprintf — see snprintf docstring):
    truncated = (n < 0) or (n >= buf.size)
    if not truncated:
        msg = bytes(buf[:n]).decode()  # "[7:11]"

.. warning::
   The truncation semantics of ``snprintf`` **differ on Windows**. POSIX and C99 ``snprintf`` return the number of
   bytes that would have been written, without the NUL, and always NUL-terminate the buffer when ``size > 0``. The
   Windows binding calls MSVCRT's `_snprintf
   <https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/snprintf-snprintf-snprintf-l-snwprintf-snwprintf-l>`_,
   which returns ``-1`` on truncation and does not guarantee a NUL terminator. Declaring the UCRT's C99 ``snprintf``
   directly crashed with an access violation: the UCRT exports an inline wrapper from its header, whose shape in the
   process does not match the plain ``declare i32 @snprintf(...)`` LLVM declaration. The check ``(rc < 0) or (rc >=
   size)`` works on every platform.

**Parsing: `sscanf <https://man7.org/linux/man-pages/man3/sscanf.3.html>`_.**
``sscanf`` is the inverse of the printf family: it parses fields from a NUL-terminated input buffer into output
pointers that the caller supplies. Its shape differs from the writers':

- ``buf`` is an ``intp`` that points at the input bytes, for example from ``get_unicode_data_p``.
- The variadic ``*args`` are ``intp`` *output pointers*. Each points at writable storage that sscanf fills according
  to the corresponding format specifier; the pointer is usually obtained with ``array_data_p`` from a one-element
  numpy array of the right dtype.
- The return value is the number of items assigned (``int32``), or ``-1`` (``EOF``) when the input fails before the
  first conversion.

Unlike the printf family, ``sscanf`` runs under **@njit only**. A plain-Python call raises ``NotImplementedError``,
and the message names Python's ``int()``, ``float()`` and ``re`` for parsing. There is **no default argument
promotion**, since pointers are not promoted, and **no string conversion**, since the arguments are output pointers
rather than values.
The binding checks only that every variadic argument has type ``intp``: a value of another type is refused, and an
integer of that type passes as a pointer would.
The storage a pointer refers to must still have the size the format
specifier needs, which the binding cannot check:

================   =============================================
Format spec        Required output points at
================   =============================================
``%hhd``           ``int8`` (1 byte)
``%hd``            ``int16`` (2 bytes)
``%d``             ``int32`` (4 bytes)
``%ld``            ``int64`` on LP64 (Linux/macOS); ``int32`` on Win64 (LLP64): ``long`` is 8 bytes on LP64 and 4 bytes on Win64
``%lld``           ``int64`` (8 bytes, portable across LP64 and LLP64)
``%u``             ``uint32``
``%llu``           ``uint64``
``%f``             ``float32`` (4 bytes; not double, which ``%lf`` reads)
``%lf``            ``float64`` (8 bytes)
``%s``             ``char`` buffer (the caller is responsible for the buffer's size and the space for the NUL)
``%n``             accepted in ``sscanf`` only; writes the number of bytes consumed so far to the
                   corresponding output pointer. ``printf``, ``fprintf`` and ``snprintf`` reject ``%n`` at
                   compile time (``TypingError``): it is a memory-write hazard that static analyzers flag and
                   glibc's ``_FORTIFY_SOURCE`` disables, and Python's ``%`` operator rejects it as well, so
                   the rejection matches both modes. Use ``snprintf`` and ``len()`` for a byte count in the
                   writer family. ``%%n`` (a literal ``%`` followed by ``n``) is allowed.
================   =============================================

The ``%ld`` row is the most common source of cross-platform errors.
Use ``%lld`` with an ``int64`` output array for a portable 8-byte width.

Example: parsing a "<int> <double>" pair into typed numpy arrays:

.. code-block:: python

    import numpy as np
    from numba import njit
    from numbox.core.bindings.fmtio import sscanf
    from numbox.utils.lowlevel import array_data_p, get_unicode_data_p

    @njit(cache=True)
    def parse_pair(text_p, n_out, x_out):
        return sscanf(text_p, "%d %lf",
                      array_data_p(n_out), array_data_p(x_out))

    n_out = np.zeros(1, dtype=np.int32)
    x_out = np.zeros(1, dtype=np.float64)
    rc = parse_pair(get_unicode_data_p("42 3.14"), n_out, x_out)
    # rc == 2; n_out[0] == 42; x_out[0] == 3.14

Modules
++++++++

numbox.core.bindings.abi
------------------------

.. automodule:: numbox.core.bindings.abi
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.libc
-------------------------

.. automodule:: numbox.core.bindings.libc
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.errno
--------------------------

.. automodule:: numbox.core.bindings.errno
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.stdio
--------------------------

.. automodule:: numbox.core.bindings.stdio
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.strerror
-----------------------------

.. automodule:: numbox.core.bindings.strerror
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.fmtio
--------------------------

.. automodule:: numbox.core.bindings.fmtio
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.libm
-------------------------

.. automodule:: numbox.core.bindings.libm
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.call
-------------------------

.. automodule:: numbox.core.bindings.call
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings.signatures
-------------------------------

.. automodule:: numbox.core.bindings.signatures
   :members:
   :show-inheritance:
   :undoc-members:


numbox.core.bindings.utils
--------------------------

.. automodule:: numbox.core.bindings.utils
   :members:
   :show-inheritance:
   :undoc-members:
