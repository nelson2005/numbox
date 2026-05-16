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

ABI dispatch
++++++++++++

LLVM's JIT treats ABI lowering as a frontend responsibility — it won't insert the right calling convention
for struct args/returns by itself. ``numbox.core.bindings.call._call_lib_func`` dispatches per platform and
per struct shape, using primitives from ``numbox.core.bindings.abi`` (platform identification via
``_current_platform``, struct-shape classification via ``_classify``, struct-size measurement via
``_struct_bytes``). The two ABI families that matter:

- **Windows x64** — passes aggregates >8 bytes via caller-allocated pointers and returns them via ``sret``;
  sizes 1/2/4/8 go directly in registers.
- **SysV x86-64 / AAPCS64** — pass and return ≤16-byte aggregates directly in GP registers; on SysV x86-64,
  >16-byte by-value args use a ``byval`` + ``optnone`` + ``noinline`` idiom so the LLVM optimizer doesn't
  elide the caller-side stack copy before the callee reads it.

References:

- `llvmlite#300 <https://github.com/numba/llvmlite/issues/300#issuecomment-327235846>`_
- `llvm-project#85417 <https://github.com/llvm/llvm-project/issues/85417>`_
- `Windows x64 calling convention <https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention>`_
- `AAPCS64 <https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst>`_

Stdio handles, errno, and thread-safe strerror
++++++++++++++++++++++++++++++++++++++++++++++

**Stdio handles.** ``stdout()``, ``stderr()``, and ``stdin()`` are exposed as JIT-callable functions
rather than module-level Python constants because the C library's stdio ``FILE *`` values can be either
data symbols or accessor functions:

- **Linux (glibc and musl)** — data symbols (``stdout``, ``stderr``, ``stdin`` global variables, declared in
  `<stdio.h> <https://www.gnu.org/software/libc/manual/html_node/Standard-Streams.html>`_)
- **macOS Darwin** — data symbols (``__stdoutp``, ``__stderrp``, ``__stdinp`` — what the libc headers'
  ``stdout`` / ``stderr`` / ``stdin`` macros expand to per
  `Apple's stdio.h <https://opensource.apple.com/source/Libc/Libc-1439.40.11/include/stdio.h.auto.html>`_)
- **Windows** — accessor function (`__acrt_iob_func(0|1|2)
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/internal-crt-globals-and-functions>`_);
  UCRT-only (Windows 10+ / VS 2015+)

Both shapes are wrapped behind a uniform ``() -> intp`` interface using extern-symbol references in LLVM IR —
never literal addresses — so that ``cache=True`` remains correct under ASLR: the address is resolved at
JIT link time on each run rather than being baked into the cached object.

Example — write to stderr from inside @njit:

.. code-block:: python

    from numba import njit, types
    from numbox.core.bindings import stderr
    from numbox.core.bindings._c import fputs, fflush

    @njit(cache=True)
    def log_to_stderr(msg_p):
        fputs(msg_p, stderr())
        fflush(stderr())

**errno.** ``errno_get()`` and ``errno_set(v)`` reach the per-thread errno location on every call via the
platform's accessor function (`__errno_location
<https://man7.org/linux/man-pages/man3/errno.3.html>`_ on Linux glibc and musl,
`__error <https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/intro.2.html>`_
on Darwin, `_errno <https://learn.microsoft.com/en-us/cpp/c-runtime-library/errno-doserrno-sys-errlist-and-sys-nerr>`_
on Windows UCRT). This makes the wrappers correct under ``@njit(parallel=True)``: each ``prange`` worker
sees its own thread's errno. A Python caller observes errno set inside a normal ``@njit`` function (same
OS thread), but not errno set inside a ``prange`` worker (different OS thread).

Example — read and report errno after a syscall-style binding:

.. code-block:: python

    from numba import njit
    from numbox.core.bindings import errno_get, errno_set

    @njit(cache=True)
    def clear_then_call_and_report(do_work):
        errno_set(0)
        result = do_work()
        return result, errno_get()

**Thread-safe strerror.** ``strerror_safe(errnum, buf, buflen)`` writes the error message into a
caller-supplied buffer, returning 0 on success and a positive errno code on failure. The underlying
symbol is selected at lowering time:

- **glibc Linux** — `__xpg_strerror_r
  <https://codebrowser.dev/glibc/glibc/string/xpg-strerror.c.html>`_ (POSIX XSI form, present on glibc
  2.3.4+ which shipped in 2004)
- **musl Linux** — also ``__xpg_strerror_r``, exported as a `weak alias
  <https://git.musl-libc.org/cgit/musl/tree/src/string/strerror_r.c>`_ to musl's own ``strerror_r``
  (which is itself the POSIX form; musl never shipped the GNU char-pointer form)
- **macOS Darwin** — `strerror_r
  <https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/strerror_r.3.html>`_
  (POSIX form)
- **Windows** — `strerror_s
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/strerror-s-strerror-s-wcserror-s-wcserror-s>`_
  with reordered args (buffer, size, errnum)

Other Linux libcs are not supported: on glibc the plain ``strerror_r`` symbol is the
`GNU form <https://man7.org/linux/man-pages/man3/strerror_r.3.html>`_ (returns ``char *``) and would
not match the POSIX-shaped IR this module generates. The Linux selector unconditionally picks
``__xpg_strerror_r``, which resolves correctly on both glibc and musl (the musl path goes through the
weak alias). A ``strerror_r`` fallback remains in the selector as defense-in-depth in case a future
libc drops ``__xpg_strerror_r``, but the fallback is currently unreachable on every supported libc.

The Linux selector logic is verified by an IR-inspection test (Linux-only) that monkeypatches
``ll.address_of_symbol`` to drive the fallback branch. The musl symbol layout is independently verified
by a small Alpine-container CI job that confirms (a) musl exports ``strerror_r``, (b) musl ALSO exports
``__xpg_strerror_r``, and (c) both names resolve to the same address (i.e. the weak alias holds).

Example — render the message for ``ENOENT`` (errno 2 on POSIX) into a buffer:

.. code-block:: python

    import errno
    import numpy as np
    from numba import njit
    from numbox.core.bindings import strerror_safe
    from numbox.utils.lowlevel import array_data_p

    @njit(cache=True)
    def explain(errnum, buf):
        rc = strerror_safe(errnum, array_data_p(buf), buf.size)
        return rc

    buf = np.zeros(128, dtype=np.uint8)
    rc = explain(errno.ENOENT, buf)
    msg = bytes(buf[:buf.tolist().index(0)]).decode()
    # rc == 0; msg is the (locale-dependent) string for ENOENT

Variadic formatted I/O — printf / fprintf / snprintf / sscanf
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``printf``, ``fprintf``, ``snprintf``, and ``sscanf`` are ``@intrinsic``
shells that emit direct extern variadic calls to libc. LLVM's backend handles the
platform-specific variadic ABI (SysV x86-64 ``AL``-register convention,
Win64 FP shadow, AAPCS64 named/anonymous split) automatically when the
function is declared with ``var_arg=True``; the bindings handle C
default-argument promotion, the pointer-as-``intp`` shape numbox uses
throughout, and embedding the format string as a UTF-8 IR global constant.

**Call convention.** Numba's ``@intrinsic`` doesn't accept Python-level
``*args``, so the variadic arguments are passed as a **tuple literal** —
the same idiom ``_call_lib_func`` uses elsewhere in the package::

    printf("x = %d, ratio = %.3f\n", (n, ratio))
    fprintf(stderr(), "warning: %s\n", (msg_p,))
    snprintf(array_data_p(buf), buf.size, "[%d:%d]", (lo, hi))
    printf("no args here\n", ())

**Format string must be a literal.** Required to embed it as an IR global
constant at typing time, the same constraint a C compiler operates under
when emitting a format-checked printf call. A runtime-built ``unicode``
raises a clean ``TypingError`` at call typing time.

**Format string encoding: UTF-8.** Non-ASCII codepoints in the literal
are encoded as UTF-8 byte sequences and embedded into the IR global.
printf treats every non-``%`` byte as opaque pass-through, so the bytes
flow through libc to stdout / FILE\\* / the snprintf buffer unmodified.
Modern terminals, files, and Windows 10+ consoles all expect UTF-8, so
``printf("Цена: %d\n", (n,))`` renders correctly out of the box.

  .. note::
     ``%-Ns`` width is byte-counted by printf in every libc, so non-ASCII
     output won't right-pad to a codepoint count. That's printf's
     contract, not the binding's. Pad in numba-side string formatting
     (``f"{s:<10}"``) before passing through ``%s`` if codepoint-counted
     widths matter.

**C ABI default-argument promotion (handled by the binding):**

============   ============================
Numba type     Promoted to (in varargs slot)
============   ============================
``float32``    ``float64`` (``fpext``)
``int8``       ``int32`` (``sext``)
``int16``      ``int32`` (``sext``)
``uint8``      ``int32`` (``zext``)
``uint16``     ``int32`` (``zext``)
``int32``      pass through
``int64``      pass through
``uint32``     pass through
``uint64``     pass through
``float64``    pass through
``intp``       pass through (use ``%s`` for char-pointer, ``%lld``/``%p`` for the integer)
============   ============================

The user is responsible for matching format specifiers to argument types
the same way a C programmer is. For example, ``%lld`` for ``int64`` on
LP64; ``%d`` for ``int32``; ``%s`` for an ``intp`` from ``get_unicode_data_p``;
``%.3f`` for ``float64`` or ``float32`` (both promote to ``double``).

**Stdout buffering.** ``stdout`` is line-buffered when attached to a terminal
and block-buffered when redirected (a pipe, file, or pytest's ``capfd``
capture). Add an explicit ``fflush(stdout())`` after a ``printf`` if you
need the output to appear before the process exits. ``stderr`` is
traditionally unbuffered; ``fflush(stderr())`` is harmless.

**Caching.** ``@njit(cache=True)`` callers of ``printf`` / ``fprintf`` /
``snprintf`` cache cleanly across processes: each call site emits a
direct extern reference to the libc symbol and a deterministic format-
string global constant. The JIT linker resolves the libc symbol at link
time in each process, so the cached IR is ASLR-safe. No
``cres_cacheable`` indirection is needed (unlike the fixed-arg bindings,
these never route through a numba dispatcher whose ``id`` would be
ASLR-randomized).

Example — log to stderr with `fprintf(3) <https://man7.org/linux/man-pages/man3/fprintf.3.html>`_:

.. code-block:: python

    from numba import njit
    from numbox.core.bindings import fprintf, fflush, stderr
    from numbox.utils.lowlevel import get_unicode_data_p

    @njit(cache=True)
    def warn(code, msg_p):
        fprintf(stderr(), "warning code=%d: %s\n", (code, msg_p))
        fflush(stderr())

    warn(7, get_unicode_data_p("disk getting full"))

Example — format into a buffer with `snprintf(3) <https://man7.org/linux/man-pages/man3/snprintf.3.html>`_,
detect truncation, decode:

.. code-block:: python

    import numpy as np
    from numba import njit
    from numbox.core.bindings import snprintf
    from numbox.utils.lowlevel import array_data_p

    @njit(cache=True)
    def fmt_range(lo, hi, buf):
        return snprintf(array_data_p(buf), buf.size, "[%d:%d]", (lo, hi))

    buf = np.zeros(64, dtype=np.uint8)
    n = fmt_range(7, 11, buf)
    # Portable truncation check (works on Linux/macOS C99 snprintf
    # *and* Windows MSVCRT _snprintf — see snprintf docstring):
    truncated = (n < 0) or (n >= buf.size)
    if not truncated:
        msg = bytes(buf[:n]).decode()  # "[7:11]"

.. warning::
   ``snprintf`` truncation semantics **diverge on Windows**. POSIX / C99
   ``snprintf`` returns the would-have-written count (excluding NUL) and
   always NUL-terminates the buffer when ``size > 0``. The Windows
   binding targets MSVCRT's `_snprintf
   <https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/snprintf-snprintf-snprintf-l-snwprintf-snwprintf-l>`_
   — that returns ``-1`` on truncation and does NOT guarantee
   NUL-termination of the buffer. We attempted to bypass this by
   declaring UCRT's C99 ``snprintf`` directly, but the call crashed with
   an access violation: UCRT exports a header-inline wrapper whose
   in-process shape does not match the naive ``declare i32 @snprintf(...)``
   LLVM declaration. The portable check
   ``(rc < 0) or (rc >= size)`` works on every platform.

**Parsing direction: `sscanf <https://man7.org/linux/man-pages/man3/sscanf.3.html>`_.**
The inverse of the printf family: parse fields from a NUL-terminated input
buffer into caller-supplied output pointers. Shape differs from the writers:

- ``buf`` is an ``intp`` pointing at the input bytes (e.g. from
  ``get_unicode_data_p``).
- ``args`` is a tuple of ``intp`` *output pointers* — each one points at
  writable storage that sscanf fills based on the corresponding format
  specifier. Typically obtained via ``array_data_p`` of a 1-element numpy
  array of the right dtype.
- Returns the count of items successfully assigned (``int32``), or
  ``-1`` (``EOF``) on input failure before the first conversion.

Unlike printf-family, there is **no default-argument promotion** for
sscanf's variadic args (pointers don't promote). The binding validates
only that every variadic arg has type ``intp``, so you can't accidentally
pass an integer value where a pointer is expected. The pointed-to storage
must still be the right size for the format spec — the binding cannot
check that:

================   =============================================
Format spec        Required output points at
================   =============================================
``%hhd``           ``int8`` (1 byte)
``%hd``            ``int16`` (2 bytes)
``%d``             ``int32`` (4 bytes)
``%lld``           ``int64`` (8 bytes)
``%u``             ``uint32``
``%llu``           ``uint64``
``%f``             ``float32`` (4 bytes — NOT double, ``%lf`` is for that)
``%lf``            ``float64`` (8 bytes)
``%s``             ``char`` buffer (caller responsible for size + NUL room)
``%n``             forbidden; security hole disabled in fortified builds
================   =============================================

Example — parse a "<int> <double>" pair into typed numpy slots:

.. code-block:: python

    import numpy as np
    from numba import njit
    from numbox.core.bindings import sscanf
    from numbox.utils.lowlevel import array_data_p, get_unicode_data_p

    @njit(cache=True)
    def parse_pair(text_p, n_out, x_out):
        return sscanf(text_p, "%d %lf",
                      (array_data_p(n_out), array_data_p(x_out)))

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

numbox.core.bindings._c
-----------------------

.. automodule:: numbox.core.bindings._c
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._errno
---------------------------

.. automodule:: numbox.core.bindings._errno
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._stdio
---------------------------

.. automodule:: numbox.core.bindings._stdio
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._strerror
------------------------------

.. automodule:: numbox.core.bindings._strerror
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._fmtio
---------------------------

.. automodule:: numbox.core.bindings._fmtio
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._math
--------------------------

.. automodule:: numbox.core.bindings._math
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._sqlite
----------------------------

.. automodule:: numbox.core.bindings._sqlite
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
