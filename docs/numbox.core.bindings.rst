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

- **Linux (glibc and musl)** — data symbols (``stdout``, ``stderr``, ``stdin`` global variables)
- **macOS Darwin** — data symbols (``__stdoutp``, ``__stderrp``, ``__stdinp`` — what the libc headers'
  ``stdout`` / ``stderr`` / ``stdin`` macros expand to)
- **Windows** — accessor function (``__acrt_iob_func(0|1|2)``); UCRT-only (Windows 10+)

Both shapes are wrapped behind a uniform ``() -> intp`` interface using extern-symbol references in LLVM IR —
never literal addresses — so that ``cache=True`` remains correct under ASLR: the address is resolved at
JIT link time on each run rather than being baked into the cached object.

**errno.** ``errno_get()`` and ``errno_set(v)`` reach the per-thread errno location on every call via the
platform's accessor function (``__errno_location`` on glibc, ``__error`` on Darwin, ``_errno`` on
Windows). This makes the wrappers correct under ``@njit(parallel=True)``: each ``prange`` worker sees
its own thread's errno. A Python caller observes errno set inside a normal ``@njit`` function (same OS
thread), but not errno set inside a ``prange`` worker (different OS thread).

**Thread-safe strerror.** ``strerror_safe(errnum, buf, buflen)`` writes the error message into a
caller-supplied buffer, returning 0 on success and a positive errno code on failure. The underlying
symbol is selected at lowering time:

- **glibc Linux** — ``__xpg_strerror_r`` (always present on glibc 2.0+; POSIX-form)
- **musl Linux** — ``strerror_r`` (POSIX-form on musl)
- **macOS Darwin** — ``strerror_r`` (POSIX-form)
- **Windows** — ``strerror_s`` with reordered args (buffer, size, errnum)

Other Linux libcs are not supported: on glibc the ``strerror_r`` symbol is the GNU form (returns
``char *``) and would not match the POSIX-shaped IR this module generates. The Linux selector only
falls through to ``strerror_r`` when ``__xpg_strerror_r`` is absent — a condition that holds on musl
but not on glibc.

The Linux probe is verified by an IR-inspection test (Linux-only) that monkeypatches
``ll.address_of_symbol`` to confirm the fallback to ``strerror_r`` works. The musl path is independently
verified by a small Alpine-container CI job that uses ``nm`` to confirm ``strerror_r`` is present in the
libc shared object. See module docstrings below for caller idioms.

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
