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
data symbols (glibc exports ``stdout``, ``stderr``, ``stdin`` as global variables) or accessor functions
(musl, macOS, and Windows export them via macros that expand to function calls). Both shapes are wrapped
behind a uniform ``() -> intp`` interface using extern-symbol references in LLVM IR — never literal
addresses — so that ``cache=True`` remains correct under ASLR: the address is resolved at JIT link time
on each run rather than being baked into the cached object.

**errno.** ``errno_get()`` and ``errno_set(v)`` reach the per-thread errno location on every call via the
platform's accessor function (``__errno_location`` on glibc, ``__error`` on Darwin, ``_errno`` on
Windows). This makes the wrappers correct under ``@njit(parallel=True)``: each ``prange`` worker sees
its own thread's errno. Note that a Python caller cannot observe an errno value set inside a ``@njit``
function because Python and the JIT run on different threads.

**Thread-safe strerror.** ``strerror_safe(errnum, buf, buflen)`` writes the error message into a
caller-supplied buffer, returning 0 on success and a positive errno code on failure. The underlying
symbol is selected at lowering time:

- **glibc Linux** — uses ``__xpg_strerror_r`` when present (POSIX form), falling back to ``strerror_r``
- **musl Linux** — uses ``strerror_r`` (and ``__xpg_strerror_r`` if exported as an alias; both are POSIX-form on musl)
- **macOS Darwin** — uses ``strerror_r``
- **Windows** — uses ``strerror_s`` with reordered args (buffer, size, errnum)

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
