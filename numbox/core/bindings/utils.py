from ctypes import CDLL
from ctypes.util import find_library
from platform import system

from llvmlite import ir as llir
from numba.core.errors import TypingError
from numba.core.types import Literal, intp


platform_ = system()


def extract_literal_str(binding_name, ty, *, field="argument"):
    """Extract the Python str value of a ``Literal[str]`` type, or raise a clear ``TypingError``.

    The error message names the binding and the field. Intrinsics that require a compile-time string use this
    function, for example for printf format strings, libc function names and stdio handle names. ``field`` labels the
    offending argument in the error message.
    """
    if not isinstance(ty, Literal):
        raise TypingError(
            f"{binding_name}: {field} must be a literal str, got {ty!r}"
        )
    val = ty.literal_value
    if not isinstance(val, str):
        raise TypingError(
            f"{binding_name}: {field} must be a Python str, got {type(val).__name__}"
        )
    return val


def intp_ll_type(context=None):
    """Return the LLVM integer type that matches numba's ``intp`` on the current platform.

    Pass the numba codegen ``context`` when it is available, so that the type comes from
    ``context.get_value_type(intp)``. That call is the canonical pattern for platform-dependent widths, such as
    size_t, ssize_t and ptrdiff_t, in intrinsics. Outside codegen, for example in test helpers and IR-rendering
    probes, pass ``None`` to use the fixed fallback ``llir.IntType(intp.bitwidth)``. numba always lowers intp to the
    same bitwidth, so the two paths produce identical LLVM types.
    """
    if context is not None:
        return context.get_value_type(intp)
    return llir.IntType(intp.bitwidth)


def _windows_bundled_dll_path(name):
    """Find a DLL bundled with the Python distribution on Windows, on a best-effort basis.

    The function tries these paths in order:

    - ``<sys.prefix>/DLLs/<name>.dll``, the CPython layout, which also covers installs outside a venv;
    - ``<sys.base_prefix>/DLLs/<name>.dll``, the base Python of a venv;
    - ``<sys.base_prefix>/Library/bin/<name>.dll``, the conda layout.

    It returns the absolute path of the first candidate that exists, or None if no bundled DLL is found.
    """
    import os
    import sys
    dirs = [
        os.path.join(sys.prefix, "DLLs"),
        os.path.join(sys.base_prefix, "DLLs"),
        os.path.join(sys.base_prefix, "Library", "bin"),
    ]
    for d in dirs:
        candidate = os.path.join(d, f"{name}.dll")
        if os.path.exists(candidate):
            return candidate
    return None


def _resolve_lib_path(name):
    """Resolve a library name to a path that CDLL can load.

    The resolution depends on the platform:

    - On Linux and Darwin, the function returns ctypes.util.find_library(name).
    - On Windows, "c" and "m" resolve to find_msvcrt(). For any other name, the function prefers
      _windows_bundled_dll_path(name), the DLL bundled with the Python distribution in <prefix>/DLLs/ or
      <prefix>/Library/bin/, over find_library(name), which searches PATH.

    The PATH lookup comes last because PATH on Windows may contain copies of common DLLs that third-party tools ship.
    Such a copy can be statically configured for the internal use of its tool, and it can fail with an access
    violation when another program calls it. The motivating example is AWS CLI v2, which the GitHub Actions
    windows-latest runners ship. Its C:\\Program Files\\Amazon\\AWSCLIV2\\sqlite3.dll is on PATH and exports the full
    SQLite symbol surface, but it writes to NULL inside sqlite3_open when a process other than aws.exe calls it.
    CPython's bundled sqlite3.dll is the canonical SQLite for the Python ecosystem on Windows, so the function prefers
    it whenever it is present.

    The function returns the path string, or None if no path can be resolved.
    """
    if platform_ in ("Darwin", "Linux"):
        return find_library(name)
    if platform_ == "Windows":
        from ctypes.util import find_msvcrt
        if name in ("c", "m"):
            return find_msvcrt()
        bundled = _windows_bundled_dll_path(name)
        if name == "sqlite3":
            # _sqlite3.pyd is dynamically linked to <prefix>/DLLs/sqlite3.dll, so that DLL is a hard prerequisite of
            # every working Python on Windows. The function never falls back to find_library here, because third-party
            # tools may ship statically configured sqlite3.dll copies that fail with an access violation when an
            # external caller uses them. AWS CLI v2 is the motivating example.
            return bundled
        if bundled is not None:
            return bundled
        return find_library(name)
    return None


_loaded_libs = {}


def load_lib(name):
    """Load the library ``name`` in global symbol mode and return its cached CDLL handle.

    The first call loads the library, and subsequent calls return the same handle from ``_loaded_libs``.

    ``find_library`` is a relatively basic utility that mostly prefixes ``lib`` and appends the platform extension.
    A user-compiled shared library may not be on the default search path of the system loader. In that case,
    consider setting ``DYLD_LIBRARY_PATH`` (macOS) or ``LD_LIBRARY_PATH`` (Linux) before the first ``load_lib``
    call. The loader consults those environment variables before its built-in search. For full control over the
    path, use :func:`load_lib_path` instead.

    The cache retains the handle for the lifetime of the process, so modules can share it without cross-module
    imports. The cache is also necessary because ``ctypes.CDLL.__del__`` calls ``dlclose`` or ``FreeLibrary``.
    Without the cached reference, the OS-level reference count would drop to zero when this function returns, and
    the library could be unloaded. The unload would invalidate any extern-ref symbols that LLVM's JIT linker has
    already resolved into module IR. The returned handle also lets ``proxy_if_available`` check whether a symbol is
    present, with ``hasattr(handle, func_name)``.
    """
    handle = _loaded_libs.get(name)
    if handle is None:
        handle = _load_lib_with_handle(name)
        _loaded_libs[name] = handle
    return handle


def _load_lib_with_handle(name):
    """Resolve ``name`` with :func:`_resolve_lib_path`, load it in global symbol mode, and return the CDLL handle.

    This helper is internal. Use :func:`load_lib` instead, because it caches the handle.
    """
    path = _resolve_lib_path(name)
    if path is None:
        # This branch preserves the historical Windows fallback for c and m: msvcrt through ctypes.cdll.
        if platform_ == "Windows" and name in ("c", "m"):
            import ctypes
            return ctypes.cdll.msvcrt
        raise RuntimeError(f"Could not find shared library for {name}")
    if platform_ in ("Darwin", "Linux"):
        from os import RTLD_GLOBAL
        return CDLL(path, mode=RTLD_GLOBAL)
    if platform_ == "Windows":
        return CDLL(path, winmode=0)
    raise RuntimeError(f"Platform {platform_} is not supported, yet.")


def load_lib_path(path):
    """Load a shared library by an identifier that ``ctypes.CDLL`` accepts, and return the handle without caching it.

    ``path`` can be any string that ``CDLL`` accepts: an absolute path, a soname such as ``libm.so.6`` (as
    ``ctypes.util.find_library`` returns it), or a bare filename that the loader can resolve. Linux and Darwin load
    the library with ``RTLD_GLOBAL``, so that its symbols are visible to LLVM's JIT. Windows loads it with
    ``winmode=0``. Prefer :func:`load_lib` when the library is referenced by name and from several modules in the
    same process.
    """
    if platform_ in ("Darwin", "Linux"):
        from os import RTLD_GLOBAL
        return CDLL(path, mode=RTLD_GLOBAL)
    if platform_ == "Windows":
        return CDLL(path, winmode=0)
    raise RuntimeError(f"Platform {platform_} is not supported, yet.")
