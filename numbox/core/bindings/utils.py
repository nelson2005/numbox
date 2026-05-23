from ctypes import CDLL
from ctypes.util import find_library
from platform import system

from llvmlite import ir as llir
from numba.core.errors import TypingError
from numba.core.types import Literal, intp


platform_ = system()


def extract_literal_str(binding_name, ty, *, field="argument"):
    """Extract the Python str value of a ``Literal[str]`` type, or raise
    a clean ``TypingError`` naming the binding and the field.

    Used by intrinsics that require a compile-time string (e.g. printf
    format strings, libc function names, stdio handle names). ``field``
    labels the offending argument in the error message.
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
    """LLVM integer type matching numba's ``intp`` on the current platform.

    Pass the numba codegen ``context`` when available so the type is
    derived via ``context.get_value_type(intp)`` — the canonical pattern
    for platform-dependent widths (size_t, ssize_t, ptrdiff_t) in
    intrinsics. When called outside codegen (test helpers, IR-rendering
    probes), pass ``None`` for the locked-in fallback
    ``llir.IntType(intp.bitwidth)``; numba's intp lowering is locked to
    the same bitwidth, so the two paths produce identical LLVM types.
    """
    if context is not None:
        return context.get_value_type(intp)
    return llir.IntType(intp.bitwidth)


def _windows_bundled_dll_path(name):
    """Best-effort: find a DLL bundled with the Python distribution on Windows.

    Tries (in order):
    - <sys.prefix>/DLLs/<name>.dll (CPython, also catches non-venv installs)
    - <sys.base_prefix>/DLLs/<name>.dll (venv -> base Python)
    - <sys.base_prefix>/Library/bin/<name>.dll (conda layout)

    Returns the absolute path of the first existing candidate, or None if no
    bundled DLL is found.
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
    """Resolve a library name to a CDLL-loadable path.

    Per-platform logic (same as the original load_lib, plus the Windows
    bundled-DLL fallback):
    - Linux / Darwin: ctypes.util.find_library(name)
    - Windows: for "c"/"m", find_msvcrt(); otherwise find_library(name) with
      _windows_bundled_dll_path(name) as a fallback for DLLs Python ships in
      its DLLs/ directory but doesn't put on PATH (notably sqlite3.dll).

    Returns the path string, or None if no path can be resolved.
    """
    if platform_ in ("Darwin", "Linux"):
        return find_library(name)
    if platform_ == "Windows":
        from ctypes.util import find_msvcrt
        if name in ("c", "m"):
            return find_msvcrt()
        path = find_library(name)
        if path is None:
            path = _windows_bundled_dll_path(name)
        return path
    return None


def load_lib(name):
    """Load library `name` in global symbol mode. Legacy contract: returns None."""
    load_lib_with_handle(name)


def load_lib_with_handle(name):
    """Load library `name` in global symbol mode AND return the CDLL handle.

    Returning the handle enables proxy_if_available / cres_if_available to
    query symbol presence via hasattr(handle, name). The handle is also kept
    alive by the caller, preventing the OS from unloading the library after
    the symbol registration completes.
    """
    path = _resolve_lib_path(name)
    if path is None:
        # Preserve the historical Windows c/m fallback (msvcrt via ctypes.cdll).
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
    """Load a shared library by ``ctypes.CDLL``-acceptable identifier.

    Accepts any string ``CDLL`` accepts — an absolute path, a soname
    (e.g. ``libm.so.6`` as returned by ``ctypes.util.find_library``), or
    a bare filename resolvable by the loader. Linux/Darwin use
    ``RTLD_GLOBAL`` so symbols reach LLVM's JIT; Windows uses
    ``winmode=0``. Unlike ``load_lib(name)``, the handle is returned so
    callers can check symbol presence with ``hasattr``.
    """
    if platform_ in ("Darwin", "Linux"):
        from os import RTLD_GLOBAL
        return CDLL(path, mode=RTLD_GLOBAL)
    if platform_ == "Windows":
        return CDLL(path, winmode=0)
    raise RuntimeError(f"Platform {platform_} is not supported, yet.")
