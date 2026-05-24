"""Windows sqlite3 diagnostic — runs first alphabetically so its output
appears in CI logs before any sqlite3 C call AVs the worker.

Writes to ``sys.__stderr__`` (not ``sys.stderr``) so pytest's capture
plugin doesn't swallow the output. Pure diagnostic: never fails.

Once Windows is working, this file can be deleted or kept as a permanent
smoke test of lib resolution + direct-ctypes calls.
"""
import sys


def test_sqlite_windows_diag():
    out = sys.__stderr__
    out.write("\n=== SQLite Windows diagnostic ===\n")
    out.flush()

    try:
        from numbox.core.bindings.utils import (
            _resolve_lib_path, _windows_bundled_dll_path, platform_,
        )
        out.write(f"platform_: {platform_}\n")
        out.write(f"_resolve_lib_path('sqlite3'): {_resolve_lib_path('sqlite3')!r}\n")
        if platform_ == "Windows":
            out.write(f"_windows_bundled_dll_path('sqlite3'): {_windows_bundled_dll_path('sqlite3')!r}\n")
            import sys as _s
            out.write(f"  sys.prefix:      {_s.prefix!r}\n")
            out.write(f"  sys.base_prefix: {_s.base_prefix!r}\n")
            import os
            for d in (_s.prefix, _s.base_prefix):
                dlls = os.path.join(d, "DLLs")
                if os.path.isdir(dlls):
                    sq = [f for f in os.listdir(dlls) if "sqlite" in f.lower()]
                    out.write(f"  {dlls}: sqlite files = {sq}\n")
    except Exception as e:
        out.write(f"resolve probe failed: {type(e).__name__}: {e}\n")
    out.flush()

    try:
        from numbox.core.bindings._sqlite_conn import sqlite3_lib
        out.write(f"sqlite3_lib type: {type(sqlite3_lib).__name__}\n")
        out.write(f"sqlite3_lib._name: {getattr(sqlite3_lib, '_name', '?')!r}\n")
        out.write(f"sqlite3_lib._handle: {getattr(sqlite3_lib, '_handle', '?')!r}\n")
        for sym in ('sqlite3_open', 'sqlite3_close', 'sqlite3_libversion',
                    'sqlite3_prepare_v2', 'sqlite3_step'):
            out.write(f"  hasattr({sym}): {hasattr(sqlite3_lib, sym)}\n")
    except Exception as e:
        out.write(f"sqlite3_lib import failed: {type(e).__name__}: {e}\n")
    out.flush()

    try:
        import ctypes
        sqlite3_lib.sqlite3_libversion.restype = ctypes.c_char_p
        v = sqlite3_lib.sqlite3_libversion()
        out.write(f"libversion (direct ctypes): {v!r}\n")
    except Exception as e:
        out.write(f"direct ctypes libversion failed: {type(e).__name__}: {e}\n")
    out.flush()

    try:
        from numbox.core.bindings._sqlite_conn import sqlite3_libversion
        from test.auxiliary_utils import str_from_p_as_int
        version_p = sqlite3_libversion()
        out.write(f"libversion (via @proxy): {str_from_p_as_int(version_p)!r}\n")
    except Exception as e:
        out.write(f"@proxy libversion failed: {type(e).__name__}: {e}\n")
    out.flush()

    out.write("=== End diagnostic ===\n")
    out.flush()
