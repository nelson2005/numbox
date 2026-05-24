# SQLite buildout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers-extended-cc:subagent-driven-development` (recommended) or `superpowers-extended-cc:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand `numbox/core/bindings/_sqlite*.py` from 3 wrappers to 60 (57 new + 3 retained) covering the standard prepare → bind → step → column → finalize lifecycle plus BLOB incremental I/O, six callback hooks, and version-sensitive 64-bit row-count accessors. Windows in-scope via a new generic bundled-DLL fallback.

**Architecture:** Thin `@proxy` wrappers around `_call_lib_func`, one wrapper file per coherent group (7 files), one constants module, no opinionated lifecycle abstractions. Version- and compile-flag-sensitive functions use the existing `proxy_if_available` decorator (lib-handle-driven presence check at module-import time). Cross-platform sqlite3 loading via a generic `_windows_bundled_dll_path` helper that searches CPython's `DLLs/` and conda's `Library/bin`.

**Tech Stack:** numba `@njit` + `@cfunc`, numbox `@proxy` / `proxy_if_available` / `_call_lib_func`, ctypes for fixture out-params, pytest, `pytest --durations=20`, flake8 `--max-line-length=127`.

**Spec:** [`docs/plans/sqlite-buildout/2026-05-23-design.md`](2026-05-23-design.md)

**Branch:** `feat/sqlite-buildout` (HEAD `205b8c4` after spec commit; based on `origin/main` @ `fa1173b` post-sync with upstream/main)

**Pre-flight (run once before Task 1):**

```bash
cd /home/erik/projects/numbox
git status --short                        # expect clean
git rev-parse HEAD                        # expect 205b8c4
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/pytest -x --durations=20 -q      # expect ~439 pass (baseline)
```

If pytest doesn't pass cleanly, stop and investigate; do not start Task 1 on a broken baseline.

## Pre-flight reading (mandatory before any task; subagent dispatchers MUST include this list in each prompt)

The buildout follows the established libc bindings pattern verbatim. **Before writing any new wrapper, read these files end-to-end** so the patterns are grounded in actual code rather than imagination. The numbduck article-campaign session learned this the hard way (drafts hallucinated library APIs until rewritten from a direct read of the source) — the same failure mode applies here. If a subagent reports "I'm not sure how X is structured" or "should this be Y or Z?", the answer is almost always in one of the files below; re-check before guessing.

**Required reading:**

1. [`numbox/core/bindings/_c.py`](../../../numbox/core/bindings/_c.py) — canonical `@proxy(sig, jit_options={"cache": True})` + `_call_lib_func("name", (args,))` shape. ~12 wrappers, very short, the model every new wrapper follows.
2. [`numbox/core/bindings/_stdio.py`](../../../numbox/core/bindings/_stdio.py) — module-scope `load_lib("c")`, signatures dict access, multi-function file structure.
3. [`numbox/core/bindings/signatures.py`](../../../numbox/core/bindings/signatures.py) — current `signatures_sqlite` shape and the `signatures = {**signatures_c, **signatures_m, **signatures_sqlite}` merge.
4. [`numbox/core/bindings/utils.py`](../../../numbox/core/bindings/utils.py) — current `load_lib` shape (Task 1 refactors this) + `load_lib_path` + `extract_literal_str` + `intp_ll_type`. Task 1's refactor extends this file; understand what's already there before editing.
5. [`numbox/core/bindings/call.py`](../../../numbox/core/bindings/call.py) — `_call_lib_func` itself, the platform-aware ABI dispatcher. **You don't need to understand the ABI logic** — just confirm the call shape is `_call_lib_func("name", (arg1, arg2, ...))`. The "Bindings: implementation gotchas" section in CLAUDE.md explains why the extern-ref pattern matters.
6. [`numbox/core/proxy/proxy.py`](../../../numbox/core/proxy/proxy.py) — `proxy(sig, jit_options=...)` workhorse AND `proxy_if_available(lib, sig, jit_options=...)`. **Read both docstrings fully** — they explain the `.as_func` contract that callers must guard with `hasattr(binding, "as_func")` for stubbed-out symbols.
7. [`numbox/utils/lowlevel.py`](../../../numbox/utils/lowlevel.py) — **per the project's own CLAUDE.md rule, "before designing anything that touches strings, pointers, or buffer ownership, read this file end-to-end first."** SQLite touches all three. Key utilities used in the test files: `array_data_p`, `get_str_from_p_as_int`, `get_unicode_data_p`.
8. [`CLAUDE.md`](../../../CLAUDE.md) — specifically the "Bindings: implementation gotchas" section (extern refs vs literal addresses, lifetime, platform-variable C types) and "Adding a New Binding" (the canonical 4-step procedure). The `long` / `time_t` / `size_t` warnings DON'T apply here (SQLite's C API uses `int` and `sqlite3_int64` explicit everywhere) — but knowing why they don't apply is part of being grounded.
9. [`docs/plans/sqlite-buildout/2026-05-23-design.md`](2026-05-23-design.md) — the design spec this plan implements. The function inventory tables (§2) and the cross-cutting design choices (§3) are the source of truth for signatures and patterns.

**Test-helper patterns (skim, not full read):**

- [`test/core/test_bindings.py`](../../../test/core/test_bindings.py) — the existing `test_sqlite`, `test_c_stdio`, `test_c_strings`, `test_c_env` show the test helper patterns this plan reuses verbatim: `_cstr` builder (returns `(keepalive, intp_address)`), `addressof(c_int64(0))` for out-params, `array_data_p` for numpy buffer addresses, `get_str_from_p_as_int` / `str_from_p_as_int` for C-string decode (njit-side vs Python-side).
- [`test/core/test_stdio_handles.py`](../../../test/core/test_stdio_handles.py) and [`test/core/test_errno.py`](../../../test/core/test_errno.py) — per-group test file pattern this buildout mirrors (one test file per wrapper file).

---

## Task 1: Refactor `load_lib` into `_resolve_lib_path` + `load_lib_with_handle`; add Windows bundled-DLL fallback

**Goal:** Make `load_lib` factor into a pure path resolver and a handle-returning loader so `proxy_if_available` has a CDLL handle to query, AND add a generic `_windows_bundled_dll_path` fallback that locates DLLs bundled with the Python distribution (CPython's `<base_prefix>/DLLs/`, conda's `<base_prefix>/Library/bin/`).

**Files:**
- Modify: `numbox/core/bindings/utils.py`
- Test: `test/core/test_bindings.py` (add a small test for `load_lib_with_handle` returning a handle; verifies `hasattr` works against it)

**Acceptance Criteria:**
- [ ] `_resolve_lib_path("c")` returns a non-None path on every supported platform
- [ ] `_resolve_lib_path("sqlite3")` returns a non-None path on Linux + macOS + Windows (the Windows path may come from the new bundled-DLL fallback)
- [ ] `load_lib("c")` still works (backwards compatible — discards the handle as before)
- [ ] `load_lib_with_handle("c")` returns a `CDLL` instance with `hasattr(handle, "strlen") is True`
- [ ] Existing tests still pass after the refactor

**Verify:**

```bash
cd /home/erik/projects/numbox
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/pytest test/core/test_bindings.py -x --durations=20 -v
venv/bin/flake8 --max-line-length=127 numbox/core/bindings/utils.py
```

Expected: all existing test_bindings tests pass; new `load_lib_with_handle` test passes; flake8 clean.

**Steps:**

- [ ] **Step 1: Replace `load_lib` implementation in `numbox/core/bindings/utils.py`**

Read the current file first, then replace the `load_lib` function with the three-function factoring. The full target shape:

```python
from ctypes import CDLL
from ctypes.util import find_library, find_msvcrt
from platform import system

from llvmlite import ir as llir
from numba.core.errors import TypingError
from numba.core.types import Literal, intp


platform_ = system()


def extract_literal_str(binding_name, ty, *, field="argument"):
    # ... unchanged ...


def intp_ll_type(context=None):
    # ... unchanged ...


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
    # ... unchanged ...
```

Keep `extract_literal_str`, `intp_ll_type`, and `load_lib_path` exactly as they were. Only `load_lib` is replaced (and the three new functions added).

- [ ] **Step 2: Add a test for `load_lib_with_handle` to `test/core/test_bindings.py`**

Add this test alongside the existing `test_load_lib_path_returns_handle_with_known_symbol`:

```python
def test_load_lib_with_handle_returns_queryable_handle():
    """load_lib_with_handle must return a CDLL the caller can hasattr-query —
    that's the whole point of the refactor (proxy_if_available uses hasattr)."""
    from numbox.core.bindings.utils import load_lib_with_handle, platform_
    name = "c" if platform_ != "Windows" else "c"
    handle = load_lib_with_handle(name)
    assert handle is not None
    # strlen is in libc on every supported platform
    assert hasattr(handle, "strlen")
    # A symbol that doesn't exist returns False
    assert not hasattr(handle, "definitely_not_a_real_symbol_xyzzy")
```

- [ ] **Step 3: Run tests + lint**

```bash
cd /home/erik/projects/numbox
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/pytest test/core/test_bindings.py -x --durations=20 -v
venv/bin/flake8 --max-line-length=127 numbox/core/bindings/utils.py
```

Expected: existing `test_c`, `test_sqlite`, `test_load_lib_path_returns_handle_with_known_symbol`, `test_c_stdio`, `test_c_strings`, `test_c_strerror`, `test_c_memory`, `test_c_env` all pass; new `test_load_lib_with_handle_returns_queryable_handle` passes; flake8 clean.

- [ ] **Step 4: Commit**

```bash
cd /home/erik/projects/numbox
git add numbox/core/bindings/utils.py test/core/test_bindings.py
cat > /tmp/commit-msg.txt <<'EOF'
bindings: refactor load_lib via _resolve_lib_path + load_lib_with_handle; add Windows bundled-DLL fallback

Splits load_lib into three pieces:
- _resolve_lib_path(name): pure path resolver, returns None if not found
- load_lib(name): legacy contract — register globally, return None
- load_lib_with_handle(name): register AND return the CDLL handle so
  proxy_if_available / cres_if_available can query symbol presence

Adds _windows_bundled_dll_path(name) as a generic fallback for DLLs the
Python distribution ships in <base_prefix>/DLLs/ (CPython) or
<base_prefix>/Library/bin/ (conda) but doesn't put on PATH. Used by
_resolve_lib_path on Windows when find_library returns None — notably
needed for sqlite3.dll.

Prep for the SQLite buildout (signatures_sqlite expanding from 3 to 60).
EOF
git commit -F /tmp/commit-msg.txt
rm /tmp/commit-msg.txt
```

---

## Task 2: Add `_sqlite_constants.py` + expand `signatures_sqlite` to 60 entries

**Goal:** Create the constants module (`_sqlite_constants.py`) with all `SQLITE_*` numeric constants needed by the buildout, and expand `signatures_sqlite` in `signatures.py` from 3 entries to 60.

**Files:**
- Create: `numbox/core/bindings/_sqlite_constants.py`
- Modify: `numbox/core/bindings/signatures.py`

**Acceptance Criteria:**
- [ ] `signatures_sqlite` contains exactly 60 entries (verifiable via `len(signatures_sqlite) == 60`)
- [ ] All entries use numba types `int32`, `int64`, `float64`, `intp`, `void` (no `long`, no `time_t`, no `size_t`)
- [ ] `_sqlite_constants.py` exposes all listed constants as module-level `int` values
- [ ] `from numbox.core.bindings._sqlite_constants import *` works (no `__all__` restrictions; module-level int names are uppercase `SQLITE_*` so star-import pollution is intentional and scoped)

**Verify:**

```bash
cd /home/erik/projects/numbox
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/python -c "
from numbox.core.bindings.signatures import signatures_sqlite
assert len(signatures_sqlite) == 60, f'expected 60, got {len(signatures_sqlite)}'
expected = {
    'sqlite3_open', 'sqlite3_open_v2', 'sqlite3_close', 'sqlite3_libversion',
    'sqlite3_libversion_number', 'sqlite3_errmsg', 'sqlite3_errcode',
    'sqlite3_extended_errcode', 'sqlite3_threadsafe', 'sqlite3_db_handle',
    'sqlite3_db_filename', 'sqlite3_db_readonly', 'sqlite3_changes',
    'sqlite3_last_insert_rowid', 'sqlite3_total_changes',
    'sqlite3_changes64', 'sqlite3_total_changes64',
    'sqlite3_prepare_v2', 'sqlite3_finalize', 'sqlite3_reset', 'sqlite3_step',
    'sqlite3_sql', 'sqlite3_expanded_sql', 'sqlite3_stmt_busy',
    'sqlite3_bind_int', 'sqlite3_bind_int64', 'sqlite3_bind_double',
    'sqlite3_bind_text', 'sqlite3_bind_blob', 'sqlite3_bind_null',
    'sqlite3_bind_parameter_count', 'sqlite3_bind_parameter_index',
    'sqlite3_bind_parameter_name',
    'sqlite3_column_int', 'sqlite3_column_int64', 'sqlite3_column_double',
    'sqlite3_column_text', 'sqlite3_column_blob', 'sqlite3_column_bytes',
    'sqlite3_column_type', 'sqlite3_column_count', 'sqlite3_column_name',
    'sqlite3_column_decltype', 'sqlite3_column_database_name',
    'sqlite3_column_table_name', 'sqlite3_column_origin_name',
    'sqlite3_exec', 'sqlite3_free',
    'sqlite3_blob_open', 'sqlite3_blob_close', 'sqlite3_blob_bytes',
    'sqlite3_blob_read', 'sqlite3_blob_write', 'sqlite3_blob_reopen',
    'sqlite3_update_hook', 'sqlite3_progress_handler', 'sqlite3_busy_handler',
    'sqlite3_commit_hook', 'sqlite3_rollback_hook', 'sqlite3_trace_v2',
}
assert set(signatures_sqlite.keys()) == expected, set(signatures_sqlite.keys()) ^ expected
print('signatures_sqlite OK: 60 entries, names match')

from numbox.core.bindings._sqlite_constants import (
    SQLITE_OK, SQLITE_ROW, SQLITE_DONE, SQLITE_BUSY, SQLITE_LOCKED,
    SQLITE_INTEGER, SQLITE_FLOAT, SQLITE_TEXT, SQLITE_BLOB, SQLITE_NULL,
    SQLITE_OPEN_READONLY, SQLITE_OPEN_READWRITE, SQLITE_OPEN_CREATE,
    SQLITE_BLOB_READONLY, SQLITE_BLOB_READWRITE,
    SQLITE_TRACE_STMT, SQLITE_TRACE_PROFILE, SQLITE_TRACE_ROW, SQLITE_TRACE_CLOSE,
    SQLITE_STATIC, SQLITE_TRANSIENT,
)
assert SQLITE_OK == 0 and SQLITE_ROW == 100 and SQLITE_DONE == 101
assert SQLITE_INTEGER == 1 and SQLITE_NULL == 5
assert SQLITE_OPEN_READONLY == 0x1 and SQLITE_OPEN_CREATE == 0x4
assert SQLITE_BLOB_READONLY == 0 and SQLITE_BLOB_READWRITE == 1
assert SQLITE_TRACE_STMT == 0x1
assert SQLITE_STATIC == 0 and SQLITE_TRANSIENT == -1
print('SQLITE_* constants OK')
"
venv/bin/flake8 --max-line-length=127 numbox/core/bindings/signatures.py numbox/core/bindings/_sqlite_constants.py
```

Expected: both probe outputs print OK lines; flake8 clean.

**Steps:**

- [ ] **Step 1: Replace `signatures_sqlite` in `numbox/core/bindings/signatures.py`**

Read the current file (it's the one we examined: `signatures_c`, `signatures_m`, then `signatures_sqlite = {...3 entries...}`, then `signatures = {...}`). Replace `signatures_sqlite` with:

```python
signatures_sqlite = {
    # === Connection ===
    "sqlite3_open": int32(intp, intp),
    "sqlite3_open_v2": int32(intp, intp, int32, intp),
    "sqlite3_close": int32(intp),
    "sqlite3_libversion": intp(),
    "sqlite3_libversion_number": int32(),
    "sqlite3_errmsg": intp(intp),
    "sqlite3_errcode": int32(intp),
    "sqlite3_extended_errcode": int32(intp),
    "sqlite3_threadsafe": int32(),
    "sqlite3_db_handle": intp(intp),
    "sqlite3_db_filename": intp(intp, intp),
    "sqlite3_db_readonly": int32(intp, intp),
    "sqlite3_changes": int32(intp),
    "sqlite3_last_insert_rowid": int64(intp),
    "sqlite3_total_changes": int32(intp),
    # 64-bit row counts — SQLite 3.37+ (Nov 2021); guarded via proxy_if_available
    "sqlite3_changes64": int64(intp),
    "sqlite3_total_changes64": int64(intp),
    # === Statement lifecycle ===
    "sqlite3_prepare_v2": int32(intp, intp, int32, intp, intp),
    "sqlite3_finalize": int32(intp),
    "sqlite3_reset": int32(intp),
    "sqlite3_step": int32(intp),
    "sqlite3_sql": intp(intp),
    "sqlite3_expanded_sql": intp(intp),
    "sqlite3_stmt_busy": int32(intp),
    # === Parameter binding ===
    "sqlite3_bind_int": int32(intp, int32, int32),
    "sqlite3_bind_int64": int32(intp, int32, int64),
    "sqlite3_bind_double": int32(intp, int32, float64),
    "sqlite3_bind_text": int32(intp, int32, intp, int32, intp),
    "sqlite3_bind_blob": int32(intp, int32, intp, int32, intp),
    "sqlite3_bind_null": int32(intp, int32),
    "sqlite3_bind_parameter_count": int32(intp),
    "sqlite3_bind_parameter_index": int32(intp, intp),
    "sqlite3_bind_parameter_name": intp(intp, int32),
    # === Column accessors ===
    "sqlite3_column_int": int32(intp, int32),
    "sqlite3_column_int64": int64(intp, int32),
    "sqlite3_column_double": float64(intp, int32),
    "sqlite3_column_text": intp(intp, int32),
    "sqlite3_column_blob": intp(intp, int32),
    "sqlite3_column_bytes": int32(intp, int32),
    "sqlite3_column_type": int32(intp, int32),
    "sqlite3_column_count": int32(intp),
    "sqlite3_column_name": intp(intp, int32),
    "sqlite3_column_decltype": intp(intp, int32),
    # Compile-flag-gated (SQLITE_ENABLE_COLUMN_METADATA); via proxy_if_available
    "sqlite3_column_database_name": intp(intp, int32),
    "sqlite3_column_table_name": intp(intp, int32),
    "sqlite3_column_origin_name": intp(intp, int32),
    # === Exec + free ===
    "sqlite3_exec": int32(intp, intp, intp, intp, intp),
    "sqlite3_free": void(intp),
    # === BLOB incremental I/O ===
    "sqlite3_blob_open": int32(intp, intp, intp, intp, int64, int32, intp),
    "sqlite3_blob_close": int32(intp),
    "sqlite3_blob_bytes": int32(intp),
    "sqlite3_blob_read": int32(intp, intp, int32, int32),
    "sqlite3_blob_write": int32(intp, intp, int32, int32),
    "sqlite3_blob_reopen": int32(intp, int64),
    # === Callback hooks ===
    "sqlite3_update_hook": intp(intp, intp, intp),
    "sqlite3_progress_handler": void(intp, int32, intp, intp),
    "sqlite3_busy_handler": int32(intp, intp, intp),
    "sqlite3_commit_hook": intp(intp, intp, intp),
    "sqlite3_rollback_hook": intp(intp, intp, intp),
    "sqlite3_trace_v2": int32(intp, int32, intp, intp),
}
```

The `signatures = {**signatures_c, **signatures_m, **signatures_sqlite}` line at the bottom stays unchanged. The `from numba.core.types import ...` at the top already has `Tuple, float64, int32, int64, intp, void` — confirm by reading; if any of those are missing, add them.

- [ ] **Step 2: Create `numbox/core/bindings/_sqlite_constants.py`**

```python
"""SQLite numeric constants (result codes, type codes, open flags, blob flags,
trace flags, destructor sentinels).

Public surface — imported via star-import by ``numbox/core/bindings/__init__.py``.
All names are uppercase ``SQLITE_*`` to avoid collision with the lowercase
C-function-named wrappers.

Numba handles Python integer literals natively in ``@njit`` code, so these
constants are usable inside JITed functions without further wrapping. The
underlying SQLite values are API-stable across all matrix versions
(3.34.0 through current).
"""

# === Primary result codes (sqlite3.h) ===
SQLITE_OK = 0
SQLITE_ERROR = 1
SQLITE_INTERNAL = 2
SQLITE_PERM = 3
SQLITE_ABORT = 4
SQLITE_BUSY = 5
SQLITE_LOCKED = 6
SQLITE_NOMEM = 7
SQLITE_READONLY = 8
SQLITE_INTERRUPT = 9
SQLITE_IOERR = 10
SQLITE_CORRUPT = 11
SQLITE_NOTFOUND = 12
SQLITE_FULL = 13
SQLITE_CANTOPEN = 14
SQLITE_PROTOCOL = 15
SQLITE_EMPTY = 16
SQLITE_SCHEMA = 17
SQLITE_TOOBIG = 18
SQLITE_CONSTRAINT = 19
SQLITE_MISMATCH = 20
SQLITE_MISUSE = 21
SQLITE_NOLFS = 22
SQLITE_AUTH = 23
SQLITE_FORMAT = 24
SQLITE_RANGE = 25
SQLITE_NOTADB = 26
SQLITE_NOTICE = 27
SQLITE_WARNING = 28
SQLITE_ROW = 100
SQLITE_DONE = 101

# === Column type codes (sqlite3_column_type return values) ===
SQLITE_INTEGER = 1
SQLITE_FLOAT = 2
SQLITE_TEXT = 3
SQLITE_BLOB = 4
SQLITE_NULL = 5

# === sqlite3_open_v2 flags (combinable with bitwise OR) ===
SQLITE_OPEN_READONLY = 0x00000001
SQLITE_OPEN_READWRITE = 0x00000002
SQLITE_OPEN_CREATE = 0x00000004
SQLITE_OPEN_URI = 0x00000040
SQLITE_OPEN_MEMORY = 0x00000080
SQLITE_OPEN_NOMUTEX = 0x00008000
SQLITE_OPEN_FULLMUTEX = 0x00010000
SQLITE_OPEN_SHAREDCACHE = 0x00020000
SQLITE_OPEN_PRIVATECACHE = 0x00040000

# === sqlite3_blob_open flags (the integer values its `flags` arg accepts) ===
SQLITE_BLOB_READONLY = 0
SQLITE_BLOB_READWRITE = 1

# === sqlite3_trace_v2 event mask bits ===
SQLITE_TRACE_STMT = 0x01
SQLITE_TRACE_PROFILE = 0x02
SQLITE_TRACE_ROW = 0x04
SQLITE_TRACE_CLOSE = 0x08

# === Destructor sentinels for sqlite3_bind_text / sqlite3_bind_blob ===
SQLITE_STATIC = 0
SQLITE_TRANSIENT = -1
```

- [ ] **Step 3: Run the verify probe**

Use the command in **Verify** above. Both prints must succeed.

- [ ] **Step 4: Commit**

```bash
cd /home/erik/projects/numbox
git add numbox/core/bindings/signatures.py numbox/core/bindings/_sqlite_constants.py
cat > /tmp/commit-msg.txt <<'EOF'
sqlite: signatures + constants module

Expand signatures_sqlite from 3 entries to 60 covering connection
+ metadata, statement lifecycle, parameter binding, column accessors,
exec + free, BLOB incremental I/O, and six callback hooks.

Add _sqlite_constants.py with primary result codes (SQLITE_OK / ROW /
DONE / BUSY / ...), column type codes (SQLITE_INTEGER / FLOAT / TEXT /
BLOB / NULL), open flags (SQLITE_OPEN_*), BLOB open flags
(SQLITE_BLOB_READONLY / READWRITE), trace event mask
(SQLITE_TRACE_STMT / PROFILE / ROW / CLOSE), and destructor sentinels
(SQLITE_STATIC = 0, SQLITE_TRANSIENT = -1).

Prep for the per-group wrapper modules.
EOF
git commit -F /tmp/commit-msg.txt
rm /tmp/commit-msg.txt
```

---

## Task 3: Create `_sqlite_conn.py` + `test_sqlite_conn.py`

**Goal:** Bind the 17 connection-group SQLite functions (3 retained — `sqlite3_open`, `sqlite3_close`, `sqlite3_libversion`; 14 new — including 2 via `proxy_if_available`). Initialize `sqlite3_lib` as the module-level CDLL handle that subsequent files reuse for their own `proxy_if_available` calls. Move the existing `test_sqlite` out of `test/core/test_bindings.py` and into the new `test_sqlite_conn.py`, **without** the Windows skip.

**Files:**
- Create: `numbox/core/bindings/_sqlite_conn.py`
- Create: `test/core/test_sqlite_conn.py`
- Modify: `test/core/test_bindings.py` (remove the existing `test_sqlite`, since its replacement lives in the new file)

**Acceptance Criteria:**
- [ ] All 14 new connection wrappers callable from `@njit` code
- [ ] `sqlite3_libversion` (renamed from existing `sqlite3_libversion_number`) returns a `const char*` that decodes via `get_str_from_p_as_int` to a dotted string containing `"."`
- [ ] `sqlite3_libversion_number` (new wrapper, new signature) returns `int >= 3_000_000`
- [ ] `sqlite3_open_v2` with `SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE` on a `tmp_path` file returns `SQLITE_OK` and the out-param db pointer is non-zero
- [ ] `sqlite3_db_filename(db, "main"_p)` returns a path that matches the input filename
- [ ] `sqlite3_threadsafe()` returns `1` or `2` (modern SQLite is built with multi-thread or serialized mode)
- [ ] `sqlite3_changes64` (proxy_if_available) — test guard via `hasattr(..., "as_func")`; if present, returns int64
- [ ] All tests pass on Windows (no `skipif`)

**Verify:**

```bash
cd /home/erik/projects/numbox
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/pytest test/core/test_sqlite_conn.py -x --durations=20 -v
venv/bin/pytest test/core/test_bindings.py -x --durations=20 -v       # confirm nothing broken
venv/bin/flake8 --max-line-length=127 numbox/core/bindings/_sqlite_conn.py test/core/test_sqlite_conn.py
```

Expected: ~12 conn tests pass; existing test_bindings.py tests still pass (sans the moved `test_sqlite`); flake8 clean.

**Steps:**

- [ ] **Step 1: Create `numbox/core/bindings/_sqlite_conn.py`**

```python
"""SQLite connection + metadata bindings.

Initializes the module-level ``sqlite3_lib`` CDLL handle via
``load_lib_with_handle("sqlite3")``. Subsequent ``_sqlite_*.py`` modules
import ``sqlite3_lib`` from here for their own ``proxy_if_available`` calls,
ensuring a single load of the shared library across the suite.

Two functions are decorated with ``proxy_if_available``:
``sqlite3_changes64`` and ``sqlite3_total_changes64``, both added in
SQLite 3.37 (Nov 2021). On Python 3.10 (which ships SQLite 3.34), they
stub to ``NotImplementedError`` so callers can ``hasattr(...,"as_func")``
to decide whether to use them or fall back to the int32 variants.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib_with_handle
from numbox.core.proxy.proxy import proxy, proxy_if_available


sqlite3_lib = load_lib_with_handle("sqlite3")


@proxy(signatures.get("sqlite3_open"), jit_options={"cache": True})
def sqlite3_open(filename_p, db_pp):
    return _call_lib_func("sqlite3_open", (filename_p, db_pp))


@proxy(signatures.get("sqlite3_open_v2"), jit_options={"cache": True})
def sqlite3_open_v2(filename_p, db_pp, flags, vfs_p):
    return _call_lib_func("sqlite3_open_v2", (filename_p, db_pp, flags, vfs_p))


@proxy(signatures.get("sqlite3_close"), jit_options={"cache": True})
def sqlite3_close(db_p):
    return _call_lib_func("sqlite3_close", (db_p,))


@proxy(signatures.get("sqlite3_libversion"), jit_options={"cache": True})
def sqlite3_libversion():
    return _call_lib_func("sqlite3_libversion")


@proxy(signatures.get("sqlite3_libversion_number"), jit_options={"cache": True})
def sqlite3_libversion_number():
    return _call_lib_func("sqlite3_libversion_number")


@proxy(signatures.get("sqlite3_errmsg"), jit_options={"cache": True})
def sqlite3_errmsg(db_p):
    return _call_lib_func("sqlite3_errmsg", (db_p,))


@proxy(signatures.get("sqlite3_errcode"), jit_options={"cache": True})
def sqlite3_errcode(db_p):
    return _call_lib_func("sqlite3_errcode", (db_p,))


@proxy(signatures.get("sqlite3_extended_errcode"), jit_options={"cache": True})
def sqlite3_extended_errcode(db_p):
    return _call_lib_func("sqlite3_extended_errcode", (db_p,))


@proxy(signatures.get("sqlite3_threadsafe"), jit_options={"cache": True})
def sqlite3_threadsafe():
    return _call_lib_func("sqlite3_threadsafe")


@proxy(signatures.get("sqlite3_db_handle"), jit_options={"cache": True})
def sqlite3_db_handle(stmt_p):
    return _call_lib_func("sqlite3_db_handle", (stmt_p,))


@proxy(signatures.get("sqlite3_db_filename"), jit_options={"cache": True})
def sqlite3_db_filename(db_p, name_p):
    return _call_lib_func("sqlite3_db_filename", (db_p, name_p))


@proxy(signatures.get("sqlite3_db_readonly"), jit_options={"cache": True})
def sqlite3_db_readonly(db_p, name_p):
    return _call_lib_func("sqlite3_db_readonly", (db_p, name_p))


@proxy(signatures.get("sqlite3_changes"), jit_options={"cache": True})
def sqlite3_changes(db_p):
    return _call_lib_func("sqlite3_changes", (db_p,))


@proxy(signatures.get("sqlite3_last_insert_rowid"), jit_options={"cache": True})
def sqlite3_last_insert_rowid(db_p):
    return _call_lib_func("sqlite3_last_insert_rowid", (db_p,))


@proxy(signatures.get("sqlite3_total_changes"), jit_options={"cache": True})
def sqlite3_total_changes(db_p):
    return _call_lib_func("sqlite3_total_changes", (db_p,))


# SQLite 3.37+; stubbed via proxy_if_available on older library versions.
@proxy_if_available(sqlite3_lib, signatures.get("sqlite3_changes64"), jit_options={"cache": True})
def sqlite3_changes64(db_p):
    return _call_lib_func("sqlite3_changes64", (db_p,))


@proxy_if_available(sqlite3_lib, signatures.get("sqlite3_total_changes64"), jit_options={"cache": True})
def sqlite3_total_changes64(db_p):
    return _call_lib_func("sqlite3_total_changes64", (db_p,))
```

- [ ] **Step 2: Create `test/core/test_sqlite_conn.py`**

```python
"""Connection + metadata binding tests for the SQLite buildout."""
from ctypes import addressof, c_char_p, c_int64, c_void_p

import pytest

from numbox.core.bindings._sqlite_conn import (
    sqlite3_changes64,
    sqlite3_close,
    sqlite3_db_filename,
    sqlite3_db_readonly,
    sqlite3_errcode,
    sqlite3_errmsg,
    sqlite3_extended_errcode,
    sqlite3_libversion,
    sqlite3_libversion_number,
    sqlite3_open,
    sqlite3_open_v2,
    sqlite3_threadsafe,
    sqlite3_total_changes64,
)
from numbox.core.bindings._sqlite_constants import (
    SQLITE_CANTOPEN,
    SQLITE_OK,
    SQLITE_OPEN_CREATE,
    SQLITE_OPEN_READONLY,
    SQLITE_OPEN_READWRITE,
)
from numbox.utils.lowlevel import get_str_from_p_as_int
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


def _cstr(s):
    """Return (keepalive, intp address) for a Python str -> NUL-terminated C string."""
    buf = c_char_p(s.encode())
    return buf, c_void_p.from_buffer(buf).value


def _open_memory():
    """Open ':memory:' via sqlite3_open. Returns the db_p as an int."""
    _, name_p = _cstr(":memory:")
    db_p = c_int64(0)
    rc = sqlite3_open(name_p, addressof(db_p))
    assert rc == SQLITE_OK, f"sqlite3_open failed: rc={rc}"
    assert db_p.value != 0
    return db_p.value


def test_libversion_returns_dotted_string():
    version_p = sqlite3_libversion()
    version = str_from_p_as_int(version_p)
    assert "." in version, version


def test_libversion_number_returns_modern_int():
    n = sqlite3_libversion_number()
    # SQLite 3.0.0 = 3_000_000; any modern build is far above this.
    assert n >= 3_000_000, n


def test_open_close_memory_db():
    db_p = _open_memory()
    rc = sqlite3_close(db_p)
    assert rc == SQLITE_OK


def test_open_v2_with_create_flag(tmp_path):
    db_file = tmp_path / "create.sqlite"
    _, name_p = _cstr(str(db_file))
    db_p = c_int64(0)
    rc = sqlite3_open_v2(name_p, addressof(db_p),
                         SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0)
    assert rc == SQLITE_OK, rc
    assert db_p.value != 0
    assert sqlite3_close(db_p.value) == SQLITE_OK
    assert db_file.exists()


def test_open_v2_bad_path_returns_cantopen(tmp_path):
    bad_path = tmp_path / "nonexistent_dir" / "x.sqlite"
    _, name_p = _cstr(str(bad_path))
    db_p = c_int64(0)
    rc = sqlite3_open_v2(name_p, addressof(db_p), SQLITE_OPEN_READONLY, 0)
    assert rc == SQLITE_CANTOPEN, rc
    # Even on failure, SQLite returns a (possibly bare) connection handle that
    # owns the errmsg. Caller must close it.
    if db_p.value != 0:
        errmsg_p = sqlite3_errmsg(db_p.value)
        assert get_str_from_p_as_int(errmsg_p)  # non-empty
        sqlite3_close(db_p.value)


def test_db_filename_returns_main_path(tmp_path):
    db_file = tmp_path / "named.sqlite"
    _, name_p = _cstr(str(db_file))
    db_p = c_int64(0)
    sqlite3_open_v2(name_p, addressof(db_p),
                    SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0)
    _, main_p = _cstr("main")
    got_p = sqlite3_db_filename(db_p.value, main_p)
    got = str_from_p_as_int(got_p)
    assert got == str(db_file), (got, str(db_file))
    sqlite3_close(db_p.value)


def test_db_readonly_zero_for_writable(tmp_path):
    db_file = tmp_path / "rw.sqlite"
    _, name_p = _cstr(str(db_file))
    db_p = c_int64(0)
    sqlite3_open_v2(name_p, addressof(db_p),
                    SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0)
    _, main_p = _cstr("main")
    assert sqlite3_db_readonly(db_p.value, main_p) == 0
    sqlite3_close(db_p.value)


def test_threadsafe_returns_nonzero():
    # Modern SQLite is always built with at least multi-thread (1) or
    # serialized (2) mode. Single-thread (0) is essentially extinct.
    assert sqlite3_threadsafe() in (1, 2)


def test_errcode_matches_errmsg_after_bad_open(tmp_path):
    bad_path = tmp_path / "nonexistent_dir" / "x.sqlite"
    _, name_p = _cstr(str(bad_path))
    db_p = c_int64(0)
    rc = sqlite3_open_v2(name_p, addressof(db_p), SQLITE_OPEN_READONLY, 0)
    assert rc != SQLITE_OK
    if db_p.value != 0:
        assert sqlite3_errcode(db_p.value) == rc
        # extended_errcode may equal errcode or be a more specific code
        assert sqlite3_extended_errcode(db_p.value) != 0
        sqlite3_close(db_p.value)


def test_changes64_when_available():
    if not hasattr(sqlite3_changes64, "as_func"):
        pytest.skip("sqlite3_changes64 not available (SQLite < 3.37)")
    db_p = _open_memory()
    n = sqlite3_changes64(db_p)
    assert n == 0  # no statements executed yet
    sqlite3_close(db_p)


def test_total_changes64_when_available():
    if not hasattr(sqlite3_total_changes64, "as_func"):
        pytest.skip("sqlite3_total_changes64 not available (SQLite < 3.37)")
    db_p = _open_memory()
    n = sqlite3_total_changes64(db_p)
    assert n == 0
    sqlite3_close(db_p)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
```

- [ ] **Step 3: Remove `test_sqlite` from `test/core/test_bindings.py`**

Delete the existing `test_sqlite` function and the surrounding `@pytest.mark.skipif(platform_ == "Windows", reason="Need to add windows support")` decorator (lines 46-65 in the current file). The other tests in the file stay.

- [ ] **Step 4: Run tests + lint**

Run the **Verify** commands. All conn tests pass; old `test_bindings.py` still green for the rest.

- [ ] **Step 5: Commit**

```bash
cd /home/erik/projects/numbox
git add numbox/core/bindings/_sqlite_conn.py test/core/test_sqlite_conn.py test/core/test_bindings.py
cat > /tmp/commit-msg.txt <<'EOF'
sqlite: connection bindings (with changes64 / total_changes64 via proxy_if_available)

Add _sqlite_conn.py covering 17 connection + metadata bindings:
sqlite3_open (retained), open_v2, close (retained), libversion
(retained, renamed from misnamed libversion_number), libversion_number
(new, returns int32), errmsg, errcode, extended_errcode, threadsafe,
db_handle, db_filename, db_readonly, changes, last_insert_rowid,
total_changes, plus changes64 / total_changes64 via proxy_if_available
(stub to NotImplementedError when running against SQLite < 3.37).

Initialize module-level sqlite3_lib via load_lib_with_handle("sqlite3")
— other _sqlite_*.py modules import this handle for their own
proxy_if_available calls, ensuring a single load.

Move the existing test_sqlite from test_bindings.py into the new
test_sqlite_conn.py, drop the Windows skip (now covered by the
bundled-DLL fallback), and add coverage for open_v2 / errmsg / errcode
/ extended_errcode / db_filename / db_readonly / threadsafe /
changes64 / total_changes64.
EOF
git commit -F /tmp/commit-msg.txt
rm /tmp/commit-msg.txt
```

---

## Task 4: Create `_sqlite_stmt.py` + `test_sqlite_stmt.py`

**Goal:** Bind the 7 statement-lifecycle functions (`prepare_v2`, `finalize`, `reset`, `step`, `sql`, `expanded_sql`, `stmt_busy`) and write tests covering the prepare → step → done → finalize loop, the reset replay, sql/expanded_sql, and stmt_busy mid-iteration.

**Files:**
- Create: `numbox/core/bindings/_sqlite_stmt.py`
- Create: `test/core/test_sqlite_stmt.py`

**Acceptance Criteria:**
- [ ] All 7 stmt wrappers callable from `@njit`
- [ ] `sqlite3_prepare_v2` out-params (stmt_pp, tail_pp) work via `addressof(c_int64(0))` from Python
- [ ] `sqlite3_step(stmt)` returns `SQLITE_ROW` for non-empty SELECTs, `SQLITE_DONE` after rows exhausted
- [ ] `sqlite3_reset(stmt)` after step lets a subsequent step return `SQLITE_ROW` again
- [ ] `sqlite3_sql(stmt)` returns a `const char*` matching the originally-prepared SQL
- [ ] `sqlite3_expanded_sql(stmt)` returns a `char*` with bound parameters substituted; caller frees via `sqlite3_free` (verified by passing the pointer through `sqlite3_free` without crash)
- [ ] `sqlite3_stmt_busy(stmt)` returns non-zero after a step → ROW

**Verify:**

```bash
cd /home/erik/projects/numbox
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/pytest test/core/test_sqlite_stmt.py -x --durations=20 -v
venv/bin/flake8 --max-line-length=127 numbox/core/bindings/_sqlite_stmt.py test/core/test_sqlite_stmt.py
```

Expected: ~7 stmt tests pass; flake8 clean.

**Steps:**

- [ ] **Step 1: Create `numbox/core/bindings/_sqlite_stmt.py`**

```python
"""SQLite statement-lifecycle bindings: prepare_v2 / finalize / reset / step /
sql / expanded_sql / stmt_busy.

Note: ``sqlite3_expanded_sql`` returns a ``char *`` the caller MUST free via
``sqlite3_free`` (bound in _sqlite_exec.py). Document this with each call site
rather than building a wrapper that auto-frees — the wrapper would hide
ownership in a way the rest of the bindings don't.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.proxy.proxy import proxy


@proxy(signatures.get("sqlite3_prepare_v2"), jit_options={"cache": True})
def sqlite3_prepare_v2(db_p, sql_p, n_byte, stmt_pp, tail_pp):
    return _call_lib_func(
        "sqlite3_prepare_v2", (db_p, sql_p, n_byte, stmt_pp, tail_pp)
    )


@proxy(signatures.get("sqlite3_finalize"), jit_options={"cache": True})
def sqlite3_finalize(stmt_p):
    return _call_lib_func("sqlite3_finalize", (stmt_p,))


@proxy(signatures.get("sqlite3_reset"), jit_options={"cache": True})
def sqlite3_reset(stmt_p):
    return _call_lib_func("sqlite3_reset", (stmt_p,))


@proxy(signatures.get("sqlite3_step"), jit_options={"cache": True})
def sqlite3_step(stmt_p):
    return _call_lib_func("sqlite3_step", (stmt_p,))


@proxy(signatures.get("sqlite3_sql"), jit_options={"cache": True})
def sqlite3_sql(stmt_p):
    return _call_lib_func("sqlite3_sql", (stmt_p,))


@proxy(signatures.get("sqlite3_expanded_sql"), jit_options={"cache": True})
def sqlite3_expanded_sql(stmt_p):
    return _call_lib_func("sqlite3_expanded_sql", (stmt_p,))


@proxy(signatures.get("sqlite3_stmt_busy"), jit_options={"cache": True})
def sqlite3_stmt_busy(stmt_p):
    return _call_lib_func("sqlite3_stmt_busy", (stmt_p,))
```

- [ ] **Step 2: Create `test/core/test_sqlite_stmt.py`**

```python
"""Statement-lifecycle binding tests for the SQLite buildout."""
from ctypes import addressof, c_char_p, c_int64, c_void_p

import pytest

from numbox.core.bindings._sqlite_conn import (
    sqlite3_close,
    sqlite3_db_handle,
    sqlite3_open,
)
from numbox.core.bindings._sqlite_constants import (
    SQLITE_DONE,
    SQLITE_OK,
    SQLITE_ROW,
)
from numbox.core.bindings._sqlite_stmt import (
    sqlite3_expanded_sql,
    sqlite3_finalize,
    sqlite3_prepare_v2,
    sqlite3_reset,
    sqlite3_sql,
    sqlite3_step,
    sqlite3_stmt_busy,
)
from numbox.utils.lowlevel import get_str_from_p_as_int
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


def _cstr(s):
    buf = c_char_p(s.encode())
    return buf, c_void_p.from_buffer(buf).value


@pytest.fixture
def memory_db():
    """Open a fresh ':memory:' database via Python-side stdlib sqlite3 to
    populate test data, then return the path so test code can re-open via
    our bindings. Using sqlite3 (stdlib) for fixture setup keeps fixture
    failures distinguishable from binding failures."""
    import sqlite3 as stdlib_sqlite3
    # We can't share a stdlib connection with our raw bindings, so populate
    # via stdlib then close, and re-open the same in-memory db is not
    # possible (each :memory: is unique per connection). Instead, drive the
    # whole setup via raw bindings.
    _, name_p = _cstr(":memory:")
    db_p = c_int64(0)
    assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    yield db_p.value
    sqlite3_close(db_p.value)


def _prepare(db_p, sql):
    _, sql_p = _cstr(sql)
    stmt_p = c_int64(0)
    tail_p = c_int64(0)
    rc = sqlite3_prepare_v2(db_p, sql_p, -1, addressof(stmt_p), addressof(tail_p))
    assert rc == SQLITE_OK, f"prepare failed: rc={rc}"
    return stmt_p.value


def test_prepare_step_done_finalize_loop(memory_db):
    stmt_p = _prepare(memory_db, "SELECT 1")
    assert sqlite3_step(stmt_p) == SQLITE_ROW
    assert sqlite3_step(stmt_p) == SQLITE_DONE
    assert sqlite3_finalize(stmt_p) == SQLITE_OK


def test_reset_replays_query(memory_db):
    stmt_p = _prepare(memory_db, "SELECT 1")
    assert sqlite3_step(stmt_p) == SQLITE_ROW
    assert sqlite3_reset(stmt_p) == SQLITE_OK
    assert sqlite3_step(stmt_p) == SQLITE_ROW  # replays
    sqlite3_finalize(stmt_p)


def test_sql_returns_original_text(memory_db):
    original = "SELECT 1 AS one"
    stmt_p = _prepare(memory_db, original)
    sql_p = sqlite3_sql(stmt_p)
    assert str_from_p_as_int(sql_p) == original
    sqlite3_finalize(stmt_p)


def test_expanded_sql_substitutes_and_must_free(memory_db):
    # Need bind_int from a later task — defer until Task 5 is in.
    # For now, expanded_sql of a parameterless query just echoes the SQL.
    from numbox.core.bindings._sqlite_exec import sqlite3_free
    original = "SELECT 1"
    stmt_p = _prepare(memory_db, original)
    expanded_p = sqlite3_expanded_sql(stmt_p)
    assert expanded_p != 0
    assert str_from_p_as_int(expanded_p) == original
    sqlite3_free(expanded_p)
    sqlite3_finalize(stmt_p)


def test_stmt_busy_is_nonzero_mid_iteration(memory_db):
    stmt_p = _prepare(memory_db, "SELECT 1")
    assert sqlite3_stmt_busy(stmt_p) == 0  # before step
    sqlite3_step(stmt_p)                   # ROW
    assert sqlite3_stmt_busy(stmt_p) != 0  # mid-iteration
    sqlite3_finalize(stmt_p)


def test_prepare_invalid_sql_returns_error(memory_db):
    _, sql_p = _cstr("SELECT FROM WHERE")  # garbage
    stmt_p = c_int64(0)
    tail_p = c_int64(0)
    rc = sqlite3_prepare_v2(memory_db, sql_p, -1,
                            addressof(stmt_p), addressof(tail_p))
    assert rc != SQLITE_OK
    assert stmt_p.value == 0  # no statement on failure


def test_db_handle_round_trip(memory_db):
    stmt_p = _prepare(memory_db, "SELECT 1")
    assert sqlite3_db_handle(stmt_p) == memory_db
    sqlite3_finalize(stmt_p)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
```

**Important:** the `test_expanded_sql_substitutes_and_must_free` test imports `sqlite3_free` from `_sqlite_exec` — that module is not yet created (Task 9). If this test runs before Task 9 lands, it will fail with ImportError. **Mark this test with `pytest.importorskip` OR move the test body to a stub now and fill it in during Task 9.** Recommended: stub now, fill in Task 9 by removing the skip:

```python
def test_expanded_sql_substitutes_and_must_free(memory_db):
    pytest.importorskip("numbox.core.bindings._sqlite_exec")
    from numbox.core.bindings._sqlite_exec import sqlite3_free
    # ... rest unchanged ...
```

- [ ] **Step 3: Run tests + lint**

Verify command from **Verify** above.

- [ ] **Step 4: Commit**

```bash
cd /home/erik/projects/numbox
git add numbox/core/bindings/_sqlite_stmt.py test/core/test_sqlite_stmt.py
cat > /tmp/commit-msg.txt <<'EOF'
sqlite: statement lifecycle bindings

Add _sqlite_stmt.py with 7 bindings: prepare_v2, finalize, reset, step,
sql, expanded_sql, stmt_busy. The expanded_sql wrapper returns memory
the caller must free via sqlite3_free (bound in _sqlite_exec.py); this
ownership note is documented in CLAUDE.md alongside the existing
lifetime gotchas.

Tests cover the prepare -> step -> done -> finalize loop, reset replay,
sql / expanded_sql, stmt_busy mid-iteration, invalid-SQL error path,
and db_handle round trip.
EOF
git commit -F /tmp/commit-msg.txt
rm /tmp/commit-msg.txt
```

---

## Task 5: Create `_sqlite_bind.py` + `test_sqlite_bind.py`

**Goal:** Bind the 9 parameter-binding functions (`bind_int/int64/double/text/blob/null`, `bind_parameter_count/index/name`) and cover roundtrip via column accessors (which arrive in Task 8 — tests in this task that depend on column accessors are deferred or skipped via `importorskip`).

**Files:**
- Create: `numbox/core/bindings/_sqlite_bind.py`
- Create: `test/core/test_sqlite_bind.py`

**Acceptance Criteria:**
- [ ] All 9 bind wrappers callable from `@njit`
- [ ] `sqlite3_bind_int / int64 / double` accept the right C-int widths
- [ ] `sqlite3_bind_text(stmt, idx, text_p, n, SQLITE_TRANSIENT)` succeeds when `text_p = get_unicode_data_p("hello")` and `n = -1`
- [ ] `sqlite3_bind_blob(stmt, idx, data_p, n, SQLITE_TRANSIENT)` succeeds when `data_p = array_data_p(numpy_uint8_array)`
- [ ] `sqlite3_bind_null(stmt, idx)` succeeds
- [ ] `sqlite3_bind_parameter_count(stmt)` for `SELECT ?1, ?2, ?3` returns 3
- [ ] `sqlite3_bind_parameter_index(stmt, ":foo"_p)` for `SELECT :foo` returns 1
- [ ] `sqlite3_bind_parameter_name(stmt, 1)` returns a `const char*` matching the named parameter

**Verify:**

```bash
cd /home/erik/projects/numbox
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/pytest test/core/test_sqlite_bind.py -x --durations=20 -v
venv/bin/flake8 --max-line-length=127 numbox/core/bindings/_sqlite_bind.py test/core/test_sqlite_bind.py
```

Expected: ~9 bind tests pass (some may xfail-pending until column accessors land); flake8 clean.

**Steps:**

- [ ] **Step 1: Create `numbox/core/bindings/_sqlite_bind.py`**

```python
"""SQLite parameter-binding bindings.

The destructor arg in bind_text / bind_blob (last intp) is one of:
- SQLITE_STATIC = 0  -> SQLite assumes the buffer outlives the statement
- SQLITE_TRANSIENT = -1 -> SQLite makes a copy
- any other value -> a C function pointer SQLite calls to free the buffer

For numpy arrays passed via array_data_p, prefer SQLITE_TRANSIENT unless the
caller can guarantee the array outlives the prepared statement.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.proxy.proxy import proxy


@proxy(signatures.get("sqlite3_bind_int"), jit_options={"cache": True})
def sqlite3_bind_int(stmt_p, idx, val):
    return _call_lib_func("sqlite3_bind_int", (stmt_p, idx, val))


@proxy(signatures.get("sqlite3_bind_int64"), jit_options={"cache": True})
def sqlite3_bind_int64(stmt_p, idx, val):
    return _call_lib_func("sqlite3_bind_int64", (stmt_p, idx, val))


@proxy(signatures.get("sqlite3_bind_double"), jit_options={"cache": True})
def sqlite3_bind_double(stmt_p, idx, val):
    return _call_lib_func("sqlite3_bind_double", (stmt_p, idx, val))


@proxy(signatures.get("sqlite3_bind_text"), jit_options={"cache": True})
def sqlite3_bind_text(stmt_p, idx, text_p, n, destructor):
    return _call_lib_func(
        "sqlite3_bind_text", (stmt_p, idx, text_p, n, destructor)
    )


@proxy(signatures.get("sqlite3_bind_blob"), jit_options={"cache": True})
def sqlite3_bind_blob(stmt_p, idx, data_p, n, destructor):
    return _call_lib_func(
        "sqlite3_bind_blob", (stmt_p, idx, data_p, n, destructor)
    )


@proxy(signatures.get("sqlite3_bind_null"), jit_options={"cache": True})
def sqlite3_bind_null(stmt_p, idx):
    return _call_lib_func("sqlite3_bind_null", (stmt_p, idx))


@proxy(signatures.get("sqlite3_bind_parameter_count"), jit_options={"cache": True})
def sqlite3_bind_parameter_count(stmt_p):
    return _call_lib_func("sqlite3_bind_parameter_count", (stmt_p,))


@proxy(signatures.get("sqlite3_bind_parameter_index"), jit_options={"cache": True})
def sqlite3_bind_parameter_index(stmt_p, name_p):
    return _call_lib_func("sqlite3_bind_parameter_index", (stmt_p, name_p))


@proxy(signatures.get("sqlite3_bind_parameter_name"), jit_options={"cache": True})
def sqlite3_bind_parameter_name(stmt_p, idx):
    return _call_lib_func("sqlite3_bind_parameter_name", (stmt_p, idx))
```

- [ ] **Step 2: Create `test/core/test_sqlite_bind.py`**

```python
"""Parameter-binding tests for the SQLite buildout."""
from ctypes import addressof, c_char_p, c_int64, c_void_p

import numpy as np
import pytest

from numbox.core.bindings._sqlite_bind import (
    sqlite3_bind_blob,
    sqlite3_bind_double,
    sqlite3_bind_int,
    sqlite3_bind_int64,
    sqlite3_bind_null,
    sqlite3_bind_parameter_count,
    sqlite3_bind_parameter_index,
    sqlite3_bind_parameter_name,
    sqlite3_bind_text,
)
from numbox.core.bindings._sqlite_conn import sqlite3_close, sqlite3_open
from numbox.core.bindings._sqlite_constants import (
    SQLITE_OK,
    SQLITE_RANGE,
    SQLITE_TRANSIENT,
)
from numbox.core.bindings._sqlite_stmt import (
    sqlite3_finalize,
    sqlite3_prepare_v2,
)
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


def _cstr(s):
    buf = c_char_p(s.encode())
    return buf, c_void_p.from_buffer(buf).value


@pytest.fixture
def stmt_three_params():
    """Open :memory: db, prepare 'SELECT ?1, ?2, ?3', yield (db_p, stmt_p),
    teardown finalizes + closes."""
    _, name_p = _cstr(":memory:")
    db_p = c_int64(0)
    assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK

    _, sql_p = _cstr("SELECT ?1, ?2, ?3")
    stmt_p = c_int64(0)
    tail_p = c_int64(0)
    assert sqlite3_prepare_v2(db_p.value, sql_p, -1,
                              addressof(stmt_p), addressof(tail_p)) == SQLITE_OK
    yield db_p.value, stmt_p.value
    sqlite3_finalize(stmt_p.value)
    sqlite3_close(db_p.value)


def test_bind_int_returns_ok(stmt_three_params):
    _, stmt_p = stmt_three_params
    assert sqlite3_bind_int(stmt_p, 1, 42) == SQLITE_OK


def test_bind_int64_returns_ok(stmt_three_params):
    _, stmt_p = stmt_three_params
    assert sqlite3_bind_int64(stmt_p, 2, 2**40) == SQLITE_OK


def test_bind_double_returns_ok(stmt_three_params):
    _, stmt_p = stmt_three_params
    assert sqlite3_bind_double(stmt_p, 3, 3.14) == SQLITE_OK


def test_bind_text_transient_returns_ok(stmt_three_params):
    _, stmt_p = stmt_three_params
    _, text_p = _cstr("hello")
    assert sqlite3_bind_text(stmt_p, 1, text_p, -1, SQLITE_TRANSIENT) == SQLITE_OK


def test_bind_blob_with_numpy_uint8_returns_ok(stmt_three_params):
    _, stmt_p = stmt_three_params
    buf = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    data_p = buf.ctypes.data
    assert sqlite3_bind_blob(stmt_p, 1, data_p, buf.nbytes, SQLITE_TRANSIENT) == SQLITE_OK


def test_bind_null_returns_ok(stmt_three_params):
    _, stmt_p = stmt_three_params
    assert sqlite3_bind_null(stmt_p, 1) == SQLITE_OK


def test_bind_parameter_count_returns_three(stmt_three_params):
    _, stmt_p = stmt_three_params
    assert sqlite3_bind_parameter_count(stmt_p) == 3


def test_bind_out_of_range_returns_sqlite_range(stmt_three_params):
    _, stmt_p = stmt_three_params
    rc = sqlite3_bind_int(stmt_p, 99, 0)
    assert rc == SQLITE_RANGE


def test_bind_parameter_index_by_name():
    _, name_p = _cstr(":memory:")
    db_p = c_int64(0)
    assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    try:
        _, sql_p = _cstr("SELECT :foo, :bar")
        stmt_p = c_int64(0)
        tail_p = c_int64(0)
        sqlite3_prepare_v2(db_p.value, sql_p, -1,
                           addressof(stmt_p), addressof(tail_p))
        _, foo_p = _cstr(":foo")
        _, bar_p = _cstr(":bar")
        assert sqlite3_bind_parameter_index(stmt_p.value, foo_p) == 1
        assert sqlite3_bind_parameter_index(stmt_p.value, bar_p) == 2
        # name lookup round trip
        name_back_p = sqlite3_bind_parameter_name(stmt_p.value, 1)
        assert str_from_p_as_int(name_back_p) == ":foo"
        sqlite3_finalize(stmt_p.value)
    finally:
        sqlite3_close(db_p.value)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
```

- [ ] **Step 3: Run tests + lint**

Use the **Verify** command.

- [ ] **Step 4: Commit**

```bash
cd /home/erik/projects/numbox
git add numbox/core/bindings/_sqlite_bind.py test/core/test_sqlite_bind.py
cat > /tmp/commit-msg.txt <<'EOF'
sqlite: parameter binding

Add _sqlite_bind.py with 9 bindings: bind_int / int64 / double / text /
blob / null / parameter_count / parameter_index / parameter_name.

For bind_text and bind_blob the destructor arg is SQLITE_STATIC (= 0,
buffer must outlive the statement) or SQLITE_TRANSIENT (= -1, SQLite
copies). Tests use SQLITE_TRANSIENT for the safe default.

Tests cover scalar binds, text via get_unicode_data_p, blob via numpy
uint8 array data pointer, null binding, parameter-count introspection,
out-of-range -> SQLITE_RANGE error path, and named-parameter
index / name round trip.
EOF
git commit -F /tmp/commit-msg.txt
rm /tmp/commit-msg.txt
```

---

## Task 6: Create `_sqlite_blob.py` + `test_sqlite_blob.py`

**Goal:** Bind the 6 BLOB incremental I/O functions (`blob_open`, `_close`, `_bytes`, `_read`, `_write`, `_reopen`). Test against a `t(b BLOB)` table populated via `sqlite3_exec` — note this requires Task 9 to land first OR fixtures using stdlib `sqlite3` to populate.

**Files:**
- Create: `numbox/core/bindings/_sqlite_blob.py`
- Create: `test/core/test_sqlite_blob.py`

**Acceptance Criteria:**
- [ ] All 6 blob wrappers callable from `@njit`
- [ ] `sqlite3_blob_open(db, dbname_p, table_p, col_p, rowid, flags, blob_pp)` returns `SQLITE_OK` and writes a non-zero blob handle
- [ ] `sqlite3_blob_read(blob_p, buf_p, n, offset)` reads expected bytes from a known BLOB
- [ ] `sqlite3_blob_write(blob_p, buf_p, n, offset)` writes to a `READWRITE` blob; subsequent read returns the new bytes
- [ ] `sqlite3_blob_bytes(blob_p)` returns the blob's declared length
- [ ] `sqlite3_blob_reopen(blob_p, new_rowid)` retargets the blob to a different row
- [ ] `sqlite3_blob_close(blob_p)` returns `SQLITE_OK`
- [ ] Opening a bad column returns a non-OK code (test error path)

**Verify:**

```bash
cd /home/erik/projects/numbox
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/pytest test/core/test_sqlite_blob.py -x --durations=20 -v
venv/bin/flake8 --max-line-length=127 numbox/core/bindings/_sqlite_blob.py test/core/test_sqlite_blob.py
```

Expected: ~6 blob tests pass; flake8 clean.

**Steps:**

- [ ] **Step 1: Create `numbox/core/bindings/_sqlite_blob.py`**

```python
"""SQLite BLOB incremental I/O bindings: blob_open / _close / _bytes / _read /
_write / _reopen.

All functions present in SQLite 3.4.0 (2007) except _reopen which arrived in
3.7.4 (2010). No version gating needed — far below the matrix floor of 3.34.
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.proxy.proxy import proxy


@proxy(signatures.get("sqlite3_blob_open"), jit_options={"cache": True})
def sqlite3_blob_open(db_p, db_name_p, table_p, col_p, rowid, flags, blob_pp):
    return _call_lib_func(
        "sqlite3_blob_open",
        (db_p, db_name_p, table_p, col_p, rowid, flags, blob_pp),
    )


@proxy(signatures.get("sqlite3_blob_close"), jit_options={"cache": True})
def sqlite3_blob_close(blob_p):
    return _call_lib_func("sqlite3_blob_close", (blob_p,))


@proxy(signatures.get("sqlite3_blob_bytes"), jit_options={"cache": True})
def sqlite3_blob_bytes(blob_p):
    return _call_lib_func("sqlite3_blob_bytes", (blob_p,))


@proxy(signatures.get("sqlite3_blob_read"), jit_options={"cache": True})
def sqlite3_blob_read(blob_p, buf_p, n, offset):
    return _call_lib_func("sqlite3_blob_read", (blob_p, buf_p, n, offset))


@proxy(signatures.get("sqlite3_blob_write"), jit_options={"cache": True})
def sqlite3_blob_write(blob_p, buf_p, n, offset):
    return _call_lib_func("sqlite3_blob_write", (blob_p, buf_p, n, offset))


@proxy(signatures.get("sqlite3_blob_reopen"), jit_options={"cache": True})
def sqlite3_blob_reopen(blob_p, new_rowid):
    return _call_lib_func("sqlite3_blob_reopen", (blob_p, new_rowid))
```

- [ ] **Step 2: Create `test/core/test_sqlite_blob.py`**

```python
"""BLOB incremental I/O tests for the SQLite buildout."""
from ctypes import addressof, c_char_p, c_int64, c_void_p

import numpy as np
import pytest

from numbox.core.bindings._sqlite_blob import (
    sqlite3_blob_bytes,
    sqlite3_blob_close,
    sqlite3_blob_open,
    sqlite3_blob_read,
    sqlite3_blob_reopen,
    sqlite3_blob_write,
)
from numbox.core.bindings._sqlite_conn import sqlite3_close, sqlite3_open
from numbox.core.bindings._sqlite_constants import (
    SQLITE_BLOB_READONLY,
    SQLITE_BLOB_READWRITE,
    SQLITE_OK,
)


def _cstr(s):
    buf = c_char_p(s.encode())
    return buf, c_void_p.from_buffer(buf).value


@pytest.fixture
def populated_db(tmp_path):
    """Create a file-backed db (not :memory:, since stdlib + our bindings
    can't share an in-memory connection) with t(b BLOB) containing one row
    of known bytes at rowid 1."""
    import sqlite3 as stdlib_sqlite3
    db_file = tmp_path / "blob.sqlite"
    conn = stdlib_sqlite3.connect(str(db_file))
    conn.executescript(
        "CREATE TABLE t(b BLOB);"
        "INSERT INTO t(rowid, b) VALUES (1, x'01020304050607');"
        "INSERT INTO t(rowid, b) VALUES (2, x'AABBCCDD');"
    )
    conn.commit()
    conn.close()
    _, name_p = _cstr(str(db_file))
    db_p = c_int64(0)
    assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    yield db_p.value
    sqlite3_close(db_p.value)


def _blob_open(db_p, rowid, flags):
    _, main_p = _cstr("main")
    _, table_p = _cstr("t")
    _, col_p = _cstr("b")
    blob_p = c_int64(0)
    rc = sqlite3_blob_open(db_p, main_p, table_p, col_p, rowid, flags,
                           addressof(blob_p))
    return rc, blob_p.value


def test_blob_open_read_known_bytes(populated_db):
    rc, blob = _blob_open(populated_db, 1, SQLITE_BLOB_READONLY)
    assert rc == SQLITE_OK
    assert blob != 0
    n = sqlite3_blob_bytes(blob)
    assert n == 7
    buf = np.zeros(n, dtype=np.uint8)
    assert sqlite3_blob_read(blob, buf.ctypes.data, n, 0) == SQLITE_OK
    assert bytes(buf) == bytes(range(1, 8))
    assert sqlite3_blob_close(blob) == SQLITE_OK


def test_blob_write_then_reread(populated_db):
    rc, blob = _blob_open(populated_db, 1, SQLITE_BLOB_READWRITE)
    assert rc == SQLITE_OK
    new_bytes = np.array([0xFE] * 7, dtype=np.uint8)
    assert sqlite3_blob_write(blob, new_bytes.ctypes.data, 7, 0) == SQLITE_OK
    sqlite3_blob_close(blob)

    rc, blob = _blob_open(populated_db, 1, SQLITE_BLOB_READONLY)
    assert rc == SQLITE_OK
    got = np.zeros(7, dtype=np.uint8)
    sqlite3_blob_read(blob, got.ctypes.data, 7, 0)
    assert bytes(got) == bytes([0xFE] * 7)
    sqlite3_blob_close(blob)


def test_blob_bytes_matches_inserted(populated_db):
    rc, blob = _blob_open(populated_db, 2, SQLITE_BLOB_READONLY)
    assert rc == SQLITE_OK
    assert sqlite3_blob_bytes(blob) == 4
    sqlite3_blob_close(blob)


def test_blob_reopen_to_different_rowid(populated_db):
    rc, blob = _blob_open(populated_db, 1, SQLITE_BLOB_READONLY)
    assert rc == SQLITE_OK
    assert sqlite3_blob_bytes(blob) == 7
    assert sqlite3_blob_reopen(blob, 2) == SQLITE_OK
    assert sqlite3_blob_bytes(blob) == 4
    sqlite3_blob_close(blob)


def test_blob_open_bad_column_returns_error(populated_db):
    _, main_p = _cstr("main")
    _, table_p = _cstr("t")
    _, bad_col_p = _cstr("nonexistent_column")
    blob_p = c_int64(0)
    rc = sqlite3_blob_open(populated_db, main_p, table_p, bad_col_p,
                           1, SQLITE_BLOB_READONLY, addressof(blob_p))
    assert rc != SQLITE_OK


def test_blob_close_returns_ok(populated_db):
    rc, blob = _blob_open(populated_db, 1, SQLITE_BLOB_READONLY)
    assert rc == SQLITE_OK
    assert sqlite3_blob_close(blob) == SQLITE_OK


if __name__ == "__main__":
    from test.auxiliary_utils import collect_and_run_tests
    collect_and_run_tests(__name__)
```

- [ ] **Step 3: Run tests + lint**

Use the **Verify** command.

- [ ] **Step 4: Commit**

```bash
cd /home/erik/projects/numbox
git add numbox/core/bindings/_sqlite_blob.py test/core/test_sqlite_blob.py
cat > /tmp/commit-msg.txt <<'EOF'
sqlite: BLOB incremental I/O bindings

Add _sqlite_blob.py with 6 bindings: blob_open, _close, _bytes, _read,
_write, _reopen. All functions are SQLite 3.4 / 3.7 vintage so no
version gating needed.

Tests use a file-backed database (file path via tmp_path) populated by
stdlib sqlite3 — a :memory: db can't be shared across stdlib and our
raw bindings since each :memory: connection is its own database.
Covers read of known bytes, write+reread, bytes/length round trip,
reopen to different rowid, bad-column error path, and close.
EOF
git commit -F /tmp/commit-msg.txt
rm /tmp/commit-msg.txt
```

---

## Task 7: Create `_sqlite_hooks.py` + `test_sqlite_hooks.py`

**Goal:** Bind the 6 callback-hook APIs (`update_hook`, `progress_handler`, `busy_handler`, `commit_hook`, `rollback_hook`, `trace_v2`). Test each one by registering a `@cfunc` from Python, exercising the trigger condition, and reading captured state from a numpy array passed as `ctx`.

**Files:**
- Create: `numbox/core/bindings/_sqlite_hooks.py`
- Create: `test/core/test_sqlite_hooks.py`

**Acceptance Criteria:**
- [ ] All 6 hook wrappers callable from `@njit`
- [ ] `update_hook` fires on INSERT/UPDATE/DELETE with correct op codes (18=INSERT, 23=UPDATE, 9=DELETE per `SQLITE_INSERT`/`UPDATE`/`DELETE`)
- [ ] `progress_handler` fires after N opcodes and can abort by returning nonzero
- [ ] `busy_handler` can retry (return nonzero) or abort (return 0)
- [ ] `commit_hook` can veto a commit by returning nonzero
- [ ] `rollback_hook` fires on rollback
- [ ] `trace_v2` fires for `SQLITE_TRACE_STMT` events

**Verify:**

```bash
cd /home/erik/projects/numbox
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/pytest test/core/test_sqlite_hooks.py -x --durations=20 -v
venv/bin/flake8 --max-line-length=127 numbox/core/bindings/_sqlite_hooks.py test/core/test_sqlite_hooks.py
```

Expected: ~6 hook tests pass; flake8 clean.

**Steps:**

- [ ] **Step 1: Create `numbox/core/bindings/_sqlite_hooks.py`**

```python
"""SQLite callback hooks: update_hook / progress_handler / busy_handler /
commit_hook / rollback_hook / trace_v2.

Each takes a function pointer (intp) the caller produces via @cfunc(...).address.
The cfunc instance MUST outlive the hook registration — keep it at module scope
in the caller.

Callback shapes (informational; signatures are caller's responsibility):
- update_hook:       void(void*, int op, const char* db, const char* tbl, sqlite3_int64 rowid)
- progress_handler:  int(void*) -- nonzero aborts
- busy_handler:      int(void*, int) -- 0 to abort, nonzero to retry
- commit_hook:       int(void*) -- nonzero vetoes commit
- rollback_hook:     void(void*)
- trace_v2:          int(unsigned, void*, void*, void*)
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.proxy.proxy import proxy


@proxy(signatures.get("sqlite3_update_hook"), jit_options={"cache": True})
def sqlite3_update_hook(db_p, cb_p, ctx_p):
    return _call_lib_func("sqlite3_update_hook", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_progress_handler"), jit_options={"cache": True})
def sqlite3_progress_handler(db_p, n_ops, cb_p, ctx_p):
    return _call_lib_func("sqlite3_progress_handler", (db_p, n_ops, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_busy_handler"), jit_options={"cache": True})
def sqlite3_busy_handler(db_p, cb_p, ctx_p):
    return _call_lib_func("sqlite3_busy_handler", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_commit_hook"), jit_options={"cache": True})
def sqlite3_commit_hook(db_p, cb_p, ctx_p):
    return _call_lib_func("sqlite3_commit_hook", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_rollback_hook"), jit_options={"cache": True})
def sqlite3_rollback_hook(db_p, cb_p, ctx_p):
    return _call_lib_func("sqlite3_rollback_hook", (db_p, cb_p, ctx_p))


@proxy(signatures.get("sqlite3_trace_v2"), jit_options={"cache": True})
def sqlite3_trace_v2(db_p, mask, cb_p, ctx_p):
    return _call_lib_func("sqlite3_trace_v2", (db_p, mask, cb_p, ctx_p))
```

- [ ] **Step 2: Create `test/core/test_sqlite_hooks.py`**

```python
"""Callback hook tests for the SQLite buildout.

Uses numpy int64 arrays as ctx — passing `arr.ctypes.data` as the ctx pointer
lets @cfunc callbacks read/write the array's bytes directly. This is the
canonical pattern for capturing state across the C->Python boundary without
ctypes-level Python object refs.
"""
from ctypes import addressof, c_char_p, c_int64, c_void_p

import numpy as np
import pytest
from numba import cfunc, types

from numbox.core.bindings._sqlite_conn import sqlite3_close, sqlite3_open
from numbox.core.bindings._sqlite_constants import (
    SQLITE_INTERRUPT,
    SQLITE_OK,
    SQLITE_TRACE_STMT,
)
from numbox.core.bindings._sqlite_hooks import (
    sqlite3_busy_handler,
    sqlite3_commit_hook,
    sqlite3_progress_handler,
    sqlite3_rollback_hook,
    sqlite3_trace_v2,
    sqlite3_update_hook,
)


def _cstr(s):
    buf = c_char_p(s.encode())
    return buf, c_void_p.from_buffer(buf).value


# Module-level cfuncs — must outlive any hook registration.

@cfunc(types.void(types.voidptr, types.int32, types.intp, types.intp, types.int64))
def _update_cb(ctx, op, db_p, tbl_p, rowid):
    """Write op into ctx[0] (rolling: ctx[idx]=op where idx = ctx[64]++)."""
    counter_p = ctx + 64 * 8  # bytes offset to counter slot
    # We can't do array_data_p in a cfunc — write via ctypes-style intp math.
    # Hack: use the carray intrinsic.
    from numba import carray
    arr = carray(ctx, 128, dtype=np.int64)
    idx = arr[64]
    arr[idx] = op
    arr[64] = idx + 1


@cfunc(types.int32(types.voidptr))
def _progress_abort_cb(ctx):
    return 1  # nonzero -> abort


@cfunc(types.int32(types.voidptr, types.int32))
def _busy_abort_cb(ctx, n):
    return 0  # zero -> abort (no retry)


@cfunc(types.int32(types.voidptr))
def _commit_veto_cb(ctx):
    return 1  # nonzero -> veto


@cfunc(types.void(types.voidptr))
def _rollback_count_cb(ctx):
    from numba import carray
    arr = carray(ctx, 1, dtype=np.int64)
    arr[0] += 1


@cfunc(types.int32(types.uint32, types.voidptr, types.voidptr, types.voidptr))
def _trace_count_cb(mask, ctx, p, x):
    from numba import carray
    arr = carray(ctx, 1, dtype=np.int64)
    arr[0] += 1
    return 0


@pytest.fixture
def populated_db(tmp_path):
    import sqlite3 as stdlib_sqlite3
    db_file = tmp_path / "hooks.sqlite"
    conn = stdlib_sqlite3.connect(str(db_file))
    conn.executescript(
        "CREATE TABLE t(a INTEGER);"
        "INSERT INTO t VALUES (1), (2), (3);"
    )
    conn.commit()
    conn.close()
    _, name_p = _cstr(str(db_file))
    db_p = c_int64(0)
    assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    yield db_p.value
    sqlite3_close(db_p.value)


def test_update_hook_records_ops(populated_db):
    # ctx layout: arr[0..63] = op log, arr[64] = next-write index
    ctx = np.zeros(128, dtype=np.int64)
    sqlite3_update_hook(populated_db, _update_cb.address, ctx.ctypes.data)
    # Use exec from Task 9; for now go through stdlib to fire INSERT.
    # Since sqlite3_exec arrives in Task 9, defer the actual trigger to that
    # task's tests OR drive via the prepare/step path now.
    pytest.importorskip("numbox.core.bindings._sqlite_exec")
    from numbox.core.bindings._sqlite_exec import sqlite3_exec
    _, sql_p = _cstr("INSERT INTO t VALUES (99); DELETE FROM t WHERE a=99;")
    rc = sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    # SQLITE_INSERT = 18, SQLITE_DELETE = 9
    assert ctx[64] == 2  # two callbacks fired
    assert ctx[0] == 18
    assert ctx[1] == 9


def test_progress_handler_aborts(populated_db):
    pytest.importorskip("numbox.core.bindings._sqlite_exec")
    from numbox.core.bindings._sqlite_exec import sqlite3_exec
    sqlite3_progress_handler(populated_db, 1, _progress_abort_cb.address, 0)
    _, sql_p = _cstr("SELECT * FROM t")
    rc = sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    assert rc == SQLITE_INTERRUPT


def test_busy_handler_registration_returns_ok(populated_db):
    # Triggering busy-condition reliably is hard in a unit test (needs two
    # connections / file locking). Verify only that registration succeeds.
    rc = sqlite3_busy_handler(populated_db, _busy_abort_cb.address, 0)
    assert rc == SQLITE_OK


def test_commit_hook_vetoes(populated_db):
    pytest.importorskip("numbox.core.bindings._sqlite_exec")
    from numbox.core.bindings._sqlite_exec import sqlite3_exec
    sqlite3_commit_hook(populated_db, _commit_veto_cb.address, 0)
    _, sql_p = _cstr("INSERT INTO t VALUES (42)")
    rc = sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    # Vetoed commit -> rolled back; sqlite3_exec returns non-OK (typically
    # SQLITE_CONSTRAINT_TRIGGER or SQLITE_CONSTRAINT in some versions).
    assert rc != SQLITE_OK


def test_rollback_hook_fires(populated_db):
    pytest.importorskip("numbox.core.bindings._sqlite_exec")
    from numbox.core.bindings._sqlite_exec import sqlite3_exec
    ctx = np.zeros(1, dtype=np.int64)
    sqlite3_rollback_hook(populated_db, _rollback_count_cb.address, ctx.ctypes.data)
    _, sql_p = _cstr("BEGIN; INSERT INTO t VALUES (77); ROLLBACK;")
    rc = sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    assert rc == SQLITE_OK
    assert ctx[0] == 1


def test_trace_v2_fires_for_stmt(populated_db):
    pytest.importorskip("numbox.core.bindings._sqlite_exec")
    from numbox.core.bindings._sqlite_exec import sqlite3_exec
    ctx = np.zeros(1, dtype=np.int64)
    rc = sqlite3_trace_v2(populated_db, SQLITE_TRACE_STMT,
                          _trace_count_cb.address, ctx.ctypes.data)
    assert rc == SQLITE_OK
    _, sql_p = _cstr("SELECT 1")
    sqlite3_exec(populated_db, sql_p, 0, 0, 0)
    assert ctx[0] >= 1


if __name__ == "__main__":
    from test.auxiliary_utils import collect_and_run_tests
    collect_and_run_tests(__name__)
```

- [ ] **Step 3: Run tests + lint**

Use the **Verify** command. Tests that depend on `sqlite3_exec` (Task 9) will skip via `pytest.importorskip` until Task 9 lands; they all turn into PASS once Task 9 is committed.

- [ ] **Step 4: Commit**

```bash
cd /home/erik/projects/numbox
git add numbox/core/bindings/_sqlite_hooks.py test/core/test_sqlite_hooks.py
cat > /tmp/commit-msg.txt <<'EOF'
sqlite: callback hooks (update / progress / busy / commit / rollback / trace_v2)

Add _sqlite_hooks.py with 6 hook-registration wrappers. Each takes a
function pointer (intp) that the caller produces via @cfunc(...).address.
The cfunc instance must outlive the hook registration — keep at module
scope in the caller.

Tests register module-scope cfuncs and verify hook firing via numpy
ctx arrays (carray inside @cfunc reads/writes the bytes pointed to by
ctx). update_hook records op codes, progress_handler aborts, busy_handler
registers cleanly, commit_hook vetoes, rollback_hook fires on ROLLBACK,
trace_v2 fires for SQLITE_TRACE_STMT.

Hook tests that drive triggers via sqlite3_exec use pytest.importorskip
since exec arrives in a later task — they auto-enable once exec lands.
EOF
git commit -F /tmp/commit-msg.txt
rm /tmp/commit-msg.txt
```

---

## Task 8: Create `_sqlite_column.py` + `test_sqlite_column.py`

**Goal:** Bind the 13 column accessors (`column_int/int64/double/text/blob/bytes/type/count/name/decltype` + 3 metadata accessors via `proxy_if_available`). End-to-end test the bind → step → column round trip.

**Files:**
- Create: `numbox/core/bindings/_sqlite_column.py`
- Create: `test/core/test_sqlite_column.py`

**Acceptance Criteria:**
- [ ] All 13 column wrappers callable from `@njit`
- [ ] `sqlite3_column_int / int64 / double` return the bound values
- [ ] `sqlite3_column_text` returns a `const unsigned char*` that decodes via `get_str_from_p_as_int` to the bound text
- [ ] `sqlite3_column_blob` + `sqlite3_column_bytes` round-trip a known byte buffer
- [ ] `sqlite3_column_type` returns the correct type code (`SQLITE_INTEGER` / `_FLOAT` / `_TEXT` / `_BLOB` / `_NULL`) for each column
- [ ] `sqlite3_column_count(stmt)` matches the SELECT's arity
- [ ] `sqlite3_column_name(stmt, i)` returns the column label
- [ ] `sqlite3_column_decltype(stmt, i)` returns the declared type from the CREATE TABLE
- [ ] `sqlite3_column_database_name / table_name / origin_name` — gated tests via `hasattr(..., "as_func")`

**Verify:**

```bash
cd /home/erik/projects/numbox
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/pytest test/core/test_sqlite_column.py -x --durations=20 -v
venv/bin/flake8 --max-line-length=127 numbox/core/bindings/_sqlite_column.py test/core/test_sqlite_column.py
```

Expected: ~10 column tests pass (or skip on `SQLITE_ENABLE_COLUMN_METADATA`-disabled builds); flake8 clean.

**Steps:**

- [ ] **Step 1: Create `numbox/core/bindings/_sqlite_column.py`**

```python
"""SQLite column accessors.

Three metadata accessors (column_database_name / column_table_name /
column_origin_name) require SQLite to be compiled with
SQLITE_ENABLE_COLUMN_METADATA. CPython's bundled sqlite3 has this enabled,
but external sqlite3.dlls on user PATH may not. proxy_if_available stubs
them out when absent so callers can hasattr-guard or fall back.

All other accessors are universally available across the matrix.
"""
from numbox.core.bindings._sqlite_conn import sqlite3_lib
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.proxy.proxy import proxy, proxy_if_available


@proxy(signatures.get("sqlite3_column_int"), jit_options={"cache": True})
def sqlite3_column_int(stmt_p, idx):
    return _call_lib_func("sqlite3_column_int", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_int64"), jit_options={"cache": True})
def sqlite3_column_int64(stmt_p, idx):
    return _call_lib_func("sqlite3_column_int64", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_double"), jit_options={"cache": True})
def sqlite3_column_double(stmt_p, idx):
    return _call_lib_func("sqlite3_column_double", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_text"), jit_options={"cache": True})
def sqlite3_column_text(stmt_p, idx):
    return _call_lib_func("sqlite3_column_text", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_blob"), jit_options={"cache": True})
def sqlite3_column_blob(stmt_p, idx):
    return _call_lib_func("sqlite3_column_blob", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_bytes"), jit_options={"cache": True})
def sqlite3_column_bytes(stmt_p, idx):
    return _call_lib_func("sqlite3_column_bytes", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_type"), jit_options={"cache": True})
def sqlite3_column_type(stmt_p, idx):
    return _call_lib_func("sqlite3_column_type", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_count"), jit_options={"cache": True})
def sqlite3_column_count(stmt_p):
    return _call_lib_func("sqlite3_column_count", (stmt_p,))


@proxy(signatures.get("sqlite3_column_name"), jit_options={"cache": True})
def sqlite3_column_name(stmt_p, idx):
    return _call_lib_func("sqlite3_column_name", (stmt_p, idx))


@proxy(signatures.get("sqlite3_column_decltype"), jit_options={"cache": True})
def sqlite3_column_decltype(stmt_p, idx):
    return _call_lib_func("sqlite3_column_decltype", (stmt_p, idx))


# Compile-flag-gated (SQLITE_ENABLE_COLUMN_METADATA)
@proxy_if_available(sqlite3_lib, signatures.get("sqlite3_column_database_name"), jit_options={"cache": True})
def sqlite3_column_database_name(stmt_p, idx):
    return _call_lib_func("sqlite3_column_database_name", (stmt_p, idx))


@proxy_if_available(sqlite3_lib, signatures.get("sqlite3_column_table_name"), jit_options={"cache": True})
def sqlite3_column_table_name(stmt_p, idx):
    return _call_lib_func("sqlite3_column_table_name", (stmt_p, idx))


@proxy_if_available(sqlite3_lib, signatures.get("sqlite3_column_origin_name"), jit_options={"cache": True})
def sqlite3_column_origin_name(stmt_p, idx):
    return _call_lib_func("sqlite3_column_origin_name", (stmt_p, idx))
```

- [ ] **Step 2: Create `test/core/test_sqlite_column.py`**

```python
"""Column accessor tests for the SQLite buildout."""
from ctypes import addressof, c_char_p, c_int64, c_void_p

import numpy as np
import pytest

from numbox.core.bindings._sqlite_bind import (
    sqlite3_bind_blob,
    sqlite3_bind_double,
    sqlite3_bind_int,
    sqlite3_bind_int64,
    sqlite3_bind_null,
    sqlite3_bind_text,
)
from numbox.core.bindings._sqlite_column import (
    sqlite3_column_blob,
    sqlite3_column_bytes,
    sqlite3_column_count,
    sqlite3_column_database_name,
    sqlite3_column_decltype,
    sqlite3_column_double,
    sqlite3_column_int,
    sqlite3_column_int64,
    sqlite3_column_name,
    sqlite3_column_origin_name,
    sqlite3_column_table_name,
    sqlite3_column_text,
    sqlite3_column_type,
)
from numbox.core.bindings._sqlite_conn import sqlite3_close, sqlite3_open
from numbox.core.bindings._sqlite_constants import (
    SQLITE_BLOB,
    SQLITE_FLOAT,
    SQLITE_INTEGER,
    SQLITE_NULL,
    SQLITE_OK,
    SQLITE_ROW,
    SQLITE_TEXT,
    SQLITE_TRANSIENT,
)
from numbox.core.bindings._sqlite_stmt import (
    sqlite3_finalize,
    sqlite3_prepare_v2,
    sqlite3_step,
)
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


def _cstr(s):
    buf = c_char_p(s.encode())
    return buf, c_void_p.from_buffer(buf).value


@pytest.fixture
def populated_table(tmp_path):
    """File db with t(i INT, big BIGINT, d REAL, s TEXT, b BLOB, n FLOAT)
    containing one row of known values."""
    import sqlite3 as stdlib_sqlite3
    db_file = tmp_path / "cols.sqlite"
    conn = stdlib_sqlite3.connect(str(db_file))
    conn.executescript(
        "CREATE TABLE t(i INT, big BIGINT, d REAL, s TEXT, b BLOB, n FLOAT);"
        "INSERT INTO t VALUES (42, 1099511627776, 3.14, 'hello', x'01020304', NULL);"
    )
    conn.commit()
    conn.close()
    _, name_p = _cstr(str(db_file))
    db_p = c_int64(0)
    assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    yield db_p.value
    sqlite3_close(db_p.value)


def _prepare_and_step(db_p, sql):
    _, sql_p = _cstr(sql)
    stmt_p = c_int64(0)
    tail_p = c_int64(0)
    sqlite3_prepare_v2(db_p, sql_p, -1, addressof(stmt_p), addressof(tail_p))
    rc = sqlite3_step(stmt_p.value)
    assert rc == SQLITE_ROW
    return stmt_p.value


def test_column_int_int64_double_match_inserted(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT i, big, d FROM t")
    assert sqlite3_column_int(stmt_p, 0) == 42
    assert sqlite3_column_int64(stmt_p, 1) == 1099511627776
    assert abs(sqlite3_column_double(stmt_p, 2) - 3.14) < 1e-9
    sqlite3_finalize(stmt_p)


def test_column_text_decodes_utf8(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT s FROM t")
    text_p = sqlite3_column_text(stmt_p, 0)
    assert str_from_p_as_int(text_p) == "hello"
    sqlite3_finalize(stmt_p)


def test_column_blob_matches_inserted(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT b FROM t")
    blob_p = sqlite3_column_blob(stmt_p, 0)
    n = sqlite3_column_bytes(stmt_p, 0)
    assert n == 4
    buf = (np.ctypeslib.as_array(
        (np.uint8 * n).from_address(blob_p))).copy()
    assert bytes(buf) == bytes([1, 2, 3, 4])
    sqlite3_finalize(stmt_p)


def test_column_type_returns_type_codes(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT i, d, s, b, n FROM t")
    assert sqlite3_column_type(stmt_p, 0) == SQLITE_INTEGER
    assert sqlite3_column_type(stmt_p, 1) == SQLITE_FLOAT
    assert sqlite3_column_type(stmt_p, 2) == SQLITE_TEXT
    assert sqlite3_column_type(stmt_p, 3) == SQLITE_BLOB
    assert sqlite3_column_type(stmt_p, 4) == SQLITE_NULL
    sqlite3_finalize(stmt_p)


def test_column_count_matches_select_arity(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT i, big, d FROM t")
    assert sqlite3_column_count(stmt_p) == 3
    sqlite3_finalize(stmt_p)


def test_column_name_returns_label(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT i AS my_int FROM t")
    name_p = sqlite3_column_name(stmt_p, 0)
    assert str_from_p_as_int(name_p) == "my_int"
    sqlite3_finalize(stmt_p)


def test_column_decltype_returns_declared(populated_table):
    stmt_p = _prepare_and_step(populated_table, "SELECT i FROM t")
    dt_p = sqlite3_column_decltype(stmt_p, 0)
    # The declared type from CREATE TABLE was "INT"
    assert str_from_p_as_int(dt_p).upper() == "INT"
    sqlite3_finalize(stmt_p)


def test_column_database_name_when_available(populated_table):
    if not hasattr(sqlite3_column_database_name, "as_func"):
        pytest.skip("SQLITE_ENABLE_COLUMN_METADATA not in this SQLite build")
    stmt_p = _prepare_and_step(populated_table, "SELECT i FROM t")
    db_name_p = sqlite3_column_database_name(stmt_p, 0)
    assert str_from_p_as_int(db_name_p) == "main"
    sqlite3_finalize(stmt_p)


def test_column_table_name_when_available(populated_table):
    if not hasattr(sqlite3_column_table_name, "as_func"):
        pytest.skip("SQLITE_ENABLE_COLUMN_METADATA not in this SQLite build")
    stmt_p = _prepare_and_step(populated_table, "SELECT i FROM t")
    tbl_p = sqlite3_column_table_name(stmt_p, 0)
    assert str_from_p_as_int(tbl_p) == "t"
    sqlite3_finalize(stmt_p)


def test_column_origin_name_when_available(populated_table):
    if not hasattr(sqlite3_column_origin_name, "as_func"):
        pytest.skip("SQLITE_ENABLE_COLUMN_METADATA not in this SQLite build")
    stmt_p = _prepare_and_step(populated_table, "SELECT i FROM t")
    orig_p = sqlite3_column_origin_name(stmt_p, 0)
    assert str_from_p_as_int(orig_p) == "i"
    sqlite3_finalize(stmt_p)


if __name__ == "__main__":
    collect_and_run_tests(__name__)
```

- [ ] **Step 3: Run tests + lint**

Use the **Verify** command.

- [ ] **Step 4: Commit**

```bash
cd /home/erik/projects/numbox
git add numbox/core/bindings/_sqlite_column.py test/core/test_sqlite_column.py
cat > /tmp/commit-msg.txt <<'EOF'
sqlite: column accessors (with column metadata accessors via proxy_if_available)

Add _sqlite_column.py with 13 bindings: column_int / int64 / double /
text / blob / bytes / type / count / name / decltype plus three metadata
accessors (database_name / table_name / origin_name) gated behind
proxy_if_available against the sqlite3_lib handle. The metadata
accessors require SQLite to be compiled with
SQLITE_ENABLE_COLUMN_METADATA; CPython's bundled sqlite3 enables it,
but user-supplied external sqlite3.dlls may not.

Tests cover scalar / text / blob / null round trip, type codes, column
count and name introspection, decltype from CREATE TABLE, and metadata
accessors with hasattr-guarded skip on builds without
COLUMN_METADATA support.
EOF
git commit -F /tmp/commit-msg.txt
rm /tmp/commit-msg.txt
```

---

## Task 9: Create `_sqlite_exec.py` + `test_sqlite_exec.py`

**Goal:** Bind `sqlite3_exec` (callback-driven one-shot SQL) and `sqlite3_free` (release SQLite-owned memory). Test the callback path end-to-end with a `@cfunc` and the errmsg+free flow.

**Files:**
- Create: `numbox/core/bindings/_sqlite_exec.py`
- Create: `test/core/test_sqlite_exec.py`

**Acceptance Criteria:**
- [ ] `sqlite3_exec(db, sql, 0, 0, 0)` runs SQL with NULL callback and returns `SQLITE_OK` on success
- [ ] `sqlite3_exec(db, sql, cb_addr, ctx_p, 0)` fires the callback per row; the callback can record row count via ctx-backed numpy int64 array
- [ ] Callback returning nonzero aborts exec with `SQLITE_ABORT`
- [ ] `sqlite3_exec` with invalid SQL + non-NULL errmsg_pp writes a `char*` to that slot; `get_str_from_p_as_int` decodes it; passing it to `sqlite3_free` returns void
- [ ] Tests in Tasks 4 / 6 / 7 that depend on `_sqlite_exec` now run (no more `importorskip`)

**Verify:**

```bash
cd /home/erik/projects/numbox
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/pytest test/core/test_sqlite_exec.py test/core/test_sqlite_stmt.py test/core/test_sqlite_blob.py test/core/test_sqlite_hooks.py -x --durations=20 -v
venv/bin/flake8 --max-line-length=127 numbox/core/bindings/_sqlite_exec.py test/core/test_sqlite_exec.py
```

Expected: exec tests pass; previously-skipped tests in stmt/blob/hooks now run and pass; flake8 clean.

**Steps:**

- [ ] **Step 1: Create `numbox/core/bindings/_sqlite_exec.py`**

```python
"""SQLite exec + free bindings.

sqlite3_exec is the one-shot SQL escape hatch — it parses, prepares, steps,
and finalizes a (potentially multi-statement) SQL string. The third arg is a
function pointer to a per-row callback; pass 0 for no callback.

sqlite3_free releases memory SQLite allocated and returned to the caller —
notably the errmsg buffer from sqlite3_exec's out-param, and the result of
sqlite3_expanded_sql.

Callback shape (informational):
    int (*sqlite3_exec_callback)(void *ctx, int ncol,
                                 char **col_values, char **col_names)
Return 0 to continue, nonzero to abort with SQLITE_ABORT.

Produce the callback address from Python via:
    @cfunc(int32(voidptr, int32, intp, intp))
    def my_row_cb(ctx, n, values_pp, names_pp): ...
    sqlite3_exec(db, sql, my_row_cb.address, ctx_p, errmsg_pp)
"""
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.proxy.proxy import proxy


@proxy(signatures.get("sqlite3_exec"), jit_options={"cache": True})
def sqlite3_exec(db_p, sql_p, cb_p, ctx_p, errmsg_pp):
    return _call_lib_func(
        "sqlite3_exec", (db_p, sql_p, cb_p, ctx_p, errmsg_pp)
    )


@proxy(signatures.get("sqlite3_free"), jit_options={"cache": True})
def sqlite3_free(mem_p):
    return _call_lib_func("sqlite3_free", (mem_p,))
```

- [ ] **Step 2: Create `test/core/test_sqlite_exec.py`**

```python
"""sqlite3_exec + sqlite3_free tests."""
from ctypes import addressof, c_char_p, c_int64, c_void_p

import numpy as np
import pytest
from numba import cfunc, types

from numbox.core.bindings._sqlite_conn import (
    sqlite3_changes,
    sqlite3_close,
    sqlite3_open,
)
from numbox.core.bindings._sqlite_constants import SQLITE_ABORT, SQLITE_OK
from numbox.core.bindings._sqlite_exec import sqlite3_exec, sqlite3_free
from numbox.utils.lowlevel import get_str_from_p_as_int
from test.auxiliary_utils import collect_and_run_tests


def _cstr(s):
    buf = c_char_p(s.encode())
    return buf, c_void_p.from_buffer(buf).value


# Module-level cfuncs — must outlive exec for the hooks/exec tests.

@cfunc(types.int32(types.voidptr, types.int32, types.intp, types.intp))
def _count_rows_cb(ctx, n, values_pp, names_pp):
    """Increment ctx[0] (numpy int64) per row."""
    from numba import carray
    arr = carray(ctx, 1, dtype=np.int64)
    arr[0] += 1
    return 0


@cfunc(types.int32(types.voidptr, types.int32, types.intp, types.intp))
def _abort_cb(ctx, n, values_pp, names_pp):
    return 1


@pytest.fixture
def memory_db():
    _, name_p = _cstr(":memory:")
    db_p = c_int64(0)
    assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    yield db_p.value
    sqlite3_close(db_p.value)


def test_exec_create_insert_null_callback(memory_db):
    _, sql_p = _cstr("CREATE TABLE t(a INTEGER); INSERT INTO t VALUES (1);")
    assert sqlite3_exec(memory_db, sql_p, 0, 0, 0) == SQLITE_OK
    assert sqlite3_changes(memory_db) == 1


def test_exec_callback_collects_row_count(memory_db):
    _, setup_p = _cstr("CREATE TABLE t(a INTEGER); "
                       "INSERT INTO t VALUES (1), (2), (3);")
    sqlite3_exec(memory_db, setup_p, 0, 0, 0)

    ctx = np.zeros(1, dtype=np.int64)
    _, sel_p = _cstr("SELECT a FROM t")
    rc = sqlite3_exec(memory_db, sel_p, _count_rows_cb.address,
                      ctx.ctypes.data, 0)
    assert rc == SQLITE_OK
    assert ctx[0] == 3


def test_exec_callback_can_abort(memory_db):
    _, setup_p = _cstr("CREATE TABLE t(a INTEGER); INSERT INTO t VALUES (1);")
    sqlite3_exec(memory_db, setup_p, 0, 0, 0)

    _, sel_p = _cstr("SELECT a FROM t")
    rc = sqlite3_exec(memory_db, sel_p, _abort_cb.address, 0, 0)
    assert rc == SQLITE_ABORT


def test_exec_invalid_sql_writes_errmsg(memory_db):
    _, sql_p = _cstr("SELECT FROM WHERE")
    errmsg_p = c_int64(0)
    rc = sqlite3_exec(memory_db, sql_p, 0, 0, addressof(errmsg_p))
    assert rc != SQLITE_OK
    assert errmsg_p.value != 0
    msg = get_str_from_p_as_int(errmsg_p.value)
    assert "syntax" in msg.lower() or "error" in msg.lower(), msg
    sqlite3_free(errmsg_p.value)  # no return value; just exercise the path


if __name__ == "__main__":
    collect_and_run_tests(__name__)
```

- [ ] **Step 3: Run tests + lint**

Use the **Verify** command. Pay special attention to the previously-skipped tests in `test_sqlite_stmt.py::test_expanded_sql_substitutes_and_must_free` and the four hook tests in `test_sqlite_hooks.py` — they should now run and pass.

- [ ] **Step 4: Commit**

```bash
cd /home/erik/projects/numbox
git add numbox/core/bindings/_sqlite_exec.py test/core/test_sqlite_exec.py
cat > /tmp/commit-msg.txt <<'EOF'
sqlite: exec + free + cfunc callback wiring

Add _sqlite_exec.py with sqlite3_exec (callback-driven one-shot SQL)
and sqlite3_free (release SQLite-owned memory: errmsg from exec,
expanded_sql result, etc.).

Tests cover null-callback (DDL via exec), callback-driven row
collection through a numpy int64 ctx array (carray inside @cfunc),
callback-can-abort returning SQLITE_ABORT, and invalid-SQL errmsg
roundtrip via get_str_from_p_as_int + sqlite3_free.

With _sqlite_exec landed, previously-skipped tests in test_sqlite_stmt
(expanded_sql) and test_sqlite_hooks (update/progress/commit/rollback/
trace) now run end-to-end.
EOF
git commit -F /tmp/commit-msg.txt
rm /tmp/commit-msg.txt
```

---

## Task 10: Retire legacy `_sqlite.py`; rewire `__init__.py`

**Goal:** Delete the legacy `_sqlite.py` (its 3 wrappers now live in `_sqlite_conn.py`) and update `numbox/core/bindings/__init__.py` to star-import from all new `_sqlite_*` modules.

**Files:**
- Delete: `numbox/core/bindings/_sqlite.py`
- Modify: `numbox/core/bindings/__init__.py`

**Acceptance Criteria:**
- [ ] `numbox/core/bindings/_sqlite.py` no longer exists on disk
- [ ] `from numbox.core.bindings import *` exposes all wrappers (`sqlite3_open`, `sqlite3_prepare_v2`, `sqlite3_bind_int`, `sqlite3_column_int`, `sqlite3_exec`, `sqlite3_blob_open`, `sqlite3_update_hook`) AND all `SQLITE_*` constants
- [ ] Full test suite still passes

**Verify:**

```bash
cd /home/erik/projects/numbox
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/python -c "
import numbox.core.bindings as nb
# Check a representative function from each new module is reachable via star-import surface
for name in ('sqlite3_open', 'sqlite3_libversion_number', 'sqlite3_prepare_v2',
            'sqlite3_bind_int', 'sqlite3_column_int', 'sqlite3_exec',
            'sqlite3_blob_open', 'sqlite3_update_hook',
            'SQLITE_OK', 'SQLITE_INTEGER', 'SQLITE_OPEN_READWRITE',
            'SQLITE_BLOB_READWRITE', 'SQLITE_TRACE_STMT',
            'SQLITE_STATIC', 'SQLITE_TRANSIENT'):
    assert hasattr(nb, name), name
print('all expected sqlite names reachable via numbox.core.bindings')
"
venv/bin/pytest -x --durations=20 -q
venv/bin/flake8 --max-line-length=127 numbox/core/bindings/__init__.py
```

Expected: probe prints success line; full pytest (~439 baseline + ~55 new = ~494 tests) all pass; flake8 clean.

**Steps:**

- [ ] **Step 1: Delete `numbox/core/bindings/_sqlite.py`**

```bash
cd /home/erik/projects/numbox
rm numbox/core/bindings/_sqlite.py
```

- [ ] **Step 2: Update `numbox/core/bindings/__init__.py`**

Read the current `__init__.py`. It will look something like:

```python
from numbox.core.bindings._c import *
from numbox.core.bindings._math import *
from numbox.core.bindings._sqlite import *
from numbox.core.bindings._errno import *
from numbox.core.bindings._stdio import *
from numbox.core.bindings._strerror import *
from numbox.core.bindings._fmtio import *
```

Replace the `_sqlite` line with the seven new module imports + constants:

```python
from numbox.core.bindings._c import *
from numbox.core.bindings._math import *
from numbox.core.bindings._sqlite_constants import *
from numbox.core.bindings._sqlite_conn import *
from numbox.core.bindings._sqlite_stmt import *
from numbox.core.bindings._sqlite_bind import *
from numbox.core.bindings._sqlite_column import *
from numbox.core.bindings._sqlite_exec import *
from numbox.core.bindings._sqlite_blob import *
from numbox.core.bindings._sqlite_hooks import *
from numbox.core.bindings._errno import *
from numbox.core.bindings._stdio import *
from numbox.core.bindings._strerror import *
from numbox.core.bindings._fmtio import *
```

Order matters slightly: `_sqlite_conn` (the one that creates `sqlite3_lib`) must come before `_sqlite_column` (which imports `sqlite3_lib` from `_sqlite_conn`). Python's import system handles this OK either way (the actual import resolution happens when each module is first executed), but keeping the order explicit avoids subtle issues if someone refactors. Constants come first so `SQLITE_OK` etc. don't get shadowed.

- [ ] **Step 3: Run probe + full test + lint**

Use the **Verify** command. The full pytest run is the real acceptance criterion.

- [ ] **Step 4: Commit**

```bash
cd /home/erik/projects/numbox
git add numbox/core/bindings/__init__.py numbox/core/bindings/_sqlite.py
cat > /tmp/commit-msg.txt <<'EOF'
sqlite: retire legacy _sqlite.py; rewire __init__.py

Delete numbox/core/bindings/_sqlite.py — its three wrappers (open, close,
libversion) now live in _sqlite_conn.py.

Rewire __init__.py to star-import from the seven new _sqlite_*.py
modules (conn, stmt, bind, column, exec, blob, hooks) plus
_sqlite_constants. Constants come first in the import order so the
uppercase SQLITE_* names don't get shadowed; _sqlite_conn precedes
_sqlite_column since the latter imports sqlite3_lib from the former.
EOF
git commit -F /tmp/commit-msg.txt
rm /tmp/commit-msg.txt
```

---

## Task 11: Update `CLAUDE.md` with Project Status entry + Key Paths additions

**Goal:** Document the buildout in `CLAUDE.md` so future sessions and reviewers have project-level context: what landed, the four lifetime gotchas (bind destructor / errmsg / expanded_sql / cfunc hook), the new file paths, and the small set of follow-ups deferred.

**Files:**
- Modify: `CLAUDE.md`

**Acceptance Criteria:**
- [ ] "Project Status" gains a new entry for the SQLite buildout (after the 0.5.12 libc bindings expansion entry)
- [ ] "Key Paths" gains entries for the 7 new wrapper files + constants module
- [ ] No formatting drift in surrounding sections
- [ ] flake8 not relevant (markdown file); doc-codeblock-flake8 CI (if any code blocks) stays clean

**Verify:**

```bash
cd /home/erik/projects/numbox
git diff CLAUDE.md | head -80     # eyeball-review the diff
ls -la docs/plans/sqlite-buildout/
```

Manual: read the new Project Status entry and confirm the 4 lifetime gotchas are listed (bind destructor, errmsg, expanded_sql, cfunc hook).

**Steps:**

- [ ] **Step 1: Add the Project Status entry**

Insert this entry into `CLAUDE.md`'s "## Project Status" section, immediately after the existing 0.5.12 / libc bindings expansion entry:

```markdown
- **SQLite bindings buildout** — Expands `signatures_sqlite` from 3 entries to 60 (57 new) across [_sqlite_conn.py](numbox/core/bindings/_sqlite_conn.py), [_sqlite_stmt.py](numbox/core/bindings/_sqlite_stmt.py), [_sqlite_bind.py](numbox/core/bindings/_sqlite_bind.py), [_sqlite_column.py](numbox/core/bindings/_sqlite_column.py), [_sqlite_exec.py](numbox/core/bindings/_sqlite_exec.py), [_sqlite_blob.py](numbox/core/bindings/_sqlite_blob.py), [_sqlite_hooks.py](numbox/core/bindings/_sqlite_hooks.py), and [_sqlite_constants.py](numbox/core/bindings/_sqlite_constants.py). Adds Windows support via a new generic `_windows_bundled_dll_path` fallback in [utils.py](numbox/core/bindings/utils.py) (locates `sqlite3.dll` in CPython's `<base_prefix>/DLLs/` and conda's `<base_prefix>/Library/bin/`). Refactors `load_lib` into `_resolve_lib_path` + `load_lib_with_handle` so `proxy_if_available` has access to the CDLL handle. Five functions are decorated with `proxy_if_available`: `sqlite3_changes64` / `sqlite3_total_changes64` (SQLite 3.37+; Python 3.10 ships 3.34) and `sqlite3_column_database_name` / `_table_name` / `_origin_name` (compile-time `SQLITE_ENABLE_COLUMN_METADATA`).

  **Four lifetime gotchas worth memorizing:**
  1. **`bind_text` / `bind_blob` destructor** — pass `SQLITE_TRANSIENT` (= -1) for SQLite to copy. `SQLITE_STATIC` (= 0) assumes the buffer outlives the statement.
  2. **`sqlite3_errmsg` lifetime** — SQLite-owned pointer, valid only until the next API call on the same connection. Decode via `get_str_from_p_as_int` immediately.
  3. **`sqlite3_expanded_sql` cleanup** — caller must release with `sqlite3_free()` (both bound in `_sqlite_exec.py`).
  4. **cfunc lifetime for hook APIs** — registered cfunc must outlive the hook registration. Keep the cfunc at module scope in the caller.

  Callback wiring pattern for `sqlite3_exec` and the six hook APIs:
  ```python
  @cfunc(types.int32(types.voidptr, types.int32, types.intp, types.intp))
  def _row_cb(ctx, ncol, values_pp, names_pp):
      return 0
  sqlite3_exec(db_p, sql_p, _row_cb.address, ctx_p, 0)
  ```

  Design spec at [docs/plans/sqlite-buildout/2026-05-23-design.md](2026-05-23-design.md).
```

- [ ] **Step 2: Update Key Paths**

Add these entries to the "## Key Paths" section:

```markdown
- `numbox/core/bindings/_sqlite_conn.py` — connection + metadata wrappers; initializes module-level `sqlite3_lib`
- `numbox/core/bindings/_sqlite_stmt.py` — statement lifecycle
- `numbox/core/bindings/_sqlite_bind.py` — parameter binding
- `numbox/core/bindings/_sqlite_column.py` — column accessors
- `numbox/core/bindings/_sqlite_exec.py` — exec + free
- `numbox/core/bindings/_sqlite_blob.py` — BLOB incremental I/O
- `numbox/core/bindings/_sqlite_hooks.py` — callback hooks
- `numbox/core/bindings/_sqlite_constants.py` — SQLite result codes, type codes, flags, destructor sentinels
```

The existing `numbox/core/bindings/_sqlite.py` entry should be removed (the file no longer exists).

- [ ] **Step 3: Add deferred-follow-ups note**

Append under "## Follow-ups" (after the existing Vector/List entry):

```markdown
- **User-defined SQL functions** — `sqlite3_create_function_v2` + the `sqlite3_value_*` / `sqlite3_result_*` API surface. Its own significant buildout; relevant for numbduck-style consumers but separate scope.
- **Backup API** — `sqlite3_backup_init` / `_step` / `_finish` / `_pagecount` / `_remaining`. Six functions plus the progress callback shape.
- **Serialize / deserialize** — `sqlite3_serialize` / `sqlite3_deserialize` for in-memory snapshots.
- **Higher-level structref wrappers** — `Connection` / `Statement` structrefs if a downstream consumer needs ergonomic types.
- **Drop `proxy_if_available` gate on `changes64` / `total_changes64`** once Python 3.10 drops out of the support matrix (Python 3.11+ all ship SQLite 3.37+).
```

- [ ] **Step 4: Verify + commit**

```bash
cd /home/erik/projects/numbox
git diff CLAUDE.md
git add CLAUDE.md
cat > /tmp/commit-msg.txt <<'EOF'
docs: CLAUDE.md SQLite buildout entry + key paths

Document the SQLite buildout in Project Status: 57 new bindings across
7 wrapper files + constants module, Windows bundled-DLL fallback,
load_lib refactor, the five proxy_if_available functions, and four
lifetime gotchas (bind destructor / errmsg / expanded_sql / cfunc
hook lifetime).

Update Key Paths to list the seven new _sqlite_*.py modules and the
constants module; remove the now-deleted _sqlite.py entry.

Record deferred follow-ups: user-defined SQL functions (CREATE
FUNCTION), backup API, serialize / deserialize, structref wrappers,
and removing the changes64 / total_changes64 gate once Python 3.10
drops from the support matrix.
EOF
git commit -F /tmp/commit-msg.txt
rm /tmp/commit-msg.txt
```

---

## Post-completion: full-suite verification + fork CI gate

After Task 11, run the full suite one more time to confirm no regressions:

```bash
cd /home/erik/projects/numbox
venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
venv/bin/pytest --durations=20 -q
venv/bin/flake8 --max-line-length=127 .
```

Expected counts (rough): ~439 baseline tests + ~55 new = ~494 total, all pass. flake8 clean across the repo.

Then push the feature branch and open fork PR (analogous to PR #20 / PR #26):

```bash
git -C /home/erik/projects/numbox push -u origin feat/sqlite-buildout
```

**Then wait for fork CI matrix green via `gh run watch <id> --exit-status`** before considering this feature-branch work done. After matrix green, the upstream-PR branch creation (per spec §5.1) is a separate workstream — explicit user consent required per the standing rule before pushing to or creating an upstream PR.
