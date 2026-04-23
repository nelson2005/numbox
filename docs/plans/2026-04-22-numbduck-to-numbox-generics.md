# Generalize numbduck generics into numbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift six project-agnostic utilities out of numbduck and into numbox so any numba-plus-C-library project can reuse them, then migrate numbduck to consume them from numbox.

**Architecture:** Phase 1 makes additive changes to numbox (6 tasks, TDD, each commit independently green). Phase 2 replaces numbduck's local copies with imports from a locally-installed numbox (5 tasks, validated by numbduck's existing test suite). Phase 3 cuts a numbox 0.5.9 release and bumps numbduck's pin. Upstream PRs (numbox, numbduck) are deliberately out-of-scope — they are follow-ups once the fork branches are green.

**Tech Stack:** numba (>=0.60, <0.66), llvmlite, numpy, pytest, flake8, setuptools. Numba `@intrinsic` for codegen, `structref` for state containers, NRT for refcount.

---

## Cross-Project Paths

- **numbox:** `/home/erik/projects/numbox`, venv: `/home/erik/projects/numbox/venv`
- **numbduck:** `/home/erik/projects/numbduck`, venv: `/home/erik/projects/numbduck/venv`
- All commands below use absolute paths and `git -C <dir>` — no `cd` required.

## Branching

- Start: numbox on `fork/setup` (13e9e93 + CI pattern commit e8e6d47), numbduck on `feat/irr-udaf-example` (per active_resume_points.md).
- Phase 1 branch: `feat/generalize-numbduck-generics` off current `fork/setup` in numbox.
- Phase 2 branch: `feat/use-numbox-generics` off current `feat/irr-udaf-example` in numbduck.
- Final numbox release cuts a tag on main after the feature branch is merged (out-of-band via GitHub PR).

## Cache-clear reminder

Per the user's standing "clean cache before tests" rule, prepend this single-line invocation to any pytest run in either project:

```bash
venv/bin/python -c "import shutil, pathlib; shutil.rmtree('/home/erik/.cache/numba', ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('venv').rglob('*.nbi')]; [p.unlink() for p in pathlib.Path('venv').rglob('*.nbc')]"
```

(Absolute-path equivalents shown in each task's `Verify` command.)

## File Structure

### numbox additions (all on `feat/generalize-numbduck-generics`)

| File | Action | Responsibility |
|------|--------|---------------|
| `numbox/utils/lowlevel.py` | modify | Add `array_data_p(arr) -> intp` |
| `numbox/core/bindings/utils.py` | modify | Add `load_lib_path(path) -> CDLL` |
| `numbox/utils/highlevel.py` | modify | Add `cres_if_available(lib, sig, **kwargs)` |
| `numbox/core/bindings/abi.py` | **create** | Struct-by-value ABI codegen helpers + platform flags |
| `numbox/utils/meminfo.py` | modify | Add `_incref_meminfo`, `_release_meminfo`, `_deref_structref_raw_ptr`, `borrow_structref`, `export_meminfo` |
| `numbox/core/vector.py` | **create** | Generic growable numba Vector (`make_vector(elem_type)`) |
| `test/utils/test_lowlevel.py` | modify | Test `array_data_p` |
| `test/core/test_bindings.py` | modify | Test `load_lib_path` |
| `test/utils/test_highlevel.py` | modify | Test `cres_if_available` |
| `test/core/test_abi.py` | **create** | Test ABI helpers via libm shims |
| `test/core/test_meminfo.py` | modify | Test bridge intrinsics (refcount ladder + borrow round-trip) |
| `test/core/test_vector.py` | **create** | Port numbduck's `test/test_vector.py` |
| `CLAUDE.md` | modify | Add `docs/plans/**` to upstream-exclude list |

### numbduck migrations (all on `feat/use-numbox-generics`)

| File | Action | Responsibility |
|------|--------|---------------|
| `numbduck/jit_utils.py` | **delete** | Replaced by `numbox.utils.lowlevel.array_data_p` |
| `numbduck/utils.py` | modify | Use numbox's `load_lib_path`; keep DuckDB discovery (`find_duckdb_shared_lib`, `_download_libduckdb`, `load_duckdb`) |
| `numbduck/ducklib.py` | modify | Import `cres_if_available` + ABI helpers from numbox; drop local copies + platform flags |
| `numbduck/vector.py` | **delete** | Replaced by `numbox.core.vector` |
| `test/test_ducklib.py` | modify | Drop inline bridge intrinsics; import from numbox |
| `test/test_vector.py` | modify | Import `make_vector` from numbox |
| `examples/irr.py` | modify | Import `make_vector` from numbox |
| `pyproject.toml` | modify | Bump `numbox~=0.5.8` → `numbox~=0.5.9` |

---

## Task 0: Setup branches and docs/plans exclusion

**Goal:** Establish feature branches on both repos; add `docs/plans/**` to numbox's upstream-exclude list so the plan artifacts stay fork-local.

**Model:** haiku

**Files:**
- Modify: `/home/erik/projects/numbox/CLAUDE.md` (Preferences section, first bullet)

**Acceptance Criteria:**
- [ ] numbox current branch = `feat/generalize-numbduck-generics`, starts from `fork/setup` HEAD
- [ ] numbduck current branch = `feat/use-numbox-generics`, starts from `feat/irr-udaf-example` HEAD
- [ ] numbox `CLAUDE.md` Preferences bullet includes `docs/plans/**` in the upstream-exclude list

**Verify:**
```bash
git -C /home/erik/projects/numbox branch --show-current
git -C /home/erik/projects/numbduck branch --show-current
grep "docs/plans" /home/erik/projects/numbox/CLAUDE.md
```
Expected: `feat/generalize-numbduck-generics`, `feat/use-numbox-generics`, one grep match.

**Steps:**

- [ ] **Step 1: Create numbox feature branch**

```bash
git -C /home/erik/projects/numbox checkout -b feat/generalize-numbduck-generics fork/setup
```

- [ ] **Step 2: Edit `/home/erik/projects/numbox/CLAUDE.md` — the first Preferences bullet**

Replace:
```
- Always exclude CLAUDE.md and fork-only `numbox_ci.yml` matrix expansions from upstream PRs (use a dedicated branch based on `upstream/main`)
```
with:
```
- Always exclude CLAUDE.md, `docs/plans/**`, and fork-only `numbox_ci.yml` matrix expansions from upstream PRs (use a dedicated branch based on `upstream/main`)
```

- [ ] **Step 3: Commit the CLAUDE.md change AND the plan + tasks file**

```bash
git -C /home/erik/projects/numbox add CLAUDE.md docs/plans/2026-04-22-numbduck-to-numbox-generics.md docs/plans/2026-04-22-numbduck-to-numbox-generics.tasks.json
git -C /home/erik/projects/numbox commit -F /tmp/t0-commit.txt
```
where `/tmp/t0-commit.txt` contains:
```
Plan: generalize numbduck generics into numbox

Exclude docs/plans from upstream PRs per feature-branch/upstream-branch
workflow. Plan document + tasks live alongside it.
```

- [ ] **Step 4: Create numbduck feature branch**

```bash
git -C /home/erik/projects/numbduck checkout -b feat/use-numbox-generics feat/irr-udaf-example
```

---

## Task 1: Add `array_data_p` to `numbox/utils/lowlevel.py`

**Goal:** Provide a numba-callable helper that returns a numpy array's data pointer as signed `intp` — matches the pointer convention used by numbox binding signatures. Replaces numbduck's `jit_utils.py`.

**Model:** sonnet

**Files:**
- Modify: `/home/erik/projects/numbox/numbox/utils/lowlevel.py` (append at end of file)
- Modify: `/home/erik/projects/numbox/test/utils/test_lowlevel.py` (append new tests)

**Acceptance Criteria:**
- [ ] `array_data_p(arr)` is callable from Python and `@njit` contexts
- [ ] Returns `int` equal to `arr.ctypes.data`
- [ ] Works for at least `float64` and `int32` dtype arrays

**Verify:**
```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree('/home/erik/.cache/numba', ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]"
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/utils/test_lowlevel.py -v -k array_data_p
```
Expected: `2 passed`.

**Steps:**

- [ ] **Step 1: Write failing tests**

Append to `/home/erik/projects/numbox/test/utils/test_lowlevel.py`:

```python
def test_array_data_p_python():
    import numpy
    from numbox.utils.lowlevel import array_data_p
    arr = numpy.arange(4, dtype=numpy.float64)
    assert array_data_p(arr) == arr.ctypes.data


def test_array_data_p_njit():
    import numpy
    from numba import njit
    from numbox.utils.lowlevel import array_data_p

    @njit
    def wrap(a):
        return array_data_p(a)

    arr = numpy.arange(4, dtype=numpy.int32)
    assert wrap(arr) == arr.ctypes.data
```

- [ ] **Step 2: Run; confirm `ImportError` / `AttributeError`**

- [ ] **Step 3: Implement in `/home/erik/projects/numbox/numbox/utils/lowlevel.py`** — append:

```python
@njit
def array_data_p(arr):
    """Return the data pointer of a numpy array as signed intp.

    ``arr.ctypes.data`` is ``uint64`` under numba; the cast aligns with the
    signed-pointer convention used by numbox binding signatures. Callable
    from Python and ``@njit`` contexts.
    """
    return intp(arr.ctypes.data)
```

(`intp` is already imported at the top of `lowlevel.py` — no new imports needed.)

- [ ] **Step 4: Re-run tests; confirm 2 passed**

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/utils/lowlevel.py test/utils/test_lowlevel.py
git -C /home/erik/projects/numbox commit -m "Add array_data_p helper to utils.lowlevel"
```

---

## Task 2: Add `load_lib_path` to `numbox/core/bindings/utils.py`

**Goal:** Provide a path-based, handle-returning `CDLL` loader with platform-appropriate flags. Existing `load_lib(name)` uses `find_library` and discards the handle; symbol-gated bindings (Task 3) need the handle back.

**Model:** sonnet

**Files:**
- Modify: `/home/erik/projects/numbox/numbox/core/bindings/utils.py`
- Modify: `/home/erik/projects/numbox/test/core/test_bindings.py`

**Acceptance Criteria:**
- [ ] `load_lib_path(path) -> CDLL` returns a loaded library
- [ ] Linux/Darwin: uses `RTLD_GLOBAL`
- [ ] Windows: uses `winmode=0`
- [ ] Unknown platform: `RuntimeError`
- [ ] The returned handle supports `hasattr(lib, sym)` checks for symbol presence

**Verify:**
```bash
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/core/test_bindings.py -v -k load_lib_path
```
Expected: `1 passed` (Linux CI) / `1 passed` (Windows) — parametrize lib path per platform, but single test with `ctypes.util.find_library("m")` is sufficient for Linux/Darwin.

**Steps:**

- [ ] **Step 1: Write the failing test**

Append to `/home/erik/projects/numbox/test/core/test_bindings.py`:

```python
def test_load_lib_path_returns_handle_with_known_symbol():
    from ctypes.util import find_library
    from numbox.core.bindings.utils import load_lib_path

    libm_path = find_library("m")
    assert libm_path is not None, "libm not discoverable on this platform"
    lib = load_lib_path(libm_path)
    assert hasattr(lib, "cos")
```

- [ ] **Step 2: Run; confirm ImportError**

- [ ] **Step 3: Implement in `/home/erik/projects/numbox/numbox/core/bindings/utils.py`** — append:

```python
def load_lib_path(path):
    """Load a shared library by absolute path and return the handle.

    Linux/Darwin: ``RTLD_GLOBAL`` so symbols reach LLVM's JIT. Windows:
    ``winmode=0``. Unlike ``load_lib(name)``, the handle is returned so
    callers can check symbol presence with ``hasattr``.
    """
    if platform_ in ("Darwin", "Linux"):
        from os import RTLD_GLOBAL
        return CDLL(path, mode=RTLD_GLOBAL)
    if platform_ == "Windows":
        return CDLL(path, winmode=0)
    raise RuntimeError(f"Platform {platform_} is not supported, yet.")
```

- [ ] **Step 4: Re-run tests; confirm passed**

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/utils.py test/core/test_bindings.py
git -C /home/erik/projects/numbox commit -m "Add load_lib_path: path-based, handle-returning CDLL loader"
```

---

## Task 3: Add `cres_if_available` to `numbox/utils/highlevel.py`

**Goal:** Provide a `cres` variant that stubs out a wrapper when the named C symbol is missing from a given library handle. Generalizes numbduck's `cres(..., if_available=True)` pattern so any version-sensitive binding set can skip missing symbols.

**Model:** sonnet

**Files:**
- Modify: `/home/erik/projects/numbox/numbox/utils/highlevel.py`
- Modify: `/home/erik/projects/numbox/test/utils/test_highlevel.py`

**Acceptance Criteria:**
- [ ] `cres_if_available(lib, sig, **kwargs)(func)` behaves exactly like `cres(sig, **kwargs)(func)` when `hasattr(lib, func.__name__)` is `True`
- [ ] When the symbol is missing: returns a Python-callable stub with the same `__name__` that raises `NotImplementedError(f"{name} is not available")`

**Verify:**
```bash
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/utils/test_highlevel.py -v -k cres_if_available
```
Expected: `2 passed`.

**Steps:**

- [ ] **Step 1: Write failing tests**

Append to `/home/erik/projects/numbox/test/utils/test_highlevel.py`:

```python
def test_cres_if_available_present_symbol_returns_real_wrapper():
    from ctypes.util import find_library
    from numba import float64
    from numbox.core.bindings.signatures import signatures
    from numbox.core.bindings.utils import load_lib_path
    from numbox.utils.highlevel import cres_if_available

    signatures["cos"] = float64(float64)
    libm = load_lib_path(find_library("m"))

    @cres_if_available(libm, signatures["cos"])
    def cos(x):
        from numbox.core.bindings.call import _call_lib_func
        return _call_lib_func("cos", (x,))

    from numba.core.types.function_type import CompileResultWAP
    assert isinstance(cos, CompileResultWAP)


def test_cres_if_available_missing_symbol_returns_stub():
    from numba import float64
    from numbox.core.bindings.signatures import signatures
    from numbox.core.bindings.utils import load_lib_path
    from numbox.utils.highlevel import cres_if_available
    from ctypes.util import find_library
    import pytest

    signatures["nonexistent_fn"] = float64(float64)
    libm = load_lib_path(find_library("m"))

    @cres_if_available(libm, signatures["nonexistent_fn"])
    def nonexistent_fn(x):
        return x

    assert nonexistent_fn.__name__ == "nonexistent_fn"
    with pytest.raises(NotImplementedError, match="nonexistent_fn is not available"):
        nonexistent_fn(1.0)
```

- [ ] **Step 2: Run; confirm ImportError**

- [ ] **Step 3: Implement in `/home/erik/projects/numbox/numbox/utils/highlevel.py`** — append after `cres`:

```python
def cres_if_available(lib, sig, **kwargs):
    """Like ``cres(sig, **kwargs)``, but stubs out the wrapper if the C
    symbol matching ``func.__name__`` is absent from ``lib``.

    Use for binding sets that target multiple library versions where some
    symbols may only exist in newer releases. Callers get a stub that
    raises ``NotImplementedError`` instead of a confusing LLVM link error
    at call time.
    """
    def _(func):
        if hasattr(lib, func.__name__):
            return cres(sig, **kwargs)(func)

        def stub(*args, **_kwargs):
            raise NotImplementedError(f"{func.__name__} is not available")
        stub.__name__ = func.__name__
        return stub
    return _
```

- [ ] **Step 4: Re-run tests; confirm passed**

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/utils/highlevel.py test/utils/test_highlevel.py
git -C /home/erik/projects/numbox commit -m "Add cres_if_available: symbol-gated cres variant"
```

---

## Task 4: Create `numbox/core/bindings/abi.py`

**Goal:** Move numbduck's struct-by-value codegen helpers into numbox so any future small-struct FFI binding can reuse them. Scope: SysV x86-64 + Windows ABI handling for ≤16-byte struct args/returns, plus the always-by-pointer variant.

**Model:** opus

**Files:**
- Create: `/home/erik/projects/numbox/numbox/core/bindings/abi.py`
- Create: `/home/erik/projects/numbox/test/core/test_abi.py`

**Acceptance Criteria:**
- [ ] Public symbols: `_is_win`, `_is_sysv_x86_64`, `_emit_byval_call`, `_call_lib_func_byval`, `_call_lib_func_struct_in`, `_call_lib_func_struct_out`
- [ ] Intrinsics read signatures from `numbox.core.bindings.signatures.signatures`
- [ ] Module imports cleanly (`python -c "import numbox.core.bindings.abi"` exits 0)
- [ ] A minimal smoke test exercises `_call_lib_func_struct_out` through `_call_lib_func_byval` to verify compilation (return of a `UniTuple(intp, 2)` via `sret` path on Windows / by-value path on Linux)

**Verify:**
```bash
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/core/test_abi.py -v
```
Expected: `2 passed` (one SysV smoke test skipped on Windows, one Windows-only smoke test skipped on Linux).

**Steps:**

- [ ] **Step 1: Write smoke tests**

Create `/home/erik/projects/numbox/test/core/test_abi.py`:

```python
import platform
import sys
import pytest


def test_abi_imports():
    from numbox.core.bindings import abi
    assert hasattr(abi, "_emit_byval_call")
    assert hasattr(abi, "_call_lib_func_byval")
    assert hasattr(abi, "_call_lib_func_struct_in")
    assert hasattr(abi, "_call_lib_func_struct_out")
    assert hasattr(abi, "_is_win")
    assert hasattr(abi, "_is_sysv_x86_64")


@pytest.mark.skipif(sys.platform == "win32", reason="SysV path only")
def test_sysv_platform_flags():
    from numbox.core.bindings import abi
    assert abi._is_win is False
    if platform.machine() == "x86_64":
        assert abi._is_sysv_x86_64 is True
```

- [ ] **Step 2: Run; confirm ImportError**

- [ ] **Step 3: Create `/home/erik/projects/numbox/numbox/core/bindings/abi.py`** — content lifted verbatim from `/home/erik/projects/numbduck/numbduck/ducklib.py` lines 139-245, plus the module-level platform flags from lines 10-19:

```python
"""Struct-by-value ABI codegen helpers for numba bindings.

LLVM's JIT treats ABI lowering as a frontend responsibility — it won't
insert the right calling convention for struct args/returns by itself.
These helpers generate the appropriate IR for SysV x86-64 and Windows.

References:
    https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    https://github.com/llvm/llvm-project/issues/85417
"""
import platform
import sys

from llvmlite import ir
from llvmlite.ir import FunctionType
from numba.core.cgutils import get_or_insert_function
from numba.extending import intrinsic

from numbox.core.bindings.signatures import signatures


_is_win = sys.platform == "win32"
_is_sysv_x86_64 = platform.machine() == "x86_64" and not _is_win


def _resolve_sig(func_name):
    func_sig = signatures.get(func_name, None)
    if func_sig is None:
        raise ValueError(f"Undefined signature for {func_name}")
    return func_sig


def _emit_byval_call(builder, context, arg, arg_ll_ty, ret_type, func_name):
    """Emit IR to pass a struct by pointer: alloca, store, call via pointer."""
    stack_p = builder.alloca(arg_ll_ty)
    builder.store(arg, stack_p)
    func_ty_ll = FunctionType(ret_type, [arg_ll_ty.as_pointer()])
    func_p = get_or_insert_function(builder.module, func_ty_ll, func_name)
    return builder.call(func_p, [stack_p])


@intrinsic(prefer_literal=True)
def _call_lib_func_byval(typingctx, func_name_ty, arg_ty):
    """Pass ``arg`` to a C function by pointer on all platforms.

    Used when the C signature takes a pointer to a struct and the caller
    holds the struct as a value (e.g. ``duckdb_result *``).
    """
    func_name = func_name_ty.literal_value
    func_sig = _resolve_sig(func_name)

    def codegen(context, builder, signature, arguments):
        _, arg = arguments
        arg_ll_ty = context.get_value_type(arg_ty)
        ret_type = context.get_value_type(signature.return_type)
        return _emit_byval_call(
            builder, context, arg, arg_ll_ty, ret_type, func_name)

    sig = func_sig.return_type(func_name_ty, arg_ty)
    return sig, codegen


@intrinsic(prefer_literal=True)
def _call_lib_func_struct_in(typingctx, func_name_ty, arg_ty):
    """Pass a ≤16-byte struct by value (SysV x86-64) or by pointer (Windows)."""
    func_name = func_name_ty.literal_value
    func_sig = _resolve_sig(func_name)
    struct_bytes = sum(t.bitwidth for t in arg_ty) / 8
    assert struct_bytes <= 16, (
        f"struct too large for by-value passing ({struct_bytes} bytes > 16)"
    )

    def codegen(context, builder, signature, arguments):
        _, arg = arguments
        arg_ll_ty = context.get_value_type(arg_ty)
        ret_type = context.get_value_type(signature.return_type)
        if _is_win:
            return _emit_byval_call(
                builder, context, arg, arg_ll_ty, ret_type, func_name)
        func_ty_ll = FunctionType(ret_type, [arg_ll_ty])
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, func_name)
        return builder.call(func_p, [arg])

    sig = func_sig.return_type(func_name_ty, arg_ty)
    return sig, codegen


@intrinsic(prefer_literal=True)
def _call_lib_func_struct_out(typingctx, func_name_ty, arg_ty):
    """Return a ≤16-byte struct by value (SysV x86-64) or via sret (Windows)."""
    func_name = func_name_ty.literal_value
    func_sig = _resolve_sig(func_name)
    ret_ty = func_sig.return_type
    struct_bytes = sum(t.bitwidth for t in ret_ty) / 8
    assert struct_bytes <= 16, (
        f"return struct too large for by-value return ({struct_bytes} bytes > 16)"
    )

    def codegen(context, builder, signature, arguments):
        _, arg = arguments
        ret_ll_ty = context.get_value_type(signature.return_type)
        if _is_win:
            sret_p = builder.alloca(ret_ll_ty)
            func_ty_ll = FunctionType(
                ir.VoidType(),
                [ret_ll_ty.as_pointer(), arg.type]
            )
            func_p = get_or_insert_function(
                builder.module, func_ty_ll, func_name)
            func_p.args[0].add_attribute("sret")
            builder.call(func_p, [sret_p, arg])
            return builder.load(sret_p)
        func_ty_ll = FunctionType(ret_ll_ty, [arg.type])
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, func_name)
        return builder.call(func_p, [arg])

    sig = func_sig.return_type(func_name_ty, arg_ty)
    return sig, codegen
```

- [ ] **Step 4: Re-run tests; confirm 2 passed (1 skipped on Windows)**

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/abi.py test/core/test_abi.py
git -C /home/erik/projects/numbox commit -m "Add core.bindings.abi: struct-by-value ABI codegen helpers"
```

---

## Task 5: Add bridge intrinsics to `numbox/utils/meminfo.py`

**Goal:** Promote numbduck's inline structref↔raw-pointer bridge (incref/decref/deref of MemInfo from an `intp`) into numbox, where `structref_meminfo` already lives. Enables any project with native callbacks to hand a structref across the FFI boundary.

**Model:** opus

**Files:**
- Modify: `/home/erik/projects/numbox/numbox/utils/meminfo.py`
- Modify: `/home/erik/projects/numbox/test/core/test_meminfo.py`

**Acceptance Criteria:**
- [ ] `_incref_meminfo(p)` and `_release_meminfo(p)` are `@intrinsic`s taking an `intp` MemInfo pointer
- [ ] `_deref_structref_raw_ptr(struct_type, p)` reconstructs a structref value from an `intp` MemInfo
- [ ] `borrow_structref(struct_type, p)` (njit) = incref + deref
- [ ] `export_meminfo(s)` (njit) = structref_meminfo + incref, returns `intp`
- [ ] Refcount ladder test: `s = MyStruct(...)` → refcount 1; `p = export_meminfo(s)` → 2; `_release_meminfo(p)` → 1
- [ ] Round-trip test: `export_meminfo` then `borrow_structref` recovers original field values

**Verify:**
```bash
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/core/test_meminfo.py -v -k bridge
```
Expected: `2 passed`.

**Steps:**

- [ ] **Step 1: Write failing tests**

Append to `/home/erik/projects/numbox/test/core/test_meminfo.py` (use existing `S1` fixture from `test.common_structrefs`):

```python
def test_bridge_refcount_ladder():
    from numbox.utils.meminfo import (
        _release_meminfo, borrow_structref, export_meminfo, get_nrt_refcount,
    )
    from test.common_structrefs import S1, S1Type

    s = S1(1, 2, 3.0)
    assert get_nrt_refcount(s) == 1
    p = export_meminfo(s)
    assert get_nrt_refcount(s) == 2

    @numba.njit
    def bounce(p_):
        t = borrow_structref(S1Type, p_)
        return t.x1

    assert bounce(p) == 1
    _release_meminfo(p)
    assert get_nrt_refcount(s) == 1


def test_bridge_round_trip_recovers_fields():
    from numbox.utils.meminfo import borrow_structref, export_meminfo
    from test.common_structrefs import S1, S1Type

    s = S1(42, 43, 3.14)
    p = export_meminfo(s)

    @numba.njit
    def read_all(p_):
        t = borrow_structref(S1Type, p_)
        return (t.x1, t.x2, t.x3)

    assert read_all(p) == (42, 43, 3.14)
```

(`numba` is already imported at the top of `test_meminfo.py`.)

- [ ] **Step 2: Run; confirm ImportError**

- [ ] **Step 3: Extend `/home/erik/projects/numbox/numbox/utils/meminfo.py`** — append:

```python
from llvmlite import ir as llir
from numba.core import cgutils


_MI_TY = types.MemInfoPointer(types.voidptr)


@intrinsic
def _incref_meminfo(typingctx, p_ty):
    """Incref a MemInfo at ``intp`` via NRT."""
    sig = types.void(p_ty)

    def codegen(context, builder, signature, args):
        mi_ll_ty = context.get_value_type(_MI_TY)
        meminfo = builder.inttoptr(args[0], mi_ll_ty)
        context.nrt.incref(builder, _MI_TY, meminfo)
    return sig, codegen


@intrinsic
def _release_meminfo(typingctx, p_ty):
    """Decref a MemInfo at ``intp`` via ``NRT_MemInfo_release``.

    Can't use ``context.nrt.decref()`` here — ``removerefctpass`` strips
    ``NRT_decref`` when the function signature has no NRT-tracked types.
    ``NRT_MemInfo_release`` does the same atomic decref + dtor call AND
    causes ``_legalize()`` to bail out of the rewrite, protecting the
    whole function.
    """
    sig = types.void(p_ty)

    def codegen(context, builder, signature, args):
        ptr_ty = llir.IntType(8).as_pointer()
        fnty = llir.FunctionType(llir.VoidType(), [ptr_ty])
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, "NRT_MemInfo_release")
        meminfo = builder.inttoptr(args[0], ptr_ty)
        builder.call(fn, [meminfo])
    return sig, codegen


@intrinsic
def _deref_structref_raw_ptr(typingctx, struct_type_ref, p_ty):
    """Reconstruct a structref value from an ``intp`` MemInfo pointer."""
    inst_type = struct_type_ref.instance_type
    sig = inst_type(struct_type_ref, p_ty)

    def codegen(context, builder, signature, args):
        p_val = args[1]
        mi_ll_ty = context.get_value_type(_MI_TY)
        meminfo = builder.inttoptr(p_val, mi_ll_ty)
        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo
        return st._getvalue()
    return sig, codegen


@numba.njit
def borrow_structref(struct_type, p):
    """Incref + reconstruct a structref from an ``intp`` MemInfo pointer.

    The caller receives a live structref that participates in normal NRT
    refcount. Net-zero for the external owner if the caller's scope exits
    without additional actions (local decref on drop balances the incref).
    """
    _incref_meminfo(p)
    return _deref_structref_raw_ptr(struct_type, p)


@numba.njit
def export_meminfo(s):
    """Export a structref as an ``intp`` MemInfo pointer with a +1 incref.

    The returned ``intp`` keeps the allocation alive until the caller
    balances it with ``_release_meminfo``.
    """
    meminfo_p, _ = structref_meminfo(s)
    _incref_meminfo(meminfo_p)
    return meminfo_p
```

- [ ] **Step 4: Re-run tests; confirm 2 passed**

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/utils/meminfo.py test/core/test_meminfo.py
git -C /home/erik/projects/numbox commit -m "Add structref-to-raw-ptr bridge intrinsics to utils.meminfo"
```

---

## Task 6: Create `numbox/core/vector.py`

**Goal:** Port numbduck's generic growable numba Vector. Cache-stable structref (single module-level class parameterized by field tuple) matching numbox's own `WorkTypeClass` pattern.

**Model:** sonnet

**Files:**
- Create: `/home/erik/projects/numbox/numbox/core/vector.py`
- Create: `/home/erik/projects/numbox/test/core/test_vector.py`

**Acceptance Criteria:**
- [ ] `make_vector(elem_type) -> (create, type_instance)` returns a factory + numba type
- [ ] `vector_push(v, val)` grows geometrically when at capacity
- [ ] `vector_extend(dst, src)` copies and grows as needed
- [ ] `len(v)`, `v[i]`, `v[i] = x` work inside `@njit`
- [ ] Memoized by `elem_type.key` so repeated calls return the same `(create, type_instance)` tuple
- [ ] All tests ported from `/home/erik/projects/numbduck/test/test_vector.py` pass against `numbox.core.vector`

**Verify:**
```bash
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/core/test_vector.py -v
```
Expected: all ported tests pass (same count as numbduck's, typically 14).

**Steps:**

- [ ] **Step 1: Copy numbduck's vector.py verbatim**

Copy `/home/erik/projects/numbduck/numbduck/vector.py` → `/home/erik/projects/numbox/numbox/core/vector.py`. No edits needed — file is numba/numpy only and already references only generic symbols.

- [ ] **Step 2: Copy numbduck's test_vector.py, rewriting the import**

Copy `/home/erik/projects/numbduck/test/test_vector.py` → `/home/erik/projects/numbox/test/core/test_vector.py`. Change any import from `numbduck.vector` to `numbox.core.vector`. Verify no other numbduck-specific imports remain.

- [ ] **Step 3: Run tests; confirm all pass**

```bash
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/core/test_vector.py -v
```

- [ ] **Step 4: Full numbox test suite — regression check**

```bash
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test -v --durations=20
```

Expected: no regressions vs. pre-plan baseline.

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/vector.py test/core/test_vector.py
git -C /home/erik/projects/numbox commit -m "Add core.vector: generic growable numba Vector container"
```

---

## Task 7: Install numbox feature branch into numbduck's venv

**Goal:** Point numbduck's venv at the locally-checked-out numbox feature branch so Phase 2 migrations can import the new symbols.

**Model:** haiku

**Files:** (none edited; environment change only)

**Acceptance Criteria:**
- [ ] `numbduck/venv/bin/python -c "from numbox.utils.lowlevel import array_data_p; from numbox.core.bindings.abi import _call_lib_func_byval; from numbox.utils.meminfo import borrow_structref; from numbox.core.vector import make_vector; from numbox.utils.highlevel import cres_if_available; from numbox.core.bindings.utils import load_lib_path"` exits 0

**Verify:**
```bash
/home/erik/projects/numbduck/venv/bin/python -c "from numbox.utils.lowlevel import array_data_p; from numbox.core.bindings.abi import _call_lib_func_byval; from numbox.utils.meminfo import borrow_structref; from numbox.core.vector import make_vector; from numbox.utils.highlevel import cres_if_available; from numbox.core.bindings.utils import load_lib_path; print('ok')"
```
Expected: `ok`.

**Steps:**

- [ ] **Step 1: Reinstall numbox into numbduck's venv in editable mode**

```bash
/home/erik/projects/numbduck/venv/bin/pip install -e /home/erik/projects/numbox
```

- [ ] **Step 2: Run the smoke import above**

---

## Task 8: Migrate numbduck's `jit_utils.py` → `numbox.utils.lowlevel.array_data_p`

**Goal:** Delete numbduck's local `jit_utils.py`; update all callers to import `array_data_p` from numbox.

**Model:** haiku

**Files:**
- Delete: `/home/erik/projects/numbduck/numbduck/jit_utils.py`
- Modify: all numbduck files that import from `numbduck.jit_utils`

**Acceptance Criteria:**
- [ ] `numbduck/jit_utils.py` does not exist
- [ ] No remaining `numbduck.jit_utils` references in numbduck source/tests/examples
- [ ] `pytest /home/erik/projects/numbduck/test` still fully green

**Verify:**
```bash
test ! -f /home/erik/projects/numbduck/numbduck/jit_utils.py
! grep -r "numbduck.jit_utils\|from numbduck import jit_utils" /home/erik/projects/numbduck --include="*.py"
/home/erik/projects/numbduck/venv/bin/pytest /home/erik/projects/numbduck/test -v
```
Expected: file absent, no grep matches, pytest green.

**Steps:**

- [ ] **Step 1: Find all callers**

```bash
grep -rn "numbduck.jit_utils\|from numbduck import jit_utils\|jit_utils.array_data_p" /home/erik/projects/numbduck --include="*.py"
```

- [ ] **Step 2: For each caller file, replace the import**

```diff
- from numbduck.jit_utils import array_data_p
+ from numbox.utils.lowlevel import array_data_p
```

- [ ] **Step 3: Delete the file**

```bash
rm /home/erik/projects/numbduck/numbduck/jit_utils.py
```

- [ ] **Step 4: Run numbduck's test suite (with cache clear)**

```bash
/home/erik/projects/numbduck/venv/bin/python -c "import shutil, pathlib; shutil.rmtree('/home/erik/.cache/numba', ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbduck').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('/home/erik/projects/numbduck/venv').rglob('*.nbi')]; [p.unlink() for p in pathlib.Path('/home/erik/projects/numbduck/venv').rglob('*.nbc')]"
/home/erik/projects/numbduck/venv/bin/pytest /home/erik/projects/numbduck/test -v
```

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbduck add -u numbduck/jit_utils.py
git -C /home/erik/projects/numbduck add -u
git -C /home/erik/projects/numbduck commit -m "Migrate jit_utils.array_data_p to numbox.utils.lowlevel"
```

---

## Task 9: Migrate numbduck's `utils.py` to use numbox's `load_lib_path`

**Goal:** Remove numbduck's local `_load_cdll` and have `load_duckdb` use `numbox.core.bindings.utils.load_lib_path` directly.

**Model:** sonnet

**Files:**
- Modify: `/home/erik/projects/numbduck/numbduck/utils.py`

**Acceptance Criteria:**
- [ ] `_load_cdll` removed from `numbduck/utils.py`
- [ ] `load_duckdb`, `_download_libduckdb` (post-download load) call `load_lib_path` instead
- [ ] DuckDB-specific discovery (`find_duckdb_shared_lib`, `_find_standalone_libduckdb`, `_download_libduckdb`, `_has_capi_symbols`, `load_duckdb`) stays in place
- [ ] numbduck tests green

**Verify:**
```bash
! grep -n "_load_cdll" /home/erik/projects/numbduck/numbduck/utils.py
/home/erik/projects/numbduck/venv/bin/pytest /home/erik/projects/numbduck/test -v
```

**Steps:**

- [ ] **Step 1: Edit `/home/erik/projects/numbduck/numbduck/utils.py`**
  - Add `from numbox.core.bindings.utils import load_lib_path` at the top
  - Delete the `_load_cdll(path)` function (lines ~113-121)
  - Replace `_load_cdll(lib_path)` call sites with `load_lib_path(lib_path)` (inside `load_duckdb` and after `_download_libduckdb`)
  - Remove the now-unused `from ctypes import CDLL` and `import platform` if nothing else in the file uses them (re-check after edit)

- [ ] **Step 2: Run numbduck tests (with cache clear — same command as Task 8 Step 4)**

- [ ] **Step 3: Commit**

```bash
git -C /home/erik/projects/numbduck add numbduck/utils.py
git -C /home/erik/projects/numbduck commit -m "Migrate _load_cdll to numbox.core.bindings.utils.load_lib_path"
```

---

## Task 10: Migrate `ducklib.py` to use numbox's ABI helpers + `cres_if_available`

**Goal:** Remove numbduck's local ABI helpers, platform flags, and `cres` wrapper. Import replacements from numbox.

**Model:** sonnet

**Files:**
- Modify: `/home/erik/projects/numbduck/numbduck/ducklib.py`

**Acceptance Criteria:**
- [ ] No local `_emit_byval_call`, `_call_lib_func_byval`, `_call_lib_func_struct_in`, `_call_lib_func_struct_out`, `_is_win`, `_is_sysv_x86_64`, `_has_symbol`, `_unavailable`, or custom `cres` in `ducklib.py`
- [ ] New imports: `from numbox.core.bindings.abi import _call_lib_func_byval, _call_lib_func_struct_in, _call_lib_func_struct_out`; `from numbox.utils.highlevel import cres_if_available`
- [ ] Call sites using `@cres(..., if_available=True)` switch to `@cres_if_available(duckdb_lib, ...)`
- [ ] Call sites using `@cres(...)` (no `if_available`) keep using numbox's plain `cres` from `numbox.utils.highlevel`
- [ ] `_build_packed_interval` stays local (DuckDB-specific)
- [ ] numbduck tests green, including the `test_ducklib.py` scalar-struct paths

**Verify:**
```bash
! grep -nE "^def _emit_byval_call|^def _call_lib_func_byval|^def _call_lib_func_struct_in|^def _call_lib_func_struct_out|^_is_win|^_is_sysv_x86_64|^def _has_symbol|^def _unavailable|^def cres\(" /home/erik/projects/numbduck/numbduck/ducklib.py
/home/erik/projects/numbduck/venv/bin/pytest /home/erik/projects/numbduck/test -v
```

**Steps:**

- [ ] **Step 1: Edit imports at the top of `ducklib.py`** — remove `platform`, `sys` if unused, remove `FunctionType`/`get_or_insert_function` imports if only used by deleted helpers; add:

```python
from numbox.core.bindings.abi import (
    _call_lib_func_byval,
    _call_lib_func_struct_in,
    _call_lib_func_struct_out,
)
from numbox.utils.highlevel import cres as _cres, cres_if_available
```

- [ ] **Step 2: Delete the following sections from `ducklib.py`:**
  - Lines ~18-19: `_is_win`, `_is_sysv_x86_64` definitions
  - Lines ~22-30: `_has_symbol`, `_unavailable` helpers
  - Lines ~33-41: local `cres` wrapper
  - Lines ~108-113: `_resolve_sig` (now lives in numbox.core.bindings.abi — remove; if `_build_packed_interval` still needs it, re-import from numbox or inline the body)
  - Lines ~139-245: `_emit_byval_call`, `_call_lib_func_byval`, `_call_lib_func_struct_in`, `_call_lib_func_struct_out`

Replace the deleted `cres` with a module-level alias that routes `if_available=True` through the new helper:

```python
def cres(sig, if_available=False, **kwargs):
    if if_available:
        return cres_if_available(duckdb_lib, sig, **kwargs)
    return _cres(sig, **kwargs)
```

(This keeps existing `@cres(..., if_available=True)` decorator call sites untouched. Alternatively, sweep decorators directly — your choice; the former is fewer edits.)

- [ ] **Step 3: Run numbduck tests (cache clear same as Task 8)**

- [ ] **Step 4: Commit**

```bash
git -C /home/erik/projects/numbduck add numbduck/ducklib.py
git -C /home/erik/projects/numbduck commit -m "Migrate ABI helpers + cres_if_available to numbox"
```

---

## Task 11: Migrate `test/test_ducklib.py` to use numbox's bridge intrinsics

**Goal:** Remove the inline `_incref_meminfo`, `_release_meminfo`, `_deref_structref_raw_ptr`, `borrow_structref`, `export_meminfo` definitions in `test_ducklib.py` (~lines 2800-2870) and import them from `numbox.utils.meminfo`.

**Model:** sonnet

**Files:**
- Modify: `/home/erik/projects/numbduck/test/test_ducklib.py`

**Acceptance Criteria:**
- [ ] No local `@intrinsic` defs of `_incref_meminfo`, `_release_meminfo`, `_deref_structref_raw_ptr` in `test_ducklib.py`
- [ ] No local `@njit` defs of `borrow_structref`, `export_meminfo` in `test_ducklib.py`
- [ ] Imports added: `from numbox.utils.meminfo import (_incref_meminfo, _release_meminfo, _deref_structref_raw_ptr, borrow_structref, export_meminfo)`
- [ ] All `test_structref_meminfo_bridge_*` tests in `test_ducklib.py` still pass (refcount ladder, nested heap, etc.)
- [ ] `_MI_TY` local is removed (now internal to numbox)

**Verify:**
```bash
! grep -nE "^@intrinsic\s*\ndef _incref_meminfo|^def _incref_meminfo|^def borrow_structref" /home/erik/projects/numbduck/test/test_ducklib.py
/home/erik/projects/numbduck/venv/bin/pytest /home/erik/projects/numbduck/test/test_ducklib.py -v -k structref_meminfo_bridge
```
Expected: no matches, all `structref_meminfo_bridge` tests pass.

**Steps:**

- [ ] **Step 1: Add the imports near the top of `test_ducklib.py`**

- [ ] **Step 2: Delete the block from the comment `# Numba type for MemInfo pointer` through the end of `def borrow_structref` and `def export_meminfo`** (roughly lines 2813-2853; verify with grep before deletion). Also delete `def _release_meminfo` and its `@intrinsic` block below (~lines 2856-2870).

- [ ] **Step 3: Run bridge tests**

- [ ] **Step 4: Full numbduck test suite (cache clear)**

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbduck add test/test_ducklib.py
git -C /home/erik/projects/numbduck commit -m "Use numbox bridge intrinsics in test_ducklib"
```

---

## Task 12: Migrate numbduck's `vector.py` → `numbox.core.vector`

**Goal:** Delete `numbduck/vector.py` and `test/test_vector.py` (already ported to numbox in Task 6); update `examples/irr.py` to import from numbox.

**Model:** haiku

**Files:**
- Delete: `/home/erik/projects/numbduck/numbduck/vector.py`
- Delete: `/home/erik/projects/numbduck/test/test_vector.py`
- Modify: `/home/erik/projects/numbduck/examples/irr.py`

**Acceptance Criteria:**
- [ ] `numbduck/vector.py` does not exist
- [ ] `test/test_vector.py` does not exist in numbduck (coverage now lives in numbox)
- [ ] `examples/irr.py` imports `make_vector` from `numbox.core.vector`
- [ ] numbduck tests green

**Verify:**
```bash
test ! -f /home/erik/projects/numbduck/numbduck/vector.py
test ! -f /home/erik/projects/numbduck/test/test_vector.py
grep -n "from numbox.core.vector import" /home/erik/projects/numbduck/examples/irr.py
/home/erik/projects/numbduck/venv/bin/pytest /home/erik/projects/numbduck/test -v
```

**Steps:**

- [ ] **Step 1: Grep for callers**

```bash
grep -rn "numbduck.vector\|from numbduck import vector\|numbduck\.vector" /home/erik/projects/numbduck --include="*.py"
```

- [ ] **Step 2: Replace each `from numbduck.vector import make_vector` with `from numbox.core.vector import make_vector`**

- [ ] **Step 3: Delete the files**

```bash
rm /home/erik/projects/numbduck/numbduck/vector.py /home/erik/projects/numbduck/test/test_vector.py
```

- [ ] **Step 4: Run numbduck tests (cache clear same as Task 8)**

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbduck add -u
git -C /home/erik/projects/numbduck commit -m "Migrate vector to numbox.core.vector; drop duplicate tests"
```

---

## Task 13: Full verification on both projects; push feature branches

**Goal:** Run every test across both projects with a clean cache; push both feature branches to origin; spot-check CI.

**Model:** sonnet

**Files:** (none)

**Acceptance Criteria:**
- [ ] numbox `pytest test -v` all green on `feat/generalize-numbduck-generics`
- [ ] numbduck `pytest test -v` all green on `feat/use-numbox-generics`
- [ ] Both branches pushed to `origin`
- [ ] numbox CI matrix queues (push on any-branch trigger from Task 2 of CI work)

**Verify:**
```bash
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test -v --durations=20
/home/erik/projects/numbduck/venv/bin/pytest /home/erik/projects/numbduck/test -v --durations=20
gh run list --repo nelson2005/numbox --branch feat/generalize-numbduck-generics --limit 3
```

**Steps:**

- [ ] **Step 1: Clear caches in both projects**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree('/home/erik/.cache/numba', ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbduck').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('/home/erik/projects/numbox/venv').rglob('*.nbi')]; [p.unlink() for p in pathlib.Path('/home/erik/projects/numbox/venv').rglob('*.nbc')]; [p.unlink() for p in pathlib.Path('/home/erik/projects/numbduck/venv').rglob('*.nbi')]; [p.unlink() for p in pathlib.Path('/home/erik/projects/numbduck/venv').rglob('*.nbc')]"
```

- [ ] **Step 2: Run numbox full suite**

- [ ] **Step 3: Run numbduck full suite**

- [ ] **Step 4: Push branches (after explicit user approval)**

```bash
git -C /home/erik/projects/numbox push -u origin feat/generalize-numbduck-generics
git -C /home/erik/projects/numbduck push -u origin feat/use-numbox-generics
```

- [ ] **Step 5: Watch CI (numbox matrix)**

```bash
gh run list --repo nelson2005/numbox --branch feat/generalize-numbduck-generics --limit 3
```

---

## Task 14: Cut numbox 0.5.9 release; bump numbduck pin

**Goal:** Publish numbox 0.5.9 and repoint numbduck's dependency pin.

**Model:** haiku

**Files:**
- Modify: `/home/erik/projects/numbduck/pyproject.toml`
- Create: git tag `v0.5.9` on numbox main (after upstream-branch PR lands — this is the handoff point)

**Acceptance Criteria:**
- [ ] numbox tag `v0.5.9` exists on main (post-merge)
- [ ] numbduck `pyproject.toml` shows `numbox~=0.5.9`
- [ ] PyPI release workflow completes successfully

**Verify:**
```bash
grep "numbox" /home/erik/projects/numbduck/pyproject.toml
git -C /home/erik/projects/numbox tag | grep v0.5.9
```

**Steps:**

- [ ] **Step 1: OUT-OF-PLAN:** Open numbox upstream PR (from a branch based on `upstream/main` excluding CLAUDE.md + fork-only CI) and merge via GitHub after review.

- [ ] **Step 2: OUT-OF-PLAN:** Tag `v0.5.9` on numbox main; push tag; wait for release workflow to publish to PyPI.

- [ ] **Step 3: Bump numbduck pin**

Edit `/home/erik/projects/numbduck/pyproject.toml`:
```diff
-    "numbox~=0.5.8"
+    "numbox~=0.5.9"
```

- [ ] **Step 4: Reinstall into numbduck venv from PyPI; run full suite**

```bash
/home/erik/projects/numbduck/venv/bin/pip install --upgrade "numbox~=0.5.9"
/home/erik/projects/numbduck/venv/bin/pytest /home/erik/projects/numbduck/test -v
```

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbduck add pyproject.toml
git -C /home/erik/projects/numbduck commit -m "Bump numbox pin to 0.5.9"
```

---

## Self-Review

**Spec coverage:** All six generic bits identified in the upstream discussion are covered — array_data_p (T1), load_lib_path (T2), cres_if_available (T3), ABI helpers (T4), bridge intrinsics (T5), Vector (T6). Five migration tasks (T8-T12) drain the numbduck side. T7 is the environmental bridge between phases. T13 is the verification gate. T14 is the release.

**Placeholder scan:** No "TBD", "implement later", "similar to Task N" escaped review. Code blocks accompany every non-mechanical step. Mechanical migration tasks (T8, T12) give explicit `grep` + import-swap recipes instead of inlining the full source — acceptable because the engineer can read the deleted file and the replacement is a single-line import change.

**Type consistency:** `array_data_p`, `load_lib_path`, `cres_if_available`, `_incref_meminfo`, `_release_meminfo`, `_deref_structref_raw_ptr`, `borrow_structref`, `export_meminfo`, `make_vector` all match across task definitions, imports, and grep patterns. Platform flags (`_is_win`, `_is_sysv_x86_64`) defined in T4 are referenced in T10's deletion list.

**Known risks / deferred items:**
- T14's upstream PR + release steps are marked OUT-OF-PLAN. They need explicit user approval per memory "No upstream branch changes without permission" and "No remote branch deletion without approval" rules.
- Windows CI coverage for T4's ABI helpers relies on numbduck's existing suite (run post-migration in T13). numbox's own matrix doesn't have a small-struct-passing binding to test directly. Acceptable because numbduck is the only downstream consumer today; if a second consumer appears, numbox should grow a dedicated test.
- `_build_packed_interval` references `_resolve_sig` (numbduck-internal). T10 Step 2 notes this — if the engineer opts to inline the body or re-import from numbox, either path is fine; neither blocks the migration.
