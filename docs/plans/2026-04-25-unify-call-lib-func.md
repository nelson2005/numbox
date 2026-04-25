# Unify `_call_lib_func` — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers-extended-cc:subagent-driven-development` (recommended) or `superpowers-extended-cc:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace [`numbox/core/bindings/abi.py`](../../numbox/core/bindings/abi.py)'s `_call_lib_func_struct_in` / `_call_lib_func_struct_out` / `_call_lib_func_args_struct_out` with a single ABI-aware `_call_lib_func` in [`numbox/core/bindings/call.py`](../../numbox/core/bindings/call.py) that handles scalar args, ≤16B struct args / returns by value (per platform), and >16B struct args by pointer with the SysV-x86-64-only `byval` + `optnone` + `noinline` idiom.

**Architecture:** `_call_lib_func` classifies each arg type and the return type as scalar / struct ≤16B / struct >16B (helper `_classify` in `abi.py`), then dispatches per platform — Win x64, SysV x86-64, AAPCS64 (helper `_current_platform` in `abi.py`). Codegen builds the LLVM function type explicitly, handles `sret` for struct returns on Win x64, and applies `byval` + `optnone` + `noinline` for SysV-x86-64 >16B struct args. `_call_lib_func_byval` (the C-`func(T*)` case) is kept; the three other intrinsics are deleted outright.

**Tech Stack:** numba (≥0.60, <0.66), llvmlite, pytest. No new dependencies.

**Spec:** [`docs/specs/2026-04-25-unify-call-lib-func-design.md`](../specs/2026-04-25-unify-call-lib-func-design.md)

**Conventions baked into every task:**
- Use the project venv: `/home/erik/projects/numbox/venv/bin/python` and `/home/erik/projects/numbox/venv/bin/pytest`. Never bare `python` / `python3` / `pytest`.
- Clean cache before any pytest run (`__pycache__` + `~/.cache/numba`). Use the Python one-liner shown in each task — never `find -exec rm`.
- Never `cd`; absolute paths everywhere. For git, `git -C /home/erik/projects/numbox ...`.
- Commit each task as a single commit on `feat/unify-call-lib-func`.
- Commit messages: plain prose, no `# headings`, no AI / Claude / Anthropic / Co-Authored-By attribution. Use `git commit -F <file>` for multi-line bodies; `-m "single line"` is fine for one-liners.

---

### Task 1: Extend `_call_lib_func` to be ABI-aware

**Goal:** Make `_call_lib_func` in [`call.py`](../../numbox/core/bindings/call.py) classify each arg and the return value, then emit the right LLVM IR for the current platform — scalar, ≤16B by-value, or >16B by-pointer with `byval` + `optnone` + `noinline` on SysV x86-64. Add `_classify` and `_current_platform` helpers in [`abi.py`](../../numbox/core/bindings/abi.py). Existing `_call_lib_func_struct_in` / `_struct_out` / `_args_struct_out` stay in place untouched (Task 2 deletes them); existing tests still pass.

**Files:**
- Modify: `/home/erik/projects/numbox/numbox/core/bindings/abi.py` — add `_classify`, `_current_platform`, platform / class string constants. `_is_win`, `_struct_bytes`, `_emit_byval_call`, the four `_call_lib_func_*` intrinsics stay as-is.
- Modify: `/home/erik/projects/numbox/numbox/core/bindings/call.py` — rewrite `_call_lib_func` codegen to be ABI-aware.

**Acceptance Criteria:**
- [ ] `abi.py` exports `_classify(ty) -> str` returning one of `"scalar"`, `"struct_small"`, `"struct_large"` for any numba type. Scalars (numba `int32`, `float64`, etc.) → `"scalar"`. `Record` / `BaseTuple` of size ≤16 → `"struct_small"`. `Record` / `BaseTuple` of size >16 → `"struct_large"`.
- [ ] `abi.py` exports `_current_platform() -> str` returning one of `"win_x64"`, `"sysv_x86_64"`, `"aapcs64"`. Raises `RuntimeError` on unsupported `(sys.platform, platform.machine())` combinations.
- [ ] `_call_lib_func` raises `TypingError` if the resolved return type is `"struct_large"`.
- [ ] `_call_lib_func` correctly handles all combinations from the spec's ABI dispatch table for arg + return classes × platform.
- [ ] All existing tests in `test/core/test_abi.py` and the math/c/sqlite binding tests still pass on every CI matrix entry.

**Verify:**
```
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/core/ -v --durations=20
```
Expected: green on every existing test (no regression). Numbduck integration tests (run separately, not in numbox CI) remain green.

**Steps:**

- [ ] **Step 1: Clean cache**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')) + [pathlib.Path('/home/erik/.cache/numba')]]"
```

- [ ] **Step 2: Add helpers to `abi.py`**

Open `/home/erik/projects/numbox/numbox/core/bindings/abi.py`. Add `import platform` near the existing `import sys` line. Below the existing `_is_win = sys.platform == "win32"` line, add this block:

```python
import platform


_PLATFORM_WIN_X64 = "win_x64"
_PLATFORM_SYSV_X86_64 = "sysv_x86_64"
_PLATFORM_AAPCS64 = "aapcs64"


def _current_platform():
    """Identify the C calling convention for the current host.

    Returns one of ``_PLATFORM_WIN_X64``, ``_PLATFORM_SYSV_X86_64``,
    ``_PLATFORM_AAPCS64``. Used by the ABI-aware codegen in
    ``numbox.core.bindings.call._call_lib_func`` to pick the right
    struct-passing convention.
    """
    if sys.platform == "win32":
        return _PLATFORM_WIN_X64
    machine = platform.machine()
    if machine in ("x86_64", "AMD64"):
        return _PLATFORM_SYSV_X86_64
    if machine in ("arm64", "aarch64", "AARCH64"):
        return _PLATFORM_AAPCS64
    raise RuntimeError(
        f"Unsupported platform for ABI dispatch: "
        f"{sys.platform}/{machine}"
    )


_CLASS_SCALAR = "scalar"
_CLASS_STRUCT_SMALL = "struct_small"
_CLASS_STRUCT_LARGE = "struct_large"


def _classify(ty):
    """Classify a numba type for ABI dispatch.

    Returns one of:

    - ``_CLASS_SCALAR`` — any non-struct numba type (e.g. ``int32``,
      ``float64``, pointers represented as ``intp``).
    - ``_CLASS_STRUCT_SMALL`` — ``Record`` / ``BaseTuple`` of size
      ≤ 16 bytes; passed by value on register-passing ABIs.
    - ``_CLASS_STRUCT_LARGE`` — ``Record`` / ``BaseTuple`` of size
      > 16 bytes; passed by pointer with ``byval`` on SysV x86-64,
      by pointer (no special attribute) on other ABIs.
    """
    if not isinstance(ty, (nb_types.Record, nb_types.BaseTuple)):
        return _CLASS_SCALAR
    if isinstance(ty, nb_types.Record):
        size = ty.size
    else:
        size = sum(t.bitwidth for t in ty.types) // 8
    return _CLASS_STRUCT_SMALL if size <= 16 else _CLASS_STRUCT_LARGE
```

The existing `_is_win`, `_struct_bytes`, `_emit_byval_call`, and the four `_call_lib_func_*` intrinsics are not touched in this task.

- [ ] **Step 3: Rewrite `call.py`**

Replace the entire contents of `/home/erik/projects/numbox/numbox/core/bindings/call.py` with:

```python
import llvmlite.binding as ll
from llvmlite import ir as llir

from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import NoneType, Tuple, UniTuple
from numba.extending import intrinsic

from numbox.core.bindings.abi import (
    _CLASS_SCALAR, _CLASS_STRUCT_SMALL, _CLASS_STRUCT_LARGE,
    _PLATFORM_AAPCS64, _PLATFORM_SYSV_X86_64, _PLATFORM_WIN_X64,
    _classify, _current_platform,
)
from numbox.core.bindings.signatures import signatures


@intrinsic(prefer_literal=True)
def _call_lib_func(typingctx, func_name_ty, args_ty=NoneType):
    """Call a C library function with ABI-correct argument and return passing.

    The C function name is resolved from numbox's ``signatures`` dict.
    Each arg in ``args_ty`` and the resolved return type are classified
    as scalar / struct ≤16 bytes / struct >16 bytes, then lowered to
    LLVM IR per the host's calling convention:

    - **Scalar args / returns** — passed and returned directly.
    - **≤16-byte struct args** — by value on SysV x86-64 and AAPCS64
      (LLVM's frontend lowers to register passing); by pointer (alloca
      + store + pass-pointer) on Windows x64.
    - **>16-byte struct args** — by pointer on every platform; on SysV
      x86-64 the ``byval`` attribute is added to the LLVM arg and the
      enclosing function gets ``optnone`` + ``noinline`` so the LLVM
      optimizer does not elide the caller-side stack copy before the
      callee reads it. See:
      https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    - **≤16-byte struct returns** — direct on SysV x86-64 and AAPCS64;
      via ``sret`` (caller-allocated hidden first arg, void return) on
      Windows x64.
    - **>16-byte struct returns** — raise ``TypingError``. No consumer
      currently needs this; add it when one does.

    For C signatures of form ``func(T*)`` (pointer to struct) rather
    than ``func(T)`` lowered to a byval pointer by the ABI, use
    ``_call_lib_func_byval`` from ``numbox.core.bindings.abi`` instead.
    The numba type system can't disambiguate ``T`` from ``T*``; the
    caller picks the intrinsic based on what the C header declares.
    """
    func_name = func_name_ty.literal_value
    func_p_as_int = ll.address_of_symbol(func_name)
    if func_p_as_int is None:
        raise RuntimeError(f"{func_name} is unavailable in the LLVM context")
    func_sig = signatures.get(func_name, None)
    if func_sig is None:
        raise ValueError(f"Undefined signature for {func_name}")

    ret_ty = func_sig.return_type
    ret_class = _classify(ret_ty)
    if ret_class == _CLASS_STRUCT_LARGE:
        raise TypingError(
            f"_call_lib_func: return struct >16 bytes is unsupported "
            f"({func_name})"
        )

    if args_ty == NoneType:
        arg_types = ()
        arg_classes = ()
    else:
        assert isinstance(args_ty, (Tuple, UniTuple))
        arg_types = tuple(args_ty)
        arg_classes = tuple(_classify(at) for at in arg_types)

    plat = _current_platform()
    use_sret = (ret_class == _CLASS_STRUCT_SMALL and plat == _PLATFORM_WIN_X64)

    def codegen(context, builder, signature, arguments):
        if args_ty == NoneType:
            arg_vals = ()
        else:
            _, args_pack = arguments
            arg_vals = tuple(
                builder.extract_value(args_pack, i)
                for i in range(len(arg_types))
            )

        ret_ll_ty = context.get_value_type(ret_ty)

        ll_arg_tys = []
        ll_arg_vals = []
        byval_arg_indices = []

        if use_sret:
            sret_ptr = builder.alloca(ret_ll_ty)
            ll_arg_tys.append(ret_ll_ty.as_pointer())
            ll_arg_vals.append(sret_ptr)
        else:
            sret_ptr = None

        for arg_ty, arg_cls, val in zip(arg_types, arg_classes, arg_vals):
            arg_ll_ty = context.get_value_type(arg_ty)
            if arg_cls == _CLASS_SCALAR:
                ll_arg_tys.append(arg_ll_ty)
                ll_arg_vals.append(val)
                continue
            pass_by_value = (
                arg_cls == _CLASS_STRUCT_SMALL
                and plat in (_PLATFORM_SYSV_X86_64, _PLATFORM_AAPCS64)
            )
            if pass_by_value:
                ll_arg_tys.append(arg_ll_ty)
                ll_arg_vals.append(val)
                continue
            stack_p = builder.alloca(arg_ll_ty)
            builder.store(val, stack_p)
            ll_arg_tys.append(arg_ll_ty.as_pointer())
            ll_arg_vals.append(stack_p)
            if arg_cls == _CLASS_STRUCT_LARGE and plat == _PLATFORM_SYSV_X86_64:
                byval_arg_indices.append(len(ll_arg_vals) - 1)

        if use_sret:
            func_ll_ty = llir.FunctionType(llir.VoidType(), ll_arg_tys)
        else:
            func_ll_ty = llir.FunctionType(ret_ll_ty, ll_arg_tys)
        func_p = get_or_insert_function(builder.module, func_ll_ty, func_name)

        if use_sret:
            func_p.args[0].add_attribute("sret")
        for idx in byval_arg_indices:
            func_p.args[idx].add_attribute("byval")
        if byval_arg_indices:
            builder.function.attributes.add("optnone")
            builder.function.attributes.add("noinline")

        if use_sret:
            builder.call(func_p, ll_arg_vals)
            return builder.load(sret_ptr)
        return builder.call(func_p, ll_arg_vals)

    sig = ret_ty(func_name_ty, args_ty)
    return sig, codegen
```

- [ ] **Step 4: Run the existing test suite**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')) + [pathlib.Path('/home/erik/.cache/numba')]]"
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/core/ -v --durations=20
```

Expected: every existing test passes. The math/c/sqlite bindings (`test_math.py`, `test_bindings.py`) exercise the scalar-only path through the rewritten `_call_lib_func` and will fail loudly if it regresses. `test_abi.py` exercises both helpers' presence and the lldiv struct-return path through the still-existing `_call_lib_func_args_struct_out` (Task 2 retires that).

If any test fails, STOP and report BLOCKED with the full traceback. Do not improvise fixes. Likely culprits: typo in the `_classify` BaseTuple size formula, mis-shifted byval index when sret is in front, missing `as_pointer()` on a struct-arg LLVM type, or `_current_platform()` raising on the matrix's macOS x86_64 entry (it shouldn't — `platform.machine()` returns `"x86_64"` there).

- [ ] **Step 5: Write the commit message to a file**

Use the Write tool to create `/tmp/task1_commit_msg.txt` with this body:

```
Extend _call_lib_func to be ABI-aware

Adds _classify and _current_platform helpers in abi.py, then rewrites
call.py's _call_lib_func codegen to dispatch per arg class and
platform: scalar pass-through, small-struct by value on SysV / AAPCS64
or by pointer on Windows, large-struct by pointer (with byval +
optnone + noinline on SysV x86-64), small-struct return direct on
SysV / AAPCS64 or via sret on Windows. Large-struct returns raise
TypingError; no consumer needs them yet.

The legacy _call_lib_func_struct_in / _struct_out / _args_struct_out
intrinsics are left in place untouched in this commit; the next
commit deletes them.
```

- [ ] **Step 6: Stage and commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/abi.py numbox/core/bindings/call.py
git -C /home/erik/projects/numbox commit -F /tmp/task1_commit_msg.txt
```

Do not push — the controller handles pushing.

---

### Task 2: Delete redundant ABI intrinsics

**Goal:** Now that `_call_lib_func` covers every case the legacy intrinsics did, delete `_call_lib_func_struct_in`, `_call_lib_func_struct_out`, `_call_lib_func_args_struct_out` and `_is_win` from `abi.py`. Update `test/core/test_abi.py` to (a) assert those symbols are gone, (b) drop their imports, and (c) rename + rewrite the existing lldiv test to invoke the unified intrinsic.

**Files:**
- Modify: `/home/erik/projects/numbox/numbox/core/bindings/abi.py` — delete the three intrinsics + `_is_win`.
- Modify: `/home/erik/projects/numbox/test/core/test_abi.py` — update import-presence test, rename + rewrite lldiv test.

**Acceptance Criteria:**
- [ ] `_call_lib_func_struct_in`, `_call_lib_func_struct_out`, `_call_lib_func_args_struct_out`, `_is_win` are no longer attributes of `numbox.core.bindings.abi`.
- [ ] `test_abi_imports` asserts both presence (survivors) and absence (retired names).
- [ ] `test_call_lib_func_lldiv_via_unified` (renamed from `test_call_lib_func_args_struct_out_lldiv`) calls `_call_lib_func("lldiv", (10, 3))` directly and asserts `(quot, rem) == (3, 1)`.
- [ ] `pytest test/core/test_abi.py -v` is green on every CI matrix entry.

**Verify:**
```
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/core/test_abi.py -v --durations=20
```
Expected: 4 passed.

**Steps:**

- [ ] **Step 1: Clean cache**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')) + [pathlib.Path('/home/erik/.cache/numba')]]"
```

- [ ] **Step 2: Trim `abi.py`**

Open `/home/erik/projects/numbox/numbox/core/bindings/abi.py`. Delete:

- The line `_is_win = sys.platform == "win32"` (currently around line 29).
- The intrinsic `_call_lib_func_struct_in` (and its `@intrinsic(prefer_literal=True)` decorator).
- The intrinsic `_call_lib_func_struct_out`.
- The intrinsic `_call_lib_func_args_struct_out`.

Keep `_resolve_sig`: it is still called by the surviving `_call_lib_func_byval` intrinsic. Keep `from numbox.core.bindings.signatures import signatures` for the same reason — `_resolve_sig` reads from it.

Verify what's left in `abi.py`:

- Module docstring.
- Imports: `import platform`, `import sys`, `from llvmlite import ir`, `from llvmlite.ir import FunctionType`, `from numba.core import types as nb_types`, `from numba.core.cgutils import get_or_insert_function`, `from numba.core.errors import TypingError`, `from numba.extending import intrinsic`, `from numbox.core.bindings.signatures import signatures`.
- `_PLATFORM_*` constants and `_current_platform`.
- `_CLASS_*` constants and `_classify`.
- `_resolve_sig`.
- `_struct_bytes`.
- `_emit_byval_call`.
- `_call_lib_func_byval` (intrinsic).

- [ ] **Step 3: Update `test/core/test_abi.py`**

Replace the entire contents of `/home/erik/projects/numbox/test/core/test_abi.py` with:

```python
import pytest


def test_abi_imports():
    """The surviving symbols are present and the retired ones are gone."""
    from numbox.core.bindings import abi

    assert hasattr(abi, "_emit_byval_call")
    assert hasattr(abi, "_call_lib_func_byval")
    assert hasattr(abi, "_struct_bytes")
    assert hasattr(abi, "_classify")
    assert hasattr(abi, "_current_platform")

    for retired in (
        "_call_lib_func_struct_in",
        "_call_lib_func_struct_out",
        "_call_lib_func_args_struct_out",
        "_is_win",
    ):
        assert not hasattr(abi, retired), (
            f"{retired} should have been removed"
        )


def test_struct_bytes_supports_all_struct_types():
    """The struct-size helper used by the ABI codegen handles every
    numba struct-shaped type: Tuple, UniTuple, NamedTuple (via .types),
    and Record (via .size)."""
    import collections
    from numba.core import types
    from numbox.core.bindings.abi import _struct_bytes

    assert _struct_bytes(
        types.Tuple([types.int32, types.int32, types.int64]), "t") == 16
    assert _struct_bytes(
        types.UniTuple(types.int32, 4), "t") == 16

    MyNT = collections.namedtuple("MyNT", ["a", "b"])
    assert _struct_bytes(
        types.NamedTuple([types.int32, types.int64], MyNT), "t") == 12

    rec = types.Record.make_c_struct([("a", types.int32), ("b", types.int64)])
    assert _struct_bytes(rec, "t") == 16  # 4 + 4 pad + 8


def test_struct_bytes_rejects_non_struct_type():
    """Scalar or otherwise non-struct types raise a clean TypingError."""
    from numba.core import types
    from numba.core.errors import TypingError
    from numbox.core.bindings.abi import _struct_bytes

    with pytest.raises(TypingError, match="struct-shaped type"):
        _struct_bytes(types.int32, "_call_lib_func_struct_in")


def test_call_lib_func_lldiv_via_unified():
    """End-to-end: call libc ``lldiv(10, 3)`` via the unified intrinsic
    and validate the 16-byte ``lldiv_t`` return value.

    Exercises the return-side ABI path on whatever platform the test
    runs on: SysV x86-64 and AAPCS64 read ``lldiv_t`` back from GP
    registers; Windows x64 reads it from a caller-allocated ``sret``
    slot. A regression on any of those three ABIs surfaces as a wrong
    quot or rem here.
    """
    from numba import njit
    from numbox.core.bindings import _c  # ensures libc is loaded  # noqa: F401
    from numbox.core.bindings.call import _call_lib_func

    @njit
    def run():
        return _call_lib_func("lldiv", (10, 3))

    quot, rem = run()
    assert quot == 3
    assert rem == 1
```

The notes block in the original test file (about coverage gaps for struct-IN) is dropped; numbduck integration tests cover the >16B struct-IN path end-to-end and the new IR-inspection tests in Task 3 cover small/large struct-IN codegen.

- [ ] **Step 4: Run the test suite**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')) + [pathlib.Path('/home/erik/.cache/numba')]]"
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/core/ -v --durations=20
```

Expected: every existing test still passes plus the four updated `test_abi.py` tests. The math/c/sqlite tests in particular must remain green — they go through `_call_lib_func` for scalar args / scalar returns, which is the regression-prone path.

If anything fails, STOP and report BLOCKED with the traceback. Likely culprits: forgot to drop a `from numbox.core.bindings.abi import _is_win` import in some other file (grep it: `grep -rn "_is_win\|_call_lib_func_struct\|_call_lib_func_args_struct" /home/erik/projects/numbox/numbox /home/erik/projects/numbox/test`); typo in the `hasattr` retired-names list.

- [ ] **Step 5: Commit**

```
git -C /home/erik/projects/numbox add numbox/core/bindings/abi.py test/core/test_abi.py
git -C /home/erik/projects/numbox commit -m "Delete redundant ABI intrinsics, migrate lldiv test"
```

---

### Task 3: Add ABI dispatch tests

**Goal:** Add three new tests to `test/core/test_abi.py`: a regression guard for the scalar-arg path, and two IR-inspection tests confirming `byval` + `optnone` + `noinline` are added on SysV x86-64 for >16B struct args and absent for ≤16B struct args.

**Files:**
- Modify: `/home/erik/projects/numbox/test/core/test_abi.py` — append new tests after `test_call_lib_func_lldiv_via_unified`.

**Acceptance Criteria:**
- [ ] `test_call_lib_func_scalar_args_unchanged` calls `_call_lib_func("cos", (0.0,))` (libm) inside `@njit`, asserts result is `1.0`.
- [ ] `test_call_lib_func_byval_attribute_in_ir_for_large_struct` asserts that on SysV x86-64, the LLVM IR for a function calling `_call_lib_func` with a 24-byte struct arg has `byval` on the corresponding parameter and `optnone` + `noinline` on the enclosing function. Skipped on Windows and AAPCS64.
- [ ] `test_call_lib_func_no_byval_attribute_for_small_struct` asserts that on SysV x86-64, a 16-byte struct arg does NOT carry `byval`. Skipped on Windows (different code path) and AAPCS64.
- [ ] `pytest test/core/test_abi.py -v` is green on every CI matrix entry; the SysV-only tests are skipped (not failed) on Windows and ARM matrix entries.

**Verify:**
```
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/core/test_abi.py -v --durations=20
```
Expected (on x86_64 Linux): 7 passed. On Windows or ARM: 5 passed, 2 skipped.

**Steps:**

- [ ] **Step 1: Clean cache**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')) + [pathlib.Path('/home/erik/.cache/numba')]]"
```

- [ ] **Step 2: Append the new tests to `test/core/test_abi.py`**

Append the following block to the end of `/home/erik/projects/numbox/test/core/test_abi.py`:

```python
def test_call_lib_func_scalar_args_unchanged():
    """Regression guard: scalar args + scalar return path goes through
    `_call_lib_func` unchanged from the pre-unification behavior.

    `cos(0.0)` from libm returns `1.0`. If the rewrite of `_call_lib_func`
    broke the scalar path that math / c / sqlite bindings depend on,
    this fails with an LLVM IR error or a wrong return value.
    """
    from numba import njit
    from numbox.core.bindings import _math  # ensures libm is loaded  # noqa: F401
    from numbox.core.bindings.call import _call_lib_func

    @njit
    def run():
        return _call_lib_func("cos", (0.0,))

    assert run() == 1.0


def _register_test_symbol(name):
    """Register a no-op address under ``name`` so ``ll.address_of_symbol``
    finds something for the IR-inspection tests. The body is never
    executed — the tests only inspect the LLVM IR emitted at compile
    time. Returns the ctypes wrapper, which the caller must keep alive
    for the symbol to remain valid.
    """
    import ctypes
    import llvmlite.binding as ll

    @ctypes.CFUNCTYPE(ctypes.c_int32)
    def _stub():
        return 0

    addr = ctypes.cast(_stub, ctypes.c_void_p).value
    ll.add_symbol(name, addr)
    return _stub


@pytest.fixture
def patch_signature():
    """Add a temporary entry to ``signatures`` and remove it after.

    Yields a function ``register(name, sig)`` that the test calls to
    install a fake signature. The fixture undoes the install on teardown,
    even if the install replaced an existing entry.
    """
    from numbox.core.bindings.signatures import signatures

    sentinel = object()
    saved = []

    def register(name, sig):
        saved.append((name, signatures.get(name, sentinel)))
        signatures[name] = sig

    yield register

    for name, prev in saved:
        if prev is sentinel:
            del signatures[name]
        else:
            signatures[name] = prev


def _platform_str():
    from numbox.core.bindings.abi import _current_platform
    return _current_platform()


@pytest.mark.skipif(
    _platform_str() != "sysv_x86_64",
    reason="byval + optnone + noinline are SysV x86-64 specific",
)
def test_call_lib_func_byval_attribute_in_ir_for_large_struct(patch_signature):
    """On SysV x86-64, a 24-byte struct arg is lowered with ``byval``
    on the LLVM parameter and ``optnone`` + ``noinline`` on the
    enclosing function. The actual C function is never called — the
    test only inspects the IR emitted by numba.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_byval_large_24b"
    keepalive = _register_test_symbol(name)
    big_struct = nb_types.UniTuple(nb_types.int64, 3)
    patch_signature(name, nb_types.int32(big_struct))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.UniTuple(nb_types.int64, 3),))
    ir_text = list(run.inspect_llvm().values())[0]

    assert "byval" in ir_text, (
        "expected 'byval' attribute on >16B struct arg on SysV x86-64;\n"
        f"IR was:\n{ir_text}"
    )
    assert "optnone" in ir_text, (
        "expected 'optnone' on enclosing function on SysV x86-64"
    )
    assert "noinline" in ir_text, (
        "expected 'noinline' on enclosing function on SysV x86-64"
    )
    del keepalive


@pytest.mark.skipif(
    _platform_str() != "sysv_x86_64",
    reason="≤16B-struct passing differs across ABIs; SysV-specific check",
)
def test_call_lib_func_no_byval_attribute_for_small_struct(patch_signature):
    """On SysV x86-64, a ≤16B struct arg is passed by value in
    registers; LLVM lowers without a ``byval`` attribute and without
    forcing ``optnone`` / ``noinline``.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_byval_small_16b"
    keepalive = _register_test_symbol(name)
    small_struct = nb_types.UniTuple(nb_types.int64, 2)
    patch_signature(name, nb_types.int32(small_struct))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.UniTuple(nb_types.int64, 2),))
    ir_text = list(run.inspect_llvm().values())[0]

    assert "byval" not in ir_text, (
        "did not expect 'byval' on ≤16B struct arg on SysV x86-64;\n"
        f"IR was:\n{ir_text}"
    )
    assert "optnone" not in ir_text, (
        "did not expect 'optnone' on enclosing function for ≤16B struct"
    )
    del keepalive
```

- [ ] **Step 3: Run the suite**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')) + [pathlib.Path('/home/erik/.cache/numba')]]"
/home/erik/projects/numbox/venv/bin/pytest /home/erik/projects/numbox/test/core/test_abi.py -v --durations=20
```

Expected on the dev host (Linux x86_64): `7 passed` — the four kept/migrated tests from Task 2 plus the three new ones.

If the IR-inspection assertions fail because the IR text doesn't have the substrings, STOP and report BLOCKED with the actual IR text the assertion message captured. Likely culprits:
- `byval` shows up as `byval(<type>)` in newer LLVM dialects — substring match still works.
- `optnone` and `noinline` may appear in the function's `attributes #N = { ... }` table rather than inline in the function header; substring match still works.
- The `_register_test_symbol` keepalive is dropping out of scope before `inspect_llvm()` resolves — the `del keepalive` is at the end of the test for that reason; if the issue persists, hold the keepalive in a module-level cache.

- [ ] **Step 4: Commit**

```bash
git -C /home/erik/projects/numbox add test/core/test_abi.py
git -C /home/erik/projects/numbox commit -m "Add ABI dispatch tests for unified _call_lib_func"
```

---

## Self-review notes (plan author)

- **Spec coverage:**
  - API change → Task 1.
  - Deletions → Task 2.
  - Module organization (helpers in `abi.py`, dispatch in `call.py`) → Tasks 1 + 2.
  - ABI dispatch table → covered by Task 1's codegen logic; verified by Task 3's IR inspection (SysV x86-64 paths) and Task 2's lldiv test (return-side ABI on whichever platform CI is running).
  - Tests → Task 2 (kept + migrated) + Task 3 (new).
  - Branch and commit plan → header + per-task commit step.
- **Type consistency:** `_classify` returns string constants `"scalar"`, `"struct_small"`, `"struct_large"` and exposes them as `_CLASS_SCALAR`/`_CLASS_STRUCT_SMALL`/`_CLASS_STRUCT_LARGE`. `_current_platform` returns `"win_x64"` / `"sysv_x86_64"` / `"aapcs64"` exposed as `_PLATFORM_*` constants. `call.py` imports both sets of constants and uses them by name; tests use `_current_platform()` directly.
- **Phase B (numbduck migration)** is out of scope per the spec.
