# libc bindings expansion — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the 8 chunks in [`2026-05-11-libc-bindings-expansion-design.md`](2026-05-11-libc-bindings-expansion-design.md): stdio handles, errno + thread-safe strerror, and a 25-function monomorphic libc batch.

**Architecture:** Three new private intrinsic modules (`_stdio.py`, `_errno.py`, `_strerror.py`) each owning one extern-symbol surface, plus 25 cres wrappers in `_c.py` over the existing `_call_lib_func`. All symbol references are extern declarations resolved by llvmlite's JIT linker at link time — never literal addresses baked into IR (would break `cache=True` under ASLR).

**Tech Stack:** numba 0.60.0, llvmlite, pytest, Sphinx. Python 3.10+.

**Reference paths (single source of truth):**
- Spec: [`docs/plans/2026-05-11-libc-bindings-expansion-design.md`](2026-05-11-libc-bindings-expansion-design.md)
- Extern-ref pattern: [`numbox/core/bindings/call.py:185`](../../numbox/core/bindings/call.py#L185)
- Low-level primitives (`array_data_p`, `get_str_from_p_as_int`, `get_unicode_data_p`): [`numbox/utils/lowlevel.py`](../../numbox/utils/lowlevel.py)
- `platform_` constant: [`numbox/core/bindings/utils.py:6`](../../numbox/core/bindings/utils.py#L6)
- Existing wrapper pattern: [`numbox/core/bindings/_c.py`](../../numbox/core/bindings/_c.py), [`_math.py`](../../numbox/core/bindings/_math.py)
- Existing test pattern: [`test/core/test_bindings.py`](../../test/core/test_bindings.py)

**Universal command prefixes** — these are **local to nelson2005's WSL2 dev environment** (absolute paths are deliberate per cross-project policy: absolute venv paths propagate cleanly to subagents whose CWD may not match the project root). If executing this plan from a different checkout, substitute the project-root absolute path; the credential-helper line is a workaround for one specific `~/.gitconfig` quirk and is not needed in other environments.

- Python/pytest: `/home/erik/projects/numbox/venv/bin/python` / `/home/erik/projects/numbox/venv/bin/pytest`
- Lint: `/home/erik/projects/numbox/venv/bin/flake8` (project config: `max-line-length=127`, `max-complexity=10`)
- Cache clean (run before every pytest invocation):
  ```bash
  /home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('~/.cache/numba').expanduser(), ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]"
  ```
- Git push (credential-helper workaround per `~/.gitconfig` quirk):
  ```bash
  git -C /home/erik/projects/numbox -c 'credential.https://github.com.helper=!/usr/bin/gh auth git-credential' push origin feature/libc-bindings-expansion
  ```

**Commit-message rule:** No mention of AI/Claude/Anthropic, no `Co-Authored-By` lines. Commit as the user.

---

## Task 0: Pre-flight checks

**Goal:** Confirm baseline-green starting state and that the implementer has read the prerequisite primitives.

**Files:**
- Read-only: [`numbox/utils/lowlevel.py`](../../numbox/utils/lowlevel.py) end-to-end
- Read-only: [`numbox/core/bindings/call.py`](../../numbox/core/bindings/call.py) — focus on `_call_lib_func` and the `get_or_insert_function` extern-ref at L185
- Read-only: [`docs/plans/2026-05-11-libc-bindings-expansion-design.md`](2026-05-11-libc-bindings-expansion-design.md) §§3.1–3.3 (architecture) and §9 (chunk order)

**Acceptance Criteria:**
- [ ] Baseline `pytest` passes on the current `feature/libc-bindings-expansion` HEAD
- [ ] `numba.__version__ == "0.60.0"`
- [ ] `__pycache__` and `~/.cache/numba` are clean before any later test step

**Verify:**
```bash
/home/erik/projects/numbox/venv/bin/python -c "import numba; print(numba.__version__)"
```
Expected output: `0.60.0`

**Steps:**

- [ ] **Step 1: Clean caches**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('~/.cache/numba').expanduser(), ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]"
```

- [ ] **Step 2: Confirm numba version**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import numba; print(numba.__version__)"
```
Expected: `0.60.0`

- [ ] **Step 3: Baseline pytest**

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest --durations=20
```
Expected: all green.

- [ ] **Step 4: Baseline flake8**

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/flake8
```
Expected: clean.

No commit for Task 0 — verification only.

---

## Task 1: Stdio handles intrinsic + wrappers

**Goal:** `stdout()`, `stderr()`, `stdin()` callable from `@njit`, returning the current process's `FILE*` as `intp`. Establish the `_stdio.py` file pattern and the extern-global LLVM ref idiom.

**Files:**
- Create: `numbox/core/bindings/_stdio.py`
- Create: `test/core/test_stdio_handles.py`
- Modify: `numbox/core/bindings/__init__.py` — add the re-export line

**Acceptance Criteria:**
- [ ] `stdout()`, `stderr()`, `stdin()` each return a non-zero `intp` when called from `@njit` code
- [ ] All three are cached (`cache=True`); a second pytest run loads cached objects
- [ ] No literal addresses in emitted IR (verified by inspection in Task 1's test or visually during dev)
- [ ] Platform branches present for Linux / Darwin / Windows

**Verify:**
```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_stdio_handles.py -v --durations=20
```
Expected: 3 passed (one per handle).

**Steps:**

- [ ] **Step 1: Write the failing test** — `test/core/test_stdio_handles.py`

```python
from numba import njit

from numbox.core.bindings import stdout, stderr, stdin


def test_stdout_handle_nonzero():
    @njit(cache=True)
    def get():
        return stdout()
    assert get() != 0


def test_stderr_handle_nonzero():
    @njit(cache=True)
    def get():
        return stderr()
    assert get() != 0


def test_stdin_handle_nonzero():
    @njit(cache=True)
    def get():
        return stdin()
    assert get() != 0
```

- [ ] **Step 2: Run test — confirm red**

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_stdio_handles.py -v
```
Expected: ImportError / ModuleNotFoundError on `stdout`/`stderr`/`stdin`.

- [ ] **Step 3: Implement `numbox/core/bindings/_stdio.py`**

```python
from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import intp
from numba.extending import intrinsic

from numbox.core.bindings.utils import platform_, load_lib
from numbox.utils.highlevel import cres


load_lib("c")


_DATA_SYMBOL_BY_NAME = {
    "Linux": {"stdout": "stdout", "stderr": "stderr", "stdin": "stdin"},
    "Darwin": {"stdout": "__stdoutp", "stderr": "__stderrp", "stdin": "__stdinp"},
}
_WIN_IOB_INDEX = {"stdin": 0, "stdout": 1, "stderr": 2}


def _get_or_insert_global(module, ll_ty, name):
    try:
        return module.get_global(name)
    except KeyError:
        gv = llir.GlobalVariable(module, ll_ty, name=name)
        gv.linkage = "external"
        return gv


@intrinsic(prefer_literal=True)
def _stdio_handle(typingctx, name_ty):
    if not hasattr(name_ty, "literal_value"):
        raise TypingError("_stdio_handle: name must be a literal string")
    name = name_ty.literal_value
    if name not in ("stdout", "stderr", "stdin"):
        raise TypingError(
            f"_stdio_handle: name must be one of stdout/stderr/stdin, got {name!r}"
        )

    def codegen(context, builder, signature, arguments):
        intp_ll = context.get_value_type(intp)
        ptr_ll = llir.IntType(8).as_pointer()
        if platform_ in ("Linux", "Darwin"):
            sym = _DATA_SYMBOL_BY_NAME[platform_][name]
            gv = _get_or_insert_global(builder.module, ptr_ll, sym)
            file_ptr = builder.load(gv)
            return builder.ptrtoint(file_ptr, intp_ll)
        if platform_ == "Windows":
            func_ty = llir.FunctionType(ptr_ll, [llir.IntType(32)])
            func_p = get_or_insert_function(
                builder.module, func_ty, "__acrt_iob_func")
            idx = llir.Constant(llir.IntType(32), _WIN_IOB_INDEX[name])
            file_ptr = builder.call(func_p, [idx])
            return builder.ptrtoint(file_ptr, intp_ll)
        raise RuntimeError(
            f"_stdio_handle: unsupported platform {platform_!r}")

    sig = intp(name_ty)
    return sig, codegen


@cres(intp(), cache=True)
def stdout():
    """Return the current process's stdout FILE* as intp.

    Callable from @njit. Uses the platform's extern symbol (Linux: stdout
    data global; macOS: __stdoutp data global; Windows: __acrt_iob_func(1)
    accessor) — no literal addresses, cache=True safe under ASLR.

    Windows requires UCRT (Universal C Runtime), bundled with Windows 10
    and later. Older Windows versions exposed FILE* via per-MSVC-version
    symbols (_iob, __iob_func) and are not supported.
    """
    return _stdio_handle("stdout")


@cres(intp(), cache=True)
def stderr():
    """Return the current process's stderr FILE* as intp."""
    return _stdio_handle("stderr")


@cres(intp(), cache=True)
def stdin():
    """Return the current process's stdin FILE* as intp."""
    return _stdio_handle("stdin")
```

- [ ] **Step 4: Wire re-export in `numbox/core/bindings/__init__.py`**

```python
from numbox.core.bindings._c import *  # noqa: F401, F403
from numbox.core.bindings._math import *  # noqa: F401, F403
from numbox.core.bindings._sqlite import *  # noqa: F401, F403
from numbox.core.bindings._stdio import *  # noqa: F401, F403
```

- [ ] **Step 5: Clean cache and run test — confirm green**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('~/.cache/numba').expanduser(), ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]"
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_stdio_handles.py -v --durations=20
```
Expected: 3 passed.

- [ ] **Step 6: Lint**

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_stdio.py test/core/test_stdio_handles.py numbox/core/bindings/__init__.py
```
Expected: clean.

- [ ] **Step 7: Full suite — no regressions**

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest --durations=20
```
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_stdio.py numbox/core/bindings/__init__.py test/core/test_stdio_handles.py
git -C /home/erik/projects/numbox commit -F - <<'EOF'
Add stdio handle intrinsics: stdout / stderr / stdin

Three @cres-wrapped functions callable from @njit code, each returning
the current process's FILE* as intp. The new _stdio_handle intrinsic
dispatches on platform_ at lowering time and emits extern declarations
(data globals on Linux/Darwin, accessor function call on Windows) —
never a literal address, so cache=True survives ASLR.
EOF
```

---

## Task 2: errno intrinsic + wrappers

**Goal:** `errno_get()` / `errno_set(v)` callable from `@njit`, with per-thread correctness guaranteed by re-calling the accessor at every use site.

**Files:**
- Create: `numbox/core/bindings/_errno.py`
- Create: `test/core/test_errno.py`
- Modify: `numbox/core/bindings/__init__.py` — add `_errno` re-export

**Acceptance Criteria:**
- [ ] `errno_set(v); errno_get() == v` round-trips for representative ints
- [ ] Two Python threads running `errno_set(distinct_val); time.sleep(0.05); assert errno_get() == distinct_val` do not contaminate each other
- [ ] `@njit(parallel=True)` `prange` iteration sees its own errno per worker

**Verify:**
```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_errno.py -v --durations=20
```

**Steps:**

- [ ] **Step 1: Write the failing tests** — `test/core/test_errno.py`

```python
import threading
import numpy as np
from numba import njit, prange

from numbox.core.bindings import errno_get, errno_set


def test_errno_set_get_roundtrip():
    @njit(cache=True)
    def rt(v):
        errno_set(v)
        return errno_get()
    for v in (0, 1, 2, 11, 13, 42):
        assert rt(v) == v


def test_errno_no_cross_thread_contamination():
    results = {}
    barrier = threading.Barrier(2)

    @njit(cache=True)
    def write_and_read(v):
        errno_set(v)
        return errno_get()

    def worker(tid, val):
        barrier.wait()
        results[tid] = write_and_read(val)

    t0 = threading.Thread(target=worker, args=(0, 101))
    t1 = threading.Thread(target=worker, args=(1, 202))
    t0.start(); t1.start(); t0.join(); t1.join()
    assert results[0] == 101
    assert results[1] == 202


def test_errno_prange_per_iteration_correctness():
    @njit(parallel=True, cache=True)
    def f(n):
        out = np.zeros(n, dtype=np.int32)
        for i in prange(n):
            errno_set(i)
            out[i] = errno_get()
        return out
    n = 256
    got = f(n)
    assert (got == np.arange(n, dtype=np.int32)).all()
```

- [ ] **Step 2: Run — confirm red** (ImportError on `errno_get`/`errno_set`).

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_errno.py -v
```

- [ ] **Step 3: Implement `numbox/core/bindings/_errno.py`**

```python
from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.types import int32, intp, void
from numba.extending import intrinsic

from numbox.core.bindings.utils import platform_, load_lib
from numbox.utils.highlevel import cres


load_lib("c")


_ERRNO_ACCESSOR = {
    "Linux": "__errno_location",
    "Darwin": "__error",
    "Windows": "_errno",
}


@intrinsic
def _errno_ptr(typingctx):
    def codegen(context, builder, signature, arguments):
        intp_ll = context.get_value_type(intp)
        i32_ptr = llir.IntType(32).as_pointer()
        sym = _ERRNO_ACCESSOR.get(platform_)
        if sym is None:
            raise RuntimeError(
                f"_errno_ptr: unsupported platform {platform_!r}")
        func_ty = llir.FunctionType(i32_ptr, [])
        func_p = get_or_insert_function(builder.module, func_ty, sym)
        ptr = builder.call(func_p, [])
        return builder.ptrtoint(ptr, intp_ll)
    return intp(), codegen


@intrinsic
def _load_int32_at(typingctx, p_ty):
    def codegen(context, builder, signature, arguments):
        (p,) = arguments
        i32_ptr = llir.IntType(32).as_pointer()
        ptr = builder.inttoptr(p, i32_ptr)
        return builder.load(ptr)
    return int32(p_ty), codegen


@intrinsic
def _store_int32_at(typingctx, p_ty, v_ty):
    def codegen(context, builder, signature, arguments):
        p, v = arguments
        i32_ptr = llir.IntType(32).as_pointer()
        ptr = builder.inttoptr(p, i32_ptr)
        builder.store(v, ptr)
    return void(p_ty, v_ty), codegen


@cres(int32(), cache=True)
def errno_get():
    """Return the current thread's errno as int32.

    Re-resolves the per-thread errno location on every call: on
    @njit(parallel=True) workers, the accessor returns that worker's
    errno. The Python caller cannot observe an errno value set inside a
    parallel region after return — different OS thread.
    """
    return _load_int32_at(_errno_ptr())


@cres(void(int32), cache=True)
def errno_set(v):
    """Set the current thread's errno to v."""
    _store_int32_at(_errno_ptr(), v)
```

- [ ] **Step 4: Update `__init__.py`**

```python
from numbox.core.bindings._c import *  # noqa: F401, F403
from numbox.core.bindings._math import *  # noqa: F401, F403
from numbox.core.bindings._sqlite import *  # noqa: F401, F403
from numbox.core.bindings._stdio import *  # noqa: F401, F403
from numbox.core.bindings._errno import *  # noqa: F401, F403
```

- [ ] **Step 5: Clean cache, run tests, lint**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('~/.cache/numba').expanduser(), ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]"
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_errno.py -v --durations=20
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_errno.py test/core/test_errno.py numbox/core/bindings/__init__.py
```
Expected: 3 passed, flake8 clean.

- [ ] **Step 6: Full suite**

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest --durations=20
```

- [ ] **Step 7: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_errno.py numbox/core/bindings/__init__.py test/core/test_errno.py
git -C /home/erik/projects/numbox commit -F - <<'EOF'
Add thread-safe errno_get / errno_set intrinsics

Per-thread correctness comes from re-calling the platform accessor
(__errno_location / __error / _errno) at every use site rather than
caching the pointer. Works under @njit(parallel=True) prange: each
worker thread's accessor returns that worker's errno. The Python
caller cannot observe errno values set inside a parallel region.
EOF
```

---

## Task 3: strerror_safe intrinsic + Alpine CI job

**Goal:** Unified `strerror_safe(errnum, buf, buflen)` across glibc / musl / macOS / Windows, plus an Alpine shell-only CI job that validates the musl-symbol-layout assumption.

**Files:**
- Create: `numbox/core/bindings/_strerror.py`
- Create: `test/core/test_strerror_safe.py`
- Modify: `numbox/core/bindings/__init__.py` — add `_strerror` re-export
- Modify: `.github/workflows/numbox_ci.yml` — append Alpine shell-only matrix entry

**Acceptance Criteria:**
- [ ] `strerror_safe(errno.ENOENT, buf_p, buflen) == 0` on a 128-byte buffer; recovered string is non-empty and contains "file" or "directory" (locale-tolerant: any non-empty ASCII suffices for the test)
- [ ] Short buffer (e.g. buflen=2): returns nonzero (`ERANGE` on POSIX, `ERANGE`/nonzero on Windows)
- [ ] Two-thread test: two threads each call `strerror_safe` on distinct buffers; no cross-contamination
- [ ] IR-inspection test on glibc CI: with `ll.address_of_symbol` monkeypatched to return None for `__xpg_strerror_r`, the emitted LLVM IR contains a call to `strerror_r`, NOT `__xpg_strerror_r`
- [ ] Alpine job present in CI matrix, runs `nm -D` symbol-layout assertion, gated on a matrix flag

**Verify:**
```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_strerror_safe.py -v --durations=20
```

**Steps:**

- [ ] **Step 1: Write the failing tests** — `test/core/test_strerror_safe.py`

```python
import errno
import threading

import numpy as np
import pytest
from numba import njit

from numbox.core.bindings import strerror_safe
from numbox.core.bindings.utils import platform_
from numbox.utils.lowlevel import array_data_p, get_str_from_p_as_int


@njit(cache=True)
def _describe(errnum, buf, buflen):
    buf_p = array_data_p(buf)
    rc = strerror_safe(errnum, buf_p, buflen)
    return rc, buf_p


def test_strerror_safe_enoent_roundtrip():
    buf = np.zeros(128, dtype=np.uint8)
    rc, buf_p = _describe(errno.ENOENT, buf, buf.size)
    assert rc == 0
    msg = get_str_from_p_as_int(buf_p)
    assert len(msg) > 0


def test_strerror_safe_short_buffer():
    buf = np.zeros(2, dtype=np.uint8)
    rc, _ = _describe(errno.ENOENT, buf, buf.size)
    assert rc != 0


def test_strerror_safe_two_threads_no_contamination():
    results = {}
    barrier = threading.Barrier(2)

    def worker(tid, errnum):
        buf = np.zeros(128, dtype=np.uint8)
        barrier.wait()
        rc, buf_p = _describe(errnum, buf, buf.size)
        msg = get_str_from_p_as_int(buf_p)
        results[tid] = (rc, msg)

    t0 = threading.Thread(target=worker, args=(0, errno.ENOENT))
    t1 = threading.Thread(target=worker, args=(1, errno.EACCES))
    t0.start(); t1.start(); t0.join(); t1.join()
    assert results[0][0] == 0
    assert results[1][0] == 0
    assert results[0][1] != results[1][1]


@pytest.mark.skipif(platform_ != "Linux", reason="glibc-only IR-inspection probe")
def test_strerror_safe_ir_uses_strerror_r_when_xpg_absent(monkeypatch):
    import llvmlite.binding as ll
    from numbox.core.bindings import _strerror as strerror_mod

    original = ll.address_of_symbol

    def fake(name):
        if name == "__xpg_strerror_r":
            return None
        return original(name)

    monkeypatch.setattr(ll, "address_of_symbol", fake)
    ir_text = strerror_mod._render_ir_for_probe()
    assert "strerror_r" in ir_text
    assert "__xpg_strerror_r" not in ir_text
```

- [ ] **Step 2: Run — confirm red.**

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_strerror_safe.py -v
```

- [ ] **Step 3: Implement `numbox/core/bindings/_strerror.py`**

```python
import llvmlite.binding as ll
from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.types import int32, intp
from numba.extending import intrinsic

from numbox.core.bindings.utils import platform_, load_lib
from numbox.utils.highlevel import cres


load_lib("c")


def _select_posix_symbol():
    if platform_ == "Linux":
        if ll.address_of_symbol("__xpg_strerror_r") is not None:
            return "__xpg_strerror_r"
        return "strerror_r"
    if platform_ == "Darwin":
        return "strerror_r"
    return None


@intrinsic
def _strerror_safe(typingctx, errnum_ty, buf_ty, buflen_ty):
    def codegen(context, builder, signature, arguments):
        errnum, buf_p, buflen = arguments
        i32 = llir.IntType(32)
        i8p = llir.IntType(8).as_pointer()
        size_t_ll = context.get_value_type(intp)
        if platform_ == "Windows":
            func_ty = llir.FunctionType(i32, [i8p, size_t_ll, i32])
            func_p = get_or_insert_function(
                builder.module, func_ty, "strerror_s")
            buf = builder.inttoptr(buf_p, i8p)
            return builder.call(func_p, [buf, buflen, errnum])
        sym = _select_posix_symbol()
        if sym is None:
            raise RuntimeError(
                f"_strerror_safe: unsupported platform {platform_!r}")
        func_ty = llir.FunctionType(i32, [i32, i8p, size_t_ll])
        func_p = get_or_insert_function(builder.module, func_ty, sym)
        buf = builder.inttoptr(buf_p, i8p)
        return builder.call(func_p, [errnum, buf, buflen])
    return int32(errnum_ty, buf_ty, buflen_ty), codegen


def _render_ir_for_probe():
    """Render the IR _strerror_safe would emit for a probe call.

    Used by the IR-inspection test (test_strerror_safe.py) to verify
    that when ll.address_of_symbol("__xpg_strerror_r") returns None,
    the chosen symbol is strerror_r and not __xpg_strerror_r. Bypasses
    end-to-end execution: on glibc, strerror_r is the GNU form (returns
    char*) — calling it under our POSIX-shaped IR would corrupt return
    reads. Direct text inspection is the safe verification.
    """
    module = llir.Module(name="probe")
    i32 = llir.IntType(32)
    i8p = llir.IntType(8).as_pointer()
    sym = _select_posix_symbol()
    func_ty = llir.FunctionType(i32, [i32, i8p, llir.IntType(64)])
    get_or_insert_function(module, func_ty, sym)
    return str(module)


@cres(int32(int32, intp, intp), cache=True)
def strerror_safe(errnum, buf, buflen):
    """Write the error message for errnum into buf (length buflen).

    Returns 0 on success, positive errno (ERANGE on short buffer,
    EINVAL on unknown errnum) on failure. Thread-safe on all supported
    platforms. Cross-platform dispatch happens at lowering time:
    __xpg_strerror_r on glibc, strerror_r on musl / macOS, strerror_s
    on Windows (with arg reorder).
    """
    return _strerror_safe(errnum, buf, buflen)
```

- [ ] **Step 4: Update `__init__.py`**

```python
from numbox.core.bindings._c import *  # noqa: F401, F403
from numbox.core.bindings._math import *  # noqa: F401, F403
from numbox.core.bindings._sqlite import *  # noqa: F401, F403
from numbox.core.bindings._stdio import *  # noqa: F401, F403
from numbox.core.bindings._errno import *  # noqa: F401, F403
from numbox.core.bindings._strerror import *  # noqa: F401, F403
```

- [ ] **Step 5: Add a separate Alpine shell-only job to `.github/workflows/numbox_ci.yml`**

Add a new top-level job alongside `build:` — keep it isolated from the matrix so the existing 25+ matrix entries stay untouched. Insert this block after the existing `build:` job:

```yaml
  musl_symbol_check:
    runs-on: ubuntu-latest
    container:
      image: alpine:3.20
    permissions:
      contents: read
    steps:
      - name: musl symbol layout
        shell: sh
        run: |
          apk add --no-cache binutils
          set -e
          test -e /lib/ld-musl-x86_64.so.1 || (echo "no musl on this image"; exit 1)
          LIBC=/lib/ld-musl-x86_64.so.1
          nm -D "$LIBC" | grep -E '^[0-9a-f]+ T strerror_r$' || (echo "musl strerror_r missing"; exit 1)
          if nm -D "$LIBC" | grep -q '__xpg_strerror_r'; then
            echo "musl now exports __xpg_strerror_r — revisit strerror_safe probe"; exit 1
          fi
          echo "musl libc has expected layout"
```

**Note:** This is a fork-only addition. When cherry-picking to the upstream PR branch, drop this entire job per the spec §10.2.

- [ ] **Step 6: Clean cache, run tests, lint**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('~/.cache/numba').expanduser(), ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]"
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_strerror_safe.py -v --durations=20
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_strerror.py test/core/test_strerror_safe.py numbox/core/bindings/__init__.py
```
Expected: 4 passed (or 3 + 1 skipped on non-Linux), flake8 clean.

- [ ] **Step 7: Full suite**

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest --durations=20
```

- [ ] **Step 8: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_strerror.py numbox/core/bindings/__init__.py test/core/test_strerror_safe.py .github/workflows/numbox_ci.yml
git -C /home/erik/projects/numbox commit -F - <<'EOF'
Add strerror_safe + Alpine musl symbol-layout CI check

Single POSIX-shaped surface (errnum, buf, buflen) -> int across glibc
(__xpg_strerror_r), musl / macOS (strerror_r), and Windows (strerror_s
with arg reorder). glibc-vs-musl detection probes
ll.address_of_symbol("__xpg_strerror_r") at lowering time. The Alpine
shell-only CI entry uses `nm -D` on the musl loader to assert the
expected symbol layout — cheap upfront alarm if musl ever ships
__xpg_strerror_r and the fallback path becomes wrong.

An IR-inspection test on glibc patches ll.address_of_symbol to return
None for __xpg_strerror_r and asserts the emitted IR contains a call
to strerror_r. Direct text inspection only — calling glibc's GNU
strerror_r under POSIX-shaped IR would corrupt return reads.
EOF
```

---

## Task 4: Monomorphic stdio batch (12 functions)

**Goal:** Add 12 non-variadic stdio functions to `signatures_c` and `_c.py` cres wrappers. Extend `test_stdio_handles.py` with a `capfd` roundtrip now that `fputs`/`fflush` exist. Add `test_c_stdio` to `test/core/test_bindings.py`.

**Files:**
- Modify: `numbox/core/bindings/signatures.py` — append a `# === stdio (non-variadic) ===` section to `signatures_c`
- Modify: `numbox/core/bindings/_c.py` — append 12 cres wrappers + a module docstring with caller idioms
- Modify: `test/core/test_stdio_handles.py` — add capfd roundtrip test
- Modify: `test/core/test_bindings.py` — add `test_c_stdio` (file round-trip)

**Acceptance Criteria:**
- [ ] `puts`, `fputs`, `fputc`, `putchar`, `fwrite`, `fread`, `fflush`, `fopen`, `fclose`, `feof`, `ferror`, `clearerr` all callable from `@njit`
- [ ] capfd roundtrip: `fputs("ok\n", stderr()); fflush(stderr())` produces `"ok\n"` on stderr
- [ ] File round-trip: write bytes via fopen+fwrite+fclose, read back via fopen+fread+fclose, assert byte-equality

**Verify:**
```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_stdio_handles.py test/core/test_bindings.py::test_c_stdio -v --durations=20
```

**Steps:**

- [ ] **Step 1: Write the failing tests** — extend `test/core/test_stdio_handles.py` with the capfd roundtrip:

```python
def test_stderr_fputs_roundtrip(capfd):
    from numbox.core.bindings import fputs, fflush
    from numbox.utils.lowlevel import get_unicode_data_p

    @njit(cache=True)
    def write_to_stderr():
        p = get_unicode_data_p("ok\n")
        fputs(p, stderr())
        fflush(stderr())

    write_to_stderr()
    out, err = capfd.readouterr()
    assert "ok" in err
```

And add `test_c_stdio` to `test/core/test_bindings.py`:

```python
def test_c_stdio(tmp_path):
    import numpy as np
    from numba import njit
    from numbox.core.bindings import fopen, fwrite, fclose, fread
    from numbox.utils.lowlevel import array_data_p, get_unicode_data_p

    path = tmp_path / "rt.bin"
    payload = b"hello-from-njit\x00\x01\x02"

    @njit(cache=True)
    def write_and_read(path_str, mode_w, mode_r, payload_arr, read_back):
        wpath = get_unicode_data_p(path_str)
        wmode = get_unicode_data_p(mode_w)
        rmode = get_unicode_data_p(mode_r)
        wfp = fopen(wpath, wmode)
        if wfp == 0:
            return -1, 0
        wbuf = array_data_p(payload_arr)
        nw = fwrite(wbuf, 1, payload_arr.size, wfp)
        fclose(wfp)
        rfp = fopen(wpath, rmode)
        if rfp == 0:
            return nw, -1
        rbuf = array_data_p(read_back)
        nr = fread(rbuf, 1, read_back.size, rfp)
        fclose(rfp)
        return nw, nr

    payload_arr = np.frombuffer(payload, dtype=np.uint8).copy()
    read_back = np.zeros(len(payload), dtype=np.uint8)
    nw, nr = write_and_read(str(path), "wb", "rb", payload_arr, read_back)
    assert nw == len(payload)
    assert nr == len(payload)
    assert bytes(read_back) == payload
```

- [ ] **Step 2: Run — confirm red** (ImportError on `fputs` / `fflush` / `fopen` / `fwrite` / `fclose` / `fread`).

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_stdio_handles.py::test_stderr_fputs_roundtrip test/core/test_bindings.py::test_c_stdio -v
```

- [ ] **Step 3: Add signatures** — append to `signatures_c` in `numbox/core/bindings/signatures.py`:

```python
    # === stdio (non-variadic) ===
    "puts": int32(intp),
    "fputs": int32(intp, intp),
    "fputc": int32(int32, intp),
    "putchar": int32(int32),
    "fwrite": intp(intp, intp, intp, intp),
    "fread": intp(intp, intp, intp, intp),
    "fflush": int32(intp),
    "fopen": intp(intp, intp),
    "fclose": int32(intp),
    "feof": int32(intp),
    "ferror": int32(intp),
    "clearerr": void(intp),
```

- [ ] **Step 4: Add cres wrappers** — append to `numbox/core/bindings/_c.py`:

```python
@cres(signatures.get("puts"), cache=True)
def puts(s):
    return _call_lib_func("puts", (s,))


@cres(signatures.get("fputs"), cache=True)
def fputs(s, fp):
    return _call_lib_func("fputs", (s, fp))


@cres(signatures.get("fputc"), cache=True)
def fputc(c, fp):
    return _call_lib_func("fputc", (c, fp))


@cres(signatures.get("putchar"), cache=True)
def putchar(c):
    return _call_lib_func("putchar", (c,))


@cres(signatures.get("fwrite"), cache=True)
def fwrite(ptr, size, nmemb, fp):
    return _call_lib_func("fwrite", (ptr, size, nmemb, fp))


@cres(signatures.get("fread"), cache=True)
def fread(ptr, size, nmemb, fp):
    return _call_lib_func("fread", (ptr, size, nmemb, fp))


@cres(signatures.get("fflush"), cache=True)
def fflush(fp):
    return _call_lib_func("fflush", (fp,))


@cres(signatures.get("fopen"), cache=True)
def fopen(path, mode):
    return _call_lib_func("fopen", (path, mode))


@cres(signatures.get("fclose"), cache=True)
def fclose(fp):
    return _call_lib_func("fclose", (fp,))


@cres(signatures.get("feof"), cache=True)
def feof(fp):
    return _call_lib_func("feof", (fp,))


@cres(signatures.get("ferror"), cache=True)
def ferror(fp):
    return _call_lib_func("ferror", (fp,))


@cres(signatures.get("clearerr"), cache=True)
def clearerr(fp):
    return _call_lib_func("clearerr", (fp,))
```

- [ ] **Step 5: Clean cache, re-run tests — confirm green**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('~/.cache/numba').expanduser(), ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]"
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_stdio_handles.py test/core/test_bindings.py -v --durations=20
```

- [ ] **Step 6: Lint + full suite**

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_c.py numbox/core/bindings/signatures.py test/core/test_stdio_handles.py test/core/test_bindings.py
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest --durations=20
```

- [ ] **Step 7: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/signatures.py numbox/core/bindings/_c.py test/core/test_stdio_handles.py test/core/test_bindings.py
git -C /home/erik/projects/numbox commit -F - <<'EOF'
Add 12 non-variadic stdio bindings + capfd / file-roundtrip tests

puts, fputs, fputc, putchar, fwrite, fread, fflush, fopen, fclose,
feof, ferror, clearerr. All fit _call_lib_func directly; size_t is
intp on every supported 64-bit platform, int is int32.

Tests: a capfd stderr round-trip via fputs("ok\n", stderr()); fflush
exercises stderr() end-to-end. A tmp_path file round-trip via
fopen/fwrite/fclose/fopen/fread/fclose asserts byte-equality.
EOF
```

---

## Task 5: Monomorphic strings batch (7 functions)

**Goal:** Add 7 string functions to signatures + `_c.py`. Add `test_c_strings` and `test_c_strerror` to `test/core/test_bindings.py`.

**Files:**
- Modify: `numbox/core/bindings/signatures.py` — append `# === strings ===` section
- Modify: `numbox/core/bindings/_c.py` — 7 wrappers
- Modify: `test/core/test_bindings.py` — add `test_c_strings`, `test_c_strerror`

**Acceptance Criteria:**
- [ ] `strcmp`, `strncmp`, `strchr`, `strrchr`, `strstr`, `strncpy`, `strerror` callable from `@njit`
- [ ] `test_c_strings` validates each on known inputs
- [ ] `test_c_strerror` validates pointer-nonzero and string non-empty (threading deferred to `test_strerror_safe.py`)

**Verify:**
```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_bindings.py::test_c_strings test/core/test_bindings.py::test_c_strerror -v --durations=20
```

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `test/core/test_bindings.py`:

```python
def test_c_strings():
    import numpy as np
    from numba import njit
    from numbox.core.bindings import (
        strcmp, strncmp, strchr, strrchr, strstr, strncpy,
    )
    from numbox.utils.lowlevel import (
        get_unicode_data_p, array_data_p, get_str_from_p_as_int,
    )

    @njit(cache=True)
    def run():
        a = get_unicode_data_p("hello")
        b = get_unicode_data_p("hello")
        c = get_unicode_data_p("world")
        eq = strcmp(a, b)
        ne = strcmp(a, c)
        n_eq = strncmp(a, c, 0)
        h = get_unicode_data_p("hello world")
        ord_l = 108
        first_l = strchr(h, ord_l)
        last_l = strrchr(h, ord_l)
        substr = strstr(h, get_unicode_data_p("world"))
        return eq, ne, n_eq, first_l - h, last_l - h, substr - h

    eq, ne, n_eq, off_first, off_last, off_sub = run()
    assert eq == 0
    assert ne != 0
    assert n_eq == 0
    assert off_first == 2
    assert off_last == 9
    assert off_sub == 6

    dst = np.zeros(8, dtype=np.uint8)

    @njit(cache=True)
    def copy(dst):
        src = get_unicode_data_p("abcdef")
        dst_p = array_data_p(dst)
        strncpy(dst_p, src, 6)
        return dst_p
    dst_p = copy(dst)
    assert bytes(dst[:6]) == b"abcdef"
    assert get_str_from_p_as_int(dst_p) == "abcdef"


def test_c_strerror():
    import errno
    from numba import njit
    from numbox.core.bindings import strerror
    from numbox.utils.lowlevel import get_str_from_p_as_int

    @njit(cache=True)
    def lookup(e):
        return strerror(e)
    p = lookup(errno.ENOENT)
    assert p != 0
    assert len(get_str_from_p_as_int(p)) > 0
```

- [ ] **Step 2: Run — confirm red** (ImportError on the string wrappers).

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_bindings.py::test_c_strings test/core/test_bindings.py::test_c_strerror -v
```

- [ ] **Step 3: Add signatures** — append to `signatures_c`:

```python
    # === strings ===
    "strcmp": int32(intp, intp),
    "strncmp": int32(intp, intp, intp),
    "strchr": intp(intp, int32),
    "strrchr": intp(intp, int32),
    "strstr": intp(intp, intp),
    "strncpy": intp(intp, intp, intp),
    "strerror": intp(int32),
```

- [ ] **Step 4: Add cres wrappers** — append to `_c.py`:

```python
@cres(signatures.get("strcmp"), cache=True)
def strcmp(a, b):
    return _call_lib_func("strcmp", (a, b))


@cres(signatures.get("strncmp"), cache=True)
def strncmp(a, b, n):
    return _call_lib_func("strncmp", (a, b, n))


@cres(signatures.get("strchr"), cache=True)
def strchr(s, c):
    return _call_lib_func("strchr", (s, c))


@cres(signatures.get("strrchr"), cache=True)
def strrchr(s, c):
    return _call_lib_func("strrchr", (s, c))


@cres(signatures.get("strstr"), cache=True)
def strstr(haystack, needle):
    return _call_lib_func("strstr", (haystack, needle))


@cres(signatures.get("strncpy"), cache=True)
def strncpy(dst, src, n):
    """Copy at most n bytes from src to dst (POSIX strncpy semantics).

    Does NOT guarantee null termination: if strlen(src) >= n, dst will
    contain n bytes from src with no trailing NUL. Callers that need a
    NUL-terminated result must reserve an extra byte and either pre-zero
    the buffer or explicitly write dst[n] = 0 after the call.
    """
    return _call_lib_func("strncpy", (dst, src, n))


@cres(signatures.get("strerror"), cache=True)
def strerror(errnum):
    """Return a pointer to the static error-message string for errnum.

    NOT thread-safe — the returned pointer references a per-process
    static buffer that subsequent strerror calls may overwrite. Use
    strerror_safe for thread-safe operation.
    """
    return _call_lib_func("strerror", (errnum,))
```

- [ ] **Step 5: Clean cache, re-run tests — confirm green**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('~/.cache/numba').expanduser(), ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]"
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_bindings.py::test_c_strings test/core/test_bindings.py::test_c_strerror -v --durations=20
```

- [ ] **Step 6: Lint + full suite**

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_c.py numbox/core/bindings/signatures.py test/core/test_bindings.py
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest --durations=20
```

- [ ] **Step 7: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/signatures.py numbox/core/bindings/_c.py test/core/test_bindings.py
git -C /home/erik/projects/numbox commit -F - <<'EOF'
Add 7 string bindings: strcmp / strncmp / strchr / strrchr / strstr / strncpy / strerror

strerror documented as not thread-safe in its docstring — callers
needing thread safety should use strerror_safe instead.
EOF
```

---

## Task 6: Monomorphic memory batch (5 functions)

**Goal:** Add 5 memory functions to signatures + `_c.py`. Add `test_c_memory`.

**Files:**
- Modify: `numbox/core/bindings/signatures.py` — append `# === memory ===` section
- Modify: `numbox/core/bindings/_c.py` — 5 wrappers
- Modify: `test/core/test_bindings.py` — add `test_c_memory`

**Acceptance Criteria:**
- [ ] `memcpy`, `memmove`, `memset`, `memcmp`, `memchr` callable from `@njit`
- [ ] `memmove` correctness verified on overlapping regions
- [ ] `memcmp` sign-of-return validated
- [ ] `memchr` offset-finding validated

**Verify:**
```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_bindings.py::test_c_memory -v --durations=20
```

**Steps:**

- [ ] **Step 1: Write the failing test** `test_c_memory` — append to `test/core/test_bindings.py`:

```python
def test_c_memory():
    import numpy as np
    from numba import njit
    from numbox.core.bindings import memcpy, memmove, memset, memcmp, memchr
    from numbox.utils.lowlevel import array_data_p

    src = np.arange(10, dtype=np.uint8)
    dst = np.zeros(10, dtype=np.uint8)

    @njit(cache=True)
    def do_copy(src, dst):
        return memcpy(array_data_p(dst), array_data_p(src), src.nbytes)
    do_copy(src, dst)
    assert (dst == src).all()

    overlap = np.arange(10, dtype=np.uint8).copy()

    @njit(cache=True)
    def do_move(arr):
        p = array_data_p(arr)
        return memmove(p + 2, p, 5)
    do_move(overlap)
    assert overlap[2] == 0 and overlap[6] == 4

    fill = np.zeros(8, dtype=np.uint8)

    @njit(cache=True)
    def do_set(arr):
        return memset(array_data_p(arr), 0x7F, arr.nbytes)
    do_set(fill)
    assert (fill == 0x7F).all()

    a = np.array([1, 2, 3, 4], dtype=np.uint8)
    b = np.array([1, 2, 3, 5], dtype=np.uint8)

    @njit(cache=True)
    def do_cmp(a, b):
        return memcmp(array_data_p(a), array_data_p(b), a.nbytes)
    assert do_cmp(a, b) < 0
    assert do_cmp(b, a) > 0
    assert do_cmp(a, a) == 0

    haystack = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)

    @njit(cache=True)
    def do_chr(h):
        p = array_data_p(h)
        return memchr(p, 3, h.nbytes) - p
    assert do_chr(haystack) == 3
```

- [ ] **Step 2: Run — confirm red** (ImportError on memcpy/memmove/memset/memcmp/memchr).

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_bindings.py::test_c_memory -v
```

- [ ] **Step 3: Add signatures** — append to `signatures_c`:

```python
    # === memory ===
    "memcpy": intp(intp, intp, intp),
    "memmove": intp(intp, intp, intp),
    "memset": intp(intp, int32, intp),
    "memcmp": int32(intp, intp, intp),
    "memchr": intp(intp, int32, intp),
```

- [ ] **Step 4: Add cres wrappers** — append to `_c.py`:

```python
@cres(signatures.get("memcpy"), cache=True)
def memcpy(dst, src, n):
    return _call_lib_func("memcpy", (dst, src, n))


@cres(signatures.get("memmove"), cache=True)
def memmove(dst, src, n):
    return _call_lib_func("memmove", (dst, src, n))


@cres(signatures.get("memset"), cache=True)
def memset(dst, c, n):
    return _call_lib_func("memset", (dst, c, n))


@cres(signatures.get("memcmp"), cache=True)
def memcmp(a, b, n):
    return _call_lib_func("memcmp", (a, b, n))


@cres(signatures.get("memchr"), cache=True)
def memchr(s, c, n):
    return _call_lib_func("memchr", (s, c, n))
```

- [ ] **Step 5: Clean cache, re-run — confirm green**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('~/.cache/numba').expanduser(), ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]"
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_bindings.py::test_c_memory -v --durations=20
```

- [ ] **Step 6: Lint + full suite**

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_c.py numbox/core/bindings/signatures.py test/core/test_bindings.py
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest --durations=20
```

- [ ] **Step 7: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/signatures.py numbox/core/bindings/_c.py test/core/test_bindings.py
git -C /home/erik/projects/numbox commit -F - <<'EOF'
Add 5 memory bindings: memcpy / memmove / memset / memcmp / memchr

All sizes are size_t which numbox represents as intp; the byte int
for memset / memchr is int32. Overlapping-region memmove and
sign-of-return memcmp are explicitly tested.
EOF
```

---

## Task 7: Monomorphic env (getenv)

**Goal:** Add `getenv` to signatures + `_c.py`. Add `test_c_env`.

**Files:**
- Modify: `numbox/core/bindings/signatures.py` — append `# === env ===` section
- Modify: `numbox/core/bindings/_c.py` — `getenv` wrapper
- Modify: `test/core/test_bindings.py` — add `test_c_env`

**Acceptance Criteria:**
- [ ] `getenv("PATH")` returns a non-zero pointer
- [ ] `getenv("NUMBOX_NONEXISTENT_XYZZY")` returns 0

**Verify:**
```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_bindings.py::test_c_env -v --durations=20
```

**Steps:**

- [ ] **Step 1: Write the failing test** — append to `test/core/test_bindings.py`:

```python
def test_c_env():
    from numba import njit
    from numbox.core.bindings import getenv
    from numbox.utils.lowlevel import get_unicode_data_p

    @njit(cache=True)
    def lookup(name):
        return getenv(get_unicode_data_p(name))
    assert lookup("PATH") != 0
    assert lookup("NUMBOX_NONEXISTENT_XYZZY") == 0
```

- [ ] **Step 2: Run — confirm red** (ImportError on `getenv`).

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_bindings.py::test_c_env -v
```

- [ ] **Step 3: Add signature** — append to `signatures_c`:

```python
    # === env ===
    "getenv": intp(intp),
```

- [ ] **Step 4: Add cres wrapper** — append to `_c.py`:

```python
@cres(signatures.get("getenv"), cache=True)
def getenv(name):
    """Return pointer to the value string in the process environ table.

    The returned pointer is owned by the platform environ — do NOT
    mutate, free, or assume it survives a subsequent setenv/putenv.
    Callers that need a stable Python str should copy via
    `get_str_from_p_as_int` before mutating environ.
    """
    return _call_lib_func("getenv", (name,))
```

- [ ] **Step 5: Clean cache, re-run — confirm green**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('~/.cache/numba').expanduser(), ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]"
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest test/core/test_bindings.py::test_c_env -v --durations=20
```

- [ ] **Step 6: Lint + full suite**

```bash
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_c.py numbox/core/bindings/signatures.py test/core/test_bindings.py
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/pytest --durations=20
```

- [ ] **Step 7: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/signatures.py numbox/core/bindings/_c.py test/core/test_bindings.py
git -C /home/erik/projects/numbox commit -F - <<'EOF'
Add getenv binding

POSIX getenv returns NULL for missing entries and a const char* to
the platform's environ table otherwise. Pointer-only surface; callers
that need a Python str use get_str_from_p_as_int.
EOF
```

---

## Task 8: Sphinx docs

**Goal:** Add a narrative section to `docs/numbox.core.bindings.rst` covering the three new surfaces, plus automodule blocks for `_stdio` / `_errno` / `_strerror`.

**Files:**
- Modify: `docs/numbox.core.bindings.rst`

**Acceptance Criteria:**
- [ ] New "Stdio handles, errno, and thread-safe strerror" section inserted between existing "ABI dispatch" and "Modules"
- [ ] Three new `.. automodule::` blocks for `_stdio` / `_errno` / `_strerror` mirroring the existing module's pattern
- [ ] `cd docs && make html` produces no Sphinx warnings touching the new content

**Verify:**
```bash
cd /home/erik/projects/numbox/docs && /home/erik/projects/numbox/venv/bin/python -m sphinx -b html -W --keep-going . _build/html
```
Expected: exit 0 (no warnings under `-W`).

**Steps:**

- [ ] **Step 1: Read the existing RST** to confirm exact insertion point and automodule format:

```bash
cat /home/erik/projects/numbox/docs/numbox.core.bindings.rst
```

- [ ] **Step 2: Insert narrative section** — between the "ABI dispatch" and "Modules" sections. Topics to cover (write as RST prose; concrete code samples may be cross-references to module docstrings):

  1. **Stdio handles** — why callable functions, not module-level constants (extern-ref pattern + ASLR + `cache=True`)
  2. **errno** — why per-thread, why the accessor is re-called at every use, the Python-observation caveat under `@njit(parallel=True)`
  3. **strerror_safe** — the platform mapping table from spec §3.3, the IR-inspection probe approach, the Alpine shell-only CI verification
  4. **Caller idioms** — render `log_to_stderr`, `append_to_file`, `describe_errno`, `buffer_equal` examples (cross-reference module docstrings to avoid drift)

- [ ] **Step 3: Add automodule blocks**:

```rst
numbox.core.bindings._stdio
---------------------------

.. automodule:: numbox.core.bindings._stdio
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._errno
---------------------------

.. automodule:: numbox.core.bindings._errno
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._strerror
------------------------------

.. automodule:: numbox.core.bindings._strerror
   :members:
   :show-inheritance:
   :undoc-members:
```

- [ ] **Step 4: Build docs**

```bash
cd /home/erik/projects/numbox/docs && /home/erik/projects/numbox/venv/bin/python -m sphinx -b html -W --keep-going . _build/html
```
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add docs/numbox.core.bindings.rst
git -C /home/erik/projects/numbox commit -F - <<'EOF'
Document stdio handles, errno, and strerror_safe in bindings RST

Narrative section explains why stdio is exposed as callables (extern
refs vs ASLR + cache=True), the per-thread invariant for errno, and
the platform mapping plus IR-inspection probe for strerror_safe.
Three new automodule blocks mirror the existing pattern.
EOF
```

---

## Post-implementation

After all eight tasks land on `feature/libc-bindings-expansion`:

1. Push the branch to fork.
2. Open fork PR to `nelson2005/numbox:main`. Wait for full matrix green + bot reviews.
3. After fork PR approval and merge, cherry-pick to a clean upstream branch based on `upstream/main`. Exclude per spec §10.2:
   - `CLAUDE.md`
   - `docs/plans/**`
   - The Alpine shell-only CI matrix entry
4. Open upstream PR.
5. After upstream merge, sync into fork as a separate small PR.
6. Tag `0.5.12` on the upstream merge commit.
