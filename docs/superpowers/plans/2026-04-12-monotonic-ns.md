# monotonic_ns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a JIT-callable `monotonic_ns() -> int64` intrinsic to `numbox/utils/clock.py` with scaling/consistency validation tests.

**Architecture:** Single new module with platform-dispatched `@intrinsic` (Linux/macOS via `clock_gettime`, Windows via `QueryPerformanceCounter`). Stack-only allocation, no NRT. Tests validate correctness, linear scaling, and wall-clock consistency.

**Tech Stack:** numba `@intrinsic`, llvmlite IR, platform libc / kernel32

**Spec:** `docs/superpowers/specs/2026-04-12-monotonic-ns-design.md`

**Reference implementation:** `numbduck/examples/_jit_clock.py` (at `/home/erik/projects/numbduck/examples/_jit_clock.py`)

---

## File Structure

| File | Responsibility |
|---|---|
| `numbox/utils/clock.py` (create) | `monotonic_ns` intrinsic — platform dispatch, IR codegen |
| `test/utils/test_clock.py` (create) | All tests: correctness, scaling, consistency |

---

### Task 1: Implement monotonic_ns intrinsic with tests

**Goal:** Create `numbox/utils/clock.py` with `monotonic_ns` and basic correctness tests.

**Files:**
- Create: `numbox/utils/clock.py`
- Create: `test/utils/test_clock.py`

**Acceptance Criteria:**
- [ ] `monotonic_ns()` returns positive `int64` from `@njit` code
- [ ] Two sequential calls are monotonically non-decreasing
- [ ] Works on Linux (CI) and Windows (CI)

**Verify:** `pytest test/utils/test_clock.py::TestMonotonicNs -v` → 2 tests PASS

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `test/utils/test_clock.py`:

```python
from numba import njit


class TestMonotonicNs:
    @staticmethod
    def test_returns_positive_int64():
        """monotonic_ns must return a positive int64 nanosecond value."""
        from numbox.utils.clock import monotonic_ns

        @njit
        def _get():
            return monotonic_ns()

        result = _get()
        assert isinstance(result, int), f"expected int, got {type(result)}"
        assert result > 0, f"expected positive, got {result}"

    @staticmethod
    def test_monotonically_non_decreasing():
        """Two sequential calls must be non-decreasing."""
        from numbox.utils.clock import monotonic_ns

        @njit
        def _pair():
            t0 = monotonic_ns()
            t1 = monotonic_ns()
            return t0, t1

        t0, t1 = _pair()
        assert t1 >= t0, f"clock went backwards: {t0} -> {t1}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/utils/test_clock.py::TestMonotonicNs -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'numbox.utils.clock'`

- [ ] **Step 3: Create `numbox/utils/clock.py`**

Lift from `/home/erik/projects/numbduck/examples/_jit_clock.py` as-is. The file is self-contained — copy the entire module:

```python
"""Cross-platform monotonic nanosecond clock callable from @njit code.

Exposes a single function::

    monotonic_ns() -> int64   # nanoseconds since an unspecified epoch

Implemented as a Numba ``@intrinsic`` that emits an LLVM ``alloca`` for
the platform's time struct, calls the OS clock function into it, and
returns the result — all on the stack, with zero heap allocation and no
NRT liveness concerns.

Platform implementations
------------------------
Linux / macOS:
    Calls libc ``clock_gettime(CLOCK_MONOTONIC, &ts)`` where ``ts`` is a
    stack-allocated ``struct timespec {int64 tv_sec; int64 tv_nsec}``.
    ``clock_gettime`` is resolved via ``address_of_symbol`` from
    RTLD_DEFAULT — libc is always globally loaded.

Windows:
    Calls ``QueryPerformanceCounter`` from ``kernel32.dll``.  The
    performance-counter frequency is read once at module import via
    ctypes and baked into the IR as a compile-time constant.
    ``kernel32`` is registered with LLVM's symbol search via
    ``load_library_permanently``.
"""
import ctypes
import platform
import time

import llvmlite.binding as ll
from llvmlite import ir
from numba.core.cgutils import get_or_insert_function
from numba.core.types import int64
from numba.extending import intrinsic

_SYSTEM = platform.system()

_i32 = ir.IntType(32)
_i64 = ir.IntType(64)
_i32_0 = ir.Constant(_i32, 0)
_i32_1 = ir.Constant(_i32, 1)
_BILLION = ir.Constant(_i64, 1_000_000_000)


if _SYSTEM == "Windows":
    ll.load_library_permanently("kernel32.dll")

    _freq_buf = ctypes.c_int64(0)
    ctypes.windll.kernel32.QueryPerformanceFrequency(
        ctypes.byref(_freq_buf))
    _QPC_FREQ = int(_freq_buf.value)
    if _QPC_FREQ <= 0:
        raise RuntimeError(
            "QueryPerformanceFrequency returned a non-positive value")
    _QPC_FREQ_CONST = ir.Constant(_i64, _QPC_FREQ)

    @intrinsic
    def monotonic_ns(typingctx):
        """Stack-only monotonic clock via QueryPerformanceCounter."""
        def codegen(context, builder, signature, arguments):
            counter_ptr = builder.alloca(_i64)
            fn_ty = ir.FunctionType(_i32, [_i64.as_pointer()])
            fn = get_or_insert_function(
                builder.module, fn_ty, "QueryPerformanceCounter")
            builder.call(fn, [counter_ptr])
            ticks = builder.load(counter_ptr)
            numer = builder.mul(ticks, _BILLION)
            return builder.sdiv(numer, _QPC_FREQ_CONST)
        return int64(), codegen

else:
    _CLOCK_MONOTONIC = getattr(time, "CLOCK_MONOTONIC", 1)
    _CLK_ID = ir.Constant(_i32, _CLOCK_MONOTONIC)
    _timespec_ty = ir.LiteralStructType([_i64, _i64])

    @intrinsic
    def monotonic_ns(typingctx):
        """Stack-only monotonic clock via clock_gettime."""
        def codegen(context, builder, signature, arguments):
            ts_ptr = builder.alloca(_timespec_ty)
            fn_ty = ir.FunctionType(
                _i32, [_i32, _timespec_ty.as_pointer()])
            fn = get_or_insert_function(
                builder.module, fn_ty, "clock_gettime")
            builder.call(fn, [_CLK_ID, ts_ptr])
            sec = builder.load(builder.gep(ts_ptr, [_i32_0, _i32_0]))
            nsec = builder.load(builder.gep(ts_ptr, [_i32_0, _i32_1]))
            return builder.add(builder.mul(sec, _BILLION), nsec)
        return int64(), codegen
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/utils/test_clock.py::TestMonotonicNs -v`
Expected: 2 passed

- [ ] **Step 5: Lint**

Run: `flake8 numbox/utils/clock.py test/utils/test_clock.py`
Expected: no output (clean)

- [ ] **Step 6: Commit**

```bash
git add numbox/utils/clock.py test/utils/test_clock.py
git commit -m "add monotonic_ns JIT clock intrinsic with basic tests

Stack-only monotonic nanosecond clock callable from @njit(nogil=True).
Linux/macOS via clock_gettime, Windows via QueryPerformanceCounter.
Ref: Goykhman/numbox#7"
```

---

### Task 2: Add scaling and consistency validation tests

**Goal:** Add tests documenting linear n-dependency and sum-of-parts consistency, fulfilling Goykhman's request for "short documentation of n-dependency and consistency."

**Files:**
- Modify: `test/utils/test_clock.py`

**Acceptance Criteria:**
- [ ] Linear scaling test: `time(2n) / time(n)` ≈ 2 within tolerance
- [ ] Consistency test: JIT-measured total ≈ Python `time.monotonic_ns()` wall-clock
- [ ] Docstrings document the theory behind each validation

**Verify:** `pytest test/utils/test_clock.py -v` → 4 tests PASS

**Steps:**

- [ ] **Step 1: Write the scaling test**

Add `import math` and `import time` to the top of `test/utils/test_clock.py` (alongside the existing `from numba import njit`), then append the new test class:

```python
class TestMonotonicNsValidation:
    """Validates monotonic_ns measurement fidelity.

    These tests establish two properties requested in numbox#7:

    1. **Linear scaling (n-dependency):** Total measured time for a
       constant-cost-per-iteration loop scales linearly with n. This
       confirms the clock is measuring real elapsed time, not being
       optimized away or reordered by LLVM.

    2. **Sum-of-parts consistency:** The total time measured by
       per-iteration JIT clock deltas closely matches the wall-clock
       time from Python's time.monotonic_ns() around the entire call.
       This confirms the JIT clock and the OS clock agree.

    Both properties hold because clock_gettime / QueryPerformanceCounter
    are external functions with no LLVM optimization attributes (readnone,
    readonly, speculatable). Each call is a full compiler barrier — LLVM
    cannot reorder, hoist, eliminate, or merge clock calls.
    """

    @staticmethod
    def test_linear_scaling():
        """Total JIT-measured time scales linearly with iteration count.

        Runs a loop of math.sin calls (opaque to LLVM) at n and 2n,
        then checks that the ratio of total times is approximately 2.
        A generous tolerance (0.5x to 4x) avoids flaky CI failures
        on virtualized or throttled hardware.
        """
        from numbox.utils.clock import monotonic_ns

        @njit
        def timed_loop(n):
            total = 0
            for i in range(n):
                t0 = monotonic_ns()
                math.sin(float(i))
                t1 = monotonic_ns()
                total += t1 - t0
            return total

        n_small = 10_000
        n_large = 2 * n_small

        # Warm up JIT
        timed_loop(100)

        t_small = timed_loop(n_small)
        t_large = timed_loop(n_large)

        assert t_small > 0, f"expected positive time, got {t_small}"
        assert t_large > 0, f"expected positive time, got {t_large}"

        ratio = t_large / t_small
        assert 0.5 < ratio < 4.0, (
            f"time(2n)/time(n) = {ratio:.2f}, expected ~2.0 "
            f"(t_small={t_small}ns, t_large={t_large}ns)"
        )

    @staticmethod
    def test_wall_clock_consistency():
        """JIT-measured total approximately matches Python wall-clock.

        Runs a JIT loop that accumulates per-iteration monotonic_ns
        deltas. Compares the sum against Python's time.monotonic_ns()
        measured around the entire call. Agreement (within an order of
        magnitude) confirms the JIT clock is not being optimized away.
        """
        from numbox.utils.clock import monotonic_ns

        @njit
        def timed_loop(n):
            total = 0
            for i in range(n):
                t0 = monotonic_ns()
                math.sin(float(i))
                t1 = monotonic_ns()
                total += t1 - t0
            return total

        n = 50_000

        # Warm up JIT
        timed_loop(100)

        wall_start = time.monotonic_ns()
        jit_total = timed_loop(n)
        wall_end = time.monotonic_ns()

        wall_total = wall_end - wall_start

        assert jit_total > 0, f"JIT total must be positive, got {jit_total}"
        assert wall_total > 0, f"wall total must be positive, got {wall_total}"

        # JIT total should be same order of magnitude as wall clock.
        # JIT total may be less (excludes call overhead) or slightly
        # more (clock call cost included in JIT but not in wall edges).
        ratio = jit_total / wall_total
        assert 0.1 < ratio < 10.0, (
            f"JIT/wall ratio = {ratio:.2f}, expected ~1.0 "
            f"(jit={jit_total}ns, wall={wall_total}ns)"
        )
```

- [ ] **Step 2: Run all tests**

Run: `pytest test/utils/test_clock.py -v`
Expected: 4 passed (2 basic + 2 validation)

- [ ] **Step 3: Lint**

Run: `flake8 test/utils/test_clock.py`
Expected: clean

- [ ] **Step 4: Commit**

```bash
git add test/utils/test_clock.py
git commit -m "add scaling and consistency validation tests for monotonic_ns

Linear n-dependency and wall-clock consistency checks per numbox#7.
Docstrings document the compiler barrier theory."
```
