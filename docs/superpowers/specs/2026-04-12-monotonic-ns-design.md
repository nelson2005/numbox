# Design: `numbox.utils.clock` — JIT-callable `monotonic_ns` intrinsic

**Issue:** [Goykhman/numbox#7](https://github.com/Goykhman/numbox/issues/7)

## Summary

A new module `numbox/utils/clock.py` exposing a single function
`monotonic_ns() -> int64` callable from `@njit(nogil=True)` code. Returns
nanoseconds from a monotonic clock with zero heap allocation (stack-only).

## Motivation

numbox's existing `utils/timer.py` provides Python-level timing via
`time.perf_counter`. There is no JIT-level equivalent. Any numbox-based
project measuring performance inside `@njit` loops (e.g. numbduck's online
scoring benchmark) needs a monotonic clock callable without the GIL.

The implementation already exists in
[numbduck/examples/_jit_clock.py](https://github.com/Goykhman/numbduck/blob/main/examples/_jit_clock.py).
This design moves it into numbox as a proper utility.

## Implementation

### New file: `numbox/utils/clock.py`

Lift the existing code from numbduck essentially as-is. Structure:

- Module-level platform dispatch via `platform.system()`
- Shared IR constants: `_i32`, `_i64`, `_BILLION`, etc.
- Platform-specific `@intrinsic` definition of `monotonic_ns`

#### Linux / macOS

Calls libc `clock_gettime(CLOCK_MONOTONIC, &ts)` where `ts` is a
stack-allocated `struct timespec {int64 tv_sec; int64 tv_nsec}`.
`clock_gettime` is resolved via `address_of_symbol` from RTLD_DEFAULT
(libc is always globally loaded).

```
alloca timespec
call clock_gettime(CLOCK_MONOTONIC, &ts)
return ts.tv_sec * 1_000_000_000 + ts.tv_nsec
```

#### Windows

Calls `QueryPerformanceCounter` from `kernel32.dll`. The
performance-counter frequency is read once at module import via ctypes
and baked into the IR as a compile-time constant. `kernel32` is
registered with LLVM's symbol search via `load_library_permanently`.

```
alloca int64
call QueryPerformanceCounter(&counter)
return counter * 1_000_000_000 / frequency
```

### Why `@intrinsic` (not `@cres` + `_call_lib_func`)

The clock needs to `alloca` a stack buffer, call the OS function into it,
and read back struct fields — all in a single codegen block. The existing
`_call_lib_func` pattern only handles simple call-and-return.

### Import path

```python
from numbox.utils.clock import monotonic_ns
```

No re-export from `numbox.utils.__init__`. This avoids triggering
platform-specific side effects (Windows `kernel32.dll` load) on bare
`import numbox.utils`.

## Tests

### New file: `test/utils/test_clock.py`

#### Basic correctness

- `monotonic_ns()` returns a positive `int64`
- Two sequential calls are monotonically non-decreasing

#### Scaling and consistency (with docstring documenting the theory)

Validates the two properties Goykhman requested:

1. **Linear scaling (n-dependency):** Run a known-cost JIT loop at several
   values of `n`. Assert total JIT-measured time scales linearly — i.e.
   the ratio `time(2n) / time(n)` is approximately 2, within a generous
   tolerance for CI environments.

2. **Sum-of-parts consistency:** Assert that the total JIT-measured time
   (sum of per-iteration `monotonic_ns()` deltas) approximately matches
   the wall-clock time measured by Python's `time.monotonic_ns()` around
   the entire JIT call. This confirms the clock isn't being optimized
   away or reordered by LLVM.

Tolerances will be generous (e.g. 0.5x to 4x for scaling ratio, order-of-
magnitude for wall-clock agreement) to avoid flaky CI failures across
different hardware and virtualized environments.

## Compiler barrier properties

`clock_gettime` and `QueryPerformanceCounter` are declared as external
functions with no LLVM attributes (`readnone`, `readonly`,
`speculatable`). Each call is a full compiler barrier: LLVM cannot reorder,
hoist, eliminate, or merge clock calls. This is desirable for
microbenchmarking — it prevents the measured work from being optimized
away, similar to Google Benchmark's `DoNotOptimize()`.
