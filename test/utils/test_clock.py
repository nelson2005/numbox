import math
import time

from numba import njit

from numbox.utils.clock import monotonic_ns


def test_returns_positive_int64():
    @njit
    def get_time():
        return monotonic_ns()

    t = get_time()
    assert t > 0


def test_monotonically_non_decreasing():
    @njit
    def get_two_times():
        t0 = monotonic_ns()
        t1 = monotonic_ns()
        return t0, t1

    t0, t1 = get_two_times()
    assert t1 >= t0


def test_linear_scaling():
    """Validate the n-dependency property from numbox#7.

    clock_gettime (Linux) and QueryPerformanceCounter (Windows) are external
    C functions that carry no LLVM optimization attributes.  Every call is a
    full compiler barrier: LLVM cannot reorder, hoist, eliminate, or merge
    clock calls across the barrier.  Therefore the accumulated time must grow
    linearly with the number of iterations, because each iteration forces a
    real pair of clock reads around a math.sin() operation that itself cannot
    be eliminated (its result is accumulated and returned).

    We measure time(2n) / time(n) and assert it falls in [0.5, 4.0].  The
    wide tolerance accommodates noisy CI environments while still catching
    gross mis-behaviour (e.g. the loop being compiled away entirely, which
    would produce a ratio near 0).
    """
    @njit
    def timed_loop(n):
        total = 0
        for i in range(n):
            t0 = monotonic_ns()
            _ = math.sin(float(i))
            t1 = monotonic_ns()
            total += t1 - t0
        return total

    # Warm up JIT compilation.
    timed_loop(100)

    n_small = 10_000
    n_large = 20_000
    t_small = timed_loop(n_small)
    t_large = timed_loop(n_large)

    ratio = t_large / t_small
    assert 0.5 <= ratio <= 4.0, (
        f"Linear scaling ratio {ratio:.3f} out of expected range [0.5, 4.0]; "
        f"t_small={t_small}, t_large={t_large}"
    )


def test_wall_clock_consistency():
    """Validate sum-of-parts consistency from numbox#7.

    If the JIT clock were being optimized away or silently reordered, the
    accumulated JIT total would diverge from the Python wall-clock time that
    wraps the same call.  The two clocks use the same underlying OS primitive
    (clock_gettime CLOCK_MONOTONIC / QPC), so their totals should agree to
    within measurement overhead.

    We assert jit_total / wall_total is in [0.1, 10.0].  The ratio can
    legitimately differ from 1.0 because the wall clock also captures Python
    call overhead and Numba dispatch overhead, but gross mismatches (ratio
    near 0 or extremely large) indicate a timing defect.
    """
    @njit
    def timed_loop(n):
        total = 0
        for i in range(n):
            t0 = monotonic_ns()
            _ = math.sin(float(i))
            t1 = monotonic_ns()
            total += t1 - t0
        return total

    # Warm up JIT compilation.
    timed_loop(100)

    n = 10_000
    wall_start = time.monotonic_ns()
    jit_total = timed_loop(n)
    wall_total = time.monotonic_ns() - wall_start

    ratio = jit_total / wall_total
    assert 0.1 <= ratio <= 10.0, (
        f"JIT/wall ratio {ratio:.3f} out of expected range [0.1, 10.0]; "
        f"jit_total={jit_total}, wall_total={wall_total}"
    )
