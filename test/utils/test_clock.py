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

    clock_gettime (Linux) and QueryPerformanceCounter (Windows) are external C functions that carry no LLVM
    optimization attributes.  Every call is a full compiler barrier: LLVM cannot reorder, hoist, eliminate, or merge
    clock calls across the barrier.  The accumulated time must therefore grow linearly with the number of iterations,
    because each iteration forces a pair of clock reads around a math.sin() operation.  That operation itself cannot
    be eliminated, because its result is accumulated and returned.

    The test measures time(2n) / time(n), averages it over multiple runs, and asserts that the average falls in
    [1.5, 3.0] (expected value 2.0).  Averaging reduces the noise from CI VMs enough for tight bounds, and the bounds
    still detect gross misbehavior.  For example, if the loop were compiled away entirely, the ratio would be near 1.0
    regardless of n.
    """
    @njit
    def timed_loop(n):
        total_ns = 0
        sink = 0.0
        for i in range(n):
            t0 = monotonic_ns()
            sink += math.sin(float(i))
            t1 = monotonic_ns()
            total_ns += t1 - t0
        return total_ns, sink

    # Warm up JIT compilation.
    timed_loop(100)

    n_small = 10_000
    n_large = 20_000
    runs = 5
    ratio_sum = 0.0
    for _ in range(runs):
        t_small, _ = timed_loop(n_small)
        t_large, _ = timed_loop(n_large)
        assert t_small > 0, f"expected positive time, got t_small={t_small}"
        assert t_large > 0, f"expected positive time, got t_large={t_large}"
        ratio_sum += t_large / t_small

    ratio = ratio_sum / runs
    assert 1.5 <= ratio <= 3.0, (
        f"Linear scaling ratio {ratio:.3f} out of expected range [1.5, 3.0]; "
        f"averaged over {runs} runs"
    )


def test_wall_clock_consistency():
    """Validate sum-of-parts consistency from numbox#7.

    If the JIT clock were being optimized away or silently reordered, the accumulated JIT total would diverge from the
    Python wall-clock time that wraps the same call.  The test uses time.perf_counter_ns() for the wall clock, because
    that function uses QueryPerformanceCounter on all CPython versions.  On CPython 3.10-3.12, time.monotonic_ns()
    uses GetTickCount64 (~15 ms resolution) and can return zero for fast loops.  CPython 3.13+ uses QPC for
    time.monotonic_ns(), but perf_counter_ns() works consistently across all versions.

    The test asserts that jit_total / wall_total is in [0.1, 1.0].  The JIT total cannot exceed the wall total,
    because the wall clock wraps the entire call.  A ratio near 0 would indicate that the JIT clock is being optimized
    away.
    """
    @njit
    def timed_loop(n):
        total_ns = 0
        sink = 0.0
        for i in range(n):
            t0 = monotonic_ns()
            sink += math.sin(float(i))
            t1 = monotonic_ns()
            total_ns += t1 - t0
        return total_ns, sink

    # Warm up JIT compilation.
    timed_loop(100)

    n = 50_000
    wall_start = time.perf_counter_ns()
    jit_total, _ = timed_loop(n)
    wall_total = time.perf_counter_ns() - wall_start

    assert wall_total > 0, "wall clock elapsed is zero — loop too fast for timer resolution"
    ratio = jit_total / wall_total
    assert 0.1 <= ratio <= 1.0, (
        f"JIT/wall ratio {ratio:.3f} out of expected range [0.1, 1.0]; "
        f"jit_total={jit_total}, wall_total={wall_total}"
    )
