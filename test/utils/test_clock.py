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
