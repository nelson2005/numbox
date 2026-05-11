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
    t0.start()
    t1.start()
    t0.join()
    t1.join()
    assert results[0] == 101
    assert results[1] == 202


def test_errno_prange_per_iteration_correctness():
    @njit(parallel=True, cache=True)
    def f(n):
        out = np.zeros(n, dtype=np.int32)
        for i in prange(n):
            errno_set(np.int64(i))
            out[i] = errno_get()
        return out
    n = 256
    got = f(n)
    assert (got == np.arange(n, dtype=np.int32)).all()
