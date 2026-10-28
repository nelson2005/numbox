import threading
import numpy as np
from numba import njit, prange

from numbox.core.bindings.errno import errno_get, errno_set


@njit(cache=True)
def _errno_set(v):
    errno_set(v)


@njit(cache=True)
def _errno_get():
    return errno_get()


def test_errno_set_get_roundtrip():
    @njit(cache=True)
    def rt(v):
        errno_set(v)
        return errno_get()
    for v in (0, 1, 2, 11, 13, 42):
        assert rt(v) == v


def test_errno_per_thread_isolation():
    """Two threads concurrently set distinct errno sentinels and read errno back after a pure-compute busy loop. Each
    thread must observe its own sentinel.

    The test uses a busy loop instead of a synchronization primitive. Any Python-side ``threading.Barrier.wait``,
    ``Event.wait`` or lock acquire goes through futex(2). The futex(2) call sets errno (typically ``EAGAIN``) in the
    calling thread before it returns to user code. That syscall would overwrite the sentinel that the thread has just
    stored. A probe based on synchronization would therefore see the syscall errno instead of the sentinel. Such a
    probe would (falsely) pass *even with thread-isolated errno*, which would invalidate the test.

    The pure-compute spin runs entirely in ``@njit(nogil=True)`` code, with no syscall between ``errno_set`` and
    ``errno_get``. One thread can influence what the other thread reads only if errno is process-global. If errno is
    TLS (the documented invariant of ``__errno_location`` and the other errno accessors), each thread reads its own
    slot. Each thread then always sees its own sentinel, regardless of scheduling. The test runs many trials so that
    the two threads overlap in wall time.
    """
    SENTINEL_A = 0xABCD
    SENTINEL_B = 0xBEEF
    N_TRIALS = 200
    BUSY = 200_000

    @njit(nogil=True, cache=True)
    def set_busy_get(v, busy):
        errno_set(v)
        s = np.int64(0)
        for i in range(busy):
            s += (i * 7) & 0xff
        # Use s in the return so LLVM cannot DCE the busy loop.
        return errno_get() + (s & 0)

    bad = {"A": 0, "B": 0, "first_bad_A": None, "first_bad_B": None}

    def worker(tid, sentinel):
        for _ in range(N_TRIALS):
            v = set_busy_get(sentinel, BUSY)
            if v != sentinel:
                bad[tid] += 1
                if bad[f"first_bad_{tid}"] is None:
                    bad[f"first_bad_{tid}"] = v

    t_a = threading.Thread(target=worker, args=("A", SENTINEL_A))
    t_b = threading.Thread(target=worker, args=("B", SENTINEL_B))
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    assert bad["A"] == 0, (
        f"thread A observed wrong errno in {bad['A']}/{N_TRIALS} trials "
        f"(first wrong value: {bad['first_bad_A']!r}, expected "
        f"{SENTINEL_A:#x}); errno is not thread-isolated"
    )
    assert bad["B"] == 0, (
        f"thread B observed wrong errno in {bad['B']}/{N_TRIALS} trials "
        f"(first wrong value: {bad['first_bad_B']!r}, expected "
        f"{SENTINEL_B:#x}); errno is not thread-isolated"
    )


def test_errno_set_get_observable_in_parallel_iterations():
    """errno_get observes the value that errno_set stored in the same iteration of a @njit(parallel=True) loop.

    The test guards against code generation that caches or hoists the errno-accessor call by accident, for example
    through a stray ``readnone`` attribute on ``__errno_location``. The test is not a thread-isolation probe. numba's
    prange assigns each thread a contiguous chunk of iterations. ``errno_set`` and ``errno_get`` within one iteration
    therefore run on the same thread, whether errno is TLS or process-global. See ``test_errno_per_thread_isolation``
    for the actual TLS probe.
    """
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
