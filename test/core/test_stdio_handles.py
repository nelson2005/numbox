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
