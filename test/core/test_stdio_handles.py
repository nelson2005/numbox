from numba import njit

from numbox.core.bindings import stdout, stderr, stdin, fputs, fflush
from numbox.utils.lowlevel import get_unicode_data_p


@njit(cache=True)
def _write_to_stderr():
    p = get_unicode_data_p("ok\n")
    fputs(p, stderr())
    fflush(stderr())


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


def test_stderr_fputs_roundtrip(capfd):
    _write_to_stderr()
    out, err = capfd.readouterr()
    assert err == "ok\n"
