from contextlib import ExitStack
from ctypes import c_char

from numba import njit
from numba.core.errors import UnsupportedBytecodeError
import pytest

from numbox.utils.cstrings import c_string
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


def test_ascii_round_trip():
    with c_string("hello") as p:
        assert p != 0
        assert str_from_p_as_int(p) == "hello"


def test_utf8_round_trip():
    s = "café — 你好 — 🚀"
    with c_string(s) as p:
        assert p != 0
        # Read the UTF-8 bytes back up to the NUL terminator.
        roundtrip = str_from_p_as_int(p)
        assert roundtrip == s


def test_empty_string_returns_pointer_to_nul():
    with c_string("") as p:
        assert p != 0
        first_byte = c_char.from_address(p).value
        assert first_byte == b"\x00"


def test_nul_terminator_present():
    s = "abc"
    with c_string(s) as p:
        # Index 3 is the position after "abc", and the byte there must be NUL.
        assert c_char.from_address(p + len(s.encode("utf-8"))).value == b"\x00"


def test_embedded_nul_rejected():
    """Embedded NUL bytes would silently truncate the C string at the first NUL, and the truncation would hide bugs in
    callers that pass untrusted input. c_string therefore rejects such a string before it creates the buffer.
    """
    with pytest.raises(ValueError, match="embedded NUL"):
        with c_string("foo\x00bar"):
            pass


def test_distinct_strings_get_distinct_pointers():
    with ExitStack() as stack:
        a_p = stack.enter_context(c_string("first"))
        b_p = stack.enter_context(c_string("second"))
        assert a_p != b_p
        assert str_from_p_as_int(a_p) == "first"
        assert str_from_p_as_int(b_p) == "second"


def test_context_manager_in_njit_raises():
    """Numba does not support arbitrary context managers. The test guards against a future numba version that silently
    accepts such a context manager and miscompiles it. If this test fails because numba starts to support such context
    managers, revisit the JIT caveat in the cstrings module docstring.

    The test's intent is to assert that compilation fails with numba's ``UnsupportedBytecodeError``. The regression
    that the test would detect raises no exception: numba silently accepts the context manager and miscompiles it.
    """
    @njit
    def kernel():
        with c_string("x") as p:
            return p

    with pytest.raises(UnsupportedBytecodeError):
        kernel()


if __name__ == "__main__":
    collect_and_run_tests(__name__)
