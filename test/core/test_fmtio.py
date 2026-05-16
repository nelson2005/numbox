"""Tests for the variadic printf/fprintf/snprintf intrinsics in
numbox.core.bindings._fmtio.

Coverage:
- Basic round-trips via capfd / buffer-write
- C ABI default-argument promotion (float32 -> double, int8/16 -> int32)
- Pointer-as-string via %s
- Truncation detection on snprintf
- TypingError on non-literal format strings
- TypingError on non-tuple args
- Empty args tuple
- @njit(cache=True) survives a subprocess round-trip (no cres_cacheable
  indirection — these intrinsics emit a direct libc extern call, JIT linker
  resolves per process)
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from numba import njit
from numba.core.errors import TypingError

from numbox.core.bindings import (
    fflush,
    fprintf,
    printf,
    snprintf,
    stderr,
    stdout,
)
from numbox.core.bindings.utils import platform_
from numbox.utils.lowlevel import array_data_p, get_unicode_data_p


# stdout is block-buffered when not a terminal (e.g. under pytest's capfd
# redirection), so each test helper flushes after printf so pytest's
# capture sees the output before the test returns. stderr is unbuffered
# but we flush it too for consistency / future-proofing.


@njit(cache=True)
def _printf_int(n):
    rc = printf("got %d\n", (n,))
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_no_args():
    rc = printf("just literal\n", ())
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_float64(x):
    rc = printf("%.3f\n", (x,))
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_float32(x):
    # float32 must be promoted to double for %f
    rc = printf("%.3f\n", (np.float32(x),))
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_int8():
    rc = printf("[%d %d]\n", (np.int8(-7), np.int8(42)))
    fflush(stdout())
    return rc


@njit(cache=True)
def _printf_string(s_p):
    rc = printf("hi %s!\n", (s_p,))
    fflush(stdout())
    return rc


@njit(cache=True)
def _fprintf_stderr(n):
    rc = fprintf(stderr(), "err %d\n", (n,))
    fflush(stderr())
    return rc


@njit(cache=True)
def _fprintf_stdout(n):
    rc = fprintf(stdout(), "out %d\n", (n,))
    fflush(stdout())
    return rc


@njit(cache=True)
def _snprintf_into(buf, lo, hi):
    return snprintf(array_data_p(buf), buf.size, "[%d:%d]", (lo, hi))


@njit(cache=True)
def _snprintf_no_args(buf):
    return snprintf(array_data_p(buf), buf.size, "literal", ())


# Non-ASCII format-string helpers. "café=" + "%d" + "\n": the 'é' is U+00E9,
# which encodes as the two-byte UTF-8 sequence b"\xc3\xa9". The expected
# rendered bytes for n=42 are b"caf\xc3\xa9=42\n" (9 bytes). With ASCII
# encoding these helpers would have raised UnicodeEncodeError at compile
# time, so the very fact that they compile + execute is the load-bearing
# proof that the format-string encoding is UTF-8.
NON_ASCII_FMT = "café=%d\n"
NON_ASCII_EXPECTED = "café=42\n".encode("utf-8")  # b'caf\xc3\xa9=42\n'


@njit(cache=True)
def _printf_utf8(n):
    rc = printf(NON_ASCII_FMT, (n,))
    fflush(stdout())
    return rc


@njit(cache=True)
def _fprintf_utf8(n):
    rc = fprintf(stdout(), NON_ASCII_FMT, (n,))
    fflush(stdout())
    return rc


@njit(cache=True)
def _snprintf_utf8(buf, n):
    return snprintf(array_data_p(buf), buf.size, NON_ASCII_FMT, (n,))


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_int_roundtrip(capfd):
    rc = _printf_int(42)
    out, err = capfd.readouterr()
    assert out == "got 42\n", repr(out)
    assert rc == len("got 42\n")
    assert err == ""


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_no_args(capfd):
    rc = _printf_no_args()
    out, _ = capfd.readouterr()
    assert out == "just literal\n"
    assert rc == len("just literal\n")


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_float64(capfd):
    _printf_float64(3.14159)
    out, _ = capfd.readouterr()
    assert out == "3.142\n", repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_float32_is_promoted_to_double(capfd):
    """C ABI default-arg promotion: float32 passed to %f must be widened to
    double before being placed in the variadic call. If the binding didn't
    do the fpext, libc printf would read 64 bits of which the high 32 are
    garbage — output would be wildly wrong."""
    _printf_float32(3.14159)
    out, _ = capfd.readouterr()
    # float32(3.14159) rounds to ~3.1415927; %.3f truncates to "3.142".
    # Allow either "3.141" or "3.142" depending on rounding; both prove the
    # value made it through cleanly (i.e. the float32 was promoted, not
    # read as garbage bits).
    assert out.strip() in ("3.141", "3.142"), repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_int8_is_promoted_to_int32(capfd):
    """C ABI default-arg promotion: int8/int16 must be widened to int32.
    Without it, libc would read 32 bits where only 8 were written and the
    high bits would be garbage."""
    _printf_int8()
    out, _ = capfd.readouterr()
    assert out == "[-7 42]\n", repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_string_via_pointer(capfd):
    """%s with a NUL-terminated string pointer (intp). At the variadic ABI
    level, an int64 and a pointer occupy the same register/slot, so passing
    the intp through unchanged works — libc's %s handler interprets the bits
    as char*."""
    s_p = get_unicode_data_p("world")
    _printf_string(s_p)
    out, _ = capfd.readouterr()
    assert out == "hi world!\n", repr(out)


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_fprintf_to_stderr(capfd):
    rc = _fprintf_stderr(13)
    out, err = capfd.readouterr()
    assert err == "err 13\n", repr(err)
    assert out == "", f"unexpected stdout: {out!r}"
    assert rc == len("err 13\n")


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_fprintf_to_stdout(capfd):
    """fprintf(stdout(), ...) must land on stdout, not stderr — guards
    against transposed FILE* handles."""
    _fprintf_stdout(99)
    out, err = capfd.readouterr()
    assert out == "out 99\n", repr(out)
    assert err == ""


def test_snprintf_basic():
    """A fits-in-buffer call writes the expected bytes and returns the
    written-count (excluding NUL). This part of the contract is identical
    across all platforms."""
    buf = np.zeros(64, dtype=np.uint8)
    rc = _snprintf_into(buf, 7, 11)
    assert rc == len("[7:11]"), rc
    nul = buf.tolist().index(0)
    assert bytes(buf[:nul]) == b"[7:11]"


def test_snprintf_no_args():
    buf = np.zeros(32, dtype=np.uint8)
    rc = _snprintf_no_args(buf)
    assert rc == len("literal")
    nul = buf.tolist().index(0)
    assert bytes(buf[:nul]) == b"literal"


def test_snprintf_truncation_detection():
    """snprintf truncation behavior diverges by platform — see the
    snprintf docstring in numbox/core/bindings/_fmtio.py.

    - **Linux/macOS** (POSIX/C99 ``snprintf``): ``rc`` is the would-have-
      written count (excluding NUL); ``rc >= size`` signals truncation;
      the buffer is always NUL-terminated when ``size > 0``.
    - **Windows** (MSVCRT ``_snprintf``, what numba's ``cgutils.snprintf``
      resolves to): ``rc < 0`` signals truncation; the buffer is NOT
      guaranteed to be NUL-terminated. The would-have-written count is
      not recoverable.

    The portable check ``(rc < 0) or (rc >= size)`` works on both."""
    buf = np.full(5, 0xFF, dtype=np.uint8)  # 0xFF pre-fill catches no-write
    rc = _snprintf_into(buf, 12345, 67890)
    full_msg = b"[12345:67890]"  # 13 bytes

    # Portable truncation signal — must hold on every platform.
    truncated = (rc < 0) or (rc >= buf.size)
    assert truncated, (
        f"expected truncation indicator: rc={rc}, buf.size={buf.size}"
    )

    if platform_ == "Windows":
        # MSVCRT _snprintf returns -1 on truncation; no NUL-term guarantee.
        assert rc == -1, f"Windows _snprintf returns -1 on truncation; got {rc}"
        # buf[-1] is not guaranteed to be 0; don't assert on it.
    else:
        # POSIX/C99 snprintf returns the would-have-written count and
        # always NUL-terminates within the buffer.
        assert rc == len(full_msg), rc
        assert buf[-1] == 0, (
            f"snprintf must NUL-terminate on POSIX; got {buf.tolist()!r}"
        )
        truncated_prefix = bytes(buf[:buf.size - 1])
        assert truncated_prefix == full_msg[:buf.size - 1], (
            truncated_prefix, full_msg
        )


def test_printf_non_literal_format_raises():
    """The format string MUST be a Python str literal at the call site —
    numba surfaces it as a Literal[str] type. A runtime-built unicode is
    rejected at typing time with a clean TypingError naming the binding.
    """
    @njit
    def caller(fmt):
        return printf(fmt, (1,))

    with pytest.raises(TypingError, match=r"printf.*literal"):
        caller("dynamic %d")


def test_fprintf_non_literal_format_raises():
    @njit
    def caller(fmt):
        return fprintf(stderr(), fmt, (1,))

    with pytest.raises(TypingError, match=r"fprintf.*literal"):
        caller("dynamic %d")


def test_snprintf_non_literal_format_raises():
    @njit
    def caller(buf, fmt):
        return snprintf(array_data_p(buf), buf.size, fmt, (1,))

    buf = np.zeros(32, dtype=np.uint8)
    with pytest.raises(TypingError, match=r"snprintf.*literal"):
        caller(buf, "dynamic %d")


def test_printf_non_tuple_args_raises():
    """args MUST be a tuple. Passing a bare scalar (or anything that's not
    a BaseTuple type at typing time) raises a clean TypingError."""
    @njit
    def caller():
        return printf("got %d\n", 42)  # bare scalar instead of (42,)

    with pytest.raises(TypingError, match=r"printf.*tuple"):
        caller()


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_printf_accepts_utf8_format_literal(capfd):
    """The format string is encoded as UTF-8 at codegen time, so non-ASCII
    literals compile cleanly and render as UTF-8 bytes through libc printf.
    With the prior ASCII encoding this would have raised UnicodeEncodeError
    at numbox-compile time (when numba lowers the @njit caller)."""
    _printf_utf8(42)
    out, _ = capfd.readouterr()
    # capfd's readouterr() decodes stdout bytes as utf-8 by default, so we
    # get back the original codepoints. Compare against the str form AND
    # the underlying UTF-8 byte sequence to pin both layers.
    assert out == "café=42\n", repr(out)
    assert out.encode("utf-8") == NON_ASCII_EXPECTED


@pytest.mark.skipif(
    platform_ == "Windows",
    reason="capfd does not reliably capture C-level stdio writes on Windows",
)
def test_fprintf_accepts_utf8_format_literal(capfd):
    _fprintf_utf8(42)
    out, _ = capfd.readouterr()
    assert out == "café=42\n", repr(out)
    assert out.encode("utf-8") == NON_ASCII_EXPECTED


def test_snprintf_accepts_utf8_format_literal():
    """snprintf gives us byte-level access to the written buffer, so we
    can assert the UTF-8 byte sequence directly without any decoding
    indirection (capfd is not involved)."""
    buf = np.zeros(32, dtype=np.uint8)
    rc = _snprintf_utf8(buf, 42)
    # rc semantics differ by platform (see snprintf docstring), but for a
    # message that fits comfortably in the buffer, both platforms return
    # the byte-count written (excluding NUL).
    assert rc == len(NON_ASCII_EXPECTED), (rc, len(NON_ASCII_EXPECTED))
    nul = buf.tolist().index(0)
    assert bytes(buf[:nul]) == NON_ASCII_EXPECTED, bytes(buf[:nul])


def test_fmtio_caller_survives_subprocess_round_trip(tmp_path):
    """@njit(cache=True) callers of printf/snprintf survive a process
    restart: the IR cached on disk references the libc extern symbol and a
    deterministic format-string global constant, never a runtime address.
    Re-running in a warm process re-uses the cached .nbc (mtime preserved).
    """
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent("""
        import numpy as np
        from numba import njit
        from numbox.core.bindings import snprintf, printf
        from numbox.utils.lowlevel import array_data_p

        @njit(cache=True)
        def go():
            buf = np.zeros(32, dtype=np.uint8)
            n = snprintf(array_data_p(buf), buf.size, "[%d]", (42,))
            return n

        v = go()
        assert v == 4, v
        print(v)
    """))
    cache_dir = tmp_path / "numba-cache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}

    r1 = subprocess.run(
        [sys.executable, str(probe)], env=env, capture_output=True, text=True,
    )
    assert r1.returncode == 0, f"cold run failed:\n{r1.stderr}"
    assert r1.stdout.strip() == "4"

    nbc_after_cold = sorted(cache_dir.rglob("*.nbc"))
    assert nbc_after_cold, f"no .nbc files written under {cache_dir}"
    mtimes_cold = {p: p.stat().st_mtime_ns for p in nbc_after_cold}

    r2 = subprocess.run(
        [sys.executable, str(probe)], env=env, capture_output=True, text=True,
    )
    assert r2.returncode == 0, f"warm run failed:\n{r2.stderr}"
    assert r2.stdout.strip() == "4"

    nbc_after_warm = sorted(cache_dir.rglob("*.nbc"))
    assert nbc_after_warm == nbc_after_cold
    for p in nbc_after_warm:
        assert p.stat().st_mtime_ns == mtimes_cold[p], (
            f"warm run rewrote {p} — cache was not hit"
        )
