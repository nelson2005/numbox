import errno
import threading

import numpy as np
import pytest
from numba import njit

from numbox.core.bindings import strerror_safe
from numbox.core.bindings.utils import platform_
from numbox.utils.lowlevel import array_data_p, get_str_from_p_as_int


@njit(cache=True)
def _describe(errnum, buf, buflen):
    buf_p = array_data_p(buf)
    rc = strerror_safe(errnum, buf_p, buflen)
    return rc, buf_p


def test_strerror_safe_enoent_roundtrip():
    buf = np.zeros(128, dtype=np.uint8)
    rc, buf_p = _describe(errno.ENOENT, buf, buf.size)
    assert rc == 0
    msg = get_str_from_p_as_int(buf_p)
    assert len(msg) > 0


def test_strerror_safe_short_buffer():
    buf = np.zeros(2, dtype=np.uint8)
    rc, _ = _describe(errno.ENOENT, buf, buf.size)
    # POSIX strerror_r returns ERANGE on short buffer; Windows strerror_s may
    # truncate-and-succeed instead. The portable contract is NUL-termination
    # within the caller's buffer — verify that directly rather than reading
    # the buffer via a NUL-scanning helper (which could read past the
    # allocation if the implementation didn't NUL-terminate).
    assert buf[-1] == 0, f"buffer not NUL-terminated (rc={rc})"


def test_strerror_safe_two_threads_no_contamination():
    results = {}
    barrier = threading.Barrier(2)

    def worker(tid, errnum):
        buf = np.zeros(128, dtype=np.uint8)
        barrier.wait()
        rc, buf_p = _describe(errnum, buf, buf.size)
        msg = get_str_from_p_as_int(buf_p)
        results[tid] = (rc, msg)

    t0 = threading.Thread(target=worker, args=(0, errno.ENOENT))
    t1 = threading.Thread(target=worker, args=(1, errno.EACCES))
    t0.start()
    t1.start()
    t0.join()
    t1.join()
    assert results[0][0] == 0
    assert results[1][0] == 0
    assert results[0][1] != results[1][1]


@pytest.mark.skipif(platform_ != "Linux", reason="glibc-only IR-inspection probe")
def test_strerror_safe_ir_uses_strerror_r_when_xpg_absent(monkeypatch):
    import llvmlite.binding as ll
    from numbox.core.bindings import _strerror as strerror_mod

    original = ll.address_of_symbol

    def fake(name):
        if name == "__xpg_strerror_r":
            return None
        return original(name)

    monkeypatch.setattr(ll, "address_of_symbol", fake)
    ir_text = strerror_mod._render_ir_for_probe()
    assert "strerror_r" in ir_text
    assert "__xpg_strerror_r" not in ir_text
