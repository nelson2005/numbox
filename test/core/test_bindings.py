import errno

import numpy as np
import pytest
from ctypes import addressof, c_char_p, c_int64, c_void_p
from numba import njit
from numbox.core.bindings import *
from numbox.core.bindings.utils import platform_
from numbox.utils.lowlevel import array_data_p, get_unicode_data_p, get_str_from_p_as_int
from test.auxiliary_utils import collect_and_run_tests, str_from_p_as_int


@njit(cache=True)
def _write_and_read(path_str, mode_w, mode_r, payload_arr, read_back):
    wpath = get_unicode_data_p(path_str)
    wmode = get_unicode_data_p(mode_w)
    rmode = get_unicode_data_p(mode_r)
    wfp = fopen(wpath, wmode)
    if wfp == 0:
        return -1, 0
    wbuf = array_data_p(payload_arr)
    nw = fwrite(wbuf, 1, payload_arr.size, wfp)
    fclose(wfp)
    rfp = fopen(wpath, rmode)
    if rfp == 0:
        return nw, -1
    rbuf = array_data_p(read_back)
    nr = fread(rbuf, 1, read_back.size, rfp)
    fclose(rfp)
    return nw, nr


def test_c():
    srand(2)
    r1 = rand()
    r2 = rand()
    assert r1 > 0
    assert r2 > 0

    s_ = "another random string"
    s = c_char_p(s_.encode())
    s_p = c_void_p.from_buffer(s).value
    assert strlen(s_p) == len(s_)


@pytest.mark.skipif(platform_ == "Windows", reason="Need to add windows support")
def test_sqlite():
    version_ = sqlite3_libversion_number()
    version_ = str_from_p_as_int(version_)
    assert "." in version_

    db_name_ = ":memory:"
    db_name = c_char_p(db_name_.encode())
    db_name_p = c_void_p.from_buffer(db_name).value

    assert str_from_p_as_int(db_name_p) == db_name_
    db_p = c_int64(0)
    assert db_p.value == 0
    db_pp = addressof(db_p)
    rc = sqlite3_open(db_name_p, db_pp)
    assert rc == 0, "could not open db connection"
    assert db_p.value != 0
    db_p = db_p.value
    rc = sqlite3_close(db_p)
    assert rc == 0, "could not close db connection"


def test_load_lib_path_returns_handle_with_known_symbol():
    from numbox.core.bindings.utils import load_lib_path

    if platform_ == "Windows":
        from ctypes.util import find_msvcrt
        lib_path = find_msvcrt()
    else:
        from ctypes.util import find_library
        lib_path = find_library("m")
    if lib_path is None:
        pytest.skip("No suitable math/C runtime library discoverable")
    lib = load_lib_path(lib_path)
    assert hasattr(lib, "cos")


def test_c_stdio(tmp_path):
    path = tmp_path / "rt.bin"
    payload = b"hello-from-njit\x00\x01\x02"
    payload_arr = np.frombuffer(payload, dtype=np.uint8).copy()
    read_back = np.zeros(len(payload), dtype=np.uint8)
    nw, nr = _write_and_read(str(path), "wb", "rb", payload_arr, read_back)
    assert nw == len(payload)
    assert nr == len(payload)
    assert bytes(read_back) == payload


@njit(cache=True)
def _strings_compare_search():
    a = get_unicode_data_p("hello")
    b = get_unicode_data_p("hello")
    c = get_unicode_data_p("world")
    eq = strcmp(a, b)
    ne = strcmp(a, c)
    n_eq = strncmp(a, c, 0)
    h = get_unicode_data_p("hello world")
    ord_l = np.int32(108)
    first_l = strchr(h, ord_l)
    last_l = strrchr(h, ord_l)
    substr = strstr(h, get_unicode_data_p("world"))
    return eq, ne, n_eq, first_l - h, last_l - h, substr - h


@njit(cache=True)
def _strings_copy(dst):
    src = get_unicode_data_p("abcdef")
    dst_p = array_data_p(dst)
    strncpy(dst_p, src, 6)
    return dst_p


def test_c_strings():
    eq, ne, n_eq, off_first, off_last, off_sub = _strings_compare_search()
    assert eq == 0
    assert ne != 0
    assert n_eq == 0
    assert off_first == 2
    assert off_last == 9
    assert off_sub == 6

    dst = np.zeros(8, dtype=np.uint8)
    dst_p = _strings_copy(dst)
    assert bytes(dst[:6]) == b"abcdef"
    assert get_str_from_p_as_int(dst_p) == "abcdef"


@njit(cache=True)
def _strerror_lookup_enoent():
    return strerror(np.int32(errno.ENOENT))


def test_c_strerror():
    p = _strerror_lookup_enoent()
    assert p != 0
    assert len(get_str_from_p_as_int(p)) > 0


@njit(cache=True)
def _mem_do_copy(src, dst):
    return memcpy(array_data_p(dst), array_data_p(src), src.nbytes)


@njit(cache=True)
def _mem_do_move(arr):
    p = array_data_p(arr)
    return memmove(p + 2, p, 5)


@njit(cache=True)
def _mem_do_set(arr):
    return memset(array_data_p(arr), np.int32(0x7F), arr.nbytes)


@njit(cache=True)
def _mem_do_cmp(a, b):
    return memcmp(array_data_p(a), array_data_p(b), a.nbytes)


@njit(cache=True)
def _mem_do_chr(h):
    p = array_data_p(h)
    return memchr(p, np.int32(3), h.nbytes) - p


def test_c_memory():
    src = np.arange(10, dtype=np.uint8)
    dst = np.zeros(10, dtype=np.uint8)
    _mem_do_copy(src, dst)
    assert (dst == src).all()

    overlap = np.arange(10, dtype=np.uint8).copy()
    _mem_do_move(overlap)
    assert overlap[2] == 0 and overlap[6] == 4

    fill = np.zeros(8, dtype=np.uint8)
    _mem_do_set(fill)
    assert (fill == 0x7F).all()

    a = np.array([1, 2, 3, 4], dtype=np.uint8)
    b = np.array([1, 2, 3, 5], dtype=np.uint8)
    assert _mem_do_cmp(a, b) < 0
    assert _mem_do_cmp(b, a) > 0
    assert _mem_do_cmp(a, a) == 0

    haystack = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
    assert _mem_do_chr(haystack) == 3


if __name__ == "__main__":
    collect_and_run_tests(__name__)
