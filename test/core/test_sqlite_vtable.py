import numpy as np
from numbox.core.bindings._sqlite_vtable import utf32_to_utf8, _nul_trimmed_len
from numbox.utils.lowlevel import array_data_p


def test_imports():
    from numbox.core.bindings import _sqlite_vtable as v
    assert hasattr(v.sqlite3_create_module, "py_func")
    assert hasattr(v.sqlite3_declare_vtab, "py_func")
    assert hasattr(v.sqlite3_malloc, "py_func")


def _encode(s, width):
    src = np.zeros(width, dtype=np.uint32)
    for i, ch in enumerate(s):
        src[i] = ord(ch)
    dst = np.zeros(4 * width + 1, dtype=np.uint8)
    n = utf32_to_utf8(array_data_p(src), width, array_data_p(dst))
    return bytes(dst[:n])


def test_utf32_ascii():
    assert _encode("abc", 6) == b"abc"


def test_utf32_multibyte():
    assert _encode("héllo", 6) == "héllo".encode("utf-8")


def test_utf32_emoji_4byte():
    assert _encode("a\U0001F600b", 6) == "a\U0001F600b".encode("utf-8")


def test_utf32_stops_at_nul():
    assert _encode("hi", 6) == b"hi"


def test_utf32_invalid_codepoint_replacement():
    src = np.array([0x41, 0xD800, 0x110000, 0], dtype=np.uint32)
    dst = np.zeros(64, dtype=np.uint8)
    n = utf32_to_utf8(array_data_p(src), 4, array_data_p(dst))
    assert bytes(dst[:n]) == b"A" + b"\xef\xbf\xbd" + b"\xef\xbf\xbd"


def test_nul_trimmed_len():
    buf = np.frombuffer(b"hi\x00\x00\x00", dtype=np.uint8).copy()
    assert _nul_trimmed_len(array_data_p(buf), 5) == 2
