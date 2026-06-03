import numpy as np
from numbox.core.bindings._sqlite_typemap import utf8_to_utf32, utf32_to_utf8
from numbox.utils.lowlevel import array_data_p


def _decode(s_bytes, width_cp):
    src = np.frombuffer(s_bytes, dtype=np.uint8).copy()
    dst = np.zeros(width_cp, dtype=np.uint32)
    n = utf8_to_utf32(array_data_p(src), len(src), array_data_p(dst), width_cp)
    return n, dst


def test_utf8_to_utf32_ascii():
    n, dst = _decode(b"abc", 6)
    assert n == 3 and list(dst[:3]) == [97, 98, 99] and dst[3] == 0


def test_utf8_to_utf32_multibyte():
    n, dst = _decode("héllo".encode("utf-8"), 6)
    assert list(dst[:n]) == [ord(c) for c in "héllo"]


def test_utf8_to_utf32_4byte():
    n, dst = _decode("a\U0001F600b".encode("utf-8"), 6)
    assert [int(x) for x in dst[:n]] == [ord("a"), 0x1F600, ord("b")]


def test_utf8_to_utf32_clamps_to_width():
    n, dst = _decode(b"abcdefgh", 3)
    assert n == 3 and list(dst[:3]) == [97, 98, 99]


def test_utf8_to_utf32_roundtrip():
    s = "abé\U0001F600cd"
    width_cp = 8
    cps = np.array([ord(c) for c in s] + [0] * (width_cp - len(s)), dtype=np.uint32)
    enc = np.zeros(4 * width_cp + 1, dtype=np.uint8)
    nbytes = utf32_to_utf8(array_data_p(cps), width_cp, array_data_p(enc))
    n, dst = _decode(enc[:nbytes].tobytes(), width_cp)
    assert n == len(s)
    assert [int(x) for x in dst[:n]] == [ord(c) for c in s]
