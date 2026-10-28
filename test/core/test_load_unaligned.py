import numpy as np
from numba import njit
from numba.core.types import int32, int64, float64
from numbox.utils.lowlevel import load_unaligned, store_unaligned, load_at, store_at, array_data_p


def test_load_unaligned_reads_misaligned_int64():
    # The test stores an int64 at byte offset 1 of a byte buffer, so the value is deliberately misaligned.
    buf = np.zeros(16, dtype=np.uint8)
    buf[1:9] = np.frombuffer(np.int64(0x0123456789ABCDEF).tobytes(), dtype=np.uint8)

    @njit
    def read(base):
        return load_unaligned(base + 1, int64)

    assert read(array_data_p(buf)) == 0x0123456789ABCDEF


def test_load_unaligned_matches_load_at_aligned():
    a = np.array([7, -3, 11], dtype=np.int32)

    @njit
    def read(base, i):
        return load_unaligned(base + 4 * i, int32)

    base = array_data_p(a)
    assert [read(base, i) for i in range(3)] == [7, -3, 11]


def test_load_unaligned_float64():
    x = np.array([3.5], dtype=np.float64)

    @njit
    def read(base):
        return load_unaligned(base, float64)

    assert read(array_data_p(x)) == 3.5


def test_store_unaligned_emits_align1_not_natural():
    # A runtime round trip cannot distinguish store_unaligned from store_at on x86. The hardware tolerates a misaligned
    # access that is declared naturally aligned, so both pass. The test therefore checks the contract in the emitted
    # IR: store_unaligned must emit an align=1 store. store_at on a misaligned address is UB. If store_unaligned
    # regressed to store_at, its store would change to natural alignment. The test asserts against that regression.
    @njit
    def w_unaligned(p):
        store_unaligned(p + 1, int64(0x0102030405060708))

    @njit
    def w_at(p):
        store_at(p + 1, int64(0x0102030405060708))

    buf = np.zeros(16, dtype=np.uint8)
    w_unaligned(array_data_p(buf))
    assert int(np.frombuffer(buf[1:9].tobytes(), dtype=np.int64)[0]) == 0x0102030405060708
    w_at(array_data_p(buf))  # w_at runs here only so that it compiles for the IR contrast below

    needle = str(0x0102030405060708)  # LLVM prints the i64 constant in decimal

    def const_stores(f):
        ir = "\n".join(f.inspect_llvm().values())
        return [ln for ln in ir.splitlines() if needle in ln and " store " in ln]

    assert any("align 1" in ln for ln in const_stores(w_unaligned)), const_stores(w_unaligned)
    assert const_stores(w_at) and all("align 1" not in ln for ln in const_stores(w_at)), const_stores(w_at)


def test_load_unaligned_emits_align1_not_natural():
    # The same contract holds on the load side: load_unaligned must emit an align=1 load.
    buf = np.zeros(16, dtype=np.uint8)
    buf[1:9] = np.frombuffer(np.int64(0x0102030405060708).tobytes(), dtype=np.uint8)

    @njit
    def r_unaligned(p):
        return load_unaligned(p + 1, int64)

    @njit
    def r_at(p):
        return load_at(p + 1, int64)

    assert r_unaligned(array_data_p(buf)) == 0x0102030405060708
    assert r_at(array_data_p(buf)) == 0x0102030405060708  # The access is UB, but x86 tolerates it

    def i64_loads(f):
        ir = "\n".join(f.inspect_llvm().values())
        return [ln for ln in ir.splitlines() if "= load i64" in ln]

    assert any("align 1" in ln for ln in i64_loads(r_unaligned)), i64_loads(r_unaligned)
    assert i64_loads(r_at) and all("align 1" not in ln for ln in i64_loads(r_at)), i64_loads(r_at)
