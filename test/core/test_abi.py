import pytest


def test_abi_imports():
    """Public helper symbols are exported by their respective modules."""
    from numbox.core.bindings import abi, call

    assert hasattr(abi, "_struct_bytes")
    assert hasattr(abi, "_classify")
    assert hasattr(abi, "_classify_eightbytes")
    assert hasattr(abi, "_is_canonical_int64_pair_layout")
    assert hasattr(abi, "_EIGHTBYTE_CLASS_INTEGER")
    assert hasattr(abi, "_EIGHTBYTE_CLASS_SSE")
    assert hasattr(abi, "_current_platform")
    assert hasattr(call, "_call_lib_func")
    assert hasattr(call, "_call_lib_func_byval")


def test_classify_eightbytes_int_int_non_canonical():
    """`Tuple([int32, int32, int64])` (the layout of duckdb_interval) has
    two pure-INTEGER eightbytes. The lo eightbyte holds two i32 fields
    (offsets 0, 4); the hi eightbyte holds one i64 (offset 8)."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.Tuple([types.int32, types.int32, types.int64])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_int_int_canonical_pair():
    """`UniTuple(int64, 2)` has the canonical `{i64, i64}` layout —
    INT/INT eightbytes, no repack needed."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.UniTuple(types.int64, 2)
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_four_i32():
    """`UniTuple(int32, 4)` has fields at offsets 0/4/8/12 — both
    eightbytes are pure INTEGER, but the LLVM type isn't `{i64, i64}`."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.UniTuple(types.int32, 4)
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_sse_sse():
    """`UniTuple(float64, 2)` has SSE eightbytes — lowered to XMM0/XMM1
    on SysV x86-64. Repack to `{i64, i64}` would be wrong."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_SSE, _classify_eightbytes,
    )

    ty = types.UniTuple(types.float64, 2)
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_SSE, _EIGHTBYTE_CLASS_SSE,
    )


def test_classify_eightbytes_sse_int():
    """`Tuple([float32, float32, int64])` has SSE in lo (two f32s),
    INT in hi (one i64)."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_SSE,
        _classify_eightbytes,
    )

    ty = types.Tuple([types.float32, types.float32, types.int64])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_SSE, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_mixed_lo_eightbyte_is_sse():
    """SysV rule: if any field in an eightbyte is SSE, the whole
    eightbyte is SSE. `Tuple([int32, float32, int64])` has int+float
    in lo → lo eightbyte is SSE."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_SSE,
        _classify_eightbytes,
    )

    ty = types.Tuple([types.int32, types.float32, types.int64])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_SSE, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_record_with_padding():
    """`Record.make_c_struct([(a, int32), (b, int64)])` is 16B with a
    4-byte gap (i32@0, pad, i64@8). Both eightbytes are pure INTEGER."""
    from numba.core import types
    from numbox.core.bindings.abi import (
        _EIGHTBYTE_CLASS_INTEGER, _classify_eightbytes,
    )

    ty = types.Record.make_c_struct([("a", types.int32), ("b", types.int64)])
    assert _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def test_classify_eightbytes_rejects_non_16b():
    """The classifier is only meaningful for 16-byte aggregates (the size
    where SysV may pass two eightbytes by-value in registers). Non-16B
    inputs raise a clean `TypingError`."""
    from numba.core import types
    from numba.core.errors import TypingError
    from numbox.core.bindings.abi import _classify_eightbytes

    with pytest.raises(TypingError, match="16-byte"):
        _classify_eightbytes(types.UniTuple(types.int64, 3))  # 24B
    with pytest.raises(TypingError, match="16-byte"):
        _classify_eightbytes(types.UniTuple(types.int32, 2))  # 8B


def test_classify_eightbytes_rejects_non_struct():
    """Scalar (non-struct) types raise a clean `TypingError`."""
    from numba.core import types
    from numba.core.errors import TypingError
    from numbox.core.bindings.abi import _classify_eightbytes

    with pytest.raises(TypingError, match="struct-shaped"):
        _classify_eightbytes(types.int64)


def test_is_canonical_int64_pair_layout_true_cases():
    """`UniTuple(int64, 2)`, `Tuple([int64, int64])`, and `Tuple([uint64,
    intp])` all lower to LLVM `{i64, i64}` — no repack needed."""
    from numba.core import types
    from numbox.core.bindings.abi import _is_canonical_int64_pair_layout

    assert _is_canonical_int64_pair_layout(types.UniTuple(types.int64, 2))
    assert _is_canonical_int64_pair_layout(
        types.Tuple([types.int64, types.int64]))
    assert _is_canonical_int64_pair_layout(
        types.Tuple([types.uint64, types.intp]))


def test_is_canonical_int64_pair_layout_false_cases():
    """Anything not exactly two 64-bit integer fields at offsets 0/8 is
    non-canonical — including `{i32, i32, i64}` (the duckdb_interval
    layout that needs repack), `{i32 × 4}`, `{f64, f64}`, and 24-byte
    aggregates."""
    from numba.core import types
    from numbox.core.bindings.abi import _is_canonical_int64_pair_layout

    assert not _is_canonical_int64_pair_layout(
        types.Tuple([types.int32, types.int32, types.int64]))
    assert not _is_canonical_int64_pair_layout(
        types.UniTuple(types.int32, 4))
    assert not _is_canonical_int64_pair_layout(
        types.UniTuple(types.float64, 2))
    assert not _is_canonical_int64_pair_layout(
        types.UniTuple(types.int64, 3))


def test_is_canonical_int64_pair_layout_record():
    """`Record.make_c_struct([(a, int64), (b, int64)])` lowers to
    `{i64, i64}` — canonical. With smaller fields the LLVM type is
    different and the helper returns False."""
    from numba.core import types
    from numbox.core.bindings.abi import _is_canonical_int64_pair_layout

    rec_canonical = types.Record.make_c_struct([
        ("a", types.int64), ("b", types.int64)])
    assert _is_canonical_int64_pair_layout(rec_canonical)

    rec_non_canonical = types.Record.make_c_struct([
        ("a", types.int32), ("b", types.int64)])
    assert not _is_canonical_int64_pair_layout(rec_non_canonical)


def test_struct_bytes_supports_all_struct_types():
    """The struct-size helper used by the ABI codegen handles every
    numba struct-shaped type: Tuple, UniTuple, NamedTuple (via .types),
    and Record (via .size)."""
    import collections
    from numba.core import types
    from numbox.core.bindings.abi import _struct_bytes

    assert _struct_bytes(
        types.Tuple([types.int32, types.int32, types.int64]), "t") == 16
    assert _struct_bytes(
        types.UniTuple(types.int32, 4), "t") == 16

    MyNT = collections.namedtuple("MyNT", ["a", "b"])
    assert _struct_bytes(
        types.NamedTuple([types.int32, types.int64], MyNT), "t") == 12

    rec = types.Record.make_c_struct([("a", types.int32), ("b", types.int64)])
    assert _struct_bytes(rec, "t") == 16  # 4 + 4 pad + 8


def test_struct_bytes_rejects_non_struct_type():
    """Scalar or otherwise non-struct types raise a clean TypingError."""
    from numba.core import types
    from numba.core.errors import TypingError
    from numbox.core.bindings.abi import _struct_bytes

    with pytest.raises(TypingError, match="struct-shaped type"):
        _struct_bytes(types.int32, "_call_lib_func_byval")


def test_call_lib_func_lldiv_via_unified():
    """End-to-end: call libc ``lldiv(10, 3)`` via the unified intrinsic
    and validate the 16-byte ``lldiv_t`` return value.

    Exercises the return-side ABI path on whatever platform the test
    runs on: SysV x86-64 and AAPCS64 read ``lldiv_t`` back from GP
    registers; Windows x64 reads it from a caller-allocated ``sret``
    slot. A regression on any of those three ABIs surfaces as a wrong
    quot or rem here.
    """
    from numba import njit
    from numbox.core.bindings.call import _call_lib_func

    @njit
    def run():
        return _call_lib_func("lldiv", (10, 3))

    quot, rem = run()
    assert quot == 3
    assert rem == 1


def test_call_lib_func_scalar_args_unchanged():
    """Regression guard: scalar args + scalar return path goes through
    `_call_lib_func` unchanged from the pre-unification behavior.

    `cos(0.0)` from libm returns `1.0`. If the rewrite of `_call_lib_func`
    broke the scalar path that math / c / sqlite bindings depend on,
    this fails with an LLVM IR error or a wrong return value.
    """
    from numba import njit
    from numbox.core.bindings.call import _call_lib_func

    @njit
    def run():
        return _call_lib_func("cos", (0.0,))

    assert run() == 1.0


def test_call_lib_func_scalar_arg_auto_wrapped():
    """A single non-tuple arg is auto-wrapped into a 1-tuple at the
    intrinsic boundary, so `_call_lib_func("cos", 0.0)` is equivalent
    to `_call_lib_func("cos", (0.0,))`.
    """
    from numba import njit
    from numbox.core.bindings.call import _call_lib_func

    @njit
    def run():
        return _call_lib_func("cos", 0.0)

    assert run() == 1.0


def _register_test_symbol(name):
    """Register a no-op address under ``name`` so ``ll.address_of_symbol``
    finds something for the IR-inspection tests. The body is never
    executed — the tests only inspect the LLVM IR emitted at compile
    time. Returns the ctypes wrapper, which the caller must keep alive
    for the symbol to remain valid.
    """
    import ctypes
    import llvmlite.binding as ll

    @ctypes.CFUNCTYPE(ctypes.c_int32)
    def _stub():
        return 0

    addr = ctypes.cast(_stub, ctypes.c_void_p).value
    ll.add_symbol(name, addr)
    return _stub


@pytest.fixture
def patch_signature():
    """Add a temporary entry to ``signatures`` and remove it after.

    Yields a function ``register(name, sig)`` that the test calls to
    install a fake signature. The fixture undoes the install on teardown,
    even if the install replaced an existing entry.
    """
    from numbox.core.bindings.signatures import signatures

    sentinel = object()
    saved = []

    def register(name, sig):
        saved.append((name, signatures.get(name, sentinel)))
        signatures[name] = sig

    yield register

    for name, prev in saved:
        if prev is sentinel:
            del signatures[name]
        else:
            signatures[name] = prev


def _platform_str():
    from numbox.core.bindings.abi import _current_platform
    try:
        return _current_platform()
    except RuntimeError:
        return "unknown"


@pytest.mark.skipif(
    _platform_str() != "sysv_x86_64",
    reason="byval + optnone + noinline are SysV x86-64 specific",
)
def test_call_lib_func_byval_attribute_in_ir_for_large_struct(patch_signature):
    """On SysV x86-64, a 24-byte struct arg is lowered with ``byval``
    on the LLVM parameter and ``optnone`` + ``noinline`` on the
    enclosing function. The actual C function is never called — the
    test only inspects the IR emitted by numba.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_byval_large_24b"
    keepalive = _register_test_symbol(name)
    big_struct = nb_types.UniTuple(nb_types.int64, 3)
    patch_signature(name, nb_types.int32(big_struct))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.UniTuple(nb_types.int64, 3),))
    ir_text = list(run.inspect_llvm().values())[0]

    assert "byval(" in ir_text, (
        "expected 'byval(' attribute on >16B struct arg on SysV x86-64;\n"
        f"IR was:\n{ir_text}"
    )
    assert "optnone" in ir_text, (
        "expected 'optnone' on enclosing function on SysV x86-64"
    )
    assert "noinline" in ir_text, (
        "expected 'noinline' on enclosing function on SysV x86-64"
    )
    del keepalive


@pytest.mark.skipif(
    _platform_str() != "sysv_x86_64",
    reason="≤16B-struct passing differs across ABIs; SysV-specific check",
)
def test_call_lib_func_no_byval_attribute_for_small_struct(patch_signature):
    """On SysV x86-64, a ≤16B struct arg is passed by value in
    registers; LLVM lowers without a ``byval`` attribute and without
    forcing ``optnone`` / ``noinline``.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_byval_small_16b"
    keepalive = _register_test_symbol(name)
    small_struct = nb_types.UniTuple(nb_types.int64, 2)
    patch_signature(name, nb_types.int32(small_struct))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.UniTuple(nb_types.int64, 2),))
    ir_text = list(run.inspect_llvm().values())[0]

    assert "byval(" not in ir_text, (
        "did not expect 'byval(' on ≤16B struct arg on SysV x86-64;\n"
        f"IR was:\n{ir_text}"
    )
    assert "optnone" not in ir_text, (
        "did not expect 'optnone' on enclosing function for ≤16B struct"
    )
    del keepalive


@pytest.mark.skipif(
    _platform_str() != "win_x64",
    reason="Windows-x64-specific 1/2/4/8-byte register-passing rule",
)
def test_call_lib_func_8byte_struct_arg_on_windows_passes_by_value(patch_signature):
    """On Windows x64, an 8-byte struct arg is passed by value in
    registers (1/2/4/8-byte aggregates take the register-passing path).
    The LLVM IR should NOT alloca + pass-by-pointer this case.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_win_pass_8b"
    keepalive = _register_test_symbol(name)
    eight_byte_struct = nb_types.UniTuple(nb_types.int32, 2)
    patch_signature(name, nb_types.int32(eight_byte_struct))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.UniTuple(nb_types.int32, 2),))
    ir_text = list(run.inspect_llvm().values())[0]

    declare_line = next(
        (line for line in ir_text.splitlines() if name in line and "declare" in line),
        None,
    )
    assert declare_line is not None, (
        f"could not find declare line for {name} in IR:\n{ir_text}"
    )
    assert "*" not in declare_line.split("(")[1], (
        f"expected struct-by-value (no pointer) on Windows for 8B arg; "
        f"declare line was:\n{declare_line}"
    )
    del keepalive


@pytest.mark.skipif(
    _platform_str() != "win_x64",
    reason="Windows-x64-specific 1/2/4/8-byte register-return rule",
)
def test_call_lib_func_8byte_struct_return_on_windows_no_sret(patch_signature):
    """On Windows x64, an 8-byte struct return goes directly in RAX —
    no ``sret`` slot, no void return. Sizes outside {1, 2, 4, 8} use
    sret; this test pins the small-size special case.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_win_ret_8b"
    keepalive = _register_test_symbol(name)
    eight_byte_struct = nb_types.UniTuple(nb_types.int32, 2)
    patch_signature(name, eight_byte_struct(nb_types.int32))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((nb_types.int32,))
    ir_text = list(run.inspect_llvm().values())[0]

    declare_line = next(
        (line for line in ir_text.splitlines() if "declare" in line and name in line),
        None,
    )
    assert declare_line is not None, (
        f"could not find declare line for {name} in IR:\n{ir_text}"
    )
    assert "sret" not in declare_line, (
        f"did not expect 'sret' on 8B struct return on Windows x64; "
        f"declare line was:\n{declare_line}"
    )
    del keepalive


def test_call_lib_func_undefined_signature_raises():
    """`_call_lib_func` raises when the function name has an LLVM symbol
    but is missing from the `signatures` dict.
    """
    from numba import njit
    from numba.core.errors import TypingError
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_no_sig_unified"
    keepalive = _register_test_symbol(name)

    @njit
    def run():
        return _call_lib_func(name, (0.0,))

    with pytest.raises((ValueError, TypingError), match="Undefined signature"):
        run()
    del keepalive


@pytest.mark.skipif(
    _platform_str() != "sysv_x86_64",
    reason="INT/INT eightbyte repack only kicks in on SysV x86-64",
)
def test_call_lib_func_int_int_eightbyte_repack_round_trip(patch_signature):
    """Round-trip a 16-byte ``{i32, i32, i64}`` struct (the
    ``duckdb_interval`` shape) through ``_call_lib_func`` on SysV
    x86-64. Without the eightbyte repack, llvmlite drops the second
    ``i32`` field when lowering the by-value call — only the first
    ``i32`` and the trailing ``i64`` survive. After the repack to
    canonical ``{i64, i64}``, all three fields arrive intact.
    """
    import ctypes
    import llvmlite.binding as ll
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    class _IntervalC(ctypes.Structure):
        _fields_ = [
            ("a", ctypes.c_int32),
            ("b", ctypes.c_int32),
            ("c", ctypes.c_int64),
        ]

    received = {}

    @ctypes.CFUNCTYPE(ctypes.c_int64, _IntervalC)
    def echo(s):
        received["a"] = s.a
        received["b"] = s.b
        received["c"] = s.c
        return s.a + s.b + s.c

    name = "numbox_test_int_int_eightbyte_round_trip"
    addr = ctypes.cast(echo, ctypes.c_void_p).value
    ll.add_symbol(name, addr)
    arg_struct = nb_types.Tuple(
        [nb_types.int32, nb_types.int32, nb_types.int64])
    patch_signature(name, nb_types.int64(arg_struct))

    @njit(nb_types.int64(nb_types.int32, nb_types.int32, nb_types.int64))
    def run(a, b, c):
        return _call_lib_func(name, ((a, b, c),))

    result = run(7, 11, 1_000_000)

    assert received == {"a": 7, "b": 11, "c": 1_000_000}, (
        f"second i32 field was dropped by SysV by-value lowering: {received}"
    )
    assert result == 7 + 11 + 1_000_000
    del echo  # keepalive


@pytest.mark.skipif(
    _platform_str() != "sysv_x86_64",
    reason="repack-skip rules are SysV x86-64 specific",
)
def test_call_lib_func_sse_eightbyte_arg_not_repacked(patch_signature):
    """A 16B ``{double, double}`` arg has SSE eightbytes — passed in
    XMM0/XMM1 on SysV x86-64. It must NOT be repacked to ``{i64, i64}``
    (which would force GP registers RDI/RSI), so the LLVM declare line
    keeps the ``double, double`` shape rather than ``i64, i64``.
    """
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_sse_pair_no_repack"
    keepalive = _register_test_symbol(name)
    sse_struct = nb_types.UniTuple(nb_types.float64, 2)
    patch_signature(name, nb_types.int32(sse_struct))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    run.compile((sse_struct,))
    ir_text = list(run.inspect_llvm().values())[0]
    declare_line = next(
        (line for line in ir_text.splitlines()
         if "declare" in line and name in line),
        None,
    )
    assert declare_line is not None, (
        f"could not find declare line for {name} in IR:\n{ir_text}"
    )
    assert "double" in declare_line, (
        f"expected SSE pair to keep its double-typed arg in declare "
        f"(numba lowers UniTuple(float64, 2) to '[2 x double]'); "
        f"got:\n{declare_line}"
    )
    assert "i64" not in declare_line, (
        f"SSE pair must not be repacked to a 64-bit-integer-typed arg "
        f"(would force GP registers instead of XMM); declare line "
        f"was:\n{declare_line}"
    )
    del keepalive


@pytest.mark.skipif(
    _platform_str() != "sysv_x86_64",
    reason="canonical-skip is SysV x86-64 specific",
)
def test_call_lib_func_canonical_int64_pair_round_trip(patch_signature):
    """A canonical 16B ``UniTuple(int64, 2)`` already lowers to
    ``{i64, i64}`` and round-trips correctly through ``_call_lib_func``
    without any repack — regression guard that the canonical-skip in
    ``_needs_int_int_eightbyte_repack`` doesn't break the path numbduck
    will use for ``duckdb_hugeint`` and ``duckdb_uhugeint``.
    """
    import ctypes
    import llvmlite.binding as ll
    from numba import njit, types as nb_types
    from numbox.core.bindings.call import _call_lib_func

    class _PairC(ctypes.Structure):
        _fields_ = [("lo", ctypes.c_int64), ("hi", ctypes.c_int64)]

    received = {}

    @ctypes.CFUNCTYPE(ctypes.c_int64, _PairC)
    def echo(s):
        received["lo"] = s.lo
        received["hi"] = s.hi
        return s.hi

    name = "numbox_test_canonical_i64_pair_round_trip"
    addr = ctypes.cast(echo, ctypes.c_void_p).value
    ll.add_symbol(name, addr)
    arg_struct = nb_types.UniTuple(nb_types.int64, 2)
    patch_signature(name, nb_types.int64(arg_struct))

    @njit(nb_types.int64(nb_types.int64, nb_types.int64))
    def run(lo, hi):
        return _call_lib_func(name, ((lo, hi),))

    result = run(0x0123456789ABCDEF, -42)

    assert received == {"lo": 0x0123456789ABCDEF, "hi": -42}
    assert result == -42
    del echo  # keepalive


def test_call_lib_func_byval_undefined_signature_raises():
    """`_call_lib_func_byval` raises when the function name is missing
    from the `signatures` dict.
    """
    from numba import njit, types as nb_types
    from numba.core.errors import TypingError
    from numbox.core.bindings.call import _call_lib_func_byval

    name = "numbox_test_no_sig_byval"
    keepalive = _register_test_symbol(name)
    arg_struct = nb_types.UniTuple(nb_types.int64, 2)

    @njit
    def run(x):
        return _call_lib_func_byval(name, x)

    with pytest.raises((ValueError, TypingError), match="Undefined signature"):
        run.compile((arg_struct,))
    del keepalive


def test_call_lib_func_missing_llvm_symbol_raises(patch_signature):
    """`_call_lib_func` raises when the function name is in the
    `signatures` dict but has no LLVM symbol registered.
    """
    from numba import njit, types as nb_types
    from numba.core.errors import TypingError
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_no_llvm_symbol"
    patch_signature(name, nb_types.float64(nb_types.float64))

    @njit
    def run():
        return _call_lib_func(name, (0.0,))

    with pytest.raises((RuntimeError, TypingError), match="unavailable in the LLVM context"):
        run()


def test_call_lib_func_large_return_struct_raises(patch_signature):
    """`_call_lib_func` raises when the C function returns a struct >16
    bytes — that path is unsupported.
    """
    from numba import njit, types as nb_types
    from numba.core.errors import TypingError
    from numbox.core.bindings.call import _call_lib_func

    name = "numbox_test_large_ret"
    keepalive = _register_test_symbol(name)
    big_ret = nb_types.UniTuple(nb_types.int64, 3)
    patch_signature(name, big_ret(nb_types.int32))

    @njit
    def run(x):
        return _call_lib_func(name, (x,))

    with pytest.raises(TypingError, match="return struct >16 bytes"):
        run.compile((nb_types.int32,))
    del keepalive
