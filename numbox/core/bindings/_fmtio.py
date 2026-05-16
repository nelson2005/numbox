"""Variadic formatted I/O — ``printf``, ``fprintf``, ``snprintf`` callable
from @njit.

These are ``@intrinsic`` shells that each emit a direct extern variadic
call to libc — no delegation through ``numba.core.cgutils`` helpers, so
the encoding and symbol choice live entirely in this module. LLVM's
backend handles the platform-specific variadic ABI (SysV x86-64
``AL``-register convention, Win64 FP shadow, AAPCS64 named/anonymous
split) automatically.

The format string must be a ``Literal[str]`` at the call site so it can
be embedded as an IR global constant — the same constraint a C compiler
operates under when emitting a format-checked printf call.

**Format string encoding: UTF-8.** Modern terminals, files, and Windows
10+ consoles all expect UTF-8, and printf treats every non-``%`` byte
as opaque pass-through. Literal format strings can therefore contain
arbitrary Unicode characters (``"Цена: %d\\n"``, ``"caf\\u00e9 %s"``);
the codepoints are encoded to UTF-8 bytes once at codegen time and
flow through libc unmodified. Note that ``%-Ns`` width is byte-counted
by printf in every libc, so non-ASCII output won't right-pad to a
codepoint count — that's printf's contract, not ours.

User-call convention follows the same shape as ``_call_lib_func`` in this
package: variadic args are passed as a **tuple literal**. ``@intrinsic``
does not accept Python-level ``*args`` (its typing-function signature
drives numba's arg folding directly), so the tuple-as-args idiom is the
standard numba pattern for variadic shapes::

    printf("x = %d, y = %.3f\\n", (n, ratio))
    fprintf(stderr(), "warning: %s\\n", (msg_p,))
    snprintf(buf_p, buf.size, "[%d:%d]", (lo, hi))
    printf("hello\\n", ())             # zero args

C ABI default-argument promotion is applied so callers can pass numba's
natural numeric types:

- ``float32`` → ``float64`` (``fpext``)
- ``int8`` / ``int16`` → ``int32`` (signed: ``sext``; unsigned: ``zext``)
- ``int32`` / ``int64`` / ``uint32`` / ``uint64`` / ``float64`` / pointers: pass through

The user is otherwise responsible for matching format specifiers to argument
types the same way a C programmer is (e.g. ``%lld`` for ``int64`` on LP64,
``%d`` for ``int32``, ``%s`` for an ``intp`` pointing at a NUL-terminated
C string).

Caching: each call site emits a direct extern reference to the libc
symbol and a deterministic UTF-8 format-string global. The JIT linker
resolves the libc symbol per-process, the format-string global is
content-deterministic, and no runtime pointer is baked into the IR.
``@njit(cache=True)`` callers cache cleanly — no ``cres_cacheable``
indirection needed (unlike fixed-arg bindings, these never route through
a numba dispatcher whose ``id`` would be ASLR-randomized).

References:

- `printf(3) <https://man7.org/linux/man-pages/man3/printf.3.html>`_
- `fprintf(3) <https://man7.org/linux/man-pages/man3/fprintf.3.html>`_
- `snprintf(3) <https://man7.org/linux/man-pages/man3/snprintf.3.html>`_
- `Microsoft _snprintf
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/snprintf-snprintf-snprintf-l-snwprintf-snwprintf-l>`_
"""
from llvmlite import ir as llir
from numba.core import cgutils
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import BaseTuple, Float, Integer, Literal, int32
from numba.extending import intrinsic

from numbox.core.bindings.utils import load_lib, platform_


__all__ = ["printf", "fprintf", "snprintf"]


load_lib("c")


# Windows ships MSVCRT's "_snprintf" (returns -1 on truncation, no NUL guarantee).
# UCRT's C99-compliant "snprintf" is exposed via the C headers as an inline
# wrapper around `__stdio_common_vsnprintf`; declaring `i32 @snprintf(...)` in
# LLVM IR and letting the JIT linker resolve it crashes with an access
# violation. So on Windows we deliberately target the MSVCRT-legacy symbol
# and document the non-C99 truncation contract (see snprintf docstring).
_SNPRINTF_SYMBOL = "_snprintf" if platform_ == "Windows" else "snprintf"


def _promote_for_varargs(builder, arg_ty, arg_val):
    """Apply C default argument promotion for variadic arg passing.

    float32 → double (fpext); int8/int16 → int32 (sext if signed, zext if
    unsigned). Larger or already-promoted types pass through. This matches
    what a C compiler does when an arg is passed to a variadic function.
    """
    if isinstance(arg_ty, Float) and arg_ty.bitwidth == 32:
        return builder.fpext(arg_val, llir.DoubleType())
    if isinstance(arg_ty, Integer) and arg_ty.bitwidth < 32:
        i32_ll = llir.IntType(32)
        if arg_ty.signed:
            return builder.sext(arg_val, i32_ll)
        return builder.zext(arg_val, i32_ll)
    return arg_val


def _literal_format_or_raise(name, fmt_ty):
    """Extract the Python str value of a Literal[str] format-string type
    or raise a clean TypingError naming the binding.
    """
    if not isinstance(fmt_ty, Literal):
        raise TypingError(
            f"{name}: format string must be a literal str, got {fmt_ty!r}"
        )
    val = fmt_ty.literal_value
    if not isinstance(val, str):
        raise TypingError(
            f"{name}: format string must be a Python str, got {type(val).__name__}"
        )
    return val


def _unpack_args_tuple(builder, args_ty, args_pack):
    """Extract individual LLVM values from a tuple-of-args LLVM aggregate.

    Returns a list of (numba type, LLVM value) pairs. Handles the empty
    tuple ``()`` case (zero variadic args) where ``args_ty`` is ``Tuple()``.
    """
    arg_types = tuple(args_ty)
    return [
        (arg_types[i], builder.extract_value(args_pack, i))
        for i in range(len(arg_types))
    ]


def _emit_variadic_call(builder, symbol, fmt_str, leading_vals, args_ty, args_pack):
    """Emit IR for ``symbol(*leading_vals, fmt_p, *promoted_args) -> i32``.

    - ``leading_vals`` is a list of LLVM values that come before the format
      string in the call (FILE* for fprintf; buf, size for snprintf; empty
      for printf). Their LLVM types are read off the values themselves.
    - ``fmt_str`` is the Python str literal; encoded as UTF-8 + NUL and
      embedded as an IR global. ``cgutils.global_constant`` auto-uniquifies
      the global name across multiple call sites.
    - ``args_ty`` + ``args_pack`` are the tuple-of-args at the typing
      layer + the LLVM aggregate at the codegen layer; both come from the
      tuple-as-args calling convention.

    The variadic ABI (AL register on SysV, FP shadow on Win64, named/
    anonymous split on AAPCS64) is handled automatically by LLVM when the
    function is declared with ``var_arg=True``.
    """
    i8p = llir.IntType(8).as_pointer()
    i32_ll = llir.IntType(32)
    mod = builder.module
    fmt_bytes = cgutils.make_bytearray((fmt_str + '\x00').encode('utf-8'))
    global_fmt = cgutils.global_constant(mod, f"{symbol}_format", fmt_bytes)
    fmt_p = builder.bitcast(global_fmt, i8p)
    unpacked = _unpack_args_tuple(builder, args_ty, args_pack)
    promoted = [_promote_for_varargs(builder, t, v) for t, v in unpacked]
    leading_tys = [v.type for v in leading_vals]
    fn_ty = llir.FunctionType(
        i32_ll, leading_tys + [i8p], var_arg=True,
    )
    fn = get_or_insert_function(mod, fn_ty, symbol)
    return builder.call(fn, list(leading_vals) + [fmt_p] + promoted)


def _require_literal_fmt_and_tuple_args(name, fmt_ty, args_ty):
    """Typing-time guards shared by all three bindings: literal format
    string + tuple args. Returns the resolved Python format string."""
    fmt_str = _literal_format_or_raise(name, fmt_ty)
    if not isinstance(args_ty, BaseTuple):
        raise TypingError(
            f"{name}: args must be a tuple, got {args_ty!r}"
        )
    return fmt_str


@intrinsic(prefer_literal=True)
def printf(typingctx, fmt_ty, args_ty):
    """libc ``printf(fmt, args)`` — write formatted text to stdout.

    ``fmt`` must be a literal Python string at the call site. ``args`` is a
    tuple of values (use ``()`` for zero args). Returns the number of
    characters written (``int32``), or a negative value on error.

    Example::

        printf("count=%d ratio=%.3f\\n", (n, ratio))
        printf("done\\n", ())
    """
    fmt_str = _require_literal_fmt_and_tuple_args("printf", fmt_ty, args_ty)

    def codegen(context, builder, sig, llvm_args):
        _, args_pack = llvm_args
        return _emit_variadic_call(
            builder, "printf", fmt_str, [], args_ty, args_pack)

    return int32(fmt_ty, args_ty), codegen


@intrinsic(prefer_literal=True)
def fprintf(typingctx, fp_ty, fmt_ty, args_ty):
    """libc ``fprintf(fp, fmt, args)`` — write formatted text to FILE\\* fp.

    ``fp`` is the FILE\\* as ``intp`` (from ``stdout()`` / ``stderr()`` /
    ``stdin()`` or ``fopen()``). ``fmt`` must be a literal Python string.
    ``args`` is a tuple of values (use ``()`` for zero args).
    Returns chars written (``int32``), or negative on error.

    Example::

        fprintf(stderr(), "error %d: %s\\n", (code, msg_p))
    """
    fmt_str = _require_literal_fmt_and_tuple_args("fprintf", fmt_ty, args_ty)

    def codegen(context, builder, sig, llvm_args):
        i8p = llir.IntType(8).as_pointer()
        fp_int, _, args_pack = llvm_args
        fp_ptr = builder.inttoptr(fp_int, i8p)
        return _emit_variadic_call(
            builder, "fprintf", fmt_str, [fp_ptr], args_ty, args_pack)

    return int32(fp_ty, fmt_ty, args_ty), codegen


@intrinsic(prefer_literal=True)
def snprintf(typingctx, buf_ty, size_ty, fmt_ty, args_ty):
    """libc ``snprintf(buf, size, fmt, args)`` — format into a caller buffer.

    ``buf`` is an ``intp`` pointer to the destination buffer (caller-owned).
    ``size`` is the buffer size in bytes (``intp``). ``fmt`` must be a literal
    Python string. ``args`` is a tuple of values (use ``()`` for zero args).

    **Return value differs by platform**:

    - **Linux glibc, Linux musl, macOS**: C99 / POSIX ``snprintf`` semantics.
      Returns the number of characters that WOULD have been written if
      ``size`` were unlimited, NOT counting the trailing NUL. The written
      portion is always NUL-terminated when ``size > 0``. Detect truncation
      via ``rc >= size``.
    - **Windows**: MSVCRT ``_snprintf`` semantics. Returns the number of
      bytes written on success (excluding NUL), or **-1 on truncation**.
      The buffer is **not guaranteed to be NUL-terminated** when truncated.
      Detect truncation via ``rc < 0``. UCRT's C99-compliant ``snprintf``
      is a header-only inline over ``__stdio_common_vsnprintf`` rather
      than a directly-linkable symbol, so we can't reach it through an
      ``i32 @snprintf(...)`` LLVM declaration — the attempt crashed with
      an access violation. The Windows binding therefore deliberately
      targets the MSVCRT-legacy symbol.

    A portable "did it fit?" check that works on every supported platform::

        n = snprintf(array_data_p(buf), buf.size, fmt, args)
        truncated = (n < 0) or (n >= buf.size)

    Example::

        buf = np.zeros(64, dtype=np.uint8)
        n = snprintf(array_data_p(buf), buf.size, "[%d:%d]", (lo, hi))
        # On Linux/macOS: n < buf.size means no truncation; bytes(buf[:n]) is the message.
        # On Windows: n >= 0 means no truncation; bytes(buf[:n]) is the message.
    """
    fmt_str = _require_literal_fmt_and_tuple_args("snprintf", fmt_ty, args_ty)

    def codegen(context, builder, sig, llvm_args):
        i8p = llir.IntType(8).as_pointer()
        buf_int, size_val, _, args_pack = llvm_args
        buf_ptr = builder.inttoptr(buf_int, i8p)
        return _emit_variadic_call(
            builder, _SNPRINTF_SYMBOL, fmt_str,
            [buf_ptr, size_val], args_ty, args_pack)

    return int32(buf_ty, size_ty, fmt_ty, args_ty), codegen
