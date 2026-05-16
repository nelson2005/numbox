"""Variadic formatted I/O — ``printf``, ``fprintf``, ``snprintf`` callable
from @njit.

These are thin ``@intrinsic`` shells over numba's own variadic-call codegen
helpers in ``numba.core.cgutils``. The format string must be a
``Literal[str]`` at the call site so it can be embedded as an IR global
constant — the same constraint numba's internal debug ``printf`` operates
under.

User-call convention follows the same shape as ``_call_lib_func`` in this
package: variadic args are passed as a **tuple literal**. ``@intrinsic`` does
not accept Python-level ``*args`` (its typing-function signature drives
numba's arg folding directly), so the tuple-as-args idiom is the standard
numba pattern for variadic shapes::

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

Caching: these intrinsics emit a direct extern call to libc ``printf`` /
``fprintf`` / ``snprintf``. The JIT linker resolves the symbol per-process,
the format string is a deterministic Python literal embedded as a global
constant, and no runtime pointer is baked into the IR. So ``@njit(cache=True)``
callers cache cleanly — no ``cres_cacheable`` indirection needed (unlike
fixed-arg bindings, these never go through a numba dispatcher whose ``id``
would be ASLR-randomized).

References:

- `numba.core.cgutils.printf
  <https://github.com/numba/numba/blob/main/numba/core/cgutils.py>`_ — the
  underlying codegen helper this module delegates to.
- `printf(3) <https://man7.org/linux/man-pages/man3/printf.3.html>`_
- `snprintf(3) <https://man7.org/linux/man-pages/man3/snprintf.3.html>`_
"""
from llvmlite import ir as llir
from numba.core import cgutils
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import BaseTuple, Float, Integer, Literal, int32
from numba.extending import intrinsic

from numbox.core.bindings.utils import load_lib


__all__ = ["printf", "fprintf", "snprintf"]


load_lib("c")


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
    fmt_str = _literal_format_or_raise("printf", fmt_ty)
    if not isinstance(args_ty, BaseTuple):
        raise TypingError(
            f"printf: args must be a tuple, got {args_ty!r}"
        )

    def codegen(context, builder, sig, llvm_args):
        _, args_pack = llvm_args
        unpacked = _unpack_args_tuple(builder, args_ty, args_pack)
        promoted = [_promote_for_varargs(builder, t, v) for t, v in unpacked]
        return cgutils.printf(builder, fmt_str, *promoted)

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
    fmt_str = _literal_format_or_raise("fprintf", fmt_ty)
    if not isinstance(args_ty, BaseTuple):
        raise TypingError(
            f"fprintf: args must be a tuple, got {args_ty!r}"
        )

    def codegen(context, builder, sig, llvm_args):
        i8p = llir.IntType(8).as_pointer()
        mod = builder.module
        fp_int, _, args_pack = llvm_args
        fp_ptr = builder.inttoptr(fp_int, i8p)
        # Embed the format string as an IR global, same shape as cgutils.printf
        # does internally for stdout. global_constant auto-uniquifies the name
        # across multiple call sites.
        fmt_bytes = cgutils.make_bytearray((fmt_str + '\x00').encode('ascii'))
        global_fmt = cgutils.global_constant(mod, "fprintf_format", fmt_bytes)
        fmt_p = builder.bitcast(global_fmt, i8p)
        unpacked = _unpack_args_tuple(builder, args_ty, args_pack)
        promoted = [_promote_for_varargs(builder, t, v) for t, v in unpacked]
        fn_ty = llir.FunctionType(
            llir.IntType(32), [i8p, i8p], var_arg=True,
        )
        fn = get_or_insert_function(mod, fn_ty, "fprintf")
        return builder.call(fn, [fp_ptr, fmt_p] + promoted)

    return int32(fp_ty, fmt_ty, args_ty), codegen


@intrinsic(prefer_literal=True)
def snprintf(typingctx, buf_ty, size_ty, fmt_ty, args_ty):
    """libc ``snprintf(buf, size, fmt, args)`` — format into a caller buffer.

    ``buf`` is an ``intp`` pointer to the destination buffer (caller-owned).
    ``size`` is the buffer size in bytes (``intp``). ``fmt`` must be a literal
    Python string. ``args`` is a tuple of values (use ``()`` for zero args).

    **Return value differs by platform**:

    - **Linux glibc, Linux musl, macOS**: C99 / POSIX semantics. Returns the
      number of characters that WOULD have been written if ``size`` were
      unlimited, NOT counting the trailing NUL. The written portion is
      always NUL-terminated when ``size > 0``. Detect truncation via
      ``rc >= size``.
    - **Windows**: MSVCRT ``_snprintf`` semantics (what numba's underlying
      ``cgutils.snprintf`` codegen helper resolves to). Returns the number
      of bytes written on success (excluding NUL), or **-1 on truncation**.
      The buffer is **not guaranteed to be NUL-terminated** when truncated.
      Detect truncation via ``rc < 0``. This non-C99 behavior is inherited
      from numba's choice of the MSVCRT-compatible symbol; calling UCRT's
      C99-compliant ``snprintf`` directly via the JIT linker turned out
      to crash with an access violation (the actual UCRT export shape is
      not what one would naively expect from the C99 headers).

    A portable "did it fit?" check that works on every supported platform::

        n = snprintf(array_data_p(buf), buf.size, fmt, args)
        truncated = (n < 0) or (n >= buf.size)

    Example::

        buf = np.zeros(64, dtype=np.uint8)
        n = snprintf(array_data_p(buf), buf.size, "[%d:%d]", (lo, hi))
        # On Linux/macOS: n < buf.size means no truncation; bytes(buf[:n]) is the message.
        # On Windows: n >= 0 means no truncation; bytes(buf[:n]) is the message.
    """
    fmt_str = _literal_format_or_raise("snprintf", fmt_ty)
    if not isinstance(args_ty, BaseTuple):
        raise TypingError(
            f"snprintf: args must be a tuple, got {args_ty!r}"
        )

    def codegen(context, builder, sig, llvm_args):
        # Delegate to numba's cgutils.snprintf — it resolves to "snprintf"
        # on Linux/macOS (C99 semantics) and to "_snprintf" on Windows
        # (MSVCRT legacy semantics). Attempting to override the Windows
        # symbol to plain "snprintf" (the UCRT C99-compliant version)
        # produced an access violation in CI; the UCRT export is not in
        # the simple C-callable shape that a `declare i32 @snprintf(...)`
        # LLVM declaration would link to. The docstring documents the
        # resulting per-platform contract divergence.
        i8p = llir.IntType(8).as_pointer()
        buf_int, size_val, _, args_pack = llvm_args
        buf_ptr = builder.inttoptr(buf_int, i8p)
        unpacked = _unpack_args_tuple(builder, args_ty, args_pack)
        promoted = [_promote_for_varargs(builder, t, v) for t, v in unpacked]
        return cgutils.snprintf(
            builder, buf_ptr, size_val, fmt_str, *promoted)

    return int32(buf_ty, size_ty, fmt_ty, args_ty), codegen
