"""Variadic formatted I/O: ``printf``, ``fprintf`` and ``snprintf`` in plain Python and ``@njit``, ``sscanf`` in ``@njit``.

Each public binding is an ordinary Python function. When ``@njit`` code calls a binding, numba dispatches the call
through ``@overload`` to a private ``@intrinsic``, which generates the code for the libc call. For the three writers,
the same source runs unchanged in either mode, as numba's own builtins such as ``print`` and ``range`` do::

    def debug_kernel(x, label):
        printf("step %d: %s\\n", x, label)
        fflush(stdout())
        return x * 2

    debug_kernel(7, "before")        # pure Python: writes via sys.stdout
    njit(debug_kernel)(7, "before")  # jitted: writes via libc printf

Call convention
---------------

The public API is C-like. The arguments come after the format string as positional ``*args``, with no tuple at the
call site::

    printf("x = %d, ratio = %.3f\\n", n, ratio)
    fprintf(stderr(), "warning: %s\\n", msg)
    snprintf(array_data_p(buf), buf.size, "[%d:%d]", lo, hi)
    sscanf(buf_p, "%d %lf", array_data_p(n_out), array_data_p(x_out))
    printf("no args here\\n")

The private ``_xxx_intrinsic`` still takes its arguments as one tuple, because numba's ``@intrinsic`` typing function
does not accept Python-level ``*args``. The ``@overload`` path packs the arguments into that tuple before it calls the
intrinsic.

Literal format strings under @njit
----------------------------------

The format string must be a literal under ``@njit``, so that the intrinsic can embed it in the IR as a global
constant. A C compiler that emits a format-checked printf call works under the same constraint. A ``unicode`` value
built at run time raises a clear ``TypingError`` when numba types the call. In pure-Python mode the format string can
be any str.

Because the format is a literal, the ``@njit`` writers also check it against the arguments at typing time, as a C
compiler's ``-Wformat`` does. The number of conversion specifiers must equal the number of arguments, and the class
of each specifier must match its argument:

- an integer conversion, such as ``%d`` or ``%x``, takes an Integer or a Boolean;
- a float conversion, such as ``%f`` or ``%g``, takes a Float;
- ``%s`` takes a unicode string or an ``intp`` or ``uintp`` pointer;
- ``%p`` takes an ``intp`` or a ``uintp``.

A mismatch raises ``TypingError``. Without the check, libc could read an unpushed varargs slot, dereference an int as
``char*``, or read a general-purpose register for a float passed in an SSE register. numba's ``intp`` is ``int64``,
so the check cannot distinguish a pointer-width int passed for ``%s`` or ``%p`` from a real pointer. The check still
rejects narrower ints, floats and booleans for those specifiers.

Format string encoding: UTF-8
-----------------------------

A non-ASCII code point in the literal is encoded as its UTF-8 byte sequence and embedded in the IR global. printf
treats every byte other than ``%`` as opaque, so the bytes reach stdout, a FILE\\* or the snprintf buffer unmodified.
Modern terminals, files and Windows 10+ consoles expect UTF-8.

.. note::
   In every libc, printf counts the ``%-Ns`` width in bytes, so it does not right-pad non-ASCII output to a width in
   code points. This behavior is part of the printf contract. If the width must count code points, pad the string
   with numba-side string formatting (``f"{s:<10}"``) before passing it through ``%s``.

Cross-mode caveats
------------------

1. **Length modifiers in format strings.** C printf accepts ``%lld``, ``%ld``, ``%lf``, ``%hd`` and similar
   directives, but Python's ``%`` operator ignores a single length modifier and rejects a doubled one. The
   pure-Python implementations strip the length modifiers
   with a regex before formatting, so ``%lld`` becomes ``%d`` and ``%.3lf`` becomes ``%.3f``. The stripped form
   produces identical output for typical values, because Python formats an int or a float at the value's own width
   whatever the length modifier.

2. **Widening of integer arguments to 64 bits.** The @njit implementation widens every integer variadic
   argument to 64 bits before the libc call, with sext or zext as the type requires. Without the widening,
   ``printf("%lld", np.int32(7))`` in @njit would make libc read 8 bytes from a 4-byte source, and the high bits
   would hold undefined register contents. With the widening, ``%lld`` formats an int32, int16, int8 or bool
   correctly. This behavior departs from the C ABI, which does not promote int to long long, but it matches user
   expectations and the behavior of the pure-Python ``%``.

3. **String arguments for ``%s``.** Python's ``%`` formats strings natively. Under @njit, the @overload converts each
   ``unicode_type`` argument with ``get_unicode_data_p``, so libc receives a NUL-terminated C string. Users no longer
   need to call ``get_unicode_data_p`` explicitly at the call site.

4. **``%ld`` on Win64 (LLP64).** ``long`` is 4 bytes on Win64 and 8 bytes on LP64. Under @njit on Win64, ``%ld``
   therefore truncates an int64 to its low 32 bits. Pure-Python mode does not show the difference, because Python's
   ``%`` ignores length modifiers. For a portable 8-byte width, use ``%lld`` with an int64.

5. **The ``snprintf`` return code after truncation on Windows.** Pure-Python mode and @njit on Linux and macOS
   follow C99: snprintf returns the number of characters it would have written. On Windows, the @njit binding calls
   MSVCRT ``_snprintf``, which returns ``-1`` on truncation and does not guarantee a NUL terminator. The portable
   check ``(rc < 0) or (rc >= size)`` works on every platform.

6. **``fprintf`` to a FILE\\* other than a stdio handle, in pure Python.** The Python implementation maps the
   ``stdout()``, ``stderr()`` and ``stdin()`` handles to the corresponding ``sys.*`` streams through a cache keyed
   by address. Python cannot dereference a FILE\\* returned by ``fopen`` without a ctypes call, so a call with such a
   handle raises a clear error in pure-Python mode. For file I/O on the Python side, use ``open()`` and
   ``f.write()``.

7. **``sscanf`` is @njit-only.** In pure Python, ``int()``, ``float()`` and ``re`` usually suit sscanf-style parsing
   better. A call to ``sscanf`` from pure Python raises ``NotImplementedError``.

References
----------

- `printf(3) <https://man7.org/linux/man-pages/man3/printf.3.html>`_
- `fprintf(3) <https://man7.org/linux/man-pages/man3/fprintf.3.html>`_
- `snprintf(3) <https://man7.org/linux/man-pages/man3/snprintf.3.html>`_
- `sscanf(3) <https://man7.org/linux/man-pages/man3/sscanf.3.html>`_
- `Microsoft _snprintf
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/snprintf-snprintf-snprintf-l-snwprintf-snwprintf-l>`_
"""
import ctypes
import re
import sys

from llvmlite import ir as llir
from numba.core import cgutils
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import (
    BaseTuple, Boolean, Float, Integer, UnicodeType, int32, intp, uintp,
    unliteral,
)
from numba.extending import intrinsic, overload

from numbox.core.bindings.utils import (
    extract_literal_str, intp_ll_type, load_lib, platform_,
)


__all__ = ["printf", "fprintf", "snprintf", "sscanf"]


load_lib("c")


# Windows MSVCRT exports "_snprintf", whose truncation semantics are not C99: it returns -1 on truncation and does
# not guarantee a NUL terminator. UCRT's C99-compliant "snprintf" is a header-only inline function over
# __stdio_common_vsnprintf, and it cannot be linked directly with the plain C99 signature. If the IR declares
# `i32 @snprintf(...)` and the JIT linker resolves it, the call crashes with an access violation. The @njit binding on
# Windows therefore calls MSVCRT on purpose (see the cross-mode caveats in the module docstring). Pure-Python mode
# follows POSIX and C99 semantics on every platform.
_SNPRINTF_SYMBOL = "_snprintf" if platform_ == "Windows" else "snprintf"


# This regex matches the C printf length modifiers 'hh', 'h', 'l', 'll' and 'L'. Python's % operator ignores a single
# one and rejects a doubled one with `ValueError: unsupported format character`, so pure-Python formatting strips them
# all first. The stripped %d or
# %f produces equivalent output for typical values, because Python's % formats a value at its natural width.
_LENGTH_MODIFIER_RE = re.compile(
    r'%([-+0# ]*[0-9*]*\.?[0-9*]*)(hh|ll|h|l|L|j|z|t|q|I32|I64)([diouxXfFeEgGaAcsp])'
)


def _python_fmt_compat(fmt):
    """Strip C printf length modifiers so Python's % accepts the format."""
    return _LENGTH_MODIFIER_RE.sub(r'%\1\3', fmt)


# ---------------------------------------------------------------------------
# Helpers for the @intrinsic codegen layer (private)
# ---------------------------------------------------------------------------


def _promote_for_varargs(builder, arg_ty, arg_val):
    """Apply the C-ABI promotion to a variadic argument, and widen every integer argument to 64 bits.

    - ``float32`` becomes ``float64`` (fpext).
    - ``bool`` becomes ``int64`` (zext).
    - A signed ``Integer`` narrower than 64 bits becomes ``int64`` (sext).
    - An unsigned ``Integer`` narrower than 64 bits becomes ``int64`` (zext).
    - 64-bit ints, doubles and pointers pass through unchanged.

    The widening to 64 bits, rather than to int32 as the strict C ABI requires, is deliberate. It makes ``%lld`` with
    an int32 work in @njit, which matches the pure-Python behavior: Python ignores length modifiers and uses the
    natural width of the value. The cost is a single LLVM sext, zext or fpext per argument, which is negligible at run
    time.

    ``_validate_writer_arg_type`` validates the argument types at typing time, and the ``else: raise`` here adds
    defense in depth. If the typing-time validation misses a type, for example because a future numba flattens tuples
    differently, code generation stops instead of passing an invalid value to libc's variadic call.
    """
    i64_ll = llir.IntType(64)
    if isinstance(arg_ty, Float):
        if arg_ty.bitwidth == 32:
            return builder.fpext(arg_val, llir.DoubleType())
        return arg_val
    if isinstance(arg_ty, Boolean):
        return builder.zext(arg_val, i64_ll)
    if isinstance(arg_ty, Integer):
        if arg_ty.bitwidth < 64:
            if arg_ty.signed:
                return builder.sext(arg_val, i64_ll)
            return builder.zext(arg_val, i64_ll)
        return arg_val
    raise TypingError(
        f"variadic arg of type {arg_ty!r} is not supported by printf-family "
        f"bindings; allowed: Float, Integer, Boolean, or intp pointer"
    )


def _validate_writer_arg_type(name, idx, ty):
    """Raise ``TypingError`` unless ``ty`` is a scalar type that ``printf``, ``fprintf`` and ``snprintf`` support.

    The supported types are Float, Integer (including intp for raw pointers), Boolean and the unicode type, which the
    @overload layer converts with ``get_unicode_data_p``.

    Without this guard, ``_promote_for_varargs`` would silently ``return arg_val`` for unsupported types. numpy
    arrays, complex numbers, tuples and similar values would then reach libc's variadic call directly as LLVM
    aggregates. The result would be a silent miscompilation instead of a clear error.
    """
    if isinstance(unliteral(ty), (Float, Integer, Boolean, UnicodeType)):
        return
    raise TypingError(
        f"{name}: arg {idx} has unsupported type {ty!r}; allowed: Float, "
        f"Integer, Boolean, unicode (auto-converted to char* via "
        f"get_unicode_data_p), or intp (raw pointer / preconverted string)"
    )


_PERCENT_N_RE = re.compile(
    r'%(?:[0-9]+\$)?[-+0# ]*[*0-9]*(?:[0-9]+\$)?'
    r'(?:\.[*0-9]*(?:[0-9]+\$)?)?'
    r'(?:hh|ll|h|l|L|j|z|t|q|I32|I64)?n'
)


def _reject_percent_n_or_raise(name, fmt_str):
    """Raise ``TypingError`` if ``fmt_str`` contains a ``%n`` directive.

    The check matches ``%n`` with or without flags, width, precision or a length modifier.

    ``%n`` makes printf write the number of bytes written so far through a caller-supplied ``int*`` pointer. The
    directive is rejected for two reasons. First, it writes memory through an argument that pure-Python's ``%``
    operator rejects with ``ValueError``, so allowing it would break the equivalence of the two modes. Second, it is a
    memory-safety hazard: static analyzers widely flag it, and glibc's ``_FORTIFY_SOURCE`` disables it for writable
    format strings.

    ``%%n``, a literal ``%`` followed by ``n``, is allowed. The regex runs on the format string after the function has
    replaced each ``%%`` pair with a sentinel, which the directive pattern cannot match.
    """
    stripped = fmt_str.replace('%%', '\x00\x00')
    if _PERCENT_N_RE.search(stripped):
        raise TypingError(
            f"{name}: %n directive in format string {fmt_str!r} is not "
            f"allowed (writes byte-count-written through caller pointer; "
            f"memory-safety hazard, also diverges from pure-Python behavior). "
            f"Use sscanf if you need %n's read-position semantics."
        )


# This pattern matches a directive whose run of flags contains the glibc ``'`` (thousands-grouping) flag. It runs on
# the format string after the ``%%`` pairs are masked.
_GROUPING_FLAG_RE = re.compile(r"%[-+0# ']*'")


def _reject_grouping_flag_or_raise(name, fmt_str):
    """Raise ``TypingError`` if ``fmt_str`` uses the glibc ``'`` grouping flag.

    The ``'`` flag (thousands grouping) is a legal glibc printf flag, but C99 does not define it and Python's ``%``
    operator does not support it. Honoring it would make the @njit and pure-Python paths diverge.

    ``_FMT_SPEC_RE`` does not recognize the flag either. A ``%'d`` directive would not match at all, so the check
    would undercount the specifiers without any error. A following directive would then read an unpushed varargs
    slot, which can return garbage, disclose memory or crash. That is the exact hazard that
    ``_validate_format_vs_args`` exists to prevent. The flag is therefore rejected with a clear error in both modes,
    in the same way as ``%n``.
    """
    stripped = fmt_str.replace('%%', '\x00\x00')
    if _GROUPING_FLAG_RE.search(stripped):
        raise TypingError(
            f"{name}: the ' (thousands-grouping) flag in format string "
            f"{fmt_str!r} is not supported (glibc-only, absent from C99 and "
            f"Python's % operator). Remove it or group the digits yourself."
        )


_FMT_SPEC_RE = re.compile(
    r'%[-+0# ]*'                          # flags
    r'(?:\*|[0-9]*)'                       # field width (number or '*')
    r'(?:\.(?:\*|[0-9]*))?'               # .precision (number or '*')
    r'(?:hh|ll|h|l|L|j|z|t|q|I32|I64)?'   # length modifier
    r'([diouxXeEfFgGaAcsp])'              # conversion (group 1); 'n' is rejected
)                                          # separately, '%%' is masked first

_CONV_INT = set("diouxXc")
_CONV_FLOAT = set("eEfFgGaA")


def _validate_format_vs_args(name, fmt_str, args_ty):
    """Check the format's conversion specifiers against the argument types, as a C compiler's ``-Wformat`` does.

    The check runs at typing time, on the literal format string. The number of specifiers must equal the number of
    arguments, and the class of each specifier must match its argument:

    - ``%d/%i/%u/%o/%x/%X/%c`` take an Integer or a Boolean;
    - ``%e/%f/%g/...`` take a Float;
    - ``%s`` takes a unicode string or an ``intp`` or ``uintp`` pointer;
    - ``%p`` takes an ``intp`` or a ``uintp``.

    Each ``*`` dynamic width or precision consumes one Integer argument, ahead of the argument that its specifier
    converts.

    Without this check, ``printf('%d %d', x)`` reads an unpushed varargs slot, which can return garbage, disclose
    memory or crash. ``printf('%s', 5)`` makes libc dereference an int as ``char*``. ``printf('%d', 3.5)`` reads a
    general-purpose register for a float passed in an SSE register, and gives a wrong result without any error.
    ``intp`` is ``int64`` in numba, so the check cannot distinguish a pointer-width int passed for ``%s`` or ``%p``
    from a real pointer. It does reject narrower ints, floats and booleans for those specifiers.
    """
    stripped = fmt_str.replace('%%', '\x00\x00')
    expected = []
    for m in _FMT_SPEC_RE.finditer(stripped):
        expected.extend(['int'] * m.group(0).count('*'))
        conv = m.group(1)
        if conv in _CONV_INT:
            expected.append('int')
        elif conv in _CONV_FLOAT:
            expected.append('float')
        elif conv == 's':
            expected.append('str')
        else:  # 'p'
            expected.append('ptr')
    arg_types = tuple(args_ty)
    if len(expected) != len(arg_types):
        raise TypingError(
            f"{name}: format {fmt_str!r} has {len(expected)} conversion "
            f"specifier(s) but {len(arg_types)} argument(s) were passed"
        )
    for i, (cls, ty) in enumerate(zip(expected, arg_types)):
        uty = unliteral(ty)
        if cls == 'int':
            ok = isinstance(uty, (Integer, Boolean))
        elif cls == 'float':
            ok = isinstance(uty, Float)
        elif cls == 'str':
            ok = isinstance(uty, UnicodeType) or uty in (intp, uintp)
        else:  # 'ptr'
            ok = uty in (intp, uintp)
        if not ok:
            raise TypingError(
                f"{name}: format specifier #{i + 1} in {fmt_str!r} expects "
                f"{cls}, but arg {i} has type {ty!r}"
            )


def _unpack_args_tuple(builder, args_ty, args_pack):
    """Extract the individual LLVM values from the LLVM aggregate of an argument tuple."""
    arg_types = tuple(args_ty)
    return [
        (arg_types[i], builder.extract_value(args_pack, i))
        for i in range(len(arg_types))
    ]


def _emit_variadic_call(builder, symbol, fmt_str, leading_vals, args_ty, args_pack, *, leading_tys):
    """Emit IR for ``symbol(*leading_vals, fmt_p, *promoted_args) -> i32``.

    ``leading_tys`` is the explicit list of LLVM types of the positional arguments that precede the format string in
    the libc signature, such as ``size_t``, ``FILE*`` or ``char*``. Callers pass it so that the function-type
    declaration states the libc ABI contract at the call site in the code generation. The implicit alternative,
    ``[v.type for v in leading_vals]``, would rely on numba's lowering to produce the right LLVM types. snprintf needs
    the explicit form for size_t: derive that type with ``intp_ll_type(context)``, which keeps it consistent with
    numba's intp lowering API.
    """
    i8p = llir.IntType(8).as_pointer()
    i32_ll = llir.IntType(32)
    mod = builder.module
    fmt_bytes = cgutils.make_bytearray((fmt_str + '\x00').encode('utf-8'))
    # cgutils.global_constant uses linkage='internal' and names the global through module.get_unique_name, which calls
    # scope.deduplicate. scope.deduplicate adds a suffix when a name repeats within a module (printf_format,
    # printf_format.1, ...). Internal-linkage globals are private to their module, and LLVM's linker renames them again
    # when it merges modules into the shared MCJIT engine. Each call site therefore has its own deduplicated global,
    # whether the format strings are the same or distinct.
    global_fmt = cgutils.global_constant(mod, f"{symbol}_format", fmt_bytes)
    fmt_p = builder.bitcast(global_fmt, i8p)
    unpacked = _unpack_args_tuple(builder, args_ty, args_pack)
    promoted = [_promote_for_varargs(builder, t, v) for t, v in unpacked]
    fn_ty = llir.FunctionType(i32_ll, list(leading_tys) + [i8p], var_arg=True)
    fn = get_or_insert_function(mod, fn_ty, symbol)
    return builder.call(fn, list(leading_vals) + [fmt_p] + promoted)


# ---------------------------------------------------------------------------
# Private @intrinsics: the @njit code generation path
# ---------------------------------------------------------------------------


@intrinsic(prefer_literal=True)
def _printf_intrinsic(typingctx, fmt_ty, args_ty):
    """Call libc printf with the arguments in one tuple. The intrinsic is internal; user code calls ``printf()``."""
    fmt_str = extract_literal_str("printf", fmt_ty, field="format string")
    _reject_percent_n_or_raise("printf", fmt_str)
    _reject_grouping_flag_or_raise("printf", fmt_str)
    if not isinstance(args_ty, BaseTuple):
        raise TypingError(f"printf: args must be a tuple, got {args_ty!r}")
    for i, ty in enumerate(tuple(args_ty)):
        _validate_writer_arg_type("printf", i, ty)
    _validate_format_vs_args("printf", fmt_str, args_ty)

    def codegen(context, builder, sig, llvm_args):
        _, args_pack = llvm_args
        return _emit_variadic_call(
            builder, "printf", fmt_str, [], args_ty, args_pack,
            leading_tys=[])

    return int32(fmt_ty, args_ty), codegen


@intrinsic(prefer_literal=True)
def _fprintf_intrinsic(typingctx, fp_ty, fmt_ty, args_ty):
    """Call libc fprintf with the arguments in one tuple. The intrinsic is internal; user code calls ``fprintf()``."""
    fmt_str = extract_literal_str("fprintf", fmt_ty, field="format string")
    _reject_percent_n_or_raise("fprintf", fmt_str)
    _reject_grouping_flag_or_raise("fprintf", fmt_str)
    if not isinstance(args_ty, BaseTuple):
        raise TypingError(f"fprintf: args must be a tuple, got {args_ty!r}")
    if fp_ty != intp:
        raise TypingError(
            f"fprintf: fp must be intp (FILE* as pointer-as-int), got {fp_ty!r}"
        )
    for i, ty in enumerate(tuple(args_ty)):
        _validate_writer_arg_type("fprintf", i, ty)
    _validate_format_vs_args("fprintf", fmt_str, args_ty)

    def codegen(context, builder, sig, llvm_args):
        i8p = llir.IntType(8).as_pointer()
        fp_int, _, args_pack = llvm_args
        fp_ptr = builder.inttoptr(fp_int, i8p)
        return _emit_variadic_call(
            builder, "fprintf", fmt_str, [fp_ptr], args_ty, args_pack,
            leading_tys=[i8p])

    return int32(fp_ty, fmt_ty, args_ty), codegen


@intrinsic(prefer_literal=True)
def _snprintf_intrinsic(typingctx, buf_ty, size_ty, fmt_ty, args_ty):
    """Call libc snprintf with the arguments in one tuple. The intrinsic is internal; user code calls ``snprintf()``."""
    fmt_str = extract_literal_str("snprintf", fmt_ty, field="format string")
    _reject_percent_n_or_raise("snprintf", fmt_str)
    _reject_grouping_flag_or_raise("snprintf", fmt_str)
    if not isinstance(args_ty, BaseTuple):
        raise TypingError(f"snprintf: args must be a tuple, got {args_ty!r}")
    if buf_ty != intp:
        raise TypingError(
            f"snprintf: buf must be intp (pointer-as-int), got {buf_ty!r}"
        )
    if size_ty != intp:
        raise TypingError(
            f"snprintf: size must be intp (size_t-as-int), got {size_ty!r}"
        )
    for i, ty in enumerate(tuple(args_ty)):
        _validate_writer_arg_type("snprintf", i, ty)
    _validate_format_vs_args("snprintf", fmt_str, args_ty)

    def codegen(context, builder, sig, llvm_args):
        i8p = llir.IntType(8).as_pointer()
        buf_int, size_val, _, args_pack = llvm_args
        buf_ptr = builder.inttoptr(buf_int, i8p)
        # size_t is pointer-width on all current 64-bit platforms. The shared helper derives its LLVM type from
        # numba's intp, so the libc snprintf signature stays correct if the platform changes, and it matches the
        # typing-time `size_ty != intp` guard above.
        return _emit_variadic_call(
            builder, _SNPRINTF_SYMBOL, fmt_str,
            [buf_ptr, size_val], args_ty, args_pack,
            leading_tys=[i8p, intp_ll_type(context)])

    return int32(buf_ty, size_ty, fmt_ty, args_ty), codegen


@intrinsic(prefer_literal=True)
def _sscanf_intrinsic(typingctx, buf_ty, fmt_ty, args_ty):
    """Call libc sscanf with the arguments in one tuple. The intrinsic is internal; user code calls ``sscanf()``.

    The arguments are intp output pointers. Pointers are not promoted, so no default argument promotion applies. See
    ``sscanf()`` for the contract that callers see.
    """
    fmt_str = extract_literal_str("sscanf", fmt_ty, field="format string")
    if not isinstance(args_ty, BaseTuple):
        raise TypingError(f"sscanf: args must be a tuple, got {args_ty!r}")
    if buf_ty != intp:
        raise TypingError(
            f"sscanf: buf must be intp (pointer-as-int), got {buf_ty!r}"
        )
    for i, ty in enumerate(tuple(args_ty)):
        if ty != intp:
            raise TypingError(
                f"sscanf: args[{i}] must be intp (output pointer), got {ty!r}"
            )

    def codegen(context, builder, sig, llvm_args):
        i8p = llir.IntType(8).as_pointer()
        buf_int, _, args_pack = llvm_args
        buf_ptr = builder.inttoptr(buf_int, i8p)
        return _emit_variadic_call(
            builder, "sscanf", fmt_str, [buf_ptr], args_ty, args_pack,
            leading_tys=[i8p])

    return int32(buf_ty, fmt_ty, args_ty), codegen


# ---------------------------------------------------------------------------
# @overload helper: build the source of an implementation that converts str arguments
# ---------------------------------------------------------------------------


def _build_args_tuple_expr_from_starargs(arg_tys):
    """Build a Python source fragment ``(...)`` that constructs the argument tuple for the intrinsic call.

    The fragment indexes into the implementation's ``*args``. It converts each ``UnicodeType`` argument with
    ``get_unicode_data_p``, so that libc receives a NUL-terminated C string for ``%s``.

    Numba requires the signature of the @overload implementation to have exactly the shape of the typing signature:
    ``*args`` in the typing function must be ``*args`` in the implementation. The implementation therefore cannot
    expand into explicit parameters per arity, and it indexes into the ``args`` tuple instead.
    """
    n = len(arg_tys)
    if n == 0:
        return "()"
    parts = []
    for i, ty in enumerate(arg_tys):
        # Both numba's runtime ``unicode_type`` and the compile-time ``Literal[str]`` (StringLiteral) need the
        # conversion. StringLiteral is not directly a subclass of UnicodeType: its MRO is ``StringLiteral``,
        # ``Literal``, ``Dummy``, ``Type``. ``unliteral(ty)`` therefore strips any Literal wrapping before the check.
        if isinstance(unliteral(ty), UnicodeType):
            parts.append(f"get_unicode_data_p(args[{i}])")
        else:
            parts.append(f"args[{i}]")
    inner = ", ".join(parts)
    return f"({inner},)" if n == 1 else f"({inner})"


def _build_overload_impl(name, fixed_params, args_tys, intrinsic_callable,
                         get_unicode_data_p):
    """Build an implementation function by running generated source through ``exec``.

    The generated implementation takes ``(fixed_params..., *args)``, which matches the shape of the typing function. It
    packs ``*args`` into a tuple with index expressions and converts each ``UnicodeType`` argument with
    ``get_unicode_data_p``. It then calls the underlying intrinsic with that tuple.
    """
    args_tuple_expr = _build_args_tuple_expr_from_starargs(args_tys)
    sig_params = ", ".join(list(fixed_params) + ["*args"])
    intrinsic_args = ", ".join(list(fixed_params) + [args_tuple_expr])
    src = f"def impl({sig_params}):\n    return _intr({intrinsic_args})\n"
    ns = {"_intr": intrinsic_callable, "get_unicode_data_p": get_unicode_data_p}
    exec(compile(src, f"<{name}-overload-impl>", "exec"), ns)  # nosec B102 - JIT codegen of internal source
    return ns["impl"]


# The import is deferred to avoid a circular import when the module loads.
def _get_unicode_data_p_lazy():
    from numbox.utils.lowlevel import get_unicode_data_p
    return get_unicode_data_p


# ---------------------------------------------------------------------------
# Pure-Python cache of stdio handle addresses, initialized on first use for fprintf routing
# ---------------------------------------------------------------------------


_PY_STREAM_BY_FP = None


def _ensure_py_stream_cache():
    global _PY_STREAM_BY_FP
    if _PY_STREAM_BY_FP is not None:
        return
    # The import is deferred, so that importing this module loads the stdio bindings only when the Python stream
    # mapping is used.
    from numbox.core.bindings.stdio import stdout, stderr, stdin
    _PY_STREAM_BY_FP = {
        int(stdout()): sys.stdout,
        int(stderr()): sys.stderr,
        int(stdin()):  sys.stdin,
    }


# ---------------------------------------------------------------------------
# Public Python-callable wrappers and their @overload registrations
# ---------------------------------------------------------------------------


def _reject_percent_n_in_python(name, fmt):
    """Raise ``ValueError`` if ``fmt`` contains ``%n``.

    This is the pure-Python equivalent of ``_reject_percent_n_or_raise``. Python's ``%`` operator would raise on
    ``%n`` by itself, because ``n`` is an unsupported format character, but its message is generic. This check gives
    users the same clear error message in Python and in @njit mode.
    """
    stripped = fmt.replace('%%', '\x00\x00')
    if _PERCENT_N_RE.search(stripped):
        raise ValueError(
            f"{name}: %n directive in format string {fmt!r} is not allowed "
            f"(writes byte-count-written through caller pointer; memory-"
            f"safety hazard). Use sscanf if you need %n's read-position "
            f"semantics."
        )


def _reject_grouping_flag_in_python(name, fmt):
    """Raise ``ValueError`` if ``fmt`` uses the glibc ``'`` grouping flag.

    This is the pure-Python equivalent of ``_reject_grouping_flag_or_raise``. Python's ``%`` operator would raise on
    ``%'d`` by itself, because ``'`` is an unsupported format character, but its message is generic. This check gives
    users the same clear error in Python and in @njit mode.
    """
    stripped = fmt.replace('%%', '\x00\x00')
    if _GROUPING_FLAG_RE.search(stripped):
        raise ValueError(
            f"{name}: the ' (thousands-grouping) flag in format string "
            f"{fmt!r} is not supported (glibc-only, absent from C99 and "
            f"Python's % operator). Remove it or group the digits yourself."
        )


def printf(fmt, *args):
    """Write ``fmt`` formatted with ``args`` to stdout, as C ``printf(fmt, *args)`` does, in plain Python and @njit.

    In plain Python, the function strips the C length modifiers (for example, ``%lld`` becomes ``%d``) and writes to
    ``sys.stdout`` through ``str.__mod__``.

    Under @njit, the ``@overload`` below converts each ``unicode_type`` argument with ``get_unicode_data_p``, so that
    libc's ``%s`` receives a NUL-terminated C string. It then routes the call to the private ``_printf_intrinsic``.
    The format string must be a literal under @njit; the module docstring describes the caveats.

    Both modes reject ``%n`` (see ``_reject_percent_n_or_raise``).

    The function returns the number of bytes written, or its pure-Python equivalent, as int32.
    """
    _reject_percent_n_in_python("printf", fmt)
    _reject_grouping_flag_in_python("printf", fmt)
    text = _python_fmt_compat(fmt) % args
    sys.stdout.write(text)
    sys.stdout.flush()
    return len(text.encode('utf-8'))


@overload(printf)
def _overload_printf(fmt, *args):
    fmt_str = extract_literal_str("printf", fmt, field="format string")
    _reject_percent_n_or_raise("printf", fmt_str)
    _reject_grouping_flag_or_raise("printf", fmt_str)
    for i, ty in enumerate(args):
        _validate_writer_arg_type("printf", i, ty)
    impl = _build_overload_impl(
        "printf", ["fmt"], args, _printf_intrinsic,
        _get_unicode_data_p_lazy(),
    )
    return impl


def fprintf(fp, fmt, *args):
    """Write ``fmt`` formatted with ``args`` to ``fp``, as C ``fprintf(fp, fmt, *args)`` does, in both modes.

    ``fp`` is a FILE\\* as an ``intp``, returned by ``stdout()``, ``stderr()``, ``stdin()`` or ``fopen()``.

    In plain Python, the function maps the ``stdout()``, ``stderr()`` and ``stdin()`` handles to the corresponding
    ``sys.*`` streams through an address cache. A call with any other FILE\\*, such as one returned by ``fopen``,
    raises ``RuntimeError``. For file I/O on the Python side, use ``open()`` and ``f.write()``.

    Under @njit, the ``@overload`` converts str arguments automatically and routes the call to ``_fprintf_intrinsic``.
    """
    _reject_percent_n_in_python("fprintf", fmt)
    _reject_grouping_flag_in_python("fprintf", fmt)
    _ensure_py_stream_cache()
    py_stream = _PY_STREAM_BY_FP.get(int(fp))
    if py_stream is None:
        raise RuntimeError(
            f"fprintf in pure-Python mode only supports stdout / stderr / "
            f"stdin handles; got fp={int(fp):#x}. For arbitrary FILE* "
            f"(e.g. fopen) use @njit, or use Python's open() + write()."
        )
    text = _python_fmt_compat(fmt) % args
    py_stream.write(text)
    py_stream.flush()
    return len(text.encode('utf-8'))


@overload(fprintf)
def _overload_fprintf(fp, fmt, *args):
    fmt_str = extract_literal_str("fprintf", fmt, field="format string")
    _reject_percent_n_or_raise("fprintf", fmt_str)
    _reject_grouping_flag_or_raise("fprintf", fmt_str)
    if fp != intp:
        raise TypingError(
            f"fprintf: fp must be intp (FILE* as pointer-as-int), got {fp!r}"
        )
    for i, ty in enumerate(args):
        _validate_writer_arg_type("fprintf", i, ty)
    impl = _build_overload_impl(
        "fprintf", ["fp", "fmt"], args, _fprintf_intrinsic,
        _get_unicode_data_p_lazy(),
    )
    return impl


def snprintf(buf_p, size, fmt, *args):
    """Write ``fmt`` formatted with ``args`` into the buffer at ``buf_p``, as C ``snprintf`` does, in both modes.

    The call is ``snprintf(buf_p, size, fmt, *args)``. ``buf_p`` is an ``intp`` pointer to the destination buffer,
    which the caller owns. It is typically ``array_data_p(numpy_array)``, a helper that works in both modes.

    The function returns, as int32, the number of characters that it would have written if ``size`` were unlimited,
    excluding the trailing NUL. Under @njit on Windows, the return value after truncation differs, as the module
    docstring describes. Python and @njit on Linux and macOS follow C99, while @njit on Windows uses MSVCRT
    ``_snprintf``, which returns ``-1`` on truncation.
    """
    _reject_percent_n_in_python("snprintf", fmt)
    _reject_grouping_flag_in_python("snprintf", fmt)
    text_bytes = (_python_fmt_compat(fmt) % args).encode('utf-8')
    n_would_have = len(text_bytes)
    if size > 0:
        n_write = min(n_would_have, size - 1)
        # Slice the content before appending the NUL, so that the NUL is at the correct position even when the text is
        # truncated. Without the slice, ``memmove(buf_p, text_bytes + b'\x00', n_write + 1)`` would leave the NUL out
        # of the copied prefix on truncation, and buf would hold n_write content bytes with no terminator.
        src = text_bytes[:n_write] + b'\x00'
        ctypes.memmove(buf_p, src, len(src))
    return n_would_have


@overload(snprintf)
def _overload_snprintf(buf_p, size, fmt, *args):
    fmt_str = extract_literal_str("snprintf", fmt, field="format string")
    _reject_percent_n_or_raise("snprintf", fmt_str)
    _reject_grouping_flag_or_raise("snprintf", fmt_str)
    if buf_p != intp:
        raise TypingError(
            f"snprintf: buf must be intp (pointer-as-int), got {buf_p!r}"
        )
    if size != intp:
        raise TypingError(
            f"snprintf: size must be intp (size_t-as-int), got {size!r}"
        )
    for i, ty in enumerate(args):
        _validate_writer_arg_type("snprintf", i, ty)
    impl = _build_overload_impl(
        "snprintf", ["buf_p", "size", "fmt"], args, _snprintf_intrinsic,
        _get_unicode_data_p_lazy(),
    )
    return impl


def sscanf(buf, fmt, *args):
    """Parse the string at ``buf`` with ``fmt`` into the output pointers ``args`` under @njit, as C ``sscanf`` does.

    The call is ``sscanf(buf, fmt, *args)``, and the binding is @njit-only. The arguments are intp output pointers,
    typically ``array_data_p`` of a 1-element numpy array of the right dtype. The docstring of ``_sscanf_intrinsic``,
    which applies under @njit only, gives the contract between the pointers and the specifiers.

    In pure Python, this binding raises ``NotImplementedError``. For parsing in pure Python, use ``int()``,
    ``float()`` or ``re``.
    """
    raise NotImplementedError(
        "sscanf is @njit-only; wrap the call in @njit, or use Python's "
        "int() / float() / re for pure-Python parsing"
    )


@overload(sscanf)
def _overload_sscanf(buf, fmt, *args):
    extract_literal_str("sscanf", fmt, field="format string")  # validates Literal[str]
    if buf != intp:
        raise TypingError(
            f"sscanf: buf must be intp (pointer-as-int), got {buf!r}"
        )
    # The sscanf arguments are output pointers, so each must be intp. They are not converted, and pointers are not
    # promoted. The implementation reuses the helper, but get_unicode_data_p converts nothing, and the intp arguments
    # pass through unchanged.
    for i, ty in enumerate(args):
        if ty != intp:
            raise TypingError(
                f"sscanf: args[{i}] must be intp (output pointer), got {ty!r}"
            )
    impl = _build_overload_impl(
        "sscanf", ["buf", "fmt"], args, _sscanf_intrinsic,
        _get_unicode_data_p_lazy(),  # unused (no UnicodeType in args)
    )
    return impl
