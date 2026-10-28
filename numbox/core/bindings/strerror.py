"""Thread-safe ``strerror_safe(errnum, buf, buflen)``, callable from @njit.

The supported libcs are listed below. Other Linux libcs are not supported.

- **glibc**: `__xpg_strerror_r <https://codebrowser.dev/glibc/glibc/string/xpg-strerror.c.html>`_, the POSIX XSI
  form, present on glibc 2.3.4 and later, which shipped in 2004.
- **musl**: ``__xpg_strerror_r`` as well. musl declares ``strerror_r`` as the POSIX XSI form, and it exports
  ``__xpg_strerror_r`` as a `weak alias <https://git.musl-libc.org/cgit/musl/tree/src/string/strerror_r.c>`_ to the
  same implementation. The same symbol therefore resolves on glibc and on musl.
- **macOS**: `strerror_r
  <https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/strerror_r.3.html>`_
  in the POSIX form.
- **Windows**: `strerror_s
  <https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/strerror-s-strerror-s-wcserror-s-wcserror-s>`_,
  with its arguments reordered as (buffer, size, errnum).

On glibc, plain ``strerror_r`` is the GNU form, which returns ``char *`` and would not match the POSIX-form IR of
this module. The Linux symbol selector always chooses ``__xpg_strerror_r``, never the GNU form. The selector keeps a
``strerror_r`` fallback as defense in depth, in case a future libc drops the ``__xpg_strerror_r`` symbol. The
fallback is unreachable on every libc currently supported (verified by the Alpine ``musl_symbol_check`` CI canary).
"""
import llvmlite.binding as ll
from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import int32, int64, intp
from numba.extending import intrinsic

from numbox.core.bindings.utils import intp_ll_type, platform_, load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy


__all__ = ["strerror_safe"]


load_lib("c")


def _select_posix_symbol():
    if platform_ == "Linux":
        if ll.address_of_symbol("__xpg_strerror_r") is not None:
            return "__xpg_strerror_r"
        # This fallback is defense in depth, in case a future libc drops the __xpg_strerror_r symbol. It is currently
        # unreachable: glibc 2.3.4+ exports the symbol directly, and musl exports it as a weak alias to its own
        # POSIX-form strerror_r. On glibc, plain strerror_r is the GNU form that returns char*, and
        # it would not match the IR signature of this module, so this line must never run on glibc.
        return "strerror_r"
    if platform_ == "Darwin":
        return "strerror_r"
    return None


@intrinsic
def _strerror_safe(typingctx, errnum_ty, buf_ty, buflen_ty):
    # The underlying libc functions (strerror_r, __xpg_strerror_r and strerror_s) all take a 32-bit int for errnum.
    # The intrinsic enforces int32 at typing time. If a future caller omits the explicit cast in the public
    # strerror_safe wrapper, typing then fails with a clear TypingError instead of an opaque type mismatch at IR
    # lowering.
    if errnum_ty != int32:
        raise TypingError(
            f"_strerror_safe: errnum must be int32, got {errnum_ty!r}")
    if platform_ == "Windows":
        sym = "strerror_s"
    else:
        sym = _select_posix_symbol()
        if sym is None:
            raise TypingError(
                f"_strerror_safe: unsupported platform {platform_!r}")

    def codegen(context, builder, signature, arguments):
        errnum, buf_p, buflen = arguments
        i32 = llir.IntType(32)
        i8p = llir.IntType(8).as_pointer()
        size_t_ll = intp_ll_type(context)
        buf = builder.inttoptr(buf_p, i8p)
        if platform_ == "Windows":
            func_ty = llir.FunctionType(i32, [i8p, size_t_ll, i32])
            func_p = get_or_insert_function(builder.module, func_ty, sym)
            return builder.call(func_p, [buf, buflen, errnum])
        func_ty = llir.FunctionType(i32, [i32, i8p, size_t_ll])
        func_p = get_or_insert_function(builder.module, func_ty, sym)
        return builder.call(func_p, [errnum, buf, buflen])
    return int32(errnum_ty, buf_ty, buflen_ty), codegen


def _render_ir_for_probe():
    """Render the IR that _strerror_safe would emit for a probe call.

    The IR-inspection test (test_strerror_safe.py) uses this function to verify that the chosen symbol is strerror_r,
    not __xpg_strerror_r, when ll.address_of_symbol("__xpg_strerror_r") returns None. The probe bypasses end-to-end
    execution, because direct inspection of the text is the safe way to verify the choice.

    The function uses ``intp_ll_type(None)``, the same shared helper that the actual code generation calls with a
    ``context``. Both paths therefore derive the size_t LLVM width from one place. numba lowers intp to
    ``IntType(intp.bitwidth)``, so the paths with and without a context produce identical types.
    """
    module = llir.Module(name="probe")
    i32 = llir.IntType(32)
    i8p = llir.IntType(8).as_pointer()
    sym = _select_posix_symbol()
    func_ty = llir.FunctionType(i32, [i32, i8p, intp_ll_type()])
    get_or_insert_function(module, func_ty, sym)
    return str(module)


@proxy(int32(int64, intp, intp), jit_options=jit_options)
def strerror_safe(errnum, buf, buflen):
    """Write the error message for errnum into buf, whose length is buflen.

    The function returns 0 on success and a positive errno on failure: ERANGE for a short buffer, and EINVAL for an
    unknown errnum. It is thread-safe on all supported platforms. The cross-platform dispatch happens at lowering time:
    __xpg_strerror_r on Linux, glibc and musl alike, strerror_r on macOS, and strerror_s on Windows, with the arguments
    reordered.
    """
    return _strerror_safe(int32(errnum), buf, buflen)
