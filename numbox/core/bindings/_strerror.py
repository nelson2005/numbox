"""Thread-safe ``strerror_safe(errnum, buf, buflen)`` callable from @njit.

Supported libcs (other Linux libcs are not supported):

- **glibc** — ``__xpg_strerror_r`` (always present on glibc 2.0+; POSIX-form)
- **musl** — ``strerror_r`` (musl's ``strerror_r`` is POSIX-form, verified by the
  Alpine ``musl_symbol_check`` CI canary)
- **macOS** — ``strerror_r`` (POSIX-form)
- **Windows** — ``strerror_s`` with reordered args (buffer, size, errnum)

On glibc, ``strerror_r`` is the GNU form (returns ``char *``) and would not
match this module's POSIX-shaped IR. The Linux symbol selector falls back to
``strerror_r`` only when ``__xpg_strerror_r`` is absent — i.e. on musl. On
glibc the fallback is unreachable in practice.
"""
import llvmlite.binding as ll
from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import int32, int64, intp
from numba.extending import intrinsic

from numbox.core.bindings.utils import platform_, load_lib
from numbox.utils.highlevel import cres_cacheable


__all__ = ["strerror_safe"]


load_lib("c")


def _select_posix_symbol():
    if platform_ == "Linux":
        if ll.address_of_symbol("__xpg_strerror_r") is not None:
            return "__xpg_strerror_r"
        # Fallback intended for musl (POSIX-form strerror_r). glibc 2.0+ always
        # exports __xpg_strerror_r, so this branch is unreachable on glibc.
        return "strerror_r"
    if platform_ == "Darwin":
        return "strerror_r"
    return None


@intrinsic
def _strerror_safe(typingctx, errnum_ty, buf_ty, buflen_ty):
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
        size_t_ll = context.get_value_type(intp)
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
    """Render the IR _strerror_safe would emit for a probe call.

    Used by the IR-inspection test (test_strerror_safe.py) to verify
    that when ll.address_of_symbol("__xpg_strerror_r") returns None,
    the chosen symbol is strerror_r and not __xpg_strerror_r. Bypasses
    end-to-end execution: direct text inspection is the safe verification.
    """
    module = llir.Module(name="probe")
    i32 = llir.IntType(32)
    i8p = llir.IntType(8).as_pointer()
    sym = _select_posix_symbol()
    # intp.bitwidth mirrors _strerror_safe's context.get_value_type(intp)
    # without needing a JIT context, which isn't available outside codegen.
    func_ty = llir.FunctionType(i32, [i32, i8p, llir.IntType(intp.bitwidth)])
    get_or_insert_function(module, func_ty, sym)
    return str(module)


@cres_cacheable(int32(int64, intp, intp))
def strerror_safe(errnum, buf, buflen):
    """Write the error message for errnum into buf (length buflen).

    Returns 0 on success, positive errno (ERANGE on short buffer,
    EINVAL on unknown errnum) on failure. Thread-safe on all supported
    platforms. Cross-platform dispatch happens at lowering time:
    __xpg_strerror_r on glibc, strerror_r on musl / macOS, strerror_s
    on Windows (with arg reorder).
    """
    return _strerror_safe(int32(errnum), buf, buflen)
