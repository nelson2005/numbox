import llvmlite.binding as ll
from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.types import int32, int64, intp
from numba.extending import intrinsic

from numbox.core.bindings.utils import platform_, load_lib
from numbox.utils.highlevel import cres


load_lib("c")


def _select_posix_symbol():
    if platform_ == "Linux":
        if ll.address_of_symbol("__xpg_strerror_r") is not None:
            return "__xpg_strerror_r"
        return "strerror_r"
    if platform_ == "Darwin":
        return "strerror_r"
    return None


@intrinsic
def _strerror_safe(typingctx, errnum_ty, buf_ty, buflen_ty):
    def codegen(context, builder, signature, arguments):
        errnum, buf_p, buflen = arguments
        i32 = llir.IntType(32)
        i8p = llir.IntType(8).as_pointer()
        size_t_ll = context.get_value_type(intp)
        if platform_ == "Windows":
            func_ty = llir.FunctionType(i32, [i8p, size_t_ll, i32])
            func_p = get_or_insert_function(
                builder.module, func_ty, "strerror_s")
            buf = builder.inttoptr(buf_p, i8p)
            return builder.call(func_p, [buf, buflen, errnum])
        sym = _select_posix_symbol()
        if sym is None:
            raise RuntimeError(
                f"_strerror_safe: unsupported platform {platform_!r}")
        func_ty = llir.FunctionType(i32, [i32, i8p, size_t_ll])
        func_p = get_or_insert_function(builder.module, func_ty, sym)
        buf = builder.inttoptr(buf_p, i8p)
        return builder.call(func_p, [errnum, buf, buflen])
    return int32(errnum_ty, buf_ty, buflen_ty), codegen


def _render_ir_for_probe():
    """Render the IR _strerror_safe would emit for a probe call.

    Used by the IR-inspection test (test_strerror_safe.py) to verify
    that when ll.address_of_symbol("__xpg_strerror_r") returns None,
    the chosen symbol is strerror_r and not __xpg_strerror_r. Bypasses
    end-to-end execution: on glibc, strerror_r is the GNU form (returns
    char*) — calling it under our POSIX-shaped IR would corrupt return
    reads. Direct text inspection is the safe verification.
    """
    module = llir.Module(name="probe")
    i32 = llir.IntType(32)
    i8p = llir.IntType(8).as_pointer()
    sym = _select_posix_symbol()
    # llir.IntType(64) mirrors _strerror_safe's context.get_value_type(intp);
    # intp == i64 on every 64-bit target we ship. Hardcoded here because no
    # JIT context is available outside the intrinsic's codegen call.
    func_ty = llir.FunctionType(i32, [i32, i8p, llir.IntType(64)])
    get_or_insert_function(module, func_ty, sym)
    return str(module)


@cres(int32(int64, intp, intp), cache=True)
def strerror_safe(errnum, buf, buflen):
    """Write the error message for errnum into buf (length buflen).

    Returns 0 on success, positive errno (ERANGE on short buffer,
    EINVAL on unknown errnum) on failure. Thread-safe on all supported
    platforms. Cross-platform dispatch happens at lowering time:
    __xpg_strerror_r on glibc, strerror_r on musl / macOS, strerror_s
    on Windows (with arg reorder).
    """
    return _strerror_safe(int32(errnum), buf, buflen)
