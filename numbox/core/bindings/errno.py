from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import int32, int64, intp, void
from numba.extending import intrinsic

from numbox.core.bindings.utils import platform_, load_lib
from numbox.core.configurations import jit_options
from numbox.core.proxy.proxy import proxy
from numbox.utils.lowlevel import load_at, store_at


__all__ = ["errno_get", "errno_set"]


load_lib("c")


_ERRNO_ACCESSOR = {
    "Linux": "__errno_location",
    "Darwin": "__error",
    "Windows": "_errno",
}


@intrinsic
def _errno_ptr(typingctx):
    sym = _ERRNO_ACCESSOR.get(platform_)
    if sym is None:
        raise TypingError(
            f"_errno_ptr: unsupported platform {platform_!r}")

    def codegen(context, builder, signature, arguments):
        intp_ll = context.get_value_type(intp)
        i32_ptr = llir.IntType(32).as_pointer()
        # This declaration deliberately does not set readnone or memory(none), although glibc declares
        # __errno_location with __attribute_const__, which maps to LLVM's readnone. With that attribute, LLVM could
        # hoist the call out of loops or across function-call boundaries. The result would be incorrect when the OS
        # thread changes, e.g., across @njit(parallel=True) workers. The call must therefore stay opaque, with
        # possible side effects.
        func_ty = llir.FunctionType(i32_ptr, [])
        func_p = get_or_insert_function(builder.module, func_ty, sym)
        ptr = builder.call(func_p, [])
        return builder.ptrtoint(ptr, intp_ll)
    return intp(), codegen


@proxy(int32(), jit_options=jit_options)
def errno_get():
    """Return the current thread's errno as int32.

    The function resolves the per-thread errno location again on every call. On @njit(parallel=True) workers, the
    accessor therefore returns the errno of the calling worker. A normal @njit function runs on the OS thread of its
    Python caller, so the caller observes an errno that the function sets. A worker of a parallel region runs on a
    different OS thread, so the Python caller does not observe an errno set there.
    """
    return load_at(_errno_ptr(), int32)


@proxy(void(int64), jit_options=jit_options)
def errno_set(v):
    """Set the current thread's errno to v.

    The function accepts int64, Python's default integer width. It narrows the value to int32 before it stores the
    value at the per-thread errno location, which matches C's ``int errno``.
    """
    store_at(_errno_ptr(), int32(v))
