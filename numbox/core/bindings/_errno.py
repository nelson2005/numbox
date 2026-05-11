from llvmlite import ir as llir
from numba.core.cgutils import get_or_insert_function
from numba.core.types import int32, int64, intp, void
from numba.extending import intrinsic

from numbox.core.bindings.utils import platform_, load_lib
from numbox.utils.highlevel import cres


load_lib("c")


_ERRNO_ACCESSOR = {
    "Linux": "__errno_location",
    "Darwin": "__error",
    "Windows": "_errno",
}


@intrinsic
def _errno_ptr(typingctx):
    def codegen(context, builder, signature, arguments):
        intp_ll = context.get_value_type(intp)
        i32_ptr = llir.IntType(32).as_pointer()
        sym = _ERRNO_ACCESSOR.get(platform_)
        if sym is None:
            raise RuntimeError(
                f"_errno_ptr: unsupported platform {platform_!r}")
        func_ty = llir.FunctionType(i32_ptr, [])
        func_p = get_or_insert_function(builder.module, func_ty, sym)
        ptr = builder.call(func_p, [])
        return builder.ptrtoint(ptr, intp_ll)
    return intp(), codegen


@intrinsic
def _load_int32_at(typingctx, p_ty):
    def codegen(context, builder, signature, arguments):
        (p,) = arguments
        i32_ptr = llir.IntType(32).as_pointer()
        ptr = builder.inttoptr(p, i32_ptr)
        return builder.load(ptr)
    return int32(p_ty), codegen


@intrinsic
def _store_int32_at(typingctx, p_ty, v_ty):
    def codegen(context, builder, signature, arguments):
        p, v = arguments
        i32_ptr = llir.IntType(32).as_pointer()
        ptr = builder.inttoptr(p, i32_ptr)
        builder.store(v, ptr)
    return void(p_ty, v_ty), codegen


@cres(int32(), cache=True)
def errno_get():
    """Return the current thread's errno as int32.

    Re-resolves the per-thread errno location on every call: on
    @njit(parallel=True) workers, the accessor returns that worker's
    errno. The Python caller cannot observe an errno value set inside a
    parallel region after return — different OS thread.
    """
    return _load_int32_at(_errno_ptr())


@cres(void(int64), cache=True)
def errno_set(v):
    """Set the current thread's errno to v.

    Accepts int64 (Python's default integer width); the value is
    narrowed to int32 before being stored at the per-thread errno
    location, matching C's ``int errno``.
    """
    _store_int32_at(_errno_ptr(), int32(v))
