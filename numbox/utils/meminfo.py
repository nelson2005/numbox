import ctypes

import numba

from llvmlite import ir as llir
from numba import types
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.core.extending import intrinsic


return_tuple_type = types.UniTuple(types.intp, 2)


@intrinsic
def _structref_meminfo(typingctx, s_type):

    def codegen(context, builder, signature, args):
        s = args[0]
        meminfo = context.nrt.get_meminfos(builder, s_type, s)[0]
        type_data, meminfo_p = meminfo

        meminfo_p_as_int = builder.ptrtoint(meminfo_p, context.get_data_type(numba.intp))
        data_p = context.nrt.meminfo_data(builder, meminfo_p)
        data_p_as_int = builder.ptrtoint(data_p, context.get_data_type(numba.intp))

        return context.make_tuple(builder, return_tuple_type, (meminfo_p_as_int, data_p_as_int))

    sig = return_tuple_type(s_type)
    return sig, codegen


@numba.njit
def structref_meminfo(s):
    """
    :param s: instance of StructRef
    :return: tuple of pointers (as 64-bit integers) to meminfo member of the StructRef and its data payload.
    """
    return _structref_meminfo(s)


def get_nrt_refcount(s):
    """
    :param s: instance of an object with payload wrapped in a meminfo
    :return: NRT reference count, the one which is controllable by nrt context's incref/decref

    Leverages meminfo memory layout::

        struct MemInfo {
            size_t            refct;
            NRT_dtor_function dtor;
            void              *dtor_info;
            void              *data;
            size_t            size;    /* only used for NRT allocated memory */
            NRT_ExternalAllocator *external_allocator;
        };

    see `nrt <https://github.com/numba/numba/blob/main/numba/core/runtime/nrt.cpp#L17>`_.
    """
    meminfo_p, data_p = structref_meminfo(s)
    return ctypes.c_int64.from_address(meminfo_p).value


_MI_TY = types.MemInfoPointer(types.voidptr)


def _require_intp(p_ty, fn_name):
    """Reject anything other than ``types.intp`` for pointer-as-int arguments.

    Accepting an arbitrary integer type would emit an ``inttoptr`` of the wrong width on 64-bit hosts: for example, an
    int32 lowers to an i32, and the pointer made from it is truncated.
    """
    if p_ty != types.intp:
        raise TypingError(f"{fn_name} expects types.intp, got {p_ty}")


@intrinsic
def _incref_meminfo(typingctx, p_ty):
    """Incref a MemInfo at ``intp`` via NRT.

    This intrinsic's signature (``void(intp)``) contains no NRT-tracked types. A caller whose signature also contains
    none passes the legality check of numba's ``removerefctpass``. The pass then deletes every ``NRT_incref`` and
    ``NRT_decref`` call in that caller by name, including this one. ``_release_meminfo`` avoids the pass by calling an
    NRT function outside the pass's allowlist, but that is not possible for an incref, because numba registers no
    such symbol. Instead, the codegen tags the enclosing function with the ``numba_args_may_always_need_nrt`` named
    metadata. That metadata is numba's own marker for functions whose references outlive the call, and it makes
    ``_legalize()`` refuse the rewrite for that function's module. The tag can only prevent the rewrite, so it changes
    nothing if another check already fails. Each function is lowered into a module of its own, so a caller's check
    never finds the tag. numba 0.66 removed the pass together with every mention of the tag, so from 0.66 on nothing
    reads it.
    """
    _require_intp(p_ty, "_incref_meminfo")
    sig = types.void(p_ty)

    def codegen(context, builder, signature, args):
        mi_ll_ty = context.get_value_type(_MI_TY)
        meminfo = builder.inttoptr(args[0], mi_ll_ty)
        context.nrt.incref(builder, _MI_TY, meminfo)
        module = builder.module
        nmd = module.add_named_metadata("numba_args_may_always_need_nrt")
        nmd.add(module.add_metadata([builder.function]))
    return sig, codegen


@intrinsic
def _release_meminfo(typingctx, p_ty):
    """Decref a MemInfo at ``intp`` via ``NRT_MemInfo_release``.

    The intrinsic calls ``NRT_MemInfo_release`` directly rather than ``context.nrt.decref``. Its signature
    (``void(intp)``) contains no NRT-tracked types, so numba's ``removerefctpass`` rewrite would strip a plain
    ``NRT_decref`` as dead code. ``NRT_MemInfo_release`` is not in the pass's accepted-nrtfns allowlist, so calling it
    keeps ``_legalize()`` from enabling the rewrite at all.
    """
    _require_intp(p_ty, "_release_meminfo")
    sig = types.void(p_ty)

    def codegen(context, builder, signature, args):
        ptr_ty = llir.IntType(8).as_pointer()
        fnty = llir.FunctionType(llir.VoidType(), [ptr_ty])
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, "NRT_MemInfo_release")
        meminfo = builder.inttoptr(args[0], ptr_ty)
        builder.call(fn, [meminfo])
    return sig, codegen


@intrinsic
def _deref_structref_raw_ptr(typingctx, struct_type_ref, p_ty):
    """Reconstruct a structref value from an ``intp`` MemInfo pointer."""
    _require_intp(p_ty, "_deref_structref_raw_ptr")
    inst_type = struct_type_ref.instance_type
    sig = inst_type(struct_type_ref, p_ty)

    def codegen(context, builder, signature, args):
        p_val = args[1]
        mi_ll_ty = context.get_value_type(_MI_TY)
        meminfo = builder.inttoptr(p_val, mi_ll_ty)
        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo
        return st._getvalue()
    return sig, codegen


@numba.njit
def borrow_structref(struct_type, p):
    """Incref an ``intp`` MemInfo pointer and reconstruct a structref from it.

    The caller receives a live structref whose reference count NRT manages normally. The external owner's count is
    unchanged when the caller's scope exits without additional actions, because the local decref on drop balances the
    incref.
    """
    _incref_meminfo(p)
    return _deref_structref_raw_ptr(struct_type, p)


@numba.njit
def export_meminfo(s):
    """Export a structref as an ``intp`` MemInfo pointer with a +1 incref.

    The returned ``intp`` keeps the allocation alive until the caller
    balances it with ``release_meminfo``.
    """
    meminfo_p, _ = structref_meminfo(s)
    _incref_meminfo(meminfo_p)
    return meminfo_p


@numba.njit
def release_meminfo(p):
    """Decref a MemInfo at ``intp`` via NRT_MemInfo_release.

    This is the public njit wrapper over ``_release_meminfo``, callable from Python and from ``@njit`` contexts.
    """
    _release_meminfo(p)
