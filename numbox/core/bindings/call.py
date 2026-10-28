import llvmlite.binding as ll
from llvmlite import ir as llir

from numba.core.cgutils import get_or_insert_function
from numba.core.errors import TypingError
from numba.core.types import BaseTuple, NoneType, Record
from numba.extending import intrinsic

from numbox.core.bindings.abi import (
    _CLASS_SCALAR, _CLASS_STRUCT_SMALL, _CLASS_STRUCT_LARGE,
    _EIGHTBYTE_CLASS_INTEGER,
    _PLATFORM_AAPCS64, _PLATFORM_SYSV_X86_64, _PLATFORM_WIN_X64,
    _classify, _classify_eightbytes, _current_platform,
    _is_canonical_int64_pair_layout, _is_windows_register_passable,
    _struct_bytes,
)
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import extract_literal_str


@intrinsic(prefer_literal=True)
def _call_lib_func(typingctx, func_name_ty, args_ty=NoneType):
    """Call a C library function with ABI-correct argument and return passing.

    The intrinsic resolves the C function name in numbox's ``signatures`` dict. It classifies each argument in
    ``args_ty`` and the resolved return type as a scalar, a struct of at most 16 bytes or a struct of more than 16
    bytes. It then lowers each of them to LLVM IR according to the calling convention of the host:

    - **Scalar arguments and returns** are passed and returned directly.
    - **Struct arguments of at most 16 bytes** are passed by value on SysV x86-64 and AAPCS64, where LLVM's frontend
      lowers them to register passing. On Windows x64, the ABI passes sizes 1, 2, 4 and 8 by value in registers, and
      other sizes by pointer, through an alloca and a store. On SysV x86-64 and AAPCS64, a 16-byte aggregate is
      repacked through a memory bitcast before the call if its eightbytes are pure-INTEGER but its LLVM type is not
      already ``{i64, i64}``, for example ``{i32, i32, i64}`` (duckdb_interval). The repack compensates for a
      limitation of llvmlite: llvmlite does not model the eightbyte-packing rule, and without the repack it loses
      fields.
    - **Struct arguments of more than 16 bytes** are passed by pointer on every platform. The SysV x86-64 lowering
      adds the ``byval`` attribute to the LLVM argument, and ``optnone`` and ``noinline`` to the enclosing function.
      These attributes prevent the LLVM optimizer from eliding the caller-side stack copy before the callee reads it.
      See https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    - **Struct returns of at most 16 bytes** are returned directly on SysV x86-64 and AAPCS64. On Windows x64, sizes
      1, 2, 4 and 8 return directly in RAX, and other sizes use ``sret``: a hidden first argument that the caller
      allocates, with a void return. On SysV and AAPCS64, a 16-byte non-canonical INT/INT return receives the same
      eightbyte repack. The LLVM call is declared to return ``{i64, i64}``, and a memory bitcast unpacks the result
      back to the original LLVM type.
    - **Struct returns of more than 16 bytes** use ``sret`` on every platform, with a hidden first argument that the
      caller allocates and a void return. SysV x86-64, AAPCS64 and Windows x64 all use an indirect result location for
      this size class. The code generation path is shared with the Windows case of a ``<=16-byte`` struct that is not
      register-passable. A ``Record`` return of this size is explicitly rejected with ``TypingError``. numba's
      ``RecordModel`` represents values as raw pointers, so a stack-alloca sret slot would dangle after the ``@njit``
      function returns. Safe support needs an NRT-allocated buffer and integration with numba's record-ownership
      model. That support can be added when a consumer needs it.

    This intrinsic handles a C signature ``func(T)``, including one that the ABI lowers to a byval pointer. A C
    signature of the form ``func(T*)``, which takes a pointer to a struct, needs the sibling intrinsic
    ``_call_lib_func_byval`` in this module instead. The numba type system cannot distinguish ``T`` from ``T*``, so
    the caller chooses the intrinsic from what the C header declares.
    """
    func_name = extract_literal_str("_call_lib_func", func_name_ty, field="func_name")
    func_p_as_int = ll.address_of_symbol(func_name)
    if func_p_as_int is None:
        raise TypingError(f"{func_name} is unavailable in the LLVM context")
    func_sig = signatures.get(func_name, None)
    if func_sig is None:
        raise TypingError(f"Undefined signature for {func_name}")

    ret_ty = func_sig.return_type
    ret_class = _classify(ret_ty)
    if ret_class == _CLASS_STRUCT_LARGE and isinstance(ret_ty, Record):
        raise TypingError(
            f"_call_lib_func: Record returns >16 bytes are not yet "
            f"supported ({func_name}). Numba's RecordModel represents "
            f"values as raw pointers, so the natural stack-alloca sret "
            f"slot would dangle after the @njit function returns. Safe "
            f"support needs an NRT-allocated buffer hooked into numba's "
            f"record-ownership model. Use a Tuple/UniTuple return shape "
            f"instead, or open an issue if you need Record."
        )

    if args_ty == NoneType:
        arg_types = ()
        arg_classes = ()
    elif isinstance(args_ty, BaseTuple):
        arg_types = tuple(args_ty)
        arg_classes = tuple(_classify(at) for at in arg_types)
    else:
        arg_types = (args_ty,)
        arg_classes = (_classify(args_ty),)

    plat = _current_platform()
    use_sret = (
        ret_class == _CLASS_STRUCT_LARGE
        or (
            ret_class == _CLASS_STRUCT_SMALL
            and plat == _PLATFORM_WIN_X64
            and not _is_windows_register_passable(
                _struct_bytes(ret_ty, "_call_lib_func"))
        )
    )
    needs_ret_repack = (
        ret_class == _CLASS_STRUCT_SMALL
        and not use_sret
        and _needs_int_int_eightbyte_repack(ret_ty, plat)
    )

    def codegen(context, builder, signature, arguments):
        if args_ty == NoneType:
            arg_vals = ()
        elif isinstance(args_ty, BaseTuple):
            _, args_pack = arguments
            arg_vals = tuple(
                builder.extract_value(args_pack, i)
                for i in range(len(arg_types))
            )
        else:
            _, scalar_arg = arguments
            arg_vals = (scalar_arg,)

        ret_ll_ty = context.get_value_type(ret_ty)

        ll_arg_tys = []
        ll_arg_vals = []
        byval_arg_indices = []

        if use_sret:
            sret_ptr = builder.alloca(ret_ll_ty)
            ll_arg_tys.append(ret_ll_ty.as_pointer())
            ll_arg_vals.append(sret_ptr)
        else:
            sret_ptr = None

        for arg_ty, arg_cls, val in zip(arg_types, arg_classes, arg_vals):
            arg_ll_ty = context.get_value_type(arg_ty)
            if arg_cls == _CLASS_SCALAR:
                ll_arg_tys.append(arg_ll_ty)
                ll_arg_vals.append(val)
                continue
            pass_by_value = arg_cls == _CLASS_STRUCT_SMALL and (
                plat in (_PLATFORM_SYSV_X86_64, _PLATFORM_AAPCS64)
                or (
                    plat == _PLATFORM_WIN_X64
                    and _is_windows_register_passable(
                        _struct_bytes(arg_ty, "_call_lib_func")
                    )
                )
            )
            if pass_by_value:
                if _needs_int_int_eightbyte_repack(arg_ty, plat):
                    val, arg_ll_ty = _repack_to_i64_pair(
                        builder, val, arg_ll_ty)
                ll_arg_tys.append(arg_ll_ty)
                ll_arg_vals.append(val)
                continue
            stack_p = builder.alloca(arg_ll_ty)
            builder.store(val, stack_p)
            ll_arg_tys.append(arg_ll_ty.as_pointer())
            ll_arg_vals.append(stack_p)
            if arg_cls == _CLASS_STRUCT_LARGE and plat == _PLATFORM_SYSV_X86_64:
                byval_arg_indices.append(len(ll_arg_vals) - 1)

        if use_sret:
            func_ll_ty = llir.FunctionType(llir.VoidType(), ll_arg_tys)
        elif needs_ret_repack:
            i64x2_ll_ty = llir.LiteralStructType(
                [llir.IntType(64), llir.IntType(64)])
            func_ll_ty = llir.FunctionType(i64x2_ll_ty, ll_arg_tys)
        else:
            func_ll_ty = llir.FunctionType(ret_ll_ty, ll_arg_tys)
        func_p = get_or_insert_function(builder.module, func_ll_ty, func_name)

        if use_sret:
            func_p.args[0].add_attribute("sret")
        for idx in byval_arg_indices:
            func_p.args[idx].add_attribute("byval")
        if byval_arg_indices:
            builder.function.attributes.add("optnone")
            builder.function.attributes.add("noinline")

        if use_sret:
            builder.call(func_p, ll_arg_vals)
            return builder.load(sret_ptr)
        result = builder.call(func_p, ll_arg_vals)
        if needs_ret_repack:
            return _repack_from_i64_pair(builder, result, ret_ll_ty)
        return result

    sig = ret_ty(func_name_ty, args_ty)
    return sig, codegen


def _emit_byval_call(builder, arg, arg_ll_ty, ret_type, func_name):
    """Emit IR to pass a struct by pointer: alloca, store, call via pointer."""
    stack_p = builder.alloca(arg_ll_ty)
    builder.store(arg, stack_p)
    func_ty_ll = llir.FunctionType(ret_type, [arg_ll_ty.as_pointer()])
    func_p = get_or_insert_function(builder.module, func_ty_ll, func_name)
    return builder.call(func_p, [stack_p])


def _needs_int_int_eightbyte_repack(ty, plat):
    """Return whether a 16-byte by-value struct (argument or return) needs a repack to ``{i64, i64}`` before the call.

    The answer depends on the host's ABI. SysV x86-64 and AAPCS64 are both affected: llvmlite's small-struct register
    lowering loses fields when both eightbytes are pure-INTEGER but the LLVM type is not already the canonical
    ``{i64, i64}`` shape. For example, ``{i32, i32, i64}``, the duckdb_interval layout, loses fields.

    Windows x64 is unaffected, because a 16-byte aggregate falls outside the register-passable set ``{1, 2, 4, 8}``.
    The call therefore passes it through an alloca and a pointer, and never takes the register-coercion path.
    """
    if plat not in (_PLATFORM_SYSV_X86_64, _PLATFORM_AAPCS64):
        return False
    if _struct_bytes(ty, "_call_lib_func") != 16:
        return False
    if _is_canonical_int64_pair_layout(ty):
        return False
    return _classify_eightbytes(ty) == (
        _EIGHTBYTE_CLASS_INTEGER, _EIGHTBYTE_CLASS_INTEGER,
    )


def _repack_to_i64_pair(builder, val, orig_ll_ty):
    """Repack a 16-byte struct value to ``{i64, i64}`` through a memory bitcast.

    Both eightbytes must already be pure-INTEGER; the caller checks this with ``_needs_int_int_eightbyte_repack``.
    The function returns ``(repacked_val, repacked_ll_ty)``, which the caller substitutes into the call site.

    The function allocates a well-aligned ``{i64, i64}`` slot (8-byte alignment) and bitcasts it to ``orig_ll_ty*``
    for the store. It then loads the same bytes back as ``{i64, i64}``. The store through the bitcast pointer is safe,
    because the slot is over-aligned relative to ``orig_ll_ty``. The subsequent ``{i64, i64}`` load is well-aligned,
    which avoids the undefined behavior that an ``alloca(orig_ll_ty)`` slot would expose. Such a slot is 4-byte
    aligned for ``[4 x i32]``, for example. The bitcast is a pure reinterpretation of the bytes, so no zext or shift
    is required.
    See: https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    """
    i64x2_ll_ty = llir.LiteralStructType(
        [llir.IntType(64), llir.IntType(64)])
    slot = builder.alloca(i64x2_ll_ty)
    slot_as_orig = builder.bitcast(slot, orig_ll_ty.as_pointer())
    builder.store(val, slot_as_orig)
    return builder.load(slot), i64x2_ll_ty


def _repack_from_i64_pair(builder, i64_pair_val, target_ll_ty):
    """Convert a ``{i64, i64}`` return value back into ``target_ll_ty``, as the inverse of ``_repack_to_i64_pair``.

    The return-side eightbyte repack uses this function. If the C function returns a 16-byte INT/INT struct that is
    not the canonical ``{i64, i64}``, llvmlite loses fields on the return, as it does on the argument side.
    ``_call_lib_func`` therefore declares the LLVM call to return ``{i64, i64}``, and this function unpacks the result.

    The function allocates a well-aligned ``{i64, i64}`` slot, stores the call result in it, bitcasts the slot to
    ``target_ll_ty*``, and loads from it.
    """
    i64x2_ll_ty = llir.LiteralStructType(
        [llir.IntType(64), llir.IntType(64)])
    slot = builder.alloca(i64x2_ll_ty)
    builder.store(i64_pair_val, slot)
    slot_as_target = builder.bitcast(slot, target_ll_ty.as_pointer())
    return builder.load(slot_as_target)


@intrinsic(prefer_literal=True)
def _call_lib_func_byval(typingctx, func_name_ty, arg_ty):
    """Pass ``arg`` to a C function by pointer on all platforms.

    Use this intrinsic when the C signature takes a pointer to a struct and the caller holds the struct as a value.
    The intrinsic allocates a stack slot, stores the value in it, and passes the address of the slot.
    """
    func_name = extract_literal_str("_call_lib_func_byval", func_name_ty, field="func_name")
    func_sig = signatures.get(func_name, None)
    if func_sig is None:
        raise TypingError(f"Undefined signature for {func_name}")

    def codegen(context, builder, signature, arguments):
        _, arg = arguments
        arg_ll_ty = context.get_value_type(arg_ty)
        ret_type = context.get_value_type(signature.return_type)
        return _emit_byval_call(
            builder, arg, arg_ll_ty, ret_type, func_name)

    sig = func_sig.return_type(func_name_ty, arg_ty)
    return sig, codegen
