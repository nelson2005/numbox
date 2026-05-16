"""Named-LLVM-global address store and indirect-call factory.

Implementation primitives for ``cres_cacheable`` (in
[`numbox.utils.highlevel`](highlevel.py)). The named-global pattern is
adapted from ``cre/utils.py`` lines 1402-1447 in the
``DannyWeitekamp/Cognitive-Rule-Engine`` repository on GitHub: custom
``@intrinsic`` functions that build and address LLVM globals by literal
symbol name via ``llvmlite.ir.GlobalVariable`` with ``common`` linkage. The
JIT linker resolves each global's symbol at link time per process, so
cached caller IR is ASLR-safe and survives across processes.

These are private utilities for ``cres_cacheable``; nothing outside this
module + ``highlevel.py`` should need them. See ``cres_cacheable``'s
docstring for the user-facing story.
"""
from functools import lru_cache

from llvmlite import ir as llir
from numba.core import cgutils
from numba.core.errors import TypingError
from numba.core.types import BaseTuple, Literal, NoneType, intp, void
from numba.extending import intrinsic


def _get_or_make_named_global(builder, ll_ty, name):
    """Find or create a named LLVM global with C-style ``common`` linkage.

    ``common`` linkage merges duplicate declarations across modules at link
    time and zero-initializes the slot — the same shape C uses for tentative
    definitions like ``int x;``. The JIT linker resolves the symbol per
    process, so the cached IR is ASLR-safe.

    The cross-module merge depends on numba's single shared MCJIT engine
    (see ``numba.core.codegen.JitEngine._load_defined_symbols`` and
    ``CodeLibrary._finalize_specific`` in ``numba/core/codegen.py``). If
    numba ever migrates to an LLJIT-with-per-module-JITDylib architecture,
    ``common``/weak-symbol merge semantics across modules become implementation-
    defined per the `LLVM JIT linkage discussion
    <https://groups.google.com/g/llvm-dev/c/qwglftF4bdQ/m/32eZrcpVBAAJ>`_,
    and this site needs revisiting.
    """
    mod = builder.module
    try:
        return mod.get_global(name)
    except KeyError:
        gv = llir.GlobalVariable(mod, ll_ty, name=name)
        gv.linkage = "common"
        gv.initializer = cgutils.get_null_value(gv.type.pointee)
        return gv


@intrinsic
def _store_addr_to_named_global(typingctx, name_ty, val_ty):
    """Store an ``intp`` into a named LLVM global addressed by literal name."""
    if not isinstance(name_ty, Literal):
        raise TypingError("_store_addr_to_named_global: name must be a literal string")
    name = name_ty.literal_value

    def codegen(context, builder, sig, args):
        _, val = args
        intp_ll = context.get_value_type(intp)
        gv = _get_or_make_named_global(builder, intp_ll, name)
        builder.store(val, gv)
    return void(name_ty, val_ty), codegen


@intrinsic
def _load_addr_from_named_global(typingctx, name_ty):
    """Load an ``intp`` from a named LLVM global addressed by literal name."""
    if not isinstance(name_ty, Literal):
        raise TypingError("_load_addr_from_named_global: name must be a literal string")
    name = name_ty.literal_value

    def codegen(context, builder, sig, args):
        intp_ll = context.get_value_type(intp)
        gv = _get_or_make_named_global(builder, intp_ll, name)
        return builder.load(gv)
    return intp(name_ty), codegen


@lru_cache(maxsize=None)
def _make_icall_for_sig(sig):
    """Build an ``@intrinsic`` that indirect-calls a function pointer matching ``sig``.

    Cached by ``sig`` so identical signatures share a single ``_Intrinsic``
    instance; numba memoizes specializations by intrinsic identity, and a
    fresh instance per ``cres_cacheable`` decoration would otherwise cause
    per-import recompilation of byte-identical wrappers.
    """
    ret_ty = sig.return_type
    arg_tys = tuple(sig.args)
    ret_is_void = ret_ty == void

    @intrinsic
    def _icall(typingctx, addr_ty, args_ty=NoneType):
        # args_ty default applies only when n_args==0 — the wrapper emits
        # _icall(_addr) with no second arg; the arg_tys==() branch handles it.
        if arg_tys == ():
            n = 0
            is_tuple = False
        elif isinstance(args_ty, BaseTuple):
            n = len(tuple(args_ty))
            is_tuple = True
            if n != len(arg_tys):
                raise TypingError(f"_icall: expected {len(arg_tys)} arguments, got {n}")
        else:
            n = 1
            is_tuple = False
            if len(arg_tys) != 1:
                raise TypingError(f"_icall: expected 1 argument, got {len(arg_tys)}")

        def codegen(context, builder, signature, arguments):
            addr = arguments[0]
            if n == 0:
                ll_call_args = []
            elif is_tuple:
                args_tuple = arguments[1]
                ll_call_args = [builder.extract_value(args_tuple, i) for i in range(n)]
            else:
                ll_call_args = [arguments[1]]
            ret_ll = llir.VoidType() if ret_is_void else context.get_value_type(ret_ty)
            arg_lls = [context.get_value_type(t) for t in arg_tys]
            func_ll_ty = llir.FunctionType(ret_ll, arg_lls)
            fn_ptr = builder.inttoptr(addr, func_ll_ty.as_pointer())
            result = builder.call(fn_ptr, ll_call_args)
            if not ret_is_void:
                return result

        out_sig = ret_ty(addr_ty, args_ty)
        return out_sig, codegen

    return _icall
