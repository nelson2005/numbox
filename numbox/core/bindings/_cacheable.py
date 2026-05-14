"""Cache-friendly indirection for @cres-style bindings.

The default @cres pattern produces a numba FunctionType value
(``CompileResultWAP``). When a user ``@njit(cache=True)`` function references
one as a Python global, numba's FunctionType lowering goes through
``add_dynamic_addr`` to materialize the value's runtime address, which sets
``has_dynamic_globals`` on the compiled IR and disables caching. The caller
emits "Cannot cache compiled function ... uses dynamic globals."

``cres_cacheable`` produces, for a single user definition, two parallel
artifacts:

- An ``@cres`` ``CompileResultWAP`` (accessible as ``wrapped.as_funcvalue``),
  preserving the FunctionType-as-value capability.
- A second ``@njit(sig, cache=True)`` dispatcher (the value the decorator
  returns) that loads the underlying function pointer from a *named LLVM
  global* and indirect-calls through it. Cached caller IR references the
  symbol name, not a runtime address — so callers cache cleanly.

The named-global pattern is adapted from `cre/utils.py` lines 1402-1447 in
the `DannyWeitekamp/Cognitive-Rule-Engine` repository on GitHub.

Cross-process safety: each process's import runs a one-time setup
``@njit(void(intp), cache=True)`` that writes the live ``CompileResultWAP``
address into the named global. The cached caller IR contains a reference to
the symbol; the JIT linker resolves it per process; ASLR-correct.
"""
from inspect import getfile

from llvmlite import ir as llir
from numba import njit
from numba.core import cgutils
from numba.core.types import BaseTuple, Literal, NoneType, intp, void
from numba.extending import intrinsic

from numbox.utils.highlevel import cres


__all__ = ["cres_cacheable"]


def _file_anchor():
    """Anchor for ``inspect.getfile`` so exec-generated wrapper code carries a
    real file path. Numba's cache locator requires ``co_filename`` to resolve
    to a file on disk; using ``getfile(_file_anchor)`` here gives the
    generated wrappers this module's path."""


def _get_or_make_named_global(builder, ll_ty, name):
    """Find or create a named LLVM global with C-style common linkage.

    'common' linkage merges duplicate declarations across modules at link
    time and zero-initializes the slot — the same shape C uses for tentative
    definitions like ``int x;``. The JIT linker resolves the symbol per
    process, so the cached IR is ASLR-safe.
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
        return
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
        return
    name = name_ty.literal_value

    def codegen(context, builder, sig, args):
        intp_ll = context.get_value_type(intp)
        gv = _get_or_make_named_global(builder, intp_ll, name)
        return builder.load(gv)
    return intp(name_ty), codegen


def _make_icall_for_sig(sig):
    """Build an ``@intrinsic`` that indirect-calls a function pointer matching ``sig``."""
    ret_ty = sig.return_type
    arg_tys = tuple(sig.args)
    ret_is_void = ret_ty == void

    @intrinsic
    def _icall(typingctx, addr_ty, args_ty=NoneType):
        if arg_tys == ():
            n = 0
            is_tuple = False
        elif isinstance(args_ty, BaseTuple):
            n = len(tuple(args_ty))
            is_tuple = True
            if n != len(arg_tys):
                return
        else:
            n = 1
            is_tuple = False
            if len(arg_tys) != 1:
                return

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


def _global_name_for(func):
    """Derive a stable cross-process symbol name for the function's address."""
    return (
        f"_numbox_cacheable_addr__{func.__module__}__{func.__name__}"
    ).replace(".", "_")


def cres_cacheable(sig, **njit_kwargs):
    """Decorator that returns a cache-friendly @njit dispatcher wrapping a binding.

    Behaves like ``@cres(sig, cache=True)`` except the returned value is a
    ``CPUDispatcher``, not a ``CompileResultWAP``. The original
    ``CompileResultWAP`` is attached as ``returned.as_funcvalue`` so callers
    that need a FunctionType (e.g. for storing in a struct field) can still
    get one.
    """
    njit_kwargs.setdefault("cache", True)
    arg_tys = tuple(sig.args)
    n_args = len(arg_tys)
    ret_is_void = sig.return_type == void

    def _decorator(func):
        funcvalue = cres(sig, **njit_kwargs)(func)
        global_name = _global_name_for(func)
        _icall = _make_icall_for_sig(sig)

        params = ", ".join(f"a{i}" for i in range(n_args))
        if n_args == 0:
            call_args = "_addr"
        elif n_args == 1:
            call_args = "_addr, a0"
        else:
            call_args = "_addr, (" + ", ".join(f"a{i}" for i in range(n_args)) + ")"
        body = (
            f"def _wrapper({params}):\n"
            f"    _addr = _load_addr({global_name!r})\n"
        )
        body += "    " + ("" if ret_is_void else "return ") + f"_icall({call_args})\n"

        ns = {
            "_load_addr": _load_addr_from_named_global,
            "_icall": _icall,
        }
        exec(compile(body, getfile(_file_anchor), mode="exec"), ns)
        wrapper = ns["_wrapper"]
        wrapper.__name__ = func.__name__
        wrapper.__qualname__ = getattr(func, "__qualname__", func.__name__)
        wrapper.__module__ = func.__module__
        wrapper.__doc__ = func.__doc__

        compiled = njit(sig, **njit_kwargs)(wrapper)

        # One-time setup: write the live cres address into the named global.
        # The setup is itself @njit(cache=True) — cacheable because the addr
        # comes in as an argument (no Python global is baked into its IR) and
        # the global name is exec-inlined as a literal string.
        setup_body = (
            f"def _setup(addr):\n"
            f"    _store_addr({global_name!r}, addr)\n"
        )
        setup_ns = {"_store_addr": _store_addr_to_named_global}
        exec(compile(setup_body, getfile(_file_anchor), mode="exec"), setup_ns)
        _setup_py = setup_ns["_setup"]
        _setup_py.__name__ = f"_setup_for_{func.__name__}"
        _setup_py.__qualname__ = _setup_py.__name__
        _setup_py.__module__ = __name__
        _setup = njit(void(intp), **njit_kwargs)(_setup_py)
        _setup(funcvalue.address)

        compiled.as_funcvalue = funcvalue
        return compiled

    return _decorator
