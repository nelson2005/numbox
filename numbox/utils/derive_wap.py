"""First-class function values whose exceptions propagate out of the call.

The mechanism
-------------

numba lowers a first-class ``FunctionType`` call in one of two ways, and chooses between them on the function model's
``jit_addr`` slot. A populated slot selects the numba calling convention together with ``return_status_propagate``,
so an exception unwinds into the caller. An empty slot selects the C wrapper, which numba documents as not supporting
exceptions: the wrapper discards the exception, zero-fills the return value and reports the failure only as an
unraisable exception on stderr.

numba fills the slot for a ``Dispatcher`` and leaves it empty for everything else, for which ``_get_jit_address``
returns 0. A call site that receives a bare compile result therefore has no entry point that could carry an exception
out.

:class:`DeriveWAP` captures the compile result's numba-callconv entry point, and :class:`DeriveFunctionType`
populates ``jit_addr`` from it on both unboxing and constant lowering. A value of this type therefore carries a usable
entry point wherever it is used.

Usage
-----

:func:`numbox.utils.highlevel.cres` builds these values, so a function compiled by `cres` is already one::

    @cres(float64(float64))
    def f(x):
        if x < 0:
            raise ValueError("negative")
        return x

Filling the slot is necessary but not sufficient, because the *call site* decides which convention it emits. A call
site that reads the wrapper address unconditionally calls the C wrapper, whatever the slot holds. A consumer opts in
by reading ``jit_addr`` and, when it is non-null, emitting ``call_conv.call_function`` together with
``return_status_propagate``. numba's own implementation of that logic is
``numba.core.lowering.Lower.__call_first_class_function_pointer``.

``numbox.core.work.work._call_derive`` is the worked example in this repository, and :doc:`numbox.core.work`
describes it. It selects the propagating convention at compile time for a :class:`DeriveFunctionType`. It tests the
slot at runtime for a plain ``FunctionType``, because numba does populate the slot for an njit dispatcher passed as a
``FunctionType``-typed argument.

Limits
------

:meth:`DeriveFunctionType.can_convert_to` documents the one conversion that does not keep the entry point: a value
passed from Python into a parameter *declared* as a plain ``FunctionType`` degrades to the C convention at that
boundary.

Reliance on numba
-----------------

The decorators below belong to numba's public extension API, except ``lower_constant``, which ``numba.extending``
does not re-export. The objects they register against are outside that API: ``FunctionModel``,
``CompileResultWAP``, ``Conversion``, ``box_function_type`` and ``lower_get_wrapper_address`` are all outside
``numba.extending``, and the constant lowering uses ``context.declare_function`` and ``context.active_code_library``
directly. No numba internals are patched, however: this module replaces no numba behavior, it only registers against
it.
"""
from numba.core import cgutils, types
from numba.core.imputils import lower_cast, lower_constant
from numba.core.typeconv import Conversion
from numba.core.types.function_type import CompileResultWAP, FunctionType
from numba.core.typing.typeof import typeof_impl
from numba.experimental.function_type import (
    FunctionModel, box_function_type, lower_get_wrapper_address,
)
from numba.extending import NativeValue, box, register_model, unbox


__all__ = ["DeriveFunctionType", "DeriveWAP", "rewrap_derive"]


#: The attribute under which :func:`rewrap_derive` stores the upgraded wrapper on the object it upgrades, so that the
#: address stored in ``py_addr`` remains valid for as long as the caller holds the original.
_UPGRADED_ATTR = "_numbox_derive_wap"


class DeriveFunctionType(FunctionType):
    """First-class function type whose values always carry a populated ``jit_addr``.

    The type is kept distinct from ``FunctionType`` so that ``_call_derive`` can select the propagating calling
    convention at compile time, with no runtime branch, for every derive that numbox itself compiled.
    """

    def can_convert_to(self, typingctx, other):
        """Permit passing a :class:`DeriveWAP` where a plain ``FunctionType`` of the same signature is declared.

        Such a call site keeps working unchanged, which is the purpose of the conversion. It also keeps its *old*
        behavior: a value passed from Python into a parameter declared as a plain ``FunctionType`` degrades to the C
        convention, so a `derive` supplied that way still discards its exception. A cast to the same declared type
        within jitted scope does not degrade the value, because :func:`lower_cast_derive_to_function_type` is an
        identity on a shared ``FunctionModel`` and the populated ``jit_addr`` is preserved. One source-level call
        site therefore has two different exception semantics, depending on the origin of the value.
        """
        if type(other) is FunctionType and other.signature == self.signature:
            return Conversion.safe
        return None


@lower_cast(DeriveFunctionType, FunctionType)
def lower_cast_derive_to_function_type(context, builder, fromty, toty, val):
    """Identity cast. Both types use ``FunctionModel``, so the struct is shared."""
    return val


class DeriveWAP(CompileResultWAP):
    """``CompileResultWAP`` that also captures the numba-callconv entry point.

    ``CompileResultWAP`` records only the cfunc wrapper address, which is the
    entry point that cannot carry an exception out.
    """

    def __init__(self, cres):
        super().__init__(cres)
        self.jit_address = cres.library.get_pointer_to_function(
            cres.fndesc.llvm_func_name)
        if self.jit_address <= 0:
            raise ValueError(
                f"no callconv entry point for {cres.fndesc.llvm_func_name}")


@typeof_impl.register(DeriveWAP)
def typeof_derive_wap(val, c):
    return DeriveFunctionType(val.signature())


register_model(DeriveFunctionType)(FunctionModel)


def _get_derive_jit_address(func, sig):
    """Resolve the callconv entry address during unboxing.

    Jitted unboxing code calls this function with the GIL held, as numba's own first-class unboxing calls back into
    Python. The function raises for a value that is not a :class:`DeriveWAP`, so the error is reported at the unbox
    boundary and does not corrupt the struct. A null ``jit_addr`` instead of the raise would make the call discard
    exceptions again, and nothing would report it.
    """
    if isinstance(func, DeriveWAP):
        return func.jit_address
    raise TypeError(
        f"DeriveFunctionType value must be a DeriveWAP, got {type(func)}")


def _lower_get_derive_jit_address(context, builder, func, sig):
    """Emit the call to :func:`_get_derive_jit_address`.

    The emitted code follows numba's ``_lower_get_address`` with ``failure_mode='return_null'``: a null result returns
    NULL from the unboxing function, which propagates the Python exception instead of discarding it.
    """
    pyapi = context.get_python_api(builder)
    modname = context.insert_const_string(builder.module, __name__)
    mod = pyapi.import_module(modname)
    fn = pyapi.object_getattr_string(mod, "_get_derive_jit_address")
    pyapi.decref(mod)
    sig_obj = pyapi.unserialize(pyapi.serialize_object(sig))
    addr = pyapi.call_function_objargs(fn, (func, sig_obj))
    # `fn` and `sig_obj` are new references and `addr` does not borrow from either,
    # so release them before the branch below: the null path returns early, and
    # unboxing runs on every call for a derive passed as an argument.
    pyapi.decref(fn)
    pyapi.decref(sig_obj)
    with builder.if_then(cgutils.is_null(builder, addr), likely=False):
        builder.ret(pyapi.get_null_object())
    return addr


@unbox(DeriveFunctionType)
def unbox_derive_function_type(typ, obj, c):
    typ = typ.get_precise()
    sfunc = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    addr = lower_get_wrapper_address(
        c.context, c.builder, obj, typ.signature, failure_mode="return_null")
    sfunc.c_addr = c.pyapi.long_as_voidptr(addr)
    c.pyapi.decref(addr)

    llty = c.context.get_value_type(types.voidptr)
    sfunc.py_addr = c.builder.ptrtoint(obj, llty)

    addr = _lower_get_derive_jit_address(c.context, c.builder, obj, typ.signature)
    sfunc.jit_addr = c.pyapi.long_as_voidptr(addr)
    c.pyapi.decref(addr)

    return NativeValue(sfunc._getvalue())


@box(DeriveFunctionType)
def box_derive_function_type(typ, val, c):
    return box_function_type(typ, val, c)


@lower_constant(DeriveFunctionType)
def lower_constant_derive_function_type(context, builder, typ, pyval):
    """Lower a derive reached as a compile-time constant from jitted code.

    ``c_addr`` and ``py_addr`` follow numba's own lowering for a ``WrapperAddressProtocol`` value. ``jit_addr`` is
    declared symbolically, and the compile result's library is linked in. numba uses the same pattern for its
    ``Dispatcher -> FunctionType`` cast. The pattern makes the entry point resolve as a symbol rather than as a bare
    pointer.

    Resolving the entry point as a symbol is also what makes a constant-lowered derive cacheable. The dead ``c_addr``
    and ``py_addr`` globals are eliminated before numba scans the final module, so a caller that only calls the
    derive reports no dynamic globals and caches. A literal address would keep those globals live, and numba would
    then refuse to cache the caller rather than store an address that is randomized per process.

    That caching has a cost on this constant-reference route alone. The caller's cached binary binds the derive's
    code, which the ``add_linking_library`` call below links in. An edit to the derive's body in another module
    therefore does not reliably take effect in such a caller, and which body the caller runs is not single-valued.

    The inlining on this route is LLVM's, over that linked-in body. The inlining is not the numba-IR ``inline='always'``
    that ``@proxy`` puts on its generated wrapper. That option applies only where a jitted caller calls the dispatcher,
    and a constant reference never reaches the wrapper. A body small enough for LLVM to inline leaves no separate
    definition behind, and the caller continues to run the old body. A larger body is embedded as a weak
    ``linkonce_odr`` definition under a mangled name. That name carries numba's per-process ``v<uid>`` abi-tag, which
    numba's cache key does not cover. The caller runs the embedded copy when the name the body carries in the loading
    process differs from the embedded name, and the body's current definition when the two match. The name the body
    carries is assigned by whichever process last recompiled the body, and it is stored in the body's own cache entry.
    The first process to recompile the body after an edit therefore determines the outcome for every later process.

    A caller that reaches the same binding through its dispatcher is unaffected: it calls through the alias, and the
    stale-alias guard repairs it on load. Clearing the cache is the only remedy for a caller that holds the derive as
    a compile-time constant, and nothing issues a warning.

    There is deliberately no fallback to a literal address. A value of this type always takes the propagating call,
    so a `jit_addr` that failed to resolve would be called unconditionally. Failing the compilation is therefore the
    only correct outcome.
    """
    typ = typ.get_precise()
    assert typ.check_signature(pyval.signature())
    sfunc = cgutils.create_struct_proxy(typ)(context, builder)
    sfunc.c_addr = context.add_dynamic_addr(
        builder, pyval.__wrapper_address__(), info=str(typ))
    sfunc.py_addr = context.add_dynamic_addr(
        builder, id(pyval), info=type(pyval).__name__)
    fn = context.declare_function(builder.module, pyval.cres.fndesc)
    sfunc.jit_addr = builder.bitcast(fn, context.get_value_type(types.voidptr))
    context.active_code_library.add_linking_library(pyval.cres.library)
    return sfunc._getvalue()


def rewrap_derive(derive):
    """Upgrade a foreign ``CompileResultWAP`` so its exceptions propagate.

    A derive compiled by numbox's own :func:`~numbox.utils.highlevel.cres` is already a :class:`DeriveWAP`. A derive
    built directly against numba is not, and would keep the convention that discards exceptions. The compile result it
    carries is enough to upgrade it.

    Anything else, including ``None``, is returned unchanged, so that callers can apply this function unconditionally.

    The upgraded wrapper is memoized as an attribute of the object it upgrades, and the memoization is required, not an
    optimization. ``py_addr`` holds the derive's address without taking a reference. A wrapper created anew for each
    call would therefore be freed as soon as the caller returned, and every `Work` built from it would point at
    released memory. Storing the wrapper on the original ties its lifetime to the object the caller already holds,
    which is the lifetime the address has assumed from the start.
    """
    if not isinstance(derive, CompileResultWAP) or isinstance(derive, DeriveWAP):
        return derive
    upgraded = getattr(derive, _UPGRADED_ATTR, None)
    if upgraded is None:
        upgraded = DeriveWAP(derive.cres)
        setattr(derive, _UPGRADED_ATTR, upgraded)
    return upgraded
