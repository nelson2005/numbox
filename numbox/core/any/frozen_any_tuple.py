"""Immutable heterogeneous snapshot container: a structref-wrapped ``UniTuple(AnyType, n)`` plus per-slot type codes,
built once from raw values and read many times.

Read discipline. The index domain is ``0 <= i < len(fat)``; negative indices are not supported on any surface.
``fat.get_as(i, ty)`` is the checked default read: at build time every slot records a ``uint32`` content digest of
its value's numba type (``zlib.crc32`` of ``str`` of the unliteral'd type) in the ``codes`` field, and ``get_as``
first bounds-checks ``i`` against the arity, a compile-time constant of the type, raising ``IndexError`` out of
range, then compares the recorded code against the digest of the requested type, baked in as a compile-time
constant, before dereferencing; on a mismatch it raises ``TypeError("FrozenAnyTuple: slot type mismatch")``. The
checked read is O(1) per call at any arity, measured at a few nanoseconds per call, and needs no hoisting. In jit
code ``fat[i]`` is the escape hatch, bounds- and type-unchecked: it returns the slot's ``Any`` in O(1), an
out-of-range ``i`` dereferences out of bounds, and ``fat[i].get_as(ty)`` with a wrong type silently returns the
reinterpreted bit pattern; it never raises. Sub-nanosecond per-element reads belong to that bare
``fat[i].get_as(ty)``, not to the checked read. From Python the proxy guards ``fat[i]`` and ``fat.get_as(i, ty)``
alike, raising ``IndexError`` before delegating unless ``0 <= i < len(fat)``; the guarded ``__getitem__`` also
terminates sequence-protocol iteration (``for x in fat``, ``list(fat)``) correctly.

Exact-match rule. The recorded code is a digest of the exact stored type, so ask with the exact type that was
stored: ``'C'`` and ``'A'`` array layouts differ, aligned and unaligned records differ.

The recorded codes are not writable through any ordinary surface. The type defines its own attribute access rather
than taking numba's generic structref accessors: ``anys`` is readable, ``codes`` does not resolve in jit code at
all, and **no setattr is defined for either field**, so neither can be rebound. Reads inside the guard go through
``_get_code``, which loads one ``uint32`` and never materializes an array, and the Python ``codes`` property returns
a fresh readonly copy per access. So the routes that defeat a merely-readonly array are all closed: element and
slice stores, ``.flat`` setitem (whose numba typing ignores array mutability), whole-field rebinding, aliasing one
codes buffer into several containers, ``setflags(write=True)``, ``numpy.frombuffer`` over the meminfo, and a forged
``__array_interface__``. Building is the sole writer, via ``_init_fat``.

Two caveats remain and neither is fixable here, so the guard is a discipline aid rather than a security boundary.
Raw-pointer escapes reach any structref regardless of its attribute surface: ``array_data_p`` plus a raw store, or
``deref_payload`` naming a payload type with a writable member, will still rewrite the codes, exactly as they will
rewrite the internals of any other numba object. And the digest is not injective: crc32 carries intrinsic
``2**-32`` collision odds, and ``str()`` of a structref type includes the class name and fields but not the defining
module, so two same-named, same-shaped structref classes from different modules share a code. Either way a checked
read is downgraded to unchecked semantics, never to a wrong-slot read.

Iteration and bulk access. A direct ``fat.anys`` field access returns an owned copy of the whole tuple, paying one
incref per slot, so it must not sit inside a loop: hoist ``t = fat.anys`` to a local exactly once, then ``for x in
t`` is a native runtime loop and ``t[i]`` is O(1). From Python, each ``.anys`` property access boxes all ``n``
elements to ``Any`` proxies, O(n) per access: bind it once; a plain ``for x in fat`` iterates through the guarded
``__getitem__``, one boundary crossing per element. The default reads (``get_as``, ``fat[i]``) do not need
hoisting.

Build. ``make_frozen_any_tuple`` is the only constructor, callable from both sides. From Python it accepts any
sequence of raw values and crosses the boundary once, with O(n) boxing paid once per build; from jit call it with a
straight-line tuple display of raw values, e.g. ``make_frozen_any_tuple((x, 2.5, "s"))``. Tuples containing ``Any``
values are refused at typing time: slot type codes require raw values, and wrapping before freezing erases the
types. Arity is fixed at build and is part of the type; ``1 <= n < 1000`` (numba's tuple-arity ceiling), with
practical guidance ``n <= 512``: compile cost is roughly quadratic in ``n`` (x86-measured) and paid once ever per
reader per machine cache under ``cache=True``.

Frozen covers the slot bindings and their recorded types, not the payloads behind them. Referenced payloads (arrays,
structrefs) stay mutable, and ``Any.reset`` through an aliased ``fat[i]`` handle changes a payload without updating
its recorded code, after which a checked ``get_as`` with the original type passes the guard and reinterprets. That
one is inherent to holding a reference: the container has no way to observe a payload mutated through a handle it
handed out.
A reference cycle (a container reachable from one of its own slots) is uncollectable, as NRT has no cycle
detection; this is inherited from ``Any``, as is the lack of support for numpy record scalars (record arrays work).

The ``FrozenAnyTuple`` constructor is deleted on both sides; use ``make_frozen_any_tuple``.
"""
import operator
import zlib

import numpy

from numba import njit
from numba.core import cgutils, imputils, types
from numba.core.datamodel import default_manager, models
from numba.core.errors import NumbaError, TypingError
from numba.core.typing.templates import AttributeTemplate
from numba.experimental import structref
from numba.experimental.structref import StructRefProxy, define_boxing, new
from numba.extending import (
    infer_getattr, intrinsic, lower_getattr_generic, lower_setattr_generic, overload, overload_method,
)

from numbox.core.any.any_type import AnyType, make_any
from numbox.core.configurations import jit_options


def _slot_code(ty):
    """uint32 content digest of a numba type, process-stable and therefore safe under ``cache=True``."""
    return zlib.crc32(str(types.unliteral(ty)).encode())


class FrozenAnyTupleTypeClass(types.StructRef):
    """Single module-level class for all arities, parameterized per ``n`` via field-tuple instances."""
    pass


#: Registered by hand rather than through ``structref.register``, which would also install generic
#: attribute access. This type defines its own: ``anys`` is readable, ``codes`` is not exposed to jit
#: code at all, and no setattr is defined for either, which is what makes the container frozen.
default_manager.register(FrozenAnyTupleTypeClass, models.StructRefModel)


@infer_getattr
class _FrozenAnyTupleAttr(AttributeTemplate):
    """Reads for ``anys`` only. ``codes`` is deliberately unresolvable in jit code.

    Handing out the codes array in jit would hand out an alias of the guard's own evidence, and
    numba's readonly array typing does not cover every store reaching it: ``.flat`` setitem ignores
    array mutability, and a raw data pointer bypasses the type entirely. The guard reads a single
    code through ``_get_code`` instead, so no array crosses into jit code.
    """
    key = FrozenAnyTupleTypeClass

    def generic_resolve(self, typ, attr):
        if attr == "anys":
            return typ.field_dict["anys"]


@lower_getattr_generic(FrozenAnyTupleTypeClass)
def _fat_getattr_impl(context, builder, typ, val, attr):
    utils = structref._Utils(context, builder, typ)
    dataval = utils.get_data_struct(val)
    return imputils.impl_ret_borrowed(context, builder, typ.field_dict[attr], getattr(dataval, attr))


frozen_fat_setattr_error = "FrozenAnyTuple is frozen; build a new one with `make_frozen_any_tuple`"


@lower_setattr_generic(FrozenAnyTupleTypeClass)
def _fat_setattr_impl(context, builder, sig, args, attr):
    """Refuse every field write with a readable message.

    Without this the refusal still happens, since no setattr is otherwise defined, but it surfaces
    as numba's bare ``No definition for lowering <type>.<attr> = ...``.
    """
    raise NumbaError(f"{frozen_fat_setattr_error} (tried to set `{attr}`)")


deleted_fat_ctor_error = "Use `make_frozen_any_tuple` instead"


class FrozenAnyTuple(StructRefProxy):
    """Python proxy; the indexed reads bounds-guard in Python, then run the same jit surface ``@njit`` callers see."""

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError(deleted_fat_ctor_error)

    def get_as(self, i, ty):
        """Checked read of slot ``i`` as ``ty``: ``IndexError`` out of range, ``TypeError`` on a mismatched type."""
        if not 0 <= i < len(self):
            raise IndexError("FrozenAnyTuple index out of range")
        return _fat_get_as_jit(self, i, ty)

    def __getitem__(self, i):
        """The slot's ``Any``, type-unchecked; bounds-guarded here, which also terminates sequence iteration."""
        if not 0 <= i < len(self):
            raise IndexError("FrozenAnyTuple index out of range")
        return _fat_getitem_jit(self, i)

    @njit(**jit_options)
    def __len__(self):
        return len(self)

    @property
    @njit(**jit_options)
    def anys(self):
        """The whole slot tuple; boxes all ``n`` elements to ``Any`` proxies per access, so bind it once."""
        return self.anys

    @property
    def codes(self):
        """A fresh readonly copy of the recorded per-slot uint32 type codes, rebuilt per access.

        A copy rather than the stored array: numpy's ``WRITEABLE`` flag describes one ndarray
        wrapper, not the memory behind it, so handing out a view of the guard's own evidence would
        leave it reachable through ``setflags``, through ``numpy.frombuffer`` over the meminfo, and
        through a forged ``__array_interface__``. Writes to what this returns go to the copy.
        """
        out = _fat_codes_copy_jit(self)
        out.setflags(write=False)
        return out


def _fat_deleted_ctor(*args):
    raise NumbaError(deleted_fat_ctor_error)


overload(FrozenAnyTuple)(_fat_deleted_ctor)
define_boxing(FrozenAnyTupleTypeClass, FrozenAnyTuple)


@njit(**jit_options)
def _fat_get_as_jit(fat, i, ty):
    return fat.get_as(i, ty)


@njit(**jit_options)
def _fat_getitem_jit(fat, i):
    return fat[i]


@njit(**jit_options)
def _fat_codes_copy_jit(fat):
    n = len(fat)
    out = numpy.empty(n, dtype=numpy.uint32)
    for i in range(n):
        out[i] = _get_code(fat, i)
    return out


_fat_type_cache = {}


def frozen_any_tuple_type(n):
    """Memoized ``FrozenAnyTuple`` type instance of arity ``n``, for signatures, containers and overload guards.

    Arity is part of the type. Requires ``1 <= n < 1000``: numba refuses tuples of 1000 or more elements, and the
    build path must enforce the ceiling itself because numba's own guard covers ``UniTuple`` arguments only.
    """
    if not (1 <= n < 1000):
        raise ValueError(f"FrozenAnyTuple arity must satisfy 1 <= n < 1000 (numba's tuple-arity ceiling), got {n}")
    if n in _fat_type_cache:
        return _fat_type_cache[n]
    inst = FrozenAnyTupleTypeClass([
        ("anys", types.UniTuple(AnyType, n)),
        ("codes", types.Array(types.uint32, 1, "C", readonly=True)),
    ])
    _fat_type_cache[n] = inst
    return inst


def make_frozen_any_tuple(values):
    """Build a ``FrozenAnyTuple`` from a sequence of raw (not ``Any``-wrapped) values.

    The one constructor, same name on both sides. From Python, ``values`` may be any sequence (list, tuple,
    generator); it is coerced to a tuple and crosses the boundary once. From jit, pass a straight-line tuple
    display of raw values.
    """
    return _make_fat_jit(tuple(values))


@njit(**jit_options)
def _make_fat_jit(values):
    return make_frozen_any_tuple(values)


@overload(make_frozen_any_tuple, strict=False, jit_options={**jit_options, "cache": False})
def ol_make_frozen_any_tuple(values):
    if not isinstance(values, types.BaseTuple):
        return None
    elem_types = tuple(types.unliteral(t) for t in values)
    n = len(elem_types)
    if n == 0:
        raise TypingError("make_frozen_any_tuple: cannot build from an empty tuple")
    if any(t == AnyType for t in elem_types):
        raise TypingError(
            "make_frozen_any_tuple: slot type codes require raw values; wrap-then-freeze erases the types"
        )
    fat_ty = frozen_any_tuple_type(n)
    _codes_const = numpy.array([_slot_code(t) for t in elem_types], dtype=numpy.uint32)
    _codes_const.setflags(write=False)
    lines = ["def _build(values):"]
    for i in range(n):
        lines.append(f"    a{i} = make_any(values[{i}])")
    lines.append("    fat = new(fat_ty)")
    lines.append("    _init_fat(fat, (%s,), _codes_const.copy())" % ", ".join(f"a{i}" for i in range(n)))
    lines.append("    return fat")
    ns = {
        "make_any": make_any, "new": new, "fat_ty": fat_ty,
        "_codes_const": _codes_const, "_init_fat": _init_fat,
    }
    exec("\n".join(lines), ns)  # nosec B102 - JIT codegen of internal source
    return ns["_build"]


@intrinsic
def _get_slot(typingctx, fat_ty, i_ty):
    """O(1) single-slot read: GEP into the stored tuple, load, and return the element with exactly one incref."""
    if not isinstance(fat_ty, FrozenAnyTupleTypeClass):
        return None
    elem_ty = fat_ty.field_dict["anys"].dtype
    sig = elem_ty(fat_ty, types.intp)

    def codegen(context, builder, signature, args):
        fat_v, i_v = args
        utils = structref._Utils(context, builder, signature.args[0])
        data = utils.get_data_struct(fat_v)
        tup_ptr = data._get_ptr_by_name("anys")
        elem_ptr = builder.gep(tup_ptr, [cgutils.int32_t(0), i_v])
        elem = builder.load(elem_ptr)
        context.nrt.incref(builder, elem_ty, elem)
        return elem

    return sig, codegen


@intrinsic
def _init_fat(typingctx, fat_ty, anys_ty, codes_ty):
    """Populate a freshly allocated container. The only writer of either field.

    No setattr is defined for this type, so the build path cannot use ordinary field assignment and
    a caller cannot rebind a field afterwards. Both fields are increfed here, matching what numba's
    generic setattr would have done on a fresh struct whose slots start null.
    """
    if not isinstance(fat_ty, FrozenAnyTupleTypeClass):
        return None
    sig = types.void(fat_ty, anys_ty, codes_ty)

    def codegen(context, builder, signature, args):
        fat_v, anys_v, codes_v = args
        _, anys_t, codes_t = signature.args
        utils = structref._Utils(context, builder, signature.args[0])
        dataval = utils.get_data_struct(fat_v)
        context.nrt.incref(builder, anys_t, anys_v)
        context.nrt.incref(builder, codes_t, codes_v)
        setattr(dataval, "anys", anys_v)
        setattr(dataval, "codes", context.cast(builder, codes_v, codes_t, signature.args[0].field_dict["codes"]))

    return sig, codegen


@intrinsic
def _get_code(typingctx, fat_ty, i_ty):
    """O(1) scalar read of slot ``i``'s recorded code, without exposing the array to jit code."""
    if not isinstance(fat_ty, FrozenAnyTupleTypeClass):
        return None
    sig = types.uint32(fat_ty, types.intp)

    def codegen(context, builder, signature, args):
        fat_v, i_v = args
        utils = structref._Utils(context, builder, signature.args[0])
        dataval = utils.get_data_struct(fat_v)
        codes_ty = signature.args[0].field_dict["codes"]
        ary = context.make_array(codes_ty)(context, builder, value=getattr(dataval, "codes"))
        ptr = builder.gep(ary.data, [i_v])
        return builder.load(ptr)

    return sig, codegen


@overload_method(FrozenAnyTupleTypeClass, "get_as", strict=False, jit_options=jit_options)
def ol_fat_get_as(fat, i, ty):
    """Jit checked read: ``IndexError`` unless ``0 <= i < n``, then the recorded-code compare, then the dereference."""
    inst = getattr(ty, "instance_type", None)
    if inst is None:
        return None
    expected = numpy.uint32(_slot_code(inst))
    n = fat.field_dict["anys"].count

    def impl(fat, i, ty):
        if i < 0 or i >= n:
            raise IndexError("FrozenAnyTuple index out of range")
        if _get_code(fat, i) != expected:
            raise TypeError("FrozenAnyTuple: slot type mismatch")
        return _get_slot(fat, i).get_as(ty)
    return impl


@overload(len, jit_options=jit_options)
def ol_fat_len(fat):
    if isinstance(fat, FrozenAnyTupleTypeClass):
        n = fat.field_dict["anys"].count

        def impl(fat):
            return n
        return impl


@overload(operator.getitem, jit_options=jit_options)
def ol_fat_getitem(fat, i):
    """Jit ``fat[i]``: the raw escape hatch, bounds- and type-unchecked; the Python proxy surface is guarded."""
    if isinstance(fat, FrozenAnyTupleTypeClass):
        def impl(fat, i):
            return _get_slot(fat, i)
        return impl
