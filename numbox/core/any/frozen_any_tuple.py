"""Immutable heterogeneous snapshot container: a structref-wrapped ``UniTuple(AnyType, n)`` plus per-slot type codes,
built once from raw values and read many times.

Read discipline. ``fat.get_as(i, ty)`` is the checked default read: at build time every slot records a ``uint32``
content digest of its value's numba type (``zlib.crc32`` of ``str`` of the unliteral'd type) in the ``codes`` field,
and ``get_as`` compares the recorded code against the digest of the requested type, baked in as a compile-time
constant, before dereferencing; on a mismatch it raises ``TypeError("FrozenAnyTuple: slot type mismatch")``. The
check is one array load plus one integer compare, O(1) per call at any arity, and needs no hoisting. ``fat[i]`` is
the unchecked escape hatch: it returns the slot's ``Any`` in O(1), and ``fat[i].get_as(ty)`` with a wrong type
silently returns the reinterpreted bit pattern; it never raises. Slot indices are not bounds-checked on either path.

Exact-match rule. The recorded code is a digest of the exact stored type, so ask with the exact type that was
stored: ``'C'`` and ``'A'`` array layouts differ, aligned and unaligned records differ.

The guard is a discipline aid, not a safety boundary: crc32 carries intrinsic ``2**-32`` collision odds; ``str()``
of a structref type includes the class name and fields but not the defining module, so two same-named, same-shaped
structref classes from different modules share a code; and ``fat.codes`` is a writable numpy array, so a user write
can lie to the guard. A collision only downgrades a checked read to unchecked semantics.

Iteration and bulk access. A direct ``fat.anys`` field access returns an owned copy of the whole tuple, paying one
incref per slot, so it must not sit inside a loop: hoist ``t = fat.anys`` to a local exactly once, then ``for x in
t`` is a native runtime loop and ``t[i]`` is O(1). From Python, each ``.anys`` property access boxes all ``n``
elements to ``Any`` proxies, O(n) per access: bind it once. The default reads (``get_as``, ``fat[i]``) do not need
hoisting.

Build. ``make_frozen_any_tuple`` is the only constructor, callable from both sides. From Python it accepts any
sequence of raw values and crosses the boundary once, with O(n) boxing paid once per build; from jit call it with a
straight-line tuple display of raw values, e.g. ``make_frozen_any_tuple((x, 2.5, "s"))``. Tuples containing ``Any``
values are refused at typing time: slot type codes require raw values, and wrapping before freezing erases the
types. Arity is fixed at build and is part of the type; ``1 <= n < 1000`` (numba's tuple-arity ceiling), with
practical guidance ``n <= 512``: compile cost is roughly quadratic in ``n`` (x86-measured) and paid once ever per
reader per machine cache under ``cache=True``.

Frozen means the slot bindings and their recorded types, not deep immutability: referenced payloads (arrays,
structrefs) stay mutable, and ``Any.reset`` through an aliased ``fat[i]`` handle changes a payload without updating
its recorded code, after which a checked ``get_as`` with the original type passes the guard and reinterprets.
A reference cycle (a container reachable from one of its own slots) is uncollectable, as NRT has no cycle
detection; this is inherited from ``Any``, as is the lack of support for numpy record scalars (record arrays work).

The ``FrozenAnyTuple`` constructor is deleted on both sides; use ``make_frozen_any_tuple``.
"""
import operator
import zlib

import numpy

from numba import njit
from numba.core import cgutils, types
from numba.core.errors import NumbaError, TypingError
from numba.experimental import structref
from numba.experimental.structref import StructRefProxy, define_boxing, new
from numba.extending import intrinsic, overload, overload_method

from numbox.core.any.any_type import AnyType, make_any
from numbox.core.configurations import jit_options


def _slot_code(ty):
    """uint32 content digest of a numba type, process-stable and therefore safe under ``cache=True``."""
    return zlib.crc32(str(types.unliteral(ty)).encode())


@structref.register
class FrozenAnyTupleTypeClass(types.StructRef):
    """Single module-level class for all arities, parameterized per ``n`` via field-tuple instances."""
    pass


deleted_fat_ctor_error = "Use `make_frozen_any_tuple` instead"


class FrozenAnyTuple(StructRefProxy):
    """Python proxy; every method below runs the same jit surface Python-side callers see from ``@njit`` code."""

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError(deleted_fat_ctor_error)

    @njit(**jit_options)
    def get_as(self, i, ty):
        """Checked read of slot ``i`` as type ``ty``; raises ``TypeError`` unless ``ty`` matches the recorded type."""
        return self.get_as(i, ty)

    @njit(**jit_options)
    def __getitem__(self, i):
        """Unchecked escape hatch: the slot's ``Any``, whose ``get_as`` reinterprets bits on a mistyped ask."""
        return self[i]

    @njit(**jit_options)
    def __len__(self):
        return len(self)

    @property
    @njit(**jit_options)
    def anys(self):
        """The whole slot tuple; boxes all ``n`` elements to ``Any`` proxies per access, so bind it once."""
        return self.anys

    @property
    @njit(**jit_options)
    def codes(self):
        """The recorded per-slot uint32 type codes."""
        return self.codes


def _fat_deleted_ctor(*args):
    raise NumbaError(deleted_fat_ctor_error)


overload(FrozenAnyTuple)(_fat_deleted_ctor)
define_boxing(FrozenAnyTupleTypeClass, FrozenAnyTuple)


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
        ("codes", types.Array(types.uint32, 1, "C")),
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
    lines.append("    fat.anys = (%s,)" % ", ".join(f"a{i}" for i in range(n)))
    lines.append("    fat.codes = _codes_const.copy()")
    lines.append("    return fat")
    ns = {"make_any": make_any, "new": new, "fat_ty": fat_ty, "_codes_const": _codes_const}
    exec("\n".join(lines), ns)
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


@overload_method(FrozenAnyTupleTypeClass, "get_as", strict=False, jit_options=jit_options)
def ol_fat_get_as(fat, i, ty):
    inst = getattr(ty, "instance_type", None)
    if inst is None:
        return None
    expected = numpy.uint32(_slot_code(inst))

    def impl(fat, i, ty):
        if fat.codes[i] != expected:
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
    if isinstance(fat, FrozenAnyTupleTypeClass):
        def impl(fat, i):
            return _get_slot(fat, i)
        return impl
