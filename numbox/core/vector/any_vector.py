"""Growable heterogeneous vector of ``Any`` values.

``AnyVector`` stores NRT MemInfo pointers to :class:`numbox.core.any.any_type.Any`
structrefs in a contiguous ``numpy`` ``intp`` buffer with amortised x2 geometric
growth. Anything ``make_any`` accepts (scalars, strings, arrays, records,
structrefs, first-class functions) can therefore live in one vector, with the
stored type recoverable per element via ``Any``'s type tag.

Ownership contract (each rule is pinned by a test in
``test/core/test_any_vector.py``):

- every slot in ``buf[0:size)`` holds a non-zero MemInfo pointer owning one
  reference to its ``Any`` element;
- reads (``v[i]``, ``v.get_as(i, ty)``) borrow: the returned ``Any`` is live and
  its reference is balanced when the caller's handle dies;
- ``pop`` transfers ownership of the last element to the caller and zeroes the
  vacated slot;
- overwriting a slot (``v[i] = x``) secures the new element before releasing the
  old one, so aliased self-assignment is safe;
- the vector's NRT destructor walks ``buf[0:size)`` when the vector itself dies
  and releases every remaining element (zeroed slots are skipped), so dropping
  the last reference to the vector cannot leak elements; ``any_vector_clear``
  does the same eagerly and composes with the destructor.

Both a bare and a sugared API are provided: module-level functions
(``any_vector_push`` etc., all callable from Python and jit) and methods on the
vector (``v.push(x)``, ``v.pop()``, ``v.clear()``, ``v.get_as(i, ty)``).
``any_vector_push`` and ``v[i] = x`` wrap a non-``Any`` argument via
``make_any`` automatically.

The destructor references the NRT API by symbol name only, never by address, so
everything here compiles under the module-default ``cache=True``.
"""
import numpy
import operator

from llvmlite import ir as llir
from numba import njit, types as nb_types
from numba.core import cgutils
from numba.core.errors import NumbaError
from numba.experimental import structref
from numba.experimental.structref import StructRefProxy, define_boxing
from numba.extending import intrinsic, overload, overload_method

from numbox.core.any.any_type import AnyType, AnyTypeClass, make_any
from numbox.core.configurations import jit_options
from numbox.utils.meminfo import (
    _incref_meminfo, borrow_structref, export_meminfo, release_meminfo,
)


@structref.register
class AnyVectorTypeClass(nb_types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)


deleted_any_vector_ctor_error = "Use `create_any_vector` instead"


class AnyVector(StructRefProxy):
    def __new__(cls, *args, **kwargs):
        raise NotImplementedError(deleted_any_vector_ctor_error)

    @property
    @njit(**jit_options)
    def size(self):
        return self.size

    @property
    @njit(**jit_options)
    def buf(self):
        return self.buf

    @njit(**jit_options)
    def get_as(self, i, ty):
        return self.get_as(i, ty)

    @njit(**jit_options)
    def push(self, x):
        return self.push(x)

    @njit(**jit_options)
    def pop(self):
        return self.pop()

    @njit(**jit_options)
    def clear(self):
        return self.clear()

    @njit(**jit_options)
    def __getitem__(self, i):
        return self[i]

    @njit(**jit_options)
    def __setitem__(self, i, x):
        self[i] = x

    @njit(**jit_options)
    def __len__(self):
        return len(self)


def _any_vector_deleted_ctor(*args):
    raise NumbaError(deleted_any_vector_ctor_error)


overload(AnyVector)(_any_vector_deleted_ctor)
define_boxing(AnyVectorTypeClass, AnyVector)

AnyVectorType = AnyVectorTypeClass([
    ("buf", nb_types.Array(nb_types.intp, 1, 'C')),
    ("size", nb_types.int64),
])

_voidptr_t = llir.IntType(8).as_pointer()


def _any_vector_dtor(context, module, inst_ty):
    """Destructor for the vector structref: release every element still owned
    by ``buf[0:size)`` (zeroed slots are skipped), then the payload's own
    fields. NRT is reached by symbol name only, keeping the IR cacheable."""
    llsize = context.get_value_type(nb_types.uintp)
    dtor_ftype = llir.FunctionType(llir.VoidType(), [_voidptr_t, llsize, _voidptr_t])
    fn = cgutils.get_or_insert_function(module, dtor_ftype, f"_numbox_any_vector_dtor.{inst_ty.name}")
    if fn.is_declaration:
        builder = llir.IRBuilder(fn.append_basic_block())
        payload_fe = inst_ty.get_data_type()
        payload_ll = context.get_value_type(payload_fe)
        payload_p = builder.bitcast(fn.args[0], payload_ll.as_pointer())
        payload = context.make_helper(builder, payload_fe, ref=payload_p)
        buf_fe = inst_ty.field_dict["buf"]
        buf_val = payload.buf
        arr = context.make_helper(builder, buf_fe, value=buf_val)
        rel_ftype = llir.FunctionType(llir.VoidType(), [_voidptr_t])
        rel_fn = cgutils.get_or_insert_function(module, rel_ftype, "NRT_MemInfo_release")
        with cgutils.for_range(builder, payload.size) as loop:
            elem = builder.load(builder.gep(arr.data, [loop.index]))
            nonzero = builder.icmp_unsigned("!=", elem, elem.type(0))
            with builder.if_then(nonzero):
                builder.call(rel_fn, [builder.inttoptr(elem, _voidptr_t)])
        context.nrt.decref(builder, buf_fe, buf_val)
        builder.ret_void()
    return fn


@intrinsic
def _new_any_vector(typingctx, vec_ty_ref):
    inst_ty = vec_ty_ref.instance_type
    sig = inst_ty(vec_ty_ref)

    def codegen(context, builder, signature, args):
        model = context.data_model_manager[inst_ty.get_data_type()]
        alloc_type = model.get_value_type()
        alloc_size = context.get_abi_sizeof(alloc_type)
        dtor = _any_vector_dtor(context, builder.module, inst_ty)
        meminfo = context.nrt.meminfo_alloc_dtor(
            builder, context.get_constant(nb_types.uintp, alloc_size), dtor)
        data_pointer = context.nrt.meminfo_data(builder, meminfo)
        data_pointer = builder.bitcast(data_pointer, alloc_type.as_pointer())
        # Field stores decref the previous slot value and the destructor may run
        # on a partially initialised payload, so the payload must start zeroed.
        builder.store(cgutils.get_null_value(alloc_type), data_pointer)
        st = context.make_helper(builder, inst_ty)
        st.meminfo = meminfo
        return st._getvalue()
    return sig, codegen


@njit(**jit_options)
def create_any_vector(capacity):
    """Create an empty ``AnyVector`` with the given initial ``capacity`` >= 1."""
    assert capacity >= 1
    v = _new_any_vector(AnyVectorType)
    v.buf = numpy.empty(capacity, dtype=numpy.intp)
    v.size = 0
    return v


@njit(**jit_options)
def _reserve_one(v):
    if v.size == v.buf.shape[0]:
        new_buf = numpy.empty(v.buf.shape[0] * 2, v.buf.dtype)
        new_buf[:v.size] = v.buf[:v.size]
        v.buf = new_buf


def _any_vector_push(v, x):
    raise NotImplementedError("Not callable from Python")


@overload(_any_vector_push, strict=False, jit_options=jit_options)
def ol_any_vector_push(v_ty, x_ty):
    if not isinstance(v_ty, AnyVectorTypeClass):
        return None
    if isinstance(x_ty, AnyTypeClass):
        def impl(v, x):
            _reserve_one(v)
            v.buf[v.size] = export_meminfo(x)
            v.size += 1
        return impl

    def impl(v, x):
        a = make_any(x)
        _reserve_one(v)
        v.buf[v.size] = export_meminfo(a)
        v.size += 1
    return impl


@njit(**jit_options)
def any_vector_push(v, x):
    """Append ``x``; a non-``Any`` value is wrapped via ``make_any`` first."""
    _any_vector_push(v, x)


@overload(len, jit_options=jit_options)
def _any_vector_len(v):
    if isinstance(v, AnyVectorTypeClass):
        def impl(v):
            return v.size
        return impl


@overload(operator.getitem, jit_options=jit_options)
def _any_vector_getitem(v, i):
    if isinstance(v, AnyVectorTypeClass):
        def impl(v, i):
            if i < 0 or i >= v.size:
                raise IndexError("AnyVector index out of range")
            return borrow_structref(AnyType, v.buf[i])
        return impl


@overload(operator.setitem, strict=False, jit_options=jit_options)
def _any_vector_setitem(v, i, x):
    if not isinstance(v, AnyVectorTypeClass):
        return None
    if isinstance(x, AnyTypeClass):
        def impl(v, i, x):
            if i < 0 or i >= v.size:
                raise IndexError("AnyVector index out of range")
            p = export_meminfo(x)
            old = v.buf[i]
            v.buf[i] = p
            if old != 0:
                release_meminfo(old)
        return impl

    def impl(v, i, x):
        if i < 0 or i >= v.size:
            raise IndexError("AnyVector index out of range")
        a = make_any(x)
        p = export_meminfo(a)
        old = v.buf[i]
        v.buf[i] = p
        if old != 0:
            release_meminfo(old)
    return impl


@njit(**jit_options)
def any_vector_pop(v):
    """Remove and return the last element, transferring ownership to the caller."""
    i = v.size - 1
    if i < 0:
        raise IndexError("pop from an empty AnyVector")
    p = v.buf[i]
    a = borrow_structref(AnyType, p)
    v.buf[i] = 0
    v.size = i
    release_meminfo(p)
    return a


@njit(**jit_options)
def any_vector_clear(v):
    """Release every element and empty the vector. Slots are zeroed first, so
    the destructor's later walk over them is a no-op."""
    for i in range(v.size):
        p = v.buf[i]
        v.buf[i] = 0
        if p != 0:
            release_meminfo(p)
    v.size = 0


@njit(**jit_options)
def any_vector_extend(dst, src):
    """Append every element of ``src`` to ``dst``; each copied slot takes its
    own reference, so the vectors own their elements independently.
    Self-extension ``any_vector_extend(v, v)`` is supported."""
    needed = dst.size + src.size
    cap = dst.buf.shape[0]
    if needed > cap:
        while cap < needed:
            cap *= 2
        new_buf = numpy.empty(cap, dst.buf.dtype)
        new_buf[:dst.size] = dst.buf[:dst.size]
        dst.buf = new_buf
    n = src.size
    for i in range(n):
        p = src.buf[i]
        _incref_meminfo(p)
        dst.buf[dst.size + i] = p
    dst.size += n


@overload_method(AnyVectorTypeClass, "get_as", strict=False, jit_options=jit_options)
def ol_vec_get_as(self_ty, i_ty, ty_ref):
    def impl(self, i, ty):
        return self[i].get_as(ty)
    return impl


@overload_method(AnyVectorTypeClass, "push", strict=False, jit_options=jit_options)
def ol_vec_push(self_ty, x_ty):
    def impl(self, x):
        _any_vector_push(self, x)
    return impl


@overload_method(AnyVectorTypeClass, "pop", strict=False, jit_options=jit_options)
def ol_vec_pop(self_ty):
    def impl(self):
        return any_vector_pop(self)
    return impl


@overload_method(AnyVectorTypeClass, "clear", strict=False, jit_options=jit_options)
def ol_vec_clear(self_ty):
    def impl(self):
        return any_vector_clear(self)
    return impl
