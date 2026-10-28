"""ABI primitives for numba bindings: the classification of platforms and structs."""
import platform
import sys

from numba.core import types as nb_types
from numba.core.errors import TypingError


_PLATFORM_WIN_X64 = "win_x64"
_PLATFORM_SYSV_X86_64 = "sysv_x86_64"
_PLATFORM_AAPCS64 = "aapcs64"


def _current_platform():
    """Identify the C calling convention for the current host.

    The function returns one of ``_PLATFORM_WIN_X64``, ``_PLATFORM_SYSV_X86_64`` and ``_PLATFORM_AAPCS64``. The
    ABI-aware code generation in ``numbox.core.bindings.call._call_lib_func`` uses the result to select the
    struct-passing convention. An unsupported ``(sys.platform, platform.machine())`` combination raises
    ``RuntimeError`` instead of being misclassified without an error. Windows ARM64 (`platform.machine() == "ARM64"`)
    is unsupported, and without the error it would default to the wrong ABI dispatch.
    """
    machine = platform.machine().lower()
    if sys.platform == "win32":
        if machine in ("x86_64", "amd64"):
            return _PLATFORM_WIN_X64
        raise RuntimeError(
            f"Unsupported Windows architecture for ABI dispatch: "
            f"{platform.machine()} (only x86_64 / AMD64 supported)"
        )
    if machine in ("x86_64", "amd64"):
        return _PLATFORM_SYSV_X86_64
    if machine in ("arm64", "aarch64"):
        return _PLATFORM_AAPCS64
    raise RuntimeError(
        f"Unsupported platform for ABI dispatch: "
        f"{sys.platform}/{platform.machine()}"
    )


_CLASS_SCALAR = "scalar"
_CLASS_STRUCT_SMALL = "struct_small"
_CLASS_STRUCT_LARGE = "struct_large"


_WIN_REGISTER_PASSABLE_SIZES = (1, 2, 4, 8)


def _is_windows_register_passable(struct_bytes):
    """Return whether the Windows x64 ABI passes and returns a struct of ``struct_bytes`` bytes in registers.

    Windows x64 passes an aggregate of 1, 2, 4 or 8 bytes directly in integer registers, and returns it in RAX. An
    aggregate of any other size (3, 5, 6, 7, or more than 8 bytes) is passed through a pointer that the caller
    allocates, and it is returned through ``sret``.
    """
    return struct_bytes in _WIN_REGISTER_PASSABLE_SIZES


def _classify(ty):
    """Classify a numba type for ABI dispatch.

    The function returns one of these classes:

    - ``_CLASS_SCALAR`` for any numba type that is not a struct, for example ``int32``, ``float64`` or a pointer
      represented as ``intp``;
    - ``_CLASS_STRUCT_SMALL`` for a ``Record`` or ``BaseTuple`` of at most 16 bytes, which register-passing ABIs pass
      by value;
    - ``_CLASS_STRUCT_LARGE`` for a ``Record`` or ``BaseTuple`` of more than 16 bytes, which is passed by pointer
      with ``byval`` on SysV x86-64, and by pointer without a special attribute on other ABIs.
    """
    if not isinstance(ty, (nb_types.Record, nb_types.BaseTuple)):
        return _CLASS_SCALAR
    size = _struct_bytes(ty, "_classify")
    return _CLASS_STRUCT_SMALL if size <= 16 else _CLASS_STRUCT_LARGE


def _align_up(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def _basetuple_layout(ty, fn_name):
    """Return the natural-alignment C layout of a ``BaseTuple`` of scalar fields.

    The function returns ``(total_size, [(offset, size, is_float), ...])``.

    numba lowers a tuple through its ``StructModel`` (the tuple data model) to a literal LLVM struct, and llvmlite
    emits that struct *non-packed* (padded) by default. Each field therefore starts at the next offset aligned to its
    own size, and the total is padded to the alignment of the largest field. This is exactly the layout of the
    equivalent ``Record``, as verified against ``get_data_type`` and ``get_abi_sizeof``. A plain bit-packed model
    with ``sum(bitwidth)`` undercounts the size and puts fields at wrong offsets. That model would corrupt both the
    SMALL/LARGE classification and the eightbyte split.

    A field with no fixed ``bitwidth``, such as a pointer, a function or another non-scalar type, raises a clear
    ``TypingError`` naming ``fn_name`` rather than an uninformative ``AttributeError``. Raw pointers are expected to
    be passed as ``intp``.
    """
    offset = 0
    max_align = 1
    fields = []
    for ft in ty.types:
        if not hasattr(ft, "bitwidth"):
            raise TypingError(
                f"{fn_name}: tuple field {ft!r} has no fixed bitwidth; only "
                f"fixed-width scalar fields (Integer, Float) are supported "
                f"(pointers must be passed as intp)."
            )
        size = ft.bitwidth // 8
        align = size  # primitive scalar fields: alignment == size
        offset = _align_up(offset, align)
        fields.append((offset, size, isinstance(ft, nb_types.Float)))
        offset += size
        max_align = max(max_align, align)
    return _align_up(offset, max_align), fields


def _struct_bytes(ty, fn_name):
    """Compute the size in bytes of a numba struct-shaped type.

    A ``types.Record`` gives its size through ``.size``. The ``types.BaseTuple`` subclasses ``types.Tuple``,
    ``types.UniTuple`` and ``types.NamedTuple`` are measured with the natural-alignment layout (see
    ``_basetuple_layout``). Any other type raises a clear ``TypingError`` naming the caller, so that a misuse does not
    end in an uninformative ``AttributeError`` or ``KeyError``.
    """
    if isinstance(ty, nb_types.Record):
        return ty.size
    if isinstance(ty, nb_types.BaseTuple):
        return _basetuple_layout(ty, fn_name)[0]
    raise TypingError(
        f"{fn_name}: expected a struct-shaped type (Record, Tuple, "
        f"UniTuple, or NamedTuple), got {ty!r}."
    )


_EIGHTBYTE_CLASS_INTEGER = "integer"
_EIGHTBYTE_CLASS_SSE = "sse"


def _iter_struct_fields(ty, fn_name):
    """Yield ``(offset, size, is_float)`` for each scalar field of a struct-shaped numba type.

    ``is_float`` is true for ``Float`` types (float and double). The SysV ABI's mapping from a float field to an SSE
    eightbyte is the consumer's responsibility (see ``_classify_eightbytes``). The size is needed to detect a field
    that spans the 8-byte eightbyte boundary.

    The fields of a ``BaseTuple`` follow the natural-alignment layout of ``_basetuple_layout``, the same layout that
    ``_struct_bytes`` uses. The padding gaps therefore match the non-packed LLVM struct that numba lowers a tuple to.
    Each ``_RecordField`` of a ``Record`` carries an explicit ``offset``, which already accounts for any C-layout
    padding gaps.
    """
    if isinstance(ty, nb_types.BaseTuple):
        yield from _basetuple_layout(ty, fn_name)[1]
        return
    if isinstance(ty, nb_types.Record):
        for fld in ty.fields.values():
            yield (
                fld.offset,
                fld.type.bitwidth // 8,
                isinstance(fld.type, nb_types.Float),
            )
        return
    raise TypingError(
        f"{fn_name}: expected a struct-shaped type (Record, Tuple, "
        f"UniTuple, or NamedTuple), got {ty!r}."
    )


def _classify_eightbytes(ty):
    """Classify the two eightbytes of a 16-byte struct.

    The function returns ``(class_lo, class_hi)``. Each class is ``_EIGHTBYTE_CLASS_INTEGER`` (ints and pointers) or
    ``_EIGHTBYTE_CLASS_SSE`` (float and double).

    The SysV merge rule applies per eightbyte. An eightbyte is SSE only if every field that touches it is a float or
    a double. If it holds *any* integer field, it is INTEGER and is passed in a general-purpose register, since SSE
    merged with INTEGER gives INTEGER. The class names come from SysV, but the repack path uses this helper on both
    SysV x86-64 and AAPCS64. The full SysV ABI also has X87, NO_CLASS and other classes. This helper covers only the
    fixed-size 16-byte aggregates of scalar integer or float fields that numbox bindings consume.

    ``_call_lib_func`` uses this function together with ``_is_canonical_int64_pair_layout``. The two decide when a
    16-byte by-value argument needs a repack to ``{i64, i64}`` through a memory bitcast. The repack compensates for a
    limitation of llvmlite: llvmlite does not model the eightbyte-packing rule, and it loses fields such as the
    second ``i32`` in ``{i32, i32, i64}`` (duckdb_interval).

    The function raises ``TypingError`` if ``ty`` is not a 16-byte struct-shaped type. The caller should already
    have classified the size.
    """
    size = _struct_bytes(ty, "_classify_eightbytes")
    if size != 16:
        raise TypingError(
            f"_classify_eightbytes: expected a 16-byte struct, got "
            f"{size}-byte ({ty!r})."
        )
    lo_has_int = hi_has_int = False
    lo_has_float = hi_has_float = False
    for offset, size, is_float in _iter_struct_fields(
            ty, "_classify_eightbytes"):
        # A field spanning the 8-byte boundary touches both eightbytes.
        touches_lo = offset < 8
        touches_hi = offset + size > 8
        if is_float:
            lo_has_float |= touches_lo
            hi_has_float |= touches_hi
        else:
            lo_has_int |= touches_lo
            hi_has_int |= touches_hi
    # SysV merge rule: an eightbyte that holds any integer field is INTEGER and is passed in a general-purpose
    # register, even if it also holds a float (SSE merged with INTEGER gives INTEGER). Only an all-float eightbyte is
    # SSE. llvmlite does not model this rule. Without the int/int repack in call.py, llvmlite would lower a mixed
    # int/float eightbyte through SSE and lose the integer field, which is why that repack exists.
    cls_lo = (_EIGHTBYTE_CLASS_SSE if (lo_has_float and not lo_has_int)
              else _EIGHTBYTE_CLASS_INTEGER)
    cls_hi = (_EIGHTBYTE_CLASS_SSE if (hi_has_float and not hi_has_int)
              else _EIGHTBYTE_CLASS_INTEGER)
    return cls_lo, cls_hi


def _is_canonical_int64_pair_layout(ty):
    """Return whether ``ty`` already lowers to LLVM ``{i64, i64}``, with two 64-bit integer fields at offsets 0 and 8.

    Such a layout has no gaps. If the result is True, the SysV-x86-64 small-struct INT/INT path needs no repack,
    because llvmlite already emits the canonical eightbyte-pair shape.
    """
    if isinstance(ty, nb_types.BaseTuple):
        return len(ty.types) == 2 and all(
            isinstance(f, nb_types.Integer) and f.bitwidth == 64
            for f in ty.types
        )
    if isinstance(ty, nb_types.Record):
        if ty.size != 16:
            return False
        # Sort by offset, so that a Record whose fields are declared out of offset order, for example through a
        # numpy structured dtype with explicit `offsets`, is still recognized as canonical when its sorted offsets
        # are [0, 8].
        flds = sorted(ty.fields.values(), key=lambda f: f.offset)
        if len(flds) != 2:
            return False
        if [f.offset for f in flds] != [0, 8]:
            return False
        return all(
            isinstance(f.type, nb_types.Integer) and f.type.bitwidth == 64
            for f in flds
        )
    return False
