"""Expose a numpy array as a read-only SQLite virtual table (register_table).

A single generic sqlite3_module (built once at import) serves every table; the
per-table base pointer / strides / dtype tags / schema live in a ctypes
descriptor passed as pClientData. See
docs/plans/2026-05-31-sqlite-vtable-design.md.
"""
import numpy as np
from numba import carray, njit
from numba.core.types import uint8, uint32

from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.core.bindings.utils import load_lib
from numbox.core.proxy.proxy import proxy
from numbox.utils.lowlevel import _cast_int_to_void_p, load_unaligned

load_lib("sqlite3")

# dtype tags (col_tags[j])
_TAG_I8, _TAG_I16, _TAG_I32, _TAG_I64 = 0, 1, 2, 3
_TAG_U8, _TAG_U16, _TAG_U32, _TAG_U64 = 4, 5, 6, 7
_TAG_F32, _TAG_F64, _TAG_BOOL = 8, 9, 10
_TAG_S, _TAG_U, _TAG_BLOB = 11, 12, 13

# descriptor field byte offsets (mirror of _NdarrayTableDescriptor)
_D_NROWS, _D_NCOLS, _D_ROW_STRIDE, _D_DATA_BASE = 0, 8, 16, 24
_D_COL_OFFSETS, _D_COL_TAGS, _D_COL_WIDTHS, _D_SCHEMA, _D_SCRATCH = 32, 40, 48, 56, 64

# vtab layout: base sqlite3_vtab is 24 bytes; the descriptor_ptr is appended at +24
_VTAB_DESC, _VTAB_SIZE = 24, 32
# cursor layout: pVtab(+0), descriptor(+8), rowid(+16), scratch(+24)
_CUR_PVTAB, _CUR_DESC, _CUR_ROWID, _CUR_SCRATCH = 0, 8, 16, 24


@proxy(signatures.get("sqlite3_create_module"), jit_options={"cache": True})
def sqlite3_create_module(db, z_name, p_module, p_client_data):
    return _call_lib_func("sqlite3_create_module", (db, z_name, p_module, p_client_data))


@proxy(signatures.get("sqlite3_declare_vtab"), jit_options={"cache": True})
def sqlite3_declare_vtab(db, z_sql):
    return _call_lib_func("sqlite3_declare_vtab", (db, z_sql))


@proxy(signatures.get("sqlite3_malloc"), jit_options={"cache": True})
def sqlite3_malloc(n):
    return _call_lib_func("sqlite3_malloc", (n,))


@njit(cache=True)
def _nul_trimmed_len(p, width):
    buf = carray(_cast_int_to_void_p(p), (width,), dtype=np.uint8)
    n = 0
    while n < width and buf[n] != 0:
        n += 1
    return n


@njit(cache=True)
def utf32_to_utf8(src, n_codepoints, dst):
    """Encode up to ``n_codepoints`` UTF-32 code points at ``src`` into UTF-8 at ``dst``.

    Stops at the first NUL code point. Returns the number of UTF-8 bytes written; the
    output is not NUL-terminated (callers pass this count explicitly, e.g. to
    ``sqlite3_result_text``). ``dst`` must hold at least ``4 * n_codepoints`` bytes.
    """
    out = carray(_cast_int_to_void_p(dst), (4 * n_codepoints + 1,), dtype=np.uint8)
    k = 0
    for i in range(n_codepoints):
        cp = load_unaligned(src + 4 * i, uint32)
        if cp == 0:
            break
        if cp > 0x10FFFF or (0xD800 <= cp <= 0xDFFF):
            cp = 0xFFFD
        if cp < 0x80:
            out[k] = uint8(cp)
            k += 1
        elif cp < 0x800:
            out[k] = uint8(0xC0 | (cp >> 6))
            out[k + 1] = uint8(0x80 | (cp & 0x3F))
            k += 2
        elif cp < 0x10000:
            out[k] = uint8(0xE0 | (cp >> 12))
            out[k + 1] = uint8(0x80 | ((cp >> 6) & 0x3F))
            out[k + 2] = uint8(0x80 | (cp & 0x3F))
            k += 3
        else:
            out[k] = uint8(0xF0 | (cp >> 18))
            out[k + 1] = uint8(0x80 | ((cp >> 12) & 0x3F))
            out[k + 2] = uint8(0x80 | ((cp >> 6) & 0x3F))
            out[k + 3] = uint8(0x80 | (cp & 0x3F))
            k += 4
    return k
