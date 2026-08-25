"""Minimal ELF64-LE symbol reader: returns (undefined, defined_weak, defined_global)."""
import struct


def symbols(obj):
    und, weak, glob = set(), set(), set()
    if obj[:4] != b"\x7fELF" or len(obj) < 0x40 or obj[4] != 2 or obj[5] != 1:
        return und, weak, glob
    (e_shoff,) = struct.unpack_from("<Q", obj, 0x28)
    e_shentsize, e_shnum = struct.unpack_from("<HH", obj, 0x3A)
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        (sh_type,) = struct.unpack_from("<I", obj, sh + 4)
        if sh_type != 2:
            continue
        sym_off, sym_size = struct.unpack_from("<QQ", obj, sh + 24)
        (sh_link,) = struct.unpack_from("<I", obj, sh + 40)
        (sh_entsize,) = struct.unpack_from("<Q", obj, sh + 56)
        (str_off,) = struct.unpack_from("<Q", obj, e_shoff + sh_link * e_shentsize + 24)
        for j in range(sym_size // sh_entsize):
            base = sym_off + j * sh_entsize
            (st_name,) = struct.unpack_from("<I", obj, base)
            (st_info,) = struct.unpack_from("<B", obj, base + 4)
            (st_shndx,) = struct.unpack_from("<H", obj, base + 6)
            if not st_name:
                continue
            end = obj.index(b"\x00", str_off + st_name)
            name = obj[str_off + st_name:end].decode("utf-8", "replace")
            if st_shndx == 0:
                und.add(name)
            elif (st_info >> 4) == 2:
                weak.add(name)
            else:
                glob.add(name)
    return und, weak, glob
