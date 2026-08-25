"""Dump the const-route caller's asm/LLVM so the inlining and vectorization state is visible."""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mods"))

from numba import njit, types  # noqa: E402

import binding_small  # noqa: E402

SMALL = binding_small.small.as_func
F64 = types.float64
I64 = types.int64


@njit(F64(I64, F64), cache=False)
def const_small_sum(n, x):
    acc = 0.0
    for i in range(n):
        acc += SMALL(x + i)
    return acc


asm = next(iter(const_small_sum.inspect_asm().values()))
llvm = next(iter(const_small_sum.inspect_llvm().values()))
tree = os.environ.get("BENCH_TREE", "?")
vec = len(re.findall(r"\b[vp]?(?:add|mul)pd\b", asm))
ymm = len(re.findall(r"%ymm", asm))
xmm = len(re.findall(r"%xmm", asm))
calls = re.findall(r"call\w*\s+([\w.$@]+)", asm)
print(f"{tree} asm_lines={len(asm.splitlines())} packed_fp_ops={vec} ymm_refs={ymm} xmm_refs={xmm}")
print(f"{tree} call_targets={sorted(set(calls))}")
print(f"{tree} llvm_declares={sorted(set(re.findall(r'^declare[^@]*(@[\\w.$]+)', llvm, re.M)))}")
out = os.environ.get("ASM_OUT")
if out:
    with open(out, "w") as fh:
        fh.write(asm)
