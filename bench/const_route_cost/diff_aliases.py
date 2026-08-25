"""Diff the shipped-binding cfunc alias values between the two trees."""
import json

ROOT = "/tmp/claude-1000/-home-erik/cd1205b3-5bc2-4131-95de-8843a157f6c4/scratchpad/bench"
b = json.load(open(f"{ROOT}/aliases_base.json"))
f = json.load(open(f"{ROOT}/aliases_fix.json"))

bc, fc = b["cfunc_aliases"], f["cfunc_aliases"]
print("base bindings:", b["n_bindings"], "fix bindings:", f["n_bindings"])
print("base jit_aliases:", b["n_jit_aliases"], "fix jit_aliases:", f["n_jit_aliases"])
only_b = sorted(set(bc) - set(fc))
only_f = sorted(set(fc) - set(bc))
changed = sorted(k for k in set(bc) & set(fc) if bc[k] != fc[k])
print("keys only in base:", only_b)
print("keys only in fix :", only_f)
print("cfunc aliases that CHANGED:", len(changed))
for k in changed[:20]:
    print("  ", k, bc[k], "->", fc[k])
print("cfunc aliases byte-identical:", len(set(bc) & set(fc)) - len(changed))
ja = f["jit_aliases"]
print("new callconv aliases minted:", len(ja))
dupes = len(set(ja.values()))
print("distinct callconv alias values:", dupes)
collide = set(ja.values()) & set(fc.values())
print("collisions between the two alias families:", sorted(collide))
sample = sorted(ja.items())[:3]
for k, v in sample:
    print("  sample", k, v)
