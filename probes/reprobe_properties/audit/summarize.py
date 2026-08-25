import collections
import json
import sys

data = json.load(open(sys.argv[1]))
print("publish_attempts", data["publish_attempts"])
print("granted", data["granted"], "refused", data["refused"])
print("distinct_aliases", data["distinct_aliases"])
print("duplicate_attempts", data["duplicate_attempts"], "of which same cres", data["duplicate_same_cres"])
print("const_lowerings", data["const_lowerings"], "without alias", data["const_without_alias"])
print()
byalias = collections.Counter(r["alias"] for r in data["records"])
dupes = {a: n for a, n in byalias.items() if n > 1}
print("colliding alias names", len(dupes), "covering", sum(dupes.values()), "publishes")
for a, n in sorted(dupes.items(), key=lambda kv: -kv[1]):
    recs = [r for r in data["records"] if r["alias"] == a]
    quals = sorted(set(r["mod"] + "." + r["qual"] for r in recs))
    codes = len(set(r["code_id"] for r in recs))
    funcs = len(set(r["py_func_id"] for r in recs))
    llvm = len(set(r["llvm_name"] for r in recs))
    addrs = len(set(r["addr"] for r in recs))
    print("  n=%d distinct_code=%d distinct_pyfunc=%d distinct_llvmname=%d distinct_addr=%d %s"
          % (n, codes, funcs, llvm, addrs, quals))
print()
print("refused records:")
for r in data["records"]:
    if not r["granted"]:
        print("  ", r["mod"] + "." + r["qual"], r["alias"])
print()
print("const lowerings:")
for c in data["const"]:
    print("  ", c["mod"] + "." + c["qual"], "alias=", c["alias"])
