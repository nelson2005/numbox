import collections
import json

d = json.load(open("/tmp/reprobe_props/work/aliases/fix.json"))
rows = d["rows"]
cf = collections.Counter(r["cfunc"] for r in rows.values())
cc = collections.Counter(r["cc"] for r in rows.values())
print("sites", len(rows))
print("distinct cfunc aliases", len(cf))
print("distinct cc aliases", len(cc))
print("cfunc aliases shared by >1 site", sum(1 for v in cf.values() if v > 1))
print("cc aliases shared by >1 site", sum(1 for v in cc.values() if v > 1))
shared = [(a, n) for a, n in cc.items() if n > 1]
for a, n in sorted(shared, key=lambda kv: -kv[1])[:10]:
    names = sorted(k for k, r in rows.items() if r["cc"] == a)
    print("  ", n, a, names[:6])
