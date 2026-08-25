import json
import sys

trees = {}
for name in ("main", "pr", "proto", "fix"):
    trees[name] = json.load(open("/tmp/reprobe_props/work/aliases/" + name + ".json"))

keys = set()
for d in trees.values():
    keys |= set(d["rows"])
print("union of attribute sites", len(keys))
for name, d in trees.items():
    print(name, "sites", d["count"], "as_func types",
          sorted(set(r["as_func_type"] for r in d["rows"].values())),
          "cc present", sum(1 for r in d["rows"].values() if r["cc"] not in ("NOATTR", "NOASFUNC", None)),
          "cc None", sum(1 for r in d["rows"].values() if r["cc"] is None))


def cmp(a, b, field):
    moved = []
    missing = []
    for k in sorted(keys):
        ra = trees[a]["rows"].get(k)
        rb = trees[b]["rows"].get(k)
        if ra is None or rb is None:
            missing.append(k)
            continue
        if ra[field] != rb[field]:
            moved.append((k, ra[field], rb[field]))
    print("%s -> %s  %s: compared %d, moved %d, absent on one side %d"
          % (a, b, field, len(keys) - len(missing), len(moved), len(missing)))
    for row in moved[:10]:
        print("   ", row)


cmp("main", "pr", "cfunc")
cmp("pr", "proto", "cfunc")
cmp("proto", "fix", "cfunc")
cmp("pr", "fix", "cfunc")
cmp("main", "fix", "cfunc")
cmp("proto", "fix", "cc")
