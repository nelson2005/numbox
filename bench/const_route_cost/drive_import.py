"""Per-module import cost, base vs fix, min over fresh processes with a warm numba cache."""
import json
import os
import subprocess
import sys

ROOT = "/tmp/claude-1000/-home-erik/cd1205b3-5bc2-4131-95de-8843a157f6c4/scratchpad"
SCRIPT = f"{ROOT}/bench/bench_import.py"
ROUNDS = int(os.environ.get("ROUNDS", "5"))

MODULES = [ln.strip().strip(",").strip('"') for ln in open(SCRIPT).read().split("MODULES = [")[1].split("]")[0].splitlines() if ln.strip()]

res = {}
info = {}
for tree in ("base", "fix"):
    res[tree] = {}
    for m in MODULES:
        times = []
        for r in range(ROUNDS + 1):
            env = dict(os.environ)
            env["NUMBA_CACHE_DIR"] = f"{ROOT}/nbc_{tree}_import"
            env["BENCH_TREE"] = tree
            env["PYTHONSAFEPATH"] = "1"
            p = subprocess.run([f"{ROOT}/{tree}/.venv/bin/python", "-P", SCRIPT, m],
                               capture_output=True, text=True, env=env)
            if p.returncode != 0:
                print(tree, m, "FAILED", p.stderr[-2000:], file=sys.stderr)
                sys.exit(1)
            d = json.loads(p.stdout.strip().splitlines()[-1])
            if r:
                times.append(d["import_s"])
            info[(tree, m)] = (d["n_proxy_bindings"], d["n_with_as_func"], d["n_modules_loaded"])
        res[tree][m] = times

print(f"{'module':42s} {'n':>3s} {'base s':>8s} {'fix s':>8s} {'delta s':>9s} {'ratio':>7s}")
tot_b = tot_f = 0.0
rows = {}
for m in MODULES:
    b = min(res["base"][m])
    f = min(res["fix"][m])
    n = info[("fix", m)][0]
    tot_b += b
    tot_f += f
    rows[m] = {"base": b, "fix": f, "n_bindings": n,
               "base_all": res["base"][m], "fix_all": res["fix"][m]}
    print(f"{m:42s} {n:3d} {b:8.4f} {f:8.4f} {f - b:9.4f} {f / b:7.3f}x")
print(f"{'TOTAL (independent imports)':42s} {'':3s} {tot_b:8.4f} {tot_f:8.4f} {tot_f - tot_b:9.4f} {tot_f / tot_b:7.3f}x")
print()
print("modules loaded after import (base/fix):")
for m in MODULES:
    print(f"  {m:42s} {info[('base', m)][2]:4d} / {info[('fix', m)][2]:4d}")
with open(f"{ROOT}/bench/import_summary.json", "w") as fh:
    json.dump(rows, fh, indent=1)
