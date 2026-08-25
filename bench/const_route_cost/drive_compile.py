"""Cold/warm compile timing and cache-size comparison, several fresh cache dirs per tree."""
import json
import os
import shutil
import subprocess
import sys

ROOT = "/tmp/claude-1000/-home-erik/cd1205b3-5bc2-4131-95de-8843a157f6c4/scratchpad"
SCRIPT = f"{ROOT}/bench/bench_compile.py"
ROUNDS = int(os.environ.get("ROUNDS", "5"))

res = {}
for tree in ("base", "fix"):
    cold_all, warm_all, sizes, warns, stats = [], [], [], set(), []
    for r in range(ROUNDS):
        cdir = f"{ROOT}/nbc_{tree}_compile_{r}"
        shutil.rmtree(cdir, ignore_errors=True)
        os.makedirs(cdir)
        env = dict(os.environ)
        env["NUMBA_CACHE_DIR"] = cdir
        env["BENCH_TREE"] = tree
        env["PYTHONSAFEPATH"] = "1"
        for phase, sink in (("cold", cold_all), ("warm", warm_all)):
            p = subprocess.run([f"{ROOT}/{tree}/.venv/bin/python", "-P", SCRIPT, phase],
                               capture_output=True, text=True, env=env)
            if p.returncode != 0:
                print(tree, phase, "FAILED", p.stderr[-3000:], file=sys.stderr)
                sys.exit(1)
            d = json.loads(p.stdout.strip().splitlines()[-1])
            sink.append(d["compile_s"])
            warns.update(d["warnings"])
            if phase == "warm":
                sizes.append(d["files"])
                stats.append(d["stats"])
        shutil.rmtree(cdir, ignore_errors=True)
    res[tree] = {"cold": cold_all, "warm": warm_all, "sizes": sizes[-1],
                 "warnings": sorted(warns), "stats": stats[-1]}

names = list(res["base"]["cold"][0].keys())
print(f"{'caller':12s} {'phase':5s} {'base s':>9s} {'fix s':>9s} {'delta s':>9s} {'ratio':>7s}")
summary = {}
for phase in ("cold", "warm"):
    for name in names:
        b = min(d[name] for d in res["base"][phase])
        f = min(d[name] for d in res["fix"][phase])
        summary[f"{phase}:{name}"] = {"base": b, "fix": f}
        print(f"{name:12s} {phase:5s} {b:9.4f} {f:9.4f} {f - b:9.4f} {f / b:7.2f}x")

print()
print("cache file sizes (warm run, last round):")
for name in names:
    for kind in ("nbi", "nbc"):
        b = res["base"]["sizes"][name][kind]
        f = res["fix"]["sizes"][name][kind]
        bs = sum(s for _, s in b)
        fs = sum(s for _, s in f)
        print(f"  {name:12s} {kind} base={bs:8d} fix={fs:8d} delta={fs - bs:+8d}")

print()
print("warm cache stats:", json.dumps({t: res[t]["stats"] for t in res}))
print("warnings base:", res["base"]["warnings"])
print("warnings fix :", res["fix"]["warnings"])
with open(f"{ROOT}/bench/compile_summary.json", "w") as fh:
    json.dump(res, fh, indent=1)
