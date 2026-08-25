"""Run bench_runtime.py in several fresh processes per tree and report min-of-min ns/call."""
import json
import os
import subprocess
import sys

ROOT = "/tmp/claude-1000/-home-erik/cd1205b3-5bc2-4131-95de-8843a157f6c4/scratchpad"
SCRIPT = f"{ROOT}/bench/bench_runtime.py"
PROCS = int(os.environ.get("DRIVE_PROCS", "3"))
N = os.environ.get("DRIVE_N", "20000000")
REPS = os.environ.get("DRIVE_REPS", "9")

out = {}
meta = {}
for tree in ("base", "fix"):
    per = {}
    for p in range(PROCS):
        env = dict(os.environ)
        env["NUMBA_CACHE_DIR"] = f"{ROOT}/nbc_{tree}_rt"
        env["BENCH_TREE"] = tree
        env["BENCH_N"] = N
        env["BENCH_REPS"] = REPS
        env["PYTHONSAFEPATH"] = "1"
        r = subprocess.run([f"{ROOT}/{tree}/.venv/bin/python", "-P", SCRIPT],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(tree, "FAILED", r.stderr[-3000:], file=sys.stderr)
            sys.exit(1)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        meta[tree] = d["meta"]
        for row in d["rows"]:
            per.setdefault(row["label"], []).append(row["ns_per_call"])
    out[tree] = per

print(json.dumps({"meta": meta}, indent=1))
labels = list(out["base"].keys())
print(f"{'shape':22s} {'base ns/call':>13s} {'fix ns/call':>13s} {'delta ns':>9s} {'ratio':>7s}")
summary = {}
for lab in labels:
    b = min(out["base"][lab])
    f = min(out["fix"][lab])
    summary[lab] = {"base": b, "fix": f, "delta_ns": f - b, "ratio": f / b,
                    "base_all": out["base"][lab], "fix_all": out["fix"][lab]}
    print(f"{lab:22s} {b:13.3f} {f:13.3f} {f - b:9.3f} {f / b:7.2f}x")
with open(f"{ROOT}/bench/runtime_summary.json", "w") as fh:
    json.dump(summary, fh, indent=1)
