"""Run the touched test files on one tree with a fresh cache and cleared __pycache__, and time it."""
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = "/tmp/claude-1000/-home-erik/cd1205b3-5bc2-4131-95de-8843a157f6c4/scratchpad"
tree = sys.argv[1]
targets = sys.argv[2:] or [
    "test/core/test_proxy.py",
    "test/core/test_proxy_cache_stale.py",
    "test/utils/test_derive_wap.py",
    "test/utils/test_highlevel.py",
    "test/core/test_caching_work.py",
    "test/core/test_cache_crossprocess.py",
]
base = pathlib.Path(f"{ROOT}/{tree}")
for p in base.rglob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)
cdir = f"{ROOT}/nbc_{tree}_pytest"
shutil.rmtree(cdir, ignore_errors=True)
os.makedirs(cdir)
env = dict(os.environ)
env["NUMBA_CACHE_DIR"] = cdir
env["PYTHONSAFEPATH"] = "1"
t0 = time.perf_counter()
p = subprocess.run([f"{ROOT}/{tree}/.venv/bin/python", "-m", "pytest", "-q", *targets],
                   cwd=str(base), capture_output=True, text=True, env=env)
el = time.perf_counter() - t0
print(f"TREE {tree} rc={p.returncode} wall={el:.1f}s")
print("\n".join(p.stdout.strip().splitlines()[-15:]))
if p.returncode:
    print("STDERR", p.stderr[-3000:])
