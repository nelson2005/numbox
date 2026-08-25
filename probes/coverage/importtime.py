"""Warm-cache import cost of numbox's shipped bindings, both trees."""
import os
import statistics
import subprocess
import sys
import time

VENVS = {"proto": "/tmp/cov/venv_proto", "branch": "/tmp/cov/venv_branch", "main": "/tmp/cov/venv_main"}
CODE = "import numbox.core.bindings.libm, numbox.core.bindings.sqlite.column, numbox.core.bindings.libc"

for tree in sys.argv[1:]:
    env = dict(os.environ)
    env["NUMBA_CACHE_DIR"] = f"/tmp/cov/runs/imp_{tree}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    subprocess.run([f"{VENVS[tree]}/bin/python", "-P", "-c", CODE], env=env, check=True,
                   capture_output=True, timeout=900)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        subprocess.run([f"{VENVS[tree]}/bin/python", "-P", "-c", CODE], env=env, check=True,
                       capture_output=True, timeout=900)
        times.append(time.perf_counter() - t)
    print(f"{tree:8s} median {statistics.median(times):.3f}s  min {min(times):.3f}s  "
          f"all {[round(t, 3) for t in times]}")
