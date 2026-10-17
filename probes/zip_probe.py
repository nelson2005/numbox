"""Zip a package's .py files, import it from the archive in two child processes and report what numba's cache did.

Usage: ``python zip_probe.py <package directory> <code for the child>``.  The children run with NUMBA_DEBUG_CACHE=1,
so numba prints one line for each cache entry it saves or loads.  A working cache saves in the first child and loads
in the second.
"""
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numba

pkg = Path(sys.argv[1]).resolve()
code = sys.argv[2]
tmp = Path(tempfile.mkdtemp())
archive = tmp / f"{pkg.name}.zip"
with zipfile.ZipFile(archive, "w") as z:
    for src in sorted(pkg.rglob("*.py")):
        z.write(src, src.relative_to(pkg.parent).as_posix())
print("platform", sys.platform, "python", sys.version.split()[0], "numba", numba.__version__)
print("archive", archive, "members", len(zipfile.ZipFile(archive).namelist()))
env = dict(os.environ, PYTHONPATH=str(archive), NUMBA_DEBUG_CACHE="1")
for name in ("NUMBA_CACHE_DIR", "NUMBOX_JIT_OPTIONS", "NUMBARROW_JIT_OPTIONS"):
    env.pop(name, None)
results = []
for attempt in (1, 2):
    run = subprocess.run([sys.executable, "-W", "always", "-c", code], env=env, cwd=str(tmp), capture_output=True,
                         text=True)
    lines = run.stdout.splitlines()
    saved = [line for line in lines if line.startswith("[cache] data saved to")]
    loaded = [line for line in lines if line.startswith("[cache] data loaded from")]
    results.append((run.returncode, len(saved), len(loaded)))
    print(f"run {attempt}: exit {run.returncode}, cache entries saved {len(saved)}, loaded {len(loaded)}")
    for line in (saved + loaded)[:1]:
        print("  first:", line)
    for line in run.stderr.splitlines():
        if "Warning" in line:
            print("  warning:", line)
    if run.returncode != 0:
        print("  --- stderr tail ---")
        print("\n".join(run.stderr.splitlines()[-25:]))
(exit1, saved1, _), (exit2, _, loaded2) = results
if exit1 or exit2:
    print("VERDICT: the import from the zip failed")
elif saved1 and loaded2:
    print(f"VERDICT: imports from the zip, first process saved {saved1} entries, second loaded {loaded2}")
else:
    print(f"VERDICT: imports from the zip without a cache (saved {saved1}, loaded {loaded2})")
