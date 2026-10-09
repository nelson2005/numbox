"""Import numbox from a zip archive in a child process and print what happens.

``sources``: every .py member is in the archive.  ``pyc``: numbox/core/bindings goes in as .pyc members compiled
to name their path inside the archive, as compileall -d does, every other member as .py.
"""
import os
import py_compile
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numba
import numbox

layout = sys.argv[1]
pkg = Path(numbox.__file__).parent
tmp = Path(tempfile.mkdtemp())
archive = tmp / "numbox.zip"
with zipfile.ZipFile(archive, "w") as z:
    for src in sorted(pkg.rglob("*.py")):
        member = src.relative_to(pkg.parent).as_posix()
        if layout == "pyc" and src.parent == pkg / "core" / "bindings":
            named = os.path.join(str(archive), *member.split("/"))
            pyc = py_compile.compile(str(src), cfile=str(tmp / "x.pyc"), dfile=named, doraise=True)
            z.write(pyc, member + "c")
        else:
            z.write(src, member)
names = zipfile.ZipFile(archive).namelist()
print("platform", sys.platform, "python", sys.version.split()[0], "numba", numba.__version__)
present = all(s.relative_to(pkg.parent).as_posix() in names for s in pkg.rglob("*.py"))
print("layout", layout, "members", len(names), "sources present:", present)
env = dict(os.environ, PYTHONPATH=str(archive))
env.pop("NUMBA_CACHE_DIR", None)
env.pop("NUMBOX_JIT_OPTIONS", None)
code = "import numbox.core.configurations, numbox.core.bindings.libm as m; print(m.__file__)"
run = subprocess.run([sys.executable, "-W", "always", "-c", code], env=env, cwd=str(tmp), capture_output=True, text=True)
print("child exit", run.returncode)
print("--- child stdout ---")
print(run.stdout)
print("--- child stderr ---")
print(run.stderr)
