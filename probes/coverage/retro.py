"""Cross-tree cache: entries written by the OLD lowering, then read after the upgrade."""
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

VENVS = {"proto": "/tmp/cov/venv_proto", "branch": "/tmp/cov/venv_branch", "main": "/tmp/cov/venv_main"}

BODY = '''
"""Proxied binding the scenario edits."""
from numba import types
from numbox.core.proxy.proxy import proxy

SIG = types.float64(types.float64)


@proxy(SIG, jit_options={{"cache": True}})
def scale(x):
    return {expr}


HANDLE = scale.as_func
'''

CALLER = textwrap.dedent('''
    from numba import njit
    from numba.core.types import float64

    from bmod import HANDLE, scale


    @njit(float64(float64), cache=True)
    def const_scale(x):
        return HANDLE(x) + 1.0


    @njit(float64(float64), cache=True)
    def disp_scale(x):
        return scale(x) + 2.0
    ''')

PROBE = textwrap.dedent('''
    import json
    import warnings

    import bmod

    out = {"alias": getattr(bmod.HANDLE, "jit_alias", None)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import cmod
        res, stats = {}, {}
        for name in ("const_scale", "disp_scale"):
            fn = getattr(cmod, name)
            try:
                res[name] = repr(fn(5.0))
            except BaseException as exc:
                res[name] = f"RAISED:{type(exc).__name__}"
            h = sum(fn.stats.cache_hits.values())
            m = sum(fn.stats.cache_misses.values())
            stats[name] = "served" if h and not m else "compiled" if m and not h else f"{h}/{m}"
    out["results"], out["stats"] = res, stats
    out["warnings"] = [w.category.__name__ for w in caught]
    print("JSON " + json.dumps(out), flush=True)
    ''')


def write(d, expr):
    d.mkdir(parents=True, exist_ok=True)
    b = d / "bmod.py"
    before = b.stat() if b.exists() else None
    b.write_text(BODY.format(expr=expr), encoding="utf-8")
    if before is not None:
        os.utime(b, (before.st_mtime + 10,) * 2)
    for name, src in (("cmod.py", CALLER), ("probe.py", PROBE)):
        p = d / name
        if not p.exists() or p.read_text(encoding="utf-8") != src:
            p.write_text(src, encoding="utf-8")


def run(d, tree):
    env = dict(os.environ)
    for hostile in ("PYTHONWARNINGS", "PYTHONDEVMODE", "NUMBA_DISABLE_JIT", "NUMBA_OPT",
                    "NUMBOX_PROXY_CACHE_STRICT", "PYTHONPATH"):
        env.pop(hostile, None)
    env.update({"NUMBA_CACHE_DIR": str(d / "nbcache"), "PYTHONPATH": str(d),
                "PYTHONDONTWRITEBYTECODE": "1"})
    r = subprocess.run([f"{VENVS[tree]}/bin/python", str(d / "probe.py")], capture_output=True,
                       text=True, env=env, timeout=900, errors="replace")
    out = {"rc": r.returncode, "stderr_tail": r.stderr[-300:]}
    for line in r.stdout.splitlines():
        if line.startswith("JSON "):
            out.update(json.loads(line[5:]))
    return out


def show(label, v):
    print(f"  {label:34s} rc={v['rc']:<4} alias={v.get('alias')} res={v.get('results')} "
          f"stats={v.get('stats')} warn={v.get('warnings')}")
    if v["rc"] != 0:
        print("     stderr:", v["stderr_tail"])


d = Path("/tmp/cov/runs/retro")
if d.exists():
    shutil.rmtree(d)
write(d, "x * 2.0")
print("### cache written by the OLD lowering (branch), then read by the fixed tree")
show("branch cold", run(d, "branch"))
show("branch warm", run(d, "branch"))
show("proto warm, body unchanged", run(d, "proto"))
write(d, "x * 3.0")
show("proto warm, body EDITED", run(d, "proto"))
show("proto warm, body EDITED, run 2", run(d, "proto"))

d2 = Path("/tmp/cov/runs/retro-fwd")
if d2.exists():
    shutil.rmtree(d2)
write(d2, "x * 2.0")
print("### cache written by the FIXED tree, then read by the old tree (downgrade)")
show("proto cold", run(d2, "proto"))
show("branch warm, body unchanged", run(d2, "branch"))
write(d2, "x * 3.0")
show("branch warm, body EDITED", run(d2, "branch"))
