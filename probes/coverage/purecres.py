"""A process that uses only `cres` -- no @proxy anywhere -- with an edited body and a warm cache.

The alias mechanism used to live entirely inside `numbox.core.proxy.proxy`, so importing a minter
installed the load guard. Extending it to `cres` means a process can hold an alias with that module
never imported. Also checks the guard really is installed in such a process, and what strict mode does.
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

VENVS = {"proto": "/tmp/cov/venv_proto", "branch": "/tmp/cov/venv_branch", "main": "/tmp/cov/venv_main"}

BODY = '''
"""A cres-minted derive. No @proxy in this process."""
from numba import types
from numbox.utils.highlevel import cres

SIG = types.float64(types.float64)


@cres(SIG, cache=True)
def scale(x):
    return {expr}
'''

CALLER = textwrap.dedent('''
    from numba import njit
    from numba.core.types import float64

    from bmod import scale


    @njit(float64(float64), cache=True)
    def const_scale(x):
        return scale(x) + 1.0


    @njit(float64(float64), cache=True)
    def plain(x):
        return x + 7.0
    ''')

PROBE = textwrap.dedent('''
    import json
    import sys
    import warnings

    import bmod
    from numba.core import caching

    out = {
        "alias": getattr(bmod.scale, "jit_alias", None),
        "guard_installed": bool(getattr(caching, "_numbox_proxy_alias_guard", False)),
        "proxy_module_imported": "numbox.core.proxy.proxy" in sys.modules,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            import cmod
            out["import"] = "ok"
        except BaseException as exc:
            out["import"] = type(exc).__name__
            cmod = None
        res, stats = {}, {}
        if cmod is not None:
            for name in ("const_scale", "plain"):
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


def run(d, tree, strict=False):
    env = dict(os.environ)
    for hostile in ("PYTHONWARNINGS", "PYTHONDEVMODE", "NUMBA_DISABLE_JIT", "NUMBA_OPT",
                    "NUMBOX_PROXY_CACHE_STRICT", "PYTHONPATH"):
        env.pop(hostile, None)
    env.update({"NUMBA_CACHE_DIR": str(d / "nbcache"), "PYTHONPATH": str(d),
                "PYTHONDONTWRITEBYTECODE": "1"})
    if strict:
        env["NUMBOX_PROXY_CACHE_STRICT"] = "1"
    r = subprocess.run([f"{VENVS[tree]}/bin/python", str(d / "probe.py")], capture_output=True,
                       text=True, env=env, timeout=900, errors="replace")
    out = {"rc": r.returncode, "stdout_tail": r.stdout[-200:], "stderr_tail": r.stderr[-400:]}
    for line in r.stdout.splitlines():
        if line.startswith("JSON "):
            out.update(json.loads(line[5:]))
    return out


def show(label, v):
    print(f"  {label:26s} rc={v['rc']:<4} guard={v.get('guard_installed')} "
          f"proxy_imported={v.get('proxy_module_imported')} alias={v.get('alias')} "
          f"res={v.get('results')} stats={v.get('stats')} warn={v.get('warnings')}")
    if v["rc"] != 0:
        print("     stdout:", v["stdout_tail"])
        print("     stderr:", v["stderr_tail"])


tree = sys.argv[1]
d = Path(f"/tmp/cov/runs/purecres-{tree}")
if d.exists():
    shutil.rmtree(d)
write(d, "x * 2.0")
print(f"### pure-cres process, tree={tree}")
show("cold", run(d, tree))
show("warm", run(d, tree))
write(d, "x * 3.0")
show("warm_after_edit", run(d, tree))
show("warm_after_edit_2", run(d, tree))
write(d, "x * 4.0")
show("edit again, STRICT", run(d, tree, strict=True))
