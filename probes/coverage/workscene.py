"""A cache=True jitted builder that makes a Work from a @proxy binding's .as_func as a constant.

This is the shape PR #96 exists for. If the const route is cacheable there too, the staleness
question applies to it, and the fix has to cover it.
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
"""Proxied binding used as a Work derive."""
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

    from numbox.core.work.work import make_work

    from bmod import HANDLE


    @njit(cache=True)
    def build_and_calculate(x):
        source = make_work("source", x)
        node = make_work("node", 0.0, sources=(source,), derive=HANDLE)
        node.calculate()
        return node.data
    ''')

PROBE = textwrap.dedent('''
    import json
    import warnings

    import bmod

    out = {"alias": getattr(bmod.HANDLE, "jit_alias", None)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from cmod import build_and_calculate
        try:
            out["result"] = repr(build_and_calculate(5.0))
        except BaseException as exc:
            out["result"] = f"RAISED:{type(exc).__name__}: {str(exc)[:120]}"
        h = sum(build_and_calculate.stats.cache_hits.values())
        m = sum(build_and_calculate.stats.cache_misses.values())
        out["stats"] = "served" if h and not m else "compiled" if m and not h else f"{h}/{m}"
    out["warnings"] = [f"{w.category.__name__}: {str(w.message)[:150]}" for w in caught]
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
    out = {"rc": r.returncode, "stderr_tail": r.stderr[-400:]}
    for line in r.stdout.splitlines():
        if line.startswith("JSON "):
            out.update(json.loads(line[5:]))
    return out


tree = sys.argv[1]
d = Path(f"/tmp/cov/runs/work-{tree}")
if d.exists():
    shutil.rmtree(d)
write(d, "x * 2.0")
steps = [("cold", run(d, tree)), ("warm", run(d, tree))]
write(d, "x * 3.0")
steps.append(("warm_after_edit", run(d, tree)))
steps.append(("warm_after_edit_2", run(d, tree)))
print(f"### jitted-scope Work, tree={tree}")
for label, v in steps:
    print(f"  {label:18s} rc={v['rc']:<4} alias={v.get('alias')} result={v.get('result')} "
          f"stats={v.get('stats')} warn={v.get('warnings')}")
    if v["rc"] != 0:
        print("     stderr:", v["stderr_tail"])
