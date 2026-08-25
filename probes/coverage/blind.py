"""What happens on a platform whose object reader returns nothing (the documented Windows/COFF path).

Simulated by neutering ``numbox.core.proxy.proxy._undefined_symbols`` in the child before the caller is
imported, which is exactly what the COFF stub does on Windows: the prefix fast path passes (the alias string
IS in the object), the reader is consulted, and it yields an empty set.
"""
import argparse
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

VENVS = {"proto": "/tmp/cov/venv_proto", "main": "/tmp/cov/venv_main", "branch": "/tmp/cov/venv_branch"}

BODY = '''
"""A proxied binding whose body the scenario edits."""
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
    import os
    import warnings

    import bmod
    import numbox.core.proxy.proxy as P

    out = {"alias": getattr(bmod.HANDLE, "jit_alias", None), "blind": os.environ.get("BLIND", "0")}
    if os.environ.get("BLIND") == "1":
        P._undefined_symbols = lambda object_code: set()
    print("BMOD_IMPORTED", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import cmod
        print("CMOD_IMPORTED", flush=True)
        res, stats = {}, {}
        for name in ("const_scale", "disp_scale"):
            fn = getattr(cmod, name)
            try:
                res[name] = repr(fn(5.0))
            except BaseException as exc:
                res[name] = f"RAISED:{type(exc).__name__}"
            print("CALLED", name, res[name], flush=True)
            h = sum(fn.stats.cache_hits.values())
            m = sum(fn.stats.cache_misses.values())
            stats[name] = "served" if h and not m else "compiled" if m and not h else f"{h}/{m}"
    out["results"], out["stats"] = res, stats
    out["warnings"] = [w.category.__name__ for w in caught]
    print("JSON " + json.dumps(out), flush=True)
    print("DONE ok", flush=True)
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


def run(d, venv, blind):
    env = dict(os.environ)
    for hostile in ("PYTHONWARNINGS", "PYTHONDEVMODE", "NUMBA_DISABLE_JIT", "NUMBA_OPT",
                    "NUMBOX_PROXY_CACHE_STRICT", "PYTHONPATH"):
        env.pop(hostile, None)
    env.update({"NUMBA_CACHE_DIR": str(d / "nbcache"), "PYTHONPATH": str(d),
                "PYTHONDONTWRITEBYTECODE": "1", "BLIND": "1" if blind else "0"})
    r = subprocess.run([f"{venv}/bin/python", str(d / "probe.py")], capture_output=True, text=True,
                       env=env, timeout=900, errors="replace")
    out = {"rc": r.returncode, "stdout_tail": r.stdout[-200:], "stderr_tail": r.stderr[-300:]}
    for line in r.stdout.splitlines():
        if line.startswith("JSON "):
            out.update(json.loads(line[5:]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True)
    a = p.parse_args()
    venv = VENVS[a.tree]
    for blind in (False, True):
        d = Path(f"/tmp/cov/runs/blind-{a.tree}-{int(blind)}")
        if d.exists():
            shutil.rmtree(d)
        write(d, "x * 2.0")
        steps = [("cold", run(d, venv, blind=False)), ("warm", run(d, venv, blind=False))]
        write(d, "x * 3.0")
        steps.append((f"warm_after_edit blind={blind}", run(d, venv, blind=blind)))
        print(f"### blind-reader tree={a.tree} blind={blind}")
        for s, v in steps:
            print(f"  {s:28s} rc={v['rc']:<4} alias={v.get('alias')} res={v.get('results')} "
                  f"stats={v.get('stats')} warn={v.get('warnings')}")
            if v["rc"] != 0:
                print(f"     stdout={v['stdout_tail']!r}")
                print(f"     stderr={v['stderr_tail']!r}")


if __name__ == "__main__":
    main()
