"""Build and run the three-door hazard scenarios against a chosen numbox tree."""
import argparse
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

VENVS = {"proto": "/tmp/cov/venv_proto", "main": "/tmp/cov/venv_main", "branch": "/tmp/cov/venv_branch",
         "proto65": "/tmp/cov/venv_proto65", "main65": "/tmp/cov/venv_main65",
         "branch65": "/tmp/cov/venv_branch65", "proto66": "/tmp/cov/venv_proto66", "proto60": "/tmp/cov/venv_proto60"}

SIGHDR = '''
"""Scenario body module."""
import os

from numba import njit, types
'''

CSYM_HDR = '''
import llvmlite.binding as ll
from numba.core.types import ExternalFunction

if os.environ.get("SYMBOL_PRESENT", "1") == "1":
    ll.load_library_permanently("/tmp/cov/lib/libcovsym.so")
_cov = ExternalFunction("cov_scale", types.float64(types.float64))
'''


def body_source(door, name, expr, absent_capable, csym=False):
    pad = "\n" * 3
    hdr = SIGHDR + (CSYM_HDR if csym else "")
    if door in ("a", "c1"):
        head = hdr + textwrap.dedent(f'''
            from numbox.core.proxy.proxy import proxy, proxy_if_available

            SIG = types.float64(types.float64)


            class _Lib:
                pass


            LIB = _Lib()
            if os.environ.get("SYMBOL_PRESENT", "1") == "1":
                setattr(LIB, "{name}", 1)

            _DECO = proxy_if_available(LIB, SIG, jit_options={{"cache": True}}) if {absent_capable!r} \\
                else proxy(SIG, jit_options={{"cache": True}})
            ''')
        tail = textwrap.dedent(f'''
            @_DECO
            def {name}(x):
                return {expr}
            ''')
        if door == "a":
            tail += textwrap.dedent(f'''

                HANDLE = {name}.as_func if hasattr({name}, "as_func") else None
                ALIAS = getattr(HANDLE, "jit_alias", None)
                CFUNC_ALIAS = getattr({name}, "_numbox_proxy_alias", None)
                ''')
        else:
            tail += textwrap.dedent(f'''

                from numbox.utils.derive_wap import rewrap_derive
                _RAW = {name}.as_func if hasattr({name}, "as_func") else None
                HANDLE = rewrap_derive(_RAW)
                ALIAS = getattr(HANDLE, "jit_alias", None)
                CFUNC_ALIAS = getattr({name}, "_numbox_proxy_alias", None)
                ''')
        return head + pad + tail
    if door == "b":
        return hdr + textwrap.dedent(f'''
            from numbox.utils.highlevel import cres, cres_if_available

            SIG = types.float64(types.float64)


            class _Lib:
                pass


            LIB = _Lib()
            if os.environ.get("SYMBOL_PRESENT", "1") == "1":
                setattr(LIB, "{name}", 1)

            _DECO = cres_if_available(LIB, SIG, cache=True) if {absent_capable!r} else cres(SIG, cache=True)


            @_DECO
            def {name}(x):
                return {expr}


            HANDLE = {name} if hasattr({name}, "cres") else None
            ALIAS = getattr(HANDLE, "jit_alias", None)
            CFUNC_ALIAS = None
            ''')
    if door == "c2":
        return hdr + textwrap.dedent(f'''
            from numba.core.types.function_type import CompileResultWAP

            from numbox.utils.derive_wap import rewrap_derive

            SIG = types.float64(types.float64)

            _PRESENT = os.environ.get("SYMBOL_PRESENT", "1") == "1"


            def _{name}(x):
                return {expr}


            if _PRESENT:
                _j = njit(SIG, cache=True)(_{name})
                FOREIGN = CompileResultWAP(_j.get_compile_result(_j.nopython_signatures[0]))
            else:
                FOREIGN = None
            HANDLE = rewrap_derive(FOREIGN)
            ALIAS = getattr(HANDLE, "jit_alias", None)
            CFUNC_ALIAS = None
            ''')
    raise ValueError(door)


STABLE_SRC = textwrap.dedent('''
    """Untouched control binding, in its own file."""
    from numba import types
    from numbox.core.proxy.proxy import proxy

    SIG = types.float64(types.float64)





    @proxy(SIG, jit_options={"cache": True})
    def offset(x):
        return x + 100.0


    STABLE_HANDLE = offset.as_func if hasattr(offset, "as_func") else None
    ''')

CALLER_SRC = textwrap.dedent('''
    """Cached callers in a file the edit never touches."""
    from numba import njit
    from numba.core.types import float64

    from bmod import HANDLE
    from smod import STABLE_HANDLE, offset


    @njit(float64(float64), cache=True)
    def const_scale(x):
        return HANDLE(x) + 1.0


    @njit(float64(float64), cache=True)
    def const_stable(x):
        return STABLE_HANDLE(x) + 3.0


    @njit(float64(float64), cache=True)
    def disp_stable(x):
        return offset(x) + 4.0


    @njit(float64(float64), cache=True)
    def plain(x):
        return x + 7.0
    ''')

PROBE_SRC = textwrap.dedent('''
    import json
    import warnings

    out = {}
    print("PROBE_START", flush=True)
    import bmod
    out["alias"] = bmod.ALIAS
    out["cfunc_alias"] = bmod.CFUNC_ALIAS
    _cres = getattr(bmod.HANDLE, "cres", None)
    out["mangled"] = _cres.fndesc.llvm_func_name if _cres is not None else None
    print("BMOD_IMPORTED", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            import cmod
            out["import"] = "ok"
        except BaseException as exc:
            out["import"] = f"{type(exc).__name__}"
            out["import_msg"] = str(exc)[:300]
            cmod = None
        print("CMOD_IMPORTED", out["import"], flush=True)
        res = {}
        stats = {}
        if cmod is not None:
            for name in ("const_scale", "const_stable", "disp_stable", "plain"):
                fn = getattr(cmod, name)
                try:
                    res[name] = repr(fn(5.0))
                except BaseException as exc:
                    res[name] = f"RAISED:{type(exc).__name__}"
                print("CALLED", name, res[name], flush=True)
            for name in ("const_scale", "const_stable", "disp_stable", "plain"):
                fn = getattr(cmod, name)
                h = sum(fn.stats.cache_hits.values())
                m = sum(fn.stats.cache_misses.values())
                stats[name] = "served" if h and not m else "compiled" if m and not h else f"{h}/{m}"
        out["results"] = res
        out["stats"] = stats
    out["warnings"] = [f"{w.category.__name__}: {str(w.message)[:250]}".replace("\\n", " ") for w in caught]
    print("JSON " + json.dumps(out), flush=True)
    print("DONE ok", flush=True)
    ''')


def write_scenario(d, door, expr, absent_capable, csym=False):
    d.mkdir(parents=True, exist_ok=True)
    bpath = d / "bmod.py"
    before = bpath.stat() if bpath.exists() else None
    bpath.write_text(body_source(door, "scale", expr, absent_capable, csym), encoding="utf-8")
    if before is not None:
        os.utime(bpath, (before.st_mtime + 10,) * 2)
        after = bpath.stat()
        if (after.st_mtime, after.st_size) == (before.st_mtime, before.st_size):
            raise RuntimeError("edit invisible to numba")
    for name, src in (("smod.py", STABLE_SRC), ("cmod.py", CALLER_SRC), ("probe.py", PROBE_SRC)):
        path = d / name
        if not path.exists() or path.read_text(encoding="utf-8") != src:
            path.write_text(src, encoding="utf-8")


def run(d, venv, present=True, strict=False, extra_env=None):
    env = dict(os.environ)
    for hostile in ("PYTHONWARNINGS", "PYTHONDEVMODE", "NUMBA_DISABLE_JIT", "NUMBA_OPT",
                    "NUMBOX_PROXY_CACHE_STRICT", "PYTHONPATH"):
        env.pop(hostile, None)
    env["NUMBA_CACHE_DIR"] = str(Path(d) / "nbcache")
    env["PYTHONPATH"] = str(d)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["SYMBOL_PRESENT"] = "1" if present else "0"
    if strict:
        env["NUMBOX_PROXY_CACHE_STRICT"] = "1"
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([f"{venv}/bin/python", str(Path(d) / "probe.py")], capture_output=True, text=True,
                       env=env, timeout=900, errors="replace")
    out = {"rc": r.returncode, "stdout_tail": r.stdout[-300:], "stderr_tail": r.stderr[-500:]}
    for line in r.stdout.splitlines():
        if line.startswith("JSON "):
            out.update(json.loads(line[5:]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True)
    p.add_argument("--door", required=True)
    p.add_argument("--hazard", required=True, choices=["stale", "absent"])
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--csym", action="store_true")
    a = p.parse_args()
    venv = VENVS[a.tree]
    tag = f"{a.tree}-{a.door}-{a.hazard}" + ("-csym" if a.csym else "")
    d = Path("/tmp/cov/runs") / tag
    if a.fresh and d.exists():
        shutil.rmtree(d)
    absent_capable = a.hazard == "absent"
    steps = []
    expr0 = "_cov(x)" if a.csym else "x * 2.0"
    expr1 = "_cov(x) + 5.0" if a.csym else "x * 3.0"
    write_scenario(d, a.door, expr0, absent_capable, a.csym)
    steps.append(("cold_present", run(d, venv, present=True)))
    if a.hazard == "stale":
        steps.append(("warm_present", run(d, venv, present=True)))
        write_scenario(d, a.door, expr1, absent_capable, a.csym)
        steps.append(("warm_after_edit", run(d, venv, present=True)))
        steps.append(("warm_after_edit_2", run(d, venv, present=True)))
    else:
        steps.append(("warm_present", run(d, venv, present=True)))
        steps.append(("warm_absent_strict", run(d, venv, present=False, strict=True)))
        steps.append(("warm_absent", run(d, venv, present=False)))
        d2 = Path(str(d) + "-coldabsent")
        if d2.exists():
            shutil.rmtree(d2)
        write_scenario(d2, a.door, expr0, absent_capable, a.csym)
        steps.append(("cold_absent", run(d2, venv, present=False)))
    blob = {"tree": a.tree, "door": a.door, "hazard": a.hazard,
            "steps": [{"step": s, **v} for s, v in steps]}
    Path(f"/tmp/cov/runs/{tag}.json").write_text(json.dumps(blob, indent=1))
    print(f"### {tag}")
    for s, v in steps:
        warn = "; ".join(w.split(":")[0] for w in v.get("warnings", []))
        print(f"  {s:18s} rc={v['rc']:<4} import={v.get('import')!s:14s} alias={v.get('alias')} "
              f"res={v.get('results')} stats={v.get('stats')} warn=[{warn}] msg={str(v.get('import_msg', ''))[:110]}")
        if v["rc"] != 0:
            print(f"     stdout_tail={v['stdout_tail']!r}")
            print(f"     stderr_tail={v['stderr_tail']!r}")
        for w in v.get("warnings", []):
            print(f"     WARN {w}")


if __name__ == "__main__":
    main()
