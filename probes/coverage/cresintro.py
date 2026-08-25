"""What is reachable from a numba CompileResult, cold vs restored from numba's own cache?

The prototype's one deliberate behaviour regression (a rewrap_derive-upgraded foreign wrapper is
no longer cacheable) rests on the claim that no Python function is reachable from a cache-restored
compile result, so no content-addressed alias can be minted for it.
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

VENV = "/tmp/cov/venv_proto"

BODY = textwrap.dedent('''
    from numba import njit, types

    SIG = types.float64(types.float64)


    @njit(SIG, cache=True)
    def f(x):
        return x * 2.0
    ''')

PROBE = textwrap.dedent('''
    import json

    import bodymod
    from numba.core.types.function_type import CompileResultWAP

    d = bodymod.f
    cres = d.get_compile_result(d.nopython_signatures[0])
    wap = CompileResultWAP(cres)
    hits = sum(d.stats.cache_hits.values())
    misses = sum(d.stats.cache_misses.values())
    out = {
        "state": "served" if hits and not misses else "compiled",
        "cres_attrs": sorted(a for a in dir(cres) if not a.startswith("__")),
        "func_id": repr(getattr(cres, "func_id", "MISSING"))[:120],
        "func_id_func": repr(getattr(getattr(cres, "func_id", None), "func", "MISSING"))[:120],
        "type_annotation_type": type(getattr(cres, "type_annotation", None)).__name__,
        "fndesc_qualname": getattr(cres.fndesc, "qualname", None),
        "fndesc_lookup_module": repr(getattr(cres.fndesc, "lookup_module", lambda: "MISSING")())[:120],
        "fndesc_lookup_globals_has_f": "f" in (getattr(cres.fndesc, "lookup_globals", dict)() or {}),
        "wap_has_dispatcher_backref": [a for a in dir(wap) if "func" in a.lower()],
    }
    try:
        out["serialize_using_object_code"] = str(type(cres.library.serialize_using_object_code()))
    except Exception as exc:
        out["serialize_using_object_code"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    mod = getattr(cres.fndesc, "lookup_module", lambda: None)()
    fn = getattr(mod, cres.fndesc.qualname.split(".")[-1], None) if mod is not None else None
    out["recovered_via_module"] = repr(fn)[:120]
    out["recovered_py_func"] = repr(getattr(fn, "py_func", "no py_func"))[:120]
    print("JSON " + json.dumps(out), flush=True)
    ''')

d = Path("/tmp/cov/runs/cresintro")
if d.exists():
    shutil.rmtree(d)
d.mkdir(parents=True)
(d / "bodymod.py").write_text(BODY, encoding="utf-8")
(d / "probe.py").write_text(PROBE, encoding="utf-8")
env = dict(os.environ)
for hostile in ("PYTHONWARNINGS", "PYTHONDEVMODE", "NUMBA_DISABLE_JIT", "NUMBA_OPT",
                "NUMBOX_PROXY_CACHE_STRICT", "PYTHONPATH"):
    env.pop(hostile, None)
env.update({"NUMBA_CACHE_DIR": str(d / "nbcache"), "PYTHONPATH": str(d),
            "PYTHONDONTWRITEBYTECODE": "1"})
for label in ("cold", "warm"):
    r = subprocess.run([f"{VENV}/bin/python", str(d / "probe.py")], capture_output=True, text=True,
                       env=env, timeout=900)
    print(f"--- {label} rc={r.returncode}")
    for line in r.stdout.splitlines():
        if line.startswith("JSON "):
            for k, v in json.loads(line[5:]).items():
                print(f"  {k}: {v}")
    if r.returncode:
        print(r.stderr[-500:])
