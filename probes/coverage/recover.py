"""Can a cache-restored compile result be tied back to its Python function, identically cold and warm?

If yes, the one deliberate behaviour regression (rewrap_derive-upgraded wrappers lowered uncacheable)
rests on a false premise.
"""
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

VENV = "/tmp/cov/venv_proto"

BODY = textwrap.dedent('''
    from numba import njit, types

    SIG = types.float64(types.float64)


    @njit(SIG, cache=True)
    def top_level(x):
        return x * 2.0


    def factory():
        @njit(SIG, cache=True)
        def nested(x):
            return x * 5.0
        return nested


    nested = factory()
    ''')

PROBE = textwrap.dedent('''
    import json

    import bodymod
    from numbox.utils.derive_wap import _stable_jit_alias

    out = {}
    for label in ("top_level", "nested"):
        d = getattr(bodymod, label)
        cres = d.get_compile_result(d.nopython_signatures[0])
        hits = sum(d.stats.cache_hits.values())
        misses = sum(d.stats.cache_misses.values())
        rec = {"state": "served" if hits and not misses else "compiled",
               "qualname": cres.fndesc.qualname}
        mod = cres.fndesc.lookup_module()
        obj = mod
        try:
            for part in cres.fndesc.qualname.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            obj = None
        rec["resolved"] = repr(obj)[:80]
        owns = obj is not None and getattr(obj, "overloads", {}).get(cres.signature, None) is cres
        rec["dispatcher_owns_this_cres"] = bool(owns)
        pf = getattr(obj, "py_func", None)
        if pf is not None:
            rec["alias_from_recovered_py_func"] = _stable_jit_alias(pf, cres.signature, {"cache": True})
        out[label] = rec
    print("JSON " + json.dumps(out), flush=True)
    ''')

d = Path("/tmp/cov/runs/recover")
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
        print(r.stderr[-600:])
