"""Read the object numba cached for a const-route caller and report what it references.

Scenario: a proxied/cres body of N statements (N chosen to sit past LLVM's inline threshold),
one cache=True caller reaching it as a compile-time constant.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, "/tmp/cov/harness")
from elfread import symbols  # noqa: E402

VENVS = {"proto": "/tmp/cov/venv_proto", "main": "/tmp/cov/venv_main", "branch": "/tmp/cov/venv_branch"}

BODY = '''
"""A body sized to sit past (or under) LLVM's inline threshold."""
from numba import types
from numbox.core.proxy.proxy import proxy

SIG = types.float64(types.float64)


@proxy(SIG, jit_options={{"cache": True}})
def big(x):
    acc = x
{stmts}
    return acc


HANDLE = big.as_func
'''

CALLER = textwrap.dedent('''
    from numba import njit
    from numba.core.types import float64

    from bigmod import HANDLE


    @njit(float64(float64), cache=True)
    def const_big(x):
        return HANDLE(x) + 1.0
    ''')

PROBE = textwrap.dedent('''
    import json

    import bigmod
    from bigcaller import const_big

    print("JSON " + json.dumps({
        "alias": getattr(bigmod.HANDLE, "jit_alias", None),
        "mangled": bigmod.HANDLE.cres.fndesc.llvm_func_name,
        "result": const_big(2.0),
    }), flush=True)
    ''')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True)
    p.add_argument("--n", type=int, default=60)
    a = p.parse_args()
    d = Path(f"/tmp/cov/runs/obj-{a.tree}-n{a.n}")
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    stmts = "\n".join(f"    acc = acc * {i + 2}.0 + x" for i in range(a.n))
    (d / "bigmod.py").write_text(BODY.format(stmts=stmts), encoding="utf-8")
    (d / "bigcaller.py").write_text(CALLER, encoding="utf-8")
    (d / "probe.py").write_text(PROBE, encoding="utf-8")
    env = dict(os.environ)
    for hostile in ("PYTHONWARNINGS", "PYTHONDEVMODE", "NUMBA_DISABLE_JIT", "NUMBA_OPT",
                    "NUMBOX_PROXY_CACHE_STRICT", "PYTHONPATH"):
        env.pop(hostile, None)
    env.update({"NUMBA_CACHE_DIR": str(d / "nbcache"), "PYTHONPATH": str(d),
                "PYTHONDONTWRITEBYTECODE": "1"})
    r = subprocess.run([f"{VENVS[a.tree]}/bin/python", str(d / "probe.py")],
                       capture_output=True, text=True, env=env, timeout=900)
    info = {}
    for line in r.stdout.splitlines():
        if line.startswith("JSON "):
            info = json.loads(line[5:])
    files = sorted((d / "nbcache").rglob("bigcaller.const_big-*.nbc"))
    out = {"tree": a.tree, "n": a.n, "rc": r.returncode, **info,
           "nbc_files": [f.name for f in files], "stderr_tail": r.stderr[-300:]}
    for f in files:
        payload = f.read_bytes()
        out["nbc_bytes"] = len(payload)
        mangled = (info.get("mangled") or "zzz").encode()
        alias = (info.get("alias") or "").encode()
        out["mangled_in_payload"] = mangled in payload
        out["alias_in_payload"] = bool(alias) and alias in payload
        out["prefix_in_payload"] = b"numbox_pxy_" in payload
        idx = payload.find(b"\x7fELF")
        if idx >= 0:
            obj = payload[idx:]
            und, weak, glob = symbols(obj)
            out["undef_numbox"] = sorted(s for s in und if s.startswith("numbox_pxy_"))
            out["undef_mangled"] = sorted(s for s in und if info.get("mangled") and info["mangled"] in s)
            out["weak_mangled"] = sorted(s for s in weak if info.get("mangled") and info["mangled"] in s)
            out["n_undef"] = len(und)
            out["n_weak"] = len(weak)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
