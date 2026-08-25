"""Two factory-made proxied bodies over different C symbols: which body does each caller run?

Before the fix a const caller declared and linked in *its own* compile result, so a colliding alias could
not misroute it. After the fix a const caller resolves the shared alias, which ``_publish_jit_alias``
binds to whichever body registered first.
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

VENVS = {"proto": "/tmp/cov/venv_proto", "branch": "/tmp/cov/venv_branch", "main": "/tmp/cov/venv_main"}

BODY = textwrap.dedent('''
    """Two proxied bindings made by one factory over two different C symbols."""
    import llvmlite.binding as ll
    from numba import types
    from numba.core.types import ExternalFunction
    from numbox.core.proxy.proxy import proxy

    ll.load_library_permanently("/tmp/cov/lib/libcovsym.so")
    SIG = types.float64(types.float64)


    def make(sym):
        ext = ExternalFunction(sym, SIG)

        def body(x):
            return ext(x)
        return body


    doubler = proxy(SIG, jit_options={"cache": True})(make("cov_scale"))
    tripler = proxy(SIG, jit_options={"cache": True})(make("cov_triple"))

    DOUBLER_AS_FUNC = doubler.as_func
    TRIPLER_AS_FUNC = tripler.as_func
    ''')

PROBE = textwrap.dedent('''
    import json

    from numba import njit
    from numba.core.types import float64

    import bodies


    @njit(float64(float64), cache=True)
    def const_doubler(x):
        return bodies.DOUBLER_AS_FUNC(x)


    @njit(float64(float64), cache=True)
    def const_tripler(x):
        return bodies.TRIPLER_AS_FUNC(x)


    @njit(float64(float64), cache=True)
    def disp_doubler(x):
        return bodies.doubler(x)


    @njit(float64(float64), cache=True)
    def disp_tripler(x):
        return bodies.tripler(x)


    print("JSON " + json.dumps({
        "cfunc_alias_doubler": bodies.doubler._numbox_proxy_alias,
        "cfunc_alias_tripler": bodies.tripler._numbox_proxy_alias,
        "cc_alias_doubler": getattr(bodies.DOUBLER_AS_FUNC, "jit_alias", None),
        "cc_alias_tripler": getattr(bodies.TRIPLER_AS_FUNC, "jit_alias", None),
        "const_doubler(5)": const_doubler(5.0),
        "const_tripler(5)": const_tripler(5.0),
        "disp_doubler(5)": disp_doubler(5.0),
        "disp_tripler(5)": disp_tripler(5.0),
        "py_doubler(5)": bodies.doubler(5.0),
        "py_tripler(5)": bodies.tripler(5.0),
    }), flush=True)
    ''')


def main():
    tree = sys.argv[1]
    d = Path(f"/tmp/cov/runs/collide-{tree}")
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "bodies.py").write_text(BODY, encoding="utf-8")
    (d / "probe.py").write_text(PROBE, encoding="utf-8")
    env = dict(os.environ)
    for hostile in ("PYTHONWARNINGS", "PYTHONDEVMODE", "NUMBA_DISABLE_JIT", "NUMBA_OPT",
                    "NUMBOX_PROXY_CACHE_STRICT", "PYTHONPATH"):
        env.pop(hostile, None)
    env.update({"NUMBA_CACHE_DIR": str(d / "nbcache"), "PYTHONPATH": str(d),
                "PYTHONDONTWRITEBYTECODE": "1"})
    r = subprocess.run([f"{VENVS[tree]}/bin/python", str(d / "probe.py")], capture_output=True,
                       text=True, env=env, timeout=900)
    print(f"### collide tree={tree} rc={r.returncode}")
    for line in r.stdout.splitlines():
        if line.startswith("JSON "):
            for k, v in json.loads(line[5:]).items():
                print(f"  {k:24s} {v}")
    if r.returncode != 0:
        print("  stderr:", r.stderr[-600:])


if __name__ == "__main__":
    main()
