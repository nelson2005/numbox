"""Write the proxied bindings and cres derives the cost benchmark calls."""
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

HEAD = '''
"""A proxied binding used by the cost benchmark."""
from numba import types
from numbox.core.proxy.proxy import proxy

SIG = types.float64(types.float64)


@proxy(SIG, jit_options={"cache": True})
def %(name)s(x):
%(body)s
'''

small_body = "    return x * 2.0 + 1.0"
(out / "binding_small.py").write_text(HEAD % {"name": "small", "body": small_body}, encoding="utf-8")

lines = ["    acc = x"]
for i in range(60):
    lines.append(f"    acc = acc * {1.0 + i / 1000.0:.6f} + x")
lines.append("    return acc")
big_body = "\n".join(lines)
(out / "binding_big.py").write_text(HEAD % {"name": "big", "body": big_body}, encoding="utf-8")

CRES_HEAD = '''
"""A cres-minted derive used by the cost benchmark."""
from numba import types
from numbox.utils.highlevel import cres

SIG = types.float64(types.float64)


@cres(SIG, cache=True)
def %(name)s(x):
%(body)s
'''
(out / "cres_small.py").write_text(CRES_HEAD % {"name": "csmall", "body": small_body}, encoding="utf-8")
print("wrote", sorted(p.name for p in out.iterdir()))
