"""Enumerate every shipped @proxy binding's cfunc alias (and, where present, its new callconv alias)."""
import importlib
import json
import os
import pkgutil
import sys

import numbox.core.bindings as B

mods = []
for m in pkgutil.walk_packages(B.__path__, B.__name__ + "."):
    mods.append(m.name)

out = {}
jit_aliases = {}
for name in sorted(mods):
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        out[f"!{name}"] = repr(exc)
        continue
    for attr, v in sorted(vars(mod).items()):
        alias = getattr(v, "_numbox_proxy_alias", None)
        if alias is None:
            continue
        out[f"{name}.{attr}"] = alias
        wap = getattr(v, "as_func", None)
        ja = getattr(wap, "jit_alias", None)
        if ja:
            jit_aliases[f"{name}.{attr}"] = ja

print(json.dumps({
    "tree": os.environ.get("BENCH_TREE", "?"),
    "n_bindings": len(out),
    "n_jit_aliases": len(jit_aliases),
    "cfunc_aliases": out,
    "jit_aliases": jit_aliases,
}, indent=1, sort_keys=True))
assert sys.argv is not None
