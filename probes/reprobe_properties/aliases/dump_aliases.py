"""Dump every shipped binding's process-stable alias values.

Walks the numbox.core.bindings package, imports every module, and records for each
module attribute that carries a @proxy alias both the cfunc alias and, where the
tree has one, the derive's callconv alias.
"""
import importlib
import json
import pkgutil
import sys

import numbox.core.bindings as pkg

mods = []
for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    mods.append(info.name)

rows = {}
failed = {}
for name in sorted(mods):
    try:
        m = importlib.import_module(name)
    except Exception as exc:
        failed[name] = type(exc).__name__ + ": " + str(exc)[:120]
        continue
    for attr in sorted(vars(m)):
        obj = getattr(m, attr)
        cf = getattr(obj, "_numbox_proxy_alias", None)
        if cf is None:
            continue
        af = getattr(obj, "as_func", None)
        rows[name + "." + attr] = {
            "cfunc": cf,
            "cc": getattr(af, "jit_alias", "NOATTR") if af is not None else "NOASFUNC",
            "as_func_type": type(af).__name__ if af is not None else "NOASFUNC",
        }

out = {"count": len(rows), "rows": rows, "failed": failed}
json.dump(out, open(sys.argv[1], "w"), indent=0, sort_keys=True)
print("sites", len(rows), "modules", len(mods), "failed", len(failed))
