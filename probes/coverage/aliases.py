"""Enumerate every shipped @proxy binding's aliases, so the two trees can be diffed."""
import importlib
import json
import pkgutil

import numbox.core.bindings as B

mods = [m.name for m in pkgutil.walk_packages(B.__path__, B.__name__ + ".")]
out = {}
for name in sorted(mods):
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        out[f"!{name}"] = f"{type(exc).__name__}: {exc}"
        continue
    for attr in sorted(dir(mod)):
        obj = getattr(mod, attr, None)
        alias = getattr(obj, "_numbox_proxy_alias", None)
        if alias is None:
            continue
        as_func = getattr(obj, "as_func", None)
        out[f"{name}.{attr}"] = {"cfunc": alias, "cc": getattr(as_func, "jit_alias", None),
                                 "as_func_type": type(as_func).__name__ if as_func is not None else None}
print(json.dumps(out, indent=0, sort_keys=True))
