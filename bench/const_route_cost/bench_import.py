"""Per-module import wall time for numbox's shipped binding modules, plus a count of the bindings each defines."""
import importlib
import json
import os
import sys
import time

MODULES = [
    "numbox.core.proxy.proxy",
    "numbox.core.bindings.libm",
    "numbox.core.bindings.libc",
    "numbox.core.bindings.stdio",
    "numbox.core.bindings.errno",
    "numbox.core.bindings.strerror",
    "numbox.core.bindings.sqlite.conn",
    "numbox.core.bindings.sqlite.stmt",
    "numbox.core.bindings.sqlite.value",
    "numbox.core.bindings.sqlite.column",
    "numbox.core.bindings.sqlite.result",
    "numbox.core.bindings.sqlite.bind",
    "numbox.core.bindings.sqlite.blob",
    "numbox.core.bindings.sqlite.exec",
    "numbox.core.bindings.sqlite.hooks",
    "numbox.core.bindings.sqlite.udf",
    "numbox.core.bindings.sqlite.vtable",
    "numbox.core.work.work",
]

target = sys.argv[1]
t0 = time.perf_counter()
mod = importlib.import_module(target)
elapsed = time.perf_counter() - t0
n_bindings = sum(1 for v in vars(mod).values() if getattr(v, "_numbox_proxy_alias", None))
n_asfunc = sum(1 for v in vars(mod).values() if hasattr(v, "as_func"))
print(json.dumps({
    "tree": os.environ.get("BENCH_TREE", "?"),
    "module": target,
    "import_s": elapsed,
    "n_proxy_bindings": n_bindings,
    "n_with_as_func": n_asfunc,
    "n_modules_loaded": len(sys.modules),
}))
assert MODULES
