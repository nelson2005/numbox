"""Direct cost of the per-decoration work the fix adds: a second _body_fingerprint plus ll.add_symbol.

Runs on the fix tree only (base has no _stable_jit_alias). Every proxied body in the named module is
timed; the total is what one import of that module now pays extra.
"""
import importlib
import json
import statistics
import sys
import time

from llvmlite import binding as ll

from numbox.utils.derive_wap import _stable_jit_alias
from numbox.utils.fingerprint import _body_fingerprint

REPS = 200


def timeit(fn, *a):
    fn(*a)
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - t0)
    return min(ts), statistics.median(ts)


rows = []
for target in sys.argv[1:]:
    mod = importlib.import_module(target)
    bodies = []
    for name, v in vars(mod).items():
        cres_wap = getattr(v, "as_func", None)
        if cres_wap is None or getattr(v, "_numbox_proxy_alias", None) is None:
            continue
        py = getattr(cres_wap, "jit_alias", None)
        bodies.append((name, v, cres_wap))
    tot_fp = tot_alias = tot_sym = 0.0
    n_alias = 0
    for name, disp, wap in bodies:
        py_func = disp.py_func if hasattr(disp, "py_func") else None
        if py_func is None:
            continue
        sig = wap.cres.signature
        fp_min, _ = timeit(_body_fingerprint, py_func)
        al_min, _ = timeit(_stable_jit_alias, py_func, sig, {})
        alias = _stable_jit_alias(py_func, sig, {})
        sy_min, _ = timeit(ll.add_symbol, alias + "_probe", wap.jit_address)
        tot_fp += fp_min
        tot_alias += al_min
        tot_sym += sy_min
        n_alias += 1
    rows.append({
        "module": target,
        "n_bindings": len(bodies),
        "n_timed": n_alias,
        "sum_body_fingerprint_s": tot_fp,
        "sum_stable_jit_alias_s": tot_alias,
        "sum_add_symbol_s": tot_sym,
        "sum_added_s": tot_alias + tot_sym,
        "per_binding_added_us": (tot_alias + tot_sym) / max(1, n_alias) * 1e6,
    })
print(json.dumps(rows, indent=1))
