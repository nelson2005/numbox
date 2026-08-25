"""pytest plugin: count derive alias publishes, refusals and constant lowerings.

Independent of the repository's own instrumentation. Writes a JSON summary to
$AUDIT_OUT at session finish.
"""
import atexit
import json
import os

import numba.core.base as nbase
from numbox.utils import derive_wap as dw

RECORDS = []
CONST = []

_orig_publish = dw._publish_jit_alias
_orig_const = nbase.BaseContext.get_constant_generic


def _publish(cres, py_func, jit_options, jit_address):
    alias = None
    try:
        if py_func is not None and dw._cache_guard_installed():
            alias = dw._stable_jit_alias(py_func, cres.signature, jit_options)
    except Exception:
        alias = None
    prior = dw._JIT_ALIASES.get(alias) if alias is not None else None
    out = _orig_publish(cres, py_func, jit_options, jit_address)
    rec = {}
    rec["alias"] = alias
    rec["granted"] = out is not None
    rec["duplicate"] = prior is not None
    rec["same_cres"] = prior is cres
    rec["mod"] = getattr(cres.fndesc, "modname", "?")
    rec["qual"] = getattr(cres.fndesc, "qualname", "?")
    rec["llvm_name"] = getattr(cres.fndesc, "llvm_func_name", "?")
    rec["cres_id"] = id(cres)
    rec["addr"] = jit_address
    rec["py_func_id"] = id(py_func) if py_func is not None else None
    rec["code_id"] = id(py_func.__code__) if py_func is not None else None
    RECORDS.append(rec)
    return out


def _const(self, builder, ty, val):
    if isinstance(ty, dw.DeriveFunctionType):
        c = {}
        c["alias"] = getattr(val, "jit_alias", "NOATTR")
        fndesc = getattr(getattr(val, "cres", None), "fndesc", None)
        c["mod"] = getattr(fndesc, "modname", "?")
        c["qual"] = getattr(fndesc, "qualname", "?")
        c["cres_id"] = id(getattr(val, "cres", None))
        CONST.append(c)
    return _orig_const(self, builder, ty, val)


dw._publish_jit_alias = _publish
nbase.BaseContext.get_constant_generic = _const


def _dump():
    out = {}
    out["publish_attempts"] = len(RECORDS)
    out["granted"] = sum(1 for r in RECORDS if r["granted"])
    out["refused"] = sum(1 for r in RECORDS if not r["granted"])
    out["distinct_aliases"] = len(set(r["alias"] for r in RECORDS if r["alias"]))
    out["duplicate_attempts"] = sum(1 for r in RECORDS if r["duplicate"])
    out["duplicate_same_cres"] = sum(1 for r in RECORDS if r["duplicate"] and r["same_cres"])
    out["const_lowerings"] = len(CONST)
    out["const_without_alias"] = sum(1 for c in CONST if c["alias"] is None)
    out["records"] = RECORDS
    out["const"] = CONST
    path = os.environ.get("AUDIT_OUT", "/tmp/audit.json")
    with open(path, "w") as fh:
        json.dump(out, fh)


atexit.register(_dump)
