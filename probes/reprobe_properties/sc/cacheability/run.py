import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

out = {}
out["tree"] = os.environ.get("PROBE_TREE", "?")
out["phase"] = os.environ.get("PROBE_PHASE", "?")
res = {}


def attempt(label, fn, *args):
    try:
        res[label] = fn(*args)
    except Exception as exc:
        res[label] = type(exc).__name__ + ": " + str(exc)[:160]


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import bodies
    import callers
    attempt("const_proxy", callers.const_proxy, 3.0)
    attempt("const_cres", callers.const_cres, 3.0)
    attempt("const_rewrap", callers.const_rewrap, 3.0)
    attempt("arg_proxy", callers.arg_any, 3.0, bodies.PROXY_AS)
    attempt("arg_cres", callers.arg_any, 3.0, bodies.CRES_AS)
    attempt("arg_rewrap", callers.arg_any, 3.0, bodies.REWRAP_AS)

out["results"] = res
stats = {}
for name in ("const_proxy", "const_cres", "const_rewrap", "arg_any"):
    d = getattr(callers, name)
    s = d.stats
    stats[name] = {"hits": sum(s.cache_hits.values()), "misses": sum(s.cache_misses.values())}
out["stats"] = stats
out["warnings"] = sorted(set(w.category.__name__ + ": " + str(w.message)[:200] for w in caught))
alias = {}
alias["proxy_as"] = getattr(bodies.PROXY_AS, "jit_alias", "NOATTR")
alias["cres_as"] = getattr(bodies.CRES_AS, "jit_alias", "NOATTR")
alias["rewrap_as"] = getattr(bodies.REWRAP_AS, "jit_alias", "NOATTR")
alias["proxy_as_type"] = type(bodies.PROXY_AS).__name__
alias["cres_as_type"] = type(bodies.CRES_AS).__name__
alias["rewrap_as_type"] = type(bodies.REWRAP_AS).__name__
out["alias"] = alias
print("PROBEJSON " + json.dumps(out))
