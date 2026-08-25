import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

out = {}
out["tree"] = os.environ.get("PROBE_TREE", "?")
out["phase"] = os.environ.get("PROBE_PHASE", "?")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import body_mod
    import ecaller
    out["const_proxy"] = ecaller.const_proxy(3.0)
    out["const_cres"] = ecaller.const_cres(3.0)
    out["dispatcher"] = body_mod.ebody(3.0)
for label, fn in (("const_proxy", ecaller.const_proxy), ("const_cres", ecaller.const_cres)):
    s = fn.stats
    out[label + "_hits"] = sum(s.cache_hits.values())
    out[label + "_misses"] = sum(s.cache_misses.values())
out["proxy_alias"] = getattr(body_mod.PROXY_AS, "jit_alias", "NOATTR")
out["cres_alias"] = getattr(body_mod.CRES_AS, "jit_alias", "NOATTR")
out["warnings"] = sorted(set(w.category.__name__ + ": " + str(w.message)[:150] for w in caught))
print("PROBEJSON " + json.dumps(out))
