import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

out = {}
out["tree"] = os.environ.get("PROBE_TREE", "?")
out["present"] = os.environ.get("PRESENT", "0")
out["body"] = os.environ.get("BODY", "py")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    try:
        import vcaller
        out["result"] = vcaller.call_it(3.0)
        s = vcaller.call_it.stats
        out["hits"] = sum(s.cache_hits.values())
        out["misses"] = sum(s.cache_misses.values())
        out["as_type"] = type(vcaller.AS).__name__
        out["as_alias"] = getattr(vcaller.AS, "jit_alias", "NOATTR")
    except Exception as exc:
        out["result"] = type(exc).__name__ + ": " + str(exc)[:200].replace("\n", " | ")
out["warnings"] = sorted(set(w.category.__name__ + ": " + str(w.message)[:130] for w in caught))
print("PROBEJSON " + json.dumps(out))
