import os
import sys
import warnings
warnings.simplefilter("always")
from caller_order import caller_o2
from derives_order import o2

r = caller_o2(0.0)
st = caller_o2.stats
print("ORDER=%s alias=%s RESULT=%s want=24.0 hits=%s misses=%s" % (
    os.environ.get("ORDER", "12"), getattr(o2, "jit_alias", "<none>"), r, st.cache_hits, st.cache_misses))
