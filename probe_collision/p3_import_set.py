"""Same collision, no order change at all: only the SET of imported bindings differs.

Two sibling modules each own one binding of a shared factory and one ``cache=True``
constant caller. Every module builds its binding at import, in one fixed order, which
is the discipline the fix's documentation offers as the way to stay clear of this.

IMPORT=b   caches ``call_b`` in a process where ``plus_one`` is the only binding.
IMPORT=ab  imports both, ``mod_a`` first, so ``hundred`` owns the alias.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

which = os.environ["IMPORT"]
out = []

if "a" in which:
    from pkg import mod_a
if "b" in which:
    from pkg import mod_b


def served(f):
    hits = sum(f.stats.cache_hits.values())
    misses = sum(f.stats.cache_misses.values())
    return "served" if hits and not misses else "compiled" if misses else f"{hits}/{misses}"


if "a" in which:
    out.append(f"call_a={mod_a.call_a(4.0)} (want 403.0) [{served(mod_a.call_a)}] "
               f"alias={'yes' if mod_a.hundred.jit_alias else 'NO'}")
if "b" in which:
    out.append(f"call_b={mod_b.call_b(4.0)} (want 8.0) [{served(mod_b.call_b)}] "
               f"alias={'yes' if mod_b.plus_one.jit_alias else 'NO'}")

print(f"IMPORT={which} " + "  ".join(out))
