"""
COR-2b: CompiledKernel.recompute shares COR-2's silent interior-override drop.

Same scenario as the CompiledGraph repro, on the compile_kernel path:
  A = external ext.a ; B = vars_.b = 10*A ; C = vars_.c = B + 1

- Isolated override  recompute({"vars_": {"b": 999}})           -> b=999, c=1000  (HONORED)
- Combined           recompute({"ext": {"a": 2}, "vars_": {"b": 999}}) -> b=20, c=21 (DROPPED)

Not a computation bug (deterministic, self-consistent b=10*a); the defect is the *silent* drop of an
explicit override when the node is downstream of another co-changed input. Root cause is the shared
CompiledGraph._collect_affected(changed_vars) reused at compile_kernel.py:711. The recompute docstring
additionally claims (unconditionally) that the overridden node's formula is "not re-run" -- false here.

Run:  venv/bin/python <thisfile>   |   venv/bin/pytest <thisfile>
"""
from inspect import getsource
from textwrap import dedent

from numbox.core.variable.variable import Graph
from numbox.core.variable.compile_kernel import compile_kernel


def _spec(name, inputs, formula):
    return {"name": name, "inputs": inputs, "formula": formula,
            "metadata": dedent(getsource(formula)) if formula else None}


def _build():
    def derive_b(a_):
        return 10 * a_

    def derive_c(b_):
        return b_ + 1

    g = Graph(
        variables_lists={"vars_": [
            _spec("b", {"a": "ext"}, derive_b),
            _spec("c", {"b": "vars_"}, derive_c),
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["vars_.b", "vars_.c"])
    return ck, g.registry["vars_"]["b"], g.registry["vars_"]["c"]


def test_compiled_kernel_recompute_honors_interior_override():
    ck, b, c = _build()
    ck.execute({"ext": {"a": 1}})
    ck.recompute({"ext": {"a": 2}, "vars_": {"b": 999}})
    assert ck._store[b] == 999, f"override discarded: b={ck._store[b]} (recomputed from a=2) instead of 999"
    assert ck._store[c] == 1000, f"downstream used the wrong b: c={ck._store[c]}"


if __name__ == "__main__":
    ck, b, c = _build()
    ck.execute({"ext": {"a": 1}})
    ck.recompute({"vars_": {"b": 999}})
    print(f"isolated override (a unchanged):        b={ck._store[b]} c={ck._store[c]}  (honored if 999/1000)")

    ck2, b2, c2 = _build()
    ck2.execute({"ext": {"a": 1}})
    ck2.recompute({"ext": {"a": 2}, "vars_": {"b": 999}})
    gb, gc = ck2._store[b2], ck2._store[c2]
    print(f"combined (override b + change upstream a): b={gb} c={gc}  (contract/assumed: 999/1000)")
    if gb == 999:
        print("RESULT: override honored in the combined case (no bug)")
    else:
        print(f"RESULT: COR-2b reproduced -- explicit override 999 silently dropped, b recomputed from a -> {gb}")
        raise SystemExit(1)
