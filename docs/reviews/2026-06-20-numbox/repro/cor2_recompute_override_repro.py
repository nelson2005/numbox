"""
Reproduction for COR-2: CompiledGraph.recompute discards an explicit override
for a computed (Variables-source) variable when it sits downstream of another
changed variable.

recompute's docstring promises to honor new values "coming from either External
or Variables source". This test passes new values for BOTH:
  - A : external input `ext.a`
  - B : computed node `vars_.b`, whose formula depends on A
and asserts the documented contract: B keeps its explicit override (999), and
the downstream node C follows from it.

On the current (buggy) code B is in `affected` (reached as a dependent of A),
so lines 400-401 null B's just-assigned override and `_calculate` recomputes
B from A's new value -> the override is silently dropped.

Run standalone:   venv/bin/python <thisfile>
Or under pytest:  venv/bin/pytest <thisfile> -v
"""
from inspect import getsource
from textwrap import dedent

from numbox.core.variable.variable import Graph, Values


def _spec(name, inputs, formula):
    return {
        "name": name,
        "inputs": inputs,
        "formula": formula,
        "metadata": dedent(getsource(formula)) if formula else None,
    }


def _build():
    def derive_b(a_):          # B = 10 * A
        return 10 * a_

    def derive_c(b_):          # C = B + 1   (downstream of B)
        return b_ + 1

    graph = Graph(
        variables_lists={
            "vars_": [
                _spec("b", {"a": "ext"}, derive_b),
                _spec("c", {"b": "vars_"}, derive_c),
            ],
        },
        external_source_names=["ext"],
    )
    compiled = graph.compile(["vars_.c"])
    b = graph.registry["vars_"]["b"]
    c = graph.registry["vars_"]["c"]
    return compiled, b, c


def test_recompute_honors_variables_override():
    compiled, b, c = _build()
    values = Values()

    # initial calc: a=1 -> b=10 -> c=11
    compiled.execute({"ext": {"a": 1}}, values)
    assert values.get(b).value == 10
    assert values.get(c).value == 11

    # recompute changing A=ext.a to 2 AND explicitly overriding computed B=vars_.b to 999.
    compiled.recompute({"ext": {"a": 2}, "vars_": {"b": 999}}, values)

    # Documented contract: the explicit override wins; downstream follows from it.
    assert values.get(b).value == 999, (
        f"COR-2 reproduced: b={values.get(b).value} (recomputed from a=2) "
        f"instead of the explicit override 999"
    )
    assert values.get(c).value == 1000, f"downstream c used the wrong b: c={values.get(c).value}"


if __name__ == "__main__":
    compiled, b, c = _build()
    values = Values()
    compiled.execute({"ext": {"a": 1}}, values)
    print(f"after execute(a=1):           b={values.get(b).value}  c={values.get(c).value}  (expect b=10 c=11)")

    compiled.recompute({"ext": {"a": 2}, "vars_": {"b": 999}}, values)
    got_b, got_c = values.get(b).value, values.get(c).value
    print(f"after recompute(a=2, b:=999): b={got_b}  c={got_c}")
    print(f"contract (docstring):         b=999  c=1000  (explicit Variables-source override honored)")
    if got_b == 999:
        print("RESULT: PASS — override honored (COR-2 is fixed)")
    else:
        print(f"RESULT: FAIL — COR-2 reproduced: override 999 discarded, b recomputed from a=2 -> {got_b}; "
              f"c followed the wrong b -> {got_c}")
        raise SystemExit(1)
