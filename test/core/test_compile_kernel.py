import pytest
from numba import njit
from numba.core.dispatcher import Dispatcher
from numbox.core.variable.compile_kernel import (
    _sanitize, _assign_identifiers, _wrap_formula, _generate_body
)
from numbox.core.variable.variable import Variable, Graph


def test_sanitize_basic():
    assert _sanitize("variables.a") == "variables_a"
    assert _sanitize("first-name") == "first_name"
    assert _sanitize("3m") == "v_3m"
    assert _sanitize("a..b") == "a_b"
    assert _sanitize("") == "v_"


def test_assign_identifiers_unique_and_valid():
    v1 = Variable(name="c", source="a_b")     # qual a_b.c -> base a_b_c
    v2 = Variable(name="b_c", source="a")     # qual a.b_c -> base a_b_c (collision)
    idents = _assign_identifiers([v1, v2])
    assert idents[v1] != idents[v2]
    assert all(s.isidentifier() for s in idents.values())


def test_assign_identifiers_formula_prefix_collision():
    node = Variable(name="x", source="variables")        # base variables_x
    clash = Variable(name="variables_x", source="f")     # base f_variables_x == f_<node temp>
    idents = _assign_identifiers([node, clash])
    temps = set(idents.values())
    fgs = {"f_" + t for t in temps}
    assert temps.isdisjoint(fgs)                          # no temp equals any formula global


def test_assign_identifiers_deterministic():
    v1 = Variable(name="c", source="a_b")
    v2 = Variable(name="b_c", source="a")
    assert _assign_identifiers([v1, v2]) == _assign_identifiers([v1, v2])


def test_assign_identifiers_invalid_char_and_leading_digit():
    v1 = Variable(name="first-name", source="ext")   # invalid char
    v2 = Variable(name="3m", source="ext")           # leading digit
    idents = _assign_identifiers([v1, v2])
    assert all(s.isidentifier() for s in idents.values())
    assert idents[v1] != idents[v2]


def _diamond_graph():
    return Graph(
        variables_lists={"variables": [
            {"name": "x", "inputs": {"y": "basket"}, "formula": njit(lambda y: 2 * y)},
            {"name": "a", "inputs": {"x": "variables"}, "formula": njit(lambda x: x - 74)},
            {"name": "b", "inputs": {"x": "variables"}, "formula": njit(lambda x: x + 0.5)},
            {"name": "u", "inputs": {"a": "variables", "b": "variables"},
             "formula": njit(lambda a, b: a + b)},
        ]},
        external_source_names=["basket"],
    )


def test_wrap_formula_passthrough_and_wrap():
    d = njit(lambda x: x)
    assert _wrap_formula(d) is d

    def plain(x):
        return x + 1
    assert isinstance(_wrap_formula(plain), Dispatcher)


def test_generate_body_shape():
    g = _diamond_graph()
    compiled = g.compile(["variables.u", "variables.a"])
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    source, bindings, params, outputs = _generate_body(compiled, ["variables.u", "variables.a"], idents)
    y_var = next(v for v in idents if v.qual_name() == "basket.y")
    assert params == [("basket", "y", idents[y_var])]
    assert outputs == ["variables.u", "variables.a"]
    assert source.startswith("def _kernel(")
    assert source.rstrip().endswith(",)")
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    expected = {"f_" + idents[n.variable] for n in compiled.ordered_nodes if n.variable not in external}
    assert set(bindings) == expected


def test_generate_body_errors():
    g = _diamond_graph()
    compiled = g.compile(["variables.u"])
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    with pytest.raises(ValueError):
        _generate_body(compiled, [], idents)
    with pytest.raises(ValueError):
        _generate_body(compiled, ["variables.nope"], idents)

    gph = Graph(
        variables_lists={"variables": [
            {"name": "x", "inputs": {"y": "basket"}, "formula": njit(lambda y: 2 * y)},
            {"name": "broken", "inputs": {"x": "variables"}, "formula": None},
        ]},
        external_source_names=["basket"],
    )
    c2 = gph.compile(["variables.broken"])
    id2 = _assign_identifiers([n.variable for n in c2.ordered_nodes])
    with pytest.raises(ValueError):
        _generate_body(c2, ["variables.broken"], id2)


def test_generate_body_external_as_only_output():
    g = _diamond_graph()
    compiled = g.compile(["basket.y"])
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    source, bindings, params, outputs = _generate_body(compiled, ["basket.y"], idents)
    assert bindings == {}
    assert outputs == ["basket.y"]
    assert "pass" not in source
    y_ident = params[0][2]
    assert source == f"def _kernel({y_ident}):\n    return ({y_ident},)\n"


def test_safe_getsource_named_function_and_cres():
    from numbox.core.variable.compile_kernel import _safe_getsource
    from numbox.utils.highlevel import cres
    from numba.core.types import float64

    @njit
    def named(x):
        return x + 41
    src = _safe_getsource(named)
    assert "return x + 41" in src

    wap = cres(float64(float64))(lambda x: x * 2.0)
    s = _safe_getsource(wap)             # must not raise
    assert isinstance(s, str) and s      # non-empty (repr fallback is acceptable)
