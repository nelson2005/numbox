from numbox.core.variable.compile_kernel import _sanitize, _assign_identifiers
from numbox.core.variable.variable import Variable


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
