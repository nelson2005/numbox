import os
import subprocess
import sys
import textwrap

import pytest
from numba import njit
from numba.core.dispatcher import Dispatcher
from numbox.core.variable.compile_kernel import (
    _sanitize, _assign_identifiers, _wrap_formula, _generate_body, _compile,
    compile_kernel, CompiledKernel,
)
from numbox.core.variable.variable import Variable, Graph, Values


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


def test_compile_runs():
    src = "def _kernel(y):\n    x = f_x(y)\n    return (x,)\n"
    bindings = {"f_x": njit(lambda y: 2 * y)}
    kernel = _compile(src, bindings, None, True)
    assert kernel(10) == (20,)


def test_compile_anchor_is_content_addressed(tmp_path, monkeypatch):
    import numbox.utils.preprocessing as pp
    monkeypatch.setattr(pp, "_anchor_root", lambda subdir: tmp_path)
    src = "def _kernel(y):\n    x = f_x(y)\n    return (x,)\n"
    _compile(src, {"f_x": njit(lambda y: 2 * y)}, None, True)
    before = set(tmp_path.glob("_kernel_*.py"))
    assert before, "first _compile must create an anchor"
    _compile(src, {"f_x": njit(lambda y: 3 * y)}, None, True)
    after = set(tmp_path.glob("_kernel_*.py"))
    assert after - before, "different formula must produce a new anchor"


def test_compile_cache_survives_fresh_process(tmp_path):
    script = textwrap.dedent('''
        from numba import njit
        from numbox.core.variable.compile_kernel import _compile
        src = "def _kernel(y):\\n    x = f_x(y)\\n    return (x,)\\n"
        k = _compile(src, {"f_x": njit(lambda y: 2 * y)}, None, True)
        print("RESULT", k(10)[0])
    ''')
    f = tmp_path / "ck_warm.py"
    f.write_text(script)
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "nbcache")}
    for _ in range(2):
        p = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, env=env)
        assert p.returncode == 0, p.stdout + p.stderr
        assert "RESULT 20" in p.stdout, p.stdout + p.stderr


def _pure(graph, required, external_values):
    compiled = graph.compile(required)
    values = Values()
    compiled.execute(external_values, values)
    by_qual = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    return {q: values.get(by_qual[q]).value for q in required}


def test_compile_kernel_matches_pure_python_diamond():
    g = _diamond_graph()
    req = ["variables.u", "variables.a"]
    ck = compile_kernel(g, req)
    assert isinstance(ck, CompiledKernel)
    ext = {"basket": {"y": 100}}
    assert ck.execute(ext) == _pure(g, req, ext)
    assert ck.params == ["basket.y"]
    assert ck.outputs == req
    assert tuple(ck.kernel(100)) == tuple(_pure(g, req, ext)[q] for q in req)


def test_compile_kernel_single_output_and_str_required():
    g = _diamond_graph()
    ck = compile_kernel(g, "variables.u")
    assert ck.outputs == ["variables.u"]
    assert ck.execute({"basket": {"y": 100}}) == {"variables.u": 326.5}


def test_compile_kernel_auto_specialization():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    assert ck.execute({"basket": {"y": 100}})["variables.u"] == 326.5
    assert ck.execute({"basket": {"y": 100.0}})["variables.u"] == 326.5


def test_compile_kernel_missing_external_raises():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    with pytest.raises(KeyError) as exc:
        ck.execute({"basket": {}})
    assert "basket.y" in str(exc.value)


def test_compile_kernel_missing_external_source_raises():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    with pytest.raises(KeyError) as exc:
        ck.execute({})                      # entire 'basket' source absent
    assert "basket.y" in str(exc.value)
