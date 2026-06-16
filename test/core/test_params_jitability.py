import pytest
from dataclasses import FrozenInstanceError
from numba import cfunc, float64, int64, vectorize
from numba import njit as _njit, int64 as _int64, float64 as _float64
from numba import njit as _njit_t3
from numbox.core.variable.variable import (
    Graph, Params, Variable, Variables, External,
)
from numbox.core.variable.compile_kernel import _classify, _validate_externals, compile_kernel
from numbox.core.variable._kernel_partition import _evaluate as _evaluate_fn
from numbox.core.variable.utils import _validate_declared_return, _wrap_formula_typed
from numbox.utils.highlevel import cres


def test_params_frozen_defaults():
    p = Params()
    assert p.jitable is True and p.type is None
    assert Params(type=float64).type is float64
    with pytest.raises(FrozenInstanceError):
        p.jitable = False  # frozen


def test_variable_params_roundtrip_and_identity_unchanged():
    a = Variable(name="a", source="m", params=Params(type=float64))
    assert a.params.type is float64
    bare = Variable(name="a", source="m")
    assert a == bare and hash(a) == hash(bare)  # params not part of identity
    assert {a, bare} == {a}  # dedup by (source, name)


def test_dict_params_rejected():
    with pytest.raises(TypeError, match="params must be a Params instance"):
        Variable(name="a", source="m", params={"jitable": True})


def test_varspec_params_passthrough():
    vs = Variables("m", [{"name": "a", "formula": lambda: 1.0, "params": Params(type=float64)}])
    assert vs["a"].params.type is float64


def test_external_declare_attaches_params():
    e = External("ext")
    e.declare("x", Params(type=int64))
    assert e["x"].params.type is int64


def _graph_all_jittable():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    return g


def test_classify_case_a_all_jittable():
    g = _graph_all_jittable()
    compiled = g.compile(["c.b"])
    case, dispositions, consumed = _classify(compiled)
    assert case == "A"
    assert all(d == "STATIC_JIT" for d in dispositions.values())


def test_classify_case_b_declared_python_mix():
    g = _graph_all_jittable()
    g.registry["c"].update("b", Variable(
        name="b", source="c", inputs={"a": "c"},
        formula=lambda a: a * 2.0, params=Params(jitable=False, type=float64)))
    g.registry["c"].update("d", Variable(
        name="d", source="c", inputs={"b": "c"},
        formula=lambda b: b + 1.0, params=Params(type=float64)))
    compiled = g.compile(["c.d"])
    case, dispositions, _ = _classify(compiled)
    assert case == "B"
    assert dispositions[g.registry["c"]["b"]] == "STATIC_PY"


def test_classify_case_c_untyped_python_boundary():
    g = _graph_all_jittable()
    g.registry["c"].update("b", Variable(
        name="b", source="c", inputs={"a": "c"},
        formula=lambda a: a * 2.0, params=Params(jitable=False)))  # type=None
    g.registry["c"].update("d", Variable(
        name="d", source="c", inputs={"b": "c"},
        formula=lambda b: b + 1.0, params=Params(type=float64)))
    compiled = g.compile(["c.d"])
    case, _, _ = _classify(compiled)
    assert case == "C"


def test_validate_externals_rejects_formula_bearing():
    g = Graph({"c": [{"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0}]}, ["e"])
    g.external["e"].update("x", Variable(name="x", source="e", formula=lambda: 1.0))
    compiled = g.compile(["c.a"])
    with pytest.raises(ValueError, match="external but carries a formula"):
        _validate_externals(compiled)


def test_njit_probe_reads_natural_return_type():
    f = _njit(lambda x: x * 1.5)
    f.compile((_int64,))
    rt = f.nopython_signatures[-1].return_type
    assert rt == _float64  # x*1.5 over int64 is float64, NOT int64


def test_validate_rejects_coercible_wrong_scalar_type():
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        _validate_declared_return(lambda x: x * 1.5, (int64,), int64, flags={})


def test_validate_accepts_correct_declaration():
    _validate_declared_return(lambda x: x * 1.5, (int64,), float64, flags={})  # no raise


def test_validate_rejects_nonconvertible_return():
    with pytest.raises(ValueError):
        _validate_declared_return(lambda x: "s", (int64,), int64, flags={})


def test_validate_rejects_dufunc_wrong_output():
    vf = vectorize(["int64(int64)", "float64(float64)"])(lambda a: a + a)
    with pytest.raises(ValueError):
        _validate_declared_return(vf, (int64,), float64, flags={})  # int+int stays int64


def test_validate_accepts_cfunc_correct_declaration():
    cf = cfunc(int64(int64))(lambda x: x + 1)
    _validate_declared_return(cf, (int64,), int64, flags={})  # no raise


def test_validate_rejects_cfunc_wrong_declaration():
    cf = cfunc(int64(int64))(lambda x: x + 1)
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        _validate_declared_return(cf, (int64,), float64, flags={})


def test_validate_accepts_cres_correct_declaration():
    cf = cres(float64(float64))(lambda x: x + 1.0)
    _validate_declared_return(cf, (float64,), float64, flags={})  # no raise


def test_validate_rejects_cres_wrong_declaration():
    cf = cres(float64(float64))(lambda x: x + 1.0)
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        _validate_declared_return(cf, (float64,), int64, flags={})


def test_validate_accepts_dispatcher_correct_declaration():
    d = _njit(lambda x: x * 1.5)
    _validate_declared_return(d, (int64,), float64, flags={})  # no raise; natural float64


def test_validate_rejects_dispatcher_wrong_declaration():
    d = _njit(lambda x: x * 1.5)
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        _validate_declared_return(d, (int64,), int64, flags={})  # natural float64 != int64


def test_wrap_formula_typed_is_uncached():
    d = _wrap_formula_typed(lambda x: x + 1.0, float64(float64), flags={})
    assert d.targetoptions.get("cache") in (None, False)


def test_wrap_formula_typed_strips_cache_flag():
    d = _wrap_formula_typed(lambda x: x + 1.0, float64(float64), flags={"cache": True})
    assert d.targetoptions.get("cache") in (None, False)


def test_wrap_formula_typed_passes_exotics_through():
    cf = cfunc(int64(int64))(lambda x: x + 1)
    assert _wrap_formula_typed(cf, int64(int64), flags={}) is cf


def test_evaluate_honors_fixed_demotion_set():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0},
    ]}, ["e"])
    compiled = g.compile(["c.b"])
    a = g.registry["c"]["a"]
    b = g.registry["c"]["b"]
    ext = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    x = next(iter(ext))
    bindings = {a: _njit_t3(a.formula), b: _njit_t3(b.formula)}
    values = {x: 3.0}
    demoted = {b}  # force b to run as plain python
    _evaluate_fn(compiled.ordered_nodes, ext, values, bindings, {}, demoted)
    assert values[a] == 4.0 and values[b] == 8.0


def test_case_a_partition_fused_at_build():
    ck = compile_kernel(_graph_all_jittable(), "c.b")
    assert ck.partition is not None and ck.partition.mode == "fused"
    assert ck.is_declared is True
    assert ck.kernel(3.0) == (8.0,)


def test_case_a_recompute_after_fused_call():
    ck = compile_kernel(_graph_all_jittable(), "c.b")
    assert ck.kernel(3.0) == (8.0,)
    assert ck.recompute({"e": {"x": 4.0}}) == (10.0,)


def test_case_a_coercible_wrong_type_raises_at_build():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x * 1.5, "params": Params(type=int64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=int64))
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        compile_kernel(g, "c.a")


def test_case_a_passthrough_external_output_compiles():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    ck = compile_kernel(g, ["c.a", "e.y"])
    assert ck.is_declared is True
    assert ck.partition is not None and ck.partition.mode == "fused"
    assert ck.execute({"e": {"x": 3.0, "y": 9.0}}) == {"c.a": 4.0, "e.y": 9.0}


def test_undeclared_graph_stays_virgin_with_no_partition():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0},
    ]}, ["e"])
    ck = compile_kernel(g, "c.b")
    assert ck.is_declared is False
    assert ck.partition is None
    assert ck.kernel(3.0) == (8.0,)


def _declared_mix():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0,
         "params": Params(jitable=False, type=float64)},
        {"name": "d", "inputs": {"b": "c"}, "formula": lambda b: b + 1.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    return g


def test_case_b_segmented_partition_and_result():
    ck = compile_kernel(_declared_mix(), "c.d")
    assert ck.partition is not None and ck.partition.mode == "segmented"
    assert ck.is_declared is True
    assert "c.b" in ck.partition.python_nodes
    assert ck.kernel(3.0) == (9.0,)  # ((3+1)*2)+1


def test_case_b_no_probing_declared_python_honored():
    # c.b is trivially jittable (lambda a: a*2.0) yet declared jitable=False;
    # it must appear as Python (NOT promoted to jit) -- no probing occurs.
    ck = compile_kernel(_declared_mix(), "c.d")
    assert "c.b" in ck.partition.python_nodes


def test_case_b_formula_bearing_external_raises():
    g = _declared_mix()
    g.external["e"].update("x", Variable(name="x", source="e", formula=lambda: 1.0,
                                         params=Params(type=float64)))
    with pytest.raises(ValueError, match="external but carries a formula"):
        compile_kernel(g, "c.d")
