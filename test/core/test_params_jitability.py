import numpy as np
import pytest
from dataclasses import FrozenInstanceError
from numba import cfunc, float32, float64, int32, int64, vectorize
from numba import njit as _njit, int64 as _int64, float64 as _float64
from numba import njit as _njit_t3
from numba.core.errors import NumbaError
from numbox.core.variable.variable import (
    Graph, Params, Variable, Variables, External,
)
from numbox.core.variable.compile_kernel import _classify, compile_kernel
from numbox.core.variable._kernel_partition import _evaluate as _evaluate_fn
from numbox.core.variable.utils import _validate_declared_return
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
    assert a == bare and hash(a) == hash(bare)  # params are not part of the identity
    assert {a, bare} == {a}  # deduplicated by (source, name)


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


def test_njit_probe_reads_natural_return_type():
    f = _njit(lambda x: x * 1.5)
    f.compile((_int64,))
    rt = f.nopython_signatures[-1].return_type
    assert rt == _float64  # x*1.5 is float64 for an int64 x, not int64


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
        _validate_declared_return(vf, (int64,), float64, flags={})  # int+int remains int64


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
    _validate_declared_return(d, (int64,), float64, flags={})  # no raise: the natural return type is float64


def test_validate_rejects_dispatcher_wrong_declaration():
    d = _njit(lambda x: x * 1.5)
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        _validate_declared_return(d, (int64,), int64, flags={})  # the natural return type is float64, not int64


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


def test_undeclared_external_only_graph_stays_undiscovered():
    # The graph has zero interior nodes and declares nothing. A graph that declares nothing must behave exactly as on
    # the discovery path, so it requires Case C (undeclared, partition None until the first call), not an eager fused
    # build.
    g = Graph({"c": []}, ["e"])
    ck = compile_kernel(g, "e.x")
    assert ck.is_declared is False
    assert ck.partition is None
    assert ck.execute({"e": {"x": 5.5}}) == {"e.x": 5.5}


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


def test_case_b_python_run_inputs_exclude_values_made_inside_the_run():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0,
         "params": Params(jitable=False, type=float64)},
        {"name": "p", "inputs": {"b": "c", "x": "e"}, "formula": lambda b, x: b + x,
         "params": Params(jitable=False, type=float64)},
        {"name": "d", "inputs": {"p": "c"}, "formula": lambda p: p + 1.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    ck = compile_kernel(g, "c.d")
    py = [s for s in ck.partition.segments if s.kind == "python"]
    assert [(s.nodes, s.inputs, s.outputs) for s in py] == [
        (("c.b", "c.p"), ("c.a", "e.x"), ("c.b", "c.p")),
    ]
    assert ck.kernel(3.0) == (12.0,)  # a=4, b=8, p=11, d=12


def test_case_b_no_probing_declared_python_honored():
    # c.b is trivially jittable (lambda a: a*2.0) but is declared jitable=False. It must be listed among the Python
    # nodes, not promoted to jit, and no probing occurs.
    ck = compile_kernel(_declared_mix(), "c.d")
    assert "c.b" in ck.partition.python_nodes
    assert ck.kernel(3.0) == (9.0,)
    # Only the runtime probe paths (_resolve_and_call and _discover_and_run) set _store. The eager Case-B build never
    # sets it, so _store is still None after a kernel call, which shows that no probe ran. _run_segmented captures
    # _last_args on its first call, so that the precondition of a later recompute holds.
    assert ck._store is None
    assert ck._last_args == (3.0,)


def test_case_b_recompute():
    # Case B, a declared mix: c.a = x+1 is jit, c.b = a*2 is declared Python, and c.d = b+1 is jit.
    ck = compile_kernel(_declared_mix(), "c.d")
    assert ck.kernel(3.0) == (9.0,)            # ((3+1)*2)+1
    out = ck.recompute({"e": {"x": 4.0}})
    assert out == (11.0,)                       # ((4+1)*2)+1


def test_declared_return_validation_is_memoized(monkeypatch):
    import numbox.core.variable.compile_kernel as ck_mod
    ck_mod._validated_returns.clear()
    calls = []
    orig = ck_mod._validate_declared_return

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(ck_mod, "_validate_declared_return", spy)
    compile_kernel(_graph_all_jittable(), "c.b")
    first = len(calls)
    assert first > 0
    compile_kernel(_graph_all_jittable(), "c.b")  # the same formula content hits the memo and is not re-probed
    assert len(calls) == first


def test_declared_type_variants_get_distinct_anchors():
    def make(t):
        g = Graph({"c": [{"name": "a", "inputs": {"x": "e"},
                          "formula": lambda x: x + 1, "params": Params(type=t)}]}, ["e"])
        g.external["e"].declare("x", Params(type=t))
        return compile_kernel(g, "c.a", cache=True)
    ck_i = make(int64)
    ck_f = make(float64)
    assert ck_i.kernel(3) == (4,)
    assert ck_f.kernel(3.0) == (4.0,)
    assert ck_i.source == ck_f.source              # source is type-free
    assert ck_i._fused.__name__ != ck_f._fused.__name__   # the declared signatures make the digests differ


def test_inner_formula_bindings_stay_uncached():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    ck = compile_kernel(g, "c.b")
    bindings_by_var = ck._ctx[2]
    for binding in bindings_by_var.values():
        assert binding.targetoptions.get("cache") in (None, False)


def test_declared_array_recompute_accepts_c_contiguous():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0,
         "params": Params(type=float64[:])},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64[:]))
    ck = compile_kernel(g, "c.a")
    base = np.zeros(4)               # C-contiguous; declared layout is 'A'
    assert ck.kernel(base)[0].tolist() == [1, 1, 1, 1]
    out = ck.recompute({"e": {"x": np.ones(4)}})   # must not raise on the layout
    assert out[0].tolist() == [2, 2, 2, 2]


def test_declared_recompute_off_contract_type_raises_crisp():
    ck = compile_kernel(_graph_all_jittable(), "c.b")
    ck.kernel(3.0)
    with pytest.raises(ValueError, match="declared type"):
        ck.recompute({"e": {"x": 3j}})  # a complex value, but float64 is declared


def test_declared_recompute_non_typeable_value_raises_crisp():
    # A value that numba cannot type at all must still produce the "declared type" contract error, not a raw typeof
    # exception.
    ck = compile_kernel(_graph_all_jittable(), "c.b")
    ck.kernel(3.0)
    with pytest.raises(ValueError, match="declared type"):
        ck.recompute({"e": {"x": object()}})


def test_failed_first_fused_call_leaves_kernel_unseeded():
    # A first call that raises must not record _last_args or change the mode, so that a later recompute refuses to
    # run instead of running on state that no full call has seeded.
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: 100 // x, "params": Params(type=int64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=int64))
    ck = compile_kernel(g, "c.a")
    with pytest.raises(ZeroDivisionError):
        ck.kernel(0)
    assert ck._last_args is None
    with pytest.raises(RuntimeError, match="requires a prior full call"):
        ck.recompute({"e": {"x": 5}})


def test_failed_first_segmented_call_leaves_kernel_unseeded():
    # The segmented path has the same invariant: a first call that raises leaves the kernel unseeded, so recompute
    # refuses to run.
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: 100 // x, "params": Params(type=int64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a + 1,
         "params": Params(jitable=False, type=int64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=int64))
    ck = compile_kernel(g, "c.b")
    with pytest.raises(ZeroDivisionError):
        ck.kernel(0)
    assert ck._last_args is None
    with pytest.raises(RuntimeError, match="requires a prior full call"):
        ck.recompute({"e": {"x": 5}})


def test_case_a_fused_recompute_still_works():
    ck = compile_kernel(_graph_all_jittable(), "c.b")
    assert ck.kernel(3.0) == (8.0,)
    assert ck.recompute({"e": {"x": 4.0}}) == (10.0,)   # the store is seeded through _evaluate


def _case_c_with_declared_node():
    # c.a is declared (STATIC_JIT) and c.b is undeclared (UNKNOWN), so the graph is Case C and is_declared is False.
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    return g


def test_case_c_declared_node_skips_contract_check():
    ck = compile_kernel(_case_c_with_declared_node(), "c.b")
    assert ck.is_declared is False
    assert ck.kernel(3.0) == (8.0,)
    # e.x carries params.type=float64, but the kernel is Case C, so the declared contract check is skipped. A
    # recompute with an int value uses the existing recovery path and must not raise the "declared type" ValueError.
    out = ck.recompute({"e": {"x": 5}})
    assert out == (12.0,)


def test_eager_kernel_no_silent_rediscover_off_contract():
    # An eager (declared) kernel must not silently re-discover or overwrite its frozen _demoted on an off-contract
    # input. It raises the "declared type" ValueError instead. The test observes the absence of a silent re-probe as
    # that error and as an unchanged _demoted.
    ck = compile_kernel(_graph_all_jittable(), "c.b")
    ck.kernel(3.0)
    demoted_before = dict(ck._demoted)
    with pytest.raises(ValueError, match="declared type"):
        ck.recompute({"e": {"x": 3j}})
    assert ck._demoted == demoted_before


def test_shared_dispatcher_multi_overload_validation():
    # Three declared interior nodes all use one shared dispatcher that has several overloads. If validation read the
    # last-compiled overload instead of the one for the node's own input types, it would falsely reject a fully
    # correct graph. The int64-input nodes naturally yield int64, but the float64 node's overload yields float64, so
    # validation that reads it would report a spurious mismatch between float64 and int64.
    shared = _njit(lambda x: x + 1)
    g = Graph({"c": [
        {"name": "a", "inputs": {"p": "e"}, "formula": shared, "params": Params(type=int64)},
        {"name": "b", "inputs": {"q": "e"}, "formula": shared, "params": Params(type=float64)},
        {"name": "d", "inputs": {"r": "e"}, "formula": shared, "params": Params(type=int64)},
    ]}, ["e"])
    g.external["e"].declare("p", Params(type=int64))
    g.external["e"].declare("q", Params(type=float64))
    g.external["e"].declare("r", Params(type=int64))
    ck = compile_kernel(g, ["c.a", "c.b", "c.d"])  # must not raise
    assert ck.is_declared is True
    assert ck.execute({"e": {"p": 3, "q": 2.0, "r": 5}}) == {"c.a": 4, "c.b": 3.0, "c.d": 6}


def test_declared_exotic_node_end_to_end():
    # A declared graph that carries a cres (CompileResultWAP) node builds eagerly and runs end-to-end through
    # compile_kernel. The natural return type is read from the exotic's own signature, so a correct declaration
    # computes the expected value.
    cf = cres(float64(float64))(lambda x: x + 1.0)
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": cf, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    ck = compile_kernel(g, "c.a")
    assert ck.is_declared is True
    assert ck.partition is not None and ck.partition.mode == "fused"
    assert ck.kernel(3.0) == (4.0,)


def test_declared_exotic_node_wrong_declaration_raises_at_build():
    # A wrong declaration on the exotic node (cres yields float64, declared int64) is caught at build by the
    # exotic-aware return probe, not at the first call.
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"},
         "formula": cres(float64(float64))(lambda x: x + 1.0), "params": Params(type=int64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        compile_kernel(g, "c.a")


def test_declared_segmented_throughput_off_contract_reraises():
    # A declared segmented kernel must re-raise on a later kernel() call whose off-contract input makes a jit segment
    # fail to compile. It must not silently re-discover and overwrite the frozen _demoted set.
    ck = compile_kernel(_declared_mix(), "c.d")
    assert ck.partition.mode == "segmented"
    assert ck.kernel(3.0) == (9.0,)             # ((3+1)*2)+1
    demoted_before = dict(ck._demoted)
    with pytest.raises(NumbaError):
        ck.kernel("not a number")               # the first jit segment (x + 1.0) fails to compile
    assert ck._demoted == demoted_before        # the demotion verdicts are unchanged


def test_declared_float32_graph_end_to_end():
    # A declared narrow-scalar (float32) graph whose body naturally yields float32
    # builds fused and runs end-to-end.
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + float32(1.0),
         "params": Params(type=float32)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float32))
    ck = compile_kernel(g, "c.a")
    assert ck.is_declared is True
    assert ck.kernel(np.float32(2.0)) == (3.0,)


def test_declared_int32_over_widening_body_rejected_at_build():
    # A declared int32 over an int-arithmetic body is rejected at build. numba widens int32 + 1 to int64, so the
    # natural return type (int64) does not match the declared int32. The test documents the rejected narrow-scalar
    # behavior.
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1, "params": Params(type=int32)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=int32))
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        compile_kernel(g, "c.a")
