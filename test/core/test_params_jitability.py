import pytest
from numba import float64, int64
from numbox.core.variable.variable import (
    Params, Variable, Variables, External,
)


def test_task0_params_defaults():
    p = Params()
    assert p.jitable is True and p.type is None
    assert Params(type=float64).type is float64
    with pytest.raises(Exception):
        p.jitable = False  # frozen


def test_task0_variable_params_roundtrip_and_identity():
    a = Variable(name="a", source="m", params=Params(type=float64))
    assert a.params.type is float64
    bare = Variable(name="a", source="m")
    assert a == bare and hash(a) == hash(bare)  # params not part of identity
    assert {a, bare} == {a}  # dedup by (source, name)


def test_task0_dict_params_rejected():
    with pytest.raises(TypeError, match="params must be a Params instance"):
        Variable(name="a", source="m", params={"jitable": True})


def test_task0_varspec_passthrough():
    vs = Variables("m", [{"name": "a", "formula": lambda: 1.0, "params": Params(type=float64)}])
    assert vs["a"].params.type is float64


def test_task0_external_declare():
    e = External("ext")
    e.declare("x", Params(type=int64))
    assert e["x"].params.type is int64
