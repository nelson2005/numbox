"""Tests for ``numbox.utils.fingerprint``, the texts that describe functions and values the same way in every process.

The tests check that ``_canon_value`` gives different texts to values that numba compiles into different code, such
as structured arrays with the same bytes, two C symbols, ctypes pointers whose ``argtypes`` differ and two kinds of
function value. They check that it raises ``_Unfingerprintable`` for a value with no text that means the same thing in
another process.
Other tests check that a fingerprint folds the globals and the module attributes that a function reads, and not a
module global that only shares an attribute's name. ``co_names`` holds both kinds of name, so ``_loaded_global_names``
reads the opcodes. Folding such a global would change the fingerprint when only the global changes, and a global
without a fingerprint would make the fingerprint fail, so the work builder would compile the derive without a cache.
An ``@intrinsic`` has no fingerprint, and the best-effort fingerprint folds a user's intrinsic by its definition
function and its options. Four rules apply with ``fold_named``. A module attribute that holds a function or a class is
folded by its content or by its name, and one that holds an ``@intrinsic`` of the user's leaves its reader without a
fingerprint. A module attribute that holds a numba type or a dtype object is folded by the identity of the type. A
helper without a fingerprint leaves its caller without one. The text of a dispatcher holds the types that it was
declared with. The tests state each rule, and that the text without ``fold_named`` does not change. The last tests cover
``_opaque_values_match`` and the ``@proxy`` aliases that a collision retires. The tests that need a C math library skip
if ``open_libm`` finds none.
"""
import abc
import cmath
import collections
import enum
import heapq
import importlib.util
import math
import random
import sys
import textwrap
import types

import numba
import numpy as np
import pytest
from numba import deferred_type, float32, float64, guvectorize, njit, stencil, vectorize
from numba.core.types import ExternalFunction
from numba.extending import intrinsic

from numbox.utils import clock, fingerprint, lowlevel
from numbox.utils.fingerprint import (
    _ABSENT_ALIASES, _MAX_REPR_INT_BITS, _Unfingerprintable, _canon_value, _fingerprint_function,
    _fingerprint_function_best_effort, _fingerprint_module_attrs, _loaded_global_names, _named_value_text,
    _opaque_values_match, _referenced_global_names,
)
from test.auxiliary_utils import open_libm


def _make(src, **globs):
    """Return the function ``f`` that the source ``src`` defines, compiled with ``globs`` as its module globals."""
    ns = dict(globs)
    exec(textwrap.dedent(src), ns)  # nosec B102 - test helper compiling a fixed snippet
    return ns["f"]


def test_canon_value_structured_dtype_preserves_layout():
    # dtype.str is |V<itemsize> for every structured dtype and drops the fields, so arrays with the same bytes and
    # different layouts would have one text. _dtype_key folds the dtype's str, descr and itemsize.
    raw = (1).to_bytes(4, "little") + (2).to_bytes(4, "little")
    two_i4 = np.frombuffer(raw, dtype=np.dtype([("a", "<i4"), ("b", "<i4")])).copy()
    two_f4 = np.frombuffer(raw, dtype=np.dtype([("a", "<f4"), ("b", "<f4")])).copy()
    one_i8 = np.frombuffer(raw, dtype=np.dtype([("x", "<i8")])).copy()

    assert two_i4.tobytes() == two_f4.tobytes() == one_i8.tobytes()
    canons = {_canon_value(a, set()) for a in (two_i4, two_f4, one_i8)}
    assert len(canons) == 3, "structured-dtype arrays with identical bytes collided"


def test_canon_value_plain_dtype_unchanged():
    canon = _canon_value(np.arange(3, dtype=np.float64), set())
    assert canon.startswith("ndarray(<f8;")


def test_canon_value_separates_the_field_orders_of_a_reordered_structured_array():
    # NumPy defines no descr for the dtype of base[["b", "a"]], whose fields are not in the order of their offsets.
    # The dtype's str lists the offsets, so the text tells that view from the views with other fields or another order.
    base = np.zeros(2, dtype=[("a", "<f8"), ("b", "<f8"), ("c", "<f8")])

    canons = {_canon_value(base[names], set()) for names in (["b", "a"], ["a", "b"], ["c", "a"])}
    assert len(canons) == 3
    assert _canon_value(base[["b", "a"]][0], set()) != _canon_value(base[["c", "a"]][0], set())


def test_canon_value_writes_an_int_the_same_way_under_every_digit_limit():
    # repr raises for an int of more decimal digits than the interpreter's limit, and a process can change the limit.
    # An int above the bound is written with hex, which has no limit. An int at the bound has no more digits than the
    # lowest limit that a process can set, so repr accepts it in every process.
    huge = 7 ** 6000

    assert _canon_value(huge, set()) == hex(huge)
    assert len({_canon_value(value, set()) for value in (huge, huge + 1, -huge)}) == 3
    assert _canon_value([huge], set()) == f"list[{hex(huge)}]"

    bound = 2 ** _MAX_REPR_INT_BITS - 1
    assert _canon_value(bound, set()) == repr(bound)
    assert len(repr(bound)) <= sys.int_info.str_digits_check_threshold


def test_named_value_text_names_a_ufunc_or_a_builtin_function_that_its_module_holds():
    # numpy holds sin under the ufunc's own name, and math holds sqrt, so the module and the name identify each of them
    # in every process. np.abs is a second name for the ufunc absolute, and the text holds the ufunc's own name.
    assert _named_value_text(np.sin) == "ufunc(numpy.sin)"
    assert _named_value_text(np.abs) == "ufunc(numpy.absolute)"
    assert _named_value_text(math.sqrt) == "builtin(math.sqrt)"
    assert _named_value_text(len) == "builtin(builtins.len)"
    # The strict walk does not use the name: a value that _canon_value has no text for makes the code that reads it
    # compile without a cache, and that stays so for these functions.
    for value in (np.sin, math.sqrt, len):
        with pytest.raises(_Unfingerprintable):
            _canon_value(value, set())


def test_named_value_text_is_none_for_a_function_that_no_module_holds_by_name(monkeypatch):
    # The text names a function by its module and its name, so it must not be given to another function of that name.
    # A ufunc that the user built and named sin is not the one that numpy holds. The parallel target of vectorize
    # returns a NumPy ufunc. A method of an object records no module. A function that was replaced in its module is
    # no longer found there, and the function that replaced it is named as its own module holds it.
    @vectorize([float64(float64)], target="parallel")
    def sin(x):
        return x + 1.0

    assert isinstance(sin, np.ufunc) and sin.__name__ == "sin"
    replaced = math.sqrt
    monkeypatch.setattr(math, "sqrt", math.sin)
    # A builtin function accepts any object as its __module__. One whose __module__ is not a string has no name.
    monkeypatch.setattr(math.fsum, "__module__", ["math"])

    for value in (sin, [].append, replaced, math.fsum, float64, 2.0, test_canon_value_structured_dtype_preserves_layout):
        assert _named_value_text(value) is None
    assert _named_value_text(math.sqrt) == "builtin(math.sin)"


def test_named_value_text_reads_only_a_loaded_module(tmp_path, monkeypatch):
    # A builtin function accepts any string as its __module__, and sys.modules can hold anything under that string.
    # Reading the __dict__ of a module that importlib.util.LazyLoader has not loaded yet runs the module, so the lookup
    # must leave such an entry alone, and an entry that is no module as well.
    source = tmp_path / "lazy_home.py"
    source.write_text("raise RuntimeError('the lookup ran a lazy module')\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("lazy_home", source)
    spec.loader = importlib.util.LazyLoader(spec.loader)
    lazy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lazy)
    monkeypatch.setitem(sys.modules, "lazy_home", lazy)
    monkeypatch.setitem(sys.modules, "no_module", types.SimpleNamespace(fsum=math.fsum))

    for home in ("lazy_home", "no_module"):
        monkeypatch.setattr(math.fsum, "__module__", home)
        assert _named_value_text(math.fsum) is None
    assert type(lazy) is not types.ModuleType, "the lookup loaded the lazy module"


def test_named_value_text_does_not_ask_an_impostor_for_its_name():
    # An object can report any class as its __class__, and isinstance believes it. Such an object is not a ufunc or a
    # builtin function, and reading its __name__ can run arbitrary code.
    class Impostor:
        def __init__(self, reported):
            self._reported = reported

        @property
        def __class__(self):
            return self._reported

        @property
        def __name__(self):
            raise RuntimeError("the lookup read the name of an impostor")

    helpers = types.ModuleType("helpers")
    for reported in (np.ufunc, type(len)):
        impostor = Impostor(reported)
        assert isinstance(impostor, reported)
        assert _named_value_text(impostor) is None
        helpers.op = impostor
        assert _fingerprint_module_attrs({"op"}, {"helpers": helpers}, set(), True) == ""


def test_the_fold_runs_no_code_of_a_str_subclass_or_of_an_odd_module(monkeypatch):
    # A module's __name__ and a builtin function's __module__ can be instances of a subclass of str, whose methods
    # are user code. A subclass of the module type can provide a __dict__ that is not a dict. The fold reads such a
    # __dict__ only by membership and by item, as the walk over data attributes does.
    class Loud(str):
        def partition(self, sep):
            raise RuntimeError("the fold called a method of a str subclass")

        def __hash__(self):
            raise RuntimeError("the fold hashed a str subclass")

        def __eq__(self, other):
            raise RuntimeError("the fold compared a str subclass")

    helpers = types.ModuleType("helpers")
    helpers.__dict__["__name__"] = Loud("numpy.lookalike")
    helpers.op = np.sin
    assert _fingerprint_module_attrs({"op"}, {"helpers": helpers}, set(), True) == (
        ";module_attrs=[helpers.op=ufunc(numpy.sin)]"
    )

    class NoGet:
        def __init__(self, items):
            self._items = items

        def __contains__(self, key):
            return key in self._items

        def __getitem__(self, key):
            return self._items[key]

    class OddModule(types.ModuleType):
        @property
        def __dict__(self):
            return NoGet({"__name__": "odd", "op": np.cos})

    assert _fingerprint_module_attrs({"op"}, {"odd": OddModule("odd")}, set(), True) == (
        ";module_attrs=[odd.op=ufunc(numpy.cos)]"
    )

    # The name of a module that an attribute holds is read in the same way, and a name of a str subclass is not folded.
    helpers.lib = types.ModuleType("held")
    helpers.lib.__dict__["__name__"] = Loud("held")
    assert _fingerprint_module_attrs({"lib"}, {"helpers": helpers}, set(), True) == ""
    helpers.lib = OddModule("a_name_that_only_getattr_returns")
    assert _fingerprint_module_attrs({"lib"}, {"helpers": helpers}, set(), True) == (
        ";module_attrs=[helpers.lib=module(odd)]"
    )

    monkeypatch.setattr(math.fsum, "__module__", Loud("math"))
    assert _named_value_text(math.fsum) is None

    # A value that is data must keep the entry that it has as data. The fold must not take it for a ufunc because
    # its class has a metaclass whose __eq__ agrees with any operand, or because it reports another __class__.
    class AgreesWithAll(type):
        def __eq__(cls, other):
            return True

        def __hash__(cls):
            return 0

    class Scale(float, metaclass=AgreesWithAll):
        pass

    class Disguised(float):
        @property
        def __class__(self):
            return np.ufunc

    plain = types.ModuleType("plain")
    for value in (Scale(2.0), Disguised(2.0)):
        plain.SCALE = value
        folded = _fingerprint_module_attrs({"SCALE"}, {"plain": plain}, set(), True)
        assert folded == _fingerprint_module_attrs({"SCALE"}, {"plain": plain}, set()) == ";module_attrs=[plain.SCALE=2.0]"


def test_a_function_that_reads_a_named_function_by_a_bare_name_has_no_fingerprint():
    # After "from math import sqrt", a function reads sqrt as a global. Such a function has no fingerprint, so code
    # that is cached by the fingerprint compiles without a cache for it. The best-effort text, which names a @proxy
    # function, holds a placeholder for sqrt and hands the function to the caller. @proxy compares the values handed
    # over, and that comparison is what tells two bodies with equal texts apart.
    src = """
        def f(x):
            return op(x)
    """
    for op in (math.sqrt, np.sin):
        f = _make(src, op=op)
        with pytest.raises(_Unfingerprintable):
            _fingerprint_function(f, set())
        opaque = []
        assert f"op=<opaque:{type(op).__name__}>" in _fingerprint_function_best_effort(f, opaque)
        assert opaque == [op]


def test_a_module_attribute_that_holds_a_ufunc_or_a_builtin_function_is_folded_by_its_name():
    # helpers.op can hold numpy.sin in one process and numpy.cos in the next. The code of f names only op, so the
    # fingerprint folds the name of the function that the attribute holds.
    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.op(x)
    """, helpers=helpers)

    texts, plain = [], []
    for op in (np.sin, np.cos, math.sqrt):
        helpers.op = op
        texts.append(_fingerprint_function(f, set(), fold_named=True))
        plain.append(_fingerprint_function(f, set()))
    assert len(set(texts)) == 3
    assert "helpers.op=ufunc(numpy.sin)" in texts[0]
    assert "helpers.op=builtin(math.sqrt)" in texts[2]
    # Without fold_named the text is the one that @proxy and cres hash into an alias. It leaves the attribute out, so
    # that an alias does not change with it.
    assert len(set(plain)) == 1 and "module_attrs" not in plain[0]
    # Another module that holds a function under the function's own name can hold another function in the next
    # process too, so that attribute is folded as well.
    helpers.sin = np.sin
    assert _fingerprint_module_attrs({"sin"}, {"helpers": helpers}, set(), True) == (
        ";module_attrs=[helpers.sin=ufunc(numpy.sin)]"
    )
    assert _fingerprint_module_attrs({"sin"}, {"helpers": helpers}, set()) == ""


def test_a_module_that_a_module_attribute_holds_is_folded_by_its_name():
    # helpers.lib can hold math in one process and cmath in the next. Each provides sqrt as its own function, which
    # is not folded, so the fingerprint folds the name of the module that the attribute holds.
    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.lib.sqrt(x)
    """, helpers=helpers)

    texts, plain = [], []
    for lib in (math, cmath):
        helpers.lib = lib
        texts.append(_fingerprint_function(f, set(), fold_named=True))
        plain.append(_fingerprint_function(f, set()))
    assert texts[0].endswith(";module_attrs=[helpers.lib=module(math)])")
    assert texts[1].endswith(";module_attrs=[helpers.lib=module(cmath)])")
    # @proxy and cres hash the text without fold_named into an alias. That text leaves the module out.
    assert plain[0] == plain[1] and "module_attrs" not in plain[0]
    # A submodule that its own package holds is not folded, as numpy.linalg is not. A package holds a submodule only
    # in a process that has loaded it, so folding it would make the text depend on which submodules are loaded. The
    # walk still reaches its attributes.
    package = types.ModuleType("package")
    package.sub = types.ModuleType("package.sub")
    package.sub.SCALE = 3.0
    assert _fingerprint_module_attrs({"sub", "SCALE"}, {"package": package}, set(), True) == (
        ";module_attrs=[package.sub.SCALE=3.0]"
    )
    # A module of another package whose name starts with the holder's name is not a submodule of the holder.
    package.sub = types.ModuleType("package_extras.sub")
    assert _fingerprint_module_attrs({"sub"}, {"package": package}, set(), True) == (
        ";module_attrs=[package.sub=module(package_extras.sub)]"
    )


@pytest.mark.parametrize("kind", [
    "global", "closure", "default", "keyword default", "tuple", "set", "dict key", "dict value", "dispatcher",
    "vectorize", "cres", "module attribute",
])
def test_fold_named_reaches_a_function_that_the_fingerprinted_function_calls(kind):
    # The attribute helpers.op is read by a helper. The fingerprinted function reaches the helper in one of the ways
    # that the walk follows, and the walk must fold the attribute on each of them.
    from numbox.utils.highlevel import cres

    helpers = types.ModuleType("helpers")
    helpers.op = np.sin
    helper = _make("""
        def f(x):
            return helpers.op(x)
    """, helpers=helpers)
    call_global = """
        def f(x):
            return helper(x)
    """
    if kind == "global":
        caller = _make(call_global, helper=helper)
    elif kind == "closure":
        def caller(x):
            return helper(x)
    elif kind == "default":
        caller = _make("""
            def f(x, g=helper):
                return g(x)
        """, helper=helper)
    elif kind == "keyword default":
        caller = _make("""
            def f(x, *, g=helper):
                return g(x)
        """, helper=helper)
    elif kind == "module attribute":
        outer = types.ModuleType("outer")
        outer.inner = njit(helper)
        caller = _make("""
            def f(x):
                return outer.inner(x)
        """, outer=outer)
    else:
        held = {
            "tuple": lambda: (helper,), "set": lambda: frozenset({helper}), "dict key": lambda: {helper: 1.0},
            "dict value": lambda: {"k": helper}, "dispatcher": lambda: njit(helper),
            "vectorize": lambda: vectorize(helper), "cres": lambda: cres(float64(float64))(helper),
        }[kind]()
        caller = _make(call_global, helper=held)

    texts, plain = [], []
    for op in (np.sin, np.cos):
        helpers.op = op
        texts.append(_fingerprint_function(caller, set(), fold_named=True))
        plain.append(_fingerprint_function(caller, set()))
    assert texts[0] != texts[1]
    assert plain[0] == plain[1]


def test_a_named_function_reached_through_its_own_package_is_not_folded():
    # numpy.add, numpy.abs and math.sqrt are the same functions in every process, so they add nothing to the
    # fingerprint of a function that calls them through their modules. A submodule of numpy that holds add as well,
    # under that name or another, is loaded in some processes and not in others. numpy.strings.add and
    # numpy.ma.core.narray are such attributes. They are not folded either, so the fingerprint does not depend on which
    # submodules are loaded.
    # The two modules are read by two functions. NumPy before 1.25 holds the module math as numpy.math, so a function
    # that names both modules would fold that attribute.
    through_numpy = _make("""
        def f(x):
            return np.abs(np.add(x, 1.0))
    """, np=np)
    through_math = _make("""
        def f(x):
            return math.sqrt(x)
    """, math=math)
    for f in (through_numpy, through_math):
        assert "module_attrs" not in _fingerprint_function(f, set(), fold_named=True)

    submodule = types.ModuleType("numpy.a_submodule")
    submodule.add = np.add
    submodule.narray = np.array
    assert _fingerprint_module_attrs({"add", "narray"}, {"sub": submodule}, set(), True) == ""
    # A module of another package whose name starts with "numpy" is not a module of numpy.
    lookalike = types.ModuleType("numpy_extras.ops")
    lookalike.add = np.add
    assert _fingerprint_module_attrs({"add"}, {"sub": lookalike}, set(), True) == (
        ";module_attrs=[sub.add=ufunc(numpy.add)]"
    )


def test_named_value_text_names_a_function_or_a_class_that_numba_compiles_from_its_own_implementation():
    # numba compiles a call of numpy.sum, heapq.nlargest, numpy.float32 or float from an implementation of its own,
    # so the module and the name identify each of them in every process. numpy.sum is not a ufunc, heapq.nlargest is
    # a plain Python function, and the last two are classes. A jitted function of numbox is named after its Python
    # function.
    assert _named_value_text(np.sum) == "named(numpy.sum)"
    assert _named_value_text(heapq.nlargest) == "named(heapq.nlargest)"
    # collections is a package of the standard library, and abc is frozen into the interpreter from Python 3.11 on.
    assert _named_value_text(collections.namedtuple) == "named(collections.namedtuple)"
    assert _named_value_text(abc.abstractmethod) == "named(abc.abstractmethod)"
    assert _named_value_text(np.float32) == "class(numpy.float32)"
    assert _named_value_text(float) == "class(builtins.float)"
    assert _named_value_text(lowlevel.array_data_p) == "named(numbox.utils.lowlevel.array_data_p)"


def test_named_value_text_is_none_for_a_function_or_a_class_of_any_other_package(monkeypatch):
    # numba compiles a function of the user from the function's own body, and the name does not identify the body.
    # A class of the user, an enum, whose metaclass is not type, and a numba type have no name either. A function that
    # was replaced in its module is no longer found there.
    class Mine:
        pass

    class Color(enum.Enum):
        RED = 1

    replaced = heapq.nlargest
    monkeypatch.setattr(heapq, "nlargest", heapq.nsmallest)

    user_function = test_canon_value_plain_dtype_unchanged
    for value in (user_function, njit(user_function), Mine, Color, enum.Enum, float64, replaced):
        assert _named_value_text(value) is None
    # heapq now holds nsmallest under two names. Its name is the one that it has itself.
    assert _named_value_text(heapq.nlargest) == "named(heapq.nsmallest)"


def test_a_module_attribute_that_holds_a_function_or_a_class_of_numpy_is_folded_by_its_name():
    # helpers.op can hold numpy.sum in one process and numpy.prod in the next, and helpers.T can hold numpy.float32 or
    # numpy.float64. The code of f names only the attribute, so the fingerprint folds the name of the value.
    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.op(x)
    """, helpers=helpers)

    texts, plain = [], []
    for op in (np.sum, np.prod, np.float32, np.float64, float, heapq.nlargest):
        helpers.op = op
        texts.append(_fingerprint_function(f, set(), fold_named=True))
        plain.append(_fingerprint_function(f, set()))
    assert len(set(texts)) == 6
    assert "helpers.op=named(numpy.sum)" in texts[0]
    assert "helpers.op=class(numpy.float32)" in texts[2]
    assert "helpers.op=class(builtins.float)" in texts[4]
    assert "helpers.op=named(heapq.nlargest)" in texts[5]
    # The text that @proxy and cres hash into an alias leaves the attribute out.
    assert len(set(plain)) == 1 and "module_attrs" not in plain[0]

    # numpy holds its own functions and classes in the same way in every process, so they add nothing to the
    # fingerprint of a function that reaches them through numpy.
    g = _make("""
        def f(x):
            return np.sum(np.ones(3) * np.float32(x)) + heapq.nlargest(1, [x])[0]
    """, np=np, heapq=heapq)
    assert "module_attrs" not in _fingerprint_function(g, set(), fold_named=True)


def test_named_value_text_names_a_function_that_is_a_method_of_a_generator_of_its_module():
    # numpy.random.normal and random.gauss are methods bound to a generator object that the module creates.
    # random.random is a builtin method of that object and records no module. The module of the object's class holds
    # each of them under the function's own name. A method of a generator that the user created is not the object
    # that the module holds, so it has no name.
    assert _named_value_text(np.random.normal) in (
        "named(numpy.random.mtrand.normal)", "builtin(numpy.random.mtrand.normal)")
    assert _named_value_text(random.gauss) == "named(random.gauss)"
    assert _named_value_text(random.random) == "builtin(random.random)"
    assert _named_value_text(np.random.RandomState(1).normal) is None
    assert _named_value_text(random.Random(1).gauss) is None

    # The lookup reads the module of the object's class only if the metaclass is type.
    class Meta(type):
        @property
        def __module__(cls):
            raise AssertionError("the lookup ran code of a metaclass")

    class Odd(metaclass=Meta):
        def method(self):
            return None

    assert _named_value_text(Odd().method) is None


def test_a_module_attribute_that_holds_a_function_of_numpy_random_or_of_random_is_folded_by_its_name():
    # helpers.draw can hold numpy.random.normal in one process and numpy.random.uniform in the next. The code of f
    # names only the attribute, so the fingerprint folds the name of the function.
    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.draw(x, x)
    """, helpers=helpers)

    texts = []
    for draw in (np.random.normal, np.random.uniform, random.gauss, random.uniform):
        helpers.draw = draw
        texts.append(_fingerprint_function(f, set(), fold_named=True))
    assert len(set(texts)) == 4
    assert "helpers.draw=named(random.gauss)" in texts[2]

    # numpy.random and random hold their own functions in the same way in every process.
    g = _make("""
        def f(x):
            return np.random.normal(x, x) + random.gauss(x, x) + random.random()
    """, np=np, random=random)
    assert "module_attrs" not in _fingerprint_function(g, set(), fold_named=True)


def test_a_module_of_the_user_with_the_name_of_a_standard_library_module_is_folded_by_its_content(
        tmp_path, monkeypatch):
    # A user's module named colorsys replaces the standard library's module in sys.modules. Its file is not in the
    # standard-library directory. numba compiles its functions from their own content, so the fingerprint folds that
    # content, and it folds a numba type that the module holds.
    texts = []
    for factor in ("2.0", "3.0"):
        source = tmp_path / factor / "colorsys.py"
        source.parent.mkdir()
        source.write_text(
            f"from numba import float32\n\nT = float32\n\n\ndef step(x):\n    return {factor} * x\n", encoding="utf-8")
        spec = importlib.util.spec_from_file_location("colorsys", source)
        mine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mine)
        monkeypatch.setitem(sys.modules, "colorsys", mine)
        assert mine.step.__module__ == "colorsys" and _named_value_text(mine.step) is None
        f = _make("""
            def f(x):
                return colorsys.step(colorsys.T(x))
        """, colorsys=mine)
        text = _fingerprint_function(f, set(), fold_named=True)
        assert "colorsys.step=function(" in text and "colorsys.T=" in text
        texts.append(text)
    assert texts[0] != texts[1]


def test_a_module_attribute_that_holds_a_function_of_the_user_is_folded_by_its_content():
    # numba compiles a @register_jitable function, a cres function, an ExternalFunction and a ctypes function pointer
    # from the value's own content. helpers.step can hold another one in the next process, or the same function after
    # an edit, so the fingerprint folds the text that the value has when it is read as a global.
    from numbox.utils.highlevel import cres

    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.step(x)
    """, helpers=helpers)
    # The two bodies have one name and differ in a constant, as a function does before and after an edit.
    double = _make("def f(x):\n    return 2.0 * x\n")
    triple = _make("def f(x):\n    return 3.0 * x\n")
    steps = [
        double, triple, cres(float64(float64))(double), cres(float64(float64))(triple),
        ExternalFunction("sin", float64(float64)), ExternalFunction("cos", float64(float64)),
    ]
    libm = open_libm()
    if libm is not None:
        steps += [libm.sin, libm.cos]

    texts, plain = [], []
    for step in steps:
        helpers.step = step
        texts.append(_fingerprint_function(f, set(), fold_named=True))
        plain.append(_fingerprint_function(f, set()))
    assert len(set(texts)) == len(steps)
    for step, text in zip(steps, texts):
        assert text.endswith(f";module_attrs=[helpers.step={_canon_value(step, set(), True)}])")
    # The text that @proxy and cres hash into an alias leaves the attribute out.
    assert len(set(plain)) == 1 and "module_attrs" not in plain[0]


def test_a_helper_without_a_fingerprint_behind_a_module_attribute_leaves_its_caller_without_one():
    # A function, a dispatcher or a @vectorize function that reads math.sqrt as a global has no fingerprint. A caller
    # that reads it as a global has none either, and the code that numbox names by the fingerprint compiles without a
    # cache. The same holds when the caller reads the helper through a module, so that numba never loads code that
    # was compiled for another body of the helper.
    opaque = _make("def f(x):\n    return sqrt(x)\n", sqrt=math.sqrt)
    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.step(x)
    """, helpers=helpers)
    # A @guvectorize function and a @stencil function never have a fingerprint, regardless of what they read.
    gufunc = guvectorize("(n)->(n)")(_make("def f(a, out):\n    out[:] = 2.0 * a\n"))
    window = stencil(_make("def f(a):\n    return 2.0 * a[0]\n"))
    for step in (opaque, njit(opaque), vectorize(opaque), gufunc, window):
        with pytest.raises(_Unfingerprintable):
            _canon_value(step, set(), True)
        helpers.step = step
        with pytest.raises(_Unfingerprintable):
            _fingerprint_function(f, set(), fold_named=True)
        # The text that @proxy and cres hash into an alias leaves the attribute out.
        assert "module_attrs" not in _fingerprint_function(f, set())

    # The walk matches names and does not check which attributes the code reads. g calls the method sum of an array,
    # and helpers holds a function under the same name, so that function is folded too, and g has no fingerprint
    # while that function has none.
    g = _make("""
        def f(x):
            return helpers.SCALE * x.sum()
    """, helpers=helpers)
    helpers.SCALE = 2.0
    helpers.sum = opaque
    with pytest.raises(_Unfingerprintable):
        _fingerprint_function(g, set(), fold_named=True)
    helpers.sum = _make("def f(x):\n    return 2.0 * x\n")
    assert "helpers.SCALE=2.0" in _fingerprint_function(g, set(), fold_named=True)


def test_data_that_holds_a_caller_of_a_helper_without_a_fingerprint_leaves_its_reader_without_one():
    # helpers.OPS is a tuple that holds a dispatcher, and the dispatcher calls low.raw, which has no fingerprint,
    # through a module. The dispatcher has a text without fold_named and none with it, and so has the tuple. A reader
    # of helpers.OPS then has no fingerprint, as a reader of an attribute that holds the dispatcher itself has none.
    # If the walk left the tuple out, an edit to the dispatcher would not change the reader's fingerprint.
    low = types.ModuleType("low")
    low.raw = njit(_make("def f(x):\n    return sqrt(x)\n", sqrt=math.sqrt))
    op = njit(_make("def f(x):\n    return 2.0 * low.raw(x)\n", low=low))
    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.OPS[0](x)
    """, helpers=helpers)
    for ops in ((op,), [op], {"first": op}):
        helpers.OPS = ops
        with pytest.raises(_Unfingerprintable):
            _fingerprint_function(f, set(), fold_named=True)
        assert "helpers.OPS=" in _fingerprint_function(f, set())

    # Data that has no text without fold_named either is left out, and its reader keeps its fingerprint.
    helpers.OPS = (float64, op)
    assert "module_attrs" not in _fingerprint_function(f, set(), fold_named=True)
    assert "module_attrs" not in _fingerprint_function(f, set())


def test_a_jitted_function_of_numbox_without_a_fingerprint_keeps_its_caller_cacheable():
    # array_data_p calls an @intrinsic, so it has no fingerprint. numbox supplies it, and numbox's version is part of
    # the key of every entry that numbox caches, so a caller that reaches it through its module keeps its fingerprint.
    # When a module of another package holds it in an attribute, it is folded by its name.
    with pytest.raises(_Unfingerprintable):
        _canon_value(lowlevel.array_data_p, set(), True)
    f = _make("""
        def f(a):
            return lowlevel.array_data_p(a)
    """, lowlevel=lowlevel)
    assert "module_attrs" not in _fingerprint_function(f, set(), fold_named=True)

    helpers = types.ModuleType("helpers")
    g = _make("""
        def f(a):
            return helpers.address(a)
    """, helpers=helpers)
    texts = []
    for address in (lowlevel.array_data_p, lowlevel.get_unicode_data_p):
        helpers.address = address
        texts.append(_fingerprint_function(g, set(), fold_named=True))
    assert texts[0].endswith(";module_attrs=[helpers.address=named(numbox.utils.lowlevel.array_data_p)])")
    assert texts[0] != texts[1]


def test_a_dtype_object_is_folded_by_its_layout():
    # numba compiles in the type that a dtype object stands for, as in np.zeros(n, dtype=DT). Two structured dtypes of
    # one size and different fields get different texts, and two equal dtypes get one text. A function that reads a
    # dtype object as a global therefore has a fingerprint, which changes with the layout.
    first = np.dtype([("a", "<i4"), ("b", "<f4")])
    texts = [_canon_value(dt, set()) for dt in (first, np.dtype([("a", "<f4"), ("b", "<i4")]), np.dtype(first.descr))]
    assert texts[0].startswith("npdtype(struct(") and texts[0] != texts[1] and texts[0] == texts[2]
    assert _canon_value(np.dtype("float32"), set()) != _canon_value(np.dtype("float64"), set())

    f = _make("""
        def f(n):
            return np.zeros(n, dtype=DT)
    """, np=np, DT=first)
    before = _fingerprint_function(f, set())
    f.__globals__["DT"] = np.dtype([("a", "<i8")])
    assert _fingerprint_function(f, set()) != before


def _make_scale(k):
    """Return an ``@intrinsic`` that multiplies a ``float64`` by ``k``, as a factory over a constant makes one."""
    @intrinsic
    def scale(typingctx, x):
        def codegen(context, builder, sig, args):
            return builder.fmul(args[0], context.get_constant(numba.types.float64, k))
        return numba.types.float64(numba.types.float64), codegen
    return scale


def _make_cast():
    """Return an ``@intrinsic`` whose definition function reads the numba type ``float64`` as a global."""
    return intrinsic(_make("""
        def f(typingctx, x):
            return float64(x), lambda context, builder, sig, args: args[0]
    """, float64=float64))


def test_an_intrinsic_has_no_fingerprint_and_its_best_effort_text_follows_its_definition():
    # numba compiles a call of an @intrinsic with the code that its definition function generates, and that code can
    # read any object through a module, so an intrinsic has no fingerprint, also one written with numba.types. The
    # best-effort walk folds its definition: two that one factory makes over different constants get different texts,
    # and two that it makes over equal constants get one text, though they are different objects.
    from numba.experimental import structref

    for scale in (_make_scale(2.0), _make_bare_scale(2.0), structref.new):
        with pytest.raises(_Unfingerprintable):
            _canon_value(scale, set())
    reader = """
        def f(x):
            return op(x)
    """
    two, three, two_again = (_make(reader, op=_make_scale(k)) for k in (2.0, 3.0, 2.0))
    texts = [_fingerprint_function_best_effort(g) for g in (two, three, two_again)]
    assert "op=intrinsic-besteffort(" in texts[0] and texts[0] != texts[1] and texts[0] == texts[2]


def test_functions_that_a_factory_makes_over_two_intrinsics_have_different_best_effort_fingerprints():
    # A factory makes the step of a SQLite aggregate over an @intrinsic, and digest names the generated code after the
    # step's fingerprint. The step has none, because the intrinsic has none, so digest uses the best-effort fingerprint,
    # which folds the closure value through the intrinsic's definition.
    def make_step(scale):
        def step(x):
            return scale(x)
        return step

    for make in (_make_scale, _make_bare_scale):
        steps = [make_step(make(k)) for k in (2.0, 3.0)]
        with pytest.raises(_Unfingerprintable):
            _fingerprint_function(steps[0], set())
        assert _fingerprint_function_best_effort(steps[0]) != _fingerprint_function_best_effort(steps[1])


def test_a_module_attribute_that_holds_an_intrinsic_of_the_user_leaves_its_reader_without_a_fingerprint():
    # helpers.scale can hold an intrinsic over 2.0 in one process and over 3.0 in the next, and the intrinsic has no
    # fingerprint. With fold_named the walk raises, as it does for a function without one, so a struct method, a
    # formula or a derive that calls it compiles without a cache. Without fold_named it is left out, as a function is.
    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.scale(x)
    """, helpers=helpers)
    for make in (_make_scale, _make_bare_scale):
        helpers.scale = make(2.0)
        with pytest.raises(_Unfingerprintable):
            _fingerprint_function(f, set(), fold_named=True)
        assert "module_attrs" not in _fingerprint_function(f, set())


def test_an_object_that_passes_for_an_intrinsic_without_a_definition_has_no_fingerprint_and_a_placeholder():
    # isinstance accepts a mock made with spec=_Intrinsic, which has no _defn. The best-effort walk never raises, so it
    # gives such an object the placeholder of its type, and digest names a callback that holds it. The strict walk with
    # fold_named raises _Unfingerprintable for it behind a module attribute, as it does for any value without a
    # fingerprint, and digest leaves it out there as it leaves out an intrinsic.
    from unittest import mock

    from numba.core.extending import _Intrinsic

    from numbox.utils.digest import digest

    spoof = mock.Mock(spec=_Intrinsic)

    def held(x):
        return spoof(x)

    assert "<opaque:Mock>" in _fingerprint_function_best_effort(held)
    assert len(digest("subject", [held])) == 16
    helpers = types.ModuleType("helpers")
    helpers.op = spoof
    f = _make("""
        def f(x):
            return helpers.op(x)
    """, helpers=helpers)
    with pytest.raises(_Unfingerprintable):
        _fingerprint_function(f, set(), fold_named=True)
    assert len(digest("subject", [f])) == 16


def test_the_best_effort_walk_tests_the_type_of_a_value_before_it_reads_an_attribute():
    # The walk meets every global that a function reads. A config dict whose keys read as attributes raises KeyError for
    # a key it lacks, and a module whose __getattr__ imports on access raises ModuleNotFoundError, so reading _defn from
    # either before testing the type made digest and @proxy raise.
    from numbox.utils.digest import digest

    class AttrDict(dict):
        __getattr__ = dict.__getitem__

    lazy = types.ModuleType("lazy_helpers_for_the_walk")

    def lazy_getattr(name):
        raise ModuleNotFoundError(f"No module named 'lazy_helpers_for_the_walk.{name}'")

    lazy.__getattr__ = lazy_getattr
    f = _make("""
        def f(x):
            return CFG.k * x + lazy.offset
    """, CFG=AttrDict(k=3.0), lazy=lazy)
    assert "CFG=" in _fingerprint_function_best_effort(f)
    assert len(digest("subject", [f])) == 16


def test_the_best_effort_walk_folds_each_keyword_only_default():
    # Two functions whose keyword-only default is one of two intrinsics that one factory makes over different constants
    # get two texts. A keyword-only default with a text keeps the text that the dict of the defaults had.
    def make_scale(k):
        @intrinsic
        def scale(typingctx, x):
            def codegen(context, builder, sig, args):
                return builder.fmul(args[0], context.get_constant(float64, k))
            return float64(float64), codegen
        return scale

    texts = []
    for k in (2.0, 3.0):
        f = _make("""
            def f(x, *, sc):
                return sc(x)
        """)
        f.__kwdefaults__ = {"sc": make_scale(k)}
        texts.append(_fingerprint_function_best_effort(f))
    assert texts[0] != texts[1]
    g = _make("""
        def f(x, *, k=2.0, j=1):
            return x * k + j
    """)
    assert f"kwdefaults={_canon_value(g.__kwdefaults__, set())};" in _fingerprint_function_best_effort(g)


def _chain_of_intrinsics(top_constant, length=400):
    """Return a reader of the last of ``length`` intrinsics, each of whose definitions reads the one before it.

    The definition of the last one also holds ``top_constant``, so two chains that differ in it differ only at the end
    that the reader reads, far above the depth at which the walk reaches the recursion limit.
    """
    previous = None
    for i in range(length):
        ns = {"float64": float64, "previous": previous, "K": top_constant if i == length - 1 else 0.0}
        if previous is None:
            exec("def d(typingctx, x):\n    return float64(x), K\n", ns)  # nosec B102 - a fixed test snippet
        else:
            exec("def d(typingctx, x):\n    return float64(previous(x)[0]), K\n", ns)  # nosec B102 - a fixed test snippet
        previous = intrinsic(ns["d"])
    return _make("""
        def f(x):
            return deepest(x)
    """, deepest=previous)


def test_the_best_effort_walk_of_a_long_chain_of_intrinsics_keeps_the_text_above_the_cut():
    # Each definition in the chain reads the next intrinsic as a global, so the walk recurses through all of them. The
    # walk exceeds the recursion limit at a few hundred, and the intrinsic at which it does gets a placeholder. The walk
    # itself returns, the intrinsics above the cut keep their texts, and two chains that differ only at the reader's end
    # get two texts. digest returns as well.
    from numbox.utils.digest import digest

    texts = [_fingerprint_function_best_effort(_chain_of_intrinsics(k)) for k in (2.0, 3.0)]
    for text in texts:
        assert text.count("intrinsic-besteffort(") > 10, text[:300]
        assert "<opaque:_Intrinsic>" in text
    assert texts[0] != texts[1]
    assert len(digest("subject", [_chain_of_intrinsics(2.0)])) == 16


def test_an_intrinsic_of_numbox_keeps_its_caller_cacheable():
    # An intrinsic has no fingerprint, and monotonic_ns is one. numbox supplies it, so a caller that reaches it through
    # its module keeps its fingerprint, as a caller of array_data_p does. When a module of another package holds it in
    # an attribute, it is folded by its name.
    with pytest.raises(_Unfingerprintable):
        _canon_value(clock.monotonic_ns, set(), True)
    f = _make("""
        def f():
            return clock.monotonic_ns()
    """, clock=clock)
    assert "module_attrs" not in _fingerprint_function(f, set(), fold_named=True)

    helpers = types.ModuleType("helpers")
    helpers.now = clock.monotonic_ns
    g = _make("""
        def f():
            return helpers.now()
    """, helpers=helpers)
    assert _fingerprint_function(g, set(), fold_named=True).endswith(
        ";module_attrs=[helpers.now=named(numbox.utils.clock.monotonic_ns)])")


def test_a_proxy_function_whose_alias_is_retired_has_no_fingerprint():
    # A retired alias identifies nothing, so _canon_value walks the wrapper of the @proxy function instead. The wrapper
    # calls an @intrinsic whose definition function holds the alias and reads the function's signature, which has no
    # fingerprint. The walk must raise, and not fold an alias that names another body in another process.
    from numbox.core.bindings.sqlite.value import sqlite3_value_int64

    alias = sqlite3_value_int64._numbox_proxy_alias
    assert alias not in _ABSENT_ALIASES
    _ABSENT_ALIASES.add(alias)
    try:
        for fold_named in (False, True):
            with pytest.raises(_Unfingerprintable):
                _canon_value(sqlite3_value_int64, set(), fold_named)
    finally:
        _ABSENT_ALIASES.discard(alias)


def test_a_subarray_dtype_object_is_folded_by_its_elements_and_its_shape():
    # Every subarray dtype of one size has the str |V<itemsize>, and numba types np.dtype((np.float64, (2,))) and
    # np.dtype((np.int32, (4,))) as two nested arrays. Both get their own text, as a global and as a module attribute.
    two_f8, four_i4 = np.dtype((np.float64, (2,))), np.dtype((np.int32, (4,)))
    assert two_f8.str == four_i4.str
    assert _canon_value(two_f8, set()) != _canon_value(four_i4, set())
    assert _canon_value(two_f8, set()) == _canon_value(np.dtype((np.float64, (2,))), set())
    # The text holds both parts: one element type in two shapes, and one shape of two element types.
    assert _canon_value(two_f8, set()) != _canon_value(np.dtype((np.float64, (3,))), set())
    assert _canon_value(two_f8, set()) != _canon_value(np.dtype((np.int64, (2,))), set())
    # Two structured elements of one size with different fields give two texts as well.
    rec_a = np.dtype([("a", np.int32), ("b", np.float32)])
    rec_b = np.dtype([("a", np.float32), ("b", np.int32)])
    assert rec_a.str == rec_b.str
    assert _canon_value(np.dtype((rec_a, (2,))), set()) != _canon_value(np.dtype((rec_b, (2,))), set())
    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.DT
    """, helpers=helpers)
    texts = []
    for dt in (two_f8, four_i4):
        helpers.DT = dt
        texts.append(_fingerprint_function(f, set(), fold_named=True))
    assert texts[0] != texts[1]


def test_the_best_effort_text_of_an_intrinsic_holds_the_names_of_its_parameters():
    # numba binds a keyword call of an intrinsic by the names of its definition's parameters, so sub(a=x, b=y) computes
    # another thing when the definition names them (typingctx, b, a). The two definitions below have the same bytecode,
    # constants and names.
    texts = []
    for params in ("a, b", "b, a"):
        texts.append(_fingerprint_function_best_effort(_bare_reader(f"""
            def f(typingctx, {params}):
                return float64(float64, float64), None
        """)))
    assert texts[0] != texts[1]


def test_an_intrinsic_whose_definition_is_not_a_function_has_no_fingerprint():
    # numba accepts a callable object as the definition of an intrinsic. The object has no code to read, so the
    # intrinsic has no text, and the best-effort fingerprint of a function that reads it puts a placeholder there.
    class Definition:
        def __call__(self, typingctx, x):
            return numba.types.float64(numba.types.float64), None

    ident = intrinsic(Definition())
    with pytest.raises(_Unfingerprintable):
        _canon_value(ident, set())
    f = _make("""
        def f(x):
            return ident(x)
    """, ident=ident)
    assert "=<opaque:_Intrinsic>" in _fingerprint_function_best_effort(f)


def _make_bare_scale(k):
    """Return an ``@intrinsic`` over ``k`` whose definition function reads the numba type ``float64`` by name."""
    make = _make("""
        def f(k):
            @intrinsic
            def scale(typingctx, x):
                def codegen(context, builder, sig, args):
                    return builder.fmul(args[0], context.get_constant(float64, k))
                return float64(float64), codegen
            return scale
    """, intrinsic=intrinsic, float64=float64)
    return make(k)


def test_the_best_effort_fingerprint_walks_an_intrinsic_of_the_user():
    # An intrinsic has no fingerprint. numba's documentation writes one with numba types imported by name, which have
    # none either. The best-effort fingerprint walks its definition function with placeholders, so two intrinsics that
    # one factory makes over 2.0 and 3.0 still give a step that reads one of them two fingerprints.
    def make_step(scale):
        def step(x):
            return scale(x)
        return step

    with pytest.raises(_Unfingerprintable):
        _canon_value(_make_bare_scale(2.0), set())
    opaque = []
    texts = [_fingerprint_function_best_effort(make_step(_make_bare_scale(k)), opaque if k == 2.0 else None)
             for k in (2.0, 3.0, 2.0)]
    assert "scale=intrinsic-besteffort(" in texts[0]
    assert texts[0] != texts[1] and texts[0] == texts[2]
    # The values without a text are float64, inside the definition, and the intrinsics themselves, which
    # _opaque_values_match compares as other such values.
    assert {type(value).__name__ for value in opaque} == {"Float", "_Intrinsic"}

    # An intrinsic of numbox gets the placeholder, so the alias of a binding does not depend on numbox's code.
    from numbox.core.bindings.call import _call_lib_func

    reader = _make("""
        def f(x):
            return _call_lib_func("cos", (x,))
    """, _call_lib_func=_call_lib_func)
    opaque = []
    assert "_call_lib_func=<opaque:_Intrinsic>" in _fingerprint_function_best_effort(reader, opaque)
    assert opaque == [_call_lib_func]

    # The intrinsic that @proxy generates for the wrapper of a function of the user's holds the body's alias, so it
    # gets the placeholder as well.
    from numbox.core.proxy.proxy import proxy

    @proxy(float64(float64), jit_options={"cache": False})
    def plus_one(x):
        return x + 1.0

    wrapper_text = _fingerprint_function_best_effort(plus_one.py_func)
    assert "=<opaque:_Intrinsic>" in wrapper_text and "intrinsic-besteffort" not in wrapper_text

    # An intrinsic whose definition reads the intrinsic itself is walked once; the second time it is a reference.
    defn = _make("""
        def f(typingctx, x):
            return float64(x), SELF
    """, float64=float64)
    defn.__globals__["SELF"] = recursive = intrinsic(defn)
    text = _fingerprint_function_best_effort(make_step(recursive))
    assert text.count("intrinsic-besteffort(") == 1 and "SELF=intrinsic-ref(0)" in text


def _bare_reader(defn_src, decorate=intrinsic):
    """Return a function that calls an ``@intrinsic`` made from ``defn_src``, which reads ``float64`` by name."""
    defn = _make(defn_src, float64=float64)
    return _make("""
        def f(x):
            return op(x)
    """, op=decorate(defn))


def test_the_best_effort_text_folds_each_default_value():
    # A factory can bind the intrinsic as a default, as in "def step(x, sc=scale)". The defaults tuple then has no text,
    # so the walk folds each default on its own, and the two steps over 2.0 and 3.0 get two texts. A tuple of defaults
    # that all have a text gives the text of the tuple, as before.
    def make_step(scale):
        def step(x, sc=scale):
            return float64(sc(x))
        return step

    texts = [_fingerprint_function_best_effort(make_step(_make_bare_scale(k))) for k in (2.0, 3.0)]
    assert "defaults=tuple[intrinsic-besteffort(" in texts[0] and texts[0] != texts[1]
    plain = _make("""
        def f(x, k=2.0, name="a"):
            return float64(x * k)
    """, float64=float64)
    assert "defaults=tuple[2.0,'a'];" in _fingerprint_function_best_effort(plain)


def test_the_best_effort_text_of_an_intrinsic_holds_its_options():
    # prefer_literal changes how numba types a call of an intrinsic, so two intrinsics made from one definition with
    # and without it get two best-effort texts, and a callback that calls either gets a digest of its own.
    src = """
        def f(typingctx, x):
            return float64(float64), None
    """
    plain = _fingerprint_function_best_effort(_bare_reader(src))
    literal = _fingerprint_function_best_effort(_bare_reader(src, intrinsic(prefer_literal=True)))
    assert "prefer_literal':False" in plain and plain != literal


def test_the_best_effort_text_holds_the_names_that_a_nested_function_uses():
    # The codegen function of an intrinsic is a nested function. Two codegens that call builder.fadd and builder.fsub
    # have the same bytecode and constants and differ in their names, and numba compiles them to different code. A
    # lambda in a function without a fingerprint is folded the same way.
    texts = []
    for op in ("fadd", "fsub"):
        texts.append(_fingerprint_function_best_effort(_bare_reader(f"""
            def f(typingctx, x):
                def codegen(context, builder, sig, args):
                    return builder.{op}(args[0], args[0])
                return float64(float64), codegen
        """)))
    assert texts[0] != texts[1]
    lambdas = [_make(f"""
        def f(a):
            return float64((lambda: a.{attr})())
    """, float64=float64) for attr in ("x", "y")]
    assert _fingerprint_function_best_effort(lambdas[0]) != _fingerprint_function_best_effort(lambdas[1])


def test_the_best_effort_text_of_nested_code_holds_its_argument_counts_flags_and_constants():
    # Each pair of lambdas has the same bytecode, constants and names and differs in one part only: the flags (*a
    # against **a), the number of arguments, or the number of keyword-only arguments. numba binds a call of the two
    # differently. Two codegens that multiply by 2.0 and by 3.0 differ only in a constant of the nested code.
    for one, other in (("lambda *a: a", "lambda **a: a"), ("lambda a, b: a", "lambda a: a"),
                       ("lambda a, *, b: a", "lambda a: a")):
        texts = [_fingerprint_function_best_effort(_make(f"""
            def f(x):
                g = {lam}
                return float64(x)
        """, float64=float64)) for lam in (one, other)]
        assert texts[0] != texts[1], (one, other)
    literals = [_fingerprint_function_best_effort(_bare_reader(f"""
        def f(typingctx, x):
            def codegen(context, builder, sig, args):
                return builder.fmul(args[0], context.get_constant(float64, {literal}))
            return float64(float64), codegen
    """)) for literal in ("2.0", "3.0")]
    assert literals[0] != literals[1]


def test_a_reference_in_the_best_effort_text_names_the_intrinsic_that_it_stands_for():
    # a(x) + b(x) + c(x) computes 7 x with c = a and 8 x with c = b. The third intrinsic is a reference in both, and its
    # number is the only part of the text that tells the two apart.
    two, three = _make_bare_scale(2.0), _make_bare_scale(3.0)
    src = """
        def f(x):
            return a(x) + b(x) + c(x)
    """
    texts = [_fingerprint_function_best_effort(_make(src, a=two, b=three, c=c)) for c in (two, three)]
    assert "c=intrinsic-ref(0)" in texts[0] and "c=intrinsic-ref(1)" in texts[1]


def test_the_best_effort_walk_enters_the_definition_of_each_intrinsic_of_the_user_once(monkeypatch):
    # Six intrinsics whose definitions read all six would make a walk that follows every path enter 1957 definitions.
    # The walk folds an intrinsic that it has met as a reference, so it enters each definition once and holds each text
    # once.
    names = [f"I{i}" for i in range(6)]
    ns = {"float64": float64}
    exec("".join(f"def d{i}(typingctx, x):\n    return float64(x), ({', '.join(names)},)\n"
                 for i in range(6)), ns)  # nosec B102 - a fixed test snippet
    for i, name in enumerate(names):
        ns[name] = intrinsic(ns[f"d{i}"])
    reader = _make("""
        def f(x):
            return (I0, I1, I2, I3, I4, I5)
    """, **{name: ns[name] for name in names})

    entered = collections.Counter()
    original = fingerprint._fingerprint_function_best_effort

    def counting(func, opaque=None, walked=None):
        entered[func.__name__] += 1
        return original(func, opaque, walked)

    monkeypatch.setattr(fingerprint, "_fingerprint_function_best_effort", counting)
    text = counting(reader)
    assert entered == collections.Counter({"f": 1, **{f"d{i}": 1 for i in range(6)}})
    assert text.count("intrinsic-besteffort(") == 6 and "I0=intrinsic-ref(0)" in text


def test_the_definition_of_an_intrinsic_of_numba_or_numbox_is_never_walked(monkeypatch):
    # 167 of numbox's 173 bindings read _call_lib_func, whose definition takes about a millisecond to walk. An intrinsic
    # has no fingerprint, so neither the strict nor the best-effort fingerprint walks the definition of one of numba or
    # numbox: the strict one raises at once and the best-effort one puts the placeholder in its place.
    from numba.experimental import structref

    from numbox.core.bindings.call import _call_lib_func

    walked = []
    strict, best_effort = fingerprint._fingerprint_function, fingerprint._fingerprint_function_best_effort

    def counting_strict(func, seen, fold_named=False):
        walked.append(func)
        return strict(func, seen, fold_named)

    def counting_best_effort(func, opaque=None, walked_intrinsics=None):
        walked.append(func)
        return best_effort(func, opaque, walked_intrinsics)

    monkeypatch.setattr(fingerprint, "_fingerprint_function", counting_strict)
    monkeypatch.setattr(fingerprint, "_fingerprint_function_best_effort", counting_best_effort)
    reader = _make("""
        def f(x):
            return _call_lib_func("cos", (x,)) + new(x)
    """, _call_lib_func=_call_lib_func, new=structref.new)
    for value in (_call_lib_func, structref.new):
        with pytest.raises(_Unfingerprintable):
            _canon_value(value, set(), True)
    text = fingerprint._fingerprint_function_best_effort(reader)
    assert "_call_lib_func=<opaque:_Intrinsic>" in text and "new=<opaque:_Intrinsic>" in text
    assert _call_lib_func._defn not in walked and structref.new._defn not in walked


def test_a_module_attribute_that_holds_a_class_of_the_user_is_left_out():
    # The fingerprint has no text for a class of the user or for an enum. Such an attribute is left out, the caller
    # keeps its fingerprint, and a change of the value does not change it.
    class Mine:
        pass

    class Color(enum.Enum):
        RED = 1

    # A function of NumPy, numba, numbox or the standard library that no module holds under its own name has no name.
    # numba does not compile it from its body, so it is left out as well. The body of such a function often reads a
    # value without a fingerprint, and folding the body would then leave the caller without a cache.
    unnamed = types.FunctionType(heapq.nlargest.__code__, heapq.nlargest.__globals__)
    assert unnamed.__module__ == "heapq" and _named_value_text(unnamed) is None
    with pytest.raises(_Unfingerprintable):
        _canon_value(unnamed, set(), True)

    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.op(x)
    """, helpers=helpers)
    for op in (Mine, Color, unnamed):
        helpers.op = op
        assert "module_attrs" not in _fingerprint_function(f, set(), fold_named=True)


def test_a_module_attribute_that_holds_a_numba_type_or_a_dtype_object_is_folded_by_the_identity_of_the_type():
    # helpers.T can hold numba's float32 in one process and float64 in the next, or two dtype objects. numba compiles
    # in the type that the value stands for, and the code of f names only the attribute.
    structured = (np.dtype([("a", "<i4"), ("b", "<f4")]), np.dtype([("a", "<f4"), ("b", "<i4")]))
    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.T(x)
    """, helpers=helpers)

    texts, plain = [], []
    for held in (float32, float64, np.dtype("float32"), np.dtype("float64"), *structured):
        helpers.T = held
        texts.append(_fingerprint_function(f, set(), fold_named=True))
        plain.append(_fingerprint_function(f, set()))
    assert len(set(texts)) == 6
    assert "helpers.T=numbatype(" in texts[0] and "helpers.T=npdtype(<f4)" in texts[2]
    # The text that @proxy and cres hash into an alias leaves the attribute out.
    assert len(set(plain)) == 1 and "module_attrs" not in plain[0]

    # numba holds its own types under the same names in every process, so they add nothing to the fingerprint of a
    # function that reaches them through numba. A module of the user that has the same attribute is folded.
    through_numba = _make("""
        def f(x):
            return nb.float32(x) + nb.types.float64(x)
    """, nb=numba)
    assert "module_attrs" not in _fingerprint_function(through_numba, set(), fold_named=True)
    helpers.float32, helpers.types = float32, numba.types
    through_helpers = _make("""
        def f(x):
            return nb.float32(x) + nb.types.float64(x)
    """, nb=helpers)
    assert "nb.float32=numbatype(" in _fingerprint_function(through_helpers, set(), fold_named=True)
    # A module whose __dict__ holds no __name__ belongs to no package, so a type that it holds is folded as well.
    helpers.inner = types.ModuleType("inner")
    helpers.inner.float32 = float32
    del helpers.inner.__dict__["__name__"]
    through_inner = _make("""
        def f(x):
            return nb.inner.float32(x)
    """, nb=helpers)
    assert "nb.inner.float32=numbatype(" in _fingerprint_function(through_inner, set(), fold_named=True)


def test_a_numba_type_whose_text_names_an_address_leaves_the_function_without_a_fingerprint():
    # numba's text for a deferred type, as for the type of a jitclass instance, holds an address of this process. A
    # digest of that text would name code that no other process finds, so the function has no fingerprint, and its
    # caller compiles without a cache.
    unstable = deferred_type()
    assert "DeferredType#" in str(unstable)
    helpers = types.ModuleType("helpers")
    helpers.T = unstable
    f = _make("""
        def f(x):
            return helpers.T(x)
    """, helpers=helpers)
    with pytest.raises(_Unfingerprintable):
        _fingerprint_function(f, set(), fold_named=True)
    assert "module_attrs" not in _fingerprint_function(f, set())

    @njit(locals={"t": unstable})
    def with_locals(x):
        t = x
        return t

    # The type of a dispatcher is another such type: numba's text for it holds the address of the Python function.
    declared = njit(float64(numba.typeof(with_locals)))(lambda g: 1.0)
    assert " at 0x" in str(declared.nopython_signatures[0])
    for dispatcher in (with_locals, declared):
        with pytest.raises(_Unfingerprintable):
            _canon_value(dispatcher, set(), True)
        assert _canon_value(dispatcher, set()).startswith("dispatcher(")


@pytest.mark.parametrize("src", [
    "def f(y, m=helpers):\n    return y * m.SCALE + m.step(y)\n",
    "def f(y, *, m=helpers):\n    return y * m.SCALE + m.step(y)\n",
    "def f(y, /, z=1.0, m=helpers):\n    return y * m.SCALE + m.step(z)\n",
])
def test_the_attributes_of_a_module_taken_as_a_default_argument_are_folded(src):
    # The function reads SCALE and step through the parameter m, whose default value is the module. The module is
    # neither a global nor a closure variable of the function.
    helpers = types.ModuleType("helpers")
    helpers.step = _make("def f(x):\n    return 2.0 * x\n")
    f = _make(src, helpers=helpers)

    texts, plain = [], []
    for scale, constant in ((2.0, "2.0"), (3.0, "2.0"), (3.0, "3.0")):
        helpers.SCALE = scale
        helpers.step = _make(f"def f(x):\n    return {constant} * x\n")
        texts.append(_fingerprint_function(f, set(), fold_named=True))
        plain.append(_fingerprint_function(f, set()))
    assert len(set(texts)) == 3
    assert "m.SCALE=2.0" in texts[0] and "m.step=function(" in texts[0]
    # The text that @proxy and cres hash into an alias does not walk the module.
    assert len(set(plain)) == 1 and "module_attrs" not in plain[0]


def test_a_parameter_hides_a_global_of_the_same_name_in_the_walk():
    # The parameter cfg hides the global cfg inside the function, so the module that the default holds is the one
    # that the function reads, and the walk folds the attributes of that module. The function also reads an
    # attribute named cfg, which puts the name among those that the walk looks up in the function's globals.
    shown, hidden = types.ModuleType("shown"), types.ModuleType("hidden")
    shown.SCALE, hidden.SCALE = 5.0, 2.0
    ns = {"cfg": hidden, "shown": shown}
    exec("def f(y, cfg=shown):\n    return y.cfg * cfg.SCALE\n\n\ndef g(y):\n    return y * cfg.SCALE\n", ns)  # nosec B102
    text = _fingerprint_function(ns["f"], set(), fold_named=True)
    assert text.endswith(";module_attrs=[cfg.SCALE=5.0])")
    hidden.SCALE = 3.0
    assert _fingerprint_function(ns["f"], set(), fold_named=True) == text
    # g has no such parameter and reads the global.
    assert _fingerprint_function(ns["g"], set(), fold_named=True).endswith(";module_attrs=[cfg.SCALE=3.0])")


def test_the_text_of_a_dispatcher_holds_its_locals_and_its_declared_signatures():
    # The locals option and a declared signature change the code of a dispatcher, and of a caller that numba links it
    # into, without changing the Python function. The text under fold_named therefore holds them.
    def body(x):
        t = x / 3.0
        return t * 1.0

    with_locals = [njit(locals={"t": ty})(body) for ty in (float32, float64)]
    texts = [_canon_value(disp, set(), True) for disp in with_locals]
    assert texts[0] != texts[1] and ";locals=[t:" in texts[0]

    declared = [njit(sig)(body) for sig in ("float64(float32)", "float64(float64)")]
    texts = [_canon_value(disp, set(), True) for disp in declared]
    assert texts[0] != texts[1] and ";signatures=[" in texts[0] and ";locals=" not in texts[0]

    loops = [vectorize([sig])(body) for sig in ("float32(float64)", "float64(float64)")]
    texts = [_canon_value(ufunc, set(), True) for ufunc in loops]
    assert texts[0].endswith(";loops=[d->f])") and texts[1].endswith(";loops=[d->d])")

    # A tuple of dispatchers that a module attribute holds is data, and the text of each dispatcher in it holds the
    # declared types as well.
    helpers = types.ModuleType("helpers")
    f = _make("""
        def f(x):
            return helpers.steps[0](x)
    """, helpers=helpers)
    texts = []
    for disp in with_locals:
        helpers.steps = (disp,)
        texts.append(_fingerprint_function(f, set(), fold_named=True))
    assert texts[0] != texts[1] and ";locals=[t:" in texts[0]

    # The text that @proxy and cres hash into an alias holds none of them, so no alias changes.
    for group in (with_locals, declared, loops):
        assert len({_canon_value(value, set()) for value in group}) == 1


def test_the_text_of_a_dispatcher_that_compiles_on_demand_does_not_depend_on_the_calls_made():
    # A dispatcher or a @vectorize function without declared signatures compiles a signature at each call with new
    # argument types. Those signatures differ from process to process, so the text holds none of them, and it is the
    # text that the dispatcher has without fold_named.
    def body(x):
        return x / 3.0

    for lazy in (njit(body), vectorize(body)):
        before = _canon_value(lazy, set(), True)
        assert before == _canon_value(lazy, set())
        lazy(1.0)
        lazy(np.float32(1.0))
        assert _canon_value(lazy, set(), True) == before


def test_proxy_binding_callback_fingerprints_cleanly_and_distinctly():
    # A @proxy function is folded as proxy(<alias>), so the walk does not reach the @intrinsic that its wrapper calls,
    # which has no fingerprint. A callback that calls a binding therefore has a fingerprint, and callbacks that call
    # different bindings have different fingerprints.
    from numbox.core.bindings.sqlite.value import sqlite3_value_int64, sqlite3_value_double

    def cb_int(v):
        return sqlite3_value_int64(v)

    def cb_dbl(v):
        return sqlite3_value_double(v)

    fp_int = _fingerprint_function(cb_int, set())   # must not raise
    fp_dbl = _fingerprint_function(cb_dbl, set())
    assert "proxy(numbox_pxy_" in fp_int
    assert fp_int != fp_dbl


def test_proxy_binding_canon_does_not_recurse_into_intrinsic():
    from numbox.core.bindings.sqlite.value import sqlite3_value_int64

    canon = _canon_value(sqlite3_value_int64, set())
    assert canon.startswith("proxy(numbox_pxy_sqlite3_value_int64_")


def test_unfingerprintable_value_still_raises_for_strict_walker():
    # _fingerprint_function raises for a value without a fingerprint, so the callers that compile such code without a
    # cache detect it. Only _fingerprint_function_best_effort puts a placeholder in place of the value.
    class Opaque:
        pass

    def cb():
        return Opaque()

    # cb2 captures an instance of Opaque in its closure, and the fingerprint folds closure values.
    marker = Opaque()

    def cb2():
        return marker

    try:
        _fingerprint_function(cb2, set())
    except _Unfingerprintable:
        pass
    else:
        raise AssertionError("expected _Unfingerprintable for an opaque closure value")


def test_loaded_global_names_distinguishes_globals_from_attributes():
    """``_loaded_global_names`` returns the names that a function reads as globals, also in a comprehension.

    ``co_names``, which ``_referenced_global_names`` reads, also holds the name of an attribute that the function reads,
    and ``_loaded_global_names`` leaves that name out.
    """
    attr = _make("def f(a):\n    return a.c1\n")
    assert _referenced_global_names(attr.__code__) == {"c1"}, "co_names should pool the attribute name"
    assert _loaded_global_names(attr.__code__) == set(), "an attribute access names no global"

    glob = _make("def f(x):\n    return x + K\n", K=5)
    assert "K" in _loaded_global_names(glob.__code__)

    nested = _make("def f(xs):\n    return [y + K for y in xs]\n", K=5)
    assert "K" in _loaded_global_names(nested.__code__), "a global read inside a comprehension counts"


def test_fingerprint_ignores_a_global_shadowed_by_an_attribute_name():
    """The fingerprint of a function that reads the attribute ``a.c1`` does not fold a module global named ``c1``.

    Folding it would change the fingerprint when only the global changes, and a global without a fingerprint would make
    the fingerprint fail.
    """
    canon = _make("def f(a):\n    return a.c1\n", c1=3)
    assert "c1=" not in _fingerprint_function(canon, set()), "folded a global the body only accesses as an attribute"

    uncanon = _make("def f(a):\n    return a.c1\n", c1=np.array([object()], dtype=object))
    # The global c1 has no fingerprint, and the fingerprint of f must not raise.
    fp = _fingerprint_function(uncanon, set())
    assert "c1=" not in fp

    # The attribute's name is still in co_names, which the fingerprint folds, so a function that reads another
    # attribute has another fingerprint.
    other = _make("def f(a):\n    return a.c2\n", c1=3)
    assert _fingerprint_function(canon, set()) != _fingerprint_function(other, set())


def test_fingerprint_still_folds_a_genuine_global_read():
    """The fingerprint folds the value of a global that the function reads, so a new value gives a new fingerprint."""
    f = _make("def f(x):\n    return x * K\n", K=2)
    assert "K=" in _fingerprint_function(f, set())
    g = _make("def f(x):\n    return x * K\n", K=9)
    assert _fingerprint_function(f, set()) != _fingerprint_function(g, set())

    # A global that the function reads and that has no fingerprint makes _fingerprint_function raise.
    bad = _make("def f(x):\n    return x * K\n", K=np.array([object()], dtype=object))
    try:
        _fingerprint_function(bad, set())
    except _Unfingerprintable:
        pass
    else:
        raise AssertionError("an un-canonicalizable genuine global read should still be _Unfingerprintable")


def test_module_attribute_read_still_rekeys(tmp_path):
    """The fingerprint folds a module attribute that the function reads, such as ``cfg.SCALE``, not one it never reads.

    The walk of module attributes takes every name in ``co_names``, so it receives the attribute names that
    ``_loaded_global_names`` leaves out.
    """
    import types

    cfg = types.ModuleType("cfgfake")
    cfg.SCALE = 2.0
    cfg.OFFSET = 9.0
    f = _make("def f(x):\n    return x * cfg.SCALE\n", cfg=cfg)

    before = _fingerprint_function(f, set())
    cfg.OFFSET = 99.0
    assert _fingerprint_function(f, set()) == before, "an unread module attribute must not re-key"
    cfg.SCALE = 3.0
    assert _fingerprint_function(f, set()) != before, "a read module attribute must re-key"


def test_an_external_function_is_canonicalized_by_symbol_and_signature():
    """``_canon_value`` gives an ``ExternalFunction`` a text of its symbol and signature, so two C symbols differ.

    The text is numba's own key for the type, and both parts mean the same thing in the next process. Without that
    case, the best-effort fingerprint that names a ``@proxy`` body would give every ``ExternalFunction`` one
    placeholder.
    """
    from numba.core.types import ExternalFunction, float64

    sin = _canon_value(ExternalFunction("sin", float64(float64)), set())
    cos = _canon_value(ExternalFunction("cos", float64(float64)), set())
    assert sin != cos, "two C symbols collapsed onto one canonical form"
    assert "sin" in sin and "cos" in cos
    assert _canon_value(ExternalFunction("sin", float64(float64)), set()) == sin, (
        "the canonical form has to be the same for an equal value, or it is not a fingerprint"
    )


def test_a_library_bound_ctypes_pointer_is_canonicalized_by_library_and_symbol():
    """A ctypes pointer taken from a loaded library has a text of the library's name and the symbol.

    Two symbols of one library have different texts, and two handles opened with one string give one text. The text
    holds the string that opened the library, so one library opened under a soname and under a path gives two texts.
    The difference can cause a recompile but never a wrong result.
    """
    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take two symbols from")
    floor = _canon_value(libm.floor, set())
    ceil = _canon_value(libm.ceil, set())
    assert floor != ceil, "two C functions from one library collapsed onto one canonical form"
    assert _canon_value(open_libm().floor, set()) == floor, (
        "the canonical form followed the handle object rather than what it names"
    )


def test_a_ctypes_pointer_built_from_an_address_stays_unfingerprintable():
    """``_canon_value`` raises ``_Unfingerprintable`` from its ctypes case for a pointer built from a raw address.

    Only the address separates two such pointers, and ASLR changes it in every process, so the caller must decide how
    to treat the pointer. The final ``raise`` of ``_canon_value`` gives the type name as its message, so the test
    requires another message, the one from the ctypes case.
    """
    import ctypes

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take a symbol address from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
    fp = proto(ctypes.cast(libm.floor, ctypes.c_void_p).value)
    with pytest.raises(_Unfingerprintable) as refusal:
        _canon_value(fp, set())
    assert str(refusal.value) != type(fp).__name__, (
        "the refusal is the walker's terminal type-name raise, so the ctypes branch was never "
        "reached and this passes against a walker that cannot fingerprint any pointer at all"
    )


def test_a_proxied_binding_and_a_cres_are_not_canonicalized_alike_as_function_values():
    """The ``.as_func`` of a ``@proxy`` function and a ``cres`` value with the same body have different texts.

    A function value holds only an address and a mangled name with a per-process counter, so numbox tags the two kinds
    of value that it creates. ``_canon_value`` folds the alias of the ``@proxy`` function for the first, and the body's
    fingerprint with the signature for the second. If the two had one text, two bodies that differ only in which of the
    values they capture would compute one alias. The collision would retire it, and neither body would be cached across
    processes.
    """
    from numba import float64

    from numbox.core.proxy.proxy import proxy
    from numbox.utils.highlevel import cres

    @proxy(float64(float64))
    def binding(x):
        return x + 1.0

    @cres(float64(float64))
    def compiled(x):
        return x + 1.0

    as_func = _canon_value(binding.as_func, set())
    derive = _canon_value(compiled, set())
    assert as_func != derive, "a proxied binding and a cres folded onto one canonical form"
    assert binding._numbox_proxy_alias in as_func, (
        "the binding's function value is not keyed by the alias that identifies its body"
    )
    assert _canon_value(binding.as_func, set()) == as_func, (
        "the canonical form has to be the same for an equal value, or it is not a fingerprint"
    )
    assert _canon_value(binding, set()) != as_func, (
        "the dispatcher and its function value fold alike, though they call differently"
    )


def test_a_function_value_numbox_did_not_mint_stays_unfingerprintable():
    """``_canon_value`` raises ``_Unfingerprintable`` for an untagged ``CompileResultWAP`` built with numba directly.

    Its compile result identifies the body only by an address and by a mangled name whose abi tag is a per-process
    counter. The test also checks that a ``cres`` value has a text, so the refusal comes from the missing tag and not
    from the type.
    """
    from numba import float64, njit
    from numba.core.types.function_type import CompileResultWAP

    from numbox.utils.highlevel import cres

    compiled = njit(float64(float64))(lambda x: x + 1.0)
    foreign = CompileResultWAP(compiled.get_compile_result(compiled.nopython_signatures[0]))

    @cres(float64(float64))
    def minted(x):
        return x + 1.0

    assert _canon_value(minted, set()), (
        "numbox's own function value is unfingerprintable too, so the refusal below is not "
        "about the missing tag and rules nothing out"
    )
    with pytest.raises(_Unfingerprintable):
        _canon_value(foreign, set())


def test_two_python_callbacks_sharing_a_name_are_not_canonicalized_alike():
    """``_canon_value`` raises ``_Unfingerprintable`` for a ctypes callback built from a Python function.

    For such a callback, ``_objects["0"]`` holds a ``CThunkObject``, which has no ``_name``, instead of a library. A
    text that folded the missing name would be the same for two callbacks with one ``__name__``, such as two callbacks
    to which a decorator copies the same name.
    """
    import ctypes

    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
    one = proto(lambda x: x + 1.0)
    two = proto(lambda x: x + 2.0)
    one.__name__ = two.__name__ = "cb"

    assert one(1.0) != two(1.0), "the two callbacks are not distinguishable to begin with"
    for cb in (one, two):
        with pytest.raises(_Unfingerprintable):
            _canon_value(cb, set())


def test_unset_and_empty_argtypes_are_canonicalized_apart():
    """A ctypes pointer whose ``argtypes`` is unset and one whose ``argtypes`` is empty have different texts.

    numba refuses a call through the first and accepts a call through the second, so one text for both would pair a
    caller with a body that it cannot call. A text that read ``argtypes or ()`` would give both pointers one text,
    because they differ in nothing else.
    """
    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take a symbol from")
    # The pointers come from two handles, because CDLL.__getattr__ stores the pointer that it builds on the handle.
    # One handle would return one object twice, and setting argtypes on it would set it for both. open_libm opens a new
    # handle on each call, which test_open_libm_hands_back_a_usable_handle in test/core/test_proxy.py checks.
    unset = open_libm().floor
    assert unset.argtypes is None, "ctypes stopped leaving argtypes unset by default"
    empty = open_libm().floor
    assert unset is not empty, "both pointers came off one handle, so setting argtypes sets both"
    empty.argtypes = ()
    assert _canon_value(unset, set()) != _canon_value(empty, set()), (
        "an unset argtypes and an empty one collapsed onto one canonical form"
    )


def test_two_prototypes_over_one_address_are_not_read_as_one_value():
    """``_opaque_values_match`` treats two ctypes pointers over one address as one value only if their prototypes match.

    numba types and lowers a call from the pointer's ``restype``, ``argtypes`` and calling convention, so two prototypes
    over one symbol describe two calls. If the two pointers counted as one value, the second body would receive the
    alias of the first, and no collision would be reported.
    """
    import ctypes

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take a symbol address from")
    # A raw-address pointer reaches _opaque_values_match because _ctypes_func_key finds no library and symbol for it.
    # The three pointers that must not match as_double differ from it in the return type, in the arguments and in both,
    # so a comparison of only one part of the prototype fails the test. again must match as_double, or every module
    # reload and every factory called twice would count as a collision and retire the alias. again is a second object,
    # because the comparison accepts an object compared with itself before it reaches the ctypes case.
    addr = ctypes.cast(libm.floor, ctypes.c_void_p).value
    as_double = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(addr)
    other_restype = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_double)(addr)
    other_argtypes = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(addr)
    as_float = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float)(addr)
    again = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(addr)

    assert as_double is not again, (
        "one object compared with itself is answered by the identity test, not by the pointer branch"
    )
    for other in (other_restype, other_argtypes, as_float, again):
        assert ctypes.cast(other, ctypes.c_void_p).value == addr, (
            "the pointers do not share an address, so nothing here reaches the comparison it is about"
        )
    for differing, part in ((other_restype, "return type"), (other_argtypes, "arguments"),
                            (as_float, "return type and arguments")):
        assert not _opaque_values_match([as_double], [differing]), (
            f"two prototypes differing in the {part} over one address read as one body"
        )
    assert _opaque_values_match([as_double], [again]), (
        "one prototype over one address read as two bodies, so a module reload and a factory "
        "called twice both turn into collisions"
    )


def test_a_retired_alias_is_not_folded_into_a_body_that_captured_it():
    """A ``@proxy`` function and its ``.as_func`` have no text once a collision has retired their alias.

    A process gives the shared alias to the colliding body that it builds first. A body that captured one of them and
    folded the alias could therefore mean the other body in another process. The comment above ``_ABSENT_ALIASES`` in
    ``numbox/utils/fingerprint.py`` describes how the walk treats such a body.
    """
    import ctypes

    from numba import float64

    from numbox.core.proxy.proxy import AliasCollisionWarning, proxy, _ABSENT_ALIASES

    libm = open_libm()
    if libm is None:
        pytest.skip("No math library discoverable to take two distinct symbol addresses from")
    proto = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

    # Each body captures a ctypes pointer built from a raw address, which has no fingerprint, so the two bodies compute
    # one alias and collide.
    def make(symbol):
        fp = proto(ctypes.cast(getattr(libm, symbol), ctypes.c_void_p).value)

        @proxy(float64(float64))
        def body(x):
            return fp(x)
        return body

    winner = make("floor")
    with pytest.warns(AliasCollisionWarning):
        loser = make("ceil")

    assert winner._numbox_proxy_alias != loser._numbox_proxy_alias, (
        "the two bodies did not collide, so nothing here is about a retired alias"
    )
    for binding, which in ((winner, "shared"), (loser, "minted")):
        alias = binding._numbox_proxy_alias
        assert alias in _ABSENT_ALIASES, (
            f"the {which} name was not retired, so this asserts nothing about folding one"
        )
        with pytest.raises(_Unfingerprintable):
            _canon_value(binding, set())
        with pytest.raises(_Unfingerprintable):
            _canon_value(binding.as_func, set())


def test_a_live_alias_is_still_folded():
    """A ``@proxy`` function with an alias that is not retired, and its ``.as_func``, are folded by that alias.

    The refusal in the test above depends on the retirement alone. If every proxied function had no text, no body that
    captures one would have a fingerprint, so code generated for such a body would compile without a cache or be named
    by the weaker fingerprint.
    """
    from numba import float64

    from numbox.core.proxy.proxy import proxy

    @proxy(float64(float64))
    def uncollided(x):
        return x * 3.0

    assert _canon_value(uncollided, set()) == f"proxy({uncollided._numbox_proxy_alias})", (
        "a binding holding a name that identifies it stopped folding by that name"
    )
    assert _canon_value(uncollided.as_func, set()) == f"proxyfunc({uncollided._numbox_proxy_alias})", (
        "the function value of an uncollided binding stopped folding by its alias"
    )
