# Static `params`-driven jitability for `compile_kernel` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a `Variable` optionally declare jit-status + numba type (`Params`) so `compile_kernel` resolves the execution mode at build time (eager compile, build-time `partition`, build-time type errors) for fully-declared graphs, while undeclared graphs stay byte-for-byte as today.

**Architecture:** A single classify→plan pipeline. `compile_kernel()` classifies the required cone into Case A (all-jittable → eager fused), Case B (declared mix → eager segmented, probe-free), or Case C (any unknown → today's lazy first-call discovery). Declared types feed the existing fused/segment codegen via `njit(sig)` bindings; a wrong scalar type is caught by an **explicit unconstrained return-type probe** (numba's `njit(sig)` silently coerces, so the bind alone does not catch it).

**Tech Stack:** Python 3.12 (venv), numba 0.65.1, numpy; pytest; flake8 (max-line-length=127); Sphinx.

**Spec:** `docs/superpowers/specs/2026-06-16-params-jitability-design.md` (v2). **Review:** `docs/superpowers/reviews/2026-06-16-adversarial-review.md`.

---

## Conventions (apply to every task)

- **venv python only:** `/home/erik/projects/numbox/venv/bin/python` (never bare `python`/`pytest`). pytest = `/home/erik/projects/numbox/venv/bin/pytest`.
- **Clean caches before every pytest run** (memory rule), as one line:
  `CLEAN = /home/erik/projects/numbox/venv/bin/python -c "import shutil,pathlib; [shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home()/'.cache'/'numba',ignore_errors=True)"`
  Each Verify runs `CLEAN && <pytest cmd>`.
- **No `cd`** except where a command must (sphinx); use absolute paths / `git -C /home/erik/projects/numbox`.
- **No planning references in code comments** (no task numbers / phase names). Comments explain the code only.
- **Commit per task** on `feat/params-jitability`. No `Co-Authored-By`. No person-names or AI-provenance in code/tests/commits.
- **Lint gate before each commit:** `/home/erik/projects/numbox/venv/bin/flake8 --max-line-length=127 <changed files>`.
- New tests live in `test/core/test_params_jitability.py` unless noted; the existing `test/core/test_compile_kernel.py` must keep passing **unchanged** (Case-C regression guard).
- Test data uses anonymous graphs (`Variables("m", [...])`); no person-names.

---

## File structure

| File | Responsibility | Tasks |
|------|----------------|-------|
| `numbox/core/variable/variable.py` | `Params` dataclass, `Variable.params` + guard, `VarSpec`, `External.declare`, `compiled_graphs` busting | 0, 8 |
| `numbox/core/variable/utils.py` | `_wrap_formula_typed`, `_validate_declared_return` (probe + exotic checks) | 2 |
| `numbox/core/variable/_kernel_partition.py` | `_evaluate` (fixed-demotion graph evaluation) | 3 |
| `numbox/core/variable/compile_kernel.py` | shared external validation, `_classify`, Case A/B/C dispatch, eager `CompiledKernel` construction, digest, recompute changes | 1, 4, 5, 6, 7 |
| `test/core/test_params_jitability.py` | all new tests | 0–8 |
| `docs/numbox.core.variable.rst`, module/function docstrings | doc updates + sphinx build | 9 |

---

## Task 0: `Params` data model + `Variable.params`

**Goal:** Add the frozen `Params` dataclass and a single optional `Variable.params` field with a construction-time guard, `VarSpec` key, and an `External.declare` helper.

**Files:**
- Modify: `numbox/core/variable/variable.py`
- Test: `test/core/test_params_jitability.py` (create)

**Acceptance Criteria:**
- [ ] `Params(jitable=True, type=None)` is a frozen dataclass; defaults as stated.
- [ ] `Variable(..., params=Params(type=float64))` round-trips; `Variable` stays hashable; identity (`==`/`hash`) unchanged by `params`.
- [ ] `Variable(..., params={"jitable": True})` raises `TypeError` at construction.
- [ ] `VarSpec` accepts `params`; `Variables("m", [{"name": "a", "params": Params(type=float64)}])` carries it.
- [ ] `External.declare(name, Params(type=float64))` stores a typed external retrievable by `external[name]`.

**Verify:** `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_params_jitability.py -v -k task0` → all pass.

**Steps:**

- [ ] **Step 1: Write failing tests.** Append to `test/core/test_params_jitability.py`:

```python
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
```

- [ ] **Step 2: Run, verify red.** `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_params_jitability.py -v -k task0` → ImportError / failures.

- [ ] **Step 3: Implement.** In `numbox/core/variable/variable.py`:

Add the dataclass above the `Variable` class:

```python
@dataclass(frozen=True)
class Params:
    """Optional per-`Variable` declaration driving static jitability in
    `compile_kernel`. `jitable=False` declares a deliberately plain-Python
    node; `type` is the numba `Type` instance of the variable's value
    (None means undeclared)."""
    jitable: bool = True
    type: Any = None
```

Add the field to `Variable` (after `metadata`) and a guard:

```python
    params: 'Params | None' = field(default=None)

    def __post_init__(self):
        if self.params is not None and not isinstance(self.params, Params):
            raise TypeError(
                f"{make_qual_name(self.source, self.name)!r}: params must be a "
                f"Params instance, not {type(self.params).__name__} (a dict is not accepted)"
            )
```

Add `params` to the `VarSpec` TypedDict:

```python
class VarSpec(VarSpecBase, total=False):
    inputs: dict[str, str]
    formula: Callable
    metadata: str
    params: 'Params'
```

Add the helper to `External`:

```python
    def declare(self, name: str, params: 'Params') -> 'Variable':
        """Pre-seed a typed external before compile (the only supported route
        to attach params to an external, which is otherwise auto-created
        untyped on lookup)."""
        variable = Variable(name=name, source=self.name, params=params)
        self._variables[name] = variable
        return variable
```

> Note: `Variable` is `@dataclass(frozen=True)` with a custom `__hash__`/`__eq__` already defined — adding `__post_init__` is compatible with frozen. `make_qual_name` is already defined in this module.

- [ ] **Step 4: Run, verify green.** `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_params_jitability.py -v -k task0` → pass.

- [ ] **Step 5: Lint + regression + commit.**
```bash
/home/erik/projects/numbox/venv/bin/flake8 --max-line-length=127 numbox/core/variable/variable.py test/core/test_params_jitability.py
git -C /home/erik/projects/numbox add numbox/core/variable/variable.py test/core/test_params_jitability.py
git -C /home/erik/projects/numbox commit -m "feat(variable): add Params dataclass and Variable.params field"
```

---

## Task 1: Shared external validation + classification

**Goal:** A pure `_classify(compiled, required)` that labels each interior node `STATIC_JIT`/`STATIC_PY`/`UNKNOWN`, returns the consumed-external set and the case (A/B/C); plus a hoisted `_validate_externals` (formula-bearing-external guard) that runs for all graphs before classification.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py`
- Test: `test/core/test_params_jitability.py`

**Acceptance Criteria:**
- [ ] `_classify` returns case `"A"` when all interior nodes are statically jittable and all consumed externals are typed; `"B"` when some declared `jitable=False` and no unknowns; `"C"` when any node is unknown.
- [ ] A `jitable=False` interior node consumed by a jittable node, but with `type=None`, makes the consumer `UNKNOWN` → case `"C"`.
- [ ] `_validate_externals` raises `ValueError` for a formula-bearing external regardless of case (including a graph that would otherwise be Case B).
- [ ] Classification reads `params` off the exact `Variable` instances in `compiled.ordered_nodes`.

**Verify:** `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_params_jitability.py -v -k task1` → pass.

**Steps:**

- [ ] **Step 1: Write failing tests.**

```python
from numbox.core.variable.variable import Graph, Params
from numbox.core.variable.compile_kernel import _classify, _validate_externals


def _g_caseA():
    # leaf c.a from external e.x; interior c.b = a + 1
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    return g


def test_task1_caseA():
    g = _g_caseA()
    compiled = g.compile(["c.b"])
    case, dispositions, consumed = _classify(compiled, ["c.b"])
    assert case == "A"
    assert all(d == "STATIC_JIT" for d in dispositions.values())


def test_task1_caseB_declared_python():
    g = _g_caseA()
    # redeclare b as a Python node WITH a type so its jit consumer (none here) / output is typed
    g.registry["c"].update("b", Variable(
        name="b", source="c", inputs={"a": "c"},
        formula=lambda a: a * 2.0, params=Params(jitable=False, type=float64)))
    # add a jit consumer d = b + 1 to force a real jit/python mix
    g.registry["c"].update("d", Variable(
        name="d", source="c", inputs={"b": "c"},
        formula=lambda b: b + 1.0, params=Params(type=float64)))
    compiled = g.compile(["c.d"])
    case, dispositions, _ = _classify(compiled, ["c.d"])
    assert case == "B"
    assert dispositions[g.registry["c"]["b"]] == "STATIC_PY"


def test_task1_caseC_untyped_python_boundary():
    g = _g_caseA()
    g.registry["c"].update("b", Variable(
        name="b", source="c", inputs={"a": "c"},
        formula=lambda a: a * 2.0, params=Params(jitable=False)))  # type=None
    g.registry["c"].update("d", Variable(
        name="d", source="c", inputs={"b": "c"},
        formula=lambda b: b + 1.0, params=Params(type=float64)))
    compiled = g.compile(["c.d"])
    case, dispositions, _ = _classify(compiled, ["c.d"])
    assert case == "C"  # d's input b has no type -> d unknown


def test_task1_formula_bearing_external_rejected_all_cases():
    g = Graph({"c": [{"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0}]}, ["e"])
    # external carrying a formula (illegal)
    g.external["e"].update("x", Variable(name="x", source="e", formula=lambda: 1.0))
    compiled = g.compile(["c.a"])
    with pytest.raises(ValueError, match="external but carries a formula"):
        _validate_externals(compiled)


from numbox.core.variable.variable import Variable  # noqa: E402  (used above)
from numba import float64  # noqa: E402
```

- [ ] **Step 2: Run, verify red.**

- [ ] **Step 3: Implement** in `compile_kernel.py`. Hoist the existing formula-bearing-external check out of `_generate_body` into a shared function, and add classification:

```python
def _validate_externals(compiled: CompiledGraph) -> set:
    """Run for ALL graphs before classification. Returns the external set and
    raises the same hard error _generate_body raised for a formula-bearing
    external (hoisted here so Case B's segment path cannot bypass it)."""
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    for var in sorted(external, key=lambda v: v.qual_name()):
        if var.formula is not None:
            raise ValueError(
                f"{var.qual_name()!r} is external but carries a formula; CompiledGraph "
                f"computes such a variable while a fused kernel treats it as a plain "
                f"input. Move it into a Variables namespace or drop the formula."
            )
    return external


def _is_typed(var: Variable) -> bool:
    return var.params is not None and var.params.type is not None


def _classify(compiled: CompiledGraph, required: list[str]):
    """Label interior nodes and pick the case. Returns
    (case, dispositions: {Variable: str}, consumed_externals: set[Variable])."""
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    consumed = set()
    dispositions = {}
    for node in compiled.ordered_nodes:
        var = node.variable
        if var in external:
            continue
        for inp in node.inputs:
            if inp in external:
                consumed.add(inp)
        p = var.params
        if p is not None and not p.jitable:
            dispositions[var] = "STATIC_PY"
        elif p is not None and p.jitable and p.type is not None and all(_is_typed(i) for i in node.inputs):
            dispositions[var] = "STATIC_JIT"
        else:
            dispositions[var] = "UNKNOWN"
    vals = set(dispositions.values())
    if "UNKNOWN" in vals or any(not _is_typed(e) for e in consumed):
        case = "C"
    elif "STATIC_PY" in vals:
        case = "B"
    else:
        case = "A"
    return case, dispositions, consumed
```

Then in `_generate_body`, remove the inline external-formula loop (it now lives in `_validate_externals`, called from `compile_kernel` — wired in Task 4). Leave the `ext_sorted`/`params` construction intact.

- [ ] **Step 4: Run, verify green.**

- [ ] **Step 5: Lint + commit** (`compile_kernel.py`, test).

```bash
git -C /home/erik/projects/numbox commit -m "feat(compile_kernel): add static jitability classification and shared external validation"
```

---

## Task 2: Typed binding + eager return-type validation

**Goal:** `_wrap_formula_typed` (uncached inner `njit(sig)`) and `_validate_declared_return` — the unconstrained probe that catches a coercible-but-wrong scalar `params.type`, plus exotic checks (CFunc/cres vs `_sig.return_type`; DUFunc shim).

**Files:**
- Modify: `numbox/core/variable/utils.py`
- Test: `test/core/test_params_jitability.py`

**Acceptance Criteria:**
- [ ] A node declared `int64` over a `lambda x: x * 1.5` body (input `int64`) **raises** at validation (the silent-coercion case `njit(sig)` alone misses).
- [ ] A correct declaration validates without error.
- [ ] A non-convertible body (returns a string) raises.
- [ ] A `DUFunc` (`@vectorize`) declared the wrong output type for an integer-preserving ufunc (`a + a` at `int64`) raises (or is rejected).
- [ ] `_wrap_formula_typed` never sets `cache=True` on the inner formula.

**Verify:** `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_params_jitability.py -v -k task2` → pass.

**Steps:**

- [ ] **Step 1: Spike the probe API (write a scratch assertion first).** The exact attribute for numba's *natural* inferred return type must be confirmed (blind spot in spec §8). Add a test that pins it:

```python
from numba import njit, int64, float64


def test_task2_probe_reads_natural_return_type():
    f = njit(lambda x: x * 1.5)
    f.compile((int64,))
    # the LAST compiled overload's naturally-inferred return type
    rt = f.nopython_signatures[-1].return_type
    assert rt == float64  # x*1.5 over int64 is float64, NOT int64
```

Run it: `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_params_jitability.py -v -k probe_reads_natural`. If `nopython_signatures[-1].return_type` is not the natural type on this numba, adjust to `f.overloads[(int64,)].signature.return_type` and update the helper below to match. **Do not proceed until this test passes** — it anchors the whole guard.

- [ ] **Step 2: Write failing validation tests.**

```python
from numba import vectorize
from numbox.core.variable.utils import _validate_declared_return, _wrap_formula_typed


def test_task2_coercible_wrong_type_raises():
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        _validate_declared_return(lambda x: x * 1.5, (int64,), int64, flags={})


def test_task2_correct_declaration_ok():
    _validate_declared_return(lambda x: x * 1.5, (int64,), float64, flags={})  # no raise


def test_task2_nonconvertible_raises():
    with pytest.raises(Exception):
        _validate_declared_return(lambda x: "s", (int64,), int64, flags={})


def test_task2_dufunc_wrong_output_raises():
    vf = vectorize(["int64(int64)", "float64(float64)"])(lambda a: a + a)
    with pytest.raises(ValueError):
        _validate_declared_return(vf, (int64,), float64, flags={})  # int+int stays int64


def test_task2_wrap_formula_typed_uncached():
    d = _wrap_formula_typed(lambda x: x + 1.0, float64(float64), flags={})
    assert d.targetoptions.get("cache") in (None, False)
```

- [ ] **Step 3: Implement** in `utils.py` (imports: `from numba import njit`; `from numba.core.ccallback import CFunc`; `from numba.np.ufunc.dufunc import DUFunc`; `from numba.core.types.function_type import CompileResultWAP`):

```python
def _wrap_formula_typed(formula, sig, flags: dict | None = None):
    """Like _wrap_formula but binds plain functions to an explicit signature.
    The inner dispatcher is NEVER cached (flags must not carry cache=True): a
    cached inner formula stale-hits on numeric-literal edits (numba hashes
    co_code, not co_consts) and would inline a stale body into the freshly
    content-addressed fused kernel."""
    if isinstance(formula, (Dispatcher, CompileResultWAP, DUFunc, CFunc)):
        return formula
    if not callable(formula):
        raise TypeError(f"formula {formula!r} is not callable")
    opts = {k: v for k, v in (flags or {}).items() if k != "cache"}
    return njit(sig, **opts)(formula)


def _validate_declared_return(formula, input_types: tuple, declared, flags: dict | None = None) -> None:
    """Raise if the formula's natural return type at `input_types` is not the
    `declared` numba type. njit(sig) silently coerces convertible scalars, so
    this compiles an UNCONSTRAINED overload and compares the inferred return."""
    opts = {k: v for k, v in (flags or {}).items() if k != "cache"}
    if isinstance(formula, (CFunc, CompileResultWAP)):
        natural = formula._sig.return_type
    elif isinstance(formula, DUFunc):
        # ufunc output dtype is a function of input dtype; resolve via a shim
        names = ", ".join(f"a{i}" for i in range(len(input_types)))
        ns = {"_f": formula}
        exec(f"def _shim({names}):\n    return _f({names})\n", ns)  # nosec B102
        probe = njit(**opts)(ns["_shim"])
        probe.compile(input_types)
        natural = probe.nopython_signatures[-1].return_type
    else:
        probe = njit(**opts)(formula)
        probe.compile(input_types)
        natural = probe.nopython_signatures[-1].return_type
    if natural != declared:
        raise ValueError(
            f"declared type {declared} but formula yields {natural} at "
            f"input types {input_types}"
        )
```

> If Step 1 showed a different attribute, use it consistently in all three branches.

- [ ] **Step 4: Run, verify green.**

- [ ] **Step 5: Lint + commit** (`utils.py`, test).

```bash
git -C /home/erik/projects/numbox commit -m "feat(variable): typed formula binding and eager declared-return validation"
```

---

## Task 3: `_evaluate` (fixed-demotion graph evaluation)

**Goal:** Add `_evaluate` to `_kernel_partition.py` — populate all node values using a **given** demotion set (no probing). `discover` is left untouched so Case C stays byte-for-byte.

**Files:**
- Modify: `numbox/core/variable/_kernel_partition.py`
- Test: `test/core/test_params_jitability.py`

**Acceptance Criteria:**
- [ ] `_evaluate` populates `values` for every interior node, honoring the passed `demoted` set (demoted → `py_func`; exotic → `_call_exotic`; else → `binding`).
- [ ] `_evaluate` never re-probes or mutates the demotion set.
- [ ] `discover` byte-for-byte unchanged (existing `test_compile_kernel.py` passes).

**Verify:** `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_params_jitability.py -v -k task3 && CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_compile_kernel.py -q`

**Steps:**

- [ ] **Step 1: Write failing test.**

```python
from numba import njit as _njit, typeof
from numbox.core.variable.variable import Graph, CompiledNode
from numbox.core.variable._kernel_partition import _evaluate


def test_task3_evaluate_honors_fixed_demotion():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0},
    ]}, ["e"])
    compiled = g.compile(["c.b"])
    a = g.registry["c"]["a"]
    b = g.registry["c"]["b"]
    ext = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    x = next(iter(ext))
    bindings = {a: _njit(a.formula), b: _njit(b.formula)}
    values = {x: 3.0}
    demoted = {b}  # force b to run as plain python
    _evaluate(compiled.ordered_nodes, ext, values, bindings, {}, demoted)
    assert values[a] == 4.0 and values[b] == 8.0
```

- [ ] **Step 2: Run, verify red.**

- [ ] **Step 3: Implement** in `_kernel_partition.py` (mirrors `discover`'s value-population branches with a fixed demotion set; the small duplication is deliberate to keep `discover` — and Case C — byte-for-byte):

```python
def _evaluate(
    ordered_nodes: list[CompiledNode],
    external: set[Variable],
    values: dict[Variable, Any],
    bindings_by_var: dict[Variable, Any],
    flags: dict | None,
    demoted: set[Variable],
) -> None:
    """Populate `values` for every interior node using a FIXED `demoted` set
    (no probing). Demoted nodes run their py_func; exotic bindings run via the
    @njit shim; the rest run their Dispatcher. Used by the declared (eager)
    path where demotions come from declarations, not discovery."""
    for node in ordered_nodes:
        var = node.variable
        if var in external:
            continue
        args = [values[inp] for inp in node.inputs]
        binding = bindings_by_var[var]
        if not isinstance(binding, Dispatcher):
            arg_types = tuple(typeof(a) for a in args)
            values[var] = _call_exotic(binding, args, arg_types, flags)
        elif var in demoted:
            py = getattr(var.formula, "py_func", var.formula)
            values[var] = py(*args)
        else:
            values[var] = binding(*args)
```

- [ ] **Step 4: Run, verify green** (both the new test and the full `test_compile_kernel.py`).

- [ ] **Step 5: Lint + commit.**

```bash
git -C /home/erik/projects/numbox commit -m "feat(variable): add _evaluate for fixed-demotion graph evaluation"
```

---

## Task 4: Eager Case A (fused) construction

**Goal:** Wire `compile_kernel` to validate externals, classify, and for Case A construct an eager-fused `CompiledKernel` (`"fused-pending"` mode, `is_declared=True`, `_fused.compile(consumed_sig)`, `partition` at build, pass-through external outputs excluded from the sig), with a `kernel`-property one-shot capture branch so `recompute` works.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py`
- Test: `test/core/test_params_jitability.py`

**Acceptance Criteria:**
- [ ] A Case-A graph yields `kernel.partition.mode == "fused"` **before** any call, and correct results on call.
- [ ] A coercible wrong `params.type` (declared `int64` over `x*1.5`) raises at `compile_kernel()` (uses Task 2's validator).
- [ ] `recompute(...)` after a single fused call returns correct values (does NOT raise `RuntimeError`) — H2.
- [ ] A required pass-through external output (untyped, no consumer) compiles in Case A — M3.
- [ ] `CompiledKernel.is_declared is True` for Case A; existing virgin path keeps `is_declared = False`.

**Verify:** `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_params_jitability.py -v -k task4`

**Steps:**

- [ ] **Step 1: Write failing tests.**

```python
from numbox.core.variable.compile_kernel import compile_kernel
from numba import int64


def _declared_chain():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    return g


def test_task4_caseA_partition_at_build():
    ck = compile_kernel(_declared_chain(), "c.b")
    assert ck.partition is not None and ck.partition.mode == "fused"  # before any call
    assert ck.is_declared is True
    assert ck.kernel(3.0) == (8.0,)


def test_task4_caseA_recompute_after_fused_call():
    ck = compile_kernel(_declared_chain(), "c.b")
    assert ck.kernel(3.0) == (8.0,)
    assert ck.recompute({"e": {"x": 4.0}}) == (10.0,)  # must not raise RuntimeError


def test_task4_coercible_wrong_type_raises_at_build():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x * 1.5, "params": Params(type=int64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=int64))
    with pytest.raises(ValueError, match="declared .* but formula yields"):
        compile_kernel(g, "c.a")
```

- [ ] **Step 2: Run, verify red.**

- [ ] **Step 3: Implement.**

(a) `CompiledKernel.__init__`: add `is_declared: bool = False` param, store it as the **public** `self.is_declared = is_declared` (referenced by tests and by Task 7).

(b) Add a `"fused-pending"` branch to the `kernel` property:

```python
    @property
    def kernel(self) -> Callable:
        if self._mode == "fused":
            return self._fused
        if self._mode == "fused-pending":
            return self._fused_pending_call
        if self._mode == "segmented":
            return self._run_segmented
        return self._resolve_and_call

    def _fused_pending_call(self, *args) -> tuple:
        self._last_args = args
        self._mode = "fused"
        self.partition = self._fused_report()
        return self._fused(*args)
```

(c) New constructor path used by `compile_kernel` for Case A — set `_mode="fused-pending"`, pre-`compile` the fused dispatcher to the consumed-external signature, set `is_declared=True`. Build the consumed signature in kernel-arg order, skipping externals not consumed by any interior node (M3):

```python
def _build_eager_fused(kernel, params, outputs, source, identifiers, ctx, required_vars,
                       external_vars, consumed, dispositions, idents, jit_options, cache):
    ck = CompiledKernel(kernel, params, outputs, source, identifiers, ctx,
                        required_vars, external_vars, is_declared=True)
    sig_vars = [v for v in external_vars if v in consumed]
    sig = tuple(v.params.type for v in sig_vars)
    if sig_vars:
        ck._fused.compile(sig)  # raises NumbaError here at build for a real type clash
    ck._mode = "fused-pending"
    ck.partition = ck._fused_report()
    return ck
```

(d) In `compile_kernel`, after building `compiled`, before constructing the kernel:

```python
    external = _validate_externals(compiled)
    case, dispositions, consumed = _classify(compiled, required)
    ...
    # build source/bindings as today, but bind STATIC_JIT nodes via _wrap_formula_typed
    # and validate each declared node's return BEFORE compiling the kernel:
    if case in ("A", "B"):
        for node in compiled.ordered_nodes:
            var = node.variable
            if var in external or dispositions.get(var) != "STATIC_JIT":
                continue
            in_types = tuple(i.params.type for i in node.inputs)
            _validate_declared_return(var.formula, in_types, var.params.type, flags)
    if case == "A":
        # bindings via _wrap_formula_typed(var.formula, sig, flags); compile eager
        ...
        return _build_eager_fused(kernel, params, outputs, source, identifiers, ctx,
                                  required_vars, external_vars, consumed, dispositions,
                                  idents, jit_options, cache)
    # case B -> Task 5; case C -> existing virgin construction (unchanged)
```

> Pass-through external output: such an external is in `external_vars` but not in `consumed`, so it is excluded from `sig` and numba infers its type from the runtime arg on the first call.

- [ ] **Step 4: Run, verify green** (task4 tests + `test_compile_kernel.py` regression).

- [ ] **Step 5: Lint + commit.**

```bash
git -C /home/erik/projects/numbox commit -m "feat(compile_kernel): eager fused construction for fully-declared graphs"
```

---

## Task 5: Eager Case B (segmented) construction

**Goal:** For Case B, build the segmented plan at `compile_kernel()` using the declared `jitable=False` set as the demotion set (no probing), compiling each segment against **declared** live-in types; seed `_demoted`, `partition`, `is_declared=True`.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py`
- Test: `test/core/test_params_jitability.py`

**Acceptance Criteria:**
- [ ] A declared jit/python mix yields `partition.mode == "segmented"` at build with `reasons` marking the declared-Python node, and correct results.
- [ ] No probing occurs (the declared-Python node is taken as Python even if it would compile).
- [ ] A formula-bearing external in a Case-B-shaped graph still raises (`_validate_externals`) — M2.

**Verify:** `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_params_jitability.py -v -k task5`

**Steps:**

- [ ] **Step 1: Write failing tests.**

```python
def _declared_mix():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0, "params": Params(type=float64)},
        {"name": "b", "inputs": {"a": "c"}, "formula": lambda a: a * 2.0,
         "params": Params(jitable=False, type=float64)},
        {"name": "d", "inputs": {"b": "c"}, "formula": lambda b: b + 1.0, "params": Params(type=float64)},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64))
    return g


def test_task5_caseB_partition_and_result():
    ck = compile_kernel(_declared_mix(), "c.d")
    assert ck.partition is not None and ck.partition.mode == "segmented"
    assert ck.is_declared is True
    assert "c.b" in ck.partition.python_nodes
    assert ck.kernel(3.0) == (9.0,)  # ((3+1)*2)+1


def test_task5_formula_bearing_external_in_B_raises():
    g = _declared_mix()
    g.external["e"].update("x", Variable(name="x", source="e", formula=lambda: 1.0,
                                         params=Params(type=float64)))
    with pytest.raises(ValueError, match="external but carries a formula"):
        compile_kernel(g, "c.d")
```

- [ ] **Step 2: Run, verify red.**

- [ ] **Step 3: Implement** the Case B branch in `compile_kernel`, reusing the existing `_discover_and_run` segment-building helpers but with a **static** demotion set and **declared** live-in types:

```python
    if case == "B":
        demoted = {n.variable for n in compiled.ordered_nodes
                   if dispositions.get(n.variable) == "STATIC_PY"}
        nodes = [n for n in compiled.ordered_nodes if n.variable not in external]
        order = linearize(nodes, demoted)
        runs = build_runs(order, demoted)
        steps, segments = [], []
        for kind, run_nodes in runs:
            quals = tuple(n.variable.qual_name() for n in run_nodes)
            if kind == "python":
                for n in run_nodes:
                    steps.append(_PyStep(
                        var=n.variable,
                        py_callable=getattr(n.variable.formula, "py_func", n.variable.formula),
                        in_vars=tuple(n.inputs)))
                reasons = {q: "declared non-jittable" for q in quals}
                ins = sorted({i for n in run_nodes for i in n.inputs
                              if i not in {x.variable for x in run_nodes}},
                             key=lambda v: v.qual_name())
                segments.append(Segment(kind="python", nodes=quals,
                                        inputs=tuple(v.qual_name() for v in ins),
                                        outputs=quals, source=None, reasons=reasons))
                continue
            live_in, live_out = segment_liveness(run_nodes, external, required_vars, order)
            src, seg_bindings, _, _ = _generate_segment_body(run_nodes, live_in, live_out, idents, flags)
            disp = _compile(src, seg_bindings, jit_options, cache)
            disp.compile(tuple(v.params.type for v in live_in))  # DECLARED types, no typeof
            steps.append(_JitStep(dispatcher=disp, in_vars=live_in, out_vars=live_out))
            segments.append(Segment(kind="jit", nodes=quals,
                                    inputs=tuple(v.qual_name() for v in live_in),
                                    outputs=tuple(v.qual_name() for v in live_out),
                                    source=src, reasons={}))
        ck = CompiledKernel(kernel, params, outputs, source, identifiers, ctx,
                            required_vars, external_vars, is_declared=True)
        ck._plan = _Plan(steps=tuple(steps), external_vars=tuple(external_vars),
                         output_vars=tuple(required_vars))
        ck._demoted = demoted
        ck._mode = "segmented"
        ck.partition = PartitionReport(mode="segmented", segments=tuple(segments))
        return ck
```

> `_PyStep`, `_Plan`, `Segment`, `linearize`, `build_runs`, `segment_liveness`, `_generate_segment_body` are already imported in `compile_kernel.py`.

- [ ] **Step 4: Run, verify green** (task5 + regression).

- [ ] **Step 5: Lint + commit.**

```bash
git -C /home/erik/projects/numbox commit -m "feat(compile_kernel): eager segmented construction for declared mixed graphs"
```

---

## Task 6: Cache-digest extension (declared signatures)

**Goal:** Fold declared signatures into the `ck-digest` hash text (via `repr`), so anchor filenames are 1:1 with a concrete typed kernel; confirm inner formulas stay uncached.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (`_compile`)
- Test: `test/core/test_params_jitability.py`

**Acceptance Criteria:**
- [ ] Two declared-type variants of the same graph produce **distinct** anchor digests and do not reuse each other's binary (distinct results).
- [ ] A literal-only edit to an inner formula does not stale-hit (inner stays uncached) — co_consts regression.

**Verify:** `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_params_jitability.py -v -k task6`

**Steps:**

- [ ] **Step 1: Write failing tests.**

```python
def test_task6_declared_variants_distinct():
    def make(t):
        g = Graph({"c": [{"name": "a", "inputs": {"x": "e"},
                          "formula": lambda x: x + 1, "params": Params(type=t)}]}, ["e"])
        g.external["e"].declare("x", Params(type=t))
        return compile_kernel(g, "c.a", cache=True)
    ck_i = make(int64)
    ck_f = make(float64)
    # distinct digests visible in generated kernel name / source-independent anchors
    assert ck_i.kernel(3) == (4,)
    assert ck_f.kernel(3.0) == (4.0,)
    # the two kernels must not collide on the anchor digest
    assert ck_i.source == ck_f.source  # source is type-free
    # digest divergence is asserted via the dispatcher name carrying the digest
    assert ck_i._fused.__name__ != ck_f._fused.__name__
```

- [ ] **Step 2: Run, verify red** (names equal until digest includes sigs).

- [ ] **Step 3: Implement** in `_compile`: accept an optional `declared_sigs` argument and append it to `hash_text`:

```python
def _compile(source, bindings, jit_options, cache, declared_sigs: tuple = ()):
    ...
    hash_text = (
        "ck-digest-v3\n" + source
        + "\n# formulas:\n" + "\n".join(fingerprints)
        + "\n# flags: " + flags_canon
        + "\n# declared_sigs: " + repr([repr(s) for s in declared_sigs])
    )
    ...
```

Thread the declared signatures from Case A (`(consumed_sig,)`) and Case B (each segment's `(live_in declared types, live_out declared types)`) into `_compile`. Bump the digest tag to `ck-digest-v3` so pre-existing cached `v2` artifacts are never mismatched.

> `repr(numba_signature)` is byte-stable; do NOT route numba Type/Signature objects through `_canon_value` (it raises `_Unfingerprintable`).

- [ ] **Step 4: Run, verify green.**

- [ ] **Step 5: Lint + commit.**

```bash
git -C /home/erik/projects/numbox commit -m "feat(compile_kernel): fold declared signatures into the cache digest"
```

---

## Task 7: `recompute` composition for declared kernels

**Goal:** `_ensure_store` uses `_evaluate` (not `discover`) for declared kernels; `_apply_changes` uses a `can_convert`-based contract check scoped to eager kernels; `_run_segmented` does not re-`discover` for declared kernels; `_flush_and_reseed` dropped for declared cones; cone live-ins compiled against declared types.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py`
- Test: `test/core/test_params_jitability.py`

**Acceptance Criteria:**
- [ ] Declared recompute returns correct values; the contract check accepts a C-contiguous numpy array where a `float64[:]` was declared (no false raise) — H3.
- [ ] A genuine off-contract type (e.g. complex where float declared) raises the crisp "declared type … got …" error.
- [ ] A declared node inside a **Case-C** kernel does NOT trip the contract check (uses existing recovery) — M8.
- [ ] An eager kernel does not silently re-`discover` and overwrite `_demoted` on an off-contract type — H4.

**Verify:** `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_params_jitability.py -v -k task7 && CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_compile_kernel.py -q`

**Steps:**

- [ ] **Step 1: Write failing tests.**

```python
import numpy as np
from numba import float64


def test_task7_declared_array_recompute_accepts_c_contiguous():
    g = Graph({"c": [
        {"name": "a", "inputs": {"x": "e"}, "formula": lambda x: x + 1.0,
         "params": Params(type=float64[:])},
    ]}, ["e"])
    g.external["e"].declare("x", Params(type=float64[:]))
    ck = compile_kernel(g, "c.a")
    base = np.zeros(4)               # C-contiguous; declared layout is 'A'
    assert ck.kernel(base)[0].tolist() == [1, 1, 1, 1]
    out = ck.recompute({"e": {"x": np.ones(4)}})   # must NOT raise on layout
    assert out[0].tolist() == [2, 2, 2, 2]


def test_task7_off_contract_type_raises_crisp():
    ck = compile_kernel(_declared_chain(), "c.b")
    ck.kernel(3.0)
    with pytest.raises(ValueError, match="declared type"):
        ck.recompute({"e": {"x": 3j}})  # complex where float64 declared
```

- [ ] **Step 2: Run, verify red.**

- [ ] **Step 3: Implement.**

(a) `_ensure_store`: branch on `self._is_declared` — for declared kernels evaluate via `_evaluate` with the frozen `self._demoted` (no `discover`):

```python
        values = dict(zip(self._external_vars, self._last_args))
        if self.is_declared:
            _evaluate(compiled.ordered_nodes, external, values, bindings_by_var, flags, self._demoted)
        else:
            self._demoted = discover(compiled.ordered_nodes, external, values, bindings_by_var, flags)
        self._store = values
```

(b) `_apply_changes`: add a contract check, scoped to eager declared kernels:

```python
                if self.is_declared and var.params is not None and var.params.type is not None:
                    from numba.core.registry import cpu_target
                    tc = cpu_target.typing_context
                    if tc.can_convert(typeof(val), var.params.type) is None:
                        raise ValueError(
                            f"declared type {var.params.type}, got {typeof(val)} for {var.qual_name()}")
```

(c) `_run_segmented`: gate the `NumbaError → _discover_and_run` fallback on `not self.is_declared`; for declared kernels re-raise.

(d) `recompute` / `_build_cone_plan`: for declared kernels, drop the `_flush_and_reseed` recovery; compile cone live-ins against `v.params.type` when typed, else `typeof(self._store[v])`.

- [ ] **Step 4: Run, verify green** (task7 + full regression).

- [ ] **Step 5: Lint + commit.**

```bash
git -C /home/erik/projects/numbox commit -m "feat(compile_kernel): declared-kernel recompute contract and seeding"
```

---

## Task 8: Before-first-compile ordering contract

**Goal:** Bust the `compiled_graphs` cache entry when `Namespace.update` replaces a `Variable`, so attaching `params` after a first compile is not silently ignored.

**Files:**
- Modify: `numbox/core/variable/variable.py`
- Test: `test/core/test_params_jitability.py`

**Acceptance Criteria:**
- [ ] Compile undeclared → attach `params` via `update` → recompile must NOT return the stale Case-C kernel (it reflects the new params).

**Verify:** `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/core/test_params_jitability.py -v -k task8`

**Steps:**

- [ ] **Step 1: Write failing test.**

```python
def test_task8_update_busts_compiled_graphs():
    g = _declared_chain()
    # first compile WITHOUT declaring b (force Case C by removing b's params)
    g.registry["c"].update("b", Variable(name="b", source="c", inputs={"a": "c"},
                                         formula=lambda a: a * 2.0))
    ck1 = compile_kernel(g, "c.b")
    assert ck1.partition is None  # Case C: unresolved until call
    # now declare b and recompile the same required set
    g.registry["c"].update("b", Variable(name="b", source="c", inputs={"a": "c"},
                                         formula=lambda a: a * 2.0, params=Params(type=float64)))
    ck2 = compile_kernel(g, "c.b")
    assert ck2.partition is not None and ck2.partition.mode == "fused"  # not stale
```

- [ ] **Step 2: Run, verify red.**

- [ ] **Step 3: Implement.** `compile_kernel` calls `graph.compile(required)`, which memoizes per required-tuple. Make `Namespace.update` invalidate the owning `Graph`'s cache for any required-tuple containing the replaced node. Simplest robust approach: give `Graph` a reference from each `Namespace` (set in `Graph.__init__`), and have `Namespace.update` clear `graph.compiled_graphs` (and `graph.reverse_dependencies`). Document the before-first-compile contract in the `Variable.params` docstring.

```python
    def update(self, key: str, var: 'Variable') -> None:
        self._variables[key] = var
        graph = getattr(self, "_graph", None)
        if graph is not None:
            graph.compiled_graphs.clear()
            graph.reverse_dependencies = None
```

In `Graph.__init__`, after building `self.registry`/`self.external`, set `ns._graph = self` for each namespace.

- [ ] **Step 4: Run, verify green** (task8 + regression).

- [ ] **Step 5: Lint + commit.**

```bash
git -C /home/erik/projects/numbox commit -m "fix(variable): invalidate compiled-graph cache on namespace update"
```

---

## Task 9: Documentation + sphinx build

**Goal:** Update the module/function docstrings and `docs/numbox.core.variable.rst` to describe declared `params`, the corrected error-timing contract, the digest extension, and the recompute contract; build the docs clean.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (module docstring L1-22; `compile_kernel` "Error timing"/"Caching"/"Non-jittable formulas" sections)
- Modify: `docs/numbox.core.variable.rst` (overview ~213-220, Caching ~236-254, recompute ~341-348)

**Acceptance Criteria:**
- [ ] Docstrings/RST state that per-node types are optional; declaring them moves errors to build (via the explicit probe, not `njit(sig)` coercion); declared sigs extend the digest; recompute applies a convertibility contract check for declared kernels.
- [ ] `sphinx-build` exits 0; doc-codeblock-flake8 passes on any RST code blocks.

**Verify:**
```
cd /home/erik/projects/numbox/docs && /home/erik/projects/numbox/venv/bin/sphinx-build -b html . _build/html
```
→ exit 0 (warning count stable).

**Steps:**

- [ ] **Step 1: Update the `compile_kernel` module docstring** (L5-8) to note that jitability is *either* discovered (undeclared) *or* declared via `Variable.params`, and that a wrong scalar `params.type` is caught by an explicit return-type probe at build (njit(sig) silently coerces).
- [ ] **Step 2: Update `compile_kernel`'s docstring "Error timing" / "Caching" / "Non-jittable formulas" sections** to match §5/§4 of the spec.
- [ ] **Step 3: Update `docs/numbox.core.variable.rst`** overview / Caching / recompute paragraphs.
- [ ] **Step 4: Build docs** with the Verify command; fix warnings introduced by the edits.
- [ ] **Step 5: doc-codeblock-flake8** on changed RST (if any python code-blocks): `/home/erik/projects/numbox/venv/bin/python .github/scripts/extract_codeblocks.py ... | /home/erik/projects/numbox/venv/bin/flake8 -` (match the CI invocation).
- [ ] **Step 6: Commit.**

```bash
git -C /home/erik/projects/numbox commit -m "docs(variable): document static params-driven jitability"
```

---

## Final gate (before any push — not a task, a checklist)

- [ ] Full suite: `CLEAN && /home/erik/projects/numbox/venv/bin/pytest test/ -q --durations=20`
- [ ] flake8 whole tree: `/home/erik/projects/numbox/venv/bin/flake8 --max-line-length=127 .`
- [ ] docs build exit 0; doc-codeblock-flake8; link-check (lychee) on changed docs.
- [ ] `test/core/test_compile_kernel.py` passes unchanged (Case-C byte-for-byte).
- [ ] Push only on explicit user consent (fork PR first; CLAUDE.md + `docs/superpowers/**` excluded from any upstream PR).
