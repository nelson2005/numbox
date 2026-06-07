# compile_kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `numbox/core/variable/compile_kernel.py` that compiles a `Variable` graph into one fused `@njit` kernel computing a requested set of variables.

**Architecture:** Reuse `Graph.compile` for topological order + external-variable discovery; assign each node a readable, collision-free identifier; generate straight-line `@njit` source (one `tmp = f_tmp(...)` per node, interior nodes as SSA temporaries) and `exec` it against a content-addressed on-disk anchor (reusing `numbox/utils/preprocessing.py`) so `cache=True` is correct. No per-node types are required — numba infers from runtime argument types. Plain-Python formulas are auto-wrapped with `njit()`.

**Tech Stack:** Python ≥3.10, numba 0.65.1, existing numbox modules (`core/variable/variable.py`, `core/configurations.py`, `utils/preprocessing.py`, `utils/highlevel.py`).

**Spec:** `docs/superpowers/specs/2026-06-07-compile-kernel-design.md`

**Conventions for every task:**
- Always use the venv interpreter: `/home/erik/projects/numbox/venv/bin/python` (never bare `python`/`pytest`).
- Run tests from the repo root and clean caches first (this feature exercises numba caching, so stale caches can mask bugs):
  ```bash
  ( cd /home/erik/projects/numbox && \
    venv/bin/python -c "import shutil,pathlib,os; r=pathlib.Path('.'); [shutil.rmtree(p,ignore_errors=True) for p in r.rglob('__pycache__')]; shutil.rmtree(pathlib.Path(os.path.expanduser('~/.cache/numba')),ignore_errors=True)" && \
    venv/bin/python -m pytest test/core/test_compile_kernel.py -q )
  ```
- No task numbers / phase references in code comments.
- Commit after each task (no `Co-Authored-By`, no AI provenance). Branch: `feat/variable-compile-kernel`.

---

### Task 1: Module scaffold + identifier assignment

**Goal:** Create the module with `_sanitize` and `_assign_identifiers`, producing unique, valid, readable Python identifiers for graph nodes.

**Files:**
- Create: `numbox/core/variable/compile_kernel.py`
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] `_sanitize` maps any string to a valid lowercase identifier (invalid chars → `_`, leading digit / empty → `v_` prefix).
- [ ] `_assign_identifiers` returns a `{Variable: identifier}` map that is unique across node temps, their `f_<temp>` formula globals, and the reserved names.
- [ ] The three collision classes resolve correctly: `("a_b","c")` vs `("a","b_c")`; an invalid-char/leading-digit name; a node whose identifier would be `f_x`.
- [ ] Identifiers are deterministic (same graph → same identifiers).

**Verify:** `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

```python
# test/core/test_compile_kernel.py
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
```

- [ ] **Step 2: Run to verify failure**

Run: `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )`
Expected: FAIL (ImportError — module/functions not defined)

- [ ] **Step 3: Write the module + functions**

```python
# numbox/core/variable/compile_kernel.py
"""Compile a `core.variable` Variable graph into one fused @njit kernel.

Alongside `core.work` (a structref graph), this turns a `Graph`/`CompiledGraph`
into a single straight-line @njit function whose interior nodes are SSA
temporaries. No per-node type info is needed: numba infers every interior type
from the kernel's runtime argument types, provided each formula is njit-able
(plain-Python formulas are auto-wrapped with njit()).
"""
import hashlib
import re

from numba.core.dispatcher import Dispatcher
from numba.core.types.function_type import CompileResultWAP

# Names injected into the kernel exec namespace; identifiers must avoid them.
_RESERVED = frozenset({"njit", "_kernel_jit_options"})


def _sanitize(qual_name):
    s = re.sub(r"[^0-9A-Za-z_]", "_", qual_name)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    if not s or s[0].isdigit():
        s = "v_" + s
    return s


def _assign_identifiers(variables):
    """Map each Variable to a unique, valid, readable Python identifier.

    Readable (from the qual_name) with a minimal deterministic sha256 suffix
    only where names would otherwise collide. Reserves both the node temp `t`
    and its formula global `f_<t>` so those namespaces never clash, and avoids
    the injected reserved names.
    """
    used = set(_RESERVED)
    idents = {}
    for var in variables:
        base = _sanitize(var.qual_name())
        digest = hashlib.sha256(var.qual_name().encode("utf-8")).hexdigest()
        cand = base
        i = 0
        while cand in used or ("f_" + cand) in used:
            i += 1
            cand = f"{base}_{digest[:i]}"
        used.add(cand)
        used.add("f_" + cand)
        idents[var] = cand
    return idents
```

- [ ] **Step 4: Run to verify pass**

Run: `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: node identifier assignment (sanitize + collision-free)"
```

---

### Task 2: Codegen — formula wrapping + source generation

**Goal:** Add `_wrap_formula`, `_safe_getsource`, and `_generate_body`, which turn a `CompiledGraph` + identifier map into kernel source text, formula bindings, params, and outputs.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py`
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] `_wrap_formula` returns numba `Dispatcher`/`CompileResultWAP` unchanged and wraps a plain function with `njit()`.
- [ ] `_generate_body` emits `def _kernel(<params>):` with one `tmp = f_tmp(args)` line per derived node (topo order, inputs in formula-arg order), a `return (outs,)` tuple, and a `bindings` dict keyed `f_<temp>`.
- [ ] External (leaf) nodes become parameters (sorted by qual_name); they emit no assignment line.
- [ ] Raises `ValueError` for: empty `required`; a non-external node with `formula is None`; a requested name absent from the graph.

**Verify:** `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

```python
# append to test/core/test_compile_kernel.py
import pytest
from numba import njit
from numba.core.dispatcher import Dispatcher
from numbox.core.variable.compile_kernel import _wrap_formula, _generate_body
from numbox.core.variable.variable import Graph


def _diamond_graph():
    # basket.y -> x=2y -> {a=x-74, b=x+0.5} -> u=a+b
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
    assert _wrap_formula(d) is d                       # dispatcher passes through

    def plain(x):
        return x + 1
    assert isinstance(_wrap_formula(plain), Dispatcher)  # plain -> njit


def test_generate_body_shape():
    g = _diamond_graph()
    compiled = g.compile(["variables.u", "variables.a"])
    from numbox.core.variable.compile_kernel import _assign_identifiers
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    source, bindings, params, outputs = _generate_body(compiled, ["variables.u", "variables.a"], idents)
    assert params == [("basket", "y", idents[next(v for v in idents if v.qual_name() == "basket.y")])]
    assert outputs == ["variables.u", "variables.a"]
    assert source.startswith("def _kernel(")
    assert source.rstrip().endswith(",)")
    assert set(bindings) == {"f_" + idents[v] for v in idents if v.qual_name() != "basket.y"}


def test_generate_body_errors():
    g = _diamond_graph()
    compiled = g.compile(["variables.u"])
    from numbox.core.variable.compile_kernel import _assign_identifiers
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    with pytest.raises(ValueError):
        _generate_body(compiled, [], idents)                       # empty required
    with pytest.raises(ValueError):
        _generate_body(compiled, ["variables.nope"], idents)       # unknown output

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
        _generate_body(c2, ["variables.broken"], id2)              # placeholder formula=None
```

- [ ] **Step 2: Run to verify failure**

Run: `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )`
Expected: FAIL (ImportError on `_wrap_formula`/`_generate_body`)

- [ ] **Step 3: Add the functions**

```python
# append to numbox/core/variable/compile_kernel.py
from inspect import getsource
from numba import njit


def _wrap_formula(formula):
    """Return an njit-callable for `formula`; plain Python is auto-njit'd."""
    if isinstance(formula, (Dispatcher, CompileResultWAP)):
        return formula
    return njit(formula)


def _safe_getsource(formula):
    """Source text of a formula for cache hashing; falls back to repr."""
    target = getattr(formula, "py_func", formula)
    try:
        return getsource(target)
    except (OSError, TypeError):
        return repr(formula)


def _generate_body(compiled, required, idents):
    """Generate `def _kernel(...): ...` source (no decorator) + bindings.

    Returns (source, bindings, params, outputs):
      source   - the kernel def as text (function name is the literal _kernel)
      bindings - {formula_global_name: njit-callable}
      params   - [(source_name, var_name, identifier)] in kernel-arg order
      outputs  - [requested_qual_name] in return-tuple order
    """
    if not required:
        raise ValueError("compile_kernel requires at least one requested variable")

    external = set()
    for vars_ in compiled.required_external_variables.values():
        external.update(vars_.values())

    ext_sorted = sorted(external, key=lambda v: v.qual_name())
    params = [(v.source, v.name, idents[v]) for v in ext_sorted]

    bindings = {}
    lines = []
    for node in compiled.ordered_nodes:
        var = node.variable
        if var in external:
            continue
        if var.formula is None:
            raise ValueError(
                f"{var.qual_name()!r} has graph placement but no formula; a fused "
                f"kernel cannot compile it. Provide a formula, or use CompiledGraph."
            )
        temp = idents[var]
        fg = "f_" + temp
        bindings[fg] = _wrap_formula(var.formula)
        arg_ids = ", ".join(idents[inp] for inp in node.inputs)
        in_names = ", ".join(inp.qual_name() for inp in node.inputs)
        lines.append(f"    {temp} = {fg}({arg_ids})  # {var.qual_name()} = f({in_names})")

    by_qual = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    outputs, out_ids = [], []
    for q in required:
        var = by_qual.get(q)
        if var is None:
            raise ValueError(f"Requested variable {q!r} is not in the compiled graph")
        outputs.append(q)
        out_ids.append(idents[var])

    sig = ", ".join(ident for _, _, ident in params)
    body = "\n".join(lines) if lines else "    pass"
    ret = f"    return ({', '.join(out_ids)},)"
    source = f"def _kernel({sig}):\n{body}\n{ret}\n"
    return source, bindings, params, outputs
```

- [ ] **Step 4: Run to verify pass**

Run: `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: formula wrapping + straight-line source generation"
```

---

### Task 3: Compile with content-addressed cache

**Goal:** Add `_compile`, which hashes the kernel source + formula sources, materializes an on-disk anchor, and `exec`s the `@njit(cache=...)` kernel.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py`
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] `_compile(source, bindings, jit_options, cache)` returns a numba dispatcher that computes correctly.
- [ ] A content-addressed anchor file is created under numba's cache dir, subdir `numbox-compile-kernel`.
- [ ] The anchor path is deterministic for identical (source, formula-sources) and differs when a formula's source differs (collision safety).

**Verify:** `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

```python
# append to test/core/test_compile_kernel.py
from numbox.core.variable.compile_kernel import _compile, _ANCHOR_SUBDIR
from numbox.utils.preprocessing import _anchor_root


def test_compile_runs():
    src = "def _kernel(y):\n    x = f_x(y)\n    return (x,)\n"
    bindings = {"f_x": njit(lambda y: 2 * y)}
    kernel = _compile(src, bindings, None, True)
    assert kernel(10) == (20,)


def test_compile_anchor_is_content_addressed():
    root = _anchor_root(_ANCHOR_SUBDIR)
    src = "def _kernel(y):\n    x = f_x(y)\n    return (x,)\n"
    _compile(src, {"f_x": njit(lambda y: 2 * y)}, None, True)
    anchors = sorted(p.name for p in root.glob("_kernel_*.py"))
    assert anchors, "expected at least one anchor file"
    # different formula source -> different anchor digest
    before = set(root.glob("_kernel_*.py"))
    _compile(src, {"f_x": njit(lambda y: 3 * y)}, None, True)
    after = set(root.glob("_kernel_*.py"))
    assert after - before, "different formula must produce a new anchor"
```

- [ ] **Step 2: Run to verify failure**

Run: `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )`
Expected: FAIL (ImportError on `_compile`/`_ANCHOR_SUBDIR`)

- [ ] **Step 3: Add `_compile` + anchor setup**

```python
# add near the top imports of numbox/core/variable/compile_kernel.py
from numbox.core.configurations import jit_options as _default_jit_options
from numbox.utils.preprocessing import (
    _anchor_path, _materialize_anchor, _orphan_anchor_sweep,
)

_ANCHOR_SUBDIR = "numbox-compile-kernel"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)
```

```python
# append to numbox/core/variable/compile_kernel.py
def _compile(source, bindings, jit_options, cache):
    """Content-addressed compile of the kernel source into an @njit dispatcher."""
    formula_src = "\n".join(_safe_getsource(f) for f in bindings.values())
    hash_text = source + "\n# formulas:\n" + formula_src
    digest = hashlib.sha256(hash_text.encode("utf-8")).hexdigest()[:16]
    name = f"_kernel_{digest}"
    opts = {**_default_jit_options, **(jit_options or {}), "cache": cache}
    final_src = "@njit(**_kernel_jit_options)\n" + source.replace(
        "def _kernel(", f"def {name}(", 1
    )
    anchor = _anchor_path(_ANCHOR_SUBDIR, name, hash_text)
    _materialize_anchor(anchor, final_src)
    code = compile(final_src, str(anchor), "exec")
    ns = {**bindings, "njit": njit, "_kernel_jit_options": opts}
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns[name]
```

- [ ] **Step 4: Run to verify pass**

Run: `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: content-addressed anchor + cached @njit compile"
```

---

### Task 4: Public API — `CompiledKernel` + `compile_kernel`

**Goal:** Tie the pieces together into the public `compile_kernel(graph, required)` returning a `CompiledKernel` with `.kernel`, `.execute`, `.params`, `.outputs`, `.source`, `.identifiers`.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py`
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] `compile_kernel(graph, required)` returns a `CompiledKernel`; `required` accepts `str | list[str]`.
- [ ] `ck.kernel(*ext)` and `ck.execute({src:{name:val}})` both equal `CompiledGraph.execute` for chain, diamond, mixed-type, multi-output, and single-output graphs.
- [ ] `ck.execute` raises `KeyError` (naming the qualified variable) when an external value is missing.
- [ ] Auto-specialization: the same `ck.kernel` works for int then float external input.

**Verify:** `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests**

```python
# append to test/core/test_compile_kernel.py
from numbox.core.variable.compile_kernel import compile_kernel, CompiledKernel
from numbox.core.variable.variable import Values


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
    # raw kernel: positional in ck.params order, tuple in ck.outputs order
    assert ck.params == ["basket.y"]
    assert ck.outputs == req
    assert tuple(ck.kernel(100)) == tuple(_pure(g, req, ext)[q] for q in req)


def test_compile_kernel_single_output_and_str_required():
    g = _diamond_graph()
    ck = compile_kernel(g, "variables.u")          # str accepted
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
    with pytest.raises(KeyError):
        ck.execute({"basket": {}})
```

- [ ] **Step 2: Run to verify failure**

Run: `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )`
Expected: FAIL (ImportError on `compile_kernel`/`CompiledKernel`)

- [ ] **Step 3: Add the wrapper + entry point**

```python
# add to imports of numbox/core/variable/compile_kernel.py
from numbox.core.variable.variable import make_qual_name
```

```python
# append to numbox/core/variable/compile_kernel.py
class CompiledKernel:
    """A fused @njit kernel compiled from a Variable graph.

    Attributes:
      kernel      - bare numba dispatcher; positional external args (in `params`
                    order) -> tuple (in `outputs` order). Zero-overhead hot path.
      params      - external input qual_names, kernel-argument order.
      outputs     - requested variable qual_names, return-tuple order.
      source      - generated kernel source text.
      identifiers - {qual_name: temp identifier} for inspection.
    """

    def __init__(self, kernel, params, outputs, source, identifiers):
        self.kernel = kernel
        self._param_keys = [(src, name) for src, name, _ in params]
        self.params = [make_qual_name(src, name) for src, name, _ in params]
        self.outputs = list(outputs)
        self.source = source
        self.identifiers = identifiers

    def execute(self, external_values):
        """Dict-in / dict-out convenience, symmetric with CompiledGraph.execute."""
        args = []
        for src, name in self._param_keys:
            try:
                args.append(external_values[src][name])
            except KeyError as e:
                raise KeyError(
                    f"Missing external value for {make_qual_name(src, name)!r}"
                ) from e
        result = self.kernel(*args)
        return dict(zip(self.outputs, result))


def compile_kernel(graph, required, *, jit_options=None, cache=True):
    """Compile `graph` into a fused @njit kernel for the `required` variables."""
    if isinstance(required, str):
        required = [required]
    required = list(required)
    compiled = graph.compile(required)
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    source, bindings, params, outputs = _generate_body(compiled, required, idents)
    kernel = _compile(source, bindings, jit_options, cache)
    identifiers = {v.qual_name(): ident for v, ident in idents.items()}
    return CompiledKernel(kernel, params, outputs, source, identifiers)
```

- [ ] **Step 4: Run to verify pass**

Run: `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: CompiledKernel wrapper + public compile_kernel entry point"
```

---

### Task 5: Robustness — identifier collisions, formula variety, error taxonomy (integration)

**Goal:** Prove the end-to-end pipeline survives adversarial names and formula kinds, and that lazy/eager errors behave as specified.

**Files:**
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] A graph using the `("a_b","c")` vs `("a","b_c")` collision pair compiles and computes both outputs correctly.
- [ ] A graph with an invalid-char / leading-digit external name compiles and runs.
- [ ] `cres`-compiled and `@njit` formulas are both callable from the kernel (or — if a `cres` `CompileResultWAP` global is not callable by name — the test documents that and the implementation requires `@njit`; see Spec §14.1).
- [ ] An auto-wrapped plain-Python formula works; a constant (no-input) formula works; an array-returning formula works.
- [ ] A non-jittable formula surfaces a numba error at first call (lazy), not at `compile_kernel()` time.

**Verify:** `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )` → all pass

**Steps:**

- [ ] **Step 1: Write the tests**

```python
# append to test/core/test_compile_kernel.py
import numpy as np
from numba.core.errors import TypingError
from numbox.utils.highlevel import cres
from numba.core.types import float64


def test_identifier_collision_graph_runs():
    # two sources whose qual_names sanitize to the same base: a_b.c and a.b_c
    g = Graph(
        variables_lists={
            "a_b": [{"name": "c", "inputs": {"y": "ext"}, "formula": njit(lambda y: y + 1)}],
            "a": [{"name": "b_c", "inputs": {"y": "ext"}, "formula": njit(lambda y: y + 2)}],
        },
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["a_b.c", "a.b_c"])
    assert ck.execute({"ext": {"y": 10}}) == {"a_b.c": 11, "a.b_c": 12}


def test_invalid_char_external_name_runs():
    g = Graph(
        variables_lists={"variables": [
            {"name": "out", "inputs": {"first-name": "ext"}, "formula": njit(lambda v: v * 2)},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.out"])
    assert ck.execute({"ext": {"first-name": 5}}) == {"variables.out": 10}


def test_cres_and_constant_and_array_formulas():
    add = cres(float64(float64, float64))(lambda a, b: a + b)
    g = Graph(
        variables_lists={"variables": [
            {"name": "k", "inputs": {}, "formula": njit(lambda: 7.0)},     # constant
            {"name": "u", "inputs": {"k": "variables", "y": "ext"}, "formula": add},
            {"name": "arr", "inputs": {"y": "ext"}, "formula": njit(lambda y: np.arange(y))},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.u", "variables.arr"])
    out = ck.execute({"ext": {"y": 3.0}})
    assert out["variables.u"] == 10.0
    assert list(out["variables.arr"]) == [0, 1, 2]


def test_non_jittable_formula_fails_at_first_call_not_compile():
    def bad(x):
        return open("/tmp/_ck_nope.txt", "w")     # unsupported in nopython
    g = Graph(
        variables_lists={"variables": [
            {"name": "b", "inputs": {"y": "ext"}, "formula": bad},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, ["variables.b"])        # compiles (lazy) — no raise here
    with pytest.raises((TypingError, Exception)):
        ck.execute({"ext": {"y": 1}})              # error surfaces at first call
```

- [ ] **Step 2: Run; if the `cres` assertion fails (CompileResultWAP not callable by global name)**, change `_wrap_formula` to also `njit`-wrap `CompileResultWAP` inputs OR document `cres` as unsupported (require `@njit`); update the test and Spec §14.1 to match the verified behavior. Re-run.

Run: `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py -q )`
Expected: PASS (after reconciling the `cres` finding)

- [ ] **Step 3: Commit**

```bash
git -C /home/erik/projects/numbox add test/core/test_compile_kernel.py numbox/core/variable/compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: robustness + formula-variety + error-taxonomy tests"
```

---

### Task 6: Cross-process cache-collision regression

**Goal:** Prove that two kernels with identical straight-line skeletons but different formulas do not collide in numba's on-disk cache across fresh processes (the verified footgun the content-addressed anchor fixes).

**Files:**
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] In a fresh subprocess, compiling two same-skeleton/different-formula kernels with `cache=True` yields each kernel's own correct result (no cross-load).
- [ ] Mirrors the existing pattern in `test/core/test_builder.py::test_make_graph_cache_key_content_independent`.

**Verify:** `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py::test_cache_no_skeleton_collision -q )` → pass

**Steps:**

- [ ] **Step 1: Read the existing pattern**

Read `test/core/test_builder.py::test_make_graph_cache_key_content_independent` to match the subprocess harness style used in this repo.

- [ ] **Step 2: Write the test**

```python
# append to test/core/test_compile_kernel.py
import subprocess
import sys
import textwrap


def test_cache_no_skeleton_collision(tmp_path):
    # Two kernels: identical skeleton (one leaf -> one formula -> return),
    # different formulas (x*10 vs x*1000). With cache=True they must NOT
    # load each other's cached binary.
    script = textwrap.dedent('''
        from numba import njit
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        def build(mult):
            g = Graph(
                variables_lists={"v": [
                    {"name": "o", "inputs": {"y": "e"}, "formula": njit(lambda y: y * mult)},
                ]},
                external_source_names=["e"],
            )
            return compile_kernel(g, ["v.o"], cache=True)

        a = build(10)
        b = build(1000)
        ra = a.execute({"e": {"y": 1}})["v.o"]
        rb = b.execute({"e": {"y": 1}})["v.o"]
        assert ra == 10, ra
        assert rb == 1000, rb
        print("OK", ra, rb)
    ''')
    f = tmp_path / "ck_cache_probe.py"
    f.write_text(script)
    # run twice in fresh interpreters so the second run reads the on-disk cache
    for _ in range(2):
        p = subprocess.run([sys.executable, str(f)], capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        assert "OK 10 1000" in p.stdout, p.stdout + p.stderr
```

- [ ] **Step 3: Run to verify pass**

Run: `( cd /home/erik/projects/numbox && venv/bin/python -m pytest test/core/test_compile_kernel.py::test_cache_no_skeleton_collision -q )`
Expected: PASS. If it fails (cross-load), the anchor hash in `_compile` is not capturing formula identity — fix `hash_text` to include every formula's source and re-run.

- [ ] **Step 4: Commit**

```bash
git -C /home/erik/projects/numbox add test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: cross-process cache-collision regression test"
```

---

### Task 7: Documentation

**Goal:** Document `compile_kernel` in the Sphinx docs and confirm the doc build + doc-codeblock lint are clean.

**Files:**
- Modify: `docs/numbox.core.variable.rst`

**Acceptance Criteria:**
- [ ] A `compile_kernel` section with prose, a worked example, and `automodule:: numbox.core.variable.compile_kernel`.
- [ ] `sphinx-build` exits 0 (warning count stable).
- [ ] `extract_codeblocks.py` reports the new `.rst` python blocks clean at max-line-length 120.

**Verify:**
```bash
( cd /home/erik/projects/numbox && venv/bin/sphinx-build -b html docs docs/_build/html ) && \
( cd /home/erik/projects/numbox && venv/bin/python .github/scripts/extract_codeblocks.py docs --flake8 venv/bin/flake8 )
```
→ sphinx exit 0; extract_codeblocks exit 0

**Steps:**

- [ ] **Step 1: Read the current doc** to match heading/automodule style: `docs/numbox.core.variable.rst` (the `[#f1]` footnote already points at the fully-JIT'ed path — cross-reference it).

- [ ] **Step 2: Append a section** to `docs/numbox.core.variable.rst` (keep code-block lines ≤ 120 chars):

```rst
Compiling a graph to a fused JIT kernel
+++++++++++++++++++++++++++++++++++++++

:func:`numbox.core.variable.compile_kernel.compile_kernel` compiles a graph into
a single fused ``@njit`` kernel for a requested set of variables. Unlike
:class:`numbox.core.work.work.Work`, it needs no per-node type information —
numba infers every interior type from the kernel's runtime arguments — provided
every ``formula`` is njit-able (plain-Python formulas are auto-wrapped)::

    from numba import njit
    from numbox.core.variable.variable import Graph
    from numbox.core.variable.compile_kernel import compile_kernel

    graph = Graph(
        variables_lists={"variables": [
            {"name": "x", "inputs": {"y": "basket"}, "formula": njit(lambda y: 2 * y)},
            {"name": "u", "inputs": {"x": "variables"}, "formula": njit(lambda x: x - 74)},
        ]},
        external_source_names=["basket"],
    )

    ck = compile_kernel(graph, ["variables.u"])
    assert ck.execute({"basket": {"y": 100}}) == {"variables.u": 126}
    assert ck.kernel(100) == (126,)

.. automodule:: numbox.core.variable.compile_kernel
   :members:
   :show-inheritance:
```

- [ ] **Step 3: Build + lint docs**

Run the **Verify** commands above. Fix any sphinx warning the new section introduces and any flake8 finding in the `.rst` python block.

- [ ] **Step 4: Commit**

```bash
git -C /home/erik/projects/numbox add docs/numbox.core.variable.rst
git -C /home/erik/projects/numbox commit -m "docs(variable): document compile_kernel (Variable graph -> fused @njit kernel)"
```

---

### Task 8: Full local CI gate + push readiness

**Goal:** Run every CI check locally (caches cleaned), confirm green, and ready the branch for the fork PR.

**Files:** none (verification only)

**Acceptance Criteria:**
- [ ] Full pytest suite passes (existing + new), caches cleaned first.
- [ ] `flake8` clean at the repo config (run from repo root).
- [ ] `doc-codeblock-flake8` (extract_codeblocks over `docs`) clean.
- [ ] `lychee` over changed docs clean (the new `.rst` has links, so the empty-diff failIfEmpty case does not apply).
- [ ] Branch ready; do NOT open the fork PR until the user approves (per workflow rules); upstream PR only on explicit per-PR approval.

**Verify:**
```bash
# clean caches
( cd /home/erik/projects/numbox && venv/bin/python -c "import shutil,pathlib,os; r=pathlib.Path('.'); [shutil.rmtree(p,ignore_errors=True) for p in r.rglob('__pycache__')]; shutil.rmtree(pathlib.Path(os.path.expanduser('~/.cache/numba')),ignore_errors=True)" )
# full suite
( cd /home/erik/projects/numbox && venv/bin/python -m pytest -q --durations=20 )
# lint (repo root for correct .flake8 discovery)
( cd /home/erik/projects/numbox && venv/bin/flake8 . --count --show-source --statistics )
# doc code blocks
( cd /home/erik/projects/numbox && venv/bin/python .github/scripts/extract_codeblocks.py docs --flake8 venv/bin/flake8 )
```
→ all green

**Steps:**

- [ ] **Step 1: Run the full gate** (Verify block). Fix anything red.
- [ ] **Step 2: Report** the test count + lint status to the user. Stop. Await approval before opening the fork PR (then upstream only on explicit approval; exclude `CLAUDE.md` and `docs/superpowers/**` from any upstream PR).

---

## Out of scope (v1 — documented, not bugs)

`cacheable` memoization, incremental `recompute`, `None`-as-value formulas,
node-identity `load`/`combine`/`harvest`, and a `Variable`→`Work` bridge. Use
`CompiledGraph` or `Work` for those (Spec §2, §7).
