# compile_kernel v2 Segmented Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved spec `docs/superpowers/specs/2026-06-12-compile-kernel-segments-design.md`: graphs with non-jittable formulas run through `compile_kernel` automatically — a Python master orchestrates fused `@njit` segments around auto-detected Python-only nodes, with fusion-maximizing linearization and a `PartitionReport` describing what runs where and why.

**Architecture:** All partition logic (report types, plan/steps, greedy linearizer, liveness, discovery probe) lives in a new private module `numbox/core/variable/_kernel_partition.py`. `compile_kernel.py` gains a generalized body emitter shared by the fused kernel and segments, and `CompiledKernel` becomes a three-mode state machine (virgin → fused | segmented) where `kernel` is a property returning the current hot-path callable. The fused path stays byte-identical to v1 (cache-digest stability is a golden test).

**Tech Stack:** Python 3.12 / numba 0.65.1 / numpy; pytest; sphinx. Venv interpreter: `/home/erik/projects/numbox/venv/bin/python` (always absolute, never bare `python`/`pytest`).

**Spec:** `docs/superpowers/specs/2026-06-12-compile-kernel-segments-design.md`. Section references (§N) below point there. Out of scope: cherry-picking to `upstream-pr/compile-kernel-hardened` / upstream PR #24 (separate, consent-gated step), any push (consent-gated), and any change to `numbox/core/variable/variable.py` or `numbox/utils/preprocessing.py` (upstream-owned substrate).

---

## Conventions (read once, used by every task)

- **Repo root:** `/home/erik/projects/numbox`; branch `feat/compile-kernel-hardened`. Never use `cd`; use absolute paths and `git -C /home/erik/projects/numbox`.
- **`<clean-caches>`** in Verify lines stands for this exact command (run before every pytest invocation):

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
```

- **`<pytest>`** stands for `/home/erik/projects/numbox/venv/bin/python -m pytest`.
- **Flake8 (both rule sets must pass before any commit):**

```bash
/home/erik/projects/numbox/venv/bin/python -m flake8 /home/erik/projects/numbox/numbox /home/erik/projects/numbox/test
/home/erik/projects/numbox/venv/bin/python -m flake8 --select=E9,F63,F7,F82,F401 /home/erik/projects/numbox/numbox /home/erik/projects/numbox/test
```

- Commit messages follow the branch's existing style (`compile_kernel: <what>`), one commit per task, no AI attribution, no Co-Authored-By.
- The feature test file is `test/core/test_compile_kernel.py`; new tests append there (existing imports at the top already cover `njit`, `Dispatcher`, `TypingError`, `Graph`, `Variable`, `Values`, `compile_kernel`). Add imports only when a snippet below needs one that is missing.
- No task numbers, phase references, or plan pointers in code comments.

---

## File structure

| File | Role in this plan |
|---|---|
| `numbox/core/variable/_kernel_partition.py` | **New, private.** Report types (`Segment`, `PartitionReport`), plan types (`_JitStep`, `_PyStep`, `_Plan`), greedy linearizer, run/liveness partitioner, discovery probe, exotic-formula shim. Tasks 0, 2. |
| `numbox/core/variable/compile_kernel.py` | Body-emitter refactor (Task 1), `CompiledKernel` state machine + `partition` (Tasks 3, 4), docstrings (Task 7). |
| `test/core/test_compile_kernel.py` | New tests for every task; v1 golden-source regression. |
| `test/compile_kernel_benchmark.py` | Task 6 only (`--python-nodes` mode). |
| `docs/numbox.core.variable.rst` | Task 7 ("Graphs with non-jittable nodes" subsection). |
| `docs/superpowers/specs/2026-06-12-compile-kernel-segments-design.md` | Task 0 amends §6 (best-of-both-starts greedy, see Task 0 Step 5). |

Dependency order: Task 1 is independent of Task 0; Tasks 2–3 need Task 0; Task 4 needs 0–3; Tasks 5–7 need 4; Task 8 needs all.

---

### Task 0: Partition data structures, plan walker, greedy linearizer, liveness

**Goal:** `_kernel_partition.py` exists with `Segment`/`PartitionReport`, `_JitStep`/`_PyStep`/`_Plan` (with a working `run`), `linearize` (greedy color-sticky, best-of-both-starts, deterministic), and `build_runs` + `segment_liveness` — all pure logic, unit-tested without any numba compilation.

**Files:**
- Create: `numbox/core/variable/_kernel_partition.py`
- Modify: `docs/superpowers/specs/2026-06-12-compile-kernel-segments-design.md` (§6, one paragraph)
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] `linearize` on the parallel-chains graph (py-chain + jit-chain joining at a jit sink) yields 2 runs, not 3 — the best-of-both-starts pick
- [ ] `linearize` is deterministic (same input → same order, twice) and respects every dependency edge
- [ ] `build_runs` groups consecutive same-color nodes; `segment_liveness` computes live-ins/outs sorted by qual_name; required outputs are always live-out
- [ ] `_Plan.run` threads values through stub callables (no numba) and returns outputs in order
- [ ] `str(PartitionReport)` renders mode, per-segment nodes, and demotion reasons

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -k "linearize or partition_report or plan_run or liveness" -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** (append to `test/core/test_compile_kernel.py`)

```python
def _nodes_from(graph, required):
    compiled = graph.compile(required)
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    return [n for n in compiled.ordered_nodes if n.variable not in external], external


def _parallel_chains_graph():
    # py-chain p1->p2 and jit-chain j1->j2 from one external, joined by jit sink.
    return Graph(
        variables_lists={"variables": [
            {"name": "p1", "inputs": {"x": "ext"}, "formula": lambda x: x + 1.0},
            {"name": "p2", "inputs": {"p1": "variables"}, "formula": lambda p1: p1 * 2.0},
            {"name": "j1", "inputs": {"x": "ext"}, "formula": lambda x: x - 1.0},
            {"name": "j2", "inputs": {"j1": "variables"}, "formula": lambda j1: j1 * 3.0},
            {"name": "out", "inputs": {"p2": "variables", "j2": "variables"},
             "formula": lambda p2, j2: p2 + j2},
        ]},
        external_source_names=["ext"],
    )


def test_linearize_minimizes_runs_on_parallel_chains():
    from numbox.core.variable._kernel_partition import build_runs, linearize
    g = _parallel_chains_graph()
    nodes, external = _nodes_from(g, ["variables.out"])
    demoted = {n.variable for n in nodes if n.variable.name in ("p1", "p2")}
    order = linearize(nodes, demoted)
    runs = build_runs(order, demoted)
    assert [kind for kind, _ in runs] == ["python", "jit"]  # 2 runs, not 3
    # dependencies respected
    pos = {n.variable.qual_name(): i for i, n in enumerate(order)}
    for n in order:
        for inp in n.inputs:
            if inp.qual_name() in pos:
                assert pos[inp.qual_name()] < pos[n.variable.qual_name()]


def test_linearize_deterministic():
    from numbox.core.variable._kernel_partition import linearize
    g = _parallel_chains_graph()
    nodes, _ = _nodes_from(g, ["variables.out"])
    demoted = {n.variable for n in nodes if n.variable.name == "j1"}
    first = [n.variable.qual_name() for n in linearize(nodes, demoted)]
    second = [n.variable.qual_name() for n in linearize(nodes, demoted)]
    assert first == second


def test_liveness_and_runs():
    from numbox.core.variable._kernel_partition import build_runs, linearize, segment_liveness
    g = _parallel_chains_graph()
    nodes, external = _nodes_from(g, ["variables.out"])
    by_name = {n.variable.name: n.variable for n in nodes}
    demoted = {by_name["p1"], by_name["p2"]}
    order = linearize(nodes, demoted)
    runs = build_runs(order, demoted)
    required_vars = [by_name["out"]]
    jit_run = next(r for kind, r in runs if kind == "jit")
    live_in, live_out = segment_liveness(jit_run, set(external), required_vars, order)
    in_names = [v.qual_name() for v in live_in]
    out_names = [v.qual_name() for v in live_out]
    assert in_names == sorted(in_names) and out_names == sorted(out_names)
    assert "ext.x" in in_names and "variables.p2" in in_names
    assert out_names == ["variables.out"]  # only the required sink escapes


def test_plan_run_threads_values():
    from numbox.core.variable._kernel_partition import _JitStep, _Plan, _PyStep
    ax = Variable(name="x", source="ext")
    av = Variable(name="v", source="calc")
    aw = Variable(name="w", source="calc")
    plan = _Plan(
        steps=(
            _JitStep(dispatcher=lambda x: (x + 1.0,), in_vars=(ax,), out_vars=(av,)),
            _PyStep(var=aw, py_callable=lambda v: v * 10.0, in_vars=(av,)),
        ),
        external_vars=(ax,),
        output_vars=(aw, av),
    )
    assert plan.run((2.0,)) == (30.0, 3.0)


def test_partition_report_str_and_python_nodes():
    from numbox.core.variable._kernel_partition import PartitionReport, Segment
    rep = PartitionReport(mode="segmented", segments=(
        Segment(kind="jit", nodes=("calc.a",), inputs=("ext.x",), outputs=("calc.a",),
                source="def _kernel(x):\n    return (x,)\n", reasons={}),
        Segment(kind="python", nodes=("calc.b",), inputs=("calc.a",), outputs=("calc.b",),
                source=None, reasons={"calc.b": "TypingError: nope"}),
    ))
    assert rep.python_nodes == {"calc.b"}
    text = str(rep)
    assert "segmented" in text and "calc.b" in text and "TypingError: nope" in text
```

- [ ] **Step 2: Run them — expect FAIL** (module `_kernel_partition` does not exist)

Run: `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -k "linearize or partition_report or plan_run or liveness" -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'numbox.core.variable._kernel_partition'`

- [ ] **Step 3: Create `numbox/core/variable/_kernel_partition.py`**

```python
"""Partitioning machinery for segmented compile_kernel.

Private module: discovery (warm-up + probe), fusion-maximizing
linearization, run/liveness partitioning, the master execution plan, and
the PartitionReport surfaced as CompiledKernel.partition. Everything here
operates on `core.variable` Variables/CompiledNodes and plain values; the
only numba interaction is Dispatcher.compile probes and typeof.
"""
from bisect import insort
from dataclasses import dataclass, field

from numba import njit, typeof
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaError


@dataclass(frozen=True)
class Segment:
    """One contiguous run of the partition, as reported to users."""
    kind: str                      # "jit" | "python"
    nodes: tuple                   # qual_names, linear order
    inputs: tuple                  # live-in qual_names
    outputs: tuple                 # live-out qual_names
    source: str = None             # generated source, jit segments only
    reasons: dict = field(default_factory=dict)   # python segments: demotion reasons


@dataclass(frozen=True)
class PartitionReport:
    """What actually runs where, and why -- see CompiledKernel.partition."""
    mode: str                      # "fused" | "segmented"
    segments: tuple                # of Segment

    @property
    def python_nodes(self):
        names = set()
        for seg in self.segments:
            if seg.kind == "python":
                names.update(seg.nodes)
        return names

    def __str__(self):
        lines = [f"compile_kernel partition: mode={self.mode}, {len(self.segments)} segment(s)"]
        for i, seg in enumerate(self.segments):
            lines.append(
                f"  [{i}] {seg.kind}: nodes={list(seg.nodes)} "
                f"inputs={list(seg.inputs)} outputs={list(seg.outputs)}"
            )
            for qual, why in sorted(seg.reasons.items()):
                lines.append(f"      {qual}: {why}")
        return "\n".join(lines)


@dataclass(frozen=True)
class _JitStep:
    dispatcher: object
    in_vars: tuple
    out_vars: tuple


@dataclass(frozen=True)
class _PyStep:
    var: object
    py_callable: object
    in_vars: tuple


@dataclass(frozen=True)
class _Plan:
    steps: tuple
    external_vars: tuple           # kernel-argument order
    output_vars: tuple             # required order

    def run(self, args):
        slots = dict(zip(self.external_vars, args))
        for step in self.steps:
            vals = [slots[v] for v in step.in_vars]
            if isinstance(step, _JitStep):
                slots.update(zip(step.out_vars, step.dispatcher(*vals)))
            else:
                slots[step.var] = step.py_callable(*vals)
        return tuple(slots[v] for v in self.output_vars)


def _qual(node):
    return node.variable.qual_name()


def _linearize_from(nodes, demoted, start_jit):
    by_var = {n.variable: n for n in nodes}
    indeg = {n: 0 for n in nodes}
    dependents = {}
    for n in nodes:
        for inp in n.inputs:
            producer = by_var.get(inp)
            if producer is not None:    # external inputs are pre-satisfied
                indeg[n] += 1
                dependents.setdefault(producer, []).append(n)
    jit_q, py_q = [], []
    for n in sorted((n for n in nodes if indeg[n] == 0), key=_qual):
        (py_q if n.variable in demoted else jit_q).append(n)
    order = []
    on_jit = start_jit
    while jit_q or py_q:
        queue = jit_q if on_jit else py_q
        if not queue:
            on_jit = not on_jit
            continue
        n = queue.pop(0)
        order.append(n)
        for d in sorted(dependents.get(n, []), key=_qual):
            indeg[d] -= 1
            if indeg[d] == 0:
                insort(py_q if d.variable in demoted else jit_q, d, key=_qual)
    return order


def linearize(nodes, demoted):
    """Topological order clustering same-color nodes into few long runs.

    Greedy color-sticky Kahn (drain the current color while possible,
    deterministic qual_name tie-break), evaluated from both possible
    starting colors; the candidate with fewer runs wins (jit-start on a
    tie). Exact run-count minimization is NP-hard; this is the documented
    heuristic.
    """
    jit_first = _linearize_from(nodes, demoted, start_jit=True)
    py_first = _linearize_from(nodes, demoted, start_jit=False)
    if len(build_runs(py_first, demoted)) < len(build_runs(jit_first, demoted)):
        return py_first
    return jit_first


def build_runs(order, demoted):
    """Split a linear order into maximal same-color runs: [(kind, [nodes])]."""
    runs = []
    for n in order:
        kind = "python" if n.variable in demoted else "jit"
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(n)
        else:
            runs.append((kind, [n]))
    return runs


def segment_liveness(run_nodes, external, required_vars, order):
    """(live_in, live_out) for one jit run, both sorted by qual_name.

    live_in: values the run consumes but does not produce (externals or
    earlier-step products). live_out: values the run produces that a later
    step consumes or that are required outputs.
    """
    produced = {n.variable for n in run_nodes}
    live_in = set()
    for n in run_nodes:
        for inp in n.inputs:
            if inp not in produced:
                live_in.add(inp)
    after = set()
    seen_any = False
    run_set = set(run_nodes)
    for n in order:
        if n in run_set:
            seen_any = True
            continue
        if seen_any:
            after.update(n.inputs)
    live_out = {v for v in produced if v in after or v in set(required_vars)}
    key = lambda v: v.qual_name()   # noqa: E731 - tiny local sort key
    return tuple(sorted(live_in, key=key)), tuple(sorted(live_out, key=key))
```

Note on `segment_liveness`: "later step" means after the run's *last* node in the linear order; nodes interleaved between two runs of the same color cannot exist (runs are maximal), so collecting consumers after the run's region is equivalent to collecting after each producer — except consumers *inside* the run, which `produced`-membership already excludes. The simple `seen_any` scan is correct because every consumer of a run-produced value is topologically after the run begins, and if it is not in the run it appears after the run ends.

- [ ] **Step 4: Run the tests — expect PASS**

Run: `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -k "linearize or partition_report or plan_run or liveness" -v`
Expected: 5 PASS

- [ ] **Step 5: Amend spec §6** — in `docs/superpowers/specs/2026-06-12-compile-kernel-segments-design.md`, replace the bullet "always drain the current color while its queue is non-empty; switch colors only when forced; initial color = jittable if any jittable node is ready." with:

```markdown
- always drain the current color while its queue is non-empty; switch colors
  only when forced; the linearization is computed from both possible starting
  colors and the candidate with fewer runs wins (jit-start on a tie) — a
  jit-start alone produces an avoidable extra run when a Python chain and a
  jit chain meet at a jit sink.
```

- [ ] **Step 6: Flake8 both passes, then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/_kernel_partition.py test/core/test_compile_kernel.py docs/superpowers/specs/2026-06-12-compile-kernel-segments-design.md
git -C /home/erik/projects/numbox commit -m "compile_kernel: partition module - report, plan walker, greedy linearizer, liveness"
```

---

### Task 1: Generalize body emission; lock v1 fused source byte-identical

**Goal:** `_generate_body` keeps its exact public behavior (byte-identical source — locked by a golden test) while the line/binding emission moves to a shared `_emit_lines`, and a new `_generate_segment_body(run_nodes, available, live_out, idents)` emits a segment kernel using live-ins as parameters and live-outs as the return tuple.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py:228-289` (`_generate_body`)
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] Golden test pins the v1 diamond-graph fused source to a literal expected string and passes
- [ ] `_generate_segment_body` output for a 2-node run has live-in params (qual-sorted), the same line format as v1, and a trailing-comma return of live-outs
- [ ] Entire existing suite still green (no behavioral change to the fused path)

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -v` → all pass incl. `test_fused_source_golden` and `test_generate_segment_body`

**Steps:**

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_fused_source_golden():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u", "variables.a"], cache=False)
    expected = (
        "def _kernel(basket_y):\n"
        "    variables_x = f_variables_x(basket_y)  # 'variables.x' = f('basket.y')\n"
        "    variables_a = f_variables_a(variables_x)  # 'variables.a' = f('variables.x')\n"
        "    variables_b = f_variables_b(variables_x)  # 'variables.b' = f('variables.x')\n"
        "    variables_u = f_variables_u(variables_a, variables_b)"
        "  # 'variables.u' = f('variables.a', 'variables.b')\n"
        "    return (variables_u, variables_a,)\n"
    )
    assert ck.source == expected


def test_generate_segment_body():
    from numbox.core.variable.compile_kernel import _generate_segment_body
    g = _diamond_graph()
    compiled = g.compile(["variables.u", "variables.a"])
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    nodes = [n for n in compiled.ordered_nodes if n.variable not in external]
    run = nodes[:2]                       # variables.x, variables.a
    by_name = {n.variable.name: n.variable for n in nodes}
    live_in = (next(iter(external)),)     # basket.y
    live_out = (by_name["a"], by_name["x"])
    source, bindings, params, outputs = _generate_segment_body(run, live_in, live_out, idents)
    assert source == (
        "def _kernel(basket_y):\n"
        "    variables_x = f_variables_x(basket_y)  # 'variables.x' = f('basket.y')\n"
        "    variables_a = f_variables_a(variables_x)  # 'variables.a' = f('variables.x')\n"
        "    return (variables_a, variables_x,)\n"
    )
    assert [p[2] for p in params] == ["basket_y"]
    assert outputs == ["variables.a", "variables.x"]
    assert set(bindings) == {"f_variables_x", "f_variables_a"}
```

- [ ] **Step 2: Run them — expect golden PASS already, segment test FAIL** (`_generate_segment_body` undefined). If the golden test fails, STOP — the literal string above mis-transcribes v1 output; fix the test to match actual `ck.source` exactly (print it), never the other way around.

Run: `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -k "golden or segment_body" -v`
Expected: `test_fused_source_golden` PASS, `test_generate_segment_body` FAIL (ImportError)

- [ ] **Step 3: Refactor.** In `compile_kernel.py`, extract the per-node loop of `_generate_body` (lines 254-274) into a shared emitter and add the segment generator:

```python
def _emit_lines(nodes, skip, idents, bindings):
    """Emit one body line per node (excluding `skip`), filling `bindings`."""
    lines = []
    for node in nodes:
        var = node.variable
        if var in skip:
            continue
        if var.formula is None:
            raise ValueError(
                f"{var.qual_name()!r} has graph placement but no formula; a fused "
                f"kernel cannot compile it. Provide a formula, or use CompiledGraph."
            )
        temp = idents[var]
        fg = "f_" + temp
        try:
            bindings[fg] = _wrap_formula(var.formula)
        except TypeError as e:
            raise TypeError(f"{var.qual_name()!r}: {e}") from e
        _check_formula_arity(var.formula, len(node.inputs), var.qual_name())
        arg_ids = ", ".join(idents[inp] for inp in node.inputs)
        in_names = ", ".join(repr(inp.qual_name()) for inp in node.inputs)
        lines.append(f"    {temp} = {fg}({arg_ids})  # {var.qual_name()!r} = f({in_names})")
    return lines


def _generate_segment_body(run_nodes, live_in, live_out, idents):
    """Like _generate_body, for one jit segment: live-ins are parameters,
    live-outs the return tuple. Same source shape so _compile applies verbatim.

    Returns (source, bindings, params, outputs) with params/outputs in the
    caller-provided (qual-sorted) live_in/live_out order.
    """
    params = [(v.source, v.name, idents[v]) for v in live_in]
    bindings = {}
    lines = _emit_lines(run_nodes, set(), idents, bindings)
    outputs = [v.qual_name() for v in live_out]
    out_ids = [idents[v] for v in live_out]
    sig = ", ".join(ident for _, _, ident in params)
    ret = f"    return ({', '.join(out_ids)},)"
    body = ("\n".join(lines) + "\n") if lines else ""
    source = f"def _kernel({sig}):\n{body}{ret}\n"
    return source, bindings, params, outputs
```

Then rewrite `_generate_body`'s node loop to use the emitter — replace its `bindings = {}` / `lines = []` / `for node in compiled.ordered_nodes:` block (current lines 254-274) with:

```python
    bindings = {}
    lines = _emit_lines(compiled.ordered_nodes, external, idents, bindings)
```

(The external-with-formula validation, params construction, outputs mapping, and final source assembly in `_generate_body` stay exactly as they are — the golden test proves the output is unchanged.)

- [ ] **Step 4: Run the full feature file — expect all green**

Run: `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -v`
Expected: all pass (existing suite + both new tests)

- [ ] **Step 5: Flake8 both passes, then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: shared body emitter + segment codegen, golden-lock fused source"
```

---

### Task 2: Discovery — one-pass warm-up + probe

**Goal:** `discover()` walks nodes in topo order with real values: probes `Dispatcher.compile(typeof(inputs))`, demotes on numba compile errors or untypeable inputs (recording reasons), evaluates every node (compiled dispatcher / `py_func` / exotic shim), propagates runtime errors, and returns the value table + demotions.

**Files:**
- Modify: `numbox/core/variable/_kernel_partition.py`
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] Non-jittable formula (unsupported module use) demoted with a `TypingError:`-prefixed reason; its value computed via Python
- [ ] Object-returning formula demoted; each downstream consumer demoted with an `is not numba-typeable` reason naming the input
- [ ] Jittable nodes' values computed via the compiled dispatcher (probe compile reused)
- [ ] A formula raising `ZeroDivisionError` propagates it — no demotion entry for that node
- [ ] Exotic formula (cres `CompileResultWAP`) evaluated via shim; works for a matching signature

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -k "discover" -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** (append; `json` import to the top of the test file, plus a module-level holder class)

```python
import json  # goes at the top of the file with the other stdlib imports


class _Opaque:
    """A value numba.typeof cannot type; arithmetic works in Python."""
    def __init__(self, v):
        self.v = v


def _bindings_by_var(graph, required):
    compiled = graph.compile(required)
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    _, bindings, _, _ = _generate_body(compiled, required, idents)
    by_var = {
        n.variable: bindings["f_" + idents[n.variable]]
        for n in compiled.ordered_nodes if n.variable not in external
    }
    return compiled, external, by_var


def test_discover_demotes_unjittable_and_keeps_values():
    from numbox.core.variable._kernel_partition import discover

    def uses_json(v):
        json.dumps({"k": 1})
        return v * 3.0

    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": lambda x: x + 1.0},
            {"name": "b", "inputs": {"a": "calc"}, "formula": uses_json},
            {"name": "c", "inputs": {"b": "calc"}, "formula": lambda b: b - 0.5},
        ]},
        external_source_names=["ext"],
    )
    compiled, external, by_var = _bindings_by_var(g, ["calc.c"])
    ext_x = next(iter(external))
    values = {ext_x: 2.0}
    demoted = discover(compiled.ordered_nodes, external, values, by_var)
    reasons = {v.qual_name(): r for v, r in demoted.items()}
    assert set(reasons) == {"calc.b"}
    assert reasons["calc.b"].startswith("TypingError:")
    by_name = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    assert values[by_name["calc.c"]] == (2.0 + 1.0) * 3.0 - 0.5


def test_discover_demotes_object_chain():
    from numbox.core.variable._kernel_partition import discover
    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": lambda x: _Opaque(x)},
            {"name": "b", "inputs": {"a": "calc"}, "formula": lambda a: a.v * 2.0},
            {"name": "c", "inputs": {"b": "calc"}, "formula": lambda b: b + 1.0},
        ]},
        external_source_names=["ext"],
    )
    compiled, external, by_var = _bindings_by_var(g, ["calc.c"])
    values = {next(iter(external)): 4.0}
    demoted = discover(compiled.ordered_nodes, external, values, by_var)
    reasons = {v.qual_name(): r for v, r in demoted.items()}
    assert set(reasons) == {"calc.a", "calc.b"}
    assert "is not numba-typeable" in reasons["calc.b"]
    by_name = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    assert values[by_name["calc.c"]] == 4.0 * 2.0 + 1.0


def test_discover_runtime_error_propagates():
    from numbox.core.variable._kernel_partition import discover
    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": lambda x: x / 0},
        ]},
        external_source_names=["ext"],
    )
    compiled, external, by_var = _bindings_by_var(g, ["calc.a"])
    values = {next(iter(external)): 1}
    with pytest.raises(ZeroDivisionError):
        discover(compiled.ordered_nodes, external, values, by_var)


def test_discover_exotic_cres_via_shim():
    from numbox.core.variable._kernel_partition import discover
    fn = cres(float64(float64))(lambda x: x * 5.0)
    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": fn},
        ]},
        external_source_names=["ext"],
    )
    compiled, external, by_var = _bindings_by_var(g, ["calc.a"])
    values = {next(iter(external)): 3.0}
    demoted = discover(compiled.ordered_nodes, external, values, by_var)
    assert demoted == {}
    by_name = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    assert values[by_name["calc.a"]] == 15.0
```

(`lambda x: x / 0` with an int input divides by zero in nopython too; numba raises `ZeroDivisionError` at runtime after a successful compile, so the test exercises "runtime error after probe-compile success", not a typing failure.)

- [ ] **Step 2: Run them — expect FAIL** (`discover` undefined)

Run: `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -k "discover" -v`
Expected: 4 ImportError/FAIL

- [ ] **Step 3: Implement in `_kernel_partition.py`** (append)

```python
_REASON_LIMIT = 200


def _error_reason(exc):
    first = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {first[:_REASON_LIMIT]}"


def _untypeable_reason(node, values):
    for inp in node.inputs:
        try:
            typeof(values[inp])
        except (ValueError, TypeError):
            return (
                f"input '{inp.qual_name()}' value of type "
                f"{type(values[inp]).__name__} is not numba-typeable"
            )
    return None


def _call_exotic(binding, args, arg_types):
    """Evaluate a CompileResultWAP/CFunc/DUFunc formula through a one-line
    @njit shim (the same global-binding shape segments use). No Python
    fallback exists for these, so a NumbaError here propagates."""
    names = ", ".join(f"a{i}" for i in range(len(args)))
    ns = {"_formula": binding}
    exec(f"def _shim({names}):\n    return _formula({names})\n", ns)  # nosec B102
    shim = njit(ns["_shim"])
    shim.compile(arg_types)
    return shim(*args)


def discover(ordered_nodes, external, values, bindings_by_var):
    """One-pass warm-up + probe (spec section 5).

    Mutates `values` ({Variable: value}, pre-seeded with externals) to hold
    every node's value; returns {Variable: reason} for demoted nodes.
    Numba *compile* errors demote; runtime errors propagate.
    """
    demoted = {}
    for node in ordered_nodes:
        var = node.variable
        if var in external:
            continue
        args = [values[inp] for inp in node.inputs]
        binding = bindings_by_var[var]
        reason = _untypeable_reason(node, values)
        arg_types = None
        if reason is None:
            arg_types = tuple(typeof(a) for a in args)
            if isinstance(binding, Dispatcher):
                try:
                    binding.compile(arg_types)
                except NumbaError as e:
                    reason = _error_reason(e)
            else:
                values[var] = _call_exotic(binding, args, arg_types)
                continue
        elif not isinstance(binding, Dispatcher):
            raise TypeError(
                f"{var.qual_name()!r}: formula {binding!r} has no Python fallback "
                f"and {reason}"
            )
        if reason is None:
            values[var] = binding(*args)
        else:
            demoted[var] = reason
            py = getattr(var.formula, "py_func", var.formula)
            values[var] = py(*args)
    return demoted
```

- [ ] **Step 4: Run the tests — expect PASS**

Run: `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -k "discover" -v`
Expected: 4 PASS

- [ ] **Step 5: Flake8 both passes, then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/_kernel_partition.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: discovery pass - probe, demote with reasons, exotic shim"
```

---

### Task 3: CompiledKernel state machine — virgin → fused path

**Goal:** `CompiledKernel` becomes mode-aware: `kernel` is a property (resolver while virgin, bare dispatcher once fused), the first successful call sets `mode="fused"` and a single-jit-segment `PartitionReport`; all v1 behavior and the whole existing suite remain green.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py:342-444` (`CompiledKernel`, `compile_kernel`)
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] `ck.partition is None` before any call; after one call on an all-jittable graph it is a `PartitionReport(mode="fused")` with one jit segment listing all non-external nodes, the fused source, empty reasons
- [ ] After the first call, `ck.kernel` **is** the bare numba dispatcher (`isinstance(ck.kernel, Dispatcher)`)
- [ ] A reference grabbed before the first call (`f = ck.kernel; f(...)`) still works and resolves the mode
- [ ] Entire existing suite green

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -v` → all pass incl. `test_fused_mode_resolution`

**Steps:**

- [ ] **Step 1: Write the failing test** (append)

```python
def test_fused_mode_resolution():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u", "variables.a"], cache=False)
    assert ck.partition is None
    early = ck.kernel                      # resolver grabbed before first call
    assert not isinstance(early, Dispatcher)
    assert early(100) == (326.5, 126)
    assert ck.partition is not None
    assert ck.partition.mode == "fused"
    (seg,) = ck.partition.segments
    assert seg.kind == "jit"
    assert seg.nodes == ("variables.x", "variables.a", "variables.b", "variables.u")
    assert seg.inputs == ("basket.y",)
    assert seg.outputs == ("variables.u", "variables.a")
    assert seg.source == ck.source and seg.reasons == {}
    assert isinstance(ck.kernel, Dispatcher)
    assert ck.kernel(100) == (326.5, 126)
    assert early(100) == (326.5, 126)      # early ref still valid post-resolution
```

- [ ] **Step 2: Run it — expect FAIL** (`ck.partition` attribute does not exist)

Run: `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py::test_fused_mode_resolution -v`
Expected: FAIL with AttributeError on `partition`

- [ ] **Step 3: Rework `CompiledKernel` and the `compile_kernel` tail.** Replace the `CompiledKernel` class body (`__init__` and `execute`; keep the class docstring, updating the `kernel` line to "hot-path callable: resolver before the first call, the bare numba dispatcher once fused, the segmented master otherwise") and the last lines of `compile_kernel`:

```python
    def __init__(self, kernel, params, outputs, source, identifiers, ctx=None):
        self._fused = kernel
        self._mode = "virgin"
        self._plan = None
        self.partition = None
        self._ctx = ctx
        self._param_keys = [(src, name) for src, name, _ in params]
        self.params = [make_qual_name(src, name) for src, name, _ in params]
        self.outputs = list(outputs)
        self.source = source
        self.identifiers = identifiers

    @property
    def kernel(self):
        if self._mode == "fused":
            return self._fused
        if self._mode == "segmented":
            return self._run_segmented
        return self._resolve_and_call

    def _fused_report(self):
        compiled, _, _, _, _, external = self._ctx
        nodes = tuple(
            n.variable.qual_name() for n in compiled.ordered_nodes
            if n.variable not in external
        )
        return PartitionReport(mode="fused", segments=(Segment(
            kind="jit", nodes=nodes, inputs=tuple(self.params),
            outputs=tuple(self.outputs), source=self.source, reasons={},
        ),))

    def _resolve_and_call(self, *args):
        if self._mode != "virgin":
            return self.kernel(*args)
        try:
            arg_types = tuple(typeof(a) for a in args)
        except (ValueError, TypeError):
            arg_types = None
        if arg_types is not None:
            try:
                self._fused.compile(arg_types)
            except NumbaError:
                pass
            else:
                self._mode = "fused"
                self.partition = self._fused_report()
                return self._fused(*args)
        return self._discover_and_run(args)

    def _run_segmented(self, *args):
        try:
            return self._plan.run(args)
        except NumbaError:
            return self._discover_and_run(args)

    def _discover_and_run(self, args):
        """Segmented execution; not yet wired -- defer to the fused dispatcher
        so error timing stays exactly v1 until the segmented path lands."""
        return self._fused(*args)

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
```

New imports at the top of `compile_kernel.py` (extend the existing import block):

```python
from numba import njit, typeof
from numba.core.errors import NumbaError
from numbox.core.variable._kernel_partition import (
    PartitionReport, Segment, discover, linearize, build_runs,
    segment_liveness, _JitStep, _Plan, _PyStep,
)
```

(`discover`/`linearize`/`build_runs`/`segment_liveness`/step types are consumed in Task 4; importing them now keeps the import block stable across the two commits — F401 does not fire because Task 4 lands before the gate, but if running the strict flake8 pass at this commit flags them, import only `PartitionReport, Segment` here and add the rest in Task 4.)

And the tail of `compile_kernel()` (last two statements) becomes:

```python
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    bindings_by_var = {
        n.variable: bindings["f_" + idents[n.variable]]
        for n in compiled.ordered_nodes if n.variable not in external
    }
    required_vars = [
        next(n.variable for n in compiled.ordered_nodes if n.variable.qual_name() == q)
        for q in outputs
    ]
    ctx = (compiled, idents, bindings_by_var, jit_options, cache, external)
    identifiers = {v.qual_name(): ident for v, ident in idents.items()}
    ck = CompiledKernel(kernel, params, outputs, source, identifiers, ctx)
    ck._required_vars = required_vars
    ck._external_vars = [
        next(v for v in external if v.source == src and v.name == name)
        for src, name in ck._param_keys
    ]
    return ck
```

- [ ] **Step 4: Run the whole feature file — expect all green** (the existing suite exercises `.kernel` and `.execute` heavily; this is the v1-compat gate)

Run: `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -v`
Expected: all pass

- [ ] **Step 5: Flake8 both passes, then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: mode state machine, kernel property, fused PartitionReport"
```

---

### Task 4: Segmented path — discovery integration, segment compile, master, plan replacement

**Goal:** `_discover_and_run` is real: discovery → linearize → runs → liveness → per-run segment codegen + `_compile` + eager `dispatcher.compile` → `_Plan` + segmented `PartitionReport`; the warm-up call returns correct outputs; later calls run the plan; a segment compile error on new types replaces the plan.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (`CompiledKernel._discover_and_run`)
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] Goykhman's example: chain `n1→n2→n3→n4→n5` with `n3` non-jittable → partition is exactly jit `(n1,n2)`, python `(n3)`, jit `(n4,n5)`; calls 1 and 2 both equal the `CompiledGraph` result
- [ ] Mixed graph with object chain (`_Opaque`): consecutive demoted nodes grouped into one python segment; results correct on calls 1 and 2
- [ ] All-demoted graph: zero jit segments, correct results
- [ ] Plan replacement: float signature learns a 2-segment plan; an `_Opaque` external breaks the first segment → re-discovery replaces the plan (all-python), correct results for the new and the old signature
- [ ] `ck.execute` works in segmented mode

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -k "segmented or goykhman or plan_replacement" -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** (append; `_Opaque` gains operator support — extend the class added in Task 2 to exactly this)

```python
class _Opaque:
    """A value numba.typeof cannot type; arithmetic works in Python."""
    def __init__(self, v):
        self.v = v

    def __mul__(self, other):
        return _Opaque(self.v * other)

    def __add__(self, other):
        return _Opaque(self.v + other)


def _chain_graph_with_python_middle():
    def n3(v):
        json.dumps({"k": 1})
        return v * 3.0

    return Graph(
        variables_lists={"calc": [
            {"name": "n1", "inputs": {"x": "ext"}, "formula": lambda x: x + 1.0},
            {"name": "n2", "inputs": {"n1": "calc"}, "formula": lambda n1: n1 * 2.0},
            {"name": "n3", "inputs": {"n2": "calc"}, "formula": n3},
            {"name": "n4", "inputs": {"n3": "calc"}, "formula": lambda n3: n3 - 4.0},
            {"name": "n5", "inputs": {"n4": "calc"}, "formula": lambda n4: n4 / 2.0},
        ]},
        external_source_names=["ext"],
    )


def _compiled_graph_result(graph, required, external_values):
    compiled = graph.compile(required)
    values = Values()
    compiled.execute(external_values, values)
    by_qual = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    return {q: values.get(by_qual[q]).value for q in required}


def test_goykhman_example_two_segments():
    g = _chain_graph_with_python_middle()
    ck = compile_kernel(g, "calc.n5", cache=False)
    expected = _compiled_graph_result(
        _chain_graph_with_python_middle(), ["calc.n5"], {"ext": {"x": 7.0}}
    )
    assert ck.execute({"ext": {"x": 7.0}}) == expected          # call 1: warm-up
    assert ck.execute({"ext": {"x": 7.0}}) == expected          # call 2: plan
    rep = ck.partition
    assert rep.mode == "segmented"
    kinds = [(s.kind, s.nodes) for s in rep.segments]
    assert kinds == [
        ("jit", ("calc.n1", "calc.n2")),
        ("python", ("calc.n3",)),
        ("jit", ("calc.n4", "calc.n5")),
    ]
    assert rep.python_nodes == {"calc.n3"}
    assert rep.segments[1].reasons["calc.n3"].startswith("TypingError:")
    assert rep.segments[0].source is not None and rep.segments[1].source is None


def test_segmented_object_chain_groups_python_run():
    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": lambda x: x * 2.0},
            {"name": "b", "inputs": {"a": "calc"}, "formula": lambda a: _Opaque(a)},
            {"name": "c", "inputs": {"b": "calc"}, "formula": lambda b: b.v + 1.0},
            {"name": "d", "inputs": {"c": "calc"}, "formula": lambda c: c * 10.0},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, "calc.d", cache=False)
    assert ck.kernel(3.0) == ((3.0 * 2.0 + 1.0) * 10.0,)
    assert ck.kernel(3.0) == ((3.0 * 2.0 + 1.0) * 10.0,)
    kinds = [(s.kind, s.nodes) for s in ck.partition.segments]
    assert ("python", ("calc.b", "calc.c")) in kinds


def test_segmented_all_python():
    def f(x):
        json.dumps({"k": 1})
        return x + 2.0

    g = Graph(
        variables_lists={"calc": [{"name": "a", "inputs": {"x": "ext"}, "formula": f}]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, "calc.a", cache=False)
    assert ck.kernel(1.0) == (3.0,)
    assert ck.kernel(1.0) == (3.0,)
    assert all(s.kind == "python" for s in ck.partition.segments)


def test_plan_replacement_on_new_signature():
    g = _chain_graph_with_python_middle()
    ck = compile_kernel(g, "calc.n5", cache=False)
    assert ck.kernel(7.0) == (((7.0 + 1.0) * 2.0 * 3.0) - 4.0) / 2.0,
    assert len([s for s in ck.partition.segments if s.kind == "jit"]) == 2
    out, = ck.kernel(_Opaque(7.0))          # breaks segment 1 -> re-discovery
    assert isinstance(out, _Opaque) and out.v == (((7.0 + 1.0) * 2.0 * 3.0) - 4.0) / 2.0
    assert [s.kind for s in ck.partition.segments] == ["python"]
    assert ck.kernel(7.0) == (((7.0 + 1.0) * 2.0 * 3.0) - 4.0) / 2.0,
```

(In `test_plan_replacement_on_new_signature` the trailing-comma lines compare against a 1-tuple, matching the kernel's tuple return.)

- [ ] **Step 2: Run them — expect FAIL** (the `_discover_and_run` stub defers to the fused dispatcher, so these graphs raise numba typing errors instead of partitioning)

Run: `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -k "segmented or goykhman or plan_replacement" -v`
Expected: 4 FAIL (TypingError/UnsupportedBytecodeError surfacing through the stub)

- [ ] **Step 2b: Rewrite the two v1 tests whose semantics this feature deliberately changes.** v2's whole point is that graphs which used to fail numba typing at first call now succeed via demotion (spec §9: "fused attempt fails to type → silent fallback to discovery"). Two pre-existing tests assert the OLD contract and must be rewritten to the new one — read each test body first, preserve its protective intent:
  - `test_non_jittable_formula_fails_at_first_call_not_compile`: the `pytest.raises(TypingError)` at first call becomes: the call **succeeds** (value asserted against the formula's Python result), `ck.partition.mode == "segmented"`, the non-jittable node's qual_name is in `ck.partition.python_nodes`, and a reason is recorded for it. Keep the part of the test asserting that `compile_kernel()` itself raises nothing (eager/lazy split — structural errors eager, jittability resolution at first call).
  - `test_object_array_closure_kernel_uncached`: the subprocess's `TypingError` expectation becomes call-succeeds-via-demotion. Preserve the test's protective intent — no on-disk cache entries are written for this graph: with the only node demoted there are zero jit segments, so the cache-dir-empty assertion stays valid; keep it.

- [ ] **Step 3: Implement `_discover_and_run`** (replace the stub in `CompiledKernel`)

```python
    def _discover_and_run(self, args):
        compiled, idents, bindings_by_var, jit_options, cache, external = self._ctx
        values = dict(zip(self._external_vars, args))
        demoted = discover(compiled.ordered_nodes, external, values, bindings_by_var)
        nodes = [n for n in compiled.ordered_nodes if n.variable not in external]
        order = linearize(nodes, set(demoted))
        runs = build_runs(order, set(demoted))
        steps, segments = [], []
        for kind, run_nodes in runs:
            quals = tuple(n.variable.qual_name() for n in run_nodes)
            if kind == "python":
                ins = set()
                produced = set()
                for n in run_nodes:
                    ins.update(i for i in n.inputs if i not in produced)
                    produced.add(n.variable)
                    steps.append(_PyStep(
                        var=n.variable,
                        py_callable=getattr(n.variable.formula, "py_func", n.variable.formula),
                        in_vars=tuple(n.inputs),
                    ))
                reasons = {n.variable.qual_name(): demoted[n.variable] for n in run_nodes}
                key = lambda v: v.qual_name()   # noqa: E731 - tiny local sort key
                segments.append(Segment(
                    kind="python", nodes=quals,
                    inputs=tuple(v.qual_name() for v in sorted(ins, key=key)),
                    outputs=quals, source=None, reasons=reasons,
                ))
                continue
            live_in, live_out = segment_liveness(
                run_nodes, external, self._required_vars, order
            )
            src, seg_bindings, _, _ = _generate_segment_body(
                run_nodes, live_in, live_out, idents
            )
            disp = _compile(src, seg_bindings, jit_options, cache)
            disp.compile(tuple(typeof(values[v]) for v in live_in))
            steps.append(_JitStep(dispatcher=disp, in_vars=live_in, out_vars=live_out))
            segments.append(Segment(
                kind="jit", nodes=quals,
                inputs=tuple(v.qual_name() for v in live_in),
                outputs=tuple(v.qual_name() for v in live_out),
                source=src, reasons={},
            ))
        self._plan = _Plan(
            steps=tuple(steps),
            external_vars=tuple(self._external_vars),
            output_vars=tuple(self._required_vars),
        )
        self._mode = "segmented"
        self.partition = PartitionReport(mode="segmented", segments=tuple(segments))
        return tuple(values[v] for v in self._required_vars)
```

- [ ] **Step 4: Run the whole feature file — expect all green**

Run: `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -v`
Expected: all pass (existing suite + Tasks 0-4 tests)

- [ ] **Step 5: Flake8 both passes, then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: segmented execution - discovery integration, per-segment compile, master plan, replacement"
```

---

### Task 5: Per-segment caching across processes

**Goal:** Prove segments are content-addressed cached: a fresh subprocess re-discovers, and its segment compiles hit numba's on-disk cache (cache files exist and outputs match); two identical jit runs in one process share a digest harmlessly.

**Files:**
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] Subprocess A (own `NUMBA_CACHE_DIR`) runs the mixed chain graph with `cache=True`, asserts segmented mode + correct outputs; cache dir gains `*.nbc` entries
- [ ] Subprocess B (same `NUMBA_CACHE_DIR`) repeats and passes; `*.nbc` count does not grow (pure cache hits)
- [ ] Same-digest sharing: a graph with two identical-source jit runs compiles and runs correctly

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -k "segment_cache or shared_digest" -v` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** (append)

```python
_SEGMENT_CACHE_SCRIPT = textwrap.dedent("""
    import json, sys
    from numba import njit
    from numbox.core.variable.compile_kernel import compile_kernel
    from numbox.core.variable.variable import Graph

    def n3(v):
        json.dumps({"k": 1})
        return v * 3.0

    g = Graph(
        variables_lists={"calc": [
            {"name": "n1", "inputs": {"x": "ext"}, "formula": lambda x: x + 1.0},
            {"name": "n2", "inputs": {"n1": "calc"}, "formula": lambda n1: n1 * 2.0},
            {"name": "n3", "inputs": {"n2": "calc"}, "formula": n3},
            {"name": "n4", "inputs": {"n3": "calc"}, "formula": lambda n3: n3 - 4.0},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, "calc.n4", cache=True)
    out = ck.kernel(7.0)
    assert out == (((7.0 + 1.0) * 2.0 * 3.0) - 4.0,), out
    assert ck.partition.mode == "segmented", ck.partition.mode
    sys.exit(0)
""")


def test_segment_cache_survives_subprocess_roundtrip(tmp_path):
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "nbcache")}
    for attempt in ("save", "load"):
        proc = subprocess.run(
            [sys.executable, "-c", _SEGMENT_CACHE_SCRIPT],
            capture_output=True, text=True, env=env,
        )
        assert proc.returncode == 0, f"{attempt}: {proc.stderr}"
        nbc = list((tmp_path / "nbcache").rglob("*.nbc"))
        if attempt == "save":
            assert nbc, "segment compile produced no cache entries"
            saved = len(nbc)
        else:
            assert len(nbc) == saved, "second process recompiled instead of loading"


def test_shared_digest_identical_segments():
    def n3(v):
        json.dumps({"k": 1})
        return v + 0.0

    g = Graph(
        variables_lists={"calc": [
            {"name": "a", "inputs": {"x": "ext"}, "formula": lambda x: x * 2.0},
            {"name": "p", "inputs": {"a": "calc"}, "formula": n3},
            {"name": "b", "inputs": {"p": "calc"}, "formula": lambda x: x * 2.0},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(g, "calc.b", cache=False)
    assert ck.kernel(3.0) == (3.0 * 2.0 * 2.0,)
    assert ck.kernel(3.0) == (3.0 * 2.0 * 2.0,)
```

(The two jit runs in `test_shared_digest_identical_segments` each contain one node whose formula is the same `lambda x: x * 2.0`-shaped body; whether their digests actually collide depends on identifier assignment, which is fine either way — the assertion is correctness, the digest-sharing is exercised opportunistically. The subprocess pair is the real cache gate.)

- [ ] **Step 2: Run — first test FAILS only if the segmented cache path is broken; both may pass immediately.** That is acceptable: this task is a verification gate, not new production code. If `test_segment_cache_survives_subprocess_roundtrip` fails on the `load` attempt with a growing `*.nbc` count, the segment source or digest is unstable across processes — debug `_generate_segment_body` determinism (identifier assignment must not depend on dict order) before proceeding.

Run: `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -k "segment_cache or shared_digest" -v`
Expected: 2 PASS (investigate per above if not)

- [ ] **Step 3: Flake8 both passes, then commit**

```bash
git -C /home/erik/projects/numbox add test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: cross-process segment cache + shared-digest tests"
```

---

### Task 6: Benchmark `--python-nodes` mode

**Goal:** `test/compile_kernel_benchmark.py` gains `--python-nodes K`: inject K evenly-spaced non-jittable nodes into the N-node chain, then report discovery (first-call) time, segmented steady-state vs `CompiledGraph` wall time, and the segment count from `ck.partition`.

**Files:**
- Modify: `test/compile_kernel_benchmark.py`

**Acceptance Criteria:**
- [ ] `--python-nodes 0` output unchanged from today (flag default keeps every existing mode untouched)
- [ ] `--python-nodes 3 --nodes 100` runs end to end and prints: segment count, demoted count, first-call (discovery) seconds, steady-state ns/node for segmented kernel and for `CompiledGraph`
- [ ] flake8-clean

**Verify:** `<clean-caches> && /home/erik/projects/numbox/venv/bin/python /home/erik/projects/numbox/test/compile_kernel_benchmark.py --nodes 100 --python-nodes 3` → prints the report block; exit 0

**Steps:**

- [ ] **Step 1: Read the benchmark's existing argument and graph-construction sections** (`test/compile_kernel_benchmark.py`; it already builds an N-node chain of generated formulas in a real module and times fused vs `CompiledGraph`). Add an argparse option `--python-nodes` (type int, default 0, help "inject K evenly-spaced non-jittable nodes into the chain").

- [ ] **Step 2: Implement injection + report.** Where the chain formulas are generated, when `--python-nodes K > 0`, replace the formula of every `i*(N//(K+1))`-th interior node (i = 1..K) with a generated non-jittable variant of the same arithmetic body that calls `json.dumps({"k": 1})` on its first line (mirroring the per-line emission pattern the benchmark already uses for distinct formulas — the injected ones live in the same generated module with `import json` at its top). Then add a timing block, following the existing timing-helper style of the file:

```python
def run_python_nodes_mode(graph, required, n_nodes, k_python, args_value):
    from numbox.core.variable.compile_kernel import compile_kernel
    t0 = time.perf_counter()
    ck = compile_kernel(graph, required)
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    first = ck.execute({"ext": {"x0": args_value}})
    t_first = time.perf_counter() - t0
    rep = ck.partition
    n_seg = len([s for s in rep.segments if s.kind == "jit"])
    n_py = len(rep.python_nodes)
    steady = time_callable(lambda: ck.execute({"ext": {"x0": args_value}}))
    cg = time_compiled_graph(graph, required, {"ext": {"x0": args_value}})
    print(f"python-nodes mode: N={n_nodes} K={k_python}")
    print(f"  partition: {n_seg} jit segment(s), {n_py} python node(s)")
    print(f"  compile_kernel() wall: {t_compile:.3f}s; first call (discovery): {t_first:.3f}s")
    print(f"  steady-state: segmented {steady / n_nodes:.1f} ns/node vs "
          f"CompiledGraph {cg / n_nodes:.1f} ns/node")
    return first
```

Adapt `time_callable` / `time_compiled_graph` to the helpers that already exist in the file (same repeat counts and best-of policy the fused-vs-CompiledGraph mode uses); if the file's helpers have different names, use those — do not duplicate timing loops.

- [ ] **Step 3: Run both acceptance commands**

Run: `<clean-caches> && /home/erik/projects/numbox/venv/bin/python /home/erik/projects/numbox/test/compile_kernel_benchmark.py --nodes 100 --python-nodes 3`
Expected: report block printed, exit 0
Run: `/home/erik/projects/numbox/venv/bin/python /home/erik/projects/numbox/test/compile_kernel_benchmark.py --nodes 100`
Expected: existing output, no behavior change

- [ ] **Step 4: Flake8 both passes, then commit**

```bash
git -C /home/erik/projects/numbox add test/compile_kernel_benchmark.py
git -C /home/erik/projects/numbox commit -m "compile_kernel benchmark: --python-nodes mode for segmented orchestration"
```

---

### Task 7: Docs — rst subsection, docstrings, module docstring

**Goal:** User-facing docs tell the v2 truth: `docs/numbox.core.variable.rst` gains a "Graphs with non-jittable nodes" subsection (auto-detection, `PartitionReport`, the worked `N1→N5` example, determinism/caching, plan-replacement limitation); `compile_kernel`'s docstring and the module docstring stop claiming every formula must be njit-able.

**Files:**
- Modify: `docs/numbox.core.variable.rst` (compile_kernel section)
- Modify: `numbox/core/variable/compile_kernel.py` (module docstring lines 1-14; `compile_kernel` docstring lines 378-407; `CompiledKernel` docstring)

**Acceptance Criteria:**
- [ ] rst subsection present with a flake8-clean `code-block:: python` worked example showing a json-using middle node, `ck.partition` inspection, and `str(ck.partition)`
- [ ] Module docstring describes both modes and the auto-demotion contract (compile errors demote, runtime errors propagate, exotic formulas have no fallback)
- [ ] `compile_kernel` docstring documents `.partition`, the `.kernel` hot-path-callable semantics, and the plan-replacement limitation
- [ ] `sphinx-build` exits 0 with warning count ≤ the branch baseline (64)

**Verify:**

```bash
/home/erik/projects/numbox/venv/bin/sphinx-build -b html /home/erik/projects/numbox/docs /tmp/numbox-docs-v2 2>&1 | tee /tmp/sphinx-v2.log; grep -c "WARNING" /tmp/sphinx-v2.log
```

→ exit 0, warning count ≤ 64

**Steps:**

- [ ] **Step 1: Write the rst subsection** (append to the compile_kernel section of `docs/numbox.core.variable.rst`):

```rst
Graphs with non-jittable nodes
------------------------------

``compile_kernel`` detects non-jittable formulas automatically at the first
call: it first tries to compile the fully fused kernel for the actual
argument types; if that fails, it probes each node against the real
intermediate values, runs the offenders in plain Python, and fuses the
jittable remainder into as few ``@njit`` segments as a greedy linearization
allows. A Python master then threads values between segments and Python
nodes. Compile-time failures demote a node; runtime errors always propagate.

.. code-block:: python

    import json

    from numbox.core.variable.compile_kernel import compile_kernel
    from numbox.core.variable.variable import Graph

    def n3(v):
        json.dumps({"k": 1})    # no nopython lowering for the json module
        return v * 3.0

    graph = Graph(
        variables_lists={"calc": [
            {"name": "n1", "inputs": {"x": "ext"}, "formula": lambda x: x + 1.0},
            {"name": "n2", "inputs": {"n1": "calc"}, "formula": lambda n1: n1 * 2.0},
            {"name": "n3", "inputs": {"n2": "calc"}, "formula": n3},
            {"name": "n4", "inputs": {"n3": "calc"}, "formula": lambda n3: n3 - 4.0},
            {"name": "n5", "inputs": {"n4": "calc"}, "formula": lambda n4: n4 / 2.0},
        ]},
        external_source_names=["ext"],
    )
    ck = compile_kernel(graph, "calc.n5")
    ck.kernel(7.0)              # first call: probes, partitions, still correct
    print(str(ck.partition))    # 2 jit segments around the python n3, with reasons

``ck.partition`` is ``None`` until the first call resolves the mode; a fully
fused graph reports a single jit segment. Each jit segment is cached
content-addressed on disk exactly like a v1 kernel; the learned partition
itself is per-process. If a later call's types break a segment, the partition
is re-learned for those values and replaces the previous plan — workloads
alternating between type families whose partitions differ re-pay discovery on
each alternation.
```

- [ ] **Step 2: Rewrite the module docstring** (`compile_kernel.py` lines 1-14) to:

```python
"""Compile a `core.variable` Variable graph into fused @njit kernel(s).

Alongside `core.work` (a structref graph), this turns a `Graph`/`CompiledGraph`
into JIT-compiled straight-line code. When every formula is njit-able the
whole graph becomes a single fused @njit function whose interior nodes are SSA
temporaries (no per-node type info needed: numba infers every interior type
from the kernel's runtime argument types). When some formulas are not
njit-able, the first call detects them automatically -- numba compile errors
demote a node to plain Python, runtime errors always propagate -- and a Python
master orchestrates fused @njit segments around the demoted nodes, with a
fusion-maximizing linearization choosing the segment boundaries. The resulting
partition is described by `CompiledKernel.partition` (a PartitionReport with
per-node demotion reasons); formulas with no Python fallback
(cres/CompileResultWAP, CFunc, DUFunc) are always treated as jittable.

The on-disk cache is content-addressed per compiled unit (the fused kernel, or
each jit segment): the digest fingerprints each formula's code, constants,
default arguments, closure-cell values, referenced globals, and the kernel's
effective jit flags, so a stale binary is never reused and two distinct
kernels never collide. A formula with no canonical fingerprint forces its unit
uncached (no anchor, no numba cache) -- never reused, never wrong.
"""
```

- [ ] **Step 3: Extend the `compile_kernel` docstring** — after the "Error timing:" paragraph (line 392), insert:

```python
    Non-jittable formulas: the first call resolves the execution mode. If the
    fully fused kernel cannot be typed for the actual argument types, each
    node is probed against the real intermediate values; nodes whose formulas
    fail to *compile* (or whose input values numba cannot type) run in plain
    Python, and the jittable remainder is fused into segments orchestrated
    from Python. `CompiledKernel.partition` describes the result, including
    per-node demotion reasons; it is `None` before the first call. Runtime
    errors never demote -- they propagate. `CompiledKernel.kernel` is the
    hot-path callable: the bare @njit dispatcher once the graph resolves
    fully fused, the Python master when segmented. A later call whose types
    break a segment re-learns and replaces the partition (one active plan).
```

- [ ] **Step 4: Update the `CompiledKernel` class docstring** `kernel` attribute line (if not already done in Task 3) and add `partition` to the attribute list:

```python
      partition   - PartitionReport describing what runs where (None until
                    the first call resolves the mode).
```

- [ ] **Step 5: Run sphinx + doc-codeblock checks**

```bash
/home/erik/projects/numbox/venv/bin/sphinx-build -b html /home/erik/projects/numbox/docs /tmp/numbox-docs-v2 2>&1 | tee /tmp/sphinx-v2.log; grep -c "WARNING" /tmp/sphinx-v2.log
```

Expected: exit 0, ≤ 64 warnings. Also run both flake8 passes (the rst code block is linted by the doc-codeblock CI; replicate locally by extracting it to a temp file and running flake8 on it if `.github/workflows/` has a doc-codeblock job — check `ls /home/erik/projects/numbox/.github/workflows/` and mirror whatever doc lint exists there).

- [ ] **Step 6: Commit**

```bash
git -C /home/erik/projects/numbox add docs/numbox.core.variable.rst numbox/core/variable/compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: document segmented orchestration and PartitionReport"
```

---

### Task 8: Full local CI gate

**Goal:** Every check the fork CI runs passes locally before any push is even proposed; the plan's tasks.json reflects completion.

**Files:**
- Modify: `docs/superpowers/plans/2026-06-12-compile-kernel-segments.md.tasks.json` (statuses)

**Acceptance Criteria:**
- [ ] Full test suite green (not just the feature file), caches cleaned first
- [ ] Both flake8 passes clean
- [ ] sphinx exit 0, warnings ≤ 64
- [ ] Every check in `.github/workflows/` replicated locally (list them; run each one's core command)
- [ ] No push performed (consent-gated, outside this plan)

**Verify:** all commands below exit 0

**Steps:**

- [ ] **Step 1: Full suite**

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
/home/erik/projects/numbox/venv/bin/python -m pytest /home/erik/projects/numbox/test --durations=20 -q
```

Expected: 0 failures (skips per existing platform gates are fine)

- [ ] **Step 2: Lint + docs**

```bash
/home/erik/projects/numbox/venv/bin/python -m flake8 /home/erik/projects/numbox/numbox /home/erik/projects/numbox/test
/home/erik/projects/numbox/venv/bin/python -m flake8 --select=E9,F63,F7,F82,F401 /home/erik/projects/numbox/numbox /home/erik/projects/numbox/test
/home/erik/projects/numbox/venv/bin/sphinx-build -b html /home/erik/projects/numbox/docs /tmp/numbox-docs-gate 2>&1 | tee /tmp/sphinx-gate.log; grep -c "WARNING" /tmp/sphinx-gate.log
```

Expected: empty flake8 output ×2; sphinx exit 0, ≤ 64 warnings

- [ ] **Step 3: Mirror remaining workflows.** `ls /home/erik/projects/numbox/.github/workflows/` and for each workflow beyond numbox_ci (doc-codeblock-flake8, link-check, …) run its core command locally against the files this plan changed (`compile_kernel.py`, `_kernel_partition.py`, both test files, `variable.rst`, the two superpowers docs). For link-check: `lychee` over the changed `.py/.rst/.md` files (offline-tolerant flags as the workflow uses).

Expected: each mirrored check clean

- [ ] **Step 4: Update tasks.json statuses to completed** (this file, ids 0-8), commit

```bash
git -C /home/erik/projects/numbox add docs/superpowers/plans/2026-06-12-compile-kernel-segments.md.tasks.json
git -C /home/erik/projects/numbox commit -m "docs(plan): compile_kernel segments tasks complete"
```

- [ ] **Step 5: STOP.** Report gate results. Pushing `feat/compile-kernel-hardened`, cherry-picking to `upstream-pr/compile-kernel-hardened`, and any #24/#52 activity are consent-gated user decisions outside this plan.
