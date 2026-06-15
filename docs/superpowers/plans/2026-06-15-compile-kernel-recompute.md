# CompiledKernel.recompute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `CompiledKernel.recompute(changed)` — incremental refresh of a compiled Variable graph that re-evaluates only the affected cone, re-fused, reading unchanged values from a persistent Python-side store.

**Architecture:** Cone-based lazy re-fusion over a Python-side value store with boundary-granularity persistence (dominator-based): a node is persisted iff it is a required output or fails to dominate a consumer; fusion happens within boundaries. Cone sub-plans are cached (LRU, keyed on cone + live-in boundary) with graceful degradation; interior overrides are supported via lazy change-source expansion. Mirrors the value-only, same-types contract of `CompiledGraph.recompute`.

**Tech Stack:** Python 3.12, numba 0.65.1, the existing `compile_kernel` v2 segmentation machinery (`discover`/`linearize`/`build_runs`/`_generate_segment_body`/`_compile`).

**Spec:** `docs/superpowers/specs/2026-06-15-compile-kernel-recompute-design.md` (authoritative; section refs below point into it).

---

## Conventions (used by every task)

- **Branch:** `feat/compile-kernel-hardened` (existing — do NOT create a new branch). Use `git -C /home/erik/projects/numbox ...`; never `cd`.
- **venv python:** `/home/erik/projects/numbox/venv/bin/python`
- **Clean caches before EVERY pytest run** (run this first, exactly):
  ```bash
  /home/erik/projects/numbox/venv/bin/python -c "import shutil,pathlib; [shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home()/'.cache'/'numba',ignore_errors=True)"
  ```
- **Run tests:** `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_compile_kernel.py -v`
- **Lint (repo `.flake8`, max-line-length=127):** `/home/erik/projects/numbox/venv/bin/flake8 numbox/core/variable/_kernel_partition.py numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py`
- **No person names** in any code, test, comment, or commit message.
- **Model:** each task is implemented with **opus** + maximum reasoning.
- TDD throughout: write the failing test, see it fail for the right reason, implement minimally, see it pass, lint, commit.

---

### Task 1: Boundary analysis (dominator-based)

**Goal:** Pure helpers `_dominators` and `compute_boundary` in `_kernel_partition.py` that compute the set of Variables that must be persisted (required outputs + nodes that fail to dominate a consumer w.r.t. the change-source set). Spec §5.

**Files:**
- Modify: `numbox/core/variable/_kernel_partition.py`
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] `compute_boundary(nodes, sources, required)` returns `{required}` for a pure chain (interior nodes are fuse-through).
- [ ] For the diamond `_diamond_graph()` with `required={variables.u}` and `sources={basket.y}`, boundary is `{variables.a, variables.b, variables.u}` (`variables.x` fuse-through).
- [ ] `compute_boundary` equals an independent brute-force reference (reachability-to-consumer-avoiding-node) on chain, diamond, fan-out, and a two-source reconvergent graph.

**Verify:** clean caches, then `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_compile_kernel.py -k boundary -v` → all pass.

**Steps:**

- [ ] **Step 1: Write the failing tests** (append to `test/core/test_compile_kernel.py`):

```python
from numbox.core.variable._kernel_partition import compute_boundary  # add to imports


def _compiled_parts(g, req):
    cg = g.compile(req)
    by_qual = {n.variable.qual_name(): n.variable for n in cg.ordered_nodes}
    sources = {v for vs in cg.required_external_variables.values() for v in vs.values()}
    required = {by_qual[q] for q in req}
    return cg, by_qual, sources, required


def _boundary_reference(cg, sources, required):
    by_var = {n.variable: n for n in cg.ordered_nodes}
    succ = {}
    for n in cg.ordered_nodes:
        for inp in n.inputs:
            if inp in by_var:
                succ.setdefault(inp, set()).add(n.variable)
    seeded = {n.variable for n in cg.ordered_nodes
              if n.variable in sources or any(i not in by_var for i in n.inputs)}

    def reachable_avoiding(target, avoid):
        seen, stack = set(), [s for s in seeded if s != avoid]
        while stack:
            cur = stack.pop()
            if cur == target:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(nxt for nxt in succ.get(cur, ()) if nxt != avoid)
        return False

    boundary = set(required)
    for n in cg.ordered_nodes:
        x = n.variable
        if any(reachable_avoiding(c, x) for c in succ.get(x, ())):
            boundary.add(x)
    return boundary


def _chain_graph():
    return Graph(
        variables_lists={"variables": [
            {"name": "m", "inputs": {"y": "basket"}, "formula": njit(lambda y: y + 1.0)},
            {"name": "n", "inputs": {"m": "variables"}, "formula": njit(lambda m: m * 2.0)},
            {"name": "p", "inputs": {"n": "variables"}, "formula": njit(lambda n: n - 3.0)},
        ]},
        external_source_names=["basket"],
    )


def _fanout_graph():
    return Graph(
        variables_lists={"variables": [
            {"name": "o1", "inputs": {"a": "basket"}, "formula": njit(lambda a: a + 1.0)},
            {"name": "o2", "inputs": {"b": "basket"}, "formula": njit(lambda b: b + 2.0)},
        ]},
        external_source_names=["basket"],
    )


def _two_source_reconvergent_graph():
    return Graph(
        variables_lists={"variables": [
            {"name": "p", "inputs": {"a": "basket"}, "formula": njit(lambda a: a + 1.0)},
            {"name": "q", "inputs": {"b": "basket"}, "formula": njit(lambda b: b - 1.0)},
            {"name": "j", "inputs": {"p": "variables", "q": "variables"},
             "formula": njit(lambda p, q: p * q)},
            {"name": "r", "inputs": {"j": "variables"}, "formula": njit(lambda j: j + 5.0)},
        ]},
        external_source_names=["basket"],
    )


def test_boundary_chain_is_required_only():
    g = _chain_graph()
    cg, _, sources, required = _compiled_parts(g, ["variables.p"])
    b = {v.qual_name() for v in compute_boundary(cg.ordered_nodes, sources, required)}
    assert b == {"variables.p"}


def test_boundary_diamond_persists_join_inputs():
    g = _diamond_graph()
    cg, _, sources, required = _compiled_parts(g, ["variables.u"])
    b = {v.qual_name() for v in compute_boundary(cg.ordered_nodes, sources, required)}
    assert b == {"variables.a", "variables.b", "variables.u"}


@pytest.mark.parametrize("factory,req", [
    (_chain_graph, ["variables.p"]),
    (_diamond_graph, ["variables.u"]),
    (_fanout_graph, ["variables.o1", "variables.o2"]),
    (_two_source_reconvergent_graph, ["variables.r"]),
])
def test_boundary_matches_reference(factory, req):
    g = factory()
    cg, _, sources, required = _compiled_parts(g, req)
    got = compute_boundary(cg.ordered_nodes, sources, required)
    assert got == _boundary_reference(cg, sources, required)
```

- [ ] **Step 2: Run tests to verify they fail.** Run the Verify command. Expected: `ImportError`/`cannot import name 'compute_boundary'`.

- [ ] **Step 3: Implement `_dominators` and `compute_boundary`** (append to `numbox/core/variable/_kernel_partition.py`):

```python
def _dominators(nodes: list[CompiledNode], sources: set[Variable]) -> dict[Variable, frozenset]:
    """Dominator set per node for the DAG rooted at a virtual super-source
    (represented by None) with an edge to every node that is in `sources` or
    has an external input. `nodes` must be topologically ordered. Each node's
    set includes the node itself."""
    by_var = {n.variable: n for n in nodes}
    dom: dict[Variable, frozenset] = {}
    for n in nodes:
        var = n.variable
        seeds = [dom[inp] for inp in n.inputs if inp in by_var]
        if var in sources or any(inp not in by_var for inp in n.inputs):
            seeds.append(frozenset({None}))
        inter = frozenset.intersection(*seeds) if seeds else frozenset({None})
        dom[var] = inter | {var}
    return dom


def compute_boundary(
    nodes: list[CompiledNode], sources: set[Variable], required: set[Variable],
) -> set[Variable]:
    """Variables that must live in the recompute store: required outputs, plus
    any node that fails to dominate a consumer (some change in `sources` can
    reach the consumer without reaching the node, making it a cone live-in).
    Nodes that dominate all their consumers are fuse-through (not persisted)."""
    by_var = {n.variable: n for n in nodes}
    dom = _dominators(nodes, sources)
    consumers: dict[Variable, list[Variable]] = {}
    for n in nodes:
        for inp in n.inputs:
            if inp in by_var:
                consumers.setdefault(inp, []).append(n.variable)
    boundary = set(required)
    for n in nodes:
        var = n.variable
        if any(var not in dom[c] for c in consumers.get(var, ())):
            boundary.add(var)
    return boundary
```

- [ ] **Step 4: Run tests to verify they pass.** Run the Verify command. Expected: PASS.

- [ ] **Step 5: Lint.** Run the flake8 command (Conventions). Expected: no output.

- [ ] **Step 6: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/_kernel_partition.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: dominator-based recompute boundary analysis"
```

---

### Task 2: Cone-liveness helper

**Goal:** Pure helper `cone_liveness(run_nodes, cone_order, required_vars, boundary)` in `_kernel_partition.py` returning `(live_in, live_out)` for one jit run inside a cone, where `live_out` includes cross-segment-consumed, required, **and boundary** nodes (so the cone persists exactly what other cones may read). Spec §7.2, §11.

**Files:**
- Modify: `numbox/core/variable/_kernel_partition.py`
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] For a single-run cone, `live_out` = produced ∩ (required ∪ boundary); interior fuse-through nodes are excluded.
- [ ] For a two-run cone, a value produced in run 1 and consumed in run 2 appears in run 1's `live_out` even if not required/boundary.
- [ ] `live_in` = inputs consumed but not produced within the run; both tuples sorted by `qual_name`.

**Verify:** clean caches, then `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_compile_kernel.py -k cone_liveness -v` → pass.

**Steps:**

- [ ] **Step 1: Write the failing test:**

```python
from numbox.core.variable._kernel_partition import cone_liveness  # add to imports


def test_cone_liveness_single_run_uses_required_and_boundary():
    g = _diamond_graph()
    cg, by_qual, _, _ = _compiled_parts(g, ["variables.u"])
    nodes = {n.variable: n for n in cg.ordered_nodes}
    a, b, u, x = (by_qual["variables.a"], by_qual["variables.b"],
                  by_qual["variables.u"], by_qual["variables.x"])
    run = [nodes[a], nodes[b], nodes[u]]          # one fused run over the cone of x
    boundary = {a, b, u}
    live_in, live_out = cone_liveness(run, run, {u}, boundary)
    assert set(live_out) == {a, u}                 # b not produced-by-later; a in boundary
    assert x in set(live_in)                       # x consumed by a,b, produced outside run
    assert list(live_out) == sorted(live_out, key=lambda v: v.qual_name())
```

- [ ] **Step 2: Run to verify it fails** (ImportError).

- [ ] **Step 3: Implement `cone_liveness`** (append to `_kernel_partition.py`):

```python
def cone_liveness(
    run_nodes: list[CompiledNode], cone_order: list[CompiledNode],
    required_vars, boundary: set[Variable],
) -> tuple[tuple[Variable, ...], tuple[Variable, ...]]:
    """(live_in, live_out) for one jit run inside a recompute cone.

    live_in: inputs consumed by the run but not produced within it (read from
    the store). live_out: produced values that are consumed by a later cone
    step, are required outputs, or are boundary nodes (persisted for other
    cones). Both sorted by qual_name."""
    produced = {n.variable for n in run_nodes}
    live_in = set()
    for n in run_nodes:
        for inp in n.inputs:
            if inp not in produced:
                live_in.add(inp)
    later = set()
    seen = False
    run_set = set(run_nodes)
    for n in cone_order:
        if n in run_set:
            seen = True
            continue
        if seen:
            later.update(n.inputs)
    req = set(required_vars)
    live_out = {v for v in produced if v in later or v in req or v in boundary}
    key = lambda v: v.qual_name()   # noqa: E731 - tiny local sort key
    return tuple(sorted(live_in, key=key)), tuple(sorted(live_out, key=key))
```

- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Lint.**
- [ ] **Step 6: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/_kernel_partition.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: cone-liveness helper for recompute write-back set"
```

---

### Task 3: Value store, seeding, and `recompute` precondition + no-op path

**Goal:** Give `CompiledKernel` a persistent value store seeded by the first full call (segmented: reuse the discarded `values` dict; fused: one `discover` pass from captured args), retain demotion verdicts, and add `recompute(changed)` that enforces the seed precondition and handles the empty/no-op change. Spec §6, §7.1, §7.4.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (`CompiledKernel.__init__`, `_resolve_and_call`, `_discover_and_run`, new `recompute`, new `_apply_changes`, `_ensure_store`)
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] `recompute` before any full call raises `RuntimeError` naming the precondition.
- [ ] After `ck.kernel(...)` (fused path) then `ck.recompute({})`, the return equals `ck.kernel(...)` outputs and the store holds correct values for all required outputs.
- [ ] After a segmented first call, `recompute({})` returns the same outputs and the store was seeded from the retained `values` dict (no second graph evaluation).

**Verify:** clean caches, then `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_compile_kernel.py -k "recompute_precondition or recompute_noop" -v` → pass.

**Steps:**

- [ ] **Step 1: Write the failing tests:**

```python
def test_recompute_precondition_before_full_call():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    with pytest.raises(RuntimeError) as exc:
        ck.recompute({"basket": {"y": 100}})
    assert "recompute" in str(exc.value).lower()


def test_recompute_noop_matches_kernel_fused():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u", "variables.a"])
    full = tuple(ck.kernel(100))               # resolves fused, seeds nothing yet
    assert tuple(ck.recompute({})) == full     # no-op recompute seeds store, returns same
```

- [ ] **Step 2: Run to verify they fail** (`AttributeError: 'CompiledKernel' object has no attribute 'recompute'`).

- [ ] **Step 3: Implement.** In `CompiledKernel.__init__`, add state (after the existing assignments):

```python
        self._store = None          # {Variable: value}; seeded on first recompute
        self._demoted = {}          # {Variable: reason}; frozen demotion verdicts
        self._last_args = None      # external args of the most recent full resolution
        self._sources = None        # change-source Variables (externals; grows in Task 6)
        self._boundary = None       # set[Variable]; computed lazily (Task 4)
        self._cone_cache = None     # OrderedDict; created in Task 5
```

In `_resolve_and_call`, capture args once at entry (it only runs while virgin):

```python
    def _resolve_and_call(self, *args) -> tuple:
        if self._mode != "virgin":
            return self.kernel(*args)
        self._last_args = args            # NEW: one-time capture for fused-mode seeding
        try:
            arg_types = tuple(typeof(a) for a in args)
        ...
```

In `_discover_and_run`, retain the values dict + demotion verdicts (just before `self._mode = "segmented"`):

```python
        self._store = values              # NEW: seed store from the already-computed values
        self._demoted = demoted           # NEW: freeze demotion verdicts for cone builds
```

Add the seeding helper and `recompute` skeleton:

```python
    def _ensure_store(self):
        if self._store is not None:
            return
        if self._mode == "virgin" or self._last_args is None:
            raise RuntimeError(
                "CompiledKernel.recompute requires a prior full call: call the kernel "
                "once with the current inputs to seed the value store before recompute()."
            )
        # fused first call discarded interiors; seed once from the captured args.
        compiled, _, bindings_by_var, jit_options, cache, external = self._ctx
        flags = _effective_flags(jit_options)
        values = dict(zip(self._external_vars, self._last_args))
        self._demoted = discover(compiled.ordered_nodes, external, values, bindings_by_var, flags)
        self._store = values

    def _apply_changes(self, changed: dict) -> set:
        """Write changed values into the store, return the set of changed Variables.
        External names resolve via required_external_variables; interior names via
        ordered_nodes (Task 6 extends this to expand the change-source set)."""
        compiled = self._ctx[0]
        changed_vars = set()
        for src, vals in changed.items():
            for name, val in vals.items():
                var = compiled.required_external_variables.get(src, {}).get(name)
                if var is None:
                    qual = make_qual_name(src, name)
                    var = next((n.variable for n in compiled.ordered_nodes
                                if n.variable.qual_name() == qual), None)
                    if var is None:
                        warnings.warn(f"{qual} is not in the calculation path, update has no effect.")
                        continue
                self._store[var] = val
                changed_vars.add(var)
        return changed_vars

    def recompute(self, changed: dict) -> tuple:
        self._ensure_store()
        changed_vars = self._apply_changes(changed)
        if not changed_vars:
            return tuple(self._store[v] for v in self._required_vars)
        # Task 4 fills in cone build + execution here.
        return tuple(self._store[v] for v in self._required_vars)
```

Ensure `make_qual_name` and `discover` are imported (already imported in `compile_kernel.py`; `warnings` already imported).

- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Lint.**
- [ ] **Step 6: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: recompute value store, seeding, precondition + no-op path"
```

---

### Task 4: Cone build, `_ConePlan.run_into`, and incremental recompute (uncached)

**Goal:** Build a fused/segmented sub-plan over the affected cone using the boundary (Task 1) + cone-liveness (Task 2), execute it in place against the store via a new `_ConePlan.run_into`, and wire `recompute` to do real incremental refresh for external changes. Equivalence vs the interpreted `CompiledGraph.recompute` across multi-call sequences. Spec §4, §5, §10, §11.

**Files:**
- Modify: `numbox/core/variable/_kernel_partition.py` (`_ConePlan`)
- Modify: `numbox/core/variable/compile_kernel.py` (`_ensure_boundary`, `_build_cone_plan`, `recompute`)
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] Chain, diamond, and fan-out: a sequence of external recomputes equals the interpreted `CompiledGraph.recompute` value-for-value across ≥3 successive different change-sets (this is where the stale-`A` diamond bug surfaces).
- [ ] Mixed jit+Python cone (`_chain_graph_with_python_middle`): recompute matches interpreted across multiple calls.
- [ ] A runtime error in a cone formula propagates (no demotion).

**Verify:** clean caches, then `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_compile_kernel.py -k "recompute_equiv or recompute_runtime_error" -v` → pass.

**Steps:**

- [ ] **Step 1: Write the failing tests** (oracle = interpreted `CompiledGraph.recompute`):

```python
def _interp_recompute_sequence(g, req, ext, changes):
    """Reference: interpreted full execute then a sequence of recomputes."""
    cg = g.compile(req)
    by_qual = {n.variable.qual_name(): n.variable for n in cg.ordered_nodes}
    values = Values()
    cg.execute(ext, values)
    out = [tuple(values.get(by_qual[q]).value for q in req)]
    for ch in changes:
        cg.recompute(ch, values)
        out.append(tuple(values.get(by_qual[q]).value for q in req))
    return out


def _kernel_recompute_sequence(g, req, ext, changes):
    ck = compile_kernel(g, req)
    out = [tuple(ck.execute(ext)[q] for q in req)]
    for ch in changes:
        out.append(tuple(ck.recompute(ch)))
    return out


@pytest.mark.parametrize("factory,req,ext,changes", [
    (_chain_graph, ["variables.p"], {"basket": {"y": 10.0}},
     [{"basket": {"y": 11.0}}, {"basket": {"y": 12.0}}, {"basket": {"y": 13.0}}]),
    (_diamond_graph, ["variables.u", "variables.a"], {"basket": {"y": 100}},
     [{"basket": {"y": 101}}, {"basket": {"y": 102}}, {"basket": {"y": 103}}]),
    (_two_source_reconvergent_graph, ["variables.r"], {"basket": {"a": 2.0, "b": 3.0}},
     [{"basket": {"a": 5.0}}, {"basket": {"b": 7.0}}, {"basket": {"a": 9.0, "b": 1.0}}]),
])
def test_recompute_equiv_external(factory, req, ext, changes):
    g = factory()
    assert _kernel_recompute_sequence(g, req, ext, changes) == \
        _interp_recompute_sequence(g, req, ext, changes)


def test_recompute_equiv_mixed_cone():
    g = _chain_graph_with_python_middle()         # existing helper, has a Python-only node
    req, ext = _python_middle_required(), _python_middle_ext()   # see note below
    changes = [_python_middle_change(1), _python_middle_change(2)]
    assert _kernel_recompute_sequence(g, req, ext, changes) == \
        _interp_recompute_sequence(g, req, ext, changes)


def test_recompute_runtime_error_propagates():
    g = Graph(
        variables_lists={"variables": [
            {"name": "z", "inputs": {"y": "basket"}, "formula": njit(lambda y: 1.0 / y)},
        ]},
        external_source_names=["basket"],
    )
    ck = compile_kernel(g, ["variables.z"])
    ck.execute({"basket": {"y": 1.0}})
    with pytest.raises(ZeroDivisionError):
        ck.recompute({"basket": {"y": 0.0}})
```

Note: `_chain_graph_with_python_middle` already exists (line ~1423); inspect it and add the three tiny `_python_middle_*` helpers (required list, ext dict, change dict) right beside the new tests, matching that graph's actual node/external names. Keep them minimal and concrete — read the helper first.

- [ ] **Step 2: Run to verify they fail** (diamond/two-source produce stale values or wrong results because `recompute` does not yet rebuild the cone).

- [ ] **Step 3: Add `_ConePlan` to `_kernel_partition.py`:**

```python
@dataclass(frozen=True)
class _ConePlan:
    """A recompute sub-plan executed in place against a persistent store.
    Unlike _Plan.run (fresh dict from args, returns output_vars), run_into
    reads live-ins from and writes every step output back to the shared store."""
    steps: tuple

    def run_into(self, store: dict) -> None:
        for step in self.steps:
            vals = [store[v] for v in step.in_vars]
            if isinstance(step, _JitStep):
                store.update(zip(step.out_vars, step.dispatcher(*vals)))
            else:
                store[step.var] = step.py_callable(*vals)
```

- [ ] **Step 4: Implement boundary memo + cone build + wire `recompute`** in `compile_kernel.py`. Add imports `compute_boundary, cone_liveness, _ConePlan` from `_kernel_partition`. Add:

```python
    def _ensure_boundary(self):
        if self._boundary is not None:
            return
        compiled = self._ctx[0]
        if self._sources is None:
            self._sources = set(self._external_vars)
        self._boundary = compute_boundary(
            compiled.ordered_nodes, self._sources, set(self._required_vars)
        )

    def _build_cone_plan(self, affected: list) -> _ConePlan:
        compiled, idents, _, jit_options, cache, _ = self._ctx
        flags = _effective_flags(jit_options)
        cone_vars = {n.variable for n in affected}
        demoted_in_cone = {v for v in self._demoted if v in cone_vars}
        order = linearize(affected, demoted_in_cone)
        runs = build_runs(order, demoted_in_cone)
        steps = []
        for kind, run_nodes in runs:
            if kind == "python":
                for n in run_nodes:
                    steps.append(_PyStep(
                        var=n.variable,
                        py_callable=getattr(n.variable.formula, "py_func", n.variable.formula),
                        in_vars=tuple(n.inputs),
                    ))
                continue
            live_in, live_out = cone_liveness(run_nodes, order, self._required_vars, self._boundary)
            src, seg_bindings, _, _ = _generate_segment_body(run_nodes, live_in, live_out, idents, flags)
            disp = _compile(src, seg_bindings, jit_options, cache)
            disp.compile(tuple(typeof(self._store[v]) for v in live_in))
            steps.append(_JitStep(dispatcher=disp, in_vars=live_in, out_vars=live_out))
        return _ConePlan(steps=tuple(steps))
```

Replace the `recompute` body's "Task 4 fills in" comment with:

```python
        self._ensure_boundary()
        affected = compiled._collect_affected(changed_vars)   # compiled = self._ctx[0]
        if not affected:
            return tuple(self._store[v] for v in self._required_vars)
        plan = self._build_cone_plan(affected)
        plan.run_into(self._store)
        return tuple(self._store[v] for v in self._required_vars)
```

(bind `compiled = self._ctx[0]` at the top of `recompute`). Import `linearize, build_runs, _generate_segment_body, _JitStep, _PyStep` as needed (most already imported).

- [ ] **Step 5: Run to verify they pass.** The diamond/two-source equivalence proves the boundary write-back keeps interiors fresh across cones.
- [ ] **Step 6: Lint.**
- [ ] **Step 7: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/_kernel_partition.py numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: incremental recompute via cone re-fusion over a value store"
```

---

### Task 5: Cone-plan cache (LRU), type-change flush, cold-path degradation

**Goal:** Cache cone sub-plans in an LRU `OrderedDict` keyed on `(frozenset(cone quals), frozenset(live-in boundary quals))`, bounded by a cap; on a boundary type change (`NumbaError`) flush the whole cache + reseed; never recompile-per-call below the interpreted baseline. Spec §7.5, §7.6, §9.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py`
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] Repeating the same change pattern reuses the cached plan (assert the cone dispatcher object identity is stable across calls, or instrument a build counter).
- [ ] Exceeding the cap evicts LRU; correctness preserved (equivalence vs interpreted still holds while thrashing).
- [ ] A boundary live-in whose type changes triggers one cache flush + reseed + rebuild and returns the correct result (no per-entry cascade).

**Verify:** clean caches, then `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_compile_kernel.py -k "recompute_cache or recompute_typechange" -v` → pass.

**Steps:**

- [ ] **Step 1: Write the failing tests:**

```python
def test_recompute_cache_reuses_plan():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    ck.execute({"basket": {"y": 100}})
    ck.recompute({"basket": {"y": 101}})
    key = next(iter(ck._cone_cache))
    plan_first = ck._cone_cache[key]
    ck.recompute({"basket": {"y": 102}})
    assert ck._cone_cache[key] is plan_first        # same cone -> reused plan


def test_recompute_cache_cap_evicts():
    g = _fanout_graph()
    ck = compile_kernel(g, ["variables.o1", "variables.o2"])
    ck._cone_cap = 1                                  # force thrash
    ck.execute({"basket": {"a": 1.0, "b": 2.0}})
    ck.recompute({"basket": {"a": 3.0}})             # cone {o1}
    ck.recompute({"basket": {"b": 4.0}})             # cone {o2} -> evicts {o1}
    assert len(ck._cone_cache) == 1
    # correctness preserved despite thrash:
    assert tuple(ck.recompute({"basket": {"a": 5.0}})) == (5.0 + 1.0, 4.0 + 2.0)


def test_recompute_typechange_flushes_and_recovers():
    g = _chain_graph()
    ck = compile_kernel(g, ["variables.p"])
    ck.execute({"basket": {"y": 10.0}})
    ck.recompute({"basket": {"y": 11.0}})
    assert len(ck._cone_cache) >= 1
    out = ck.recompute({"basket": {"y": np.int64(12)}})   # type change at the boundary
    assert out == (((12 + 1.0) * 2.0) - 3.0,)
```

- [ ] **Step 2: Run to verify they fail** (`AttributeError: _cone_cache is None` / wrong results).

- [ ] **Step 3: Implement.** Add `from collections import OrderedDict` at the top of `compile_kernel.py`. In `__init__`, set `self._cone_cache = OrderedDict()` and `self._cone_cap = 64`. Add the key + flush helpers and route the build through the cache:

```python
    def _cone_key(self, affected, live_in_boundary):
        return (frozenset(n.variable.qual_name() for n in affected),
                frozenset(v.qual_name() for v in live_in_boundary))

    def _flush_and_reseed(self):
        self._cone_cache.clear()
        self._store = None
        self._boundary = None
        self._ensure_store()
        self._ensure_boundary()

    def _cone_plan_cached(self, affected) -> _ConePlan:
        self._ensure_boundary()
        # live-in boundary identities are part of the key (external vs interior
        # override with the same cone produce different live-ins -> different plans).
        cone_vars = {n.variable for n in affected}
        live_in_boundary = {inp for n in affected for inp in n.inputs if inp not in cone_vars}
        key = self._cone_key(affected, live_in_boundary)
        plan = self._cone_cache.get(key)
        if plan is not None:
            self._cone_cache.move_to_end(key)
            return plan
        plan = self._build_cone_plan(affected)
        self._cone_cache[key] = plan
        if len(self._cone_cache) > self._cone_cap:
            self._cone_cache.popitem(last=False)
        return plan
```

Update `recompute` to use the cache + the type-change guard:

```python
        plan = self._cone_plan_cached(affected)
        try:
            plan.run_into(self._store)
        except NumbaError:
            self._flush_and_reseed()
            self._apply_changes(changed)            # re-apply onto the fresh store
            plan = self._cone_plan_cached(affected)
            plan.run_into(self._store)
        return tuple(self._store[v] for v in self._required_vars)
```

(Keep `NumbaError` imported — it already is.) For un-fingerprintable formulas, `_compile` already sets `cache=False`; the on-disk cache handles the cacheable rebuild path, so no extra code is needed for §7.6 beyond the LRU + flush — document the fallback-to-interpreted as an open item only if a thrash benchmark (Task 8) shows a regression.

- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Lint.**
- [ ] **Step 6: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: bounded cone-plan cache with type-change flush"
```

---

### Task 6: Interior overrides via lazy change-source expansion

**Goal:** Support `recompute` of an interior node's value (mirroring `CompiledGraph.recompute`): when a changed name resolves to an interior node not yet a source, add it to `self._sources`, recompute the boundary, and flush the cone cache. Spec §8.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (`_apply_changes`)
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] Overriding an interior node matches interpreted `CompiledGraph.recompute` override semantics across a multi-call sequence.
- [ ] First interior override expands `self._sources`, recomputes `self._boundary`, and clears the cone cache.
- [ ] An external change and an interior override that yield the same cone produce distinct cached plans (no key collision).

**Verify:** clean caches, then `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_compile_kernel.py -k "recompute_interior" -v` → pass.

**Steps:**

- [ ] **Step 1: Write the failing tests:**

```python
def test_recompute_interior_override_matches_interpreted():
    g = _diamond_graph()
    req, ext = ["variables.u"], {"basket": {"y": 100}}
    changes = [{"variables": {"x": 5.0}}, {"basket": {"y": 101}}, {"variables": {"x": 9.0}}]
    assert _kernel_recompute_sequence(g, req, ext, changes) == \
        _interp_recompute_sequence(g, req, ext, changes)


def test_recompute_interior_override_expands_sources_and_flushes():
    g = _diamond_graph()
    ck = compile_kernel(g, ["variables.u"])
    ck.execute({"basket": {"y": 100}})
    ck.recompute({"basket": {"y": 101}})
    before = set(ck._cone_cache)
    assert before                                   # at least one external cone cached
    ck.recompute({"variables": {"x": 5.0}})         # interior override
    by_qual = {v.qual_name(): v for v in ck._sources}
    assert "variables.x" in by_qual                 # x added to sources
    # boundary recomputed + cache flushed at expansion (then repopulated by this call):
    assert all(k for k in ck._cone_cache)           # cache valid; old keys cleared at flush
```

- [ ] **Step 2: Run to verify they fail** (interior override currently just sets the value; boundary/cache not adjusted → wrong results or stale plan).

- [ ] **Step 3: Implement.** Extend `_apply_changes` to detect new interior sources and trigger expansion:

```python
    def _apply_changes(self, changed: dict) -> set:
        compiled = self._ctx[0]
        changed_vars = set()
        new_interior_sources = set()
        for src, vals in changed.items():
            for name, val in vals.items():
                var = compiled.required_external_variables.get(src, {}).get(name)
                is_external = var is not None
                if var is None:
                    qual = make_qual_name(src, name)
                    var = next((n.variable for n in compiled.ordered_nodes
                                if n.variable.qual_name() == qual), None)
                    if var is None:
                        warnings.warn(f"{qual} is not in the calculation path, update has no effect.")
                        continue
                self._store[var] = val
                changed_vars.add(var)
                if not is_external and (self._sources is None or var not in self._sources):
                    new_interior_sources.add(var)
        if new_interior_sources:
            self._expand_sources(new_interior_sources)
        return changed_vars

    def _expand_sources(self, new_sources: set):
        if self._sources is None:
            self._sources = set(self._external_vars)
        self._sources |= new_sources
        self._boundary = None            # force recompute (Task 4 _ensure_boundary)
        self._cone_cache.clear()         # boundaries (hence plans) may have changed
```

`_ensure_store` must run before `_apply_changes` (already the case: `recompute` calls `_ensure_store()` first), so the store exists when interior values are written.

- [ ] **Step 4: Run to verify they pass.**
- [ ] **Step 5: Lint.**
- [ ] **Step 6: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: interior-override recompute via lazy source expansion"
```

---

### Task 7: Docs

**Goal:** Document `recompute`: the value-only/same-types contract, the seed precondition, the boundary-persistence model (diamond example), the don't-interleave-throughput note, the interior Python-node stable-type requirement, and cache/degradation behavior. Spec §15.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (module docstring + `CompiledKernel`/`recompute` docstrings)
- Modify: `docs/numbox.core.variable.rst`
- Test: doc build + doc-codeblock-flake8 (no new unit tests)

**Acceptance Criteria:**
- [ ] `recompute` has a docstring covering contract, precondition, interior overrides, and limitations.
- [ ] `docs/numbox.core.variable.rst` has an "Incremental recompute" subsection with the diamond example.
- [ ] Sphinx builds with warning count at or below baseline; doc code blocks flake8-clean.

**Verify:**
- `cd /home/erik/projects/numbox/docs && /home/erik/projects/numbox/venv/bin/sphinx-build -b html . _build/html` → exit 0, warnings ≤ baseline.
- doc-codeblock-flake8 over the changed `.rst` (run the repo's `.github/workflows/doc-codeblock-flake8` extractor + flake8).

**Steps:**

- [ ] **Step 1:** Add a `recompute` docstring (contract, precondition, boundary model summary, interior-override support, the two documented limitations: don't interleave input-changing `kernel()` calls between recomputes; interior Python nodes must return stable types). Update the module docstring's "non-goals" note that previously excluded incremental recompute.
- [ ] **Step 2:** Add the `.rst` "Incremental recompute" subsection with a runnable diamond example (`compile_kernel` → `kernel(...)` → `recompute({...})`), flake8-clean.
- [ ] **Step 3:** Run sphinx-build; confirm exit 0 and warning count ≤ baseline.
- [ ] **Step 4:** Run doc-codeblock-flake8 over the changed `.rst`.
- [ ] **Step 5: Commit.**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py docs/numbox.core.variable.rst
git -C /home/erik/projects/numbox commit -m "docs: CompiledKernel.recompute incremental refresh"
```

---

### Task 8: Benchmark extension

**Goal:** Add a recompute mode to `test/compile_kernel_benchmark.py`: seed once, then drive M recomputes changing a small input subset; report recompute steady-state vs full `kernel()` re-run vs interpreted `CompiledGraph.recompute`, across a deep chain, a diamond, and fan-out columns. Spec §14.

**Files:**
- Modify: `test/compile_kernel_benchmark.py`
- Test: run the benchmark (it is a script, not a pytest target)

**Acceptance Criteria:**
- [ ] A `--recompute` mode runs the three regimes and prints recompute vs full-rerun vs interpreted timings + cone/segment counts.
- [ ] Recompute steady-state beats full `kernel()` re-run on the deep chain; the fan-out regime exercises cache pressure and stays ≥ interpreted-baseline speed.

**Verify:** `/home/erik/projects/numbox/venv/bin/python test/compile_kernel_benchmark.py --recompute` → prints a table; recompute < full re-run on the chain.

**Steps:**

- [ ] **Step 1:** Read the existing benchmark structure (`vs0 = Values()` etc.) and add a `--recompute` argparse mode following the established `--compile-report`/`--profile` patterns.
- [ ] **Step 2:** Implement the three-regime measurement (chain / diamond / fan-out), timing `ck.recompute(small_change)` vs `ck.kernel(*full)` vs `cg.recompute(change, values)`.
- [ ] **Step 3:** Run it; confirm the steady-state recompute speedup and that fan-out thrash stays at or above interpreted baseline.
- [ ] **Step 4: Commit.**

```bash
git -C /home/erik/projects/numbox add test/compile_kernel_benchmark.py
git -C /home/erik/projects/numbox commit -m "bench: recompute steady-state vs full re-run vs interpreted"
```

---

## Final gate (before declaring the feature done)

Run the full local CI-equivalent gate (per repo memory), caches cleaned first:
- `pytest --durations=20` over the whole suite (not just the new file);
- both flake8 passes (repo `.flake8`);
- doc-codeblock-flake8;
- sphinx build (warnings ≤ baseline);
- lychee link check over changed `.rst`/`.md`.

Do NOT push or cherry-pick to `upstream-pr/compile-kernel-hardened` (#24) or post the #24 reply without explicit per-action user consent (spec §16).
