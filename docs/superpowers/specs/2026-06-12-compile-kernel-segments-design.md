# Design: `compile_kernel` v2 — segmented orchestration for partially-jittable graphs

- **Date:** 2026-06-12
- **Status:** Approved design; implementation pending
- **Module:** `numbox/core/variable/compile_kernel.py` (extended),
  `numbox/core/variable/_kernel_partition.py` (new, private)
- **Relates to:** `docs/superpowers/specs/2026-06-07-compile-kernel-design.md` (v1),
  upstream [PR #24](https://github.com/Goykhman/numbox/pull/24) discussion
  ([issuecomment-4693805704](https://github.com/Goykhman/numbox/pull/24#issuecomment-4693805704)).
- **Delivery vehicle:** extends #24 (and fork review PR #52) — not a follow-up PR.

## 1. Motivation

v1 `compile_kernel` requires every formula in the requested subgraph to be
njit-able; one non-jittable node makes the whole kernel fail to type at first
call. Goykhman's review proposal on #24: let the "true" kernel be a Python
function orchestrating both jitted fused segments and pure-Python nodes — for
`N1→N2→N3→N4→N5` with `N3` not jittable, call `fuse(N1,N2)`, compute `N3` in
Python, then `fuse(N4,N5)`. And since topological order is not unique, choose
a linearization that clusters jittable nodes so fusion is maximized.

v2 implements exactly that, with **automatic** detection of non-jittable
nodes — no user marking. The key enabler over v1: the master is Python, so the
first call has concrete argument values, which is precisely the type
information needed to try-jit each node and yank the failures.

## 2. Goals / Non-goals

**Goals**
- Graphs with non-jittable formulas work end-to-end through `compile_kernel`
  with no API change and no user annotation.
- Zero regression for all-jittable graphs: same single fused kernel, same
  byte-identical generated source, same cache digests, same bare-dispatcher
  hot path after first call.
- Fusion-maximizing linearization (deterministic, cache-stable).
- Per-segment content-addressed caching through the existing v1 machinery.
- A structured `PartitionReport` describing what actually runs where, and why.

**Non-goals (v2)**
- **Explicit marking** (`python_nodes=` kwarg or a `Variable` field) —
  considered and rejected in favor of pure auto-detection; trivially addable
  later if a use case appears (e.g. formulas that take a long time to *fail*
  compilation).
- **Warnings on demotion** — considered and rejected; `PartitionReport` is the
  single description surface.
- **Partition persistence across processes** — a fresh process re-pays one
  failed typing pass + probes at first call; the expensive segment compiles
  load from numba's on-disk cache. Revisit if the benchmark says it matters.
- **Per-signature plan dictionary** — one active plan; see §4 limitation.
- **`numba.objmode` lifting** — objmode blocks inside one fused kernel would
  avoid segment boundaries but require explicit output type annotations per
  Python node; intrusive, different trade-off, out of scope.
- **Proxy bindings for compile cost** (Goykhman's earlier #23 idea) — a
  different axis (compile cost vs jittability coverage); separate discussion.
- v1's non-goals (per-node `cacheable` memoization, incremental `recompute`,
  `None`-as-value formulas, node-identity APIs) all still hold.

## 3. Public API

Unchanged signature; one new attribute:

```
ck = compile_kernel(graph, required, *, jit_options=None, cache=None)

ck.partition   # None until the first call resolves the mode; then a
               # PartitionReport (see §8)
```

`ck.kernel` semantics are restated as **"the hot-path callable"**:

- before the first call: a resolver callable (runs the first-call logic);
- after the first call, fused mode: the **bare numba dispatcher** — identical
  to v1's steady state, zero wrapper overhead (`kernel` becomes a property
  returning the current callable; a caller that grabbed the resolver early
  keeps a working callable that delegates, paying one indirection);
- segmented mode: the Python master (§7).

`ck.execute`, `ck.params`, `ck.outputs`, `ck.source` (the *fused* source),
`ck.identifiers` are unchanged.

## 4. Call-mode state machine

```
            ┌──────── first call ────────┐
  virgin ───┤ typeof(args) + fused       │──── success ──▶ fused (permanent)
            │ dispatcher.compile(sig)    │
            └──── numba compile error ───┘
                  or untypeable arg
                        │
                        ▼
                  discovery (§5)  ──────────────────────▶ segmented(plan)
                                                              │
                              segment numba-compile error     │ later call,
                              on a later-call signature ◀─────┘ new types
                                        │
                                        ▼
                              re-discovery; REPLACE the plan
```

- **virgin → fused:** compute `numba.typeof()` on each arg, call
  `dispatcher.compile(types)` on the v1 fused kernel. Success → this graph
  runs fully fused, permanently (for new signatures numba's own dispatch
  applies, exactly as v1; a later-signature typing failure raises to the user,
  v1 behavior — no auto-discovery from fused mode).
- **virgin → discovery:** any numba compile-time error from the fused attempt,
  or any external arg `numba.typeof()` cannot type.
- **segmented:** the master runs the single active plan. Segments are ordinary
  dispatchers and auto-specialize on new input types; if a segment raises a
  numba compile-time error for new types, re-run discovery with the current
  call's values and **replace** the plan. No per-call `typeof`, no plan
  dictionary — the hot path stays free of type inspection. Documented
  limitation: workloads alternating between two type families whose partitions
  differ will churn plans (re-discovery per alternation).

Catch scope everywhere in this state machine: `numba.core.errors.NumbaError`
raised by *compilation* steps only. Runtime errors always propagate (§5).

## 5. Discovery: warm-up + probe in one pass

Runs with the actual argument values of the triggering call; the call still
returns correct results (discovery doubles as execution).

Walk the nodes in the v1 topo order (`compiled.ordered_nodes`), maintaining
`{Variable: value}`:

1. **Type the inputs.** `numba.typeof()` each input value. Failure (e.g. an
   upstream node emitted a DataFrame) → **demote** this node, reason
   `"input '<qual>' value of type <T> is not numba-typeable"`.
2. **Probe-compile.** The node's binding is the njit-wrapped formula from v1
   codegen (a `Dispatcher` for plain-Python and `@njit` formulas):
   `binding.compile(input_types)` inside `try/except NumbaError` → on error,
   **demote**, reason = first line of the error text (truncated ~200 chars).
3. **Evaluate.**
   - jittable node → call the (now compiled) dispatcher with the input values;
   - demoted node → call `getattr(formula, "py_func", formula)` in plain
     Python.
   Either way the node's value lands in the table for downstream probes.
   **Runtime exceptions propagate** — the compile/execute split is what keeps
   a `ZeroDivisionError` from being misread as "not jittable".
4. **Exotic formulas** (`CompileResultWAP`, `CFunc`, `DUFunc`): no
   eager-compile hook and no Python fallback exists for them (a
   `CompileResultWAP` that cannot type in nopython cannot run in Python
   either). They are assumed jittable. For warm-up evaluation they are invoked
   through a generated single-node `@njit` shim (`def _shim(a, b): return
   f(a, b)` with the formula bound as a global — the same binding trick
   segments use), compiled against the known input types. A `NumbaError` from
   the shim compile is a **hard error propagated to the caller** (v1
   equivalent: the whole fused kernel fails to type), not a demotion.

Probe compiles are not wasted: the probed dispatchers are the same objects the
segments bind as globals, so their typed overloads are reused when the segment
compiles.

After the walk: demoted set + per-node reasons + all intermediate values
(used to eagerly compile segments, §7, and to return the first call's
outputs).

## 6. Fusion-maximizing linearization

With the demoted set known, re-linearize the node set (edges from
`CompiledNode.inputs`) with greedy color-sticky Kahn:

- two ready-queues, jittable and Python, each ordered by `qual_name`
  (determinism — the content-addressed cache needs byte-stable segment
  sources);
- always drain the current color while its queue is non-empty; switch colors
  only when forced; the linearization is computed from both possible starting
  colors and the candidate with fewer runs wins (jit-start on a tie) — a
  jit-start alone produces an avoidable extra run when a Python chain and a
  jit chain meet at a jit sink.

Exact minimization of run count in a 2-colored DAG linearization is NP-hard;
greedy color-stickiness is the documented heuristic and handles the practical
shapes (pipelines, diamonds, layered DAGs) well. Goykhman's `N1→N5` example
partitions into exactly `fuse(N1,N2) → py(N3) → fuse(N4,N5)` and is a literal
test case.

External variables take no position in the linearization (they are
pre-satisfied dependencies, not steps); only non-external nodes are
scheduled, mirroring v1 codegen's skip of externals.

## 7. Segments and the master plan

**Partition.** Split the linear order into maximal runs of same-colored nodes.
Jittable runs become fused segments; consecutive demoted nodes form one
Python segment (one plan step per node inside it).

**Liveness.** A value produced in step *i* is a **live-out** if consumed by
any later step or listed in `required`; a segment's **live-ins** are the
external variables and earlier-produced values its nodes consume. Both lists
ordered by `qual_name` (determinism).

**Segment codegen.** Generalize v1's `_generate_body` to
`(nodes, available_inputs, needed_outputs, idents)`: available inputs become
parameters, needed outputs become the return tuple, body lines and per-line
comments unchanged. The v1 full-kernel call becomes the special case
(`available = required_external_variables`, `needed = required`) and **must
keep producing byte-identical source** (cache-digest stability for existing
v1 users is a regression test). `_assign_identifiers` runs once over all
nodes; identifiers are shared across segments.

**Segment compile.** Each segment goes through the existing `_compile`
verbatim: per-segment content-addressed digest (segment source + that
segment's formula fingerprints + flags), per-segment anchor file, same `cache`
tri-state semantics, same uncacheable-formula downgrade — all v1 machinery,
unmodified. Segments are compiled **eagerly during discovery** (live-in types
are known from the warm-up values), so every compile-time error for the
learned signature surfaces at the same moment and the plan is fully ready
when the first call returns.

**Master.** A plain Python plan-walker, not exec-generated source:

```
slots = {var: value}            # externals pre-loaded from args
for step in plan.steps:
    jit step:    outs = step.dispatcher(*[slots[v] for v in step.live_ins])
                 slots.update(zip(step.live_outs, outs))
    python step: slots[step.var] = step.py_callable(*[slots[v] for v in step.ins])
return tuple(slots[v] for v in plan.output_vars)
```

Per-call overhead is one Python call plus boxing per step — the boundary cost
the benchmark quantifies (§11). All boundary values are ordinary boxed
Python/numpy objects; nothing special is marshalled.

**Module layout.** Discovery, linearization, liveness, plan and report types
live in a new private module `numbox/core/variable/_kernel_partition.py`
(pure logic; the ordering/liveness parts unit-test without any compilation).
`compile_kernel.py` keeps codegen, caching, `CompiledKernel`, and the public
API. Public surface stays exactly `compile_kernel` + `CompiledKernel` (the
report class is public-by-traversal via `ck.partition`, exported under
`compile_kernel.py`'s namespace).

## 8. PartitionReport

```
ck.partition                    # None while virgin
  .mode          "fused" | "segmented"
  .segments      ordered tuple of Segment:
      .kind      "jit" | "python"
      .nodes     (qual_name, ...)     # linear order within the segment
      .inputs    (qual_name, ...)     # live-ins
      .outputs   (qual_name, ...)     # live-outs
      .source    str | None           # generated source, jit segments only
      .reasons   {qual_name: str}     # python segments: why each node demoted
  .python_nodes  # convenience union across python segments
  str(report)    # human-readable summary (mode, per-segment node lists,
                 # demotion reasons)
```

Uniform across modes: fused mode reports one jit segment holding every
non-external node (source = the fused source, reasons empty), so "what is
actually running" is always one attribute away after any first call. Frozen
dataclasses; rebuilt (replaced) on plan replacement. No warning is emitted
anywhere — the report is the whole story (explicit decision, §2).

## 9. Error taxonomy

| Condition | When | Outcome |
|---|---|---|
| all v1 eager structural errors | `compile_kernel()` | unchanged (ValueError/TypeError/RuntimeError) |
| fused attempt fails to type | first call | silent fallback to discovery |
| external arg not numba-typeable | first call | skip fused attempt, discovery |
| Dispatcher/plain formula fails probe compile | discovery | demoted (reason recorded) |
| node input value not numba-typeable | discovery | demoted (reason recorded) |
| exotic formula (cres/CFunc/DUFunc) shim fails to compile | discovery | numba error **propagates** (no Python fallback exists) |
| formula raises at runtime (any mode) | execution | propagates unchanged, never demotes |
| segment fails to compile new-signature types | later call | re-discovery, plan replaced |
| fused-mode kernel fails new-signature types | later call | numba error propagates (v1 behavior) |
| missing external value | `ck.execute()` | `KeyError` (unchanged) |

## 10. Testing (`test/core/test_compile_kernel.py` additions)

Equivalence & modes:
- mixed graph (jittable nodes around a Python-only node) equals
  `CompiledGraph.execute` on call 1 (warm-up path) and call 2 (segmented
  fast path);
- all-jittable regression: mode fused, `ck.kernel` is the bare dispatcher
  after first call, generated source byte-identical to v1 (digest stability);
- Goykhman's `N1→N5`, `N3` Python → exactly two jit segments
  `(N1,N2)`/`(N4,N5)` with `N3` between them;
- ordering: a DAG where DFS order interleaves colors but greedy clustering
  yields fewer segments — assert segment count;
- all-nodes-demoted graph → zero jit segments, pure-Python plan, correct
  results.

Discovery semantics:
- demotion reasons recorded (TypingError text; untypeable-input text);
- object-value producer demoted and its consumers demoted via untypeable
  inputs;
- runtime error (e.g. ZeroDivisionError) propagates from call 1 and call 2,
  no demotion;
- `@njit` formula that cannot compile for the arg types → demoted to
  `py_func`;
- plan replacement: signature A learns a partition; signature B breaks a
  segment → re-discovery, new plan, both signatures still compute correctly.

PartitionReport:
- `None` before first call; fused mode → single jit segment spanning all
  nodes; segmented mode → segments/nodes/inputs/outputs/reasons populated;
  `str()` renders.

Caching:
- fresh-subprocess test: second process re-discovers and segment compiles hit
  numba's on-disk cache (per-segment anchors exist; mirror the v1 subprocess
  cache-test pattern);
- two segments in one process with identical source+formulas share a digest
  harmlessly.

Run with the venv python, caches cleaned first (`__pycache__` + numba cache),
per the standing test protocol.

## 11. Benchmark (`test/compile_kernel_benchmark.py` extension)

New mode injecting K Python-only nodes into the N-node chain (default
N=1000): reports segmented-vs-`CompiledGraph` wall time, discovery (first
call) cost vs steady-state, segment count, and per-boundary overhead. This is
the justify-the-feature number for the #24 thread, matching the benchmark
practice established on #23.

## 12. Docs

- `docs/numbox.core.variable.rst`: new "Graphs with non-jittable nodes"
  subsection — auto-detection, `PartitionReport`, the `N1→N5` worked example,
  determinism/caching notes, the fused-mode zero-regression guarantee.
- `compile_kernel` docstring: detection semantics, `.partition`, `.kernel`
  hot-path-callable wording, plan-replacement limitation.
- Module docstring: currently promises every formula must be njit-able —
  rewrite.
- Code blocks flake8-clean (doc-codeblock-flake8); sphinx warning count at or
  below baseline.

## 13. Branch / PR workflow

- Implement on `feat/compile-kernel-hardened` (fork review PR #52; bots
  re-review there first).
- Full local CI gate before any push (pytest `--durations=20` with caches
  cleaned, both flake8 passes, doc-codeblock-flake8, sphinx baseline, lychee).
- Cherry-pick to `upstream-pr/compile-kernel-hardened` (#24) — only the
  feature files (`compile_kernel.py`, `_kernel_partition.py`, tests,
  benchmark, `variable.rst`); never `docs/superpowers/**`, `CLAUDE.md`, or
  fork-only CI. Upstream push requires explicit per-action consent.
- The #24 design-question reply is drafted after implementation (answer with
  working code + benchmark numbers); posting requires explicit consent.

## 14. Open risks / verification items

1. **`Dispatcher.compile(types)` probe behavior across numba 0.60–0.65** —
   stable public API, but verify the error classes raised on typing failure
   are uniformly `NumbaError` subclasses on both pinned versions (CI matrix
   covers both).
2. **CompileResultWAP/CFunc/DUFunc warm-up shims** — verify early (first
   implementation task) that a single-node shim binding each exotic type as a
   global compiles and runs for representative cases; v1 tests prove the
   in-kernel call works, the shim is the same shape.
3. **Greedy linearization quality** — heuristic, not optimal; acceptable by
   design. If a pathological real case appears, the linearizer is isolated in
   `_kernel_partition.py` and swappable.
4. **Plan-replacement churn** under alternating type families — documented
   limitation (§4); the benchmark's discovery-cost number quantifies the
   worst case.
5. **First-call cost in fresh processes** for graphs with Python nodes (one
   failed fused typing pass + probes, every process) — measured by the
   benchmark; if material, partition persistence (§2 non-goal) is the known
   escape hatch.
