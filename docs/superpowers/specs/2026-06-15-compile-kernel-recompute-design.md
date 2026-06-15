# Design: `CompiledKernel.recompute` — incremental refresh of a compiled Variable graph

- **Date:** 2026-06-15
- **Status:** Approved design; #24 reply drafted (not posted); implementation pending.
- **Module:** `numbox/core/variable/compile_kernel.py` (extended),
  `numbox/core/variable/_kernel_partition.py` (extended, private).
- **Relates to:** `docs/superpowers/specs/2026-06-07-compile-kernel-design.md` (v1),
  `docs/superpowers/specs/2026-06-12-compile-kernel-segments-design.md` (v2 segmentation),
  upstream [PR #24](https://github.com/Goykhman/numbox/pull/24) recompute discussion.
- **Delivery vehicle:** extends the `compile_kernel` feature on
  `feat/compile-kernel-hardened` (fork review PR #52); fold into #24 unless that PR
  merges first, in which case a follow-up PR. Upstream push requires explicit consent.

## 1. Motivation

The interpreted
[`CompiledGraph.recompute(changed, values)`](https://github.com/Goykhman/numbox/blob/6dd5f8a39dbf3309f6dbbecc82a98ae9e4a9be6d/numbox/core/variable/variable.py#L358)
re-evaluates only the nodes affected by a change: it seeds the new values, walks the
forward affected cone via
[`_collect_affected`](https://github.com/Goykhman/numbox/blob/6dd5f8a39dbf3309f6dbbecc82a98ae9e4a9be6d/numbox/core/variable/variable.py#L313),
nulls those nodes, and re-runs `_calculate` over the cone alone. `compile_kernel`
has no analogue: a fused kernel is one straight-line `@njit` function, so "recompute
after a small input change" degenerates to a full re-run — there is no interior seam
to stop at. That is the price of fusion, and it is wrong for the workload where the
same graph is refreshed many times as a few inputs change (the amortized regime).

`CompiledKernel.recompute(changed)` adds incremental refresh: re-execute only the
affected cone, re-fused, reading the unchanged upstream values from a persistent
Python-side store, with the cone's compiled sub-plan cached so repeated change
patterns pay nothing after warm-up.

The architecture below was selected after a five-approach design panel (each from a
distinct bias) independently converged on it, then hardened against an adversarial
pass whose findings are encoded as the guards in §7 and the tests in §10.

## 2. Goals / Non-goals

**Goals**
- `CompiledKernel.recompute(changed)` mirroring the value-only, same-types contract of
  `CompiledGraph.recompute`: refresh the required outputs after an input change,
  re-evaluating only the affected cone.
- Real speedup in the **fused** case (the flagship), not just the segmented case —
  i.e. recompute must beat a full re-run even when the original graph fused into a
  single kernel.
- Correctness across **many repeated** recompute calls with **different** change-sets
  (the staleness traps in §7 are the hard part, not the happy path).
- A hard memory/compile ceiling: recompute must never degrade *below* the interpreted
  `CompiledGraph.recompute` baseline, even under adversarial change patterns.
- Reuse the existing discovery / linearization / segmentation / content-addressed
  compile machinery wherever it genuinely fits; add new code only where it must.

**Non-goals (this version)**
- **`Variable.params={jitable,type}` hints** and per-node `proxy(sig)` construction —
  a separate, larger change that touches the `Variable` class and the graph-wide
  typeability model. Recompute does not need it (auto-discovery already classifies
  jit vs Python nodes). One interaction is acknowledged as a documented limitation
  in §7.7.
- **Cross-process persistence of the value store or cone cache** — a fresh process
  re-primes; compiled sub-kernels reload from numba's on-disk cache.
- **Concurrent recompute on one `CompiledKernel`** — the store is mutable shared state;
  single-threaded use per instance (same as the interpreted path's `Values`).
- v1/v2 non-goals (per-node `cacheable` memoization, `None`-as-value formulas,
  node-identity APIs, objmode lifting) still hold.

## 3. Public API

One new method on `CompiledKernel`; everything else unchanged.

```
ck = compile_kernel(graph, required, *, jit_options=None, cache=None)
ck.kernel(*externals)            # full evaluation (v1/v2) — also seeds recompute (§6)
outputs = ck.recompute(changed)  # changed: {source: {name: new_value}}, like
                                 # CompiledGraph.recompute's `changed`; returns a
                                 # tuple in `ck.outputs` order (same shape as kernel())
```

- `changed` is the same nested `{source: {name: value}}` shape as the interpreted
  `recompute`. Names are resolved exactly as
  [`CompiledGraph.recompute`](https://github.com/Goykhman/numbox/blob/6dd5f8a39dbf3309f6dbbecc82a98ae9e4a9be6d/numbox/core/variable/variable.py#L358)
  does: external first, then interior (§8); a name in neither warns and is skipped.
- **Precondition:** a prior full evaluation must have seeded the store (`ck.kernel(...)`
  at least once). Calling `recompute` first raises `RuntimeError` with a message
  pointing at the precondition — never an opaque `KeyError` from a half-seeded store.
- Return is dict-free for symmetry with `ck.kernel`; a `recompute_execute(changed)`
  dict-in/dict-out convenience (symmetric with `ck.execute`) is a trivial optional add.

## 4. Architecture

Three pieces, all Python-side; the JIT seam is crossed only by ordinary boxed call args.

1. **Persistent value store** `self._store: dict[Variable, value]` — holds the current
   value of every **boundary** node (§5) plus required outputs. Seeded once (§6); read
   for cone live-ins; written back after every recompute.
2. **Affected cone → sub-plan**: on `recompute(changed)`, compute the affected cone
   (`compiled._collect_affected`), build a fused/segmented sub-plan over the cone with
   the cone's live-in boundary as its inputs, run it reading/writing `self._store`.
3. **Cone sub-plan cache** `self._cone_cache: OrderedDict[key, _ConePlan]` — LRU-bounded,
   keyed on (cone node-set, live-in boundary identities) (§9). Many recomputes of the
   same change pattern reuse one compiled sub-plan; the steady-state hot path is a
   store-gather + the cone's dispatcher call(s).

The cone sub-plan is built with the **same** pipeline v2 already uses for the full
graph — `discover`'s demotion verdicts (frozen at seed time, §7.4),
[`linearize`](https://github.com/Goykhman/numbox/blob/6dd5f8a39dbf3309f6dbbecc82a98ae9e4a9be6d/numbox/core/variable/_kernel_partition.py#L118)
/ `build_runs` over the cone nodes, `_generate_segment_body` + `_compile` per jit run —
so a cone that is all-jittable fuses into one dispatcher and a cone with Python nodes
segments exactly as the master plan does. What is **new** is enumerated in §11; the
"reuse everything verbatim" framing is explicitly false and the differences are the
load-bearing part of this design.

## 5. The persistence boundary (core correctness invariant)

The central trap (found by every correctness adversary): **you cannot fuse a whole cone
and also persist its interiors.** A fused segment's interior nodes are SSA temporaries
the `@njit` function never returns, so they cannot be written to the store. If
`cone(E_a)` fuses `A→M→N` and returns only `N`, a later `cone(E_b)` that reads `A` as a
boundary live-in gets a **stale `A`** → silently wrong result. Forcing every interior
to be returned un-fuses the cone and defeats the feature.

Resolution: persist at the **sharing boundary**, fuse within it. Define the
**change-source set** `S` (the nodes a caller can change; default `S = externals`,
extended for interior overrides in §8). Augment the DAG with a virtual super-source `⊤`
with an edge to each node in `S`, and compute dominators in one topological pass
(for a DAG, `dom(M) = {M} ∪ ⋂ dom(pred)` ). Then:

> **Boundary rule.** A node `N` must live in the store iff `N` is a required output,
> **or** `N` has a consumer `M` with `N ∉ dom(M)` — i.e. some change in `S` can reach
> `M` without reaching `N`, making `N` a live-in to that cone. Equivalently, `N` is
> *fuse-through* (never persisted) iff `N` dominates **all** its consumers and is not
> required.

The boundary set is computed **once** from graph structure (and recomputed only if `S`
grows, §8). Worked example — `A=fa(E.a)`, `B=fb(E.b)`, `M=f(A,B)`, `N=g(M)`,
`required={N}`, `S={E.a,E.b}`: `dom(M)={⊤,M}` so `A,B ∉ dom(M)` → `A,B` boundary;
`M` dominates its only consumer `N` → fuse-through; `N` required → boundary. Boundary
set `{A,B,N}`. `cone(E.a)={A,M,N}` fuses as one segment whose **live-out widens to
(required ∪ boundary) ∩ cone = {A,N}** (`M` stays SSA); the store gets fresh `A`, so
`cone(E_b)` later reads a correct `A`. Pure pipelines have empty interior boundary
(every node dominates its successor) → the whole cone fuses and only the output
persists; diamonds persist exactly their join inputs. This both fixes the staleness bug
and keeps fusion maximal — only genuine shared/join nodes widen a return tuple.

## 6. Seeding the store

`recompute` needs a store consistent with the **current** inputs before it can run.

- **Segmented first call.** `_discover_and_run` already builds a `{Variable: value}`
  dict holding *every* node's value and currently discards it
  ([`compile_kernel.py:388`](https://github.com/Goykhman/numbox/blob/6dd5f8a39dbf3309f6dbbecc82a98ae9e4a9be6d/numbox/core/variable/compile_kernel.py#L388)).
  Retain it as the seed (free); also store the demotion verdicts (§7.4).
- **Fused first call.** The fused dispatcher returns only required outputs; interiors
  are never materialized in Python. Capture the external args **once**, at the
  virgin→fused transition inside `_resolve_and_call` (which runs only while virgin — no
  per-call hot-path cost), then on first `recompute` seed the store by evaluating the
  **boundary ∪ required** nodes once from those args. Formulas in a fused graph are
  njit-pure by construction, so this single seed evaluation has no observable
  side-effect hazard (contrast §7.1).

The store is seeded lazily on the first `recompute`, not eagerly, so kernels never used
incrementally pay nothing.

## 7. Correctness guards (each from an adversarial finding)

1. **No side-effect double-fire / no O(N) re-probe on seeding.** Seeding must not
   blindly re-run `discover` over the whole graph after a fused first call (that
   re-executes every formula and re-probes every node). Use the retained values dict in
   segmented mode; in fused mode evaluate only the boundary∪required closure once. The
   contract documents that recompute requires **pure** formulas (already true for jit
   formulas; the hazard only exists for impure Python nodes, which fused graphs do not
   have).
2. **Interior staleness across cones** — handled by the boundary rule (§5): the cone
   write-back set is `(required ∪ boundary) ∩ cone`, computed via the dominator-based
   boundary set, **not** `segment_liveness` with cone-local order (which cannot see
   out-of-cone consumers).
3. **Throughput/recompute desync.** A bare `ck.kernel(*args)` after fused resolution
   writes nothing back to the store, so interleaving throughput calls that change inputs
   with `recompute` would read stale store values. Contract: `recompute` is the
   **stateful** entry point seeded once; do not interleave input-changing `kernel()`
   calls between recomputes (mirrors the interpreted contract, where bypassing the
   shared `Values` storage desyncs equally). Documented; not silently "supported".
4. **Demotion reuse, frozen at seed time.** `discover`'s `demoted` verdicts are stored
   on the instance (`self._demoted`) and **filtered** to each cone — never re-probed
   per cone (re-probing is wasted under same-types and would re-execute formulas).
   Today `demoted` is a local in `_discover_and_run`; retaining it is a required new
   field.
5. **Type-change handling = full flush, not per-entry.** A boundary live-in whose type
   changes (contract violation, but guarded) makes a cone dispatcher raise `NumbaError`.
   Do **not** rebuild only that entry (it cascades O(cache) failures across entries
   sharing the live-in). On any such `NumbaError`, **flush** `self._cone_cache`, re-seed
   the store, and rebuild — one re-discovery, mirroring v2's `_run_segmented`
   re-discovery but flushing the cone cache too.
6. **Cold path reuses compiled artifacts, never recompiles per call.** On an LRU miss
   past the cap, rebuilding hits numba's content-addressed on-disk cache for cacheable
   formulas (cheap reload, not full compile). For un-fingerprintable formulas
   (cres/CFunc — already `cache=False` in `_compile`), fall back to the **segmented
   master `_Plan`** (already compiled) or the interpreted
   `CompiledGraph.recompute` rather than recompiling the cone every call, so recompute
   never drops below the interpreted baseline (§9).
7. **Interior Python-node type drift (documented limitation).** Under the same-types
   contract applied at the *boundary*, an interior **demoted** node can still return a
   different type across recomputes (its output is value-dependent) with no
   boundary-type change and no `NumbaError` — a cached cone dispatcher would then run on
   an unexpected seam type. v1 documents that interior demoted nodes must return stable
   types across recomputes (the same-types contract extended to demoted outputs). This
   is the one place declared `params.type` (the deferred hints work) would let us check
   rather than document — noted as the natural follow-up.

## 8. Change-source set `S` and interior overrides

The interpreted `recompute` permits overriding an **interior** node's value (it resolves
the name against `ordered_nodes` after externals). To honor that mirror contract while
keeping fusion maximal for the common external-only case:

- `S` starts as the externals. The boundary set (§5) is computed against `S`.
- If `recompute` is given a `changed` name resolving to an interior node not yet in `S`,
  add it to `S`, **recompute the boundary set**, and **flush** the cone cache (the
  boundaries — hence sub-plans — may have changed). Subsequent recomputes are fast again.
- This is lazy: callers who only ever change externals get full fusion and never expand
  `S`; callers who override interiors pay one boundary recompute + cache flush the first
  time each new interior node is touched.

**Decision (resolved):** interior overrides are **supported** in v1 via the lazy `S`
expansion above. It preserves the mirror contract, the cost is modest, and it keeps
`recompute` a faithful analogue of `CompiledGraph.recompute`.

## 9. Cone cache: key, bound, degradation

- **Key:** `(frozenset(cone node qual_names), frozenset(live-in boundary qual_names))`.
  The cone node-set alone is insufficient: an external `E` and an interior override `I`
  with the same forward cone produce identical node-sets but different live-in
  boundaries (and different generated source) — keying on the boundary too prevents that
  collision. Demotion/boundary are stable within a cache epoch (an `S` change flushes the
  cache), so they need not enter the key.
- **Collapse, honestly scoped.** Keying on the cone (not the raw change-set) collapses
  change-sets that share a forward closure — diamonds / fan-in. For **fan-out** graphs
  (N independent input→chain→output columns) every input subset yields a distinct cone,
  so there is **no** collapse; cone-keying degenerates to change-set keying. The bound
  there is the LRU cap + §7.6 degradation, **not** collapse. (This corrects an earlier
  overstated claim.)
- **Bound:** `self._cone_cache` is an `OrderedDict` capped at `self._cone_cap` (default
  e.g. 64), LRU-evicted. Any secondary `changed→cone` memo (to skip `_collect_affected`
  on exact-repeat change-sets) must be bounded too, or omitted.
- **Degradation:** §7.6 — rebuild-on-miss leans on the on-disk cache; un-fingerprintable
  formulas fall back to the segmented `_Plan` / interpreted path. Never recompile-per-call.

## 10. Algorithm (`recompute(changed)`)

1. If the store is unseeded, raise `RuntimeError` (precondition, §3).
2. Resolve each `changed` name to a `Variable` (external, then interior per §8); write
   the new value into `self._store`; collect `changed_vars`. Interior names may expand
   `S` (§8). Unknown names warn and skip (mirror).
3. `affected = compiled._collect_affected(changed_vars)`; if empty, return current
   outputs from the store (no-op fast path).
4. `key = (frozenset(cone quals), frozenset(live-in boundary quals))`. On hit, move to
   end (LRU) and go to step 7.
5. Build the cone sub-plan: `demoted_in_cone = self._demoted ∩ cone`;
   `order = linearize(cone_nodes, demoted_in_cone)`; `runs = build_runs(...)`. For each
   jit run, live-out `= (required ∪ boundary ∪ later-consumed) ∩ produced` via the new
   cone-liveness helper (§11) over the **full** graph `dependents`; `_generate_segment_body`
   + `_compile`; eager `disp.compile(boundary live-in types)`. Python runs become
   `_PyStep`s. Wrap as a `_ConePlan` (a `run_into(store)` variant of `_Plan`, §11).
6. Insert into the cache; if over `self._cone_cap`, evict LRU (§9, §7.6 for the cold path).
7. Execute the `_ConePlan` against `self._store` in place: read live-ins from the store,
   run each step, write **all step outputs (which include the boundary nodes)** back to
   the store. Guard with `try/except NumbaError → §7.5 flush+reseed+rebuild`.
8. Return `tuple(self._store[v] for v in self._required_vars)`.

## 11. New code vs reuse

**Reused unmodified:** `_collect_affected`, `linearize`, `build_runs`,
`_generate_segment_body`, `_compile`, `_PyStep`/`_JitStep`, the content-addressed digest
and anchor machinery, the discovery demotion logic.

**New (the load-bearing parts the "verbatim reuse" framing got wrong):**
1. **Boundary analysis** — dominator pass over the DAG-with-super-source (§5),
   recomputed on `S` growth (§8). New; `segment_liveness` cannot express
   "consumed by a node outside the subset".
2. **Cone-liveness helper** — live-out `= (required ∪ boundary ∪ cross-segment-consumed)`
   over the **full** graph `dependents`, not `segment_liveness` with cone-local `order`
   (whose `external` param is dead and whose `order` semantics are wrong for
   non-contiguous cones).
3. **`_ConePlan.run_into(store)`** — reads live-ins from and writes **all** produced
   step outputs to the persistent `self._store`, unlike `_Plan.run` which builds a fresh
   `dict(zip(external_vars, args))` and returns only `output_vars`.
4. **Store seeding + `self._demoted` retention + one-time fused-args capture** (§6, §7.4).
5. **Cone cache** (OrderedDict LRU, keying, flush-on-`S`-change / on-type-change) (§9).

## 12. Error taxonomy (additions over v2)

| Condition | When | Outcome |
|---|---|---|
| `recompute` before any full eval | `recompute()` | `RuntimeError` (precondition, §3) |
| `changed` name in neither externals nor nodes | `recompute()` | warn + skip (mirror) |
| interior `changed` name | `recompute()` | expand `S`, recompute boundaries, flush cache (§8) |
| boundary live-in type changed | `recompute()` execution | `NumbaError` caught → flush cache, reseed, rebuild (§7.5) |
| cone formula raises at runtime | `recompute()` execution | propagates unchanged, never demotes (as v2) |
| LRU miss past cap | `recompute()` | rebuild via on-disk cache; un-fingerprintable → segmented/interpreted fallback (§7.6) |

## 13. Testing (`test/core/test_compile_kernel.py` additions)

Equivalence vs the interpreted baseline (the oracle):
- For a battery of graph shapes (pure chain, diamond/join, fan-out independent columns,
  mixed jit+Python cone), a sequence of recomputes equals
  `CompiledGraph.recompute` on the same `changed` sequence — **value-for-value over
  multiple successive calls with different change-sets** (this is where staleness bugs
  surface, per §5/§7.2).
- Fused-case speedup: a single-fused graph's `recompute` re-evaluates only the cone
  (assert via instrumentation / timing that it is not a full re-run).

Boundary / staleness (the §5 crux) — each as a failing-first test:
- Diamond `A,B→M→N`: recompute `E_a` then `E_b`; assert `E_b`'s result uses the fresh
  `A` (the canonical stale-`A` bug); assert the boundary set is `{A,B,N}` and `M` is
  fuse-through.
- Reconvergent / shared-interior cones: overlapping cones read fresh interiors.
- Pure pipeline: boundary set is `{required}` only; whole cone fuses.

Contract & guards:
- precondition `RuntimeError` before any full eval;
- interior override (`changed` names an interior node) — `S` expands, boundary recomputed,
  result matches interpreted override semantics;
- external-vs-interior cache-key collision does **not** occur (same cone, different
  live-in → distinct sub-plans);
- boundary live-in type change → cache flush + correct recompute (no O(cache) cascade);
- mixed-cone (`_ConePlan` with Python steps) keeps interior slots fresh across calls;
- runtime error in a cone formula propagates, no demotion.

Cache / scaling:
- LRU eviction past the cap, then re-hit a thrashed cone → rebuild loads from on-disk
  cache (subprocess pattern), not a full recompile;
- fan-out graph cycling more cones than the cap stays correct and never slower than the
  interpreted baseline (assert against the oracle; timing sanity).

Run with the venv python, `__pycache__` + numba cache cleaned first, per protocol.

## 14. Benchmark (`test/compile_kernel_benchmark.py` extension)

New mode: seed once, then drive `M` recomputes changing a small input subset; report
recompute steady-state vs full `kernel()` re-run vs interpreted
`CompiledGraph.recompute`, across (a) a deep chain (fusion-favorable), (b) a diamond
(boundary-persistence cost), (c) fan-out independent columns (cache-pressure case). This
is the justify-the-feature number for the #24 thread, matching the benchmark practice on
#23/#24.

## 15. Docs

- `docs/numbox.core.variable.rst`: new "Incremental recompute" subsection — the
  value-only/same-types contract, the seed precondition, the boundary-persistence model
  (with the diamond example), the don't-interleave-throughput note, the interior
  Python-node stable-type requirement, and the cache/degradation behavior.
- `compile_kernel`/`CompiledKernel` docstrings: the `recompute` method, precondition,
  contract, and limitations.
- Code blocks flake8-clean (doc-codeblock-flake8); sphinx warning count at or below
  baseline.

## 16. Branch / PR workflow

- Implement on `feat/compile-kernel-hardened` (fork PR #52; bots re-review there first),
  via TDD per §13 (failing test → implementation), one capability at a time
  (store+seed → boundary analysis → cone build/exec → cache/degradation → interior
  overrides).
- Full local CI gate before any push (pytest `--durations=20`, caches cleaned; both
  flake8 passes; doc-codeblock-flake8; sphinx baseline; lychee).
- Fold into #24 by cherry-pick (feature files only — never `docs/superpowers/**`,
  `CLAUDE.md`, or fork-only CI), or open a follow-up PR if #24 has merged. **Upstream
  push and the #24 reply both require explicit per-action consent.**

## 17. Open risks / verification items

1. **Dominator pass correctness on real DAG shapes** — validate the one-pass DAG
   dominator computation against an independent reference on randomized graphs (seeded
   regression), since the boundary rule rests on it.
2. **Seed-evaluation cost in fused mode** — evaluating boundary∪required once on first
   recompute; measure on a large fused graph; if material, a dedicated prime kernel
   (`compile_kernel(graph, boundary∪required)`) reused across reseeds is the escape hatch.
3. **Fan-out cache pressure** — the §7.6 fallback must be measured to confirm it stays at
   or above interpreted-baseline speed when cones thrash.
4. **Interior Python-node type drift** (§7.7) — documented limitation; revisit when the
   `params.type` hints land.
5. **Same-types contract enforcement point** — confirm the boundary type-change guard
   (§7.5) catches the realistic violations and the flush+reseed recovers correctly.
