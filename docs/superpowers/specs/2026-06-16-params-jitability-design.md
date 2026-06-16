# Static `params`-driven jitability for `compile_kernel`

- **Date:** 2026-06-16
- **Status:** Design — approved, pre-implementation
- **Branch:** `feat/params-jitability` (fork-only artifact; excluded from any upstream PR, like all of `docs/superpowers/**` and `CLAUDE.md`)
- **Origin:** Goykhman's comment on [`Goykhman/numbox#24`](https://github.com/Goykhman/numbox/pull/24#issuecomment-4702046203)

## 1. Context

The merged `compile_kernel` (PR #24) turns a `Variable` graph into a fused
`@njit` kernel. It determines jitability **dynamically**: it builds the fused
source, leaves the kernel uncompiled (`_mode == "virgin"`), and on the **first
call** tries to compile the fully-fused kernel for `typeof(args)`. On a
`NumbaError` it falls into `_discover_and_run`, which *probes every node*
against real intermediate values, demotes numba-compile failures to plain
Python, and orchestrates fused segments around the demoted nodes
(`compile_kernel.py:320-402`). `recompute` then layers incremental cone
re-fusion over a Python-side value store.

Goykhman proposes a **static** alternative: give each `Variable` an optional
field declaring whether its formula is jittable and the numba type of its
value, so jitability is *declared* rather than *discovered*, and a formula's
dispatcher can be built from a fully-known signature.

This design adopts the idea but resolves it into a single **classify → plan**
pipeline: when a graph (or part of it) is fully declared, resolution happens at
`compile_kernel()` time — eager compilation, build-time `partition`, errors at
build — and the existing runtime-discovery path remains the fallback for
whatever is *not* declared. Declared and discovered typing become two ends of
one mechanism, not two parallel code paths.

### Goals

- Let a `Variable` optionally declare jit-status and value type.
- When fully declared, resolve the execution mode at build time: eager compile,
  build-time `partition`, build-time type errors.
- Eliminate the runtime probe for declared graphs — especially mixed
  jit/Python graphs, the most expensive case under today's discovery.
- Keep undeclared graphs **byte-for-byte** identical to today's behavior.
- Compose cleanly with `recompute`.

### Non-goals

- Replacing runtime discovery. It stays as the fallback (Case C).
- Per-node `jit_options`. Kernel-level `jit_options` only, as today.
- `None`-as-value formulas, node-identity load/combine — already excluded from
  `compile_kernel` v1; unchanged here.
- Changing the fusion/segmentation model. Declared types feed the **existing**
  fused codegen; they do not introduce per-node Python-orchestrated dispatch.

## 2. Data model

A new frozen dataclass in `numbox/core/variable/variable.py`:

```python
@dataclass(frozen=True)
class Params:
    jitable: bool = True        # False => declare this node deliberately plain-Python
    type: Any = None            # numba Type of this Variable's value (None => undeclared)
```

`Variable` gains **one** optional field (not a list — one `Params` per
`Variable`, describing that variable's own jit-status and value type):

```python
@dataclass(frozen=True)
class Variable:
    name: str
    source: str = ""
    inputs: Mapping[str, str] = field(default_factory=lambda: {})
    formula: Callable = field(default=None)
    metadata: str | None = None
    params: Params | None = None     # None => behaves exactly as today
```

- `VarSpec` (TypedDict) gains optional `params: Params`.
- `External.__getitem__` and `Variables.__init__` pass `params` through.
- Identity is unchanged: `__hash__`/`__eq__` stay `(source, name)`. `Params`
  being frozen keeps `Variable` hashable. `params` is **not** part of identity
  — two variables that differ only in `params` are the same node (this matters
  for the compiled-graph caches keyed on `Variable`).

`type` holds a numba `Type` *instance* (e.g. `numba.float64`), matching
Goykhman's `isinstance(variable_type, Type)` check — not a string spec.

### Classification rule

For an interior node (one with a formula), in the required cone:

- **statically jittable** iff `params is not None and params.jitable is True`
  **and** `params.type is not None` **and** every input variable (interior or
  external) has `params is not None and params.type is not None`;
- **statically Python** iff `params is not None and params.jitable is False`;
- **unknown** otherwise (`params is None`, or jitable-but-incomplete types).

This is Goykhman's rule with the sketch bugs fixed (no `None` deref; input
types read from each `input_variable.params.type`; the in-loop assertion checks
the input type, not the output type). cres/`CFunc`/`DUFunc` formulas — which
the shipped code already treats as always-jittable — are honored as jittable
when `params.jitable` is True; their `params.type` is still required so
consuming nodes' input types are known, but the formula itself is not re-wrapped
(see §4).

Externals (no formula) may carry `Params(type=...)`. An external used as input
to a jittable node **must** declare `type`, otherwise that consuming node is
*unknown* by the rule above.

## 3. Resolution pipeline

At `compile_kernel()`, classify the required cone. The rule makes the cases
exhaustive and disjoint:

### Case A — all interior nodes statically jittable, all used externals typed

Build the external signature `sig = tuple(ev.params.type for ev in external_vars)`
in kernel-arg order (the order `_generate_body` already fixes), and **eagerly**
`fused.compile(sig)` at build time. Mode `"fused"`, `partition` populated now,
type errors raised now.

> Minimality note: numba can infer every interior type from the external arg
> types, so *strictly* only external types are needed to compile a fully-fused
> kernel. The design nonetheless binds each jittable node to its declared
> signature (§4) so a wrong declaration is caught eagerly and the kernel is
> type-consistent by construction. The classification rule requires all interior
> types anyway, so they are always present to bind against.

### Case B — a mix of statically-jittable and statically-Python, no unknowns

Use the declared `jitable=False` set **directly** as the demotion set — no
probing. Run the existing `linearize` / `build_runs` (`_kernel_partition`) to
get the provably-minimal segmentation, then build each jit segment's signature
from **declared** types at its live-in / live-out boundary and eagerly compile
it:

```python
disp.compile(tuple(v.params.type for v in live_in))   # was: typeof(values[v])
```

This is where interior `params.type` is essential and not merely validation: a
jit segment downstream of a Python node has live-ins whose types numba *cannot*
infer through the opaque Python node — they must be declared. Mode
`"segmented"`, `partition` populated now with `reasons` = "declared
non-jittable" per Python node.

### Case C — any unknown node

Fall back to today's lazy first-call resolution, but **declared info shrinks the
probe**: statically-jittable nodes skip probing, statically-Python nodes are
pre-demoted, and only *unknown* nodes are probed in `discover`. For an
undeclared graph (`params is None` everywhere) this is byte-for-byte today's
behavior; `partition` stays `None` until the first call.

### Exhaustiveness

The rule routes any node with a missing required type to *unknown*, so a
half-declared graph that could not build a segment lands in Case C — never a
broken Case B. Cases partition the space: "every node jittable" → A; "every node
known, some Python" → B; "any node unknown" → C.

### CompiledKernel construction

`CompiledKernel` gains an already-resolved construction path (today it always
starts `"virgin"` and resolves on first call):

- **eager fused:** `_mode="fused"`, `_fused` pre-`.compile(sig)`'d, `partition` set.
- **eager segmented:** `_mode="segmented"`, `_plan` built by compiling each
  segment against **declared** live-in types, `partition` set, `_demoted` seeded
  from declarations.

The hot-path `kernel` property is unchanged — it already dispatches on `_mode`
(`compile_kernel.py:298-307`).

## 4. Binding construction & cache digest

Today every binding is `_wrap_formula(formula, flags)` → `njit(**flags)(formula)`
(lazy, no signature) for plain functions; cres/`CFunc`/`DUFunc`/`Dispatcher`
pass through untouched (`utils.py:65`). The fused kernel — itself `@njit` and
content-addressed — **inlines** those calls when it compiles.

### Use `njit(sig)`, not `proxy(sig)` — deliberate divergence from the sketch

Goykhman's sketch builds `proxy(sig, **jit_options)(formula)` per node. For the
fused/segmented model that is the wrong tool:

- The fused kernel (and each segment) is one `@njit` artifact that numba inlines
  the formula into and that numbox already content-addresses and caches as a
  whole. So `proxy`'s value-adds — its own per-formula cache anchor, the
  intrinsic-based static linking, `inline='always'`, `.as_func`
  (`proxy.py:49,57-67,96`) — are redundant *here*: the formula's standalone
  compiled object never runs on the hot path (the inlined copy inside the fused
  artifact does), so caching it separately is pure overhead.
- `njit(sig)` inlines identically, is already handled by `_formula_fingerprint`
  (it is a `Dispatcher`), and eagerly compiles the formula to its declared
  signature at bind time — which is what surfaces a wrong declaration as a
  build-time error (§5).
- `proxy` asserts a plain `FunctionType` target (`proxy.py:52`), so it could not
  wrap cres/`CFunc`/`DUFunc` anyway.

New helper `_wrap_formula_typed(formula, sig, flags)` alongside `_wrap_formula`:
`njit(sig, **flags)(formula)` for plain functions; pass-through for
cres/`CFunc`/`DUFunc`/`Dispatcher` (for those, `params.type` is declared but the
formula is not re-wrapped — the user owns the declaration matching the
artifact's real return type). `_check_formula_arity` is unchanged.

A node's signature is built from its own and its inputs' declared types:
`sig = node.params.type(*[inp.params.type for inp in node.inputs])`. Because each
node's sig is built from its inputs' declared output types, the fused kernel's
node-to-node calls are **type-consistent by construction**; the only failure mode
is a formula that cannot meet its declared sig, which raises at bind time.

### Cache digest

The generated kernel *source* never mentions types, so two kernels differing
only in declared types produce identical source + identical formula
fingerprints. numba's own per-signature cache would disambiguate them (each
declared external sig compiles a distinct overload), so this is **not** a
correctness hole — but it is cheap and clearly better to make the numbox anchor
1:1 with a concrete typed kernel: **append the declared signatures to the
`ck-digest` hash text** (the external sig for Case A; each segment's live-in/out
sig for Case B). `_kernel_<digest>.py` filenames then never alias across type
variants, which also keeps cache debugging legible. A verification test pins
that two declared-type variants of the same graph get distinct digests and do
not reuse each other's binary.

## 5. Error timing, `partition`, and `recompute`

### Error timing

- **Case A / B:** typing errors move from first-call to `compile_kernel()` time
  — a formula that cannot type at its declared sig (`njit(sig)` bind), a
  cross-node type mismatch (caught at the eager fused/segment compile), and a
  **wrong `params.type`** (a new eager check: `njit(sig)` enforces the declared
  return type and raises on mismatch).
- **Case C:** timing is unchanged from today (first-call). This is not laziness
  for its own sake — a declared node in a Case C graph often *cannot* be
  compiled eagerly because its inputs trace back to unknown upstream nodes whose
  types are unknown until a real call. The contract is crisp: **fully-declared
  graphs (A/B) fail fast at build; any-undeclared graph (C) fails at first call,
  exactly as today.**
- Runtime errors still propagate and never demote — unchanged.

### `partition` at build time

For A and B, `partition` is populated at construction (Case A: single jit
segment, `reasons={}`; Case B: the static segmentation with `reasons` =
"declared non-jittable"). Case C leaves `partition = None` until the first call.
So for declared graphs the partition is inspectable and assertable **without
calling the kernel** — a real testability win.

### Composition with `recompute`

`recompute` needs the value store seeded with **all** node values
(externals + interiors) plus the frozen `_demoted` set. Today that seeding
happens inside `_discover_and_run` (segmented first call) or `_ensure_store`
(fused first call, which re-runs `discover` from `_last_args`). Eager modes skip
`_discover_and_run`, so four things change:

1. **Seed `_last_args` on the first call in eager modes** (mirrors the lazy
   design, where `_resolve_and_call` captures args once; the documented contract
   is "don't interleave throughput calls between recomputes"). Eager-segmented
   captures it at the top of `_run_segmented`; eager-fused uses a one-shot
   capturing wrapper for its first call, then hands back the bare dispatcher (the
   hot path stays the bare dispatcher thereafter).

2. **Seed `_demoted` at construction from declarations** — empty for Case A, the
   declared-`jitable=False` set for Case B. No probe.

3. **`_ensure_store` must NOT re-run `discover` for declared graphs.** `discover`
   re-derives demotions by probing, and would **disagree** with declarations: a
   node the user declared `jitable=False` may actually compile fine, so
   `discover` would treat it as jittable, contradicting the authoritative
   declaration. Refactor the existing `discover` (`_kernel_partition`) into:
   - `_evaluate(ordered_nodes, external, values, bindings, flags, demoted)` —
     run the graph to populate interior values using a **fixed** demotion set;
   - `discover(...)` — = evaluate *while discovering* demotions (today's
     behavior), now expressed in terms of `_evaluate`'s evaluation loop.

   `_ensure_store` calls `_evaluate` with the declared `_demoted` for declared
   graphs (declarations are authoritative) and keeps calling `discover` for Case
   C. This also de-duplicates the evaluation logic the two paths share today.

4. **Drop the speculative type-change recovery for declared nodes.**
   `_flush_and_reseed` exists to recover when a live-in's *type* changes and a
   cached cone dispatcher fails to compile. For a declared variable the type is
   fixed by contract; a changed type is a violation, not something to silently
   recover from (and the recover would fail anyway, since the cone is compiled to
   the declared type). So in `_apply_changes`, validate `typeof(new_value)`
   against `params.type` for declared variables and raise a crisp
   `"declared type <X>, got <Y> for <qual_name>"` error. Undeclared (Case C)
   nodes keep the existing flush-and-reseed recovery untouched. Declared cone
   live-ins compile against `v.params.type` rather than `typeof(self._store[v])`,
   uniform with eager-segmented build.

Net: declared graphs get a precise recompute contract check and shed the
speculative recovery machinery; undeclared graphs behave exactly as today.

## 6. Affected code

- `numbox/core/variable/variable.py` — `Params` dataclass; `Variable.params`
  field; `VarSpec` `params` key; passthrough in `External`/`Variables`.
- `numbox/core/variable/utils.py` — `_wrap_formula_typed`.
- `numbox/core/variable/_kernel_partition.py` — split `discover` into
  `_evaluate` + `discover`; reuse for static segmentation in Case B.
- `numbox/core/variable/compile_kernel.py` — classification; Case A/B/C
  dispatch; eager `CompiledKernel` construction; digest extension;
  `_ensure_store`/`_apply_changes`/`_build_cone_plan` changes; docstring
  "Error timing" update.

## 7. Testing strategy

- **Data model:** `Params` defaults; `Variable.params` passthrough via
  `VarSpec`, `External`, `Variables`; identity/hash unaffected by `params`.
- **Classification → mode at build (no call):** representative graphs that land
  in A, B, C; assert `_mode` and `partition` immediately after `compile_kernel`.
- **Case A:** fully-declared all-jittable graph → eager fused; `partition` known
  pre-call; correct results; wrong `params.type` raises at `compile_kernel()`;
  cross-node type mismatch raises at `compile_kernel()`.
- **Case B:** mixed declared jittable/non-jittable → eager segmented; minimal
  segments; `partition` reasons "declared non-jittable"; declared boundary types
  used; correct results.
- **Case C:** undeclared and partially-declared graphs behave exactly as today
  (probe path); `partition` is `None` until first call; only unknown nodes
  probed.
- **recompute on declared graphs:** works after a first full call (store seeded
  via `_evaluate`); declared-type contract violation raises the crisp error; the
  flush-and-reseed machinery is not invoked for declared cones.
- **Cache:** two declared-type variants of one graph → distinct digests/anchors,
  no binary reuse; cross-process reload of a declared kernel.
- **cres/`CFunc`/`DUFunc` declared nodes:** bound as-is; `params.type` used for
  consumers.
- **Regression:** the existing `test/core/test_compile_kernel.py` suite passes
  unchanged (it exercises only undeclared graphs).

Per the project's testing rules: clean `__pycache__` and the numba cache before
each pytest run; run via the venv python at
`/home/erik/projects/numbox/venv/bin/...`.

## 8. Risks / open questions / verification tasks

- **numba per-signature cache vs the shared anchor.** Pin (test) that two
  declared-type variants do not collide and do not reuse each other's binary,
  with and without the digest extension. *(Verification task.)*
- **`discover` / `_evaluate` refactor must not change Case C.** Pin that
  undeclared-graph behavior is byte-for-byte unchanged. *(Verification task.)*
- **macOS-arm64 + py3.14 DCE hazard.** The platform that produced the
  `query_to_array` all-zeros bug (see CLAUDE.md). Ensure the declared-type eager
  path does not regress there. *(Verification task — CI matrix already covers
  this platform.)*
- **Eager-fused first-call capture overhead.** Confirm the one-shot `_last_args`
  wrapper does not slow the hot path after the first call (it should not — the
  bare dispatcher is returned thereafter).
- **`params.type` as numba `Type` instance** (decided): instances, not string
  specs, matching the sketch's `isinstance(..., Type)`.

## 9. Pre-implementation note

`numbox` fork `origin/main` is current with `upstream/main` (synced via PR #57,
`18b4b24`, latest upstream tag `0.5.18`), so `feat/params-jitability` is
correctly based — no sync needed before implementation. (This is fork-internal
work; whether any of it is ever offered upstream, and in what form, is a
separate decision and out of scope for this spec.)
