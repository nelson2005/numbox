# Static `params`-driven jitability for `compile_kernel`

- **Date:** 2026-06-16
- **Status:** Design v2 — revised per the adversarial review in
  [`docs/superpowers/reviews/2026-06-16-adversarial-review.md`](../reviews/2026-06-16-adversarial-review.md); pre-implementation.
- **Branch:** `feat/params-jitability` (fork-only artifact; excluded from any upstream PR, like all of `docs/superpowers/**` and `CLAUDE.md`)
- **Origin:** an upstream maintainer comment on [`Goykhman/numbox#24`](https://github.com/Goykhman/numbox/pull/24#issuecomment-4702046203)

## 0. Revision note (v2)

v1 was reviewed adversarially (8 dimensions, empirical numba probes). The
architecture (classify→plan A/B/C, reuse of the fusion/segmentation machinery,
`njit(sig)` over `proxy(sig)`, the cache-digest extension) survived; several
**guarantee claims were empirically false** and two mechanisms were
mis-specified. v2 corrects them. The load-bearing change: **`njit(sig)` does
NOT raise on a coercible-but-wrong declared scalar type — it silently casts**
(`int64`-declared over a `x*1.5` body returns `7`, not `7.5`). Eager
type-error detection therefore requires an **explicit unconstrained
return-type probe**, not the `njit(sig)` bind alone. See §4.

## 1. Context

The merged `compile_kernel` (PR #24) turns a `Variable` graph into a fused
`@njit` kernel. It determines jitability **dynamically**: it builds the fused
source, leaves the kernel uncompiled (`_mode == "virgin"`), and on the **first
call** tries to compile the fully-fused kernel for `typeof(args)`. On a
`NumbaError` it falls into `_discover_and_run`, which *probes every node*
against real intermediate values, demotes numba-compile failures to plain
Python, and orchestrates fused segments around the demoted nodes. `recompute`
then layers incremental cone re-fusion over a Python-side value store.

The upstream maintainer proposed a **static** alternative: give each `Variable`
an optional field declaring whether its formula is jittable and the numba type
of its value, so jitability is *declared* rather than *discovered*, and a
formula's dispatcher can be built from a fully-known signature.

This design adopts the idea and resolves it into a single **classify → plan**
pipeline: when a graph (or part of it) is fully declared, resolution happens at
`compile_kernel()` time — eager compilation, build-time `partition`, errors at
build — and the existing runtime-discovery path remains the fallback for
whatever is *not* declared. Declared and discovered typing become two ends of
one mechanism, not two parallel code paths.

### Goals

- Let a `Variable` optionally declare jit-status and value type.
- When fully declared, resolve the execution mode at build time: eager compile,
  build-time `partition`, build-time type errors **(via an explicit return-type
  probe — see §4 — not via `njit(sig)` alone)**.
- Eliminate the runtime probe for declared graphs — especially mixed
  jit/Python graphs, the most expensive case under today's discovery.
- Keep undeclared graphs **byte-for-byte** identical to today's behavior.
- Compose cleanly with `recompute`.

**Ergonomic cost (stated honestly):** reaching the fully-fused build-time path
(Case A) requires declaring `type` on **every** interior node *and* every used
external, plus `jitable=True` everywhere — there is no minimal-annotation
fast path (numba can infer interior types from external types alone *to
compile*, but the classification rule needs them to *classify*, and Case B
needs them at Python→jit boundaries; see §3). The payoff for that annotation is
build-time errors, build-time `partition`, probe-free mixed graphs, and a
recompute contract check. A graph that declares nothing behaves exactly as
today.

### Non-goals

- Replacing runtime discovery. It stays as the fallback (Case C).
- Per-node `jit_options`. Kernel-level `jit_options` only, as today.
- `None`-as-value formulas, node-identity load/combine — already excluded from
  `compile_kernel` v1; unchanged here.
- Changing the fusion/segmentation model. Declared types feed the **existing**
  fused codegen; they do not introduce per-node Python-orchestrated dispatch.
- A minimal-annotation Case A (interior types inferred). Rejected: it conflicts
  with Case B's mandatory boundary types and the digest extension.

## 2. Data model

A new frozen dataclass in `numbox/core/variable/variable.py`:

```python
@dataclass(frozen=True)
class Params:
    jitable: bool = True        # False => declare this node deliberately plain-Python
    type: Any = None            # numba Type of this Variable's value (None => undeclared)
```

`Variable` gains **one** optional field (one `Params` per `Variable`, describing
that variable's own jit-status and value type):

```python
@dataclass(frozen=True)
class Variable:
    name: str
    source: str = ""
    inputs: Mapping[str, str] = field(default_factory=lambda: {})
    formula: Callable = field(default=None)
    metadata: str | None = None
    params: Params | None = None     # None => behaves exactly as today

    def __post_init__(self):
        if self.params is not None and not isinstance(self.params, Params):
            raise TypeError(
                f"{self.qual_name()!r}: params must be a Params instance, not "
                f"{type(self.params).__name__} (a dict is not accepted)"
            )
```

- `VarSpec` (TypedDict) gains optional `params: Params`. **It must be a `Params`
  instance, not a dict** — the `__post_init__` guard above raises a crisp
  construction-time error if a dict (or anything else) is passed, since the
  codebase's pervasive dict-spec idiom (`inputs`, `jit_options`) would otherwise
  lead a user to write `params={"jitable": True, ...}`, which would pass the
  `is not None` checks and then fail with an opaque `AttributeError` at classify
  time. *(Fixes M6.)*
- `Variables.__init__` spreads `**variable` and so passes `params` through
  cleanly when present in a `VarSpec`.
- Identity is unchanged: `__hash__`/`__eq__` stay `(source, name)`; `params` is
  **not** part of identity. `Params` being frozen keeps `Variable` hashable.

`type` holds a numba `Type` *instance* (e.g. `numba.float64`), not a string
spec.

### Externals carry `params` via pre-seeding (not auto-create)

`External.__getitem__(self, name)` lazily mints an **untyped** `Variable` on
first lookup — there is no parameter to "pass through." A typed external is
declared by **pre-seeding before the first compile**:

```python
graph.external[src].update(name, Variable(name=name, source=src, params=Params(type=float64)))
```

This `update` path is an established pattern in the test-suite. The spec does
**not** claim `External.__getitem__` carries params. *(Fixes M5.)* An optional
ergonomic helper `External.declare(name, params)` may wrap this `update`.

### Before-first-compile ordering contract

`Graph.compile` caches a `CompiledGraph` per **sorted required-tuple**
(`compiled_graphs`), keyed only on the tuple — never on `Variable` content.
Attaching or replacing a `Variable`'s `params` (via `Namespace.update`, or by
swapping a frozen `Variable` instance) **after** a first `compile()` of the same
required set returns the *stale* cached graph, and classification/digest/
`_demoted` then read the old `params`. Therefore: **`params` (like `formula`)
must be attached before the first compile of any required set that includes the
node.** This is a documented hard contract; the implementation additionally
busts the relevant `compiled_graphs` entry on a `Namespace.update` that replaces
a node. *(Fixes M4. The pre-existing-for-`formula` aliasing is now load-bearing
because `params` drives mode selection.)*

### Classification rule

For an interior node (one with a formula), in the required cone:

- **statically jittable** iff `params is not None and params.jitable` is True
  **and** `params.type is not None` **and** every input variable (interior or
  external) has `params is not None and params.type is not None`;
- **statically Python** iff `params is not None and params.jitable is False`;
- **unknown** otherwise (`params is None`, or jitable-but-incomplete types).

This is the maintainer's rule with the sketch bugs fixed (no `None` deref; input
types read from each `input_variable.params.type`; the in-loop assertion checks
the *input* type, not the output type).

**Precondition for a `STATIC_PY` node to participate in Case B:** a
`jitable=False` interior node that is *consumed by a jittable node* **must also
declare `type`** (its declared value type is the live-out the downstream jit
segment's signature is built from — numba cannot infer a type through an opaque
Python node). A `STATIC_PY` node defaults `type=None`, so this must be declared
explicitly; if it is not, the jittable consumer fails rule clause 3 and the
graph routes to Case C (correct, but the rule must state the precondition so the
default does not silently push graphs out of Case B). A **terminal**
`STATIC_PY` node (consumed by nothing jittable) needs no `type`. *(Fixes M1.)*

### Shared external-validation step (runs for ALL graphs, before classification)

The existing hard error for a **formula-bearing external** lives only inside
`_generate_body` (the fully-fused source path), which Case A and Case C reach
but Case B's segment-source path does **not**. A formula-bearing external that
also carries `Params(type=...)` would make its consumer jittable; add a Python
node and the graph becomes Case B, whose path silently ignores the external's
formula and treats its value as a plain input — the exact silent miscompile the
guard exists to prevent. Therefore the formula-bearing-external check (and any
other external-shape validation) is **hoisted into a shared validation step run
once at `compile_kernel()` entry, before classification, for every graph and
every case.** *(Fixes M2.)*

cres/`CFunc`/`DUFunc` formula nodes are honored as jittable when
`params.jitable` is True; their `params.type` is still required so consuming
nodes' input types are known, but the formula itself is not re-wrapped. Their
declarations are validated where possible (§4, fixes H6).

## 3. Resolution pipeline

At `compile_kernel()`, after the shared external-validation step, classify the
required cone. The rule makes the cases exhaustive and disjoint:

### Case A — all interior nodes statically jittable, all *consumed* externals typed

Build the external signature **over only the externals actually consumed by an
interior node** (not all externals): `sig = tuple(ev.params.type for ev in
consumed_externals)` in kernel-arg order, then **eagerly `fused.compile(sig)`**
at build time. A required output that is itself a pass-through external (no
formula, no consumer) is *excluded* from this signature — numba infers its type
from the runtime arg, and demanding a declared type there would otherwise
produce `fused.compile((None, ...))` → an uncaught `TypeError` (not a
`NumbaError`, so the runtime fallback would not catch it). A fully-undeclared
external-only graph (zero interior nodes) routes to Case C. *(Fixes M3.)*
Mode `"fused"` (see construction caveat below), `partition` populated now,
type errors raised now (via the §4 probe).

> Minimality vs annotation cost: numba can infer every *interior* type from the
> external arg types, so strictly only external types are needed to *compile* a
> fully-fused kernel. But the classification rule needs all interior types to
> *reach* Case A, and §4 uses them for the eager return-type probe. There is no
> minimal-annotation Case A (see Goals/Non-goals). This is a real ergonomic
> cost, disclosed in Goals and §8.

### Case B — a mix of `STATIC_JIT` and `STATIC_PY`, no `UNKNOWN`

Use the declared `jitable=False` set **directly** as the demotion set — no
probing. Run the existing `linearize` / `build_runs` to get the
provably-minimal segmentation, then build each jit segment's signature from
**declared** types at its live-in boundary and eagerly compile it:

```python
disp.compile(tuple(v.params.type for v in live_in))   # was: typeof(values[v])
```

Interior `params.type` is essential here: a jit segment downstream of a Python
node has live-ins numba cannot infer through the opaque Python node (this is
why §2 requires a consumed `STATIC_PY` node to declare `type`). Mode
`"segmented"`, `partition` populated now with `reasons` = "declared
non-jittable" per Python node.

### Case C — any `UNKNOWN` node

Fall back to today's lazy first-call resolution, but **declared info shrinks
the probe**: `STATIC_JIT` nodes skip probing, `STATIC_PY` nodes are
pre-demoted, and only `UNKNOWN` nodes are probed in `discover`. For an
undeclared graph this is byte-for-byte today's behavior; `partition` stays
`None` until the first call. Note a Case-C kernel **may contain individually
declared nodes** — this matters for the recompute contract check (§5, M8).

### Exhaustiveness

A node with any missing required type is *unknown* by the rule, so a
half-declared graph that could not build a segment lands in Case C — never a
broken Case B. Cases partition the space: "every interior node jittable + all
needed types present" → A; "every node known, some Python, boundary types
present" → B; "any node unknown / a needed boundary type missing" → C.

### CompiledKernel construction & the eager-fused state machine

`CompiledKernel` today always starts `"virgin"` and resolves on first call. The
eager paths construct it already-resolved, **with one correction the review
forced (H2)**: the `kernel` property returns the *bare* `_fused` dispatcher when
`_mode == "fused"`, so if construction sets `_mode = "fused"` directly, a call
never records `_last_args`, and `recompute`'s `_ensure_store` precondition
(`_mode == "virgin" or _last_args is None`) raises **on every** `recompute()`.
A bare dispatcher also cannot capture args via `__call__` reassignment (Python
resolves dunders on the type).

Resolution: construct eager-fused in a **`"fused-pending"` sub-state**, and add
a branch to the `kernel` property that returns a **self-delegating one-shot
wrapper** which records `self._last_args`, flips `_mode → "fused"` and sets
`partition`, then returns `self._fused(*args)`. Subsequent calls take the bare
dispatcher (hot path unchanged after the first call). This mirrors the lazy
`_resolve_and_call` self-delegation. *(The `kernel` property therefore is NOT
"unchanged" — it gains one branch; §6/§8 updated accordingly.)*

- *Eager fused*: `_mode="fused-pending"`, `_fused` pre-`.compile(sig)`'d,
  `partition` computed, `_ctx` populated as the 6-tuple `compile_kernel` builds.
  First call (via the property's one-shot wrapper) records `_last_args`, flips
  to `"fused"`.
- *Eager segmented*: `_mode="segmented"`, `_plan` built by compiling each
  segment to its **declared** live-in types, `partition` set, `_demoted` seeded
  from declarations; `_run_segmented` records `_last_args` on its first call
  (guarded `if self._last_args is None`).

`CompiledKernel` also records an **`is_declared` flag** (True for eager A/B,
False for Case C) used by `_run_segmented` and the recompute contract check
(§5, fixes H4/M8).

## 4. Binding construction, eager type validation & cache digest

Today every binding is `_wrap_formula(formula, flags)` → `njit(**flags)(formula)`
(lazy, no signature); cres/`CFunc`/`DUFunc`/`Dispatcher` pass through untouched.
The fused kernel — itself `@njit` and content-addressed — **inlines** those
calls when it compiles.

### `njit(sig)` over `proxy(sig)` (validated by review)

Confirmed sound: the fused kernel inlines the formula into the single cached
artifact (verified — the formula's machine code appears inside `_kernel`), so
`proxy`'s per-formula cache anchor / intrinsic static-linking / `inline='always'`
/ `.as_func` are redundant here. `proxy` also asserts a plain `FunctionType`, so
it cannot wrap exotics. New helper `_wrap_formula_typed(formula, sig, flags)`:
`njit(sig, **flags)(formula)` for plain functions; pass-through for exotics.

**Inner `njit(sig)` MUST stay uncached.** `_wrap_formula_typed` threads
`_effective_flags` (which strips `'cache'`); it must **never** set `cache=True`
on an inner formula. The numba cache keys on `co_code` + file `(st_mtime,
st_size)` but **not `co_consts`**, so a `cache=True` inner formula stale-hits on
a pure-numeric-literal body edit and gets inlined *stale* into a freshly
content-addressed fused kernel (verified: `1.0→9.0` edit yielded the stale
result under `cache=True`, correct under `cache=False`). Only the outer
`_kernel_<digest>` artifact is cached; its content-addressed anchor already
defends the literal-edit hazard. *(Fixes M11 — this invariant was implicit in
v1.)*

### Eager type validation — the real guard (corrects H1)

**`njit(sig)` does NOT enforce the declared return type.** It raises only when
the formula's natural return type is *non-convertible* to the declared type
(complex→real, str→numeric, array↔scalar, wrong dtype/ndim). Every coercible
scalar mismatch — int↔float, narrowing (`float64→int64`, `int64→int32`), sign
(`int→uint`) — is **silently cast**, so a node declared `int64` over a `x*1.5`
body computes `7` where the correct/undeclared result is `7.5`. The declared
path would thus *silently miscompute* and be **more dangerous than the
undeclared Case-C baseline**. So:

1. The §3/§5 text must NOT claim a wrong `params.type` "raises at bind time."
   Node-to-node type flow is "numba coerces or rejects each call per its
   conversion rules; only non-convertible mismatches raise" — not
   "type-consistent by construction."
2. **Explicit return-type probe (the eager guard).** For each plain-Python
   `STATIC_JIT` node, compile an **unconstrained** probe overload over the
   declared input types and compare numba's *naturally inferred* return type to
   the declaration:
   ```python
   probe = njit(**flags)(formula)                       # NO explicit return sig
   probe.compile(tuple(inp.params.type for inp in cnode.inputs))
   inferred = probe.nopython_signatures[-1].return_type
   if inferred != node.params.type:                     # identity, not convertibility
       raise <crisp build-time error>
   ```
   The naïve check (reading the return type off an `njit(sig)` dispatcher) is a
   **tautology** — that dispatcher *reports the declared return*. The
   unconstrained probe is what reveals the natural type. Default policy:
   **identity** (forbid all coercion) so "fail fast" is real; a documented
   safe-widening policy (`can_convert` at exact/promote) may be adopted
   explicitly instead. Once validated (natural == declared), the kernel binding
   may **reuse the probe dispatcher** (njit(sig) is then redundant) — avoiding a
   second compile; the eager path already pays compile cost.
3. **Exotic carve-out (fixes H6):**
   - `CFunc`/cres: validate `params.type` against `cfunc_obj._sig.return_type`
     at build (cheap, data present).
   - `DUFunc` (`@vectorize`): has **no single** return type (output dtype
     follows input dtype via ufunc promotion). Either **reject** a
     `params.type` on a DUFunc node (forcing Case C for any graph containing
     one), or validate the declared type against numba's inferred output at the
     node's declared input types via a one-line `@njit` shim that applies the
     DUFunc (reuse the `_call_exotic` shim in `_kernel_partition.py`), raising on
     disagreement. The existing DUFunc tests use `a + 0.5` (always promotes to
     float64), which masks this — a test with an integer-preserving ufunc
     (`a + a`) must be added.

**Per-node signature** (corrects M10 pseudocode): for a `CompiledNode cnode`,
`sig = cnode.variable.params.type(*[inp.params.type for inp in cnode.inputs])`
where `cnode.inputs` is the `list[Variable]` of inputs. (`Variable.inputs` is a
`Mapping[str,str]` and does not carry `.params`; `CompiledNode` has no `.params`
— so the v1 line `node.params.type(*[inp.params.type for inp in node.inputs])`
did not typecheck under either reading.)

**Narrow-scalar caveat (blind spot to settle in implementation):** for
non-native-width declared scalars (`float32`, `int8..32`, `complex64`), numba
may box a node's output back at a different type than declared, which interacts
with the recompute contract check (§5). The initial implementation targets
native-width types; if narrow types are supported, coerce boundary values back
to the declared type before storing and add a narrow-typed equivalence test.

### Cache digest

The generated kernel *source* never mentions types, so two kernels differing
only in declared types produce identical source + formula fingerprints. numba's
own per-signature cache **does** disambiguate them (verified: two declared
variants under one anchor produce two `.nbc` and reload type-correctly), so this
is **not** a correctness hole. The extension is for legibility/1:1 anchors:
**append the declared signatures to the `ck-digest` hash text** (the external
sig for Case A; each segment's live-in/out sig for Case B), encoded via
`repr(signature)` (byte-stable across processes — do **not** route numba `Type`/
`Signature` objects through `_canon_value`, which raises `_Unfingerprintable` on
them). A verification test pins that two declared-type variants get distinct
anchors and do not reuse each other's binary.

## 5. Error timing, `partition`, and `recompute`

### Error timing (corrected per H1)

- **Case A / B:** these errors move to `compile_kernel()` time —
  (i) a formula whose natural return at the declared input types is
  *non-convertible* to the declared type (raised by the `njit(sig)` bind), and
  (ii) **a coercible-but-wrong `params.type`**, raised by the **explicit
  return-type probe in §4** (NOT by `njit(sig)` — that silently casts). Without
  the §4 probe, (ii) would not be caught at all. A cross-node type mismatch is
  caught at the eager fused/segment compile.
- **Case C:** timing unchanged from today (first-call) — a declared node whose
  inputs trace to `UNKNOWN` upstream nodes cannot be compiled eagerly.
- Contract: **fully-declared graphs (A/B) fail fast at build** *for both
  non-convertible mismatches (njit) and coercible mismatches (the §4 probe)*;
  any-undeclared graph (C) fails at first call, exactly as today.
- Runtime errors still propagate and never demote — unchanged.

### `partition` at build time

For A and B, `partition` is populated at construction (Case A: single jit
segment, `reasons={}`; Case B: the static segmentation with `reasons` =
"declared non-jittable"). Case C leaves `partition = None` until the first call.
So declared graphs' partitions are inspectable and assertable without calling
the kernel.

### Composition with `recompute`

`recompute` needs the value store seeded with **all** node values plus the
frozen `_demoted` set. Eager modes skip `_discover_and_run`, so:

1. **Seed `_last_args` on the first call in eager modes** — eager-segmented at
   the top of `_run_segmented` (guarded `if self._last_args is None`);
   eager-fused via the `"fused-pending"` one-shot wrapper (§3). This is what
   makes `recompute` usable after a fused call (fixes H2).

2. **Seed `_demoted` at construction from declarations** — empty for Case A,
   the declared-`jitable=False` set for Case B.

3. **`_ensure_store` must not re-run `discover` for declared graphs.** `discover`
   *re-derives* demotions by probing and would **disagree** with declarations (a
   node declared `jitable=False` that happens to compile fine would be treated
   as jittable). Factor the existing `discover` into:
   - `_evaluate(ordered_nodes, external, values, bindings, flags, demoted)` —
     run the graph to populate interior values using a **fixed** demotion set,
     preserving `discover`'s three per-node branches (exotic via the
     `_call_exotic` shim + `continue`; `Dispatcher` via `binding(*args)`;
     demoted via `py(*args)`), and the no-fallback `TypeError` for an untypeable
     node;
   - `discover(...)` = evaluate *while discovering* demotions (today's
     behavior), expressed via `_evaluate`'s loop.

   `_ensure_store` calls `_evaluate` with the declared `_demoted` for declared
   graphs; keeps `discover` for Case C.

4. **`_run_segmented` must not re-`discover` for declared kernels (fixes H4).**
   Today `_run_segmented` catches `NumbaError` and falls into
   `_discover_and_run`, which **unconditionally overwrites** `self._demoted`,
   `_store`, and `partition` by re-probing — silently discarding authoritative
   declarations (and re-classifying a declared `jitable=False` node that
   compiles). Gate this on `is_declared`: for eager (A/B) kernels,
   `_run_segmented` does **not** fall back to `_discover_and_run`; a segment
   `NumbaError` for an off-contract input type re-raises as the crisp
   "declared type X, got Y" violation. Only Case-C kernels re-discover.

5. **recompute type-contract check: use `can_convert`, scoped to eager mode
   (fixes H3/M8).** v1's implied `typeof(new_value) == params.type` is wrong:
   it returns `False` for a C-contiguous array vs a declared `float64[:]`
   (layout `A`), and for `float32` vs declared `float64` — both of which numba
   **accepts** (verified). Replace with a numba assignability check in
   `_apply_changes`: raise only when
   `typingctx.can_convert(typeof(new_value), var.params.type)` is `None` (accept
   exact/promote/safe), or equivalently normalize array layout before comparing.
   **Scope the check to eager (A/B) kernels** (`if self._is_declared and
   var.params is not None and var.params.type is not None`): a Case-C kernel may
   hold individually-declared nodes whose cones compile against discovered
   `typeof` (not `params.type`), so applying the check there would wrongly raise
   on a type change that the existing flush-and-reseed recovers from.

   The drop of `_flush_and_reseed` for declared cones stands, but **not** on
   v1's stated rationale ("the recover would fail anyway, since the cone is
   compiled to the declared type") — that is **false**, because cone dispatchers
   are lazy (`disp.compile((sig,))`, not signature-locked), so a C-layout
   overload compiles and runs fine. The correct justification: for declared
   nodes the type is fixed by *contract*, so an out-of-contract type is a
   user error to report crisply, not to recover from; undeclared (Case C) nodes
   keep the existing recovery.

Net: declared graphs get a precise (convertibility-based) recompute contract
check and shed speculative recovery; undeclared graphs behave exactly as today.

## 6. Affected code

- `numbox/core/variable/variable.py` — `Params` dataclass; `Variable.params`
  field + `__post_init__` Params-instance guard; `VarSpec` `params` key;
  passthrough in `Variables.__init__`; optional `External.declare` helper; bust
  `compiled_graphs` entry on `Namespace.update` node replacement.
- `numbox/core/variable/utils.py` — `_wrap_formula_typed` (uncached inner
  `njit(sig)`); the unconstrained return-type probe helper.
- `numbox/core/variable/_kernel_partition.py` — split `discover` into
  `_evaluate` + `discover`; reuse `linearize`/`build_runs` for static Case B;
  the DUFunc shim validation reusing `_call_exotic`.
- `numbox/core/variable/compile_kernel.py` — shared external-validation step
  (hoisted formula-bearing-external guard) run for all graphs; classification;
  Case A/B/C dispatch (Case A sig over *consumed* externals only); eager
  `CompiledKernel` construction with `"fused-pending"` sub-state +
  `is_declared` flag; the `kernel`-property one-shot-capture branch; `_compile`
  digest extension (repr of declared sigs); `_ensure_store`/`_apply_changes`/
  `_build_cone_plan`/`_run_segmented`/`_flush_and_reseed` changes; docstring
  updates (module docstring L1-22 + "Error timing" + "Caching" +
  "Non-jittable formulas" sections).
- **Docs (mandatory):** `docs/numbox.core.variable.rst` — overview (per-node
  types now *optional*; declaring all of them moves errors to build), Caching
  (declared sigs extend the digest), recompute (declared-type contract check).
  Run `cd docs && /home/erik/projects/numbox/venv/bin/sphinx-build -b html .
  _build/html` and confirm exit 0. *(Fixes H5.)*

## 7. Testing strategy

- **Data model:** `Params` defaults; `Variable.params` passthrough via
  `VarSpec`/`Variables`; identity/hash unaffected by `params`; **negative test:
  a dict-form `params` raises `TypeError` at construction** (M6).
- **Classification → mode at build (no call):** representative A/B/C graphs;
  assert `_mode` and `partition` immediately after `compile_kernel`. Assert the
  Case-B `PartitionReport` shape at construction (the §5.2 testability win).
- **Case A:** fully-declared all-jittable graph → eager fused; `partition`
  pre-call; correct results; **a coercible wrong `params.type` (declared `int64`
  over a `x*1.5` body) raises at `compile_kernel()`** — this passes only if the
  §4 return-type probe is implemented, not `njit(sig)` alone (H1); a
  non-convertible mismatch also raises; cross-node mismatch raises at build;
  pass-through external output graph compiles (M3).
- **Case B:** mixed declared jittable/non-jittable → eager segmented; minimal
  segments; declared boundary types used; `reasons` "declared non-jittable";
  **formula-bearing external + `Params(type)` still raises the external guard in
  Case B** (M2).
- **Case C:** undeclared and partially-declared graphs behave exactly as today;
  `partition` None until first call; only unknown nodes probed; **a declared
  node inside a Case-C kernel does NOT trip the recompute contract check** (M8).
- **recompute on declared graphs:** works after a first full call (H2);
  declared-type contract violation raises crisp "declared type X, got Y";
  `_flush_and_reseed` not invoked for declared cones; **declared-array recompute
  equivalence** (C-contiguous array seed against a `float64[:]` declaration does
  NOT raise — H3); **eager kernel does not re-`discover` on an off-contract
  type** (H4).
- **Exotics:** `CFunc`/cres `params.type` validated against `_sig.return_type`;
  **DUFunc with an integer-preserving ufunc (`a + a`)** validated or rejected
  (H6); narrow-typed (`float32`) equivalence if narrow types are supported.
- **Cache:** two declared-type variants → distinct anchors, no binary reuse;
  cross-process reload; **co_consts-edit regression** confirming inner formulas
  are uncached (M11).
- **Inner-uncached invariant** and **docs build** (sphinx exit 0).
- **Regression:** `test/core/test_compile_kernel.py` passes unchanged.

Per project rules: clean `__pycache__` and the numba cache before each pytest
run; run via `/home/erik/projects/numbox/venv/bin/...`; full local CI gate
(flake8 max-line-length=127, `pytest --durations=20`, doc build,
doc-codeblock-flake8) before any push.

## 8. Risks / open questions / verification tasks

- **H1 enforcement unverified end-to-end (blind spot).** The "unconstrained
  probe → compare natural inferred return to `params.type`" guard was reasoned
  about, not prototyped against the real `_wrap_formula_typed` codegen. The
  implementer must confirm
  `njit(**flags)(formula).compile((inp_types,)).nopython_signatures[-1].return_type`
  yields the *natural* (un-coerced) type, and fix the safe-widening policy
  (default: identity).
- **Narrow-scalar box-back (blind spot).** `float32`/`int8..32`/`complex64`
  boundary values may box at a type other than declared, interacting with the
  §5.5 contract check. Native-width first; coerce-back + test if narrow types
  are supported.
- **numba per-signature cache vs the shared anchor.** Confirmed not a
  correctness hole (verified); the digest extension is for legible 1:1 anchors.
  Pin with a test that two declared variants don't collide/reuse binaries.
- **`discover`/`_evaluate` refactor must not change Case C** — pin
  byte-for-byte.
- **Ergonomic cost (M9).** Case A requires declaring `type` on every interior
  node; there is no minimal-annotation path. Stated in Goals; not a defect.
- **macOS-arm64 + py3.14 DCE (M12 — structural, not just a "verification
  task").** The `query_to_array` DCE bug was specific to raw-pointer stores
  (`array_data_p` + `store_unaligned`). The eager/fused/segmented codegen emits
  only SSA arithmetic + a tuple return (no raw-pointer stores) and shares the
  identical source with the already-green fused path the CI matrix exercises, so
  the DCE mechanism is **structurally absent**. Keep a smoke test as
  belt-and-suspenders.
- **Eager-fused first-call overhead.** The one-shot capture runs once; the bare
  dispatcher is the hot path thereafter — confirm no per-call overhead post
  first call.

## 9. Pre-implementation note

`numbox` fork `origin/main` is current with `upstream/main` (synced via PR #57,
`18b4b24`, latest upstream tag `0.5.18`), so `feat/params-jitability` is
correctly based — no sync needed before implementation. This is fork-internal
work; whether any of it is ever offered upstream, and in what form, is a
separate decision and out of scope for this spec.
