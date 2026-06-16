# Adversarial Review — `2026-06-16-params-jitability-design.md`

> Reviews the design spec `docs/superpowers/specs/2026-06-16-params-jitability-design.md`.
> Method: 8 independent adversarial dimension reviewers → per-finding adversarial verification (empirical numba probes where the claim is about runtime behavior) → completeness critic → synthesis. 56 verified findings (30 real: 12 high, 12 medium, 14 low; 18 refuted).
> Fork-only artifact — excluded from upstream PRs, like the spec.

## 1. Executive summary

The design is **architecturally sound but rests on a false premise** that recurs in its three most load-bearing sections. The classify→plan pipeline (A/B/C), the decision to reuse the existing fusion/segmentation machinery, the choice of `njit(sig)` over `proxy(sig)`, and the cache-digest extension are all correct and well-reasoned. However, the central safety claim — that binding each node via `njit(sig)` makes a wrong `params.type` "raise at bind time" (§4 L208-209, §5 L232-233, §3 L122-124) — is **empirically false for the common case**: numba silently *coerces* a convertible-but-wrong scalar return type (`int64`↔`float64`, narrowing, sign), so a mis-declared graph silently miscomputes instead of failing fast, and is *strictly worse* than the undeclared Case-C path it complements. Separately, the eager-fused `recompute` seeding is self-contradictory as specified (guaranteed `RuntimeError`), and the `recompute` type-contract check (`typeof(new) == params.type`) spuriously rejects everyday array/scalar inputs. None of these block the architecture, but four spec claims are factually wrong as written and must be corrected before implementation, plus three real mechanism gaps need resolving.

**Verdict: APPROVE-WITH-CHANGES** — the design is the right shape; specific guarantee claims are false and two mechanisms are under/mis-specified.

---

## 2. Critical / High findings

### H1. `njit(sig)` silently COERCES a wrong scalar `params.type` — the eager-error guarantee is false for the common case
**Spec:** §3 L122-124 ("a wrong declaration is caught eagerly"); §4 L208-209 ("the only failure mode is a formula that cannot meet its declared sig, which raises at bind time"); §5 L232-233 ("a wrong `params.type` … `njit(sig)` enforces the declared return type and raises on mismatch"); §7 L316 test bullet.

The single most important finding, corroborated independently across **seven** verifier dimensions. Re-verified directly:

```
int64(int64) lambda x*1.5  d1(5): 7 (true 7.5) sigs [(int64,) -> int64]   # NO RAISE, truncated
uint64(int64) lambda x-100 d2(3): 18446744073709551519                    # NO RAISE, wraparound
float64(int64) lambda x+1  d3(3): 4.0                                      # NO RAISE, widened
float64(complex128): RAISED TypingError
int64(int64) str:     RAISED TypingError
int64(int64) array:   RAISED TypingError
```

`njit(sig)` raises **only** when the formula's natural return is *non-convertible* to the declared type (complex→real, str→numeric, array↔scalar, array dtype/ndim). For every coercible scalar mismatch — `int↔float`, narrowing (`float64→int64`, `int64→int32`), sign (`int→uint`) — it inserts a silent cast and compiles clean. End-to-end this propagates through the fused kernel: a node declared `int64` over a `x*1.5` body returns `7` where the undeclared Case-C path correctly returns `7.5`. So a mis-declared coercible type is not merely *uncaught* — the declared (Case A/B) path **actively miscomputes**, diverging from both the correctly-declared graph and today's undeclared baseline, with no build-time or run-time error.

The feature's headline value proposition ("fail fast at build", §5 L238-239) is **not delivered** for the most likely declaration mistake in a numeric graph, and the declared path is *more dangerous* than the undeclared one it sits alongside.

**Recommended spec change (do all four):**
1. **Correct the false text.** §4 L208-209 and §5 L232-233: replace "raises on mismatch" / "the only failure mode … raises at bind time" with the truth — *"`njit(sig)` raises only when the formula's natural return type has no implicit numba cast to the declared type; a convertible-but-wrong declared type (including lossy scalar narrowing) is silently cast, so a wrong scalar `params.type` is NOT caught at build."* Narrow §4's "type-consistent by construction" to "numba coerces or rejects each node-to-node call per its conversion rules; only non-convertible mismatches raise."
2. **Add the real guard** (eager detection is the stated reason to prefer `njit(sig)` over `proxy`). The naïve fix is a **tautology**: comparing `dispatcher.nopython_signatures[0].return_type` after `njit(sig)` against `params.type` always matches, because that signature *reports the declared return*. The sound check: for each plain-function node, compile a probe overload with **no return constraint** over the declared input types (`njit(**flags)(formula).compile(tuple(inp.params.type …))`), read numba's *naturally inferred* `return_type`, and require it `== node.params.type` (identity, or an explicitly-stated safe-widening policy). Raise a crisp build-time error on mismatch. Cost: one extra unconstrained compile per declared plain node — which the eager path already pays compile cost for.
3. **Carve out exotics.** cres/`CFunc`/`DUFunc` pass-through nodes (§4 L199-203) — CFunc/cres *can* be validated against `_sig.return_type` cheaply (see H6); DUFunc cannot the same way (see H6). Document the carve-out.
4. **Fix the §7 test (L316).** As written ("wrong `params.type` raises at `compile_kernel()`") it passes only for a non-coercible type and gives false confidence. With fix (2), make it assert a *coercible* wrong type (declared `int64` on a `x*1.5` body) raises — which the implementation can only satisfy with the explicit return-type probe.

### H2. Eager-fused never sets `_last_args` → `recompute()` always raises after a fused call
**Spec:** §3 L165 ("eager fused: `_mode="fused"`"), §3 L170 ("hot-path `kernel` property is unchanged"), §5.1 L262-263 ("one-shot capturing wrapper for its first call, then hands back the bare dispatcher").

The spec holds three statements that cannot coexist. Verified directly: with `_mode="fused"` set at construction and the unchanged property (`compile_kernel.py:300-304`) returning `self._fused`, a call leaves `_last_args=None`:

```
kernel is bare dispatcher: True
call result: (8.0,)
_last_args after call: None
_store after call: None
after __call__ override, k(5.0): (10.0,) hook ran: False
```

`_ensure_store` (`compile_kernel.py:407-411`) raises `RuntimeError` whenever `_mode=="virgin" or _last_args is None`, so **every** `recompute()` on an eager-fused kernel raises — defeating Goal "Compose cleanly with recompute" (L40). The wrapper can't be installed by reassigning the dispatcher's `__call__` (Python resolves dunders on the type — confirmed: `hook ran: False`). Refined to **high**: realizable with a small change, and this is a pre-implementation doc so nothing ships broken — but the literal reading produces a guaranteed `RuntimeError`.

**Recommended spec change:** Do NOT set `_mode="fused"` at eager-fused construction and do NOT claim the property is "unchanged." Either (i) construct with a sentinel `"fused-pending"` mode and ADD a branch to the `kernel` property returning a one-shot wrapper that sets `self._last_args`, flips `_mode→"fused"` + `partition`, then returns `self._fused(*args)`; or (ii) keep `_mode="virgin"` and pre-populate `_fused`/`partition` so the existing `_resolve_and_call` seeds `_last_args` at L323 and takes the already-compiled fused branch (no recompile, sig pre-`.compile`'d). Update §3 L170 to say the property *gains* a one-shot-capture branch, and reconcile §8's overhead note.

### H3. `recompute` type-contract check `typeof(new) == params.type` spuriously rejects valid inputs
**Spec:** §5 point 4 (L287-292) "validate `typeof(new_value)` against `params.type` … raise a crisp 'declared type X, got Y'", framed by "the type is fixed by contract; a changed type is a violation."

The spec never names the comparison operator; the "got Y / a changed type is a violation" framing points an implementer straight at `==` (or set membership), which is wrong. Verified:

```
typeof(np.zeros(5)) == float64[:]:   False   (layout C vs declared A)
typeof(np.zeros(5)) == float64[::1]: True
can_convert(C-array -> float64[:]): 3        (safe — numba ACCEPTS it)
float64 == typeof(np.float32(1)):   False
can_convert(float32 -> float64):    2        (promote — numba ACCEPTS it)
int64   == typeof(np.int32(1)):     False
```

A user who declares the natural `float64[:]` (layout `A`) and seeds with an ordinary C-contiguous numpy array — or passes `np.float32`/`np.int32` where `float64`/`int64` was declared — hits the "crisp" error on their own *valid* data. Breaks the declared==interpreted equivalence contract for array-valued nodes and regresses vs the undeclared path. Array live-ins/interiors are real here (`test_compile_kernel.py:264,540,686`).

Sub-claim corrected: cone dispatchers are **lazy** (`compile_kernel.py:491-492` `disp.compile((sig,))`, not signature-locked), so the §5.4 justification *"the recover would fail anyway, since the cone is compiled to the declared type"* is **false for array types** — the C-layout overload compiles and runs fine.

**Recommended spec change:**
1. §5 point 4: replace `typeof(new_value) == params.type` with a numba assignability check — raise only when `typingctx.can_convert(typeof(new_value), params.type)` is `None` (or worse than a chosen `Conversion` threshold). Equivalently normalize array layout before comparison.
2. Drop the "a changed type is a violation, fixed by contract" framing as justification for an exact check — the contract is *convertibility*, not bit-identical `typeof`.
3. Correct or remove the false "the recover would fail anyway" claim.
4. Add a declared-**array** recompute equivalence test (current equivalence tests use only scalars).

### H4. Eager-segmented `NumbaError` fallback re-runs `discover()` and overwrites the declaration-seeded `_demoted`
**Spec:** §5 item 3 ("declarations are authoritative"; `discover` "would disagree"); §3 L167-170.

The design fixes `_ensure_store` to not re-run `discover`, but leaves `_run_segmented`'s `except NumbaError → _discover_and_run` path untouched (`compile_kernel.py:338-346`). `_discover_and_run` re-probes and **unconditionally** sets `self._demoted = demoted` (line 399), rebuilds the plan, reseeds `_store`. For a declared Case-B kernel, a later signature that breaks a jit segment silently re-probes — exactly the "discover disagrees with declarations" situation §5 item 3 says must be prevented — and a user-declared `jitable=False` node that happens to compile fine is silently re-classified jittable (discover only demotes on *compile failure*, `_kernel_partition.py:333-337`), discarding the authoritative declaration and contaminating `partition`/`_store`/`_demoted`. Refined to **medium**: trigger is a user contract violation and numerics stay correct, but it silently violates the design's own authority-of-declarations contract.

**Recommended spec change:** Apply §5-item-4's stance uniformly to the throughput path. For eager (A/B) kernels, `_run_segmented` must NOT fall back to `_discover_and_run`; gate the fallback to Case-C graphs only, and for declared graphs re-raise the crisp "declared type X, got Y" violation (record an `is_declared` flag on `CompiledKernel` and branch on it in both `_run_segmented` and `_discover_and_run`). Add `_run_segmented` + the no-overwrite-of-declared-`_demoted` invariant to §6 and a §7 test.

### H5. Sphinx RST (and most of the module docstring) is omitted from the docs deliverable
**Spec:** §6 (only "docstring 'Error timing' update"); §7 (no doc-build item).

`docs/numbox.core.variable.rst` states contracts the design changes: L213-220 ("No per-node type information needs to be supplied … the first call detects them"), L236-254 (Caching digest — no declared sigs), L341-348 (recompute "same types across calls"). The same prose lives in the **module docstring** at `compile_kernel.py:1-22` (L5-8), and the function docstring's "Caching:" / "Non-jittable formulas:" sections (L633-658) are affected — §6 names only the single "Error timing" docstring section and no RST. Per the project's mandatory-docs rule, an RST + module-docstring update and a `sphinx-build` (exit 0) verification are required deliverables the spec omits. Refined to **medium**: undeclared-graph claims stay *true* (Case C preserved), so the prose is **incomplete for declared graphs**, not "falsified."

**Recommended spec change:** Add a §6 documentation bullet covering `docs/numbox.core.variable.rst` (overview ~213-220: per-node types now *optional*; Caching ~236-254: declared sigs extend the digest; recompute ~341-348: declared-type contract check) AND the `compile_kernel.py` module docstring (L5-8) + "Caching:"/"Non-jittable formulas:" sections. Add a §7 "sphinx-build clean (exit 0)" deliverable.

### H6. DUFunc nodes have no single validatable `params.type`; CFunc/cres carry one the design declines to check
**Spec:** §2 L98-102, §4 L199-203 ("the user owns the declaration matching the artifact's real return type").

For cres/`CFunc`/`DUFunc` the spec requires `params.type` but never validates it against the artifact. A `CFunc`/cres carries a fixed `_sig.return_type` that *could* be validated cheaply — declining is a missed eager check. A `DUFunc` (`@vectorize`) has **no single return type**: output dtype is a function of input dtype via ufunc promotion (`du(int64)→int64`, `du(float64)→float64`). Existing DUFunc tests (`test_compile_kernel.py:783,815,858,1347`) all use `a + 0.5` which always promotes to float64, masking the gap; an integer-preserving ufunc (`a+a`) exposes it. A wrong/inapplicable `params.type` then silently coerces at the consumer boundary (same mechanism as H1). Refined to **high**: within one fixed external signature a DUFunc output *is* single-valued, so the defect is the *absence of validation* + silent coercion, not impossibility.

**Recommended spec change:** (a) Validate `CFunc`/cres `params.type` against `cfunc_obj._sig.return_type` at build. (b) For DUFunc, either reject a `params.type` (forcing Case C for any graph containing one), or validate the declared type against numba's inferred output at the node's declared input type via a one-line `@njit` shim (reuse the `_call_exotic` shim at `_kernel_partition.py:258`), raising on disagreement. Part of the same enforcement gate as H1.

---

## 3. Medium / Low findings

### M1. Case B reachability under-specified (Low)
§2 L90-94 vs §3 L139-141,L155-158. The rule requires every input typed for a jittable verdict, but a `jitable=False` interior node defaults `type=None`. A typeless Python node feeding a jit consumer makes that consumer *unknown* → graph falls to Case C, not Case B. Routing-to-C is *correct* (Case B can't compile the downstream segment without the Python node's declared live-out type), but the rule never states the precondition and the default discourages it. **Fix:** near L93/106 add the explicit requirement mirroring the external rule ("an interior `jitable=False` node consumed by a jittable node must also declare `type`"); tighten Exhaustiveness prose at L157; note a *terminal* Python node needs no type and reaches B unconditionally.

### M2. Formula-bearing external + `Params(type)` silently bypasses the hard-error guard in Case B (Medium)
§2 L104-106, §3 Case A/B. The ValueError for a formula-bearing external lives only in `_generate_body` (`compile_kernel.py:177-183`), which Case A/C reach but Case B's `_generate_segment_body` path does not. The new rule checks input `params.type` presence but not `input.formula is None`, so a formula-bearing external carrying `Params(type=…)` makes its consumer jittable; add a Python node → Case B → the external's formula is silently ignored and its value taken as a plain input — the exact silent miscompile the ValueError exists to prevent. **Fix:** state an invariant that the formula-bearing-external check runs for ALL graphs at `compile_kernel()` time, before/independent of classification (hoist to a shared external-validation step). Add a Case-B test.

### M3. Pass-through external output (no formula, untyped) breaks Case A's external-sig construction (Medium)
§2 L88,104-106; §3 Case A L113-118. A required output can be an untyped external with no consumer (`test_external_only_output_end_to_end` passes today). Case A builds `sig` over *all* externals; an output-only external's `None` type yields `fused.compile((None,))` → uncaught `TypeError` (not `NumbaError`, so the runtime fallback at L331 wouldn't catch it). The type demand is gratuitous: numba infers a pass-through output's type from the runtime arg. **Fix:** build the Case A sig only over externals actually *consumed* by an interior node; make the predicates total for the zero-interior-node corner; route a fully-undeclared external-only graph to Case C.

### M4. `compiled_graphs` cache returns a stale `CompiledGraph` after post-compile `params`/formula replacement (Medium)
§2 L78-81, §4, §5. `Graph.compile` keys `compiled_graphs` only on the sorted required-tuple (`variable.py:419-422`), never on Variable content. If a user attaches/changes `params` by swapping a Variable via `Namespace.update` after a first compile of the same required set, `graph.compile` returns the OLD CompiledGraph; classification/digest/`_demoted` then read stale `params`, silently producing a kernel under old declarations. Already ships in #24 for `formula`, but `params` now drives mode selection + build-time classification, raising stakes. §2's identity acknowledgment covers Variable-keyed `affected_cache`/`dependents`, NOT string-keyed `compiled_graphs`. **Fix:** document a hard contract that `params` (like formulas) must be attached *before* first compile (add a construction-time typed-external API so `update` isn't the only route — see M5), or bust the `compiled_graphs` entry on Variable replacement. Test: compile undeclared → attach params via update → recompile must not return the stale Case-C kernel.

### M5. External `params` has no clean construction path; §6 "passthrough in External" misdescribes it (Medium)
§2 L77,105-106; §6 L299-300. `External.__getitem__(self, name)` takes only a name and lazy-mints an untyped Variable; there is no parameter to "pass through." The only working route is pre-seeding via `graph.external[src].update(name, Variable(..., params=...))` *before* compile — an established pattern (`test_compile_kernel.py:946`) the spec never names. The `Variables.__init__` half (`Variable(source=…, **variable)` at `variable.py:192`) IS a genuine passthrough. **Fix:** reword §6 L300 to name the `update` pre-seed mechanism; add the before-compile ordering contract (ties to M4); optionally add a thin `External.declare(name, params)` helper. Add a typed-external-via-`update` test.

### M6. VarSpec `params` boundary contract unspecified; a dict-form `params` raises a cryptic late AttributeError (Low)
§2 L76. `Variables.__init__` spreads `**variable` verbatim with no coercion; TypedDict is unenforced. Given the codebase's pervasive dict-spec idiom, a user will plausibly write `params={"jitable": True, "type": float64}`. A truthy dict passes the `is not None` guard, then `params.jitable` raises `AttributeError` at classify time. Loud failure, never silent — hence Low. **Fix:** state in §2 that VarSpec `params` must be a `Params` instance (deliberately not a dict); add a `Variable.__post_init__` `isinstance` guard; fold a negative test into the §7 passthrough test.

### M7. `_evaluate` must replicate `discover`'s three population branches — editorial (Low; verifier: non-issue)
§5 point 3. `discover` has three per-node paths (exotic `_call_exotic`+`continue`; Dispatcher `binding(*args)`; demoted `py(*args)`). Verifier *refuted* harmful consequences empirically (exotics are directly Python-callable; `_call_exotic` yields identical values/types), and §5-point-3 already constrains `discover` to "today's behavior expressed via `_evaluate`." **Fix (optional):** expand the §5 one-liner to enumerate the three branches + the no-fallback-untypeable `TypeError`.

### M8. §5.4 contract check unscoped vs kernel mode — declared node inside a Case-C kernel (Medium)
§5 point 4. Classification is per-*kernel* but `params` are per-*node*; a Case-C kernel can hold individually-declared nodes (§3 L147-151). In Case C, cones compile against discovered `typeof`, not `params.type` (`compile_kernel.py:492`), so the §5.4 "recover would fail anyway" justification is **false for Case C** — flush-and-reseed works. Read literally, the check fires for any node with `params.type` set, so a declared node in a Case-C kernel would *raise* on a type change instead of recovering — a Case-C regression. **Fix:** gate the contract check on eager (A/B) kernel mode; keep typeof-based recovery for all nodes in Case-C kernels; correct the §5.4 justification.

### M9. Minimality note doesn't surface the all-interior-types ergonomic cost (Low)
§3 L120-125 vs §2 L90-94. The minimality note is empirically correct that a fully-fused Case-A kernel compiles from external types alone, but §2 transitively forces *every* interior node typed for Case A — no minimal-annotation path. Goals/Non-goals/Risks never list the full-annotation cost. **Fix (option b only):** add a one-paragraph note in Goals/§8 that Case A's build-time benefits require declaring `type` on every interior node + `jitable=True`. Do NOT make interior types optional/inferred (option a) — conflicts with Case B's essential interior types and the digest extension.

### M10. §4 per-node sig pseudocode doesn't typecheck under either reading of `node` (Low)
§4 L205-206 `sig = node.params.type(*[inp.params.type for inp in node.inputs])`. `Variable.inputs` is `Mapping[str,str]` (iterating yields name *strings*, no `.params`); `CompiledNode.inputs` is `list[Variable]` but `CompiledNode` has no `.params` (only `.variable`). Both readings raise `AttributeError`. §2 L97 prose states the correct intent. **Fix:** rewrite as `sig = cnode.variable.params.type(*[inp.params.type for inp in cnode.inputs])` and state `cnode` is a `CompiledNode`.

### M11. Inner `njit(sig)` must stay uncached — load-bearing invariant left implicit (Medium)
§4 L199-200. The co_consts cache hazard (numba hashes `co_code` not `co_consts`, keyed on file `(st_mtime, st_size)`) means a `cache=True` inner formula stale-hits on numeric-literal edits and gets inlined *stale* into a freshly content-addressed fused kernel — verified: literal change `1.0→9.0` yielded `11.0`/stale under `cache=True`, `19.0`/correct under `cache=False`. The design is safe *only* because `_effective_flags` strips `'cache'` (`compile_kernel.py:58-64`), an invisible invariant. Since the spec elevates eager bind-time compilation to a feature, "should I cache the eagerly-compiled inner?" is a natural question an implementer can answer wrongly → silent miscompile. **Fix:** add one sentence to §4: the inner `njit(sig)` MUST stay uncached (thread `_effective_flags`, never `cache=True`); only the outer `_kernel_<digest>` artifact is cached. Optionally add a co_consts-edit regression test.

### M12. macOS-arm64 DCE risk entry is a non-mitigation; add the structural rationale (Low)
§8 L345-348. The `query_to_array` DCE bug was specific to raw-pointer stores (`array_data_p` + `store_unaligned`); the eager/fused/segmented codegen emits only SSA arithmetic + a tuple return (`_emit_lines`/`_assemble_source`, no raw-pointer stores) and shares the identical source with the already-green fused path. The original DCE mechanism is structurally absent. **Fix:** replace the bullet with the one-line structural argument (no raw-pointer stores; shares source with the shipped fused path the CI matrix already exercises; keep a smoke test as belt-and-suspenders).

---

## 4. Refuted / non-issues (coverage appendix)

- **"1:1 anchor with a concrete typed kernel" overstated (overload growth under one anchor)** — *Non-issue.* §4 defines "type variants" as distinct declared-type *builds*; the digest extension disambiguates those. Runtime overload growth is already acknowledged ("not a correctness hole").
- **External-types-only compile / helper inlining / non-coercible build-time error / `njit(sig)` vs `proxy` inlining** — *Confirmed correct (non-defect).* All four reproduce; they validate §3/§4/§5. Residual risk owned by H1.
- **`jitable=True, type=None` silently demoted to unknown** — *Non-issue.* Routes to Case C → correct results; a targeted "you forgot type" diagnostic is *impossible* (`Params()` ≡ `Params(jitable=True)` ≡ `Params(jitable=True, type=None)`, equal-hashing — verified). Docs nicety at most.
- **Diamond/multi-cone classification consistency** — *Non-issue.* `ordered_nodes` is deduped; one cone per call; classification location implicitly pinned.
- **Captured-once kernel reference bypasses property capture** — *Non-issue.* A self-delegating wrapper satisfies held-reference validity AND a bare hot path AND one-shot capture (verified). Subsumed by H2's fix.
- **Unconditional `_last_args` capture corrupts flush-and-reseed** — *Refuted as silent corruption; Low residual.* Flush-fires-vs-result-returned conditions are mutually exclusive. Residual: guard the shared `_run_segmented` capture with `if self._last_args is None`.
- **Construction-path wiring for `_fused.compile(sig)`** — *Non-issue.* §3 fully specifies the end-state; eager compile pinned as build-time, so a raising compile aborts the whole call.
- **"No digest change needed for correctness"** — *Confirmed TRUE.* numba's per-sig `.nbi` keys on `(sig, magic, co_code/closure hashes)`; two declared variants under one anchor produce two `.nbc` and reload type-correctly (verified). Affirms the spec.
- **`_canon_value` raises on numba Type/Signature** — *Non-issue → Low.* True, but element order is pinned by existing qual_name sorts, `repr` is byte-stable, digest non-correctness-critical. Optional "encode via `repr`, don't route through `_canon_value`" note.
- **Digest fragments cache across declared/undeclared boundary** — *Non-issue → Low.* Real but bounded (one extra trivial compile + anchor).
- **Digest not sufficient for true 1:1 anchor under polymorphic dispatch** — *Non-issue.* Semantic nitpick on "1:1" against a goal the spec never set.
- **§7 cache test "no binary reuse" wording invites a brittle file-shape assertion** — *Non-issue → Low.* Codebase already uses a behavioral oracle.
- **CFunc-bearing declared kernels uncached → digest inert** — *Non-issue.* DUFunc IS cacheable, so the cache test is exercisable with the exotic family.
- **Per-call classification doesn't avoid staleness** — *Non-issue.* Rebuts a strawman; staleness is pre-existing (genuine concern is M4).
- **Within-compile params-out-of-identity safety rests on registry-singleton** — *Non-issue.* Safety rests on `__eq__`/`__hash__` on `(source,name)` (verified).
- **recompute change-source resolution masks params mismatches** — *Non-issue.* The §5.4 check reads `params.type` off the *same* pinned instance the cone is compiled against. Conditional restatement of M4.
- **params-not-in-identity dedup, last-writer-undefined** — *Non-issue.* Namespace-caching guarantees one canonical instance per `(source,name)`.
- **Thread-safety / concurrent compile unaddressed** — *Non-issue.* Pre-existing; numba's `global_compiler_lock` serializes compilation; eager build is *more* thread-confined (verified 8-thread probes).
- **Empty-external Case A recompute seeding** — *Non-issue.* `sig=()` compiles; store seeded via `_evaluate`/`discover` from nullary roots (verified).
- **`explain`/`dependents_of`/`debug=True` interactions** — *Non-issue.* These read only metadata/inputs/identity (none of which `params` touches); `debug=True` is CompiledGraph-only, disjoint from `compile_kernel`.
- **Steelman of proxy's caching/declaration-linking** — *Non-issue (affirms spec).* `njit(sig)` inlines the formula into the single fused artifact (verified: `fmul` present), so proxy's standalone cache/declaration-link has nothing to share; the §4 `njit(sig)`-over-`proxy` choice is sound.

---

## 5. Blind-spots / residual coverage note

- **The H1 enforcement fix is unverified end-to-end.** The "compile an unconstrained probe overload and compare the natural inferred return to `params.type`" guard was reasoned about but not prototyped against the real `_wrap_formula_typed` codegen. Confirm `njit(**flags)(formula).compile((inp_types,)).nopython_signatures[-1].return_type` yields the *natural* (un-coerced) type before relying on it — and decide the safe-widening policy.
- **Narrow-scalar box-back (float32/int8-32/complex64) was flagged High by one verifier but is not in the top findings.** It interacts with H3 (box-back type differs from declared narrow type, bypassing the declared overload and tripping an exact contract check), masked by the all-float64/int64 suite. If narrow types are supported, coerce boundary values back to declared types before storing (or accept best-effort eager compile) + add a narrow-typed equivalence test.
- **No probe confirmed `partition` is fully populated and assertable at construction for a real Case-B graph** (the §5.2 testability win). Low risk; mechanical.

---

## 6. Final verdict

**APPROVE-WITH-CHANGES.** The classify→plan architecture, the A/B/C partition, the `njit(sig)`-over-`proxy` decision, the digest extension, and the Case-C-preservation goal are all correct. The blockers are false guarantee claims and two mis-specified mechanisms, all fixable at the spec level without redesign.

**Ordered must-fix checklist (before implementation):**

1. **H1 / H6** — Correct the false "wrong `params.type` raises at bind time" claims (§3 L122-124, §4 L208-209, §5 L232-233), and add the real eager guard: a natural-return-type probe for plain nodes (not the tautological post-`njit(sig)` comparison), `_sig.return_type` check for CFunc/cres, reject-or-shim policy for DUFunc. Fix the §7 wrong-type test to use a coercible mismatch.
2. **H2** — Resolve the eager-fused state-machine contradiction: introduce a `"fused-pending"` sub-state with a self-delegating one-shot capture wrapper; do not set `_mode="fused"` at construction; update §3 L170 + §8.
3. **H3 / M8** — Replace the `==` recompute contract check with `can_convert`-based assignability, scoped to eager (A/B) kernel mode; correct the false "recover would fail anyway" justification (cone dispatchers are lazy); add declared-array and Case-C-declared-node tests.
4. **H4** — Stop `_run_segmented`'s `NumbaError` fallback from re-`discover`-ing and overwriting declaration-seeded `_demoted` for declared kernels; add to §6/§7.
5. **M2 / M3 / M5 / M4** — Make external validation total: preserve the formula-bearing-external guard for all cases (incl. Case B); exempt pass-through external outputs from the Case-A sig; name the `update` pre-seed path for typed externals and the before-first-compile ordering contract (stale `compiled_graphs`).
6. **M11 / M10 / M6** — Pin the inner-`njit(sig)`-must-stay-uncached invariant; fix the §4 `node.params.type(...)` pseudocode to typecheck; require a `Params` instance at the VarSpec boundary with a construction-time guard.
7. **H5 / M9 / M12** — Add the RST + module-docstring + sphinx-build deliverable to §6/§7; surface the all-interior-types ergonomic cost in Goals/§8; replace the macOS-DCE risk bullet with its structural rationale.

**Relevant code sites:** `compile_kernel.py` (state machine 274-307, `_resolve_and_call` 320-336, `_run_segmented`/`_discover_and_run` 338-402, `_ensure_store`/`_apply_changes`/`_build_cone_plan`/`_flush_and_reseed` 404-525, `_effective_flags` 58-64, formula-bearing-external guard 177-183); `variable.py` (identity 163-167, `compiled_graphs` 419-436, `External.__getitem__` 109-124, `Variables.__init__` 192); `utils.py` (`_wrap_formula` 65-72); `_kernel_partition.py` (`discover` 310-352, `_call_exotic` 258); docs at `docs/numbox.core.variable.rst` (213-220, 236-254, 341-348).
