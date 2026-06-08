# Design: `compile_kernel` — fuse a `Variable` graph into one `@njit` kernel

- **Date:** 2026-06-07
- **Status:** Draft (awaiting review)
- **Module:** `numbox/core/variable/compile_kernel.py` (new)
- **Relates to:** `numbox/core/variable/variable.py` (`Graph`/`CompiledGraph`),
  `numbox/core/work/` (`Work`/`builder.make_graph`),
  `numbox/utils/preprocessing.py` (anchor machinery),
  `numbox/utils/highlevel.py` (`cres`).

## 1. Motivation

`core/variable` builds a pure-Python DAG: `Graph.compile(required)` returns a
`CompiledGraph` that runs each node's `formula` in the Python interpreter
(`CompiledGraph._calculate`). The formulas *may* be JIT-compiled, but the
traversal and the value passing between nodes are Python, so every node crossing
pays interpreter + boxing overhead.

`core/work` already proves that a graph can be turned into JIT-compiled code:
`builder.make_graph` generates Python source for an `@njit` function, `exec`s it,
and returns a graph of `Work` structref nodes; `Work.calculate()` walks that
graph in nopython mode. But `Work` is a *structref graph* — per-node nodes,
dynamic dispatch through a function pointer, and it **requires per-node numba
types** (`builder.get_ty = spec.ty or typeof(spec.init_value)`), which a
`Variable` does not carry.

This module adds a **different compiled target, alongside `Work`**: compile a
`Variable` graph into a **single fused `@njit` kernel** — straight-line code,
interior nodes become SSA temporaries, LLVM can inline and fuse arithmetic
across formulas, and **no per-node type declarations are needed** because numba
infers every interior type from the kernel's runtime argument types.

A proof-of-concept (run against numba 0.65.1) confirmed the approach end to end:
a real `Variable` graph compiled to a fused `@njit` kernel produced results
identical to `CompiledGraph.execute`, auto-specialized on a second input dtype
with zero annotations, and rejected the boundary cases (non-jitted formula,
`formula=None` placeholder) as expected.

## 2. Goals / Non-goals

**Goals**
- `compile_kernel(graph, required)` → a `CompiledKernel` that computes the
  requested variables as one fused `@njit` function.
- Reuse, not reimplement: topological order + external-variable discovery come
  from `Graph.compile`; the content-addressed cache anchor comes from
  `utils/preprocessing.py`.
- Correct under the full range of `Variable` name strings (no identifier
  collisions, no invalid identifiers).
- Cross-process caching that is *correct* (no skeleton-collision reuse).

**Non-goals (v1) — documented limitations, not bugs**
- `cacheable` per-node memoization (a `CompiledGraph`-only feature).
- Incremental `recompute` (a monolithic kernel recomputes everything).
- Formulas that return Python `None` as a value (no nopython representation).
- Node-identity `load`/`combine`/`harvest` (Work-only APIs).
- A `Variable` → `Work`/`Derived` bridge (separate, harder, type-requiring;
  out of scope here).

For any of those, callers keep using `CompiledGraph` or `Work`.

## 3. Public API

```
from numbox.core.variable.compile_kernel import compile_kernel

ck = compile_kernel(graph, ["variables.u", "variables.a"],
                    jit_options=None, cache=True)

# Hot path: bare @njit kernel, positional external args (in ck.params order),
# returns a tuple (in ck.outputs order). Zero Python overhead.
u, a = ck.kernel(100)

# Convenience: dict-in / dict-out, symmetric with CompiledGraph.execute.
out = ck.execute({"basket": {"y": 100}})      # -> {"variables.u": 326.5, "variables.a": 126}

ck.params      # ["basket.y"]                  external inputs, kernel-arg order
ck.outputs     # ["variables.u", "variables.a"]  requested vars, return-tuple order
ck.source      # generated kernel source text (with per-line comments)
ck.identifiers # {qual_name: temp_identifier}  for debugging
```

- `compile_kernel(graph, required, *, jit_options=None, cache=True)` is a free
  function (no method added to `Graph`, keeping the capability self-contained
  and the frozen `Variable` dataclass untouched).
- `graph` is a `core.variable.variable.Graph`; `required` is `str | list[str]`
  (same shape `Graph.compile` accepts). Internally calls `graph.compile(required)`.
- Returns a `CompiledKernel`.

`CompiledKernel` is a small wrapper holding: the compiled `@njit` dispatcher
(`.kernel`), `.params`, `.outputs`, `.source`, `.identifiers`, and `.execute()`.

## 4. Architecture & data flow

```
Graph ──compile(required)──▶ CompiledGraph
                              │  ordered_nodes (topo)            assemble identifiers
                              │  required_external_variables ──▶ + generate source ──▶ exec ──▶ @njit kernel
                              ▼                                  (content-addressed anchor + cache)
                            (reused as-is)                              │
                                                                       ▼
                                                              CompiledKernel(.kernel/.execute/...)
```

`compile_kernel` is one module with internal helpers; the only public symbols
are `compile_kernel` and `CompiledKernel`.

## 5. Identifier scheme (and why the naive one is unsound)

Node temporaries and formula globals in the generated source need identifiers.
`qual_name().replace(".", "_")` is **not safe**, because `source`/`name` are
arbitrary strings (`External.__getitem__` mints a `Variable` for any requested
name; `VarSpec["name"]` is unvalidated):

1. **Underscore ambiguity** — `.` and `_` both collapse to `_`, so distinct
   variables alias: `("a_b","c")` → `a_b_c` and `("a","b_c")` → `a_b_c`.
2. **Invalid identifiers** — `"first-name"`, `"3m"`, spaces, unicode → a
   `SyntaxError` at `exec` (or a valid-but-wrong identifier).
3. **Prefix-namespace collision** — a node whose identifier is `f_x` collides
   with another node `x`'s formula global `f_x`; likewise with injected names
   (`njit`).

**Chosen scheme: readable name + minimal deterministic suffix.**

- `base = sanitize(qual_name)`: replace every char not in `[A-Za-z0-9_]` with
  `_`, collapse runs of `_`, strip leading/trailing `_`, lowercase, and prefix
  `v_` if empty or if it starts with a digit.
- Assemble all symbols that must be unique **as one namespace**: every node's
  temp, every formula global (`f_<temp>`), and the injected helper names
  (`njit`, and any others bound into the exec namespace).
- Detect collisions at assembly time (we hold all nodes up front). When two
  symbols would collide, append `_<suffix>`, where `<suffix>` is the shortest
  prefix of `sha256(true_qual_name).hexdigest()` (lowercase base-16) that makes
  the whole symbol set unique. Grow the suffix length until unique.

Properties: the common case is clean and readable (`variables_a`); suffixes
appear only on genuine clashes (`a_b_c_4f`, `a_b_c_9q`); the scheme is
**deterministic** (suffix seeded from the qual_name, not an RNG), so the
generated source is byte-stable across runs — which the content-addressed cache
(§7) depends on. Names never need to be valid Python identifiers; they survive
only as dict keys in `.execute`/`.params`/`.outputs`.

## 6. Codegen algorithm

Inputs: `compiled.ordered_nodes` (topo order), `compiled.required_external_variables`,
`required` (the requested qual_names).

1. Build `external_set` = the set of `Variable`s in
   `required_external_variables` (these become kernel parameters).
2. Assign each node in `ordered_nodes` an identifier per §5; build
   `var → identifier` and the formula-global names. Collision resolution runs
   over the union of all temps + formula globals + injected names.
3. Parameters: the external nodes, **sorted by qual_name** (deterministic
   signature); record `params` (qual_names) and the matching identifiers.
4. For each node in topo order:
   - external (`variable in external_set`) → it is a parameter; emit nothing.
   - `formula is None` and **not** external → **raise `ValueError`** (a
     derived placeholder cannot be compiled).
   - otherwise → emit `tmp = f_tmp(input_tmps...)` in `variable.inputs` order
     (the 1:1 input↔formula-arg correspondence the variable docs make
     load-bearing), with a trailing comment `# <qual_name> = formula(<inputs>)`.
     Bind the (possibly auto-wrapped, §7) formula as global `f_tmp`.
5. Outputs: map each requested qual_name to its node's identifier; **raise
   `ValueError`** if a requested name is absent or never computed. Emit
   `return (out_tmps,)` (trailing comma → always a tuple, including the
   single-output case).
6. Empty `required` → `ValueError`.

The generated function signature is `def kernel(<param tmps>):` decorated with
`@njit(cache=<cache>, **jit_options)`.

## 7. Formula handling (auto-wrap)

Per derived node's `formula`:
- plain Python function (`types.FunctionType`, not a numba `Dispatcher`) →
  wrap with `njit()` (lazy, no signature) and bind the dispatcher.
- already a numba `Dispatcher` (`@njit`) or a `cres` `CompileResultWAP` → bind
  as-is.
- anything else → bind as-is and let numba decide at compile time.

Genuinely non-jittable bodies surface as a numba `TypingError` pointing at the
formula. Because the kernel is lazily `@njit`-compiled, **that occurs on the
first `ck.kernel(...)` / `ck.execute(...)` call**, not at `compile_kernel()`
time (no argument types are known earlier). Structural errors (§6) raise
eagerly. This eager/lazy split is documented in the API docstring.

Caveat: `lambda` formulas work functionally but their `getsource` is unreliable
for the cache hash (multiple lambdas can share a source line) — documented, not
blocked; callers wanting cache stability should use named functions.

Note: `cres`/`CompileResultWAP` formulas and source-less lambdas (defined
outside any source file) have no recoverable `getsource`, so the cache hash
falls back to `repr(formula)` (per the `_safe_getsource` docstring); this is
correctness-safe (repr is unique per object, never a hash collision) but, being
per-process, gets no cross-process cache reuse. (A `cres` formula additionally
disables caching of the whole kernel via numba's dynamic-globals rule — see
§14 item 1.)

## 8. Caching (content-addressed)

Reuse `numbox.utils.preprocessing`:
- `hash_text` = generated kernel source **+** `getsource()` of every formula in
  node order. Including formula sources is mandatory: two graphs with identical
  straight-line skeletons but different formulas must not share a cache entry
  (a verified failure mode — without it, `cache=True` silently loads the wrong
  binary). The content hash includes each formula's source text **and its
  closed-over constants** (the formula's closure cell values), so two formulas
  built by the same closure factory — identical source text, different captured
  values — get distinct cache entries instead of colliding.
- `anchor = _anchor_path("numbox-compile-kernel", "kernel", hash_text)`;
  `_materialize_anchor(anchor, source)`; `compile(source, str(anchor), "exec")`;
  fold the digest into the kernel function name too (belt-and-suspenders, as
  `builder.make_graph` does).
- `@njit(cache=cache, **jit_options)` over that real on-disk anchor. Compiling
  exec'd source under `<string>` with `cache=True` raises `RuntimeError`; the
  anchor file is mandatory whenever caching is on.
- Call `_orphan_anchor_sweep("numbox-compile-kernel")` at module import.

`jit_options` defaults to `numbox.core.configurations.jit_options` merged with
the caller's overrides; `cache` defaults to `True`.

## 9. Execution & marshalling

- `ck.kernel(*args)` — the bare dispatcher. Args are the external inputs in
  `ck.params` order; returns a tuple in `ck.outputs` order. This is the
  zero-overhead hot path.
- `ck.execute(external_values)` — accepts the `CompiledGraph.execute` dict shape
  `{source: {name: value}}`, looks up each param's `(source, name)`, builds the
  positional args, calls `ck.kernel(*args)`, and zips the returned tuple back to
  `{qual_name: value}` using `ck.outputs`. A missing external raises a `KeyError`
  with the qualified name (mirroring `CompiledGraph._assign_external_values`).

## 10. Error taxonomy

| Condition | When | Error |
|---|---|---|
| `formula is None` on a non-external node | `compile_kernel()` | `ValueError` |
| requested name absent / never computed | `compile_kernel()` | `ValueError` |
| empty `required` | `compile_kernel()` | `ValueError` |
| cycle in the graph | `graph.compile()` (reused) | `RuntimeError` |
| missing external value | `ck.execute()` | `KeyError` |
| non-jittable formula | first `ck.kernel()`/`ck.execute()` | numba `TypingError` |

## 11. Testing (`test/core/test_compile_kernel.py`)

Equivalence vs `CompiledGraph.execute`:
- chain; diamond (merge point); mixed int/float; constant formula (no inputs);
  array-returning formula; multi-output `required`; single-output (1-tuple).

Numba behavior:
- auto-specialization (same kernel, int then float input);
- `cres` formula and `@njit` formula both callable from the kernel (verify
  `cres`/`CompileResultWAP`-by-global-name works; if not, document and fall back
  to plain `@njit` wrapping);
- auto-wrap of a plain-Python formula.

Identifier safety (regression for §5):
- the `("a_b","c")` vs `("a","b_c")` alias pair compiles to distinct temps and
  computes both correctly;
- a name with invalid chars / leading digit (`"first-name"`, `"3m"`) compiles;
- the `f_x`-prefix collision case.

Marshalling & errors:
- `.execute` round-trip; missing-external `KeyError`;
- `formula=None` placeholder → `ValueError`; unknown/uncomputed output →
  `ValueError`; empty `required` → `ValueError`.

Caching:
- two kernels with identical skeleton but different formulas, exercised in a
  **fresh subprocess**, must not collide (the regression test for the
  skeleton-collision footgun) — mirror the existing
  `test_builder.py::test_make_graph_cache_key_content_independent` pattern.

Run via `venv/bin/python -m pytest` from the repo root, caches cleaned first.

## 12. Docs

Extend `docs/numbox.core.variable.rst` with a `compile_kernel` section
(prose + `automodule:: numbox.core.variable.compile_kernel`) and a short
worked example. Run `sphinx-build` and confirm exit 0 (warning count stable).
Code blocks in the `.rst` must be flake8-clean (doc-codeblock-flake8).

## 13. Branch / PR workflow

- Branch: `feat/variable-compile-kernel`, based on the current-upstream synced
  tip (includes `1c9fb24`).
- Full local CI gate before push (pytest `--durations=20`, flake8 at the repo
  config, doc-codeblock-flake8, lychee), caches cleaned first.
- Fork PR first (bot review). Upstream PR only on explicit per-PR approval.
- Exclude `CLAUDE.md` and `docs/superpowers/**` (incl. this spec) from any
  upstream PR.

## 14. Open risks / verification items

1. **`cres`-by-global-name callability. — VERIFIED: works as-is.** The PoC
   verified `@njit` (`CPUDispatcher`) globals are callable by name inside the
   kernel; `cres` returns a `CompileResultWAP` (first-class function value).
   Confirmed by `test_compile_kernel.py::test_cres_formula`: a
   `CompileResultWAP` bound as a kernel global is callable by name and produces
   the correct result, with no wrapping needed (no Option A re-`njit`, no Option
   B fail-fast). Only side effect: numba emits a `NumbaWarning` and disables
   on-disk caching for that kernel ("uses dynamic globals"), because a
   `CompileResultWAP` is treated as a dynamic global; the result is still
   correct, the kernel just recompiles each fresh process. `cres` formulas are
   therefore supported but not cache-eligible.
2. **Cache correctness across processes** — covered by the subprocess test, but
   it is the highest-risk area; treat a green subprocess collision test as the
   gate for shipping `cache=True` as the default.
3. **Eager vs lazy errors** — accepted: non-jittable formulas fail at first
   call. If eager validation is later wanted, add an optional
   `signature=` / `sample_values=` parameter to force compilation; out of scope
   for v1.
