"""Compile a `core.variable` Variable graph into fused @njit kernel(s).

This module turns a `Graph`/`CompiledGraph` into JIT-compiled straight-line code, alongside `core.work` (a structref
graph). When every formula can be compiled with njit, the whole graph is compiled into a single fused
@njit function whose interior nodes are SSA temporaries. No per-node type information is needed, because numba
infers every interior type from the kernel's runtime argument types. When some formulas cannot be compiled with
njit, the first call detects them automatically: a numba compile error demotes a node to plain Python, and a runtime
error always propagates. A Python master then orchestrates fused @njit segments around the demoted nodes, and a
fusion-maximizing linearization chooses the segment boundaries. `CompiledKernel.partition` describes the resulting
partition as a PartitionReport with per-node demotion reasons. Formulas that have no Python fallback
(cres/CompileResultWAP, CFunc, DUFunc) are always treated as jittable.

Jitability may be discovered at the first call, as above, or declared in advance. Each `Variable` carries an optional
`params` (`Params(jitable, type)`), and a node with no `params` is discovered exactly as described above. When every
node in the required cone is declared, and every consumed external is typed, `compile_kernel()` resolves the
execution mode at build time instead of at the first call. An all-jittable graph compiles eagerly into one fused
kernel ("fused"). A declared mix of jit and Python nodes compiles eagerly into a static segment plan ("segmented"),
with no probing. `CompiledKernel.partition` is populated at build and can be inspected before any call. If a declared
`params.type` is not the type that the formula naturally yields, an explicit unconstrained return-type probe catches
the mismatch at build. Binding the formula to the declared signature would not catch it, because numba would
silently coerce a convertible but wrong scalar type: a node declared int64 over a float-returning body would return a
truncated value. Any node left undeclared, or only partially typed, keeps the runtime-discovery path, and a graph
that declares nothing behaves exactly as described above.

The on-disk cache is content-addressed per compiled unit: the fused kernel, or each jit segment. The digest
fingerprints each formula's code, constants, default arguments, closure-cell values, referenced globals and the
values of referenced module attributes. It also holds the kernel's effective jit flags, the targetoptions of any
Dispatcher or DUFunc formula, and the ``NUMBA_*`` settings that change the emitted code
(``numbox.core.cache_key.KEYED_SETTINGS``). The kernel itself is a function that numbox compiles, so its own cache
key holds them as well. The kernel source never mentions types, so declared signatures are appended to the digest
too, and two declared-type variants of one graph get distinct cache anchors. A formula with no canonical fingerprint
makes its unit uncached: the unit has no anchor and no numba cache.

This freshness guarantee covers the code that numbox generates and njit-wraps from raw Python. It does *not* extend
to a formula that carries its *own* numba on-disk cache, a ``@njit(cache=True)`` Dispatcher. That cache's key does
not include the global values the dispatcher reads, and numba's own key does not include its jit flags either.
Folding those values into the outer kernel name cannot invalidate that cache, and a fresh outer kernel would
otherwise link the stale inner binary and serialize it. Such a unit is therefore compiled without an on-disk cache,
and numbox issues a warning. numbox does not compound the staleness of that inner cache, but it cannot correct it:
after a change to the user dispatcher's flags or globals, that dispatcher still runs its own stale binary
in-process. A value or dispatcher that a module exposes *only* through a PEP 562 ``__getattr__`` is outside the
digest, because resolving it would call the ``__getattr__`` of unrelated modules on every compile. Clear the cache
after changing such a value.
"""
import hashlib
import warnings

from collections import OrderedDict
from types import FunctionType, ModuleType
from typing import Any, Callable, NamedTuple

from numba import njit, typeof
from numba.core.caching import NullCache
from numba.core.ccallback import CFunc
from numba.core.dispatcher import Dispatcher
from numba.core.errors import NumbaError
from numba.np.ufunc.dufunc import DUFunc

from numbox.core.configurations import jit_options as _default_jit_options
from numbox.core.variable._kernel_partition import (
    PartitionReport, Segment, _ConePlan, _JitStep, _Plan, _PyStep,
    _evaluate, build_runs, compute_boundary, cone_liveness, discover, linearize, segment_liveness,
)
from numbox.core.variable.utils import (
    _assign_identifiers, _check_formula_arity, _validate_declared_return, _wrap_formula,
)
from numbox.core.variable.variable import (
    QUAL_SEP, CompiledGraph, CompiledNode, Graph, Variable, make_qual_name,
)
from numbox.core.cache_key import _settings_canon
from numbox.utils.fingerprint import (
    _Unfingerprintable, _canon_value, _effective_flags, _fingerprint_function,
    _flags_canon, _referenced_global_names, _safe_repr,
)
from numbox.utils.highlevel import _type_identity
from numbox.utils.preprocessing import (
    _anchor_root, _anchored_or_uncached, _orphan_anchor_sweep,
)

_ANCHOR_SUBDIR = "numbox-compile-kernel"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)


class _KernelCtx(NamedTuple):
    """Build context passed into a CompiledKernel; it stays on the Python side and never enters njit code."""
    compiled: Any
    idents: dict
    bindings_by_var: dict
    jit_options: dict | None
    cache: bool | None
    external: set


def _formula_fingerprint(formula) -> tuple[str, bool]:
    """Behavioral identity of a formula for the cache digest.

    Returns ``(text, cacheable)``. The text covers every value channel that numba freezes into a compiled artifact.
    These are the code object's bytecode, constants and names, the default-argument values, the closure-cell values,
    the values of referenced module-level globals, the defining module, and the targetoptions of a dispatcher or
    DUFunc. The walk over the globals recurses into helper functions and dispatchers, with cycle protection. Builtins
    resolve outside ``__globals__`` and are deliberately not hashed.

    Any value with no canonical form makes the formula un-fingerprintable. The returned text is then a per-object
    placeholder and ``cacheable`` is False, so the kernel is compiled without an on-disk cache: it is never reused,
    and it is never wrong. A CFunc formula embeds an ASLR-randomized address, so numba can never cache on disk a
    kernel that calls it. Such a formula is fingerprinted deterministically but marked uncacheable.
    """
    target = getattr(formula, "py_func", None)
    extra = ""
    if target is None and isinstance(formula, (DUFunc, CFunc)):
        target = getattr(formula, "__wrapped__", None)
        extra = f";kind={type(formula).__name__}"
    if target is None:
        target = formula
    if not isinstance(target, FunctionType):
        return f"{_safe_repr(formula)} @{id(formula)}", False
    try:
        if isinstance(formula, (Dispatcher, DUFunc)):
            extra += ";targetoptions=" + _canon_value(dict(getattr(formula, "targetoptions", {}) or {}), set())
        return _fingerprint_function(target, set()) + extra, not isinstance(formula, CFunc)
    except (_Unfingerprintable, RecursionError):
        return f"{_safe_repr(formula)} @{id(formula)}", False


def _is_self_cached(formula) -> bool:
    """True if ``formula`` carries its own numba on-disk cache: a Dispatcher built with ``cache=True``.

    The ``_cache`` of such a Dispatcher is a ``FunctionCache``, not a ``NullCache``. That cache's key does not include
    the global values the dispatcher reads, so folding them into the outer kernel name cannot keep that cache fresh.
    A fresh compile of the outer kernel links the stale inner binary, and would serialize it. Such a unit must not be
    cached on disk. A ``cache=False`` Dispatcher recompiles in-process and is safe. A DUFunc's cache cannot be
    introspected independently, but its flag identity is folded into the digest, so the key of the outer kernel
    changes with it.
    """
    if isinstance(formula, Dispatcher):
        cache = getattr(formula, "_cache", None)
        return cache is not None and not isinstance(cache, NullCache)
    return False


def _pyfunc_of(formula):
    """The plain Python function underlying a formula, or None.

    It is the dispatcher's ``py_func``, a DUFunc's or CFunc's ``__wrapped__``, or a bare function itself.
    """
    func = getattr(formula, "py_func", None)
    if func is None and isinstance(formula, (DUFunc, CFunc)):
        func = getattr(formula, "__wrapped__", None)
    if func is None and isinstance(formula, FunctionType):
        func = formula
    return func if isinstance(func, FunctionType) else None


def _references_self_cached(value, seen: set[int]) -> bool:
    """True if ``value`` is itself, or transitively holds or references, a Dispatcher with its own numba cache.

    A formula can reach a ``@njit(cache=True)`` dispatcher through a call, a default argument, a container, a
    closure cell, or a global. The formula then links that dispatcher's binary, whose key does not include the
    globals the dispatcher reads. The outer digest changes with the dispatcher, but it cannot invalidate the
    callee's cache. The unit must therefore not be cached on disk either, or a stale inner binary is serialized into
    an artifact under a fresh digest. This walk mirrors the traversal of ``_canon_value``, so that it covers every
    channel the digest folds. A narrower walk would leave a gap through which a stale binary poisons the cache.
    """
    if id(value) in seen:
        return False
    seen.add(id(value))
    if _is_self_cached(value):
        return True
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_references_self_cached(v, seen) for v in value)
    if isinstance(value, dict):
        return any(_references_self_cached(v, seen) for v in value.values())
    func = _pyfunc_of(value)
    if func is None:
        return False
    for cell in func.__closure__ or ():
        try:
            contents = cell.cell_contents
        except ValueError:
            continue
        if _references_self_cached(contents, seen):
            return True
    for default in (func.__defaults__ or ()) + tuple((func.__kwdefaults__ or {}).values()):
        if _references_self_cached(default, seen):
            return True
    referenced = _referenced_global_names(func.__code__)
    for name in referenced:
        if name not in func.__globals__:
            continue
        gval = func.__globals__[name]
        if isinstance(gval, ModuleType):
            # A dispatcher reached as a module attribute, possibly chained, such as helpers.helper(x), is linked
            # exactly as a direct global dispatcher is.
            if _module_reaches_self_cached(gval, referenced, seen, set()):
                return True
        elif _references_self_cached(gval, seen):
            return True
    return False


def _module_reaches_self_cached(mod, ref_names, seen: set[int], mods_seen: set[int]) -> bool:
    """True if any referenced name in ``mod``'s ``__dict__`` reaches a cache-carrying Dispatcher.

    The search recurses into referenced submodules. It scans by ``__dict__`` membership, never by ``getattr``, so
    that a co-name collision cannot trigger an unrelated module's PEP 562 ``__getattr__``. A dispatcher that a module
    exposes *only* through PEP 562 ``__getattr__`` is a documented case that this search does not cover.
    """
    if id(mod) in mods_seen:
        return False
    mods_seen.add(id(mod))
    mod_dict = getattr(mod, "__dict__", {})
    for name in ref_names:
        if name not in mod_dict:
            continue
        val = mod_dict[name]
        if isinstance(val, ModuleType):
            if _module_reaches_self_cached(val, ref_names, seen, mods_seen):
                return True
        elif _references_self_cached(val, seen):
            return True
    return False


def _self_cached_name(formula) -> str:
    return getattr(getattr(formula, "py_func", formula), "__qualname__", None) or _safe_repr(formula)


# Build-time return-type validations are memoized by (formula fingerprint,
# input types, declared type) so repeated compile_kernel() calls do not re-probe.
_validated_returns: set = set()


def _is_typed(var: Variable) -> bool:
    return var.params is not None and var.params.type is not None


def _classify(compiled: CompiledGraph):
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
    # An empty `dispositions` is a fully undeclared graph of externals only, with zero interior nodes. Route it to
    # "C", so that a graph that declares nothing always behaves exactly as on the discovery path: no eager build,
    # and partition stays None until the first call.
    if not dispositions or "UNKNOWN" in vals or any(not _is_typed(e) for e in consumed):
        case = "C"
    elif "STATIC_PY" in vals:
        case = "B"
    else:
        case = "A"
    return case, dispositions, consumed


def _assemble_source(params: list[tuple[str, str, str]], lines: list[str], out_ids: list[str]) -> str:
    """Assemble the canonical kernel source.

    The fused kernel and every segment must share this exact shape, so that their cache digests cannot diverge.
    """
    sig = ", ".join(ident for _, _, ident in params)
    ret = f"    return ({', '.join(out_ids)},)"
    body = ("\n".join(lines) + "\n") if lines else ""
    return f"def _kernel({sig}):\n{body}{ret}\n"


def _emit_lines(
    nodes: list[CompiledNode], skip: set[Variable],
    idents: dict[Variable, str], bindings: dict[str, Any], flags: dict | None = None,
) -> list[str]:
    """Emit one body line per node, excluding `skip`, and fill `bindings`.

    Raises on a missing formula, a non-callable formula, or an arity mismatch.
    """
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
            bindings[fg] = _wrap_formula(var.formula, flags)
        except TypeError as e:
            raise TypeError(f"{var.qual_name()!r}: {e}") from e
        _check_formula_arity(var.formula, len(node.inputs), var.qual_name())
        arg_ids = ", ".join(idents[inp] for inp in node.inputs)
        in_names = ", ".join(repr(inp.qual_name()) for inp in node.inputs)
        lines.append(f"    {temp} = {fg}({arg_ids})  # {var.qual_name()!r} = f({in_names})")
    return lines


def _generate_segment_body(
    run_nodes: list[CompiledNode], live_in: tuple[Variable, ...],
    live_out: tuple[Variable, ...], idents: dict[Variable, str], flags: dict | None = None,
) -> tuple[str, dict, list, list]:
    """Generate the source of one jit segment, as _generate_body does for the fused kernel.

    The live-ins are the parameters, and the live-outs are the return tuple. The source has the same shape as the
    fused kernel's, so _compile applies to it unchanged.

    Returns (source, bindings, params, outputs), with params and outputs in the live_in and live_out order that the
    caller provides (sorted by qual name).
    """
    params = [(v.source, v.name, idents[v]) for v in live_in]
    bindings = {}
    lines = _emit_lines(run_nodes, set(), idents, bindings, flags)
    outputs = [v.qual_name() for v in live_out]
    out_ids = [idents[v] for v in live_out]
    source = _assemble_source(params, lines, out_ids)
    return source, bindings, params, outputs


def _generate_body(
    compiled: CompiledGraph, required: list[str], idents: dict[Variable, str], flags: dict | None = None,
) -> tuple[str, dict, list, list]:
    """Generate the `def _kernel(...): ...` source, without a decorator, and its bindings.

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
    lines = _emit_lines(compiled.ordered_nodes, external, idents, bindings, flags)

    by_qual = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    outputs, out_ids = [], []
    for q in required:
        var = by_qual.get(q)
        if var is None:
            raise ValueError(f"Requested variable {q!r} is not in the compiled graph")
        outputs.append(q)
        out_ids.append(idents[var])

    source = _assemble_source(params, lines, out_ids)
    return source, bindings, params, outputs


def _compile(
    source: str, bindings: dict[str, Any], jit_options: dict | None, cache: bool | None,
    declared_sigs: tuple = (),
) -> Dispatcher:
    """Compile the kernel source, content-addressed, into an @njit dispatcher.

    `declared_sigs` carries the unit's declared signature(s): the consumed external signature for an eager fused
    kernel, or each segment's live-in and live-out signature for an eager segment. They are folded into the digest
    through `repr`, so two declared-type variants of one type-free source get distinct anchors. `declared_sigs` is
    empty for undeclared (Case C) units.
    """
    fingerprints = []
    cacheable = True
    self_cached = []
    for fg, formula in bindings.items():
        fp, ok = _formula_fingerprint(formula)
        fingerprints.append(f"{fg}: {fp}")
        cacheable = cacheable and ok
        if _references_self_cached(formula, set()):
            cacheable = False
            self_cached.append(_self_cached_name(formula))
    if self_cached:
        warnings.warn(
            "compile_kernel: formula(s) " + ", ".join(sorted(set(self_cached)))
            + " are, or reference, a @njit(cache=True) dispatcher whose cache key is "
            "blind to referenced global values; compiling this kernel "
            "without an on-disk cache so a stale inner binary cannot be linked into a "
            "cached artifact. Pass cache=False to those dispatchers (or supply "
            "plain-Python formulas) to restore caching."
        )
    opts = {**_default_jit_options, **(jit_options or {})}
    if cache is not None:
        opts["cache"] = cache
    opts.setdefault("cache", True)
    flags = _effective_flags(jit_options)
    flags_canon, ok_flags = _flags_canon(flags)
    if not ok_flags:
        cacheable = False
        warnings.warn(
            f"compile_kernel: jit flags {sorted(flags)} have no canonical "
            "fingerprint; compiling this kernel without an on-disk cache."
        )
    # Each declared sig is a tuple of numba types. Fold them through the process-stable identity rather than the
    # raw repr: a Dispatcher's repr, ``type(CPUDispatcher(<function f at 0x...>))``, carries an ASLR address and
    # would leave one orphaned cache pair per run. An un-fingerprintable declared type, such as a Dispatcher that
    # wraps an @intrinsic-referencing body, has only a best-effort identity, so the kernel is compiled uncached.
    declared_ids = [[_type_identity(t) for t in s] for s in declared_sigs]
    declared_canon = [[h for h, _ok in sig] for sig in declared_ids]
    if not all(ok for sig in declared_ids for _h, ok in sig):
        cacheable = False
    hash_text = (
        "ck-digest-v3\n" + source
        + "\n# formulas:\n" + "\n".join(fingerprints)
        + "\n# flags: " + flags_canon
        # The NUMBA_* settings that change the emitted code (NUMBA_BOUNDSCHECK
        # overrides even an explicit jit flag at lowering) are in neither the jit
        # flags nor numba's own cache key; see numbox.core.cache_key.
        + "\n# settings: " + _settings_canon()
        + "\n# declared_sigs: " + repr(declared_canon)
    )
    if not cacheable:
        opts["cache"] = False
    digest = hashlib.sha256(hash_text.encode("utf-8")).hexdigest()[:16]
    name = f"_kernel_{digest}"
    final_src = "@njit(**_kernel_jit_options)\n" + source.replace(
        "def _kernel(", f"def {name}(", 1
    )
    anchor = _anchor_root(_ANCHOR_SUBDIR) / f"_kernel_{digest}.py"
    if opts["cache"]:
        # An uncached kernel still gets no anchor. The structref and the sqlite registrations write their anchors for
        # numba's messages in either case.
        opts = _anchored_or_uncached(anchor, final_src, opts)
    code = compile(final_src, str(anchor), "exec")
    # __name__ must name an importable module, so that numba can rebuild the cached overload's environment in
    # another process: importlib.import_module needs a real name, not None. make_graph and make_structref do the
    # same.
    ns = {**bindings, "njit": njit, "_kernel_jit_options": opts, "__name__": __name__}
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns.pop(name)


def _build_plan(
    runs: list, idents: dict, flags: dict, jit_options: dict | None, cache: bool | None,
    liveness: Callable, live_in_type: Callable, declared_sigs: bool = False, demoted: dict | None = None,
) -> tuple[tuple, tuple]:
    """Turn `runs` from `build_runs` into plan steps, and return (steps, segments).

    Each Python run becomes a chain of `_PyStep` steps, and each jit run becomes one compiled `_JitStep` segment.
    `liveness(run_nodes)` gives a jit run's (live_in, live_out), and `live_in_type(v)` gives the numba type that each
    live-in is compiled against. `declared_sigs` folds the segment's declared live-in and live-out types into the
    compile digest. If `demoted` maps each variable to its demotion reason, a `Segment` is recorded per run for the
    `PartitionReport`; without `demoted`, `segments` is empty.
    """
    steps, segments = [], []
    for kind, run_nodes in runs:
        quals = tuple(n.variable.qual_name() for n in run_nodes)
        if kind == "python":
            for n in run_nodes:
                steps.append(_PyStep(
                    var=n.variable,
                    py_callable=getattr(n.variable.formula, "py_func", n.variable.formula),
                    in_vars=tuple(n.inputs),
                ))
            if demoted is not None:
                produced = {n.variable for n in run_nodes}
                ins = sorted({i for n in run_nodes for i in n.inputs if i not in produced},
                             key=lambda v: v.qual_name())
                segments.append(Segment(
                    kind="python", nodes=quals,
                    inputs=tuple(v.qual_name() for v in ins),
                    outputs=quals, source=None,
                    reasons={n.variable.qual_name(): demoted[n.variable] for n in run_nodes},
                ))
            continue
        live_in, live_out = liveness(run_nodes)
        src, seg_bindings, _, _ = _generate_segment_body(run_nodes, live_in, live_out, idents, flags)
        seg_sigs = ((tuple(v.params.type for v in live_in), tuple(v.params.type for v in live_out))
                    if declared_sigs else ())
        disp = _compile(src, seg_bindings, jit_options, cache, seg_sigs)
        disp.compile(tuple(live_in_type(v) for v in live_in))
        steps.append(_JitStep(dispatcher=disp, in_vars=live_in, out_vars=live_out))
        if demoted is not None:
            segments.append(Segment(
                kind="jit", nodes=quals,
                inputs=tuple(v.qual_name() for v in live_in),
                outputs=tuple(v.qual_name() for v in live_out),
                source=src, reasons={},
            ))
    return tuple(steps), tuple(segments)


class CompiledKernel:
    """A fused @njit kernel compiled from a Variable graph.

    Attributes::

      kernel      - hot-path callable: the resolver before the first call, the
                    bare numba dispatcher once fused, and the segmented master
                    otherwise. It takes the external args positionally, in
                    `params` order, and returns a tuple in `outputs` order.
      recompute   - value-only incremental refresh of only the cone that a
                    change affects, over a store seeded by a prior `kernel`
                    call; returns a tuple in `outputs` order (see `recompute`).
      params      - external input qual_names, in kernel-argument order.
      outputs     - requested variable qual_names, in return-tuple order.
      source      - generated kernel source text.
      identifiers - {qual_name: temp identifier} for inspection.
      partition   - PartitionReport describing which nodes run in jit
                    segments and which run in Python. None until the first
                    call for undeclared graphs; set at build time for fully
                    declared (eager) graphs.
      is_declared - True when the graph was fully declared and the mode was
                    resolved eagerly at build; False for a discovery
                    (undeclared) kernel. Declared kernels enforce the
                    `recompute()` type contract instead of re-discovering.
    """

    def __init__(self, kernel: Dispatcher, params: list[tuple[str, str, str]],
                 outputs: list[str], source: str, identifiers: dict[str, str],
                 ctx: _KernelCtx, required_vars: list[Variable],
                 external_vars: list[Variable], is_declared: bool = False) -> None:
        self._fused = kernel
        self.is_declared = is_declared
        self._mode = "virgin"
        self._plan = None
        self.partition = None
        self._ctx = ctx
        self._param_keys = [(src, name) for src, name, _ in params]
        self.params = [make_qual_name(src, name) for src, name, _ in params]
        self.outputs = list(outputs)
        self.source = source
        self.identifiers = identifiers
        self._required_vars = required_vars
        self._external_vars = external_vars
        self._store = None          # {Variable: value}; seeded on first recompute
        self._demoted = {}          # {Variable: reason}; frozen demotion verdicts
        self._last_args = None      # external args of the most recent full resolution
        self._sources = None        # change-source Variables (externals; may grow on interior override)
        self._boundary = None       # set[Variable]; persisted-node set, computed lazily
        self._cone_cache = OrderedDict()  # LRU cache of cone sub-plans, keyed on cone+live-in
        self._cone_cap = 64         # max distinct cone plans retained before LRU eviction

    @property
    def kernel(self) -> Callable:
        if self._mode == "fused":
            # Fused mode is permanent: later signatures go through numba's own dispatch, and a typing failure there
            # raises. There is no segmentation fallback from fused mode.
            return self._fused
        if self._mode == "fused-pending":
            return self._fused_pending_call
        if self._mode == "segmented":
            return self._run_segmented
        return self._resolve_and_call

    def _fused_pending_call(self, *args) -> tuple:
        # A caller may hold a reference to this bound method across calls. Once the mode has changed to "fused", call
        # the dispatcher directly, without setting _last_args again: the first call already captured the seeding args.
        if self._mode == "fused":
            return self._fused(*args)
        result = self._fused(*args)
        self._last_args = args
        self._mode = "fused"
        return result

    def _fused_report(self) -> PartitionReport:
        compiled, external = self._ctx.compiled, self._ctx.external
        nodes = tuple(
            n.variable.qual_name() for n in compiled.ordered_nodes
            if n.variable not in external
        )
        return PartitionReport(mode="fused", segments=(Segment(
            kind="jit", nodes=nodes, inputs=tuple(self.params),
            outputs=tuple(self.outputs), source=self.source, reasons={},
        ),))

    def _resolve_and_call(self, *args) -> tuple:
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
                result = self._discover_and_run(args)
                self._last_args = args
                return result
            self._mode = "fused"
            self.partition = self._fused_report()
            result = self._fused(*args)
            self._last_args = args
            return result
        result = self._discover_and_run(args)
        self._last_args = args
        return result

    def _run_segmented(self, *args) -> tuple:
        try:
            result = self._plan.run(args)
        except NumbaError:
            # The except clause is deliberately broad: a segment that fails to compile for new input types triggers
            # re-discovery. A NumbaError raised inside a python-step formula is raised again when re-discovery
            # evaluates the same args in Python, so no error is masked. Declared kernels do not re-discover, because
            # that would overwrite the authoritative _demoted; an off-contract input raises again instead.
            if self.is_declared:
                raise
            result = self._discover_and_run(args)
        if self._last_args is None:
            self._last_args = args
        return result

    def _discover_and_run(self, args: tuple) -> tuple:
        compiled, idents, bindings_by_var, jit_options, cache, external = self._ctx
        flags = _effective_flags(jit_options)
        values = dict(zip(self._external_vars, args))
        demoted = discover(compiled.ordered_nodes, external, values, bindings_by_var, flags)
        nodes = [n for n in compiled.ordered_nodes if n.variable not in external]
        order = linearize(nodes, set(demoted))
        runs = build_runs(order, set(demoted))
        steps, segments = _build_plan(
            runs, idents, flags, jit_options, cache,
            liveness=lambda run_nodes: segment_liveness(run_nodes, external, self._required_vars, order),
            live_in_type=lambda v: typeof(values[v]),
            demoted=demoted,
        )
        self._plan = _Plan(
            steps=steps,
            external_vars=tuple(self._external_vars),
            output_vars=tuple(self._required_vars),
        )
        self._store = values              # seed store from the already-computed values
        self._demoted = demoted           # freeze demotion verdicts for cone builds
        self._mode = "segmented"
        self.partition = PartitionReport(mode="segmented", segments=segments)
        return tuple(values[v] for v in self._required_vars)

    def _ensure_store(self):
        if self._store is not None:
            return
        if self._mode == "virgin" or self._last_args is None:
            raise RuntimeError(
                "CompiledKernel.recompute requires a prior full call: call the kernel "
                "once with the current inputs to seed the value store before recompute()."
            )
        compiled, _, bindings_by_var, jit_options, cache, external = self._ctx
        flags = _effective_flags(jit_options)
        # A fused first call discarded the interior values, so seed the store by evaluating the whole graph once from
        # the captured args. Fused-graph formulas are njit-pure, so the one extra evaluation has no observable effect.
        # Seeding only the persisted-node closure instead would be a possible optimization.
        values = dict(zip(self._external_vars, self._last_args))
        if self.is_declared:
            # Seed interior values with the frozen declared demotion set; re-running
            # discover would re-probe and could disagree with the declarations.
            _evaluate(compiled.ordered_nodes, external, values, bindings_by_var, flags, self._demoted)
        else:
            self._demoted = discover(compiled.ordered_nodes, external, values, bindings_by_var, flags)
        self._store = values

    def _apply_changes(self, changed: dict) -> set:
        """Write changed values into the store, and return the set of changed Variables.

        External names resolve through required_external_variables; interior names resolve through ordered_nodes.
        Overriding an interior node expands the change-source set.
        """
        compiled = self._ctx.compiled
        # First pass: resolve every (src, name) to its variable, run the declared contract check, and collect
        # (var, val, is_external). If any check fails, raise here, before _store is mutated, so that a ValueError
        # that is caught and retried cannot leave the store partially written.
        resolved = []
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
                if self.is_declared and _is_typed(var):
                    try:
                        got = typeof(val)
                        ok = self._fused.typingctx.can_convert(got, var.params.type) is not None
                    except (ValueError, TypeError):
                        ok, got = False, type(val).__name__
                    if not ok:
                        raise ValueError(
                            f"declared type {var.params.type}, got {got} for {var.qual_name()}")
                resolved.append((var, val, is_external))
        # Second pass: write the values, collect changed_vars, and record the interior overrides.
        changed_vars = set()
        new_interior_sources = set()
        for var, val, is_external in resolved:
            self._store[var] = val
            changed_vars.add(var)
            if not is_external and (self._sources is None or var not in self._sources):
                new_interior_sources.add(var)
        if new_interior_sources:
            self._expand_sources(new_interior_sources)
        return changed_vars

    def _expand_sources(self, new_sources: set):
        """Add overridden interior nodes to the change-source set.

        The method also forces the persisted-node boundary to recompute, and invalidates the cached cone plans,
        because their boundaries have changed.
        """
        if self._sources is None:
            self._sources = set(self._external_vars)
        self._sources |= new_sources
        self._boundary = None
        self._cone_cache.clear()

    def _ensure_boundary(self):
        if self._boundary is not None:
            return
        compiled = self._ctx.compiled
        if self._sources is None:
            self._sources = set(self._external_vars)
        self._boundary = compute_boundary(
            compiled.ordered_nodes, self._sources, set(self._required_vars)
        )

    def _cone_live_in_type(self, v: Variable):
        """The numba type that a cone jit-segment live-in is compiled against.

        A declared kernel uses the declared type of a live-in that declares one, so that the cone matches the eager
        build's signature. Otherwise the type falls back to the stored value's runtime type, as the undeclared path
        always does.
        """
        if self.is_declared and _is_typed(v):
            return v.params.type
        return typeof(self._store[v])

    def _build_cone_plan(self, affected: list) -> _ConePlan:
        """Linearize the affected cone, split it into runs, and compile each jit run against its live-in types.

        The live-in types are the declared types for a declared kernel, and otherwise the store's current runtime
        types. Python runs become _PyStep chains. `_demoted` is restricted to the cone nodes, so only nodes demoted at
        seed time stay Python. `_cone_plan_cached` calls this method on a cache miss.
        """
        compiled, idents, _, jit_options, cache, _ = self._ctx
        flags = _effective_flags(jit_options)
        cone_vars = {n.variable for n in affected}
        demoted_in_cone = {v for v in self._demoted if v in cone_vars}
        order = linearize(affected, demoted_in_cone)
        runs = build_runs(order, demoted_in_cone)
        steps, _ = _build_plan(
            runs, idents, flags, jit_options, cache,
            liveness=lambda run_nodes: cone_liveness(run_nodes, order, self._required_vars, self._boundary),
            live_in_type=self._cone_live_in_type,
        )
        return _ConePlan(steps=steps)

    def _cone_key(self, affected) -> tuple[frozenset, frozenset]:
        """Cache key for a cone sub-plan: (cone node qual_names, live-in boundary qual_names).

        Including the live-in boundary prevents an external change and an interior override that share a cone from
        colliding, because their boundaries differ.
        """
        cone_vars = {n.variable for n in affected}
        live_in_boundary = {inp for n in affected for inp in n.inputs if inp not in cone_vars}
        return (frozenset(v.qual_name() for v in cone_vars),
                frozenset(v.qual_name() for v in live_in_boundary))

    def _cone_plan_cached(self, affected) -> _ConePlan:
        key = self._cone_key(affected)
        plan = self._cone_cache.get(key)
        if plan is not None:
            self._cone_cache.move_to_end(key)
            return plan
        plan = self._build_cone_plan(affected)
        self._cone_cache[key] = plan
        if len(self._cone_cache) > self._cone_cap:
            self._cone_cache.popitem(last=False)
        return plan

    def _flush_and_reseed(self):
        """Drop the cone-plan cache, and re-seed the store and the boundary from the last full call.

        The method recovers when a type change of a boundary live-in makes a cached cone dispatcher fail to compile.
        """
        self._cone_cache.clear()
        self._store = None
        self._boundary = None
        self._ensure_store()
        self._ensure_boundary()

    def recompute(self, changed: dict) -> tuple:
        """Incrementally re-evaluate only the cone that `changed` affects.

        The method mirrors :meth:`numbox.core.variable.variable.CompiledGraph.recompute`: it is a value-only refresh,
        not a recompile. `changed` is ``{source: {name: value}}``, and the returned tuple is in `outputs` order. The
        same variables may carry different values across calls, but their numba types must stay the same as in the
        seeding call. A type change is recovered from once, as described below, but the contract is that the types
        stay the same.

        Precondition: a prior full call (``kernel(...)`` or ``execute(...)``) must have seeded the value store.
        Calling `recompute` first raises ``RuntimeError``.

        The method writes the changed values into a persistent value store, collects the downstream cone of the
        changed nodes, and re-fuses only that cone, reading every unchanged input from the store. The cone sub-plan
        is compiled on first use and kept in a bounded LRU cache, keyed on the cone and its live-in boundary, so a
        recurring change pattern reuses its compiled plan without re-fusing. Nodes that were demoted to plain Python
        at seed time stay Python in the cone, and the jittable remainder fuses into ``@njit`` segments.

        Interior overrides: a `changed` name may resolve to an interior (computed) node rather than an external
        input, as on the interpreted path. Its value is then overridden in the store, and only its downstream cone
        recomputes. The overridden node's own formula is *not* re-run, unless the node is itself downstream of
        another changed input in the same call. In that case graph priority applies, and the node is recomputed
        from its formula. The first override of an interior node not seen before expands the change-source set and
        rebuilds the persisted-node boundary. It also invalidates the cached cone plans, whose boundaries have
        changed.

        Limitations:

        - Do not interleave input-changing ``kernel(...)`` throughput calls between `recompute` calls. `recompute`
          is the stateful entry point: the store is seeded once, and a throughput call does not update it, so a
          later `recompute` would read stale values for the unchanged inputs. Use `recompute` for the incremental
          workflow, and the bare kernel for independent one-shot calls.
        - An interior plain-Python (demoted) node must return a stable numba type across recomputes. The same-types
          contract extends to demoted outputs: a demoted node whose output type changes between recomputes is not
          supported.

        If the type of a live-in changes, a cached cone dispatcher fails to compile. The whole cone-plan cache is
        then flushed once, the store is re-seeded from the last full call, the change is applied again, and the cone
        is rebuilt against the new types.

        Declared kernels enforce a contract instead of recovering. A kernel built from a fully declared graph checks
        each changed value for numba assignability to the node's declared type. A value that numba cannot
        assign to the declared type raises a ``declared type X, got Y`` error. A benign difference that numba
        accepts, such as a C-contiguous array against an ``'A'``-layout array declaration or a safe scalar
        promotion, is accepted. The check is convertibility, not type identity, and it applies only to declared
        (eager) kernels. An individually declared node inside an otherwise discovered kernel keeps the one-time
        flush and reseed described above.
        """
        self._ensure_store()
        changed_vars = self._apply_changes(changed)
        if not changed_vars:
            return tuple(self._store[v] for v in self._required_vars)
        compiled = self._ctx.compiled
        self._ensure_boundary()
        affected = compiled._collect_affected(changed_vars)
        if not affected:
            return tuple(self._store[v] for v in self._required_vars)
        plan = self._cone_plan_cached(affected)
        try:
            plan.run_into(self._store)
        except NumbaError:
            # A live-in type change invalidates every compiled cone plan. Flush the whole cache once, reseed the
            # store, apply the change again against the fresh store, and rebuild against the new types. Declared
            # kernels fix every type by contract, so the declared contract check in _apply_changes has already
            # rejected off-contract types with a clear error; flushing here would serve no purpose.
            if self.is_declared:
                raise
            self._flush_and_reseed()
            self._apply_changes(changed)
            affected = compiled._collect_affected(changed_vars)
            plan = self._cone_plan_cached(affected)
            plan.run_into(self._store)
        return tuple(self._store[v] for v in self._required_vars)

    def execute(self, external_values: dict) -> dict:
        """Take external values as a dict and return the outputs as a dict.

        This convenience wrapper is symmetric with CompiledGraph.execute.
        """
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


def compile_kernel(
    graph: Graph, required: str | list[str], *,
    jit_options: dict | None = None, cache: bool | None = None,
) -> CompiledKernel:
    """Compile `graph` into a fused @njit kernel for the `required` variables.

    :param graph: The graph whose dependency structure and formulas are fused into one straight-line @njit function
        (see `CompiledKernel`).
    :param required: The requested variables. Their order is preserved and fixes the order of
        `CompiledKernel.outputs` and of the kernel's return tuple. A duplicate entry raises `ValueError`: each output
        is requested once, because the return tuple is positional and a repeat carries no information.
    :param jit_options: Options merged over numbox's defaults and passed to @njit. The defaults are the
        `NUMBOX_JIT_OPTIONS` environment value, with `cache` off if numba can write no cache for numbox's own files
        (see `numbox.core.configurations`). All options except `cache` participate in the content-addressed digest.
    :param cache: A tri-state flag. `None` (the default) defers to `jit_options["cache"]`, and then to numbox's
        default: the `NUMBOX_JIT_OPTIONS` environment value, or `True` unless numba can write no cache for numbox's
        own files. In that case the default is `False`, even though the kernel's own anchor might be cacheable. An
        explicit `True` or `False` takes precedence over both, and the anchor's own check still turns caching off if
        that anchor cannot be cached.

    Error timing: structural problems raise here. These are unknown or malformed `required` entries, non-callable
    formulas, arity mismatches against the declared inputs, and graphs deeper than the recursion limit. For an
    undeclared or partially declared graph, numba typing problems are reported at the kernel's first call, because
    the automatic njit wrapping of plain-Python formulas is lazy. A graph is fully declared when every node carries
    `params` and every consumed external is typed. The mode of such a graph resolves eagerly here, so its type errors
    move to build time. These errors are a formula whose natural return at the declared input types does not convert
    to the declared type, a cross-node type mismatch, and, in particular, a coercible but wrong `params.type`. An
    explicit unconstrained return-type probe catches the last of these: it compares the formula's naturally inferred
    return type against the declaration. Binding the formula to the declared signature does not catch it, because
    numba silently coerces a convertible scalar. Without the probe, declaring int64 over a `x * 1.5` body would
    compute 7, not 7.5. A fully declared graph thus fails fast at build, and a graph with any undeclared node fails at
    the first call, exactly as on the discovery path. Runtime errors never demote a node; they propagate.

    Caching: the kernel digest fingerprints each formula's bytecode, constants, default values, closure-cell values,
    referenced module-level globals (including helper functions, recursively), referenced module-attribute values
    and defining module. It also holds the effective jit flags, the targetoptions of any Dispatcher or DUFunc
    formula, and the ``NUMBA_*`` settings that change the emitted code (see :mod:`numbox.core.cache_key`). The
    generated source never mentions types, so a declared graph's signatures are appended to the digest too: the
    consumed external signature for an eager fused kernel, and each segment's live-in and live-out signature for an
    eager segment. Two declared-type variants of one type-free graph therefore get distinct cache anchors, and never
    reuse each other's binary.

    Two kinds of formula downgrade their kernel to cache=False, so that it is always recompiled and never stale. One
    is a formula whose state cannot be fingerprinted, such as a cres/CompileResultWAP object or a value with no
    canonical form. The other carries or reaches its own numba cache: a `@njit(cache=True)` Dispatcher, whose key
    does not include the globals it reads and cannot be invalidated from outside. The downgrades for a self-cached
    dispatcher and for an un-canonicalizable flag emit a warning. An inherently unfingerprintable formula kind (CFunc,
    cres) is downgraded silently, as documented. See the module docstring for the scope of the guarantee. When
    caching is enabled, a content-addressed anchor `.py` file is written under numba's cache directory. Nothing is
    written with caching off, or with an unwritable cache directory, which warns and degrades.

    Non-jittable formulas: for an undeclared graph, the first call resolves the execution mode. If the fully fused
    kernel cannot be typed for the actual argument types, each node is probed against the real intermediate values.
    Nodes whose formulas fail to *compile*, or whose input values numba cannot type, run in plain Python, and the
    jittable remainder is fused into segments orchestrated from Python. A declared `jitable=False` node is instead
    demoted by declaration, with no probing. A graph that mixes declared jittable and declared-Python nodes resolves
    eagerly to a "segmented" plan at build, and `CompiledKernel.partition` is populated immediately with per-node
    reasons.

    `CompiledKernel.partition` describes the result, including per-node demotion reasons. It is `None` before the
    first call only for an undeclared graph, and is set at build for a fully declared one. `CompiledKernel.kernel` is
    the hot-path callable: the bare @njit dispatcher once the graph resolves fully fused, and the Python master when
    it is segmented. For an undeclared graph, a later call whose types do not fit a segment learns the partition
    again and replaces it, so one plan is active at a time. A declared kernel does not re-discover.

    The `declared type X, got Y` contract is enforced at build time and on `recompute()`, by `can_convert` against
    each declared type. It is not enforced on the throughput `kernel(...)` path, which retains numba's polymorphic
    widening across calls. A later `kernel(...)` call whose off-contract type does not fit a jit segment raises
    numba's own typing error, and the declared `_demoted` stays unchanged, so no silent re-discovery happens. Once
    the kernel is fully fused, it stays fused: a typing failure for a later signature raises. The discovery call
    computes jit-node values through per-node dispatchers, while later calls use fused segments. The results are
    identical under default IEEE semantics, but non-default jit_options such as fastmath could in principle make
    them differ across fusion boundaries.
    """
    required = [required] if isinstance(required, str) else list(required)
    for entry in required:
        if not isinstance(entry, str):
            raise TypeError(f"required entries must be qualified-name strings; got {entry!r}")
        if QUAL_SEP not in entry:
            raise ValueError(
                f"required entry {entry!r} is not qualified (expected 'source{QUAL_SEP}name')"
            )
    dupes = sorted({entry for entry in required if required.count(entry) > 1})
    if dupes:
        raise ValueError(
            f"required has duplicate entries {dupes}; each requested output must "
            f"appear once (the kernel's return tuple is positional)"
        )
    try:
        compiled = graph.compile(required)
    except KeyError as e:
        raise ValueError(
            f"required name or one of its dependencies cannot be resolved in the graph: {e}"
        ) from e
    except RecursionError:
        raise RecursionError("Consider raising recursion limit.") from None
    external = {v for vs in compiled.required_external_variables.values() for v in vs.values()}
    case, dispositions, consumed = _classify(compiled)
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    flags = _effective_flags(jit_options)
    flags_key, _ = _flags_canon(flags)
    if case in ("A", "B"):
        for node in compiled.ordered_nodes:
            var = node.variable
            if var in external or dispositions.get(var) != "STATIC_JIT":
                continue
            in_types = tuple(i.params.type for i in node.inputs)
            fp, fingerprintable = _formula_fingerprint(var.formula)
            key = (fp, in_types, var.params.type, flags_key)
            if fingerprintable and key in _validated_returns:
                continue
            _validate_declared_return(var.formula, in_types, var.params.type, flags)
            if fingerprintable:
                _validated_returns.add(key)
    source, bindings, params, outputs = _generate_body(compiled, required, idents, flags)

    def _registry_var(qual):
        src, name = qual.rsplit(QUAL_SEP, 1)
        return graph.registry[src][name]

    required_vars = [_registry_var(q) for q in outputs]
    external_vars = [graph.registry[src][name] for src, name, _ in params]
    # Case A folds the consumed-external declared signature into the digest, in kernel-arg order to match the eager
    # `.compile` below, so that int64 and float64 variants of one type-free source get distinct anchors. Cases B and
    # C pass ().
    declared_sigs = ()
    if case == "A":
        consumed_sig = tuple(v.params.type for v in external_vars if v in consumed)
        declared_sigs = (consumed_sig,)
    kernel = _compile(source, bindings, jit_options, cache, declared_sigs)
    bindings_by_var = {
        n.variable: bindings["f_" + idents[n.variable]]
        for n in compiled.ordered_nodes if n.variable not in external
    }
    ctx = _KernelCtx(compiled, idents, bindings_by_var, jit_options, cache, external)
    identifiers = {v.qual_name(): ident for v, ident in idents.items()}
    if case == "A":
        ck = CompiledKernel(kernel, params, outputs, source, identifiers, ctx,
                            required_vars, external_vars, is_declared=True)
        sig_vars = [v for v in external_vars if v in consumed]
        # Compile eagerly only when every kernel param is consumed. A pass-through external output is a kernel arg
        # but has no declared type, so a partial signature would have the wrong arity. Such graphs compile lazily on
        # the first call.
        if len(sig_vars) == len(external_vars):
            ck._fused.compile(tuple(v.params.type for v in sig_vars))
        ck._mode = "fused-pending"
        ck.partition = ck._fused_report()
        return ck
    if case == "B":
        demoted = {n.variable: "declared non-jittable" for n in compiled.ordered_nodes
                   if dispositions.get(n.variable) == "STATIC_PY"}
        nodes = [n for n in compiled.ordered_nodes if n.variable not in external]
        order = linearize(nodes, demoted)
        runs = build_runs(order, demoted)
        # The segment structure is the same as in _discover_and_run, but with declared types and a static demotion
        # set (no probing).
        steps, segments = _build_plan(
            runs, idents, flags, jit_options, cache,
            liveness=lambda run_nodes: segment_liveness(run_nodes, external, required_vars, order),
            live_in_type=lambda v: v.params.type,
            declared_sigs=True,
            demoted=demoted,
        )
        ck = CompiledKernel(kernel, params, outputs, source, identifiers, ctx,
                            required_vars, external_vars, is_declared=True)
        ck._plan = _Plan(steps=steps, external_vars=tuple(external_vars),
                         output_vars=tuple(required_vars))
        ck._demoted = demoted
        ck._mode = "segmented"
        ck.partition = PartitionReport(mode="segmented", segments=segments)
        return ck
    return CompiledKernel(
        kernel, params, outputs, source, identifiers, ctx, required_vars, external_vars
    )
