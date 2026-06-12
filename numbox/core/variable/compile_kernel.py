"""Compile a `core.variable` Variable graph into one fused @njit kernel.

Alongside `core.work` (a structref graph), this turns a `Graph`/`CompiledGraph`
into a single straight-line @njit function whose interior nodes are SSA
temporaries. No per-node type info is needed: numba infers every interior type
from the kernel's runtime argument types, provided each formula is njit-able
(plain-Python formulas are auto-wrapped with njit()).

The on-disk cache is content-addressed: the digest fingerprints each formula's
code, constants, default arguments, closure-cell values, referenced globals,
and the kernel's effective jit flags, so a stale binary is never reused and two
distinct kernels never collide. A formula with no canonical fingerprint forces
the kernel uncached (no anchor, no numba cache) -- never reused, never wrong.
"""
import hashlib
import keyword
import re
import warnings

from types import CodeType, FunctionType, ModuleType

import numpy as np

from numba import njit
from numba.core.dispatcher import Dispatcher
from numba.core.types.function_type import CompileResultWAP

from numbox.core.configurations import jit_options as _default_jit_options
from numbox.core.variable.variable import make_qual_name
from numbox.utils.preprocessing import (
    _anchor_root, _materialize_anchor, _orphan_anchor_sweep,
)

# Names injected into the kernel exec namespace; identifiers must avoid them.
_RESERVED = frozenset({"njit", "_kernel_jit_options"})

_ANCHOR_SUBDIR = "numbox-compile-kernel"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)


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
        while cand in used or ("f_" + cand) in used or keyword.iskeyword(cand):
            i += 1
            if i > len(digest):
                raise RuntimeError(
                    f"Cannot assign a unique identifier for {var.qual_name()!r}; "
                    f"all sha256 prefixes exhausted"
                )
            cand = f"{base}_{digest[:i]}"
        used.add(cand)
        used.add("f_" + cand)
        idents[var] = cand
    return idents


def _wrap_formula(formula):
    """Return an njit-callable for `formula`; non-Dispatcher/CompileResultWAP callables are njit-wrapped."""
    if isinstance(formula, (Dispatcher, CompileResultWAP)):
        return formula
    return njit(formula)


class _Unfingerprintable(Exception):
    """A value the cache digest cannot canonicalize; the kernel goes uncached."""


def _canon_value(value, seen):
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return repr(value)
    if isinstance(value, np.ndarray):
        data = np.ascontiguousarray(value)
        raw = hashlib.sha256(data.tobytes()).hexdigest()
        return f"ndarray({data.dtype.str};{value.shape};{raw})"
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__}[" + ",".join(_canon_value(v, seen) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}[" + ",".join(sorted(_canon_value(v, seen) for v in value)) + "]"
    if isinstance(value, dict):
        items = sorted((_canon_value(k, seen), _canon_value(v, seen)) for k, v in value.items())
        return "dict[" + ",".join(f"{k}:{v}" for k, v in items) + "]"
    if isinstance(value, ModuleType):
        return f"module({value.__name__})"
    if isinstance(value, Dispatcher):
        topts = _canon_value(dict(getattr(value, "targetoptions", {}) or {}), seen)
        return f"dispatcher({_fingerprint_function(value.py_func, seen)};{topts})"
    if isinstance(value, FunctionType):
        return f"function({_fingerprint_function(value, seen)})"
    raise _Unfingerprintable(type(value).__name__)


def _fingerprint_codeobj(code, seen):
    consts = ",".join(
        _fingerprint_codeobj(c, seen) if isinstance(c, CodeType) else _canon_value(c, seen)
        for c in code.co_consts
    )
    return (
        f"code({code.co_code.hex()};flags={code.co_flags};argc={code.co_argcount};"
        f"kwonly={code.co_kwonlyargcount};names={','.join(code.co_names)};consts=[{consts}])"
    )


def _referenced_global_names(code):
    names = set(code.co_names)
    for c in code.co_consts:
        if isinstance(c, CodeType):
            names |= _referenced_global_names(c)
    return names


def _fingerprint_function(func, seen):
    if id(func) in seen:
        return f"recursive({func.__qualname__})"
    seen = seen | {id(func)}
    code = func.__code__
    cells = []
    for name, cell in zip(code.co_freevars, func.__closure__ or ()):
        try:
            contents = cell.cell_contents
        except ValueError as e:
            raise _Unfingerprintable("empty closure cell") from e
        cells.append(f"{name}={_canon_value(contents, seen)}")
    hashed_globals = []
    for name in sorted(_referenced_global_names(code)):
        if name in func.__globals__:
            hashed_globals.append(f"{name}={_canon_value(func.__globals__[name], seen)}")
    return (
        f"func({func.__module__}:{func.__qualname__};{_fingerprint_codeobj(code, seen)};"
        f"defaults={_canon_value(func.__defaults__ or (), seen)};"
        f"kwdefaults={_canon_value(func.__kwdefaults__ or {}, seen)};"
        f"closure=[{';'.join(cells)}];globals=[{';'.join(hashed_globals)}])"
    )


def _formula_fingerprint(formula):
    """Behavioral identity of a formula for the cache digest.

    Returns ``(text, cacheable)``. The text covers every value channel
    numba freezes into a compiled artifact: code-object bytecode/consts/
    names, default-argument values, closure-cell values, the values of
    referenced module-level globals (recursing into helper functions and
    dispatchers, with cycle protection), the defining module, and
    dispatcher targetoptions. Builtins resolve outside ``__globals__``
    and are deliberately not hashed. Any value with no canonical form
    makes the formula un-fingerprintable: the returned text is then a
    per-object placeholder and ``cacheable`` is False, so the kernel is
    compiled without an on-disk cache -- never reused, never wrong.
    """
    target = getattr(formula, "py_func", None)
    if target is None:
        target = formula
    if not isinstance(target, FunctionType):
        return f"{repr(formula)} @{id(formula)}", False
    extra = ""
    if isinstance(formula, Dispatcher):
        extra = ";targetoptions=" + _canon_value(dict(formula.targetoptions or {}), set())
    try:
        return _fingerprint_function(target, set()) + extra, True
    except (_Unfingerprintable, RecursionError):
        return f"{repr(formula)} @{id(formula)}", False


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
        in_names = ", ".join(repr(inp.qual_name()) for inp in node.inputs)
        lines.append(f"    {temp} = {fg}({arg_ids})  # {var.qual_name()!r} = f({in_names})")

    by_qual = {n.variable.qual_name(): n.variable for n in compiled.ordered_nodes}
    outputs, out_ids = [], []
    for q in required:
        var = by_qual.get(q)
        if var is None:
            raise ValueError(f"Requested variable {q!r} is not in the compiled graph")
        outputs.append(q)
        out_ids.append(idents[var])

    sig = ", ".join(ident for _, _, ident in params)
    ret = f"    return ({', '.join(out_ids)},)"
    body = ("\n".join(lines) + "\n") if lines else ""
    source = f"def _kernel({sig}):\n{body}{ret}\n"
    return source, bindings, params, outputs


def _compile(source, bindings, jit_options, cache):
    """Content-addressed compile of the kernel source into an @njit dispatcher."""
    fingerprints = []
    cacheable = True
    for fg, formula in bindings.items():
        fp, ok = _formula_fingerprint(formula)
        fingerprints.append(f"{fg}: {fp}")
        cacheable = cacheable and ok
    opts = {**_default_jit_options, **(jit_options or {})}
    if cache is not None:
        opts["cache"] = cache
    opts.setdefault("cache", True)
    flags = {k: v for k, v in opts.items() if k != "cache"}
    try:
        flags_canon = _canon_value(flags, set())
    except _Unfingerprintable:
        flags_canon = repr(sorted(flags.items(), key=repr))
        cacheable = False
    hash_text = (
        "ck-digest-v2\n" + source
        + "\n# formulas:\n" + "\n".join(fingerprints)
        + "\n# flags: " + flags_canon
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
        try:
            anchor.parent.mkdir(parents=True, exist_ok=True)
            _materialize_anchor(anchor, final_src)
        except OSError as e:
            warnings.warn(
                f"compile_kernel: cache directory unusable ({e}); "
                f"compiling without an on-disk cache"
            )
            opts["cache"] = False
    code = compile(final_src, str(anchor), "exec")
    # __name__ must be an importable module so numba can rebuild the cached
    # overload's environment in another process (importlib.import_module needs
    # a real name, not None); mirrors make_graph / make_structref.
    ns = {**bindings, "njit": njit, "_kernel_jit_options": opts, "__name__": __name__}
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns.pop(name)


class CompiledKernel:
    """A fused @njit kernel compiled from a Variable graph.

    Attributes::

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


def compile_kernel(graph, required, *, jit_options=None, cache=None):
    """Compile `graph` into a fused @njit kernel for the `required` variables."""
    required = [required] if isinstance(required, str) else list(required)
    required = list(dict.fromkeys(required))  # dedupe, preserve first-seen order
    compiled = graph.compile(required)
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    source, bindings, params, outputs = _generate_body(compiled, required, idents)
    kernel = _compile(source, bindings, jit_options, cache)
    identifiers = {v.qual_name(): ident for v, ident in idents.items()}
    return CompiledKernel(kernel, params, outputs, source, identifiers)
