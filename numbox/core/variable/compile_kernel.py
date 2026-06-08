"""Compile a `core.variable` Variable graph into one fused @njit kernel.

Alongside `core.work` (a structref graph), this turns a `Graph`/`CompiledGraph`
into a single straight-line @njit function whose interior nodes are SSA
temporaries. No per-node type info is needed: numba infers every interior type
from the kernel's runtime argument types, provided each formula is njit-able
(plain-Python formulas are auto-wrapped with njit()).
"""
import hashlib
import re

from inspect import getsource
from numba import njit
from numba.core.dispatcher import Dispatcher
from numba.core.types.function_type import CompileResultWAP

from numbox.core.configurations import jit_options as _default_jit_options
from numbox.utils.preprocessing import (
    _anchor_path, _materialize_anchor, _orphan_anchor_sweep,
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
        while cand in used or ("f_" + cand) in used:
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


def _safe_getsource(formula):
    """Source text of a formula, for the content-addressed cache hash.

    Content-sensitive on purpose: two formulas with the same signature but
    different bodies must hash differently, so we never substitute the
    signature here. When no source is recoverable (e.g. a cres/CompileResultWAP
    formula, or a lambda defined outside a source file), we fall back to
    ``repr(formula)``. repr is unique per object, so it never causes a hash
    *collision* (results stay correct); it is just not stable across processes,
    so such a formula does not get cross-process cache reuse.
    """
    target = getattr(formula, "py_func", formula)
    try:
        return getsource(target)
    except (OSError, TypeError):
        return repr(formula)


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
        in_names = ", ".join(inp.qual_name() for inp in node.inputs)
        lines.append(f"    {temp} = {fg}({arg_ids})  # {var.qual_name()} = f({in_names})")

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
    formula_src = "\n".join(_safe_getsource(f) for f in bindings.values())
    hash_text = source + "\n# formulas:\n" + formula_src
    digest = hashlib.sha256(hash_text.encode("utf-8")).hexdigest()[:16]
    name = f"_kernel_{digest}"
    opts = {**_default_jit_options, **(jit_options or {}), "cache": cache}
    final_src = "@njit(**_kernel_jit_options)\n" + source.replace(
        "def _kernel(", f"def {name}(", 1
    )
    anchor = _anchor_path(_ANCHOR_SUBDIR, name, hash_text)
    _materialize_anchor(anchor, final_src)
    code = compile(final_src, str(anchor), "exec")
    # __name__ must be an importable module so numba can rebuild the cached
    # overload's environment in another process (importlib.import_module needs
    # a real name, not None); mirrors make_graph / make_structref.
    ns = {**bindings, "njit": njit, "_kernel_jit_options": opts, "__name__": __name__}
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns[name]
