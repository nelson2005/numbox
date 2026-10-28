"""Formula and identifier helpers for the fused-kernel compiler.

This module holds shared utilities that `numbox.core.variable.compile_kernel` uses: identifier assignment for the
generated kernel source, the njit wrapping of formulas, and the validation of formula arity. The utilities are defined
here so that their location stays stable as the kernel machinery grows, and so that a second consumer can reuse them
without importing compile_kernel.
"""
import hashlib
import inspect
import keyword
import re

from typing import Callable

from numba import njit
from numba.core.ccallback import CFunc
from numba.core.dispatcher import Dispatcher
from numba.core.types import Array
from numba.core.types.function_type import CompileResultWAP
from numba.np.ufunc.dufunc import DUFunc

from numbox.core.variable.variable import Variable

# Names injected into the kernel exec namespace; identifiers must avoid them. The set must match the exec
# namespace that compile_kernel._compile assembles.
_RESERVED = frozenset({"njit", "_kernel_jit_options"})


def _sanitize(qual_name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_]", "_", qual_name)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    if not s or s[0].isdigit():
        s = "v_" + s
    return s


def _assign_identifiers(variables: list[Variable]) -> dict[Variable, str]:
    """Map each Variable to a unique, valid, readable Python identifier.

    The identifier is derived from the qual_name, so it stays readable, and it receives a minimal deterministic sha256
    suffix only if names would otherwise collide. The function reserves both the node temp `t` and its formula global
    `f_<t>`, so that those namespaces never clash, and it avoids the injected reserved names.
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


def _wrap_formula(formula: Callable, flags: dict | None = None) -> Dispatcher | CompileResultWAP | DUFunc | CFunc:
    """Return an njit-callable for `formula`.

    A plain-Python callable is njit-wrapped with the kernel's effective jit `flags`, so that its semantics match the
    fused kernel.
    """
    if isinstance(formula, (Dispatcher, CompileResultWAP, DUFunc, CFunc)):
        return formula
    if not callable(formula):
        raise TypeError(f"formula {formula!r} is not callable")
    return njit(**(flags or {}))(formula)


def _strip_cache(flags: dict | None) -> dict:
    """Drop `cache` from the flags used for the transient return-type probe in `_validate_declared_return`.

    The probe is a compile whose result is discarded, so it must not read or write numba's on-disk cache.
    Fused-kernel staleness is handled separately, by folding each formula's digest into the kernel name.
    """
    return {k: v for k, v in (flags or {}).items() if k != "cache"}


def _validate_declared_return(formula, input_types: tuple, declared, flags: dict | None = None) -> None:
    """Raise if the formula's NATURAL return type at `input_types` differs from `declared`.

    njit(sig) silently coerces every convertible scalar mismatch (between int and float, narrowing, sign), so the
    declared return cannot be relied on to fail fast. The natural type is read as follows:

    - CFunc: `_sig.return_type` (data already present).
    - cres (CompileResultWAP): `signature().return_type` (no `_sig` attribute).
    - DUFunc (@vectorize): no single return type; probe it through an unconstrained @njit shim that applies the
      DUFunc at the declared input types.
    - plain Python: an unconstrained @njit probe over the declared input types.

    The comparison is equality, with no coercion allowed, so that the check actually raises on a mismatch. Array
    layout is the one exception, and is ignored. An elementwise formula naturally yields a C-contiguous array, while
    a declared `float64[:]` has layout 'A', and numba allows assignment between the two in either direction. A
    layout difference is therefore not a declared-type violation.
    """
    opts = _strip_cache(flags)
    if isinstance(formula, CFunc):
        natural = formula._sig.return_type
    elif isinstance(formula, CompileResultWAP):
        natural = formula.signature().return_type
    elif isinstance(formula, DUFunc):
        names = ", ".join(f"a{i}" for i in range(len(input_types)))
        ns = {"_f": formula}
        exec(f"def _shim({names}):\n    return _f({names})\n", ns)  # nosec B102
        probe = njit(**opts)(ns["_shim"])
        probe.compile(input_types)
        natural = probe.nopython_signatures[-1].return_type
    elif isinstance(formula, Dispatcher):
        formula.compile(input_types)
        natural = formula.overloads[input_types].signature.return_type
    else:
        probe = njit(**opts)(formula)
        probe.compile(input_types)
        natural = probe.nopython_signatures[-1].return_type
    if isinstance(natural, Array) and isinstance(declared, Array):
        # Normalize both to layout 'A', so that only dtype, ndim and readonly mismatches count; numba allows
        # assignment between the C and A layouts in either direction.
        mismatch = natural.copy(layout="A") != declared.copy(layout="A")
    else:
        mismatch = natural != declared
    if mismatch:
        raise ValueError(
            f"declared type {declared} but formula yields {natural} at "
            f"input types {input_types}"
        )


def _check_formula_arity(formula, n_inputs: int, qual_name: str) -> None:
    target = getattr(formula, "py_func", None) or getattr(formula, "__wrapped__", None) or formula
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return
    try:
        sig.bind(*range(n_inputs))
    except TypeError as e:
        raise ValueError(
            f"{qual_name!r}: formula signature {sig} cannot accept its "
            f"{n_inputs} declared input(s) passed positionally ({e})"
        ) from None
