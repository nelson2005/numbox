import os
import json

from importlib.metadata import version

import numba.experimental.function_type  # noqa: F401  registers `FunctionModel` against `FunctionType`
from numba.core.datamodel import default_manager
from numba.core.types import FunctionType, void


def get_jit_options():
    """
    E.g., export NUMBOX_JIT_OPTIONS='{"cache": false}'
    """
    as_str = os.environ.get("NUMBOX_JIT_OPTIONS")
    if as_str is None:
        return {"cache": True}
    try:
        as_json = json.loads(as_str)
        return as_json
    except json.JSONDecodeError:
        raise ValueError("NUMBOX_JIT_OPTIONS must be valid JSON")


jit_options = get_jit_options()


_PROXY_CACHE_STRICT_ENV = "NUMBOX_PROXY_CACHE_STRICT"

#: Prefix of every LLVM symbol numbox publishes with ``llvmlite.binding.add_symbol`` so that a
#: cached caller can reference it by name instead of by a per-process address. The cache guard in
#: `numbox.core.proxy.proxy` recognises a numbox-owned symbol by this prefix alone, so it lives here
#: rather than in either of the two modules that mint one: `numbox.core.proxy.proxy` publishes a
#: proxied body's cfunc wrapper, and `numbox.utils.derive_wap` publishes a derive's numba-callconv
#: entry point.
_ALIAS_PREFIX = "numbox_pxy_"

#: Aliases two different bodies both minted, so that the name does not identify what it resolves to.
#: `numbox.utils.derive_wap` refuses the second body an alias, which keeps the two call routes agreeing
#: within one process, but a caller cached in a process where the *other* body won the name is served
#: against whichever body happens to hold it here. Recording the collision lets the cache guard in
#: `numbox.core.proxy.proxy` treat such an alias the way it treats an absent binding's: resolvable, and
#: still not safe to serve. Lives here for the same reason `_ALIAS_PREFIX` does -- the module that
#: detects a collision and the module that acts on one are on opposite sides of a deferred import.
_COLLIDED_ALIASES = set()


def _strict_cache_mode():
    """True when ``NUMBOX_PROXY_CACHE_STRICT`` selects strict validation of ``@proxy`` cache loads.

    Strict mode makes every fail-open path in the ``numbox.core.proxy.proxy`` cache guard loud: a payload
    that cannot be read, or an object whose container cannot be parsed, aborts the load with
    ``UnvalidatedProxyCacheError`` instead of loading unchecked; and a detected stale alias raises
    ``StaleProxyCacheError`` before the heal, leaving the stale entry on disk. It is a debugging aid, off by
    default. Unlike ``jit_options`` above, the value is read on each call rather than once at import, so the
    knob can be toggled within a process; the cost is one environment lookup against a multi-millisecond
    cache load. Anything other than the unset/empty/``0``/``false``/``no``/``off`` set (case-insensitive)
    enables it.
    """
    value = os.environ.get(_PROXY_CACHE_STRICT_ENV)
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


MAX_STR_LENGTH = 2 ** 31 - 1


numba_version = int(version("numba").split(".")[1])
assert numba_version >= 60, numba_version

#: numba's `FunctionModel`, resolved once. Looking a data model up is type machinery and
#: compiles nothing, which is what lets the layout be read here: `numbox.utils.lowlevel` and
#: `numbox.utils.derive_wap` both need it on paths that must stay free of compilation side
#: effects, and importing the former compiles a cached eager `@njit` helper.
_function_model = default_manager.lookup(FunctionType(void()))

#: Number of slots in the `FunctionModel` struct: (c_addr, py_addr, jit_addr) on numba 0.61
#: and later, (addr, pyaddr) before it. Read off the data model rather than inferred from the
#: numba version, so the layout is discovered rather than assumed.
function_struct_size = _function_model.field_count


def function_struct_has(field: str) -> bool:
    """Whether numba's `FunctionModel` carries a slot named `field`.

    `get_field_position` is numba's own accessor for the question and raises `KeyError` when
    the slot is absent, which is how a numba predating it reads.
    """
    try:
        _function_model.get_field_position(field)
    except KeyError:
        return False
    return True
