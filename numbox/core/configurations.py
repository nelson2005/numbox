import os
import json
import warnings

from importlib.metadata import version

import numba.experimental.function_type  # noqa: F401  registers `FunctionModel` against `FunctionType`
from numba import njit
from numba.core.datamodel import default_manager
from numba.core.types import FunctionType, int64, void


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


def _cache_probe():
    return 0


def uncached_where_no_cache_can_be_written(options):
    """``options`` as given, or with ``cache`` off after one warning where numba can write no cache for numbox.

    numba sets a cached function up when it is decorated and raises ``RuntimeError`` there when no cache location
    can be written: a read-only install whose user cache directory cannot be written either, or an import from an
    ``.egg``, ``.whl`` or ``.pyz`` archive, which Spark's ``--py-files`` ships. For a ``.zip`` it takes the user's
    cache directory without checking that it can be written, and the first write raises ``OSError`` instead. Either
    way the import died at the first decorated function, and nothing named the way out.

    Every module here decorates under the one ``jit_options``, and a placement that gives numba no cache for one
    of numbox's functions gives it none for the rest, so the question is put once, to a function of this module
    compiled and saved here, and answered for the package: the options come back with ``cache`` off and one
    warning names the remedy. That is ``NUMBA_CACHE_DIR`` for a source file on disk, and for an archive, where
    numba never reads it, an unpacked install or a ``.zip``, which numba 0.61 and later cache in the user's cache
    directory. ``NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns caching off and silences the warning. An error
    that is not the cache's is raised as it was.
    """
    if not options.get("cache"):
        return options
    try:
        njit(int64(), cache=True)(_cache_probe)
    except (RuntimeError, OSError) as error:
        if not (isinstance(error, OSError) or "no locator available" in str(error)):
            raise
        silence = "NUMBOX_JIT_OPTIONS='{\"cache\": false}' to turn caching off and silence this warning"
        if os.path.exists(__file__):
            remedy = f"Set NUMBA_CACHE_DIR to a writable directory, or {silence}"
        else:
            # Every location numba reads NUMBA_CACHE_DIR for needs the source
            # file on disk, so for an archive the variable changes nothing.
            remedy = (
                "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, "
                "install numbox unpacked or import it from a .zip, which numba 0.61 and later cache in the "
                f"user's cache directory. Set {silence}"
            )
        warnings.warn(
            f"numba cannot cache numbox here ({error}); it compiles without a cache. {remedy}",
            RuntimeWarning, stacklevel=2,
        )
        return {**options, "cache": False}
    return options


jit_options = uncached_where_no_cache_can_be_written(get_jit_options())


_PROXY_CACHE_STRICT_ENV = "NUMBOX_PROXY_CACHE_STRICT"


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
