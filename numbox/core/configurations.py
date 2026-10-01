import importlib.util
import inspect
import os
import json
import marshal
import sys
import types
import warnings
import zipfile

from importlib.metadata import version

import numba.experimental.function_type  # noqa: F401  registers `FunctionModel` against `FunctionType`
from numba.core.caching import CompileResultCacheImpl
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
    except json.JSONDecodeError:
        raise ValueError("NUMBOX_JIT_OPTIONS must be valid JSON")
    if not isinstance(as_json, dict):
        # Its keys go to njit as keyword arguments, and the fallback below
        # reads "cache" from it, so a value of another shape stops here, named.
        raise ValueError('NUMBOX_JIT_OPTIONS must be a JSON object, e.g. {"cache": false}')
    return as_json


def _cache_probe():
    """The function whose cache location is asked for; never compiled."""


def check_cache_location(py_file):
    """Raise as numba would where a function whose source is ``py_file`` cannot be cached; else return.

    The question is put the way numba puts it: the cache set-up that decoration runs, which picks the location
    for the file or raises ``RuntimeError`` with no locator, then the writability check, which decoration runs
    for every location but a ``.zip``'s and the first save runs for all, raising ``OSError``. The probe's code is
    given ``py_file`` as its file, which is all a locator reads of it. Nothing is compiled, and nothing is
    written but the cache directory itself.
    """
    code = _cache_probe.__code__.replace(co_filename=os.fspath(py_file))
    probe = types.FunctionType(code, _cache_probe.__globals__, _cache_probe.__name__)
    CompileResultCacheImpl(probe).locator.ensure_cache_path()


def is_a_cache_error(error):
    """Whether ``error`` is numba's for a cache it cannot set up: no locator, or a location it cannot use.

    numba itself passes over a location on any ``OSError`` from making its directory or writing a file there,
    permission denied, a read-only file system, a path into a file, a component too long, a full disk, so any
    ``OSError`` counts. So does the ``ValueError`` numba raises for an archive under a directory whose name
    holds ``.zip``: its ``.zip`` locator takes the file by that substring and then finds no ``.zip`` in it.
    """
    if isinstance(error, OSError):
        return True
    if isinstance(error, ValueError):
        return "No zip file found" in str(error)
    return isinstance(error, RuntimeError) and "no locator available" in str(error)


def _module_files():
    """The source files numba's answer for numbox rests on, this module's own first.

    numba's in-tree cache is a ``__pycache__`` beside each source, so the directories answer separately and one
    can be writable while another is not: one module of each directory stands for the directory, whether or not
    that directory's modules cache anything, so a directory numba cannot cache in turns caching off for the
    package even where every cached function's own directory is fine; the answer errs toward uncached, which
    is never wrong. And numba finds a location for a module by its source, so a module that survives as
    ``.pyc`` alone, beside sourced ones, answers for itself: it is named by the ``.py`` that is gone, which is
    what numba looks up. An archive shows no directories to walk, so there the probe's own file, not on disk
    either, is the whole answer.
    """
    own = inspect.getfile(_cache_probe)
    yield own
    package = os.path.dirname(os.path.dirname(own))
    if not os.path.isdir(package):
        yield from _archived_module_files(own)
        return
    for directory, subdirectories, files in os.walk(package, followlinks=True):
        subdirectories[:] = sorted(name for name in subdirectories if name != "__pycache__")
        stems = sorted({name.rsplit(".", 1)[0] for name in files if name.endswith((".py", ".pyc"))})
        for index, stem in enumerate(stems):
            source = os.path.join(directory, stem + ".py")
            if index == 0 or not os.path.exists(source):
                yield source


def _archived_module_files(own):
    """One module per directory of numbox inside the ``.zip`` that holds ``own``, and the file each ``.pyc`` in it
    was compiled from; nothing for any other archive.

    numba caches a ``.zip`` per directory of it, each in a location of its own under the user's cache directory,
    so the directories answer separately there too. A ``.pyc`` imported from the archive, which zipimport takes
    before the ``.py`` beside it, keeps the file it was compiled from as its code's file, and that is what numba
    looks up for its functions: the archive's own path for the module is not; so a ``.pyc`` member asks by that
    file, there or gone. Any other archive has no location at all, and the probe's own file has already asked.
    """
    parts = own.split(os.sep)
    depth = next((index for index, part in enumerate(parts) if part.endswith(".zip")), None)
    if depth is None:
        return
    zip_path = os.sep.join(parts[:depth + 1])
    if not zipfile.is_zipfile(zip_path):
        return
    package = "/".join(parts[depth + 1:-2])
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(name for name in archive.namelist()
                       if name.startswith(package + "/") and name.endswith((".py", ".pyc"))
                       and "/__pycache__/" not in name)
        seen = set()
        for name in names:
            directory = name.rpartition("/")[0]
            if name.endswith(".pyc"):
                data = archive.read(name)
                if data[:4] == importlib.util.MAGIC_NUMBER:
                    # The bytes zipimport unmarshals to run this module.
                    yield marshal.loads(data[16:]).co_filename  # nosec B302
            elif directory not in seen:
                seen.add(directory)
                yield os.path.join(zip_path, *name.split("/"))


def uncached_where_no_cache_can_be_written(options):
    """``options`` as given, or with ``cache`` off after one warning where numba can write no cache for numbox.

    numba sets a cached function up when it is decorated and raises ``RuntimeError`` there when no cache location
    can be written: a read-only install whose user cache directory cannot be written either, or an import from an
    ``.egg``, ``.whl`` or ``.pyz`` archive, which Spark's ``--py-files`` ships. For a ``.zip`` it takes the user's
    cache directory without checking at decoration that it can be written, and the first write raises ``OSError``
    instead. Either way the import died at the first decorated function, and nothing named the way out.

    Every function numbox caches decorates under the one ``jit_options``, so the question is put here, once, for
    a source file in each directory of the package that holds a module, and answered for the package. Where
    numba can cache no function of one of those files the options come back with ``cache`` off and one warning
    names the remedy: ``NUMBA_CACHE_DIR`` for a source file on disk; for a ``.zip``, the user's cache directory
    made writable, since numba reads ``NUMBA_CACHE_DIR`` only for a source file on disk; for any other archive,
    or a ``.pyc``-only install, or a ``.pyc`` in a ``.zip``, the source files on disk or a ``.zip`` holding them,
    which numba 0.61 and later cache in the user's cache directory. ``NUMBOX_JIT_OPTIONS='{"cache": false}'``
    turns caching off and silences the warning. An error that is not the cache's, as ``is_a_cache_error`` draws
    the line, is raised as it was.

    A ``.zip`` whose cache directory holds every entry but can no longer be written takes the fallback too, where
    numba alone would have loaded the entries: the writability check is the rule numba applies to every other
    placement, and the one the ``.zip`` locator is missing.
    """
    if not options.get("cache"):
        return options
    for py_file in _module_files():
        try:
            check_cache_location(py_file)
        except (RuntimeError, OSError, ValueError) as error:
            if not is_a_cache_error(error):
                raise
            failure = error
            break
    else:
        return options
    silence = "NUMBOX_JIT_OPTIONS='{\"cache\": false}' to turn caching off and silence this warning"
    if os.path.exists(py_file):
        remedy = f"Set NUMBA_CACHE_DIR to a writable directory, or {silence}"
    elif isinstance(failure, OSError) or getattr(sys, "frozen", False):
        # Two placements numba caches without the source on disk, a .zip and a
        # frozen application, both in the user's cache directory; the error
        # names the directory.
        remedy = (
            "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR "
            f"has no effect here, because the source is not a file on disk: make that directory writable, or "
            f"{silence}"
        )
    else:
        # Every location numba reads NUMBA_CACHE_DIR for needs the source
        # file on disk, so for an archive the variable changes nothing.
        remedy = (
            "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, "
            "install numbox with its source files on disk, unpacked from any archive, or import it from a "
            f".zip holding its source files, which numba 0.61 and later cache in the user's cache directory. "
            f"Set {silence}"
        )
    warnings.warn(
        f"numba cannot cache numbox here ({failure}); it compiles without a cache. {remedy}",
        RuntimeWarning, stacklevel=2,
    )
    return {**options, "cache": False}


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
