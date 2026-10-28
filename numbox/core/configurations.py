import ast
import contextlib
import errno
import importlib.machinery
import inspect
import itertools
import os
import pathlib
import json
import linecache
import re
import sys
import tempfile
import types
import warnings
import zipfile
import zipimport

from importlib.metadata import version

import numba.experimental.function_type  # noqa: F401  registers `FunctionModel` against `FunctionType`
from numba.core.caching import CompileResultCacheImpl
from numba.core.datamodel import default_manager
from numba.misc.appdirs import AppDirs
from numba.core.types import FunctionType, void


numba_version = int(version("numba").split(".")[1])
assert numba_version >= 63, numba_version


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
        # njit takes the keys as keyword arguments, and the fallback below
        # reads "cache" from the object. A value of another shape is therefore
        # refused here, with a message that names the shape expected.
        raise ValueError('NUMBOX_JIT_OPTIONS must be a JSON object, e.g. {"cache": false}')
    if "cache" in as_json and not isinstance(as_json["cache"], bool):
        # numba tests the option for truth, so the string "false" would turn caching on.
        raise ValueError('NUMBOX_JIT_OPTIONS "cache" must be true or false, e.g. {"cache": false}')
    return as_json


def _cache_probe():
    """A function of this module that is never compiled; numba looks up its code's file as the module's file."""


# LONGEST_CACHE_FILE_NAME is an upper bound, with a margin, on the length of a cache file name that numba writes for
# a function of numbox's own files. The name holds the module's stem, the function's qualified name, numba's line
# number, the interpreter tag and the index number. numba writes the file under a temporary name 21 bytes longer, and
# the bound counts those bytes too. The function of the package with the longest name comes to 105 bytes, and the
# builder's generated kernel, anchored to its file, comes to 117. A test keeps every function of the package under
# this bound.
LONGEST_CACHE_FILE_NAME = 128

# The number of digits reserved for the number that numba gives a function's data file. numba writes one data file
# per signature and numbers the files from one without bound. Three digits are reserved, because no function of the
# package or of its generated code reaches a thousand signatures. If an error carries a name with a wider number, the
# remedy reads that number at its own width. Two digits would not suffice: the name of the hundredth signature's file
# is one byte longer than a two-digit reserve allows.
_OVERLOAD_DIGITS = 3

# Each check puts the probe's module in sys.modules under a unique name while
# numba picks the locator; this counter supplies those names.
_probes = itertools.count()


class _MemberSource:
    """The loader of the probe's module, through which inspect reads a ``.zip`` member's source.

    inspect reads the member through this loader as it would read the member's own module through zipimport.
    ``get_source`` returns the member's bytes decoded as UTF-8, which is how zipimport decodes them. It returns None
    if no archive above the file holds the member, or if the member is not UTF-8. zipimport raises for a member that
    is not UTF-8, and linecache passes that error on; None makes inspect raise its own error instead. That error marks
    the member, and the remedy then reads the reason."""

    def __init__(self, py_file):
        self._py_file = py_file

    def get_source(self, name):
        found = _archive_above(self._py_file)
        if found is None:
            return None
        zip_path, member = found
        try:
            with zipfile.ZipFile(zip_path) as archive:
                return archive.read(member).decode()
        except (KeyError, OSError, zipfile.BadZipFile, UnicodeDecodeError):
            return None


@contextlib.contextmanager
def _probing(py_file):
    """A probe for the block: a function that is never compiled, whose code names ``py_file`` as its file.

    The probe is a function of a module of its own, which the block puts in ``sys.modules``. Inside the block, a
    caller passes the probe to a locator, as numba passes a function to it at decoration.

    ``py_file`` can be a string, bytes or a path object. The block reads it as a string, because the probe's code
    carries its file name as a string; ``compile`` makes a string of bytes too.

    numba's IPython locator reads the function's source when it takes the file, as inspect reads it. inspect reads a
    file on disk from the file itself, a cell from linecache, where IPython holds the cell while it runs, and a
    ``.zip`` member through the loader of the function's module. For a real module, that loader is zipimport's. It
    gives the member's source from the archive, or None for a member compiled to ``.pyc`` alone. The loader of the
    probe's module reads the member as zipimport's does, and inspect finds the module in ``sys.modules``.

    After inspect reads the member, linecache holds the member's own lines for the file, as any reader would leave
    them. If the member's archive holds no source, linecache keeps the probe's loader, to call it again later. The
    block removes that linecache entry when it ends, so that no loader of the probe remains after the block.
    """
    file = os.fsdecode(py_file)
    # The probe's source is never put in linecache under the member's name:
    # a reader of the member in another thread would then read the probe's
    # two lines instead of the member's own. The loader below gives the
    # member's own source.
    module = types.ModuleType(f"_numbox_cache_probe_{next(_probes)}")
    module.__file__ = file
    module.__loader__ = _MemberSource(file)
    exec(compile("def _cache_probe():\n    pass\n", file, "exec"), module.__dict__)  # nosec B102 - fixed source
    held = file in linecache.cache
    sys.modules[module.__name__] = module
    try:
        yield module._cache_probe
    finally:
        sys.modules.pop(module.__name__, None)
        if not held and len(linecache.cache.get(file, ())) == 1:
            linecache.cache.pop(file, None)


def _takes(cls, py_file):
    """Whether ``cls``, a locator class outside numba's families, takes ``py_file``.

    The function calls the class's ``from_function`` with a probe, as numba calls it at decoration.
    ``from_function`` returns a locator for a file the class takes, and None for a file it passes over. A class whose
    ``from_function`` raises counts as taking the file, because the error is the class's own and numba raises it at
    decoration. Without the call, the remedy would treat such a class as one that takes every file. It would then
    read a file as the class's own when the class had passed the file over and numba had gone on to the locators
    listed after that class."""
    with _probing(py_file) as probe:
        try:
            return cls.from_function(probe, os.fsdecode(py_file)) is not None
        except Exception:
            return True


def check_cache_location(py_file, longest_file_name=0):
    """Raise the error numba would raise if a function whose source file is ``py_file`` cannot be cached; else return.

    ``py_file`` can be a string, bytes or a path object; the check reads it as a string.

    The check makes the calls that decoration makes, in three steps. The first is the cache set-up, which picks the
    location for the file, or raises ``RuntimeError`` if no locator takes it. The second reads the source's stamp,
    which decoration reads too. For a ``.zip``, numba stats the archive, and from numba 0.68 it also reads the
    member's source from the archive, as from 0.68 it reads a source on disk for its hash. The third is the
    writability check, which raises ``OSError``. Decoration runs it for every location except a ``.zip``'s, and the
    first save runs it for all.

    The probe is compiled with ``py_file`` as its file. A locator reads only that file name from the probe, except
    numba's IPython locator, which also reads the function's source, as inspect reads it. inspect reads a file on disk
    from the file itself, a cell from linecache, where IPython holds the cell while it runs, and a ``.zip`` member
    through the loader of the function's module. For a real module, that loader is zipimport's, which gives the
    member's source from the archive. The probe is therefore a function of a module of its own, whose loader reads
    the member as zipimport's does, and that module is in ``sys.modules`` while numba picks the locator. After
    inspect reads the member, linecache holds the member's own lines for the file, as any reader would leave them. If
    the member's archive holds no source, linecache keeps the probe's loader, to call it again later. The check
    removes that linecache entry, so that no loader of the probe remains after the check. inspect raises its error
    for a cell that linecache does not hold, or for a member compiled to ``.pyc`` alone, as it does for numba at
    decoration.

    The check compiles nothing. It writes nothing but the cache directory itself, with one exception. An IPython that
    can write neither its cache directory nor its own makes a temporary directory on every call. The check causes
    three such calls: numba makes two while it makes the directory, and the check makes one of its own.

    An ``OSError`` from the check names the location numba picked, with three exceptions. The first is inspect's error,
    which names no file. If numba's IPython locator takes a cell file on disk, inspect reads the source from the file
    itself before the locator tries its location. inspect raises if it finds no source there: in a file it cannot read
    or, before Python 3.13, in an empty one, which linecache then holds as a blank line. The second is the stamp's
    error, which names the archive that a ``.zip`` member's code names, if that archive has since moved or cannot be
    read. The check reads the stamp once the locator is picked, as numba reads it at decoration. The third applies from
    numba 0.68, whose stamp hashes the source: the stamp's error can then name the source itself, if the source cannot
    be read. From 0.68, the stamp's error can also name a directory named ``.zip`` above a file on disk, which numba's
    ``.zip`` locator treats as the archive and then opens.

    A location can be absent after numba makes it. numba's IPython locator asks IPython for its directory at every
    save. An IPython that can write neither its cache directory nor its own gives a new temporary directory on every
    call, so the directory numba made is not the one IPython gives next. The check then raises ``FileNotFoundError``
    naming ``numba_cache`` under the last directory IPython gave, which numba never made. numba's first save would
    instead name a file in ``numba_cache`` under the directory IPython gave when numba set the cache up, a different
    temporary directory.

    numba's own writability check misses a location near the path limit. It makes a temporary file with a short name, or
    no file at all on Linux, while the names of the files numba saves reach a hundred bytes and more. A location within
    that length of the path limit therefore passes numba's check, and the first save fails. If ``longest_file_name`` is
    given, the check also makes and removes a file with a name that long in the location, and the ``OSError`` it raises
    names the location. numbox passes ``LONGEST_CACHE_FILE_NAME``, the bound on the names numba gives the package's own
    files. If the name is longer than the file system holding the location allows, no shorter path fixes that. The error
    then carries the length as ``longest_file_name``, and the remedy asks for a file system that takes the name.
    """
    # numba picks the locator for a probe, a function in a module of its own.
    with _probing(py_file) as probe:
        locator = CompileResultCacheImpl(probe).locator
    # numba reads the source's stamp at decoration too. For a .zip, the stamp
    # is the archive's, and reading it raises if the code names an archive
    # that has since moved.
    locator.get_source_stamp()
    try:
        locator.ensure_cache_path()
    except OSError as error:
        # numba's error names whatever failed, possibly a temporary file. A
        # user can act on the location that holds it, so the error raised
        # here names the location.
        raise OSError(error.errno, error.strerror, locator.get_cache_path()) from error
    # The check asks for the location once more, as numba does at every save.
    # numba's IPython locator asks IPython for it. An IPython that can write
    # neither its cache directory nor its own gives a new temporary directory
    # on every call. numba then makes numba_cache in one directory and saves
    # in another, and the save fails there although numba's own check passed.
    # Without this call, the check would pass the file as numba's check does.
    # numba asks twice while it makes the directory, and the check asks once
    # more here; each call returns a temporary directory of its own.
    location = locator.get_cache_path()
    if not os.path.isdir(location):
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", location)
    if longest_file_name:
        try:
            # tempfile appends eight random characters to the prefix.
            with tempfile.NamedTemporaryFile(dir=location, prefix="x" * (longest_file_name - 8)):
                pass
        except OSError as error:
            raised = OSError(error.errno, error.strerror, location)
            if error.errno == errno.ENAMETOOLONG and _name_limit(location) < longest_file_name:
                # The name alone, not the path, is too long for the file
                # system. The error carries the name's length, so that the
                # remedy does not attribute the error to the location and
                # offer a shorter path.
                raised.longest_file_name = longest_file_name
            raise raised from error


def is_a_cache_error(error):
    """Whether ``error`` is numba's error for a cache it cannot set up.

    Such an error has one of three causes: no locator takes the file, numba cannot use a location, or, from numba
    0.68, numba cannot read a ``.zip`` member's source from the archive.

    Any ``OSError`` counts, because numba itself passes over a location on any ``OSError`` from making its directory
    or writing a file there. Such errors include permission denied, a read-only file system, a path through a file, a
    component too long and a full disk.

    The ``ValueError`` that numba raises for an archive under a directory whose name holds ``.zip`` counts too. numba's
    ``.zip`` locator takes the file because of that substring, and then finds no ``.zip`` in the path.

    The ``KeyError`` that numba 0.68 and later raise when they read a ``.zip`` member's source from the archive counts
    as well; numba hashes that source into the archive's stamp. It is zipfile's error, naming the member, for a name
    the archive does not hold. The name is missing for a module shipped as ``.pyc`` alone. It is also missing on
    Windows under numba 0.68.0 for any member, because that release builds the name with the platform's separator
    (numba issue 10889, fixed in 0.68.1).

    The ``UnicodeEncodeError`` that numba raises when it hashes the source's directory path as UTF-8 counts as well.
    numba hashes that path for the name of its directory under ``NUMBA_CACHE_DIR`` or the user's cache directory, and
    raises for a path that holds bytes that are not UTF-8, decoded with surrogates. A locator that names its cache
    directory by that hash raises the error at decoration when numba reaches the locator, after the locators before it
    passed the file over, so no location takes the file. Without this case, a package installed under such a
    directory, with its ``__pycache__`` unwritable, would fail at import, although it imports with caching off.
    """
    if isinstance(error, OSError):
        return True
    if isinstance(error, UnicodeEncodeError):
        return error.reason == "surrogates not allowed"
    if isinstance(error, ValueError):
        return "No zip file found" in str(error)
    if isinstance(error, KeyError):
        return numba_version >= 68 and "There is no item named" in str(error)
    return isinstance(error, RuntimeError) and "no locator available" in str(error)


def _module_files():
    """The source files that decide whether numba can cache numbox, this module's own file first.

    numba's in-tree cache is a ``__pycache__`` beside each source, so each directory gives its own answer, and one can
    be writable while another is not. One module of each directory represents the directory, whether or not that
    directory's modules cache anything. If numba cannot cache in one directory, caching is therefore off for the whole
    package, even if the directory of every cached function is usable. A doubtful answer is uncached, which is never
    wrong.

    numba finds a location for a module by the file its code names. A module that exists as ``.pyc`` alone, beside
    modules with sources, is therefore checked on its own, by the file it was compiled from, which its code keeps.
    That file is the ``.py`` that no longer exists, if the module was compiled in place, or a file in a tree
    elsewhere, on disk or not.

    The package is found by this module's ``__file__``, which is where it was imported from, an archive or a
    directory. This module's code can name a file elsewhere in the same way. numba looks that file up for the probe,
    and the check runs for that file first. An archive has no directories to walk, so its members are listed instead.
    """
    yield inspect.getfile(_cache_probe)
    package = os.path.dirname(os.path.dirname(__file__))
    if not os.path.isdir(package):
        yield from _archived_module_files(__file__)
        return
    walked = set()
    for directory, subdirectories, files in os.walk(package, followlinks=True):
        # Symlinks are followed, and a directory reached twice through links,
        # as in a cycle, is walked once.
        real = os.path.realpath(directory)
        if real in walked:
            subdirectories[:] = []
            continue
        walked.add(real)
        subdirectories[:] = sorted(name for name in subdirectories if name != "__pycache__")
        stems = sorted({name.rsplit(".", 1)[0] for name in files if name.endswith((".py", ".pyc"))})
        sources = [os.path.join(directory, stem + ".py") for stem in stems]
        on_disk = [source for source in sources if os.path.exists(source)]
        if on_disk:
            yield on_disk[0]
        for source in sources:
            if source not in on_disk:
                compiled_from = _sourceless_compiled_from(source[:-3] + ".pyc")
                if compiled_from is not None:
                    yield compiled_from


def _sourceless_compiled_from(pyc_path):
    """The file the ``.pyc`` at ``pyc_path``, alone beside sourced modules, was compiled from, or None.

    The import system runs such a ``.pyc`` through its sourceless loader, which keeps the compile-time file name on
    the code. The module's import fails if the loader cannot run the ``.pyc``, as for one with another interpreter's
    magic number or an unreadable one. The function therefore reads the loader's answer, and if the loader gives none,
    there is nothing to check.
    """
    loader = importlib.machinery.SourcelessFileLoader(os.path.basename(pyc_path)[:-4], pyc_path)
    try:
        return loader.get_code(loader.name).co_filename
    except Exception:
        return None


def _archived_module_files(own):
    """The files to check for numbox inside the ``.zip`` that holds ``own``, this module's ``__file__``.

    The function yields one module per directory of numbox in the archive, as the file its code names. It also yields
    every ``.pyc`` whose code names a member the archive does not hold, and the check runs for that ``.pyc`` on its
    own. Such a ``.pyc`` has no ``.py`` beside it, or was compiled to name another path in the archive. The function
    yields nothing for any other kind of archive.

    numba caches a ``.zip`` per directory of it, each in a location of its own under the user's cache directory, so
    each directory gives its own answer there too. A ``.pyc`` run from the archive keeps the file it was compiled from
    as its code's file. numba looks that file up for the module's functions, not the archive's own path for the
    module, so the check runs for such a member by that file, whether the file exists or not.

    numba 0.68 and later read the member's source from the archive by the name its code gives. A ``.pyc`` alone has
    no source there. A ``.pyc`` compiled to name another path in the archive (``compileall -d``) has no source there
    either, whether or not its ``.py`` is beside it. The check therefore runs for either kind of ``.pyc`` on its own,
    whichever member represents its directory.

    The archive is the first part of the path that is named ``.zip`` and is an archive; a directory so named above it
    is not an archive. Any other kind of archive has no cache location at all, and the check has already run for the
    probe's own file.
    """
    parts = own.split(os.sep)
    for depth, part in enumerate(parts):
        zip_path = os.sep.join(parts[:depth + 1])
        if part.endswith(".zip") and zipfile.is_zipfile(zip_path):
            break
    else:
        return
    package = "/".join(parts[depth + 1:-2])
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(name for name in archive.namelist()
                       if name.startswith(package + "/") and name.endswith((".py", ".pyc"))
                       and "/__pycache__/" not in name)
    held = set(names)
    seen = set()
    for name in names:
        directory, _, file = name.rpartition("/")
        if file.endswith(".pyc"):
            member = _compiled_from(zip_path, directory, file[:-4])
            if member is None:
                continue
            if not member.startswith(zip_path + os.sep):
                yield member
                continue
            named = member[len(zip_path) + 1:].replace(os.sep, "/")
            if named not in held:
                # The archive does not hold the member the code names. A .pyc
                # alone names the .py it was compiled from, which no longer
                # exists. A .pyc compiled to name another path in the archive
                # (compileall -d) names a member that never existed, with or
                # without the .py beside it. numba 0.68 and later read the
                # member's source from the archive by that name and find none,
                # so the check runs for the .pyc on its own, whichever member
                # represents the .pyc's directory. If a .py member represented
                # the directory alone, the check would never run for the .pyc,
                # and its module would fail at decoration. The same would
                # happen if a .pyc beside its .py were taken to name the
                # archive.
                seen.add(directory)
                yield member
                continue
        else:
            member = os.path.join(zip_path, *name.split("/"))
        if directory not in seen:
            seen.add(directory)
            yield member


def _compiled_from(zip_path, directory, stem):
    """The file that is named by the code which ``zipimport`` runs for module ``stem`` of ``directory``, or None if it
    runs no code.

    zipimport decides which code runs, so the function reads the code that zipimport gives. zipimport runs the
    ``.pyc`` rather than the ``.py`` beside it, unless the ``.pyc`` has another interpreter's magic number, is stale
    against the ``.py``, or cannot be unmarshalled. zipimport compiles the source instead in the first two cases, and
    the module's import fails in the last.

    The code names the archive's own path for the module if zipimport compiled the ``.py``, or if the ``.pyc`` was
    compiled to name a member the archive holds. Such a file represents its directory, as a ``.py`` member does.
    Otherwise the code names a file the archive does not hold, elsewhere, or a path in the archive that the ``.pyc``
    was compiled to name, present or not. The check then runs for that file on its own. If zipimport runs no code,
    because it cannot compile the source, cannot decompress the member, or fails for any other reason, there is
    nothing to check.
    """
    if stem == "__init__":
        directory, _, stem = directory.rpartition("/")
    importer = zipimport.zipimporter(os.path.join(zip_path, *directory.split("/")))
    try:
        return importer.get_code(stem).co_filename
    except Exception:
        return None


def _moved_through():
    """A clause for a remedy naming what moves numba's user cache directory on this platform; empty on Windows.

    numba's ``appdirs`` gets the directory from the system on Windows, where no variable moves it. On macOS it puts
    the directory under ``HOME``. Elsewhere it uses ``XDG_CACHE_HOME``, or a directory under ``HOME`` if
    ``XDG_CACHE_HOME`` is unset.
    """
    if sys.platform == "win32":
        return ""
    if sys.platform == "darwin":
        return ", through HOME"
    return ", through XDG_CACHE_HOME or HOME"


def _measured(name):
    """The length of ``name`` as the file system counts it, and the unit's name.

    Windows and macOS count a name in UTF-16 units; the file systems elsewhere count bytes. A surrogate counts as one
    unit. A name decoded with surrogateescape carries a surrogate for each byte that is not UTF-8, and Windows accepts
    a surrogate in a name. Encoding without ``surrogatepass`` would raise on a surrogate, and the remedy would then
    fail on a name that numba's error carried instead of returning its text."""
    if os.name == "nt" or sys.platform == "darwin":
        return len(name.encode("utf-16-le", "surrogatepass")) // 2, "characters"
    return len(os.fsencode(name)), "bytes"


def _name_limit(location):
    """The longest name a file can have where ``location`` is.

    It is 255 if the system cannot report the limit, if the system reports 0, or if ``location`` is None; a file system
    with no limit of its own reports 0. The count is in bytes outside Windows and macOS, and in UTF-16 units on those
    two.

    The function reads the limit of the nearest directory above ``location`` that exists, because the location itself
    may not exist yet.
    """
    if location is None or not hasattr(os, "statvfs"):
        return 255
    path = os.path.abspath(location)
    while not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    try:
        return os.statvfs(path).f_namemax or 255
    except OSError:
        return 255


def _locators(listed):
    """numba's cache locator classes in the order numba tries them.

    ``listed`` is ``NUMBA_CACHE_LOCATOR_CLASSES``. Each entry is a class of numba's
    caching module, by its name, or any class by its dotted path, resolved as numba resolves it. If ``listed`` is
    empty, the order is numba's own order.

    An entry that numba could not resolve is not a locator here. numba has already refused such an entry before trying
    any location, or raised importlib's error for it, or raised a ``TypeError`` for a relative module path. An entry
    that names no class, or a class with no ``from_function``, is not a locator here either. numba fails on such an
    entry, because it lacks ``from_function``, at a decorated function that reaches the entry. In every case numba's
    own error is raised unchanged.
    """
    from numba.core import caching
    if not listed:
        return list(caching.CacheImpl._locator_classes)
    classes = []
    for entry in listed.split(","):
        entry = entry.strip()
        if "." in entry:
            module_path, class_name = entry.rsplit(".", 1)
            try:
                cls = getattr(importlib.import_module(module_path), class_name, None)
            except Exception:
                # importlib's errors, a relative path's TypeError and any error
                # the module raises on import. numba raises these as its own at
                # decoration, and the check raises them before any remedy runs.
                # The remedy runs only for a cache error, so it must not raise
                # the module's error instead.
                cls = None
        else:
            cls = getattr(caching, entry, None)
        if isinstance(cls, type) and callable(getattr(cls, "from_function", None)):
            classes.append(cls)
    return classes


def _existing_above(path):
    """The first path at or above ``path`` that exists.

    That path is the directory under which the first missing directory on the way to ``path`` would be made. If the
    location does not exist, that directory is the one that refused it. The result can also be a file in the path,
    under which no directory can be made, or, at the end of the walk, the root. ``path`` is read as absolute. A
    relative path with no existing ancestor would walk up to an empty string, and the remedy would then report a file
    with no name in the path."""
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def _ipython_numba_cache():
    """numba's cache location for a cell file that its IPython locator takes, or None.

    The location is ``numba_cache`` under IPython's cache directory, with no directory per file. It is None if IPython
    is not importable. numba's locator raises ``ImportError`` for that at decoration, which is not a cache error, so
    the location is then none of numba's. It is None too if IPython raises for its directory. For an ``OSError``,
    numba's locator passes the file over, as it does for a location it cannot make, so numba took no location of
    IPython's. numba's locator would raise any other error at decoration. The remedy makes the same call only to word
    a warning, and takes such an error as no location."""
    try:
        try:
            from IPython.paths import get_ipython_cache_dir
        except ImportError:
            from IPython.utils.path import get_ipython_cache_dir
        return os.path.join(get_ipython_cache_dir(), "numba_cache")
    except Exception:
        return None


# The messages of inspect's error for a source it does not read. inspect raises that error for a cell that linecache
# does not hold, a member whose archive holds no source, a function whose line lies past the lines inspect read, and a
# function of no module.
_INSPECTS_ERRORS = ("could not get source code", "source code not available", "lineno is out of bounds")


def _is_inspects(failure):
    """Whether ``failure`` is inspect's error for a source it could not read.

    Such an error is an ``OSError`` with no errno, no file name and one of inspect's own messages. A caller's error
    with no errno and no file name has the same form, and only its message differs."""
    return (isinstance(failure, OSError) and failure.errno is None and not getattr(failure, "filename", None)
            and str(failure) in _INSPECTS_ERRORS)


def _ipython_cache_is_temporary(first):
    """Whether IPython gives a new temporary directory as its cache directory on every call.

    IPython does that, with a warning, if it can write neither its cache directory nor its own. numba's IPython locator
    asks for the directory at every save, so numba then makes ``numba_cache`` in one directory and saves in another.
    ``first`` is a location that ``_ipython_numba_cache`` already returned. The function asks it for one more, and each
    directory IPython makes stays on disk. An IPython that returns a directory once and raises the next time gives no
    temporary directory, so a second result of None does not count as a new directory."""
    if first is None:
        return False
    second = _ipython_numba_cache()
    return second is not None and second != first


def _taken_by_ipython(py_file):
    """Whether numba's IPython locator takes ``py_file`` by its name alone, whether or not the file is on disk.

    The locator takes a cell, ``<ipython-...>``, and a file in an ipykernel directory, and reads the function's source
    when it takes the file."""
    return py_file.startswith("<ipython-") or os.path.basename(os.path.dirname(py_file)).startswith("ipykernel_")


def _archive_above(py_file):
    """The archive that ``py_file`` is a member of, and the member's name in it, as zipimport finds them; or None.

    The first part of the path that is a file, whatever its name, must be a zip archive, and the rest of the path is
    the member. The result is None if no part of the path is a file, or if the first part that is a file is not an
    archive.

    The function splits the path by the platform's separator, ``os.sep``, through the pure path classes. The concrete
    ``pathlib.Path`` would build a path of the platform that ``os.name`` names, which a test can set to Windows's.
    Python before 3.12 refuses to build such a path on Linux, so the remedy would fail for a member that numba names
    with Windows's separator.
    """
    pure = pathlib.PureWindowsPath if os.sep == "\\" else pathlib.PurePosixPath
    parts = pure(os.fsdecode(py_file)).parts
    for at in range(len(parts) - 1):
        prefix = str(pure(*parts[:at + 1]))
        if os.path.isdir(prefix):
            continue
        if zipfile.is_zipfile(prefix):
            return prefix, "/".join(parts[at + 1:])
        return None
    return None


def _ipython_locator_reads(py_file):
    """Whether a locator of numba's IPython family that numba reaches takes ``py_file`` by its name.

    Such a locator reads the function's source when it takes the file. numba reaches the locators in its own order, or
    in the order ``NUMBA_CACHE_LOCATOR_CLASSES`` lists them. For a path with ``.zip`` in it, numba reaches no locator
    listed after its ``.zip`` locator, because that locator takes the file without trying its location. numba also
    reaches no locator listed after one that takes the file, whether that locator is one of numba's own or a class
    outside its families. The function walks the list in numba's order. A locator of IPython's family reads the file.
    numba's ``.zip`` locator takes a path with ``.zip`` in it, and numba reaches no locator after it. The function calls
    every other locator with a probe, as numba calls it, and one that takes the file ends the walk. The in-tree locator,
    for example, takes a cell file on disk whose ``__pycache__`` it can write. If a locator passes the file over, numba
    goes on to the locators after it."""
    from numba import config
    from numba.core import caching
    if not _taken_by_ipython(py_file):
        return False
    for cls in _locators(config.CACHE_LOCATOR_CLASSES):
        if issubclass(cls, caching.IPythonCacheLocator):
            return True
        if issubclass(cls, caching.ZipCacheLocator) and ".zip" in py_file:
            return False
        if _takes(cls, py_file):
            # A locator before IPython's took the file, so numba does not reach
            # the IPython locator. Without this test, the remedy would read a
            # caller's error that names some other numba_cache as the IPython
            # locator's error, for a cell file on disk whose __pycache__ the
            # in-tree locator writes.
            return False
    return False


def _numbas_archive_holds(py_file, name):
    """Whether the archive that numba's ``.zip`` locator uses for ``py_file`` holds a member named ``name``.

    That archive is the first part of the path ending in ``.zip``; the result is False if that part is not an
    archive."""
    parts = py_file.split(os.sep)
    at = next((i for i, part in enumerate(parts) if part.endswith(".zip")), None)
    if at is None:
        return False
    try:
        with zipfile.ZipFile(os.sep.join(parts[:at + 1])) as archive:
            return name in archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def _pyc_naming(py_file):
    """The ``.pyc`` member whose code names ``py_file``, and whether a ``.py`` is beside it; or None.

    The archive searched is the one that numba's ``.zip`` locator uses for ``py_file``, and the code is the code that
    zipimport runs for the member. The result is None if no member's code names ``py_file``. The function finds a
    ``.pyc`` compiled to name another path in the archive (``compileall -d``). numba 0.68 and later read the source by
    that name and find no source under it, although the archive holds the source beside the ``.pyc``."""
    parts = py_file.split(os.sep)
    at = next((i for i, part in enumerate(parts) if part.endswith(".zip")), None)
    if at is None:
        return None
    zip_path = os.sep.join(parts[:at + 1])
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return None
    for name in sorted(names):
        if name.endswith(".pyc") and "/__pycache__/" not in name:
            directory, _, file = name.rpartition("/")
            if _compiled_from(zip_path, directory, file[:-4]) == py_file:
                return name, name[:-1] in names
    return None


def _member_not_utf8(py_file):
    """Why the member of the archive above ``py_file`` is not UTF-8: the byte and the reason; or None.

    zipimport decodes a member's source as UTF-8, and numba's IPython locator reads the source as zipimport decodes
    it. The result is None if no archive above the file holds the member, or if the member decodes as UTF-8."""
    found = _archive_above(py_file)
    if found is None:
        return None
    zip_path, member = found
    try:
        with zipfile.ZipFile(zip_path) as archive:
            data = archive.read(member)
    except (KeyError, OSError, zipfile.BadZipFile):
        return None
    try:
        data.decode()
    except UnicodeDecodeError as error:
        return f"{error.reason} at byte {error.start}"
    return None


def _missing_member(error):
    """The member that zipfile's ``KeyError`` names, "There is no item named 'x' in the archive", or its whole message.

    zipfile quotes the name as Python does, so the message doubles a backslash in the name, and the function reads the
    name back from the quotes. numba 0.68.0 builds a member's name with backslashes on Windows.
    """
    message = str(error.args[0]) if error.args else str(error)
    found = re.search(r"^There is no item named (.+) in the archive$", message)
    if not found:
        return message
    try:
        return str(ast.literal_eval(found.group(1)))
    except (ValueError, SyntaxError):
        return found.group(1)


def cache_remedy(py_file, failure, silence, package="numbox"):
    """The remedy for ``failure``, numba's error for a function whose file is ``py_file``, ending in ``silence``.

    The remedy is one sentence. It names what numba tried, why that failed and what would move it, and it ends in
    ``silence``, the clause that says how to turn caching off. ``py_file`` can be a string, bytes or a path object,
    and so can the file name the error carries; the function reads both as strings.

    ``package`` is the package that the remedy asks the user to install again, at a shorter path or with its source
    files on disk. A package built on numbox can check its own files with ``check_cache_location`` and
    ``is_a_cache_error``. Such a package passes its own name here, as it passes ``check_cache_location`` a
    ``longest_file_name`` for its own functions; ``LONGEST_CACHE_FILE_NAME`` is numbox's.

    The remedy depends mostly on which locator took the file. numba caches a source file on disk in one of three
    places: a directory under ``NUMBA_CACHE_DIR`` if that variable is set, else the ``__pycache__`` beside the
    source, else a directory under the user's cache directory. In the first and the third, numba caches under a
    directory named after the source's directory, by its name and a hash of its path. The three places can nest,
    and ``NUMBA_CACHE_DIR`` set to the user's cache directory makes two of them one path. The remedy therefore
    matches the whole path the error names against the location that each locator numba reaches would give the
    file, in the order numba tries the locators. It attributes the error to the first locator that matches. numba's
    own errors mean that the locators before that one passed the file over. An error that a caller passes on, naming
    a location numba did not take, need not mean that. For a path with ``.zip`` in it, numba reaches no locator
    listed after its ``.zip`` locator, because that locator takes the file without trying its location. A location
    that one of those later locators would give is then none of numba's.

    ``NUMBA_CACHE_LOCATOR_CLASSES`` decides the order. Each entry is one of three
    kinds. The first is a class of numba's caching module. The second is a subclass of such a class, by its dotted
    path, which takes the files its parent takes and caches where its parent caches; the remedy treats it as its
    parent. The third is a class outside numba's families, with a ``from_function`` and a place of its own. It takes
    whatever its ``from_function`` takes when called with a probe, as numba calls it, and caches in a place that only
    the class can name. If an error names a location that no locator of numba's gives the file, the remedy names that
    location as the error names it, and the location is then that class's. If an error names no file, the remedy
    lists the location numba took as the outside class's own, with no path to name, among the places of the numba
    locators that numba reaches.

    numba's IPython locator takes a file on disk only in an ipykernel directory, and its ``.zip`` locator takes only a
    path with ``.zip`` in it. Listing either before the others therefore changes nothing for any other file. If the
    ``.zip`` locator is listed first, it takes such a path before any other locator does. It caches the file under
    the user's cache directory if a part of the path ends in ``.zip``, whether that part is an archive or a
    directory. If no part does, the locator finds no archive and raises, and the remedy asks for a locator for a file
    on disk listed before it. The user-provided locator counts only if ``NUMBA_CACHE_DIR`` is set, because it takes
    nothing without the variable. If the ``.zip`` locator is listed after some of the locators for a file on disk, it
    finds no archive and raises once those have passed the file over. numba then tries no locator after it. The
    remedy asks for one of their locations to be made writable, or for such a locator to be listed before it.

    If the list has no locator that takes the file, the remedy says so. The remedy also says so if the only locator
    for the file in the list is the user-provided one, with ``NUMBA_CACHE_DIR`` unset, and then asks for the variable
    to be set. The remedy for an error that a caller passes on under such a list is the same as for numba's
    no-locator error, because numba took no location under the list.

    For a source file on disk, the remedy offers ``NUMBA_CACHE_DIR``, because numba reads the variable for no other
    kind of source. If the variable is set and numba passed it over, the remedy names the variable and asks for a
    writable directory at a short path. numba passes over a directory too deep to make its own directory in, as it
    passes over an unwritable one. If ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves the user-provided locator out, numba
    never reads the variable, and the remedy asks for the locations the list names to be made writable.

    The location numba took can refuse a file for another reason: a full disk, or permissions changed since numba's
    own check. The remedy then names the location and the reason, and asks for free space or a writable directory
    there. It offers ``NUMBA_CACHE_DIR`` as the alternative if numba tries the variable before that location's
    locator. If the location is the variable's own, it offers the variable set to another directory. numba passes
    over a location it cannot make or write in, so only numba's no-locator error means that numba passed the
    variable over.

    The remedy names a location that is none of numba's as the error names it. If numba tries ``NUMBA_CACHE_DIR``
    before any other locator, the remedy also names the variable. It asks for a set variable to be made a writable
    directory at a short path. It asks for an unset variable to be set to a short path for a path too long, and to a
    writable directory otherwise. The remedy does not say that numba passed the variable over, because such a
    location cannot show that.

    If a path is too long for the file system, the error names the location, and the remedy depends on which of the
    three places it is. For the first, the remedy is a shorter ``NUMBA_CACHE_DIR``. For the second, it is the
    package at a shorter path. For the third, it is the user's cache directory at a shorter path, through
    ``XDG_CACHE_HOME`` or ``HOME``; on macOS only ``HOME`` moves that directory, and on Windows nothing does. For
    either of the last two, the remedy also offers ``NUMBA_CACHE_DIR`` set to a short path. It makes that offer if
    numba tries the variable before the locator that took the location. It also makes the offer if numba tries the
    variable before the last of the locators that give the file that one path, as the user-wide and ``.zip`` locators
    do. If the variable is set and numba passed it over, the remedy names the variable and asks for it to be made a
    writable directory at a short path.

    A directory name can be too long for the file system. numba names a directory after the source's directory, under
    ``NUMBA_CACHE_DIR`` or the user's cache directory, and that directory's name is the source directory's name plus
    forty-one bytes. A name longer than the file system holding the location allows cannot be made at any path, and
    the system refuses it whatever the errno, ``EINVAL`` on Windows. The remedy reads the limit from the system, and
    takes 255 if the system cannot report it. The limit is in bytes, or in the UTF-16 units that Windows and macOS
    count a name in.

    For three errors, the remedy gives the name and its length, and asks for the file in a directory of a shorter
    name. The first is an error naming such a location. The second is numba's no-locator error for a file on disk, or
    its ``.zip`` locator's error once the locators before it passed the file over. That error is one of the three if
    every locator numba tried that names its cache directory by that hash found the name too long. The variable's
    locator measures by the limit of the directory the variable is set to, or by 255 if the variable is unset. If the
    variable's locator is the only locator numba tried and the variable is unset, the remedy also asks for the
    variable to be set. The third is a frozen application's error.

    For a file on disk, the remedy adds the ``__pycache__`` beside the source, which caches under no such name. It
    asks for that directory to be made writable if numba reaches the in-tree locator. Otherwise, under a list, it
    asks for the in-tree locator to be listed, before the ``.zip`` locator or the outside class that took the file if
    the list puts the in-tree locator after one.

    A cache file name can be too long for the file system. numba names the files it writes for a function after the
    source's name and the function's qualified name. numba can fail to make or open such a file in the location it
    took, because the name is longer than that file system allows. The remedy then gives the name and its length,
    and says that no shorter path fixes it. It asks for a shorter name for the function or its file, which only the
    package can give.

    If a file system in use takes the name, the remedy also offers ``NUMBA_CACHE_DIR`` pointed at a directory on such a
    file system. It names the variable as set if the variable's own location took the file. Otherwise it offers the
    variable only if numba tries the variable before the locator that took the file, as it offers the variable at a
    short path for a path too long. For a ``.zip`` member, it offers nothing, because the variable has no effect on one.
    A file system in use takes the name if the name's base fits in 255, the common limit, less the 29 bytes that numba
    adds for the longest name it writes for the function. That name belongs to a data file, with three digits for its
    number, written through a temporary name. numba numbers the data files from one without bound, and a wider number in
    the error's name counts at its own width.

    The check's own error can report a name as long as the names numba gives the package's functions, on a file
    system that takes no name that long. The remedy then gives the length. It offers ``NUMBA_CACHE_DIR`` pointed at a
    directory on a file system that takes such a name, on the same terms, whatever the name's length. The remedy
    offers nothing for a ``.zip`` member, because the variable has no effect on one.

    A path can hold bytes that are not UTF-8. numba hashes the source's directory path as UTF-8 for the name of its
    directory, and raises at decoration for such bytes when a locator that names its cache directory by that hash
    reaches the file. For such a path, the remedy names no location under ``NUMBA_CACHE_DIR`` or the user's cache
    directory. numba's ``.zip`` locator hashes the archive's path the same way for its directory under the user's
    cache directory. The remedy for a ``.zip`` under such a path therefore asks for the archive under a UTF-8 path.
    The user-wide locator hashes a frozen application's path the same way, so the remedy for a frozen application
    under such a path asks for the application under a UTF-8 path.

    The remedy for a file on disk under such a path also names the ``__pycache__`` beside it, because the in-tree
    locator hashes no path for that directory. If the list leaves the in-tree locator out, or puts it after a locator
    that hashes other than the variable's, the remedy asks for it to be listed before the first locator that hashes.
    If only the variable's locator hashes before the in-tree locator, the remedy asks for ``NUMBA_CACHE_DIR`` to be
    unset. If the in-tree locator came before every locator that hashes and passed the directory over, the remedy
    asks for the ``__pycache__`` to be made writable. With any of these, the remedy also asks for the ``__pycache__``
    to be made writable if it cannot be written.

    numba's IPython locator caches a cell file in ``numba_cache`` under IPython's cache directory, with no directory
    per file. The locator passes the location over if it cannot make or write in it. No other locator of numba's takes
    a cell, so a cell then gets numba's no-locator error. The remedy names that location, to be made writable or put
    at a shorter path. If IPython raised for its cache directory, the remedy instead names that directory and asks for
    a writable home for it. The remedy names a class outside numba's families listed with the locator as having
    passed the cell over too. If that location is too long and the IPython locator is listed before the ``.zip`` one,
    the remedy asks for IPython's cache directory at a shorter path. It does so for a cell file on disk and for a
    ``.zip`` member in an ipykernel directory alike, and offers no ``NUMBA_CACHE_DIR`` for a source not on disk.

    The remedy asks IPython for its directory only if a locator of that family is listed and takes the file. It never
    asks for any other file, a plain ``.zip`` member included, because IPython warns on such a call under a home it
    cannot write, and leaves a temporary directory behind. numba imports IPython when it makes the location, and
    catches only ``OSError`` there. If IPython is not importable, numba therefore raises ``ImportError`` at decoration
    for a file that the IPython locator takes, and that error is not a cache error. A caller can pass on an error that
    names that location. The remedy then names the location as none of numba's if another locator numba reaches has a
    place for the file. If no such locator has one, the remedy names the list, as for numba's no-locator error. If
    IPython raises ``OSError`` for its directory, numba passes the locator over, as it does a location it cannot make.
    The remedy makes the same call to IPython, so it has no location of IPython's to match the error against.

    IPython can give a new temporary directory as its cache directory on every call, if it can write neither that
    directory nor its own. This affects a file in an ipykernel directory that numba's IPython locator takes, a cell
    file on disk or a ``.zip`` member. numba then makes ``numba_cache`` in one such directory and saves in another.
    Its first save raises ``FileNotFoundError``, naming a file in the ``numba_cache`` directory it could not save in.
    For such an error naming a file, the remedy says what IPython does, and asks for ``IPYTHONDIR`` or
    ``XDG_CACHE_HOME`` at a writable path. An error of another errno that names a ``numba_cache`` comes from a
    caller, and the remedy names the location the error names. For an error naming no file, the remedy lists the
    locations numba could have taken, with the error's reason.

    An error can name no file, carry an empty string or empty bytes as its name, or carry a name of another kind, an
    integer file descriptor, which counts as no name. A caller can build such an error, and numba's first save raises
    one when a write fails, as on a full disk. inspect raises one too when, for numba's IPython locator, it reads no
    source in a cell file on disk that cannot be read or, before Python 3.13, is empty. inspect's error has no errno
    either, because no call into the file system raised it.

    The remedy reads that error as inspect's for a file on disk that a locator of IPython's family takes, once numba
    reaches that locator, because the locator tried no location. It asks for the file to be made readable, with the
    source in it. numba reads a cell's source from linecache while IPython holds it to run the cell, so for a cell the
    remedy says that numba caches a cell only while IPython holds its source. The remedy for a ``.zip`` member whose
    archive holds no source, a ``.pyc`` alone, is the archive's: it asks for the source files to be shipped.

    Every other ``OSError`` that ``check_cache_location`` raises names a file. For an error with no name, or an empty
    one, the remedy lists the locations numba could have taken, in numba's order and each once. For a path too long,
    it asks for whichever of them is too long at a shorter path. For a file refused for another reason, such as a
    full disk, it gives the reason and asks for free space or a writable directory there. A path object is never
    empty, because pathlib reads an empty string as the working directory, ``.``, which the remedy names as a
    location that is none of numba's.

    numba caches a ``.zip`` and a frozen application under the user's cache directory. The remedy for either asks
    for that location to be made writable or given free space, with the reason the error gives, a full disk or
    permissions. The ``.zip``'s error names the location, a directory of numba's under the user's cache directory, or
    a file numba writes in it, as the first save's error does, and the remedy names the directory. If an error names
    a location that is none of numba's, under the list and in its order, the remedy names that location as the error
    names it, as for a source on disk. The remedy then says nothing of what moves the user's cache directory, or of
    where numba caches the file, because that location is none of numba's.

    The frozen application's error is numba's no-locator error, which gives no reason. numba passed the location over on
    an error of its own, whether the location was unwritable or too deep. The remedy therefore names the user's cache
    directory as one numba could not use, to be made writable or put at a shorter path. For a ``.zip`` or a frozen
    application, if the name numba gives the directory after the source's directory is too long for the file system, the
    remedy says that instead. The remedy says the same of a cache file name too long for the file system, in that
    location or in IPython's ``numba_cache`` for a member in an ipykernel directory, which no shorter path fixes. If a
    ``.zip``'s location is too long for the file system, the user's cache directory is the cause, because the names
    numba makes are bounded. The remedy is then that directory at a shorter path, through what moves it on the platform.

    A frozen application's path with ``.zip`` in it and no part ending in ``.zip`` can get the ``.zip`` locator's error
    instead. That happens if the list puts that locator before the user-wide one, or once the user-wide one passed
    its location over. The remedy then asks for the user-wide locator to be listed before the ``.zip`` one, or for
    the user-wide location to be made writable or put at a shorter path.

    For a ``.zip`` or a frozen application, the remedy lists the locations numba could have taken for an error that
    names no file, as for a source on disk, in order and each once. Such an error is a caller's, or the first save's on
    a full disk. The remedy gives the reason, or, for a path too long, asks for whichever location is too long at a
    shorter path. For such an error, the remedy does not name the user's cache directory as where numba caches a member
    in an ipykernel directory, if numba reaches a locator of IPython's family. IPython's ``numba_cache`` is among the
    possible locations whenever IPython gives its directory.

    numba caches a ``.zip`` through its ``.zip`` locator alone, so a list that leaves that locator out gives numba no
    locator for a source in a ``.zip``. The remedy then asks for the locator to be listed. That holds for a member of
    an archive that exists; a directory named ``.zip`` above a source that does not exist is not an archive. The same
    holds for a notebook cell and numba's IPython locator, and for a frozen application and numba's user-wide
    locator, the only locator that takes one. The remedy is the same for an error that a caller passes on under such
    a list as for numba's no-locator error, because numba took no location.

    For any other kind of archive, or a module without its source, the remedy is the source files on disk or a
    ``.zip`` holding them. If ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves out the locator that either needs, the remedy
    also asks for that locator to be listed. The source files on disk need one of the three locators for a file on
    disk, and ``NUMBA_CACHE_DIR`` set if the user-provided locator is the one listed; the ``.zip`` needs the ``.zip``
    locator. A source that is neither in a ``.zip`` nor a frozen application's, which no locator takes, gets the same
    remedy for any error.

    From numba 0.68, the archive's stamp hashes the member's source. numba reads that source from the archive by the
    name the module's code gives under the archive's path, and zipfile raises ``KeyError`` for a name the archive does
    not hold. For a module shipped as ``.pyc`` alone, the remedy names the member and asks for the source files, on disk
    or in a ``.zip``. A ``.pyc`` can be compiled to name another path in the archive (``compileall -d``) while the
    archive holds its source beside it. The remedy for such a ``.pyc`` is to compile it to name its own path, or to
    leave it out. On Windows, numba 0.68.0 raises that error for every member, because that release builds the name with
    the platform's separator (numba issue 10889, fixed for 0.68.1). A release with the fix reads the name with forward
    slashes. If the archive holds the member's source under the forward-slash name, the remedy is the install on disk,
    or a numba release that contains the fix. Otherwise the remedy is the one for any other member.

    numba treats the first part of the path ending in ``.zip`` as the archive, including a directory so named above
    the file, and from 0.68 opens it to read the source. The remedy for the ``IsADirectoryError`` that follows names
    the directory, and asks for the archive under a path with no such directory above it. The ``.zip`` locator takes
    a file on disk under such a directory once the locators before it passed the file over. The remedy for such a
    file asks for one of their locations to be made writable. If the ``.zip`` locator is listed first, the remedy asks
    instead for a locator for a file on disk to be listed before it. The remedy also gives the alternative of putting
    the file elsewhere.
    """
    from numba import config
    from numba.core import caching
    from numba.core.caching import _CacheLocator
    # check_cache_location takes any path-like object, bytes included, and so
    # does cache_remedy. The remedy reads a str, and os.fspath would leave
    # bytes as bytes.
    py_file = os.fsdecode(py_file)
    # The error names the refused file in the form the raising call received
    # it, as bytes for a call given bytes, or in the form a caller built the
    # error with, a path object included. The remedy reads the name as a str,
    # as it reads the source's path. An empty str or bytes is no name. A path
    # object built from an empty string names the working directory, ".", as
    # pathlib reads it. A name of another kind is no name, such as the integer
    # file descriptor that os.stat(fd) raises with.
    filename = getattr(failure, "filename", None)
    filename = os.fsdecode(filename) if filename and isinstance(filename, (str, bytes, os.PathLike)) else None
    listed = config.CACHE_LOCATOR_CLASSES
    user_provided = caching.UserProvidedCacheLocator
    in_tree = caching.InTreeCacheLocator
    user_wide = caching.UserWideCacheLocator
    for_ipython = caching.IPythonCacheLocator
    for_a_zip = caching.ZipCacheLocator

    def one_of(cls, base):
        # Whether cls is base or a subclass of it. A subclass listed by its
        # dotted path takes the files its parent takes and caches where its
        # parent caches, as numba's InTreeCacheLocatorFsAgnostic does. Each of
        # numba's locators therefore represents its whole family.
        return issubclass(cls, base)

    def of_numbas(cls):
        # Whether cls belongs to one of numba's locator families, whose places
        # the remedy knows. A class outside them has a from_function and a
        # place of its own, and caches in a place that only the class can name.
        return any(one_of(cls, base) for base in (user_provided, in_tree, user_wide, for_ipython, for_a_zip))

    def could_have_taken(locations):
        # An error that names no file does not show which location numba
        # took. The remedy therefore lists the locations numba could have
        # taken, the values of locations in numba's order, each once. On disk,
        # NUMBA_CACHE_DIR set to the user's cache directory makes two of them
        # one path. A class outside numba's families has a place the remedy
        # cannot name, so the list calls that place the class's own.
        known = list(dict.fromkeys(
            location if location else f"{cls.__name__}'s own"
            for cls, location in locations.items() if location or not of_numbas(cls)))
        if len(known) > 1:
            return f"the location numba took, one of {', '.join(known[:-1])} or {known[-1]}"
        return f"the location numba took, {known[0]}"

    def listed_without(base):
        # Whether NUMBA_CACHE_LOCATOR_CLASSES is set and names no class of
        # base's family.
        return bool(listed) and not any(one_of(cls, base) for cls in _locators(listed))

    try:
        subpath = _CacheLocator.get_suitable_cache_subpath(py_file)
    except UnicodeEncodeError:
        # numba hashes the source's directory path as UTF-8 for the name of
        # its directory. It raises at decoration for bytes that are not UTF-8,
        # when a locator that names its cache directory by that hash reaches
        # the file. The remedy then has no such location to name.
        subpath = None

    def too_long_a_name(location):
        # numba names the cache directory it makes for the file after the
        # source's directory, by its name and a hash of its path. That
        # directory is under NUMBA_CACHE_DIR or the user's cache directory. A
        # name longer than the file system holding the location allows cannot
        # be made at any path, so no shorter path fixes it. The result is the
        # clause that says so, or None if the name fits or numba has no name to
        # give.
        if subpath is None:
            return None
        limit = _name_limit(location)
        length, counted_in = _measured(subpath)
        if length <= limit:
            return None
        return (
            f"numba names the directory it caches this file in after the source's directory, {subpath}, {length} "
            f"{counted_in}, which is longer than a name can be on that file system ({limit})"
        )

    a_shorter_name = f"a shorter name for the function or its file, which is {package}'s to give"

    def a_name_limit_below(location):
        # The check raises an error for a name as long as the names numba
        # gives the package's functions, on a file system that takes no name
        # that long. That error carries the name's length. The result is the
        # clause that says so, or None for any other error. Without this
        # clause, the remedy would read the error as the location's and offer
        # a shorter path, which has no effect on a name that is too long.
        longest = getattr(failure, "longest_file_name", 0)
        if not longest or location is None:
            return None
        limit = _name_limit(location)
        if limit >= longest:
            return None
        length, counted_in = _measured("x" * longest)
        return (
            f"numba's cache files for {package}'s functions have names of up to {length} {counted_in}, longer than "
            f"a name can be on that file system ({limit})"
        )

    def too_long_a_file_name(location):
        # numba names the files it caches a function in after the source's
        # name and the function's qualified name. A name longer than the file
        # system holding the location allows cannot be made at any path, so
        # no shorter path fixes it; only a shorter name does. The result is
        # the clause that says so for the file the error names in the
        # location, and whether any file system in use takes the name at all.
        # The result is None if the error names no such file, or if the
        # file's name fits.
        #
        # ext4, NTFS and APFS all limit a name to 255, in the units each
        # counts. The name is measured against the longest name that numba
        # writes for the function. numba writes the index, ".nbi", and a data
        # file per signature, ".<n>.nbc", numbered from one without bound. It
        # writes each through a temporary name 21 longer: ".tmp." and sixteen
        # hex digits. The data file's number takes the digits reserved for it,
        # or the width of the number the error names if that is wider. Without
        # this clause, the remedy would ask for a shorter path for the package,
        # the user's cache directory, IPython's cache directory or the
        # variable, and none of these shortens a name. The remedy offers a file
        # system that takes the name only if one in use does. Its test counts
        # the temporary name and a data file name with three digits for its
        # number, or with the number the error names if that is wider. The
        # remedy always names the shorter name.
        if not filename or location is None:
            return None
        named = os.path.abspath(filename)
        if os.path.dirname(named) != os.path.abspath(location):
            return None
        name = os.path.basename(named)
        length, counted_in = _measured(name)
        limit = _name_limit(location)
        if length <= limit:
            return None
        clause = (
            f"numba names the cache files for a function after the source's name and the function's qualified "
            f"name, {name}, {length} {counted_in}, which is longer than a name can be on that file system ({limit})"
        )
        stripped = re.sub(r"\.tmp\.[0-9a-f]{16}$", "", name)
        numbered = re.search(r"\.(\d+)\.nbc$", stripped)
        digits = max(_OVERLOAD_DIGITS, len(numbered.group(1))) if numbered else _OVERLOAD_DIGITS
        base = re.sub(r"\.nbi$|\.\d+\.nbc$", "", stripped)
        return clause, _measured(base)[0] + len(".") + digits + len(".nbc") + 21 <= 255

    a_file_on_disk = os.path.exists(py_file)

    def may_take(cls):
        # Whether a locator takes the file by its name alone, before any
        # location is tried. numba's IPython locator takes a cell, or a file in
        # an ipykernel directory, on disk or not. Its .zip locator takes a path
        # with .zip in it. That locator caches the file if a part of the path
        # ends in .zip, an archive or a directory, and raises if no part does.
        # Its user-wide locator takes a frozen application's file as well as
        # any file on disk. numba's other locators take any file on disk. A
        # class outside numba's families has a from_function and a place of
        # its own. It takes what its from_function takes when called with a
        # probe, as numba calls it, and caches in a place that only the class
        # can name. Without the call, the remedy would treat such a class as
        # one that takes every file. It would then report a file the class
        # passed over as the class's own, when numba went on to the next
        # locator listed.
        if one_of(cls, for_ipython):
            return _taken_by_ipython(py_file)
        if one_of(cls, for_a_zip):
            return ".zip" in py_file
        if one_of(cls, user_wide):
            return a_file_on_disk or bool(getattr(sys, "frozen", False))
        if of_numbas(cls):
            return a_file_on_disk
        return _takes(cls, py_file)

    def names_a_path_above():
        # Whether the error names a path above the file: the archive that a
        # member's code names, or a directory in the path. Both are read as
        # absolute, because a relative file name matches no absolute name.
        # Otherwise the remedy would report a member's archive that has since
        # moved as a location that refused the file. The working directory
        # does not count. A path object built from an empty string names it,
        # and the remedy names that path as a location that is none of
        # numba's. Otherwise the working directory, which lies above any file
        # under it, would count as a moved archive.
        above = os.path.abspath(filename) if filename else ""
        return bool(above) and above != os.getcwd() and os.path.abspath(py_file).startswith(above + os.sep)

    def names_another_location():
        # Whether the error names the location of one of numba's locators for
        # a file on disk other than IPython's, or a file in such a location.
        # Those locations are NUMBA_CACHE_DIR's, the __pycache__ beside the
        # source and the user's cache directory. A location that is none of
        # those, such as a temporary directory IPython gave, is taken as the
        # IPython locator's.
        if not filename:
            return False
        named = os.path.abspath(filename)
        if named == os.path.abspath(py_file):
            # The error names the source itself, which numba reads. If the
            # source cannot be read, the remedy asks for it to be made
            # readable, whatever directory IPython gives as its cache
            # directory.
            return True
        others = [os.path.abspath(config.CACHE_DIR)] if config.CACHE_DIR else []
        others.append(os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__")))
        others.append(os.path.abspath(AppDirs(appname="numba", appauthor=False).user_cache_dir))
        return any(named == other or named.startswith(other + os.sep) for other in others)

    asked = []

    def ipython_numba_cache():
        # IPython's cache directory, which the remedy asks IPython for only
        # once. IPython makes a temporary directory on every call, and warns,
        # under a home directory that it cannot write.
        if not asked:
            asked.append(_ipython_numba_cache())
        return asked[0]

    def names_a_numba_cache():
        # Whether the error names IPython's location for numba, a numba_cache
        # directory or a file in one, as missing. If IPython gives a temporary
        # directory on every call, numba's first save raises FileNotFoundError
        # naming a file in the numba_cache directory it could not save in. The
        # save fails because numba made that directory in a different
        # temporary directory. The check's error names the directory. A
        # caller's error that names another location shows nothing of
        # IPython's locator. Without this test, the remedy would offer
        # IPYTHONDIR for a cell on disk that the in-tree locator took. It
        # would also offer it for a member under such an IPython on any errno,
        # including a refused directory of the caller's own.
        return (
            bool(filename) and failure.errno == errno.ENOENT
            and "numba_cache" in os.path.abspath(filename).split(os.sep))

    if isinstance(failure, OSError) and names_a_numba_cache() and not names_a_path_above() and (
            _ipython_locator_reads(py_file)) and (
            not a_file_on_disk or not names_another_location()
    ) and _ipython_cache_is_temporary(ipython_numba_cache()):
        # numba reaches its IPython locator first for a member in an ipykernel
        # directory. numba reaches that locator for a cell file on disk in such
        # a directory once the locators before it passed the file over. The
        # locator asks IPython for its cache directory at every save. An
        # IPython that can write neither that directory nor its own gives a
        # new temporary directory on every call. numba then makes numba_cache
        # in one directory and saves in another, and the save in the second
        # fails. Without this case, the remedy would report the directory the
        # error names, a temporary one, as refusing a file. For a file on
        # disk, it would also offer NUMBA_CACHE_DIR beside the source.
        #
        # Three errors in particular are left out of this case. The first is
        # an error that names no file. The remedy lists the locations numba
        # could have taken for it, with its reason, as the docstring says,
        # whatever directory IPython gives. For a file on disk, such an error
        # also shows nothing of which locator took the file. The error numba's
        # first save raises here names a file in the numba_cache directory it
        # could not save in. A caller's error with a reason and no name, such
        # as a full disk's, differs: it keeps its reason instead of getting
        # IPYTHONDIR.
        #
        # The second is inspect's error, with no errno and no name, which
        # inspect raised before the locator tried any location. The cases
        # below report it as inspect's for a cell or a file on disk, and as
        # the archive's for a member shipped as .pyc, whatever directory
        # IPython gives. IPython's directory is not the cause for a cell whose
        # source linecache does not hold, or for such a member.
        #
        # The third is an error naming a path above the file: the archive that
        # a member's code names, when that archive has since moved. It is the
        # stamp's error, and the cases below report it. The temporary
        # directory's error names a numba_cache under the temporary directory.
        #
        # From IPython 8, IPython reads XDG_CACHE_HOME for its cache directory
        # on every POSIX system, macOS included; IPython 7 did not read it on
        # macOS. The message therefore says POSIX, not Linux, so that it does
        # not lead a macOS user away from the variable.
        return (
            "numba caches this file in numba_cache under IPython's cache directory, and IPython, able to write neither "
            "that directory nor its own, gives a new temporary directory on every call, so numba makes numba_cache "
            "in one and saves in another: set IPYTHONDIR, or XDG_CACHE_HOME on a POSIX system, to a writable "
            "directory; or "
            f"{silence}"
        )

    if a_file_on_disk:
        def reads_the_variable(cls):
            return one_of(cls, user_provided)

        def caches_beside_the_source(cls):
            return one_of(cls, in_tree)

        def caches_under_the_user_cache_dir(cls):
            return one_of(cls, user_wide) or one_of(cls, for_a_zip)

        order = [cls for cls in _locators(listed) if may_take(cls)]
        # numba's .zip locator takes a path with .zip in it without trying
        # its location. A class outside numba's families that takes the file
        # when called with a probe also ends the order. numba reaches no
        # locator listed after either. Without cutting the order there, the
        # remedy would treat the locators after such a class as reached. It
        # would then report an error that names IPython's location as one
        # where numba caches the file.
        zip_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip)), None)
        stops_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip) or not of_numbas(cls)), None)
        reached = order if stops_at is None else order[:stops_at + 1]
        # numba does not reach a .zip locator listed after such a class, so
        # under that list any error the .zip locator would raise comes from a
        # caller. Without this, the remedy would read those errors as the .zip
        # locator's own, as if the locators before it had passed the file
        # over, when one of them had taken the file.
        zip_at = zip_at if zip_at == stops_at else None
        variable_at = next((at for at, cls in enumerate(order) if reads_the_variable(cls)), None)
        cache_dir_read = variable_at is not None

        def instead(taken, asks="a short path", shared=None, made="a writable directory at a short path"):
            # The clause that offers NUMBA_CACHE_DIR as the alternative to the
            # location numba took. The clause offers the variable if numba
            # tries the variable before that location's locator. If the
            # variable is set, numba passed it over because the directory was
            # unwritable or too deep, and the made argument says what the
            # remedy asks the user to make of the directory. If the variable is
            # unset, numba would take it, and the asks argument says what the
            # remedy asks the user to set the variable to.
            #
            # The taken argument is None for a location that is none of
            # numba's, and such a location counts as the first locator in the
            # order other than the variable's. It shows nothing of what numba
            # did with the variable, so the clause names a set variable
            # without the claim that numba passed it over.
            #
            # The shared argument is the last locator numba reaches whose place
            # the error also names. The user-wide and .zip locators give the
            # file one path under the user's cache directory, and the error
            # cannot show which of them took the file. The clause therefore
            # offers the variable if numba tries it before the last of them,
            # without the claim. If the clause read the first of those two
            # locators as the one that took the file, it would leave the
            # variable out whenever the variable's locator comes between them.
            if subpath is None:
                # numba names its directory under NUMBA_CACHE_DIR by hashing
                # the source's directory path as UTF-8, and raises for a path
                # that is not UTF-8. If the remedy offered the variable here,
                # a user who set it would make numba hash the path and raise
                # at decoration. The warning would then become an import
                # error.
                return (
                    ", and NUMBA_CACHE_DIR cures nothing, numba hashing the source's directory as UTF-8 for its "
                    "directory there, which this one is not"
                )
            if taken in order:
                position = order.index(taken)
                passed_over = " and numba could not use"
            else:
                position = next((at for at, cls in enumerate(order) if not reads_the_variable(cls)), len(order))
                passed_over = ""
            if shared is not None and shared is not taken and shared in order:
                position = max(position, order.index(shared))
                passed_over = ""
            if not cache_dir_read or variable_at >= position:
                return ""
            if config.CACHE_DIR:
                return f", or NUMBA_CACHE_DIR, which is set to {config.CACHE_DIR}{passed_over}, made {made}"
            return f", or NUMBA_CACHE_DIR set to {asks}"

        user_cache_dir = AppDirs(appname="numba", appauthor=False).user_cache_dir

        def beside_the_source():
            # The in-tree locator caches beside the source, in its __pycache__,
            # under no directory named after the source's directory. It is
            # therefore the one locator of numba's for a file on disk that
            # still works when a directory name is too long. The remedy offers
            # the __pycache__ made writable if numba reaches the in-tree
            # locator and the locator passed the file over. It offers the
            # in-tree locator listed if NUMBA_CACHE_LOCATOR_CLASSES leaves it
            # out. It offers the in-tree locator listed before the .zip one if
            # the .zip locator, listed first, took a path with .zip in it and
            # numba never reached the in-tree locator. A writable __pycache__
            # would fix nothing under a list that never reaches the in-tree
            # locator.
            if any(caches_beside_the_source(cls) for cls in reached):
                beside = os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__"))
                return (
                    f", or make the __pycache__ beside the source, {beside}, writable, where InTreeCacheLocator "
                    "caches it under no such name"
                )
            if listed:
                # If the in-tree locator is listed after the locator that took
                # the file, the .zip one or a class outside numba's families,
                # the in-tree locator must come before that locator. The
                # remedy names that locator, which is not always the .zip one.
                before = ""
                if stops_at is not None and any(caches_beside_the_source(cls) for cls in order):
                    stopper = order[stops_at]
                    before = f" before {'ZipCacheLocator' if one_of(stopper, for_a_zip) else stopper.__name__}"
                return (
                    f", or list InTreeCacheLocator{before}, which caches beside the source, in its __pycache__, under "
                    "no such name"
                )
            return ""

        def location_of(cls):
            # Where the class places the file. The .zip locator places it under
            # the user's cache directory, as the user-wide one does. The result
            # is None for a class that is not numba's, whose place is its own.
            # It is None for the user-provided locator without NUMBA_CACHE_DIR,
            # which takes nothing without the variable. It is None for the
            # IPython locator if IPython is not importable, or if IPython
            # raises OSError for its directory; numba then passes the locator
            # over.
            if reads_the_variable(cls):
                return os.path.abspath(os.path.join(config.CACHE_DIR, subpath)) if config.CACHE_DIR and subpath else None
            if caches_beside_the_source(cls):
                return os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__"))
            if caches_under_the_user_cache_dir(cls):
                return os.path.abspath(os.path.join(user_cache_dir, subpath)) if subpath else None
            if one_of(cls, for_ipython):
                ipython_cache = ipython_numba_cache()
                return os.path.abspath(ipython_cache) if ipython_cache else None
            return None

        locations = {cls: location_of(cls) for cls in reached}
        if _is_inspects(failure) and any(one_of(cls, for_ipython) for cls in reached):
            # inspect's error, which no call into the file system raised, so
            # it has no errno and no file name. numba's IPython locator reads
            # the function's source, from the file itself for a file on disk.
            # inspect raises if it reads no source there: in a file it cannot
            # read or, before Python 3.13, in an empty one. That happens before
            # the locator tries its location, once the locators numba tried
            # before it passed the file over. numba therefore took no location
            # that the remedy could report as refusing the file.
            return (
                f"numba's IPython locator reads the function's source from its file before trying its location, and "
                f"read none in {py_file} ({failure}): make the file readable, with the source in it; or {silence}"
            )
        if isinstance(failure, OSError) and filename and os.path.abspath(filename) == os.path.abspath(py_file):
            # The error names the source itself, which numba reads. Its IPython
            # locator reads the function's source, and from 0.68 every locator
            # reads the file for the stamp, which hashes the source. The remedy
            # reports a file that cannot be read as unreadable, not as the
            # location numba took refusing the file.
            reason = failure.strerror or failure
            return (
                f"numba reads the function's source from its file, {py_file}, which cannot be read ({reason}): make "
                f"it readable; or {silence}"
            )
        # numba took no location under a list with no locator that takes the
        # file, or with no locator that has a place for it. A list of the
        # user-provided locator alone, with NUMBA_CACHE_DIR unset, is one such
        # list. The list itself is then the remedy below for an error a caller
        # passes on, as it is for numba's no-locator error. A class outside
        # numba's families has a place of its own, which an error can name.
        if isinstance(failure, OSError) and (any(locations.values()) or not all(of_numbas(cls) for cls in reached)):
            # The error names the location numba took, or a file numba writes
            # in that location. numba's own check passed the location because
            # its temporary file fit where numba's cache files would not. Or
            # the file the error names, which the package's check made, was
            # refused later, by a full disk or by permissions changed since the
            # check. The locator that took the location decides the remedy: the
            # first locator in numba's order whose place the error names. The
            # remedy matches paths whole, because the places can nest, as they
            # do for an install under either cache directory, or for
            # NUMBA_CACHE_DIR set above the user's cache directory. It matches
            # them in order, because NUMBA_CACHE_DIR set to the user's cache
            # directory makes two places one path.
            if filename:
                named = os.path.abspath(filename)
                named = {named, os.path.dirname(named)}
            else:
                # A caller can pass on an error that names nothing, and numba's
                # first save raises one when a write fails, as on a full disk.
                named = set()
            taken = next((cls for cls in reached if locations[cls] and locations[cls] in named), None)
            shared = next((cls for cls in reversed(reached) if locations[cls] and locations[cls] in named), None)
            # The system refuses a name that is too long whatever the errno;
            # on Windows the errno is EINVAL.
            named_after_the_source = taken is not None and (
                reads_the_variable(taken) or caches_under_the_user_cache_dir(taken))
            too_long = too_long_a_name(locations[taken]) if named_after_the_source else None
            if too_long:
                return f"{too_long}: put the file in a directory of a shorter name{beside_the_source()}; or {silence}"
            too_long_file = too_long_a_file_name(locations[taken]) if taken is not None else None
            if too_long_file:
                # The remedy offers the variable pointed at a directory on a
                # file system that takes the name. It makes the offer if numba
                # tries the variable before the locator that took the file, and
                # if a file system in use takes the name. The first condition
                # is the one under which the remedy offers the variable at a
                # short path for a path too long. The variable's own location
                # took the file only because the variable is set, so the
                # remedy names the variable as set.
                clause, a_file_system_takes_it = too_long_file
                offer = ""
                if a_file_system_takes_it:
                    fits = "directory on a file system that takes a name that long"
                    if reads_the_variable(taken):
                        offer = f", or NUMBA_CACHE_DIR, which is set to {config.CACHE_DIR}, pointed at a {fits}"
                    else:
                        offer = instead(taken, asks=f"a {fits}", shared=shared, made=f"a writable {fits}")
                return f"{clause}, and no shorter path cures it: {a_shorter_name}{offer}; or {silence}"
            too_long_for_the_names = a_name_limit_below(locations[taken]) if taken is not None else None
            if too_long_for_the_names:
                fits = "directory on a file system that takes a name that long"
                if reads_the_variable(taken):
                    offer = f", or NUMBA_CACHE_DIR, which is set to {config.CACHE_DIR}, pointed at a {fits}"
                else:
                    offer = instead(taken, asks=f"a {fits}", shared=shared, made=f"a writable {fits}")
                return f"{too_long_for_the_names}, so no shorter path cures it{offer}; or {silence}"
            if failure.errno == errno.ENAMETOOLONG:
                if taken is not None and reads_the_variable(taken):
                    # The variable's location, which numba's own check passed
                    # and the first save's longer name overflows. numba goes
                    # on using the variable as it is set, so the user's cache
                    # directory at a shorter path would change nothing that
                    # numba reads, even if the variable is set to that
                    # directory.
                    cure = "a shorter NUMBA_CACHE_DIR"
                elif taken is not None and caches_beside_the_source(taken):
                    cure = f"{package} installed at a shorter path{instead(taken)}"
                elif taken is not None and caches_under_the_user_cache_dir(taken):
                    cure = (
                        f"the user's cache directory, {user_cache_dir}, at a shorter path{_moved_through()}"
                        f"{instead(taken, shared=shared)}"
                    )
                elif taken is not None and one_of(taken, for_ipython):
                    cure = f"IPython's cache directory, {os.path.dirname(locations[taken])}, at a shorter path{instead(taken)}"
                elif not filename:
                    cure = f"{could_have_taken(locations)}, at a shorter path{instead(None)}"
                else:
                    cure = f"that location, {filename}, at a shorter path{instead(None)}"
                return f"the path is too long for the file system: {cure}; or {silence}"
            if (failure.errno == errno.EISDIR and zip_at is not None and filename and os.path.isdir(filename)
                    and filename.endswith(".zip") and names_a_path_above()):
                # numba's .zip locator takes a file on disk because the path
                # contains ".zip", after the locators before it have passed
                # the file over. It treats the first part of the path ending in
                # .zip as the archive, including a directory so named. From
                # 0.68, numba opens that archive to read the source, and a
                # directory refuses that. The directory is not a cache
                # location, so the remedy does not ask for free space in it;
                # the remedy for a member handles the same case. Under a list
                # with no .zip locator, the error comes from a caller, and the
                # remedy names the location the error names.
                if zip_at:
                    reached_how = "reached once the locators before it passed this file over"
                    cure = f"make one of those locators' locations writable{instead(None, 'a writable directory')}, or "
                else:
                    # If the .zip locator is listed first, it takes the file
                    # before any locator for a file on disk, so no locators
                    # come before it to name.
                    reached_how = "which NUMBA_CACHE_LOCATOR_CLASSES puts before every locator for a source file on disk"
                    cure = "list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or " \
                           "UserWideCacheLocator before it, or "
                return (
                    f"numba's .zip locator, {reached_how}, takes the first part of the path ending in .zip, {filename}, "
                    "for an archive, and it is a directory, which numba 0.68 and later open for the source: "
                    f"{cure}put the file under a path with no such directory; or {silence}"
                )
            # numba took the location, so it did not pass the variable over for
            # that location. If the variable is set to another directory, it
            # moves the cache off a full disk. The remedy names a location that
            # is none of numba's as the error names it. For an error that names
            # no location, it lists the locations numba could have taken.
            if taken is not None and reads_the_variable(taken):
                alternative = "" if subpath is None else ", or NUMBA_CACHE_DIR set to another writable directory"
            else:
                alternative = instead(taken, "a writable directory", shared=shared)
            if taken:
                opening = f"numba caches this file in {locations[taken]},"
            elif filename:
                opening = f"that location, {filename},"
            else:
                opening = f"{could_have_taken(locations)},"
            # An error a caller builds from a message alone has no strerror.
            reason = failure.strerror or failure
            return (
                f"{opening} where no file can be written ({reason}): make room there, or make it "
                f"writable{alternative}; or {silence}"
            )
        # For a source on disk, numba itself passes over a location it cannot
        # make or write in. An error that reaches this point is therefore one
        # of two. The first is the no-locator error: numba tried every locator,
        # and each passed the file over. The second is the .zip locator's error
        # for a path with .zip in it and no part ending in .zip. numba raises
        # it when it reaches that locator: the locators before it passed the
        # file over, and numba tried none after it.
        zip_raised = isinstance(failure, ValueError) and zip_at is not None
        tried = order[:zip_at] if zip_raised else order
        if not tried:
            if zip_raised:
                return (
                    "numba's .zip locator, which NUMBA_CACHE_LOCATOR_CLASSES puts before every locator for a source "
                    'file on disk, takes this file for the ".zip" in its path and finds no archive there: list '
                    "UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or UserWideCacheLocator "
                    f"before it, or {silence}"
                )
            return (
                f"numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, {listed}, says, and none of those locators takes "
                "a source file on disk: list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator "
                f"or UserWideCacheLocator, or {silence}"
            )
        named_after_the_source = [
            cls for cls in tried
            if cls in locations and (reads_the_variable(cls) or caches_under_the_user_cache_dir(cls))]
        if named_after_the_source and all(too_long_a_name(locations[cls]) for cls in named_after_the_source):
            # Every locator numba tried that names its directory after the
            # source's directory found that name too long for its file system,
            # and passed the file over for that reason. numba does not reach a
            # class listed after the .zip locator for a path that the .zip
            # locator takes. The variable's locator measures by the file system
            # of the directory the variable is set to. If the variable is
            # unset, it measures by the limit nearly every file system has. No
            # writable directory at a short path fixes a name that is too long,
            # so the remedy asks for none. If the .zip locator then raised for
            # a path with .zip in it and no part ending in .zip, the remedy
            # still gives the name, and does not ask for the variable alone
            # either.
            too_long = too_long_a_name(locations[named_after_the_source[0]])
            # Under the user-provided locator alone, with the variable unset,
            # numba took no location at all and never measured the name. That
            # locator takes nothing without the variable, and the name would
            # stop the locator once the variable is set, so the remedy names
            # both.
            variable = (
                ", with NUMBA_CACHE_DIR set, which UserProvidedCacheLocator takes nothing without"
                if not config.CACHE_DIR and all(reads_the_variable(cls) for cls in tried) else "")
            return (
                f"{too_long}: put the file in a directory of a shorter name{variable}{beside_the_source()}; or {silence}"
            )
        hashing = [
            cls for cls in tried if (reads_the_variable(cls) and config.CACHE_DIR) or caches_under_the_user_cache_dir(cls)]
        if subpath is None and (isinstance(failure, UnicodeEncodeError) or hashing):
            # numba hashes the source's directory path as UTF-8 for its
            # directory under NUMBA_CACHE_DIR or the user's cache directory,
            # and the locator that names its cache directory by that hash
            # raises. That locator is the variable's, if the variable is set,
            # before numba reaches the in-tree locator, which caches beside the
            # source and hashes no path. Or it is the user-wide locator, once
            # the in-tree one passed an unwritable __pycache__ over. With the
            # variable unset, the same install imports with caching on. The
            # remedy must therefore say more than that the variable fixes
            # nothing and a UTF-8 path is needed. Under a list without the
            # variable's locator, making the listed locations writable fixes
            # nothing either, because hashing refuses them however they are
            # set up. numba hashed the path if the error is its own
            # UnicodeEncodeError, or if numba tried a locator that names its
            # cache directory by that hash. Such a locator is the variable's,
            # when the variable is set, the user-wide one or the .zip one. The
            # in-tree locator alone hashes nothing, and its no-locator error
            # means that the __pycache__ was refused.
            hashed = (
                "numba hashes the source's directory as UTF-8 for the directory it caches this file in under "
                f"NUMBA_CACHE_DIR or the user's cache directory, and {os.path.dirname(py_file)} is not"
            )

            # The locators that hash the source's directory path are the
            # variable's locator, when the variable is set, the user-wide
            # locator, and the .zip locator for a path with .zip in it. The
            # in-tree locator caches beside the source, in its __pycache__,
            # and hashes no path. numba never reaches the in-tree locator if it
            # is listed after any locator that hashes. Making the __pycache__
            # writable, or listing the in-tree locator after the hashing one,
            # then fixes nothing: numba still raises. The remedy is to list the
            # in-tree locator before the first locator that hashes. If the
            # variable's locator is the only hashing one before it, the remedy
            # is to unset the variable. If the in-tree locator came first and
            # passed the file over, the remedy is to make the __pycache__
            # writable.
            def hashes(cls):
                return ((reads_the_variable(cls) and bool(config.CACHE_DIR)) or caches_under_the_user_cache_dir(cls)
                        or (one_of(cls, for_a_zip) and ".zip" in py_file))

            beside = os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__"))
            in_tree_at = next((at for at, cls in enumerate(order) if caches_beside_the_source(cls)), None)
            before_in_tree = [cls for cls in (order if in_tree_at is None else order[:in_tree_at]) if hashes(cls)]
            # Whether the in-tree locator takes the file, called as numba calls
            # it: the in-tree locator takes a file whose __pycache__ it can
            # write. Without that call, the remedy could only hedge with "where
            # that directory can be written", and would never ask for the
            # directory to be made writable. numba would then raise the same
            # error after the user unsets the variable as the remedy asks.
            if _takes(in_tree, py_file):
                caches_beside = f"numba caches beside the source, in its __pycache__, {beside}, hashing no path"
            else:
                caches_beside = (
                    f"make the __pycache__ beside the source, {beside}, writable, and numba caches there, hashing no "
                    "path")
            if in_tree_at is None:
                where = f" before {before_in_tree[0].__name__}" if before_in_tree else ""
                return (
                    f"{hashed}: put the file under a UTF-8 path, or list InTreeCacheLocator{where}, and "
                    f"{caches_beside}; or {silence}"
                )
            if before_in_tree and all(reads_the_variable(cls) for cls in before_in_tree):
                return (
                    f"{hashed}, and NUMBA_CACHE_DIR, which is set to {config.CACHE_DIR}, is hashed for first: unset "
                    f"it, and {caches_beside}; or put the file under a UTF-8 path; or {silence}"
                )
            if before_in_tree:
                return (
                    f"{hashed}, and NUMBA_CACHE_LOCATOR_CLASSES puts {before_in_tree[0].__name__}, which hashes it, "
                    f"before InTreeCacheLocator: list InTreeCacheLocator before it, and {caches_beside}; or put the "
                    f"file under a UTF-8 path; or {silence}"
                )
            return (
                f"{hashed}: put the file under a UTF-8 path, or make the __pycache__ beside the source, {beside}, "
                f"writable, where InTreeCacheLocator caches it hashing no path; or {silence}"
            )
        if not any(reads_the_variable(cls) for cls in tried):
            if zip_raised:
                return (
                    'numba\'s .zip locator takes this file for the ".zip" in its path and finds no archive there, after '
                    f"every locator NUMBA_CACHE_LOCATOR_CLASSES, {listed}, puts before it passed the file over: make "
                    "one of those locations writable, or list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, "
                    f"InTreeCacheLocator or UserWideCacheLocator before it, or {silence}"
                )
            return (
                f"numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, {listed}, says: make one of those locations "
                f"writable, or {silence}"
            )
        if config.CACHE_DIR:
            # The variable is set and numba passed it over: the directory it
            # points at is unwritable, or too deep for numba to make its own
            # directory there. A writable directory at the same depth would
            # not fix the depth.
            return (
                f"NUMBA_CACHE_DIR is set to {config.CACHE_DIR}, which numba could not use: set it to a writable "
                f"directory at a short path, or {silence}"
            )
        return f"Set NUMBA_CACHE_DIR to a writable directory, or {silence}"

    def the_source_files():
        # The remedy's two offers of source files, on disk or in a .zip. Each
        # names the locator it needs if NUMBA_CACHE_LOCATOR_CLASSES leaves
        # that locator out.
        on_disk = f"install {package} with its source files on disk, unpacked from any archive"
        if listed:
            # The install on disk caches through a locator for a file on disk
            # that the list names. If the list names no such locator, the
            # install caches no more than the archive does. If the list names
            # the user-provided locator alone, that locator takes nothing
            # without NUMBA_CACHE_DIR.
            for_a_file_on_disk = [
                cls for cls in _locators(listed)
                if any(one_of(cls, base) for base in (user_provided, in_tree, user_wide))
            ]
            if not for_a_file_on_disk:
                on_disk += (
                    ", and list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or "
                    f"UserWideCacheLocator in NUMBA_CACHE_LOCATOR_CLASSES, {listed}, which has none of them"
                )
            elif all(one_of(cls, user_provided) for cls in for_a_file_on_disk) and not config.CACHE_DIR:
                on_disk += (
                    ", with NUMBA_CACHE_DIR set, which UserProvidedCacheLocator, the one locator for a file on disk in "
                    f"NUMBA_CACHE_LOCATOR_CLASSES, {listed}, takes nothing without"
                )
        a_zip = (
            "import it from a .zip holding its source files, which numba caches in the user's cache "
            "directory"
        )
        found = _archive_above(py_file)
        if found is not None and not found[0].endswith(".zip"):
            # The file is already in an archive, such as APP.ZIP or an .egg.
            # numba's .zip locator never takes such an archive, because the
            # locator looks for a part of the path ending in ".zip",
            # lower-case. The remedy asks a user importing from such an
            # archive to rename it, and names the extension that numba does
            # not read. It does not offer a .zip, which the user already uses.
            # The remedy offers the rename even if a directory above the
            # archive has ".zip" in its name, which makes the locator take the
            # file and find no archive. numba treats the first part of the path
            # ending in .zip as the archive, so a directory so named above the
            # archive makes the rename insufficient. After the rename alone,
            # numba would still open the directory. The remedy therefore also
            # asks for the archive to be moved to a path outside that
            # directory.
            archive_parts = pathlib.Path(found[0]).parts
            in_the_way = next(
                (str(pathlib.Path(*archive_parts[:at + 1])) for at, part in enumerate(archive_parts[:-1])
                 if part.endswith(".zip")), None)
            moved = (
                f", and put it under a path with no part ending in .zip above it, {in_the_way} being one, which that "
                "locator takes for the archive first" if in_the_way else "")
            a_zip = (
                f"rename the archive, {found[0]}, to end in .zip, lower-case, which numba's .zip locator looks for in "
                f"the path, with its source files in it{moved}, so that numba caches it in the user's "
                "cache directory"
            )
        if listed_without(for_a_zip):
            # Only numba's .zip locator takes a .zip, so under a list without
            # it numba would have no locator for the offered .zip either.
            a_zip += f" through ZipCacheLocator, once NUMBA_CACHE_LOCATOR_CLASSES, {listed}, lists it"
        return on_disk, a_zip

    if isinstance(failure, KeyError):
        # numba 0.68 and later read a .zip member's source from the archive
        # for the stamp, by the name the module's code gives under the
        # archive's path. zipfile raises for a name the archive does not
        # hold: a module shipped as .pyc alone, or, on Windows under 0.68.0,
        # any member. The Windows case happens because that release builds
        # the name with the platform's separator (numba issue 10889, fixed in
        # 0.68.1). The remedy checks the archive for the member, so that it
        # does not ask for source files the archive holds. A backslash alone
        # does not identify the case either. A module shipped as .pyc alone
        # has no source under the name numba 0.68.1 reads, so that release
        # does not fix it.
        member = _missing_member(failure)
        held = member.replace("\\", "/")
        on_disk, a_zip = the_source_files()
        if os.name == "nt" and "\\" in member and _numbas_archive_holds(py_file, held):
            # The install on disk fixes the error now. A numba release with the
            # fix reads the member as the archive names it. No such release
            # was out as of 0.68.0, so the remedy names that release by the
            # fix, not as a release to install.
            return (
                f"numba 0.68.0 reads a .zip member's source by a name built with Windows's separator, {member}, which "
                "no archive holds (numba issue https://github.com/numba/numba/issues/10889, fixed for numba 0.68.1): "
                f"to cache, {on_disk}, which no separator touches, or a numba release that carries the fix, which "
                f"reads the member as the archive names it; or {silence}"
            )
        naming = _pyc_naming(py_file)
        if naming is not None and naming[1]:
            # The archive holds the module's source beside a .pyc compiled to
            # name another path in the archive. zipimport runs that .pyc, and
            # numba reads the source by the name the .pyc gives. The module is
            # not shipped as .pyc alone, so the remedy does not ask for source
            # files the archive already holds.
            compiled, source = naming[0], naming[0][:-1]
            return (
                f"numba 0.68 and later read a .zip member's source from the archive by the name the module's code "
                f"gives, {held}, which {compiled} was compiled to name, and this one holds none so named, while it "
                f"holds the module's source, {source}: to cache, compile the .pyc to name its path in the archive, "
                f"or leave it out, so that Python compiles the source and numba reads it under that name; or {silence}"
            )
        return (
            f"numba 0.68 and later read a .zip member's source from the archive, and this one holds none under "
            f"{held}, the module shipped as .pyc alone: to cache, {on_disk}, or {a_zip}; or {silence}"
        )
    # The locators numba reaches for a file not on disk, in numba's order.
    # numba's .zip locator takes the file without trying its location. A class
    # outside numba's families that takes the file when called with a probe
    # also ends the order, so numba reaches no locator listed after either.
    # numba's IPython and user-wide locators pass over a location they cannot
    # make or write in, and numba then tries the next locator.
    order = [cls for cls in _locators(listed) if may_take(cls)]
    zip_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip)), None)
    stops_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip) or not of_numbas(cls)), None)
    reached = order if stops_at is None else order[:stops_at + 1]
    # numba does not reach a .zip locator listed after such a class, so under
    # that list the locator's error for a path with no part ending in .zip
    # comes from a caller.
    zip_at = zip_at if zip_at == stops_at else None
    user_cache_dir = AppDirs(appname="numba", appauthor=False).user_cache_dir
    in_the_user_cache = os.path.join(user_cache_dir, subpath) if subpath else None

    def location_of(cls):
        # Where the class places the file. The IPython locator places it in
        # numba_cache under IPython's cache directory. The result is None
        # without IPython, or if IPython raises OSError for its directory, and
        # numba then passes the locator over. The remedy asks IPython for the
        # directory only if numba reaches a locator of that family, never for
        # any other file. If the remedy asked for any .zip member, an IPython
        # with its own directory unwritable would warn on every import of the
        # package and leave a temporary directory behind. The .zip and
        # user-wide locators place the file in a directory of their own under
        # the user's cache directory. A class outside numba's families has a
        # place of its own, which is None here.
        if one_of(cls, for_ipython):
            ipython_cache = ipython_numba_cache()
            return os.path.abspath(ipython_cache) if ipython_cache else None
        if not of_numbas(cls) or in_the_user_cache is None:
            return None
        return os.path.abspath(in_the_user_cache)

    locations = {cls: location_of(cls) for cls in reached}
    took_no_location = not any(locations.values()) and all(of_numbas(cls) for cls in reached)
    if getattr(sys, "frozen", False) and listed_without(user_wide) and (
            took_no_location or not isinstance(failure, (OSError, ValueError))):
        # The user-wide locator is the only one of numba's that takes a frozen
        # application, so under a list without it numba has no locator of its
        # own for one. This case covers numba's no-locator error, whatever
        # classes of the user's own the list holds, because those classes
        # passed the file over. It also covers any error a caller passes on
        # when the list has no place for the file. Asking for a writable
        # directory would name a directory that numba never tried.
        return (
            "numba caches a frozen application through UserWideCacheLocator alone, and "
            f"NUMBA_CACHE_LOCATOR_CLASSES, {listed}, leaves it out: list it, or {silence}"
        )
    # numba took no location under a list with no locator that takes the
    # file, or with none that has a place for it. The list is then the remedy
    # below for an error a caller passes on, as for numba's no-locator error
    # and as for a source on disk. A class outside numba's families has a
    # place of its own.
    parts = pathlib.Path(py_file).parts
    in_a_zip = any(part.endswith(".zip") for part in parts)
    # Whether a directory named .zip lies above the file. numba's .zip locator
    # still treats such a directory as an archive, and caches nothing under
    # it. A cell under such a directory, whose source linecache does not hold,
    # must not get the archive remedy, which asks for the source files to be
    # shipped.
    a_directory_named_zip = any(
        os.path.isdir(os.path.join(*parts[:at + 1])) for at, part in enumerate(parts) if part.endswith(".zip"))
    # Whether the file is a member of any archive, an .egg or .whl as well as
    # a .zip, which zipimport runs and whose source inspect reads through its
    # loader. The .zip is the one kind that numba's .zip locator takes, and the
    # only kind it caches.
    in_an_archive = (in_a_zip and not a_directory_named_zip) or _archive_above(py_file) is not None
    # A caller's error with no errno and no name looks like inspect's. The
    # remedy therefore recognizes inspect's errors by their messages, and
    # never reports a caller's error as inspect's.
    inspects = _is_inspects(failure) and any(one_of(cls, for_ipython) for cls in reached)
    if inspects and not in_an_archive:
        # inspect's error, with no errno and no name, for a cell whose source
        # linecache does not hold. numba's IPython locator reads a cell's
        # source from linecache, where IPython holds it while the cell runs.
        # inspect raises before the locator tries its location. numba therefore
        # took no location that the remedy could report as refusing a file. A
        # member of an archive that holds no source for it, a .pyc alone, gets
        # the archive remedy below for inspect's error. That remedy asks for
        # the source files to be shipped. The remedy finds the archive by testing
        # the files in the path, not by the archive's name, so a member of an
        # .egg gets that remedy too.
        return (
            f"numba's IPython locator reads a cell's source from linecache, which holds none for {py_file} "
            f"({failure}), and numba caches a cell only while IPython holds its source; or {silence}"
        )
    not_utf8 = _member_not_utf8(py_file) if inspects else None
    if not_utf8:
        # inspect's error for a member of an archive that holds its source,
        # when that source is not UTF-8. zipimport decodes a member's source as
        # UTF-8, and raises for one that is not. The probe's loader reads the
        # member as zipimport does, but returns None. The check therefore
        # raises inspect's error, which is a cache error, instead of the
        # decoding error. The decoding error is not a cache error, and it would
        # fail the import, as numba's would at decoration.
        return (
            f"numba's IPython locator reads the member's source as UTF-8, which {py_file} is not ({not_utf8}): ship "
            f"it as UTF-8; or {silence}"
        )
    hashed_by = next((cls for cls in reached if (one_of(cls, user_wide) and getattr(sys, "frozen", False))
                      or (one_of(cls, for_a_zip) and in_a_zip)), None)
    if isinstance(failure, UnicodeEncodeError) and hashed_by is not None:
        # numba's .zip locator names its directory under the user's cache
        # directory after the source's directory, with the archive's path in
        # it hashed as UTF-8. It raises for bytes that are not UTF-8, at
        # decoration and in the check alike. The user-wide locator does the
        # same for a frozen application's file. The error does not come from a
        # caller with no place for the file, so the remedy names the path, not
        # the .zip the user imports from or the source files on disk. The
        # first of the two locators that numba reaches is the one that raised.
        # The user-wide locator comes before the .zip locator in numba's order,
        # so for a frozen application under a directory named .zip, the remedy
        # names the application, not an archive.
        what, where = (("a .zip", "the archive") if one_of(hashed_by, for_a_zip)
                       else ("a frozen application", "the application"))
        return (
            f"numba caches {what} in the user's cache directory, and NUMBA_CACHE_DIR has no effect here, because the "
            "source is not a file on disk: numba hashes the source's directory as UTF-8 for the directory it caches "
            f"this file in there, and {os.path.dirname(py_file)} is not: put {where} under a UTF-8 path, or "
            f"{silence}"
        )
    if not took_no_location and not inspects and (isinstance(failure, OSError) or getattr(sys, "frozen", False)):
        if isinstance(failure, ValueError) and zip_at is not None:
            # numba's .zip locator takes a frozen application's file because of
            # the ".zip" in its path, without trying its location. It raises if
            # no part of the path ends in .zip, whether an archive or a
            # directory. If the .zip locator is listed before the user-wide
            # locator, the only one that takes the file, numba never reaches
            # the user-wide locator. The remedy is then to list the user-wide
            # locator before the .zip one. If the .zip locator is listed after
            # it, the user-wide locator passed its location over, unwritable or
            # too deep, and the remedy names that location, as for a source on
            # disk.
            if not any(one_of(cls, user_wide) for cls in order[:zip_at]):
                return (
                    f"numba's .zip locator, listed in NUMBA_CACHE_LOCATOR_CLASSES, {listed}, with no "
                    'UserWideCacheLocator before it, takes this file for the ".zip" in its path and finds no archive '
                    f"there: list UserWideCacheLocator, the one locator for a frozen application, before it, or {silence}"
                )
            return (
                'numba\'s .zip locator takes this file for the ".zip" in its path and finds no archive there, after '
                "UserWideCacheLocator, which numba tries before it, could not use the user's cache directory, "
                f"{user_cache_dir}: make it writable, or put it at a shorter path{_moved_through()}, or {silence}"
            )
        if filename:
            named = os.path.abspath(filename)
            named = {named, os.path.dirname(named)}
        else:
            named = set()
        # The error names the location numba took, or a file numba writes in
        # it, as the first save's error does. The locator that took the
        # location decides the remedy: the first locator reached whose place
        # the error names. The frozen application's no-locator error names
        # nothing.
        taken = next((cls for cls in reached if locations[cls] and locations[cls] in named), None)
        if taken is not None and one_of(taken, for_ipython):
            # numba caches the member in numba_cache under IPython's cache
            # directory. The remedy is the one for a cell file on disk in that
            # location, without NUMBA_CACHE_DIR, which has no effect for a
            # source not on disk.
            ipython_cache = locations[taken]
            too_long_file = too_long_a_file_name(ipython_cache)
            if too_long_file:
                return f"{too_long_file[0]}, and no shorter path cures it: {a_shorter_name}; or {silence}"
            too_long_for_the_names = a_name_limit_below(ipython_cache)
            if too_long_for_the_names:
                return f"{too_long_for_the_names}, so no shorter path cures it; or {silence}"
            if failure.errno == errno.ENAMETOOLONG:
                return (
                    f"the path is too long for the file system: IPython's cache directory, "
                    f"{os.path.dirname(ipython_cache)}, at a shorter path; or {silence}"
                )
            reason = failure.strerror or failure
            return (
                f"numba caches this file in {ipython_cache}, where no file can be written ({reason}): make room "
                f"there, or make it writable; or {silence}"
            )
        # The user can act on the .zip's location, a directory of numba's own
        # under the user's cache directory. The error names that location, or
        # the location is the directory of the file the error names. For an
        # error that names no file, the remedy lists the locations numba could
        # have taken, IPython's numba_cache among them if numba reaches a
        # locator of that family. The frozen application's no-locator error
        # names nothing, and the remedy names the user's cache directory.
        if taken is not None:
            location = filename if os.path.abspath(filename) == locations[taken] else os.path.dirname(filename)
        else:
            location = filename or user_cache_dir
        # Whether the remedy says where numba caches the file. A .zip's or a
        # frozen application's location is under the user's cache directory. A
        # location that is none of numba's is not under it. numba can reach a
        # locator of IPython's family, or a class outside numba's families, for
        # a member. An error that names no file may then come from that
        # locator, whether or not IPython gave its directory. The remedy
        # therefore says nothing of the user's cache directory in any of these
        # cases.
        in_the_user_cache_alone = taken is not None or (
            not filename and all(of_numbas(cls) and not one_of(cls, for_ipython) for cls in reached))
        where = (
            "numba caches a .zip, or a frozen application, in the user's cache directory, and "
            if in_the_user_cache_alone else "")
        opening = f"{where}NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: "
        if isinstance(failure, FileNotFoundError) and names_a_path_above():
            # The error comes from the stamp that numba reads at decoration,
            # for the archive the code names: .pyc members compiled to name an
            # archive that has since moved. A location that cannot be made,
            # under a dangling link, also gives ENOENT, but it does not lie
            # above the code's file.
            suggestion = "" if numba_version >= 68 else "compile the archive's .pyc members to name its path now, or "
            return (
                f"the module's code names {filename}, which is not there: its .pyc was compiled to name "
                f"that path, so {suggestion}ship its source files; "
                f"or {silence}"
            )
        if (isinstance(failure, OSError) and failure.errno == errno.EISDIR and filename and os.path.isdir(filename)
                and filename.endswith(".zip") and names_a_path_above()):
            # numba treats the first part of the path ending in .zip as the
            # archive, including a directory so named above the file. numba
            # 0.68 and later open that archive to read the member's source,
            # and a directory refuses that. The directory is not a cache
            # location, so the remedy does not ask for free space in it.
            return (
                f"{opening}numba takes the first part of the path ending in .zip, {filename}, for the archive, and it "
                "is a directory, which numba 0.68 and later open to read the member's source: put the archive under "
                f"a path with no such directory above it, or {silence}"
            )
        if isinstance(failure, OSError) and names_a_path_above():
            # The archive that the member's code names cannot be read. numba
            # reads it for the stamp, and from 0.68 opens it for the member's
            # source. The remedy reports the archive as unreadable, because
            # making it writable, as for a location that refused a file, would
            # not make it readable.
            reason = failure.strerror or failure
            return (
                f"{opening}numba reads the archive, {filename}, for the member's stamp, and from 0.68 for the "
                f"member's source, and cannot ({reason}): make it readable, or {silence}"
            )
        if isinstance(failure, OSError) and filename and taken is None:
            # A location that is none of numba's, under the list and in its
            # order, which an error a caller passes on can name. The remedy
            # names this location as the error names it, as for a source on
            # disk. It says nothing of what moves the user's cache directory,
            # because moving that directory does not move this location. It
            # says nothing of where numba caches the file either, because this
            # location is none of numba's. One such case is a member that
            # numba's IPython locator takes while IPython is not importable;
            # numba does not cache that member at all.
            if failure.errno == errno.ENAMETOOLONG:
                return (
                    f"{opening}the path is too long for the file system, so put that location, {filename}, at a "
                    f"shorter path, or {silence}"
                )
            reason = failure.strerror or failure
            return (
                f"{opening}no file can be written at that location, {filename} ({reason}), so make room there, or "
                f"make it writable, or {silence}"
            )
        too_long = too_long_a_name(locations[taken]) if isinstance(failure, OSError) and taken is not None else None
        if too_long:
            # The system refuses the .zip's directory under the user's cache
            # directory, named after the source's directory, whatever the
            # errno; on Windows the errno is EINVAL. The user's cache directory
            # at a shorter path does not fix a name that is too long, so the
            # remedy does not ask for a shorter path.
            return f"{opening}{too_long}, so put the file in a directory of a shorter name, or {silence}"
        too_long_file = (
            too_long_a_file_name(locations[taken]) if isinstance(failure, OSError) and taken is not None else None)
        if too_long_file:
            return f"{opening}{too_long_file[0]}, and no shorter path cures it: {a_shorter_name}, or {silence}"
        too_long_for_the_names = (
            a_name_limit_below(locations[taken]) if isinstance(failure, OSError) and taken is not None else None)
        if too_long_for_the_names:
            return f"{opening}{too_long_for_the_names}, so no shorter path cures it, or {silence}"
        if isinstance(failure, OSError) and failure.errno == errno.ENAMETOOLONG:
            if taken is None:
                # The error names no file, so the remedy cannot tell which of
                # the locations is too long. Moving the user's cache directory
                # moves them all if numba reaches no locator of IPython's
                # family.
                moved_through = _moved_through() if in_the_user_cache_alone else ""
                return (
                    f"{opening}the path is too long for the file system, so put {could_have_taken(locations)}, at a "
                    f"shorter path{moved_through}, or {silence}"
                )
            return (
                f"{opening}the path is too long for the file system, so put that directory, {location}, at a shorter "
                f"path{_moved_through()}, or {silence}"
            )
        if isinstance(failure, OSError):
            # The location refused a file for another reason, a full disk or
            # permissions, which the error gives. The remedy repeats the
            # reason, as it does for a source on disk. The frozen application's
            # no-locator error gives no reason, because numba passed the
            # location over on an error of its own.
            reason = failure.strerror or failure
            if taken is None:
                return (
                    f"{opening}no file can be written in {could_have_taken(locations)} ({reason}), so make room "
                    f"there, or make it writable, or {silence}"
                )
            # numba's .zip locator takes the file without making its location.
            # The first save or the check makes the location under the first
            # directory above it that exists, and that directory refused. The
            # remedy therefore names that directory. Asking for free space in a
            # directory that does not exist, or for it to be made writable,
            # would name the wrong directory. An error naming a file in the
            # location means that the location exists.
            made_under = ""
            if os.path.abspath(filename) == locations[taken] and not os.path.isdir(location):
                above = _existing_above(location)
                if not os.path.isdir(above):
                    # A file is in the path. The remedy names it, because the
                    # directory above it is writable, and free space there
                    # would fix nothing.
                    return (
                        f"{opening}no file can be written in that directory, {location}, which cannot be made, "
                        f"{above} being a file in its path ({reason}), so put that file elsewhere, or {silence}"
                    )
                made_under = f", which cannot be made under {above}"
            return (
                f"{opening}no file can be written in that directory, {location}{made_under} ({reason}), so make room "
                f"there, or make it writable, or {silence}"
            )
        # numba's no-locator error for a frozen application gives no reason,
        # because numba passed the user-wide location over on an error of its
        # own. The location was an unwritable directory, one too deep to make,
        # or one whose name, after the source's directory, is too long for the
        # file system. The remedy reports the last case as such, because no
        # shorter path fixes it.
        too_long = too_long_a_name(in_the_user_cache)
        if too_long:
            return f"{opening}{too_long}, so put the file in a directory of a shorter name, or {silence}"
        return (
            f"{opening}numba could not use that directory, {location}: make it writable, or put it at a shorter "
            f"path{_moved_through()}, or {silence}"
        )
    of_ipythons = [cls for cls in reached if one_of(cls, for_ipython)]
    if not isinstance(failure, (OSError, ValueError)) and of_ipythons and all(
            one_of(cls, for_ipython) or not of_numbas(cls) for cls in reached):
        # numba's no-locator error for a file that only numba's IPython locator
        # takes: a cell, or a .zip member in an ipykernel directory under a
        # list with no .zip locator after the IPython one. The locator passed
        # its location over, as numba's user-wide locator passes over a frozen
        # application's location. The location was unwritable or too deep to
        # make, or IPython raised for its directory and gave none. A class
        # outside numba's families listed with the IPython locator passed the
        # file over too, as numba's no-locator error shows, and the remedy
        # names it as such. Once called with a probe, such a class is no
        # longer in the order. If the remedy took it for one of numba's
        # locators, it would answer with the archive remedy.
        ipython_cache = locations[of_ipythons[0]]
        passed_over = [cls.__name__ for cls in _locators(listed) if not of_numbas(cls)]
        passed_over = f", and {' and '.join(passed_over)}, listed too, passed the file over" if passed_over else ""
        if ipython_cache:
            return (
                f"numba caches this file in {ipython_cache}, which it could not make or write in{passed_over}: make "
                f"it writable, or put IPython's cache directory, {os.path.dirname(ipython_cache)}, at a shorter path; "
                f"or {silence}"
            )
        return (
            "numba caches this file in numba_cache under IPython's cache directory, which IPython raised for, under a "
            f"home it cannot write{passed_over}: make that home writable; or {silence}"
        )
    if _taken_by_ipython(py_file) and not in_a_zip and listed_without(for_ipython):
        # Only numba's IPython locator takes a cell, or a file in an ipykernel
        # directory that is not on disk, so under a list without it numba has
        # no locator for one. The archive remedy would offer source files on
        # disk and locators that take none of these files.
        return (
            f"numba caches a notebook cell through IPythonCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, {listed}, "
            f"leaves it out: list it; or {silence}"
        )
    if in_a_zip and not a_directory_named_zip and listed_without(for_a_zip):
        # Only numba's .zip locator takes a source in a .zip, so under a list
        # without it numba has no locator for one. The archive remedy would
        # offer the .zip that the user already imports from. A source that does
        # not exist, under a directory named .zip, is left out, because that
        # locator treats the directory as an archive and caches nothing.
        return (
            f"numba caches a .zip through ZipCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, {listed}, leaves "
            f"it out: list it; or {silence}"
        )
    on_disk, a_zip = the_source_files()
    return (
        f"NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, {on_disk}, or "
        f"{a_zip}; or {silence}"
    )


def uncached_where_no_cache_can_be_written(options):
    """``options`` as given, or with ``cache`` off, after one warning, if numba can write no cache for numbox.

    numba sets a cached function up when the function is decorated, and raises ``RuntimeError`` there if no cache
    location can be written. That happens in a read-only install whose user cache directory cannot be written either,
    and on an import from an ``.egg``, ``.whl`` or ``.pyz`` archive, which Spark's ``--py-files`` ships. numba takes the
    user's cache directory for a ``.zip`` without checking at decoration that it can be written, and the first write
    raises ``OSError`` instead. Either way the import would fail at the first decorated function, with no message that
    names the remedy.

    Every function numbox caches is decorated under the one ``jit_options``, so this function runs the check once,
    for a source file in each directory of the package that holds a module. The answer holds for the whole package.
    If numba can cache no function of one of those files, the function returns the options with ``cache`` off, and
    one warning names the remedy that ``cache_remedy`` gives. ``NUMBOX_JIT_OPTIONS='{"cache": false}'`` turns caching
    off and silences the warning. An error that ``is_a_cache_error`` does not count as the cache's is raised
    unchanged.

    The remedy depends on the source. For a source file on disk, it is ``NUMBA_CACHE_DIR``, and, if the location is too
    long for the file system, whatever shortens the location numba took. For a ``.zip``, the remedy is one of three. The
    first is the location numba took under the user's cache directory, made writable or given free space, with the
    reason. The second is that directory at a shorter path, if the path is too long. The third is the ``.zip`` locator
    listed, if ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves that locator out. numba reads ``NUMBA_CACHE_DIR`` only for a
    source file on disk.
    The remedy for any other archive, a ``.pyc``-only install or a ``.pyc`` in a ``.zip`` is the source
    files on disk, or a ``.zip`` holding them, which numba caches in the user's cache directory. The
    same remedy applies, from numba 0.68, to a ``.zip`` member whose archive holds no source for it, a ``.pyc`` alone,
    because numba then reads the member's source from the archive for the stamp. The remedy names the member. On
    Windows, numba 0.68.0 builds the name with the platform's separator (numba issue 10889), so every member is such a
    member. The remedy there is the install on disk, or a numba release that contains the fix.

    A ``.zip`` whose cache directory holds every entry but can no longer be written falls back too, although numba
    alone would have loaded the entries. numba applies the writability check to every other placement, and its
    ``.zip`` locator lacks it, as reported in https://github.com/numba/numba/issues/10888.

    ``NUMBOX_JIT_OPTIONS`` can give options without a ``cache`` key, and ``njit`` then uses its default, which is off.
    The sqlite virtual-table and table-valued-function callbacks cache under those options, however. This function
    therefore checks such options too, and returns them with ``cache`` off, explicitly, if numba can write no cache.
    """
    if not options.get("cache", True):
        return options
    for py_file in _module_files():
        try:
            check_cache_location(py_file, LONGEST_CACHE_FILE_NAME)
        except (RuntimeError, OSError, KeyError, ValueError) as error:
            if not is_a_cache_error(error):
                raise
            failure = error
            break
    else:
        return options
    silence = "NUMBOX_JIT_OPTIONS='{\"cache\": false}' to turn caching off and silence this warning"
    remedy = cache_remedy(py_file, failure, silence)
    warnings.warn(
        f"numba cannot cache numbox here ({failure}); it compiles without a cache. {remedy}",
        RuntimeWarning, stacklevel=2,
    )
    return {**options, "cache": False}


jit_options = uncached_where_no_cache_can_be_written(get_jit_options())


_PROXY_CACHE_STRICT_ENV = "NUMBOX_PROXY_CACHE_STRICT"


def _strict_cache_mode():
    """True when ``NUMBOX_PROXY_CACHE_STRICT`` selects strict validation of ``@proxy`` cache loads.

    In strict mode, every fail-open path in the ``numbox.core.proxy.proxy`` cache guard raises instead. A payload that
    cannot be read, or an object whose container cannot be parsed, aborts the load with ``UnvalidatedProxyCacheError``
    instead of loading unchecked. A detected stale alias raises ``StaleProxyCacheError`` before the guard repairs it,
    and the stale entry stays on disk. Strict mode is a debugging aid, off by default.

    Unlike ``jit_options`` above, the value is read on each call rather than once at import, so the setting can be
    changed within a process. The cost is one environment lookup, against a cache load of several milliseconds. Any
    value outside the unset, empty, ``0``, ``false``, ``no`` and ``off`` set (case-insensitive) enables strict mode.
    """
    value = os.environ.get(_PROXY_CACHE_STRICT_ENV)
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


MAX_STR_LENGTH = 2 ** 31 - 1

#: numba's `FunctionModel`, resolved once. A data model lookup is type machinery and compiles
#: nothing, so the layout can be read here. `numbox.utils.lowlevel` and `numbox.utils.derive_wap`
#: both need it on paths that must stay free of compilation side effects, and importing
#: `numbox.utils.lowlevel` compiles a cached eager `@njit` helper.
_function_model = default_manager.lookup(FunctionType(void()))

#: Slots in the `FunctionModel` struct: (c_addr, py_addr, jit_addr)
function_struct_size = _function_model.field_count
