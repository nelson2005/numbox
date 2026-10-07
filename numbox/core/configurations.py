import errno
import importlib.machinery
import inspect
import os
import pathlib
import json
import linecache
import sys
import tempfile
import warnings
import zipfile
import zipimport

from importlib.metadata import version

import numba.experimental.function_type  # noqa: F401  registers `FunctionModel` against `FunctionType`
from numba.core.caching import CompileResultCacheImpl
from numba.core.datamodel import default_manager
from numba.misc.appdirs import AppDirs
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
    if "cache" in as_json and not isinstance(as_json["cache"], bool):
        # numba reads the option's truth, so the string "false" turned caching on.
        raise ValueError('NUMBOX_JIT_OPTIONS "cache" must be true or false, e.g. {"cache": false}')
    return as_json


def _cache_probe():
    """A function of this module, never compiled: its code's file is the one numba looks up for the module."""


# The longest file numba writes for a function of numbox's own files, with room to spare: the module's stem,
# the function's qualified name, numba's line number, interpreter tag and index number, and the 21 bytes of the
# temporary name it is written under. The longest-named function of the package comes to 105 bytes and the
# builder's generated kernel, anchored to its file, to 117; a test holds every function of the package under
# this bound.
LONGEST_CACHE_FILE_NAME = 128


def check_cache_location(py_file, longest_file_name=0):
    """Raise as numba would where a function whose source is ``py_file`` cannot be cached; else return.

    The question is put the way numba puts it: the cache set-up that decoration runs, which picks the location
    for the file or raises ``RuntimeError`` with no locator, then the source's stamp, which decoration reads
    and which stats the archive for a ``.zip``, then the writability check, which decoration runs for every
    location but a ``.zip``'s and the first save runs for all, raising ``OSError``. The probe is compiled with
    ``py_file`` as its file, which is all a locator reads of it but for numba's IPython locator, which reads the
    function's source too: the probe's is given it under the file's name while the locator is picked, the probe
    having no module for inspect to read a ``.zip`` member's source through. Nothing is compiled, and nothing is
    written but the cache directory itself. An ``OSError`` from the check names the location numba picked.

    numba's writability check makes a temporary file whose name is short, or none at all on Linux, and the
    files it saves have names of up to a hundred bytes and more, so a location within their length of the path
    limit passes the check and the first save dies. Given ``longest_file_name``, a file with a name that long
    is made and removed in the location too, and the ``OSError`` is the location's: the package asks with
    ``LONGEST_CACHE_FILE_NAME``, the bound on numba's names for its own files.
    """
    source = "def _cache_probe():\n    pass\n"
    namespace = {}
    exec(compile(source, os.fspath(py_file), "exec"), namespace)  # nosec B102 - fixed source
    probe = namespace["_cache_probe"]
    # numba's IPython locator reads the function's source when it takes the
    # file, which inspect reads from the file, or through the module's loader
    # where the file is not on disk, a member of a .zip; the probe has no
    # module, so its source is put where inspect reads first, under the
    # file's name, while numba picks the locator, and what was there put back.
    file = probe.__code__.co_filename
    was_cached = linecache.cache.get(file)
    linecache.cache[file] = (len(source), None, source.splitlines(True), file)
    try:
        locator = CompileResultCacheImpl(probe).locator
    finally:
        if was_cached is None:
            linecache.cache.pop(file, None)
        else:
            linecache.cache[file] = was_cached
    # numba reads the source's stamp at decoration too, the archive's for a
    # .zip, which is not there where the code names an archive since moved.
    locator.get_source_stamp()
    try:
        locator.ensure_cache_path()
        if longest_file_name:
            # tempfile's name is the prefix and eight characters of its own.
            with tempfile.NamedTemporaryFile(dir=locator.get_cache_path(), prefix="x" * (longest_file_name - 8)):
                pass
    except OSError as error:
        # The error names what failed, a temporary file's name among the
        # possibilities; the location it is in is what a reader can act on.
        raise OSError(error.errno, error.strerror, locator.get_cache_path()) from error


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
    is never wrong. And numba finds a location for a module by its code's file, so a module that survives as
    ``.pyc`` alone, beside sourced ones, answers for itself, by the file it was compiled from, which its code
    keeps: the ``.py`` that is gone where it was compiled in place, or a tree elsewhere, on disk or not. The
    package is found by this module's ``__file__``, which is where it was imported from, an archive or a
    directory: its code's file, which numba looks up for the probe and which is asked first, can be elsewhere
    the same way. An archive shows no directories to walk, so there its members are listed instead.
    """
    yield inspect.getfile(_cache_probe)
    package = os.path.dirname(os.path.dirname(__file__))
    if not os.path.isdir(package):
        yield from _archived_module_files(__file__)
        return
    walked = set()
    for directory, subdirectories, files in os.walk(package, followlinks=True):
        # A symlink is followed, and a directory reached twice through links,
        # a cycle among them, is walked once.
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

    The import system runs such a ``.pyc`` through its sourceless loader, which keeps the compile-time file on
    the code, and fails the module's import on one it cannot run, of another interpreter's magic number or
    unreadable; so the loader's answer is read, and none leaves nothing to ask.
    """
    loader = importlib.machinery.SourcelessFileLoader(os.path.basename(pyc_path)[:-4], pyc_path)
    try:
        return loader.get_code(loader.name).co_filename
    except Exception:
        return None


def _archived_module_files(own):
    """One module per directory of numbox inside the ``.zip`` that holds ``own``, this module as imported, and the
    file each ``.pyc`` in it that zipimport would run was compiled from; nothing for any other archive.

    numba caches a ``.zip`` per directory of it, each in a location of its own under the user's cache directory,
    so the directories answer separately there too. A ``.pyc`` run from the archive keeps the file it was compiled
    from as its code's file, and that is what numba looks up for its functions, the archive's own path for the
    module is not; so such a member asks by that file, there or gone. The archive is the first part of the path
    named ``.zip`` that is one, a directory so named above it being no archive. Any other archive has no location
    at all, and the probe's own file has already asked.
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
        else:
            member = os.path.join(zip_path, *name.split("/"))
        if directory not in seen:
            seen.add(directory)
            yield member


def _compiled_from(zip_path, directory, stem):
    """The file the code zipimport runs module ``stem`` of ``directory`` from names, or None where it runs none.

    zipimport decides: it runs the ``.pyc`` before the ``.py`` beside it, but not one of another interpreter's
    magic number, nor one stale against that ``.py``, which it passes over for the source, nor one it cannot
    unmarshal, and the module's import fails with it; so its answer, its code, is read. The file that names is
    the archive's own path for the module where zipimport compiled the ``.py``, or where the ``.pyc`` was
    compiled to name the archive, and stands for the directory as a ``.py`` member does; else it is the file
    elsewhere the ``.pyc`` was compiled from, there or gone, and asks for itself. No code at all, for whatever
    stops zipimport, a source it cannot compile or a member it cannot decompress among the reasons, leaves
    nothing to ask.
    """
    if stem == "__init__":
        directory, _, stem = directory.rpartition("/")
    importer = zipimport.zipimporter(os.path.join(zip_path, *directory.split("/")))
    try:
        return importer.get_code(stem).co_filename
    except Exception:
        return None


def _moved_through():
    """What moves numba's user cache directory on this platform, as a clause for a remedy; nothing on Windows.

    numba's ``appdirs`` asks the system for it on Windows, which no variable changes, puts it under ``HOME`` on
    macOS, and takes ``XDG_CACHE_HOME``, else a directory under ``HOME``, elsewhere.
    """
    if sys.platform == "win32":
        return ""
    if sys.platform == "darwin":
        return ", through HOME"
    return ", through XDG_CACHE_HOME or HOME"


def _class_named(module, name):
    """The class of this name in ``module``, as numba spells its locators from 0.62 or with the underscore it put
    before them until then; None where the module has neither."""
    return getattr(module, name, None) or getattr(module, "_" + name, None)


def _numba_locator(name):
    """numba's cache locator class of this name, or None where this numba has none: the ``.zip`` locator arrived in
    0.61."""
    from numba.core import caching
    return _class_named(caching, name)


def _locators(listed):
    """numba's cache locator classes in the order it tries them.

    ``listed`` is ``NUMBA_CACHE_LOCATOR_CLASSES`` as numba 0.62 and later read it, each entry a class of numba's
    caching module by its name or its dotted path, resolved as numba resolves it; an entry numba could not resolve
    it has refused already, before any location was tried, and is left out here. Empty, the order is numba's own.
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
                cls = _class_named(importlib.import_module(module_path), class_name)
            except ImportError:
                cls = None
        else:
            cls = _numba_locator(entry)
        if cls is not None:
            classes.append(cls)
    return classes


def _ipython_numba_cache():
    """numba's cache location for a cell file its IPython locator takes: ``numba_cache`` under IPython's cache
    directory, with no directory per file; None where IPython is not importable, which numba's locator raises
    ImportError for at decoration, no cache error, so the location is then none of numba's."""
    try:
        try:
            from IPython.paths import get_ipython_cache_dir
        except ImportError:
            from IPython.utils.path import get_ipython_cache_dir
    except ImportError:
        return None
    return os.path.join(get_ipython_cache_dir(), "numba_cache")


def cache_remedy(py_file, failure, silence, package="numbox"):
    """The remedy for ``failure``, numba's for a function whose file is ``py_file``, ending in ``silence``.

    ``NUMBA_CACHE_DIR`` for a source file on disk, which is the only kind numba reads the variable for; where it is
    set and numba passed it over, the variable named and a writable directory at a short path asked for, since
    numba passes over one too deep to make its directory in as it does an unwritable one; and where
    ``NUMBA_CACHE_LOCATOR_CLASSES``, from numba 0.62, leaves the user-provided locator out, the locations it names
    made writable, numba never reading the variable. Where the
    location is too long for the file system, the error names it, and the remedy is for the one it is: numba takes
    a directory under ``NUMBA_CACHE_DIR`` where that is set, else the ``__pycache__`` beside the source, else a
    directory under the user's cache directory, each of the two under a cache directory named for the source's
    directory, by its name and a hash of its path; the path the error names is matched whole against each, in the
    order numba tries them, since the three can nest and ``NUMBA_CACHE_DIR`` set to the user's cache directory makes
    one path of two. So a shorter ``NUMBA_CACHE_DIR`` for the first; the
    package at a shorter path for the second; the user's cache directory at a shorter path, through
    ``XDG_CACHE_HOME`` or ``HOME``, ``HOME`` alone on macOS and nothing on Windows, for the third; and for either
    of the last two ``NUMBA_CACHE_DIR`` set to a short path, where numba tries it before the locator that took the
    location, or, where it is set and numba passed it over, named and made a writable directory at a short path. A
    location that is none of numba's is named as the error names it, with the variable, where numba tries it before
    any other locator, named as set or asked for at a short path, and nothing said of numba passing it over, which
    that location cannot show. An error that names no file, or an empty name, which a caller can pass on where
    numba's and the package's own name the file refused, is told the locations numba could have taken, in numba's
    order and each once, to put the one that is too long at a shorter path.
    ``NUMBA_CACHE_LOCATOR_CLASSES`` decides that order, each entry a class of numba's caching module or, by its dotted
    path, a subclass of one, which takes what its parent takes, caches where it does and is told as it is. numba's
    IPython locator caches a cell file it takes in ``numba_cache`` under IPython's cache directory, with no directory
    per file, and that location too long is told as IPython's cache directory at a shorter path, for a cell file on
    disk and for a member of a ``.zip`` in an ipykernel directory alike, which that locator takes before the ``.zip``
    one is reached, with no ``NUMBA_CACHE_DIR`` offered for a source not on disk; numba imports
    IPython for the location when it makes it and catches only OSError there, so where IPython is not importable it
    raises ImportError at decoration for a file that locator takes, no cache error, and an error a caller passes on
    naming that location is told it as none of numba's. numba's IPython locator takes a file on disk only in
    an ipykernel directory and its ``.zip`` locator only a path with ``.zip`` in it, so either ahead of the rest
    changes nothing for any other file; the ``.zip`` locator ahead of them all takes such a path first and caches it
    under the user's cache directory where a part of the path ends in ``.zip``, an archive or a directory, or, where
    none does, finds no archive in it, and the remedy
    then asks for a locator for a file on disk listed before it, the user-provided one with ``NUMBA_CACHE_DIR`` set,
    which alone it takes nothing without; after some of them, it finds no archive once those have passed the file
    over, numba trying none after it, and the remedy asks for one of their locations made writable or such a
    locator listed before it. A list with no locator that takes the file is told so, for an error a caller
    passes on under it as for numba's no-locator one, and so is one whose only locator for the file is the
    user-provided one with ``NUMBA_CACHE_DIR`` unset, which takes nothing without it: the variable to set, as
    numba's no-locator error is told there. Where the location numba took refuses a file for another reason, a
    full disk or permissions changed since numba's own check, the remedy names the location and the reason and asks for room
    or a writable directory there, with ``NUMBA_CACHE_DIR`` as the alternative where numba tries it before that
    location, or set to another directory where the location is the variable's own; numba passes over a location it
    cannot make or write in, so only its no-locator error means the variable was passed over. The error can name the
    location or a file numba writes in it, as a string, bytes or a path object, read as a string; one that names a
    location that is none of numba's is told that location as the error names it, and one that names no file, a
    caller's, the locations numba could have taken. For a ``.zip`` or
    a frozen application, both cached under the user's cache directory, the location made writable or room made
    there, with the reason the error gives, a full disk or permissions: the ``.zip``'s error names it, a directory of
    numba's under the user's cache directory, or a file numba writes in it, as the first save's does, and the
    directory is told; the frozen application's is the no-locator one, which gives no reason, numba having passed
    the location over on its error, so the user's cache directory is named, to be made writable, as it is for an
    error a caller passes on that names no file. For any other archive, or a
    module without its source, the source files on disk or a ``.zip`` holding them, each with the locator it needs
    listed where ``NUMBA_CACHE_LOCATOR_CLASSES`` leaves it out: one of the three for a file on disk for the first,
    the ``.zip`` locator for the second, and ``NUMBA_CACHE_DIR`` set where the user-provided locator is the one
    listed for a file on disk, which takes nothing without it. A ``.zip``'s location too long for the file system is
    the user's cache directory's doing, with the names numba makes bounded, and the remedy is that directory at a
    shorter path, through what moves it on the platform. numba caches a ``.zip`` through its ``.zip`` locator alone, so a
    ``NUMBA_CACHE_LOCATOR_CLASSES`` that leaves that locator out gives numba no locator for a source in a ``.zip``,
    and the remedy is to list it; so for a frozen application and numba's user-wide locator, which alone takes one.

    ``package`` is the one the remedy tells the reader to install again, at a shorter path or with its source
    files on disk: a package built on numbox that puts the question for its own files with
    ``check_cache_location`` and ``is_a_cache_error`` passes its own name, as it passes ``check_cache_location`` a
    ``longest_file_name`` for its own functions, ``LONGEST_CACHE_FILE_NAME`` being numbox's.
    """
    from numba import config
    from numba.core.caching import _CacheLocator
    # check_cache_location takes any path-like, bytes too, and so does this:
    # the str the remedy reads, which os.fspath would leave bytes as bytes.
    py_file = os.fsdecode(py_file)
    # The error names the file refused as the call that raised it was given
    # it, bytes for one given bytes, or as a caller built it, a path object
    # too, and the remedy reads the name as str, as it reads the source's.
    filename = getattr(failure, "filename", None)
    filename = os.fsdecode(filename) if filename else None
    listed = getattr(config, "CACHE_LOCATOR_CLASSES", "")
    user_provided = _numba_locator("UserProvidedCacheLocator")
    in_tree = _numba_locator("InTreeCacheLocator")
    user_wide = _numba_locator("UserWideCacheLocator")
    for_ipython = _numba_locator("IPythonCacheLocator")
    for_a_zip = _numba_locator("ZipCacheLocator")

    def one_of(cls, base):
        # Whether ``cls`` is ``base`` or a subclass of it: a subclass, listed
        # by its dotted path, takes what its parent takes and caches where it
        # does, as numba's InTreeCacheLocatorFsAgnostic does, so each of
        # numba's locators stands for its family.
        return base is not None and issubclass(cls, base)

    def listed_without(base):
        # Whether NUMBA_CACHE_LOCATOR_CLASSES is set and names no class of
        # ``base``'s family.
        return base is not None and bool(listed) and not any(one_of(cls, base) for cls in _locators(listed))

    if os.path.exists(py_file):
        def may_take(cls):
            # numba's IPython locator takes a file on disk only in an ipykernel
            # directory, and its .zip locator only a path with .zip in it, which
            # it caches where a part ends in .zip, an archive or a directory,
            # and raises for where none does; every other class may take any.
            if one_of(cls, for_ipython):
                return os.path.basename(os.path.dirname(py_file)).startswith("ipykernel_")
            if one_of(cls, for_a_zip):
                return ".zip" in py_file
            return True

        def reads_the_variable(cls):
            return one_of(cls, user_provided)

        def caches_beside_the_source(cls):
            return one_of(cls, in_tree)

        def caches_under_the_user_cache_dir(cls):
            return one_of(cls, user_wide) or one_of(cls, for_a_zip)

        order = [cls for cls in _locators(listed) if may_take(cls)]
        variable_at = next((at for at, cls in enumerate(order) if reads_the_variable(cls)), None)
        cache_dir_read = variable_at is not None

        def instead(taken, asks="a short path"):
            # NUMBA_CACHE_DIR as the alternative to the location numba took,
            # offered where numba tries the variable before that location's
            # locator: set, it was passed over, unwritable or too deep; unset,
            # it would be taken, and ``asks`` is what it is asked to be. None
            # is a location that is none of numba's, taken for the first in
            # the order other than the variable's own; it shows nothing of
            # what numba did with the variable, so a set one is named without
            # the claim that numba passed it over.
            if taken in order:
                position = order.index(taken)
                passed_over = " and numba could not use"
            else:
                position = next((at for at, cls in enumerate(order) if not reads_the_variable(cls)), len(order))
                passed_over = ""
            if not cache_dir_read or variable_at >= position:
                return ""
            if config.CACHE_DIR:
                return (
                    f", or NUMBA_CACHE_DIR, which is set to {config.CACHE_DIR}{passed_over}, made a writable "
                    "directory at a short path"
                )
            return f", or NUMBA_CACHE_DIR set to {asks}"

        subpath = _CacheLocator.get_suitable_cache_subpath(py_file)
        user_cache_dir = AppDirs(appname="numba", appauthor=False).user_cache_dir

        def location_of(cls):
            # Where the class places the file: the .zip locator under the user's
            # cache directory like the user-wide one. None for a class that is
            # not numba's, whose place is its own, for the user-provided one
            # without NUMBA_CACHE_DIR, which takes nothing without it, and for
            # the IPython one without IPython.
            if reads_the_variable(cls):
                return os.path.abspath(os.path.join(config.CACHE_DIR, subpath)) if config.CACHE_DIR else None
            if caches_beside_the_source(cls):
                return os.path.abspath(os.path.join(os.path.dirname(py_file), "__pycache__"))
            if caches_under_the_user_cache_dir(cls):
                return os.path.abspath(os.path.join(user_cache_dir, subpath))
            if one_of(cls, for_ipython):
                ipython_cache = _ipython_numba_cache()
                return os.path.abspath(ipython_cache) if ipython_cache else None
            return None

        locations = {cls: location_of(cls) for cls in order}
        # Under a list with no locator that takes the file, or none with a
        # place for it, the user-provided one alone with NUMBA_CACHE_DIR unset,
        # numba took no location, and the list is the remedy, below, for an
        # error a caller passes on as for numba's no-locator one.
        if isinstance(failure, OSError) and any(locations.values()):
            # The error names the location numba took, or a file numba writes
            # in it: numba's own check passed the location, its temporary file
            # fitting where its cache files would not, or the package's named
            # file was refused since, a full disk or permissions changed. The
            # locator that took it decides the remedy, the first in numba's
            # order whose place the error names: whole paths, since the places
            # can nest, an install under either cache directory or
            # NUMBA_CACHE_DIR above the user's; and in order, since
            # NUMBA_CACHE_DIR set to the user's cache directory makes one path
            # of two.
            if filename:
                named = os.path.abspath(filename)
                named = {named, os.path.dirname(named)}
            else:
                # A caller can pass on an error that names nothing; numba's
                # and the package's own name the file refused.
                named = set()
            taken = next((cls for cls in order if locations[cls] and locations[cls] in named), None)

            def could_have_taken():
                # Which location numba took cannot be told from an error that
                # names no file, so the ones it could have taken are listed, in
                # its order, each once: NUMBA_CACHE_DIR set to the user's cache
                # directory makes one path of two.
                known = list(dict.fromkeys(locations[cls] for cls in order if locations[cls]))
                which = ", ".join(known[:-1]) + " or " + known[-1] if len(known) > 1 else "".join(known)
                return f"the location numba took{', one of ' + which if which else ''}"
            if failure.errno == errno.ENAMETOOLONG:
                if taken is not None and reads_the_variable(taken):
                    cure = "a shorter NUMBA_CACHE_DIR"
                elif taken is not None and caches_beside_the_source(taken):
                    cure = f"{package} installed at a shorter path{instead(taken)}"
                elif taken is not None and caches_under_the_user_cache_dir(taken):
                    cure = (
                        f"the user's cache directory, {user_cache_dir}, at a shorter path{_moved_through()}"
                        f"{instead(taken)}"
                    )
                elif taken is not None and one_of(taken, for_ipython):
                    cure = f"IPython's cache directory, {os.path.dirname(locations[taken])}, at a shorter path{instead(taken)}"
                elif not filename:
                    cure = f"{could_have_taken()}, at a shorter path{instead(None)}"
                else:
                    cure = f"that location, {filename}, at a shorter path{instead(None)}"
                return f"the path is too long for the file system: {cure}; or {silence}"
            # numba took the location, so the variable was not passed over for
            # it; set elsewhere it moves the cache off a disk that is full. A
            # location that is none of numba's is named as the error names it,
            # and an error that names none is told the ones numba could have
            # taken.
            if taken is not None and reads_the_variable(taken):
                alternative = ", or NUMBA_CACHE_DIR set to another writable directory"
            else:
                alternative = instead(taken, "a writable directory")
            if taken:
                opening = f"numba caches this file in {locations[taken]},"
            elif filename:
                opening = f"that location, {filename},"
            else:
                opening = f"{could_have_taken()},"
            # An error a caller builds from a message alone has no strerror.
            reason = failure.strerror or failure
            return (
                f"{opening} where no file can be written ({reason}): make room there, or make it "
                f"writable{alternative}; or {silence}"
            )
        # numba itself passes a location it cannot make or write over, for a
        # source on disk, so the error here is the no-locator one, every
        # locator tried and passed over, or the .zip locator's for a path with
        # .zip in it and no part ending in it, which numba raises where it
        # reaches that locator: the locators before it passed the file over,
        # and none after it was tried.
        zip_at = next((at for at, cls in enumerate(order) if one_of(cls, for_a_zip)), None)
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
            # Set and passed over: unwritable, or too deep for numba to make
            # its directory there, which a writable one as deep would not cure.
            return (
                f"NUMBA_CACHE_DIR is set to {config.CACHE_DIR}, which numba could not use: set it to a writable "
                f"directory at a short path, or {silence}"
            )
        return f"Set NUMBA_CACHE_DIR to a writable directory, or {silence}"
    if isinstance(failure, OSError) or getattr(sys, "frozen", False):
        if not isinstance(failure, OSError) and listed_without(user_wide):
            # numba's user-wide locator alone takes a frozen application, so
            # a list without it leaves numba no locator for one, and the
            # remedy asked for a writable directory numba never tried.
            return (
                "numba caches a frozen application through UserWideCacheLocator alone, and "
                f"NUMBA_CACHE_LOCATOR_CLASSES, {listed}, leaves it out: list it, or {silence}"
            )
        ipython_cache = _ipython_numba_cache() if filename else None
        if ipython_cache and os.path.abspath(ipython_cache) in {
            os.path.abspath(filename), os.path.dirname(os.path.abspath(filename))
        }:
            # numba's IPython locator takes a member of a .zip in an ipykernel
            # directory before the .zip locator is reached, and caches it in
            # numba_cache under IPython's cache directory, which the error
            # names, or a file numba writes in it: the remedy a cell file on
            # disk gets there, with no NUMBA_CACHE_DIR to offer for a source
            # not on disk.
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
        # The .zip's error names the location numba picked, a directory of
        # its own under the user's cache directory, or a file numba writes in
        # it, as the first save's does, and the location is what a reader can
        # act on; the frozen application's names nothing.
        user_cache_dir = AppDirs(appname="numba", appauthor=False).user_cache_dir
        location = filename or user_cache_dir
        in_the_user_cache = os.path.join(user_cache_dir, _CacheLocator.get_suitable_cache_subpath(py_file))
        if os.path.dirname(os.path.abspath(location)) == os.path.abspath(in_the_user_cache):
            location = os.path.dirname(location)
        if isinstance(failure, FileNotFoundError) and filename and py_file.startswith(filename + os.sep):
            # The stamp numba reads at decoration, of the archive the code
            # names: .pyc members compiled to name an archive since moved. A
            # location that cannot be made, under a dangling link, is ENOENT
            # too, and is not under the code's file.
            return (
                f"the module's code names {filename}, which is not there: its .pyc was compiled to name "
                "that path, so compile the archive's .pyc members to name its path now, or ship its source files; "
                f"or {silence}"
            )
        if isinstance(failure, OSError) and failure.errno == errno.ENAMETOOLONG:
            return (
                "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR "
                f"has no effect here, because the source is not a file on disk: the path is too long for the file "
                f"system, so put that directory, {location}, at a shorter path{_moved_through()}, or {silence}"
            )
        if isinstance(failure, OSError):
            # The location refused a file for another reason, a full disk or
            # permissions, which the error gives, as the remedy for a source on
            # disk gives it; the frozen application's no-locator error gives
            # none, numba having passed the location over on it.
            reason = failure.strerror or failure
            return (
                "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR "
                f"has no effect here, because the source is not a file on disk: no file can be written in that "
                f"directory, {location} ({reason}), so make room there, or make it writable, or {silence}"
            )
        return (
            "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR "
            f"has no effect here, because the source is not a file on disk: make that directory, {location}, "
            f"writable, or {silence}"
        )
    in_a_zip = any(part.endswith(".zip") for part in pathlib.Path(py_file).parts)
    if in_a_zip and listed_without(for_a_zip):
        # numba's .zip locator alone takes a source in a .zip, so a list
        # without it leaves numba no locator for one, and the archive remedy
        # offered the .zip the reader was importing from.
        return (
            f"numba caches a .zip through ZipCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, {listed}, leaves "
            f"it out: list it; or {silence}"
        )
    on_disk = f"install {package} with its source files on disk, unpacked from any archive"
    if listed:
        # The install caches through a locator for a file on disk the list
        # names: none, and it caches no more than the archive does; the
        # user-provided one alone, and that takes nothing without
        # NUMBA_CACHE_DIR.
        for_a_file_on_disk = [
            cls for cls in _locators(listed) if any(one_of(cls, base) for base in (user_provided, in_tree, user_wide))
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
    a_zip = "import it from a .zip holding its source files, which numba 0.61 and later cache in the user's cache directory"
    if listed_without(for_a_zip):
        # numba's .zip locator alone takes a .zip, so a list without it
        # would leave numba no locator for the .zip offered either.
        a_zip += f" through ZipCacheLocator, once NUMBA_CACHE_LOCATOR_CLASSES, {listed}, lists it"
    return (
        f"NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, {on_disk}, or "
        f"{a_zip}; or {silence}"
    )


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
    names the remedy: ``NUMBA_CACHE_DIR`` for a source file on disk, and where the location is too long for the
    file system what shortens the one it is, as ``cache_remedy`` sets out; for a ``.zip``, the user's cache
    directory made writable, since
    numba reads ``NUMBA_CACHE_DIR`` only for a source file on disk; for any other archive,
    or a ``.pyc``-only install, or a ``.pyc`` in a ``.zip``, the source files on disk or a ``.zip`` holding them,
    which numba 0.61 and later cache in the user's cache directory. ``NUMBOX_JIT_OPTIONS='{"cache": false}'``
    turns caching off and silences the warning. An error that is not the cache's, as ``is_a_cache_error`` draws
    the line, is raised as it was.

    A ``.zip`` whose cache directory holds every entry but can no longer be written takes the fallback too, where
    numba alone would have loaded the entries: the writability check is the rule numba applies to every other
    placement, and the one the ``.zip`` locator is missing, reported as https://github.com/numba/numba/issues/10888.

    Options without a ``cache`` key, as ``NUMBOX_JIT_OPTIONS`` can give, leave ``njit`` to its default, off, but
    the sqlite virtual-table and table-valued-function callbacks cache under them, so they are asked about too and
    come back with ``cache`` off, said, where the answer is no.
    """
    if not options.get("cache", True):
        return options
    for py_file in _module_files():
        try:
            check_cache_location(py_file, LONGEST_CACHE_FILE_NAME)
        except (RuntimeError, OSError, ValueError) as error:
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
