"""Source-anchor machinery for dynamically-exec'd code.

Content-addressed anchors keep numba's per-overload cache correct
when two exec'd code blocks differ only in ``co_consts``. See the
"Cache-anchor mechanism" section in ``docs/numbox.utils.rst`` for
the rationale and references.
"""
import ast
import errno
import hashlib
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

from numbox.core.configurations import (LONGEST_CACHE_FILE_NAME, _OVERLOAD_DIGITS, cache_remedy, check_cache_location,
                                        is_a_cache_error)


def _anchor_root(subdir: str = "numbox-structref") -> Path:
    from numba import config
    from numba.misc.appdirs import AppDirs
    if config.CACHE_DIR:
        return Path(config.CACHE_DIR) / subdir
    return Path(AppDirs(appname="numba", appauthor=False).user_cache_dir) / subdir


_STEM_MAX = 40


def _longest_cache_file_name(stem: str, code_txt: str) -> int:
    """The longest name numba writes for a function of the code anchored at a file of this stem, in bytes.

    numba names a function's cache files after the file's stem, the function's qualified name with the angle brackets
    of ``<locals>`` dropped, its line, and Python's version and ABI flags. A data file's name also carries the
    overload's number. numba writes each file through a temporary name that is 21 bytes longer: ``.tmp.`` and
    sixteen hex digits.

    The result is the longest of these names over the functions that the code jits, nested functions and methods
    included. A function counts if it is decorated itself, with ``njit`` or with a decorator of the package's own that
    jits. A function defined within a decorated function counts too. An ``overload``'s constructor is defined in that
    way, and numba compiles it with the overload's options and writes its cache files under the overload's name and
    ``locals``. For the data file's number, the result counts the three digits that ``_OVERLOAD_DIGITS`` reserves,
    because numba numbers the data files from one without bound.

    A plain function at the top level or in a class body gets no cache file. A struct's method under its own name is
    such a function: it calls the jitted thunk. If the result counted these functions, a method named with two hundred
    characters would put the longest name over the file system's 255, and the struct would compile uncached. If the
    walk recognized decorators by five names alone, it would leave out the builder's derive function, jitted through
    the package's own decorator, and the constructor overload's inner function. numba would then fail to save after
    the check passed.

    The anchor's check makes a file with a name this long in numba's location, as the package's check does for its own
    files with ``LONGEST_CACHE_FILE_NAME``. A cache directory within that length of the path limit therefore fails the
    check here, and the code compiles uncached after one warning. In such a directory, numba's own check passes with a
    name of eight, and numba would raise at its first save, while the struct is made. A bound of 251 for every struct
    would turn caching off on a file system whose name limit is lower, such as eCryptfs's 143, for a struct whose
    names are 58 bytes long.
    """
    names = []

    def walk(node, outer, jitted):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{outer}{child.name}"
                counts = jitted or bool(child.decorator_list)
                if counts:
                    names.append((qualname, child.lineno))
                walk(child, f"{qualname}.<locals>.", counts)
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{outer}{child.name}.", jitted)
            else:
                walk(child, outer, jitted)

    walk(ast.parse(code_txt), "", False)
    version = f"py{sys.version_info[0]}{sys.version_info[1]}{getattr(sys, 'abiflags', '')}"
    return max(
        (len(f"{stem}.{qualname.replace('<', '').replace('>', '')}-{lineno}.{version}.{'9' * _OVERLOAD_DIGITS}.nbc.tmp."
             .encode("utf-8")) + 16
         for qualname, lineno in names), default=0)


# Generated code compiles under the package's options unless the caller gives its own, which make_structref,
# compile_kernel and the builder accept. compile_kernel's cache argument overrides all of them. NUMBOX_JIT_OPTIONS
# applies only to the package's options.
_SILENCE = (
    "compile without a cache to silence this warning: \"cache\" off in the jit options this code was given, or "
    "in its cache argument where it takes one, or NUMBOX_JIT_OPTIONS='{\"cache\": false}' where the options are "
    "the package's"
)


def bounded_stem(name: str) -> str:
    """Return ``name`` as a file-name stem: unchanged up to 40 bytes, else its whole characters within 31 and a digest.

    An anchor's name, and the names numba gives the cache files of the functions the anchor holds, carry the name of
    the struct or the function they were generated for. A file system allows a name 255 bytes long. numba's file
    names repeat the stem and the generated function's name, so without the bound a struct named with about 93
    characters would overflow them.

    With the names bounded, the longest of numba's file names belongs to a method thunk. It consists of the anchor's
    stem of at most 57 bytes, the struct's and the method's bounded names, the method's 64-character hash, and
    numba's line number and suffixes. It stays under 230 bytes whatever the names' lengths, and within the 255 with
    the 21 bytes that numba's temporary name adds when numba writes the file. A name of 40 bytes or fewer, which is
    nearly every name, keeps the file names it had. The measure is the name's UTF-8 length, which is the measure the
    file system uses: a character of another script takes up to four bytes there.
    """
    encoded = name.encode("utf-8")
    if len(encoded) <= _STEM_MAX:
        return name
    head = encoded[:_STEM_MAX - 9].decode("utf-8", errors="ignore")
    return f"{head}_{hashlib.sha256(encoded).hexdigest()[:8]}"


def _anchor_path(subdir: str, stem: str, code_txt: str) -> Path:
    """Content-addressed on-disk source anchor for dynamically-exec'd code, with a bounded stem.

    See the "Cache-anchor mechanism" section in
    ``docs/numbox.utils.rst`` for the rationale.
    """
    digest = hashlib.sha256(code_txt.encode("utf-8")).hexdigest()[:16]
    return _anchor_root(subdir) / f"{bounded_stem(stem)}_{digest}.py"


def _structref_anchor_path(struct_name: str, code_txt: str) -> Path:
    return _anchor_path("numbox-structref", struct_name, code_txt)


def _materialize_anchor(path: Path, code_txt: str) -> None:
    if path.exists():
        return
    fd, tmp_str = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(code_txt)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _anchor_or_error(path: Path, code_txt: str):
    """Write the anchor at ``path`` and ask numba for its cache location; return the error if there is none, else None.

    Any ``OSError`` from making the directory or writing the file means that there is no cache here, whatever its
    cause. The cause can be ``NUMBA_CACHE_DIR`` pointing into a file, a full disk, or a path too long for the file
    system, because of either the directory or the name that numbox made from a struct's or a function's name. numba
    itself passes a location over on any ``OSError``.

    The function then asks numba the question that the package asks for its own modules. If no locator takes the
    file, or numba cannot use the location, numba's error is a cache error, and the function returns it. numba may
    answer with the user's cache directory if the anchor's own directory cannot be written. numba's
    ``UnicodeEncodeError`` for the anchor's directory path is a cache error as well: numba hashes that path as UTF-8
    to name its cache directory, and raises for a path that is not UTF-8. The function asks the question with the
    longest name that numba writes for the code's functions, read from the code. A location within that length of
    the path limit therefore fails here, although numba's own check passes it with a name of eight.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _materialize_anchor(path, code_txt)
    except OSError as error:
        return error
    try:
        check_cache_location(path, _longest_cache_file_name(path.stem, code_txt))
    except (RuntimeError, OSError, ValueError) as error:
        if not is_a_cache_error(error):
            raise
        return error
    return None


def _anchored_or_uncached(path: Path, code_txt: str, jit_options: dict) -> dict:
    """Return the ``jit_options`` under which the code anchored at ``path`` compiles, and write the anchor if possible.

    numba reads the anchor to cache the code that it names, and quotes the source from it in its messages, a typing
    error's message among them. The function therefore writes the anchor whenever it can. With caching off, a failed
    write has no effect, and the path serves as the code's filename, as it did before. With caching on,
    ``_anchor_or_error`` writes the anchor and asks numba for its location. If that returns an error, the code
    compiles without a cache after one warning that names the remedy, instead of failing at the write or at the first
    decorated function.
    """
    if not jit_options.get("cache"):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _materialize_anchor(path, code_txt)
        except OSError:
            pass
        return jit_options
    error = _anchor_or_error(path, code_txt)
    if error is None:
        return jit_options
    if isinstance(error, OSError) and getattr(error, "longest_file_name", 0):
        # The file system accepts no name as long as numba's longest name for the anchor's functions, the length that
        # the check asks for. No shorter path removes that limit, and without this case the warning would offer one.
        from numbox.core.configurations import _name_limit
        remedy = (
            f"numba's cache files for the functions anchored here have names of up to {error.longest_file_name} "
            f"bytes, longer than a name can be on that file system ({_name_limit(error.filename)}): NUMBA_CACHE_DIR "
            f"at a directory on a file system that takes a name that long, or {_SILENCE}"
        )
    elif isinstance(error, OSError) and error.errno == errno.ENAMETOOLONG:
        # The names numbox makes are bounded, so the part of the path that is too long is the directory,
        # NUMBA_CACHE_DIR or the user's cache directory. A short NUMBA_CACHE_DIR moves the anchor out of either.
        remedy = f"The path is too long for the file system: NUMBA_CACHE_DIR at a shorter path, or {_SILENCE}"
    elif isinstance(error, UnicodeEncodeError):
        # numba hashes the anchor's directory path as UTF-8 to name its cache directory under NUMBA_CACHE_DIR or the
        # user's cache directory, and raises for bytes that are not UTF-8. Without this case, the warning would ask
        # for a writable directory, and the anchor's directory is writable.
        remedy = (
            f"numba hashes the anchor's directory path as UTF-8 to name its cache directory, and {path.parent} is "
            f"not: NUMBA_CACHE_DIR at a UTF-8 path, or {_SILENCE}"
        )
    else:
        remedy = f"Set NUMBA_CACHE_DIR to a writable directory, or {_SILENCE}"
    warnings.warn(
        f"numba cannot cache {path.name} here ({error}); it compiles without a cache. {remedy}",
        RuntimeWarning, stacklevel=3,
    )
    return {**jit_options, "cache": False}


def _cached_at_or_uncached(py_file: str, jit_options: dict) -> dict:
    """Return the ``jit_options`` under which code anchored to the module file ``py_file`` compiles.

    The options are returned as given if numba can cache a function of that file. Otherwise the function turns
    ``cache`` off, after one warning that names the remedy.

    ``make_graph``'s kernel is anchored to the builder's own file and cached beside it. The options that a caller
    gives reach numba unchanged, so the package's own answer does not cover them. The function therefore asks numba
    the question for that file, the way the package asks it for its own modules, and the kernel's name is within the
    package's bound. The remedy is the package's remedy for that placement.
    """
    if not jit_options.get("cache"):
        return jit_options
    try:
        check_cache_location(py_file, LONGEST_CACHE_FILE_NAME)
    except (RuntimeError, OSError, KeyError, ValueError) as error:
        if not is_a_cache_error(error):
            raise
        failure = error
    else:
        return jit_options
    warnings.warn(
        f"numba cannot cache the code generated at {os.path.basename(py_file)} here ({failure}); it compiles "
        f"without a cache. {cache_remedy(py_file, failure, _SILENCE)}",
        RuntimeWarning, stacklevel=3,
    )
    return {**jit_options, "cache": False}


_ORPHAN_AGE_SECONDS = 60


def _orphan_anchor_sweep(subdir: str) -> None:
    """Remove, as a best effort, the orphaned ``.tmp-*`` anchor files of writers killed by SIGKILL.

    The module calls this function at import. The function ignores every failure, because an orphan is at worst
    harmless disk usage.

    The function sweeps only files whose ``mtime`` is older than ``_ORPHAN_AGE_SECONDS``. A concurrent
    ``_materialize_anchor`` call in another process has a file of the same ``.tmp-*`` shape between ``mkstemp`` and
    ``replace``, and the age limit keeps the sweep from unlinking that file while the write is in progress.
    Otherwise, the ``FileNotFoundError`` at the in-progress writer's ``replace`` would abort its caller's import.
    """
    try:
        root = _anchor_root(subdir)
        if not root.exists():
            return
        cutoff = time.time() - _ORPHAN_AGE_SECONDS
        for orphan in root.glob("*.tmp-*"):
            try:
                if orphan.stat().st_mtime < cutoff:
                    orphan.unlink()
            except OSError:
                pass
    except Exception:
        pass


_orphan_anchor_sweep("numbox-structref")
