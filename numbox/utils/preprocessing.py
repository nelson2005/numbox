"""Source-anchor machinery for dynamically-exec'd code.

Content-addressed anchors keep numba's per-overload cache correct
when two exec'd code blocks differ only in ``co_consts``. See the
"Cache-anchor mechanism" section in ``docs/numbox.utils.rst`` for
the rationale and references.
"""
import errno
import hashlib
import os
import tempfile
import time
import warnings
from pathlib import Path

from numbox.core.configurations import check_cache_location, is_a_cache_error


def _anchor_root(subdir: str = "numbox-structref") -> Path:
    from numba import config
    from numba.misc.appdirs import AppDirs
    if config.CACHE_DIR:
        return Path(config.CACHE_DIR) / subdir
    return Path(AppDirs(appname="numba", appauthor=False).user_cache_dir) / subdir


def _anchor_path(subdir: str, stem: str, code_txt: str) -> Path:
    """Content-addressed on-disk source anchor for dynamically-exec'd code.

    See the "Cache-anchor mechanism" section in
    ``docs/numbox.utils.rst`` for the rationale.
    """
    digest = hashlib.sha256(code_txt.encode("utf-8")).hexdigest()[:16]
    return _anchor_root(subdir) / f"{stem}_{digest}.py"


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
    """Write the anchor at ``path`` and ask numba for its cache location; the error where there is none, else None.

    Three steps, each with its own errors. Making the anchor's directory: any ``OSError`` there is the location's,
    ``NUMBA_CACHE_DIR`` pointing into a file, a component too long, a full disk, and numba would pass such a
    location over too. Writing the anchor: an error there is the location's as well, but a name too long for
    the file system is numbox's own, made from a struct's or a function's name, and is raised. Asking numba,
    the question the package puts for its own modules: no locator, or a location it cannot use, is the cache's,
    and the answer may be the user's cache directory where the anchor's own cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return error
    try:
        _materialize_anchor(path, code_txt)
    except OSError as error:
        if error.errno == errno.ENAMETOOLONG:
            raise
        return error
    try:
        check_cache_location(path)
    except (RuntimeError, OSError, ValueError) as error:
        if not is_a_cache_error(error):
            raise
        return error
    return None


def _anchored_or_uncached(path: Path, code_txt: str, jit_options: dict) -> dict:
    """``jit_options`` to compile the code anchored at ``path`` under, the anchor written where it will be read.

    numba reads the anchor only to cache the code it names, so with caching off nothing is written and the
    path serves as the code's filename. With caching on the anchor is written and numba asked for its location,
    by ``_anchor_or_error``; where that gives an error, the code compiles without a cache after one warning
    naming the remedy, rather than dying at the write or at the first decorated function.
    """
    if not jit_options.get("cache"):
        return jit_options
    error = _anchor_or_error(path, code_txt)
    if error is not None:
        warnings.warn(
            f"numba cannot cache {path.name} here ({error}); it compiles without a cache. Set NUMBA_CACHE_DIR "
            "to a writable directory, or NUMBOX_JIT_OPTIONS='{\"cache\": false}' to turn caching off and "
            "silence this warning",
            RuntimeWarning, stacklevel=3,
        )
        return {**jit_options, "cache": False}
    return jit_options


_ORPHAN_AGE_SECONDS = 60


def _orphan_anchor_sweep(subdir: str) -> None:
    """Best-effort cleanup of orphaned ``.tmp-*`` anchor files from SIGKILL'd
    writers. Called at module import; failures are silent (the orphan is at
    worst harmless disk usage).

    Only sweeps files whose ``mtime`` is older than ``_ORPHAN_AGE_SECONDS``
    so a concurrent ``_materialize_anchor`` call in another process --
    which has the same ``.tmp-*`` shape between ``mkstemp`` and
    ``replace`` -- isn't unlinked mid-flight (the resulting
    ``FileNotFoundError`` on the in-progress writer's ``replace`` would
    abort its caller's import).
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
