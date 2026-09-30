"""Importing numbox where numba can write no cache for it.

numba sets a cached function up when it is decorated and raises there when no
cache location can be written, so ``import numbox`` died at its first module
under an archive import or a read-only install. Every check here runs in a
subprocess with its own tree and cache directory, so the placement under test
is the one the subprocess sees and nothing else.
"""
import compileall
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent

# chmod takes write access from a directory neither on Windows nor from root.
needs_a_directory_it_cannot_write = pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="needs a directory this user cannot write to")

# A lazily compiled helper, an eagerly compiled one and a proxied binding, each
# decorated at import under the package's options, then used.
IMPORT_AND_USE = (
    "import math\n"
    "import numpy as np\n"
    "import numbox.core.configurations as configurations\n"
    "from numbox.core.bindings.libm import cos\n"
    "from numbox.utils.lowlevel import array_data_p, get_str_from_p_as_int\n"
    "assert abs(cos(0.5) - math.cos(0.5)) < 1e-15\n"
    "text = np.frombuffer(b'numbox\\0', dtype=np.uint8).copy()\n"
    "assert get_str_from_p_as_int(array_data_p(text)) == 'numbox'\n"
    "print(configurations.__file__)\n"
)


def _archive(path):
    """numbox's modules zipped into ``path``, which goes on PYTHONPATH as it is."""
    with zipfile.ZipFile(path, "w") as zipped:
        for source in sorted((REPO / "numbox").rglob("*.py")):
            zipped.write(source, str(source.relative_to(REPO)))
    return path


def _run(env, cwd, warnings="always"):
    # From a neutral cwd the child imports the placement PYTHONPATH names,
    # not the tree the test runs from.
    return subprocess.run([sys.executable, "-W", warnings, "-c", IMPORT_AND_USE],
                          capture_output=True, text=True, env=env, cwd=str(cwd))


def _index_files(cache_dir):
    return sorted(path.name for path in Path(cache_dir).rglob("*.nbi"))


@pytest.mark.parametrize("name", ["numbox-0.0.0-py3.12.egg", "numbox-0.0.0-py3-none-any.whl"])
def test_an_import_from_an_archive_compiles_uncached_with_one_warning_naming_the_remedy(tmp_path, name):
    # numba's cache locators need the source file on disk, so an import from
    # an .egg, .whl or .pyz archive, which Spark's --py-files ships, raised
    # RuntimeError at the first decorated function, naming neither the way to
    # a cache nor the option that turns caching off. The package answers the
    # question once, so there is one warning, not one per function.
    # NUMBA_CACHE_DIR is no way to a cache here: it is set and writable, the
    # warning fires all the same and nothing is written there, so the warning
    # says so instead of offering it.
    archive = _archive(tmp_path / name)
    env = dict(os.environ, PYTHONPATH=str(archive), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "NUMBA_CACHE_DIR has no effect here" in run.stderr and "Set NUMBA_CACHE_DIR" not in run.stderr
    assert _index_files(tmp_path / "cache") == []
    quiet = _run(dict(env, NUMBOX_JIT_OPTIONS='{"cache": false}'), tmp_path, warnings="error")
    assert quiet.returncode == 0, quiet.stderr


def test_a_zip_import_is_cached_by_numba_from_0_61(tmp_path):
    # The warning sends an archive's user to a .zip, which numba caches from
    # 0.61 on, in the user's cache directory whatever NUMBA_CACHE_DIR says.
    # Before that a .zip is one more archive. This pins the remedy the warning
    # names, numba's behaviour, not the fallback, which has nothing to do here.
    import numba
    archive = _archive(tmp_path / "numbox.zip")
    home = tmp_path / "home"
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    cached = tuple(int(part) for part in numba.__version__.split(".")[:2]) >= (0, 61)
    assert ("compiles without a cache" not in run.stderr) == cached, run.stderr
    assert _index_files(tmp_path / "cache") == []
    if os.name != "nt":
        # On Windows numba asks the system for the user's cache directory, and
        # no variable set here moves it.
        assert bool(_index_files(home)) == cached


@needs_a_directory_it_cannot_write
def test_one_read_only_directory_among_writable_ones_takes_the_fallback(tmp_path):
    # numba's in-tree cache is a __pycache__ beside each source, so the
    # package's directories answer separately. A check on configurations.py's
    # directory alone passed here, and the import died at libm's first binding.
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    home = tmp_path / "home"
    home.mkdir()
    read_only = [home, site / "numbox" / "core" / "bindings"]
    for path in read_only:
        path.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(site), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
        env.pop("NUMBA_CACHE_DIR", None)
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(site) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "Set NUMBA_CACHE_DIR" in run.stderr
        cured = _run(dict(env, NUMBA_CACHE_DIR=str(tmp_path / "cache")), tmp_path, warnings="error")
        assert cured.returncode == 0, cured.stderr
        assert _index_files(tmp_path / "cache")
    finally:
        for path in read_only:
            path.chmod(0o755)


def test_a_sourceless_install_is_told_the_source_is_not_on_disk(tmp_path):
    # A .pyc-only install: numba looks the source up by the code's co_filename,
    # which names the .py that was removed, and finds no locator. The module's
    # __file__ is the .pyc, on disk, and a check on it offered NUMBA_CACHE_DIR,
    # which numba ignores without the source.
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    assert compileall.compile_dir(str(site), quiet=1, legacy=True)
    for source in list(site.rglob("*.py")):
        source.unlink()
    env = dict(os.environ, PYTHONPATH=str(site), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(site) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "source is not a file on disk" in run.stderr and "Set NUMBA_CACHE_DIR" not in run.stderr
    assert _index_files(tmp_path / "cache") == []


def _zip_is_cached():
    import numba
    return tuple(int(part) for part in numba.__version__.split(".")[:2]) >= (0, 61)


@needs_a_directory_it_cannot_write
def test_a_zip_import_with_no_writable_user_cache_directory_compiles_uncached_with_a_warning(tmp_path):
    # numba takes the user's cache directory for a .zip without checking that
    # it can be written, so where it cannot, an executor with a read-only
    # home, the first save raised PermissionError and the import died on it.
    # The remedy is that directory made writable, not the .zip the archive
    # warning offers, which is what the user already has.
    archive = _archive(tmp_path / "numbox.zip")
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
                   NUMBA_CACHE_DIR=str(tmp_path / "cache"))
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "NUMBA_CACHE_DIR has no effect here" in run.stderr
        if _zip_is_cached():
            assert "make that directory writable" in run.stderr and "import it from a .zip" not in run.stderr
    finally:
        home.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_a_zip_import_whose_cache_directory_stopped_being_writable_compiles_uncached(tmp_path):
    # Every entry is in the user's cache directory from an earlier import, and
    # the directory can no longer be written. A probe that only loaded its own
    # entry would have said the cache works and left the import to die at
    # numba's first write for anything not there; the writability check numba
    # runs for every other placement is run here for the .zip too.
    archive = _archive(tmp_path / "numbox.zip")
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    warm = _run(env, tmp_path)
    assert warm.returncode == 0, warm.stderr
    assert bool(_index_files(home)) == _zip_is_cached()
    read_only = [home, *(path for path in home.rglob("*") if path.is_dir())]
    for path in read_only:
        path.chmod(0o555)
    try:
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
    finally:
        for path in read_only:
            path.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_a_read_only_install_warns_naming_numba_cache_dir_and_setting_it_caches(tmp_path):
    # The other way to have no cache location: the source is on disk, and
    # neither its directory nor the user's cache directory can be written.
    # NUMBA_CACHE_DIR is the remedy there, and nothing showed that the warning
    # names it or that setting it works.
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    home = tmp_path / "home"
    home.mkdir()
    read_only = [home, *(path for path in site.rglob("*") if path.is_dir())]
    for path in read_only:
        path.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(site), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
        env.pop("NUMBA_CACHE_DIR", None)
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(site) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "Set NUMBA_CACHE_DIR" in run.stderr
        cured = _run(dict(env, NUMBA_CACHE_DIR=str(tmp_path / "cache")), tmp_path, warnings="error")
        assert cured.returncode == 0, cured.stderr
        assert _index_files(tmp_path / "cache")
    finally:
        for path in read_only:
            path.chmod(0o755)


ANOTHER_ERROR_AT_THE_CACHE_SET_UP = (
    "import numba.core.caching as caching\n"
    "def refuse(self, py_func):\n"
    "    raise RuntimeError('a locator of another kind refused')\n"
    "caching.CompileResultCacheImpl.__init__ = refuse\n"
    "import numbox.core.configurations\n"
)


def test_an_error_that_is_not_the_caches_is_raised_as_it_was(tmp_path):
    # The fallback answers two errors of numba's cache set-up, no locator and
    # a directory that cannot be written. A RuntimeError of another kind at
    # the same step, a locator of the user's own refusing, say, is not its to
    # turn into an uncached import with a remedy that does not apply.
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", ANOTHER_ERROR_AT_THE_CACHE_SET_UP],
                         capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode != 0 and "a locator of another kind refused" in run.stderr, run.stderr
    assert "compiles without a cache" not in run.stderr


# The code numbox generates at run time, each child importing the helpers of
# the test module that exercises it.
MAKE_A_STRUCTREF = (
    "from numba.core.types import StructRef, float32\n"
    "from numba.experimental.structref import register\n"
    "from numbox.utils.highlevel import make_structref\n"
    "@register\n"
    "class TypeClass(StructRef):\n"
    "    pass\n"
    "Struct = make_structref('Struct', {'value': float32}, TypeClass)\n"
    "assert Struct(2.5).value == 2.5\n"
    "print('made')\n"
)

REGISTER_AN_AGGREGATE = (
    "from test.core.test_sqlite_udf_helpers import (_open_memory, _make_table, _read1_int64, sum_state_type,\n"
    "                                               sum_init, sum_step, sum_finalize)\n"
    "from numbox.core.bindings.sqlite.udf_helpers import register_aggregate\n"
    "from numbox.core.bindings.sqlite.conn import sqlite3_close\n"
    "db = _open_memory()\n"
    "_make_table(db, [1, 2, 3, 4, 5])\n"
    "register_aggregate(db, 'my_sum', 1, sum_state_type, sum_init, sum_step, sum_finalize)\n"
    "value, _ = _read1_int64(db, 'SELECT __cap(my_sum(v)) FROM t')\n"
    "sqlite3_close(db)\n"
    "assert value == 15, value\n"
    "print('summed')\n"
)

REGISTER_A_TVF = (
    "import numpy as np\n"
    "from test.core.test_sqlite_tvf import _open, _select_int, _series, _OUT\n"
    "from numbox.core.bindings.sqlite.tvf import register_tvf\n"
    "from numbox.core.bindings.sqlite.conn import sqlite3_close\n"
    "db = _open()\n"
    "handle = register_tvf(db.value, 'series', (np.int64, np.int64), _OUT, _series)\n"
    "rc, rows = _select_int(db, 'SELECT n FROM series(2, 5)')\n"
    "assert rc == 0 and [row[0] for row in rows] == [2, 3, 4], (rc, rows)\n"
    "sqlite3_close(db.value)\n"
    "print('selected')\n"
)


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("child, word", [
    (MAKE_A_STRUCTREF, "made"), (REGISTER_AN_AGGREGATE, "summed"), (REGISTER_A_TVF, "selected"),
], ids=["make_structref", "register_aggregate", "register_tvf"])
def test_generated_code_compiles_uncached_where_its_anchor_cannot_be_written(tmp_path, child, word):
    # The code numbox generates at run time is anchored to a file under
    # NUMBA_CACHE_DIR or the user's cache directory, and the anchor was written
    # whatever the cache option said. So with the tree writable, where the
    # package's own functions cache, and the user's cache directory not,
    # make_structref and the sqlite registrations died at the anchor's
    # directory. The anchor is written to be cached from, and where it cannot
    # be, the code compiles without a cache after a warning; NUMBA_CACHE_DIR at
    # a writable directory cures it and holds the anchor.
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(REPO), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
        env.pop("NUMBA_CACHE_DIR", None)
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True,
                             env=env, cwd=str(tmp_path))
        assert run.returncode == 0 and word in run.stdout, run.stderr
        assert "compiles without a cache" in run.stderr and "Set NUMBA_CACHE_DIR" in run.stderr, run.stderr
        cured = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", child], capture_output=True,
                               text=True, env=dict(env, NUMBA_CACHE_DIR=str(tmp_path / "cache")), cwd=str(tmp_path))
        assert cured.returncode == 0 and word in cured.stdout, cured.stderr
        assert list((tmp_path / "cache").rglob("*.py")), "no anchor under NUMBA_CACHE_DIR"
    finally:
        home.chmod(0o755)


@needs_a_directory_it_cannot_write
@pytest.mark.parametrize("child, word", [
    (MAKE_A_STRUCTREF, "made"), (REGISTER_AN_AGGREGATE, "summed"), (REGISTER_A_TVF, "selected"),
], ids=["make_structref", "register_aggregate", "register_tvf"])
def test_a_warm_anchor_in_a_directory_that_stopped_being_writable(tmp_path, child, word):
    # The anchor is on disk from an earlier run, in a NUMBA_CACHE_DIR that can
    # no longer be written. numba then caches the code in the user's cache
    # directory, and a check on the anchor's own directory would have turned
    # caching off where numba had a location. With the user's cache directory
    # unwritable too there is none, and a check that only wrote the anchor
    # would have let the first decorated function die at numba's set-up.
    home = tmp_path / "home"
    home.mkdir()
    cache = tmp_path / "cache"
    env = dict(os.environ, PYTHONPATH=str(REPO), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(cache))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    warm = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", child], capture_output=True,
                          text=True, env=env, cwd=str(tmp_path))
    assert warm.returncode == 0 and word in warm.stdout, warm.stderr
    anchors = list(cache.rglob("*.py"))
    assert anchors, "no anchor under NUMBA_CACHE_DIR"
    read_only = [cache, *(path for path in cache.rglob("*") if path.is_dir())]
    for path in read_only:
        path.chmod(0o555)
    try:
        in_user_cache = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", child],
                                       capture_output=True, text=True, env=env, cwd=str(tmp_path))
        assert in_user_cache.returncode == 0 and word in in_user_cache.stdout, in_user_cache.stderr
        assert _index_files(home), "numba did not cache the generated code in the user's cache directory"
        home_tree = [home, *(path for path in home.rglob("*") if path.is_dir())]
        read_only.extend(home_tree)
        for path in home_tree:
            path.chmod(0o555)
        uncached = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True,
                                  env=env, cwd=str(tmp_path))
        assert uncached.returncode == 0 and word in uncached.stdout, uncached.stderr
        assert "compiles without a cache" in uncached.stderr and "Set NUMBA_CACHE_DIR" in uncached.stderr
    finally:
        for path in read_only:
            path.chmod(0o755)
