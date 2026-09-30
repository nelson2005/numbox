"""Importing numbox where numba can write no cache for it.

numba sets a cached function up when it is decorated and raises there when no
cache location can be written, so ``import numbox`` died at its first module
under an archive import or a read-only install. Every check here runs in a
subprocess with its own tree and cache directory, so the placement under test
is the one the subprocess sees and nothing else.
"""
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
    # Before that a .zip is one more archive.
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

