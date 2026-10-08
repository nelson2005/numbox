"""Importing numbox where numba can write no cache for it.

numba sets a cached function up when it is decorated and raises there when no
cache location can be written, so ``import numbox`` died at its first module
under an archive import or a read-only install. Every check here runs in a
subprocess with its own tree and cache directory, so the placement under test
is the one the subprocess sees and nothing else.
"""
import ast
import compileall
import errno
import importlib.util
import marshal
import os
import py_compile
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from numbox.core.configurations import numba_version

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


def _archive(path, bytecode_naming_the_archive=()):
    """numbox's modules zipped into ``path``, which goes on PYTHONPATH as it is.

    A directory named in ``bytecode_naming_the_archive`` goes in as ``.pyc`` alone, each compiled to name its
    path inside the archive, as ``compileall -d`` and ``py_compile``'s ``dfile`` do.
    """
    with zipfile.ZipFile(path, "w") as zipped:
        for source in sorted((REPO / "numbox").rglob("*.py")):
            member = str(source.relative_to(REPO))
            if source.parent.relative_to(REPO).as_posix() in bytecode_naming_the_archive:
                compiled = py_compile.compile(str(source), cfile=str(path.parent / (source.name + "c")),
                                              dfile=os.path.join(str(path), member), doraise=True)
                zipped.write(compiled, member + "c")
                os.unlink(compiled)
            else:
                zipped.write(source, member)
    return path


def _run(env, cwd, warnings="always"):
    # From a neutral cwd the child imports the placement PYTHONPATH names,
    # not the tree the test runs from.
    return subprocess.run([sys.executable, "-W", warnings, "-c", IMPORT_AND_USE],
                          capture_output=True, text=True, env=env, cwd=str(cwd))


def _index_files(cache_dir):
    return sorted(path.name for path in Path(cache_dir).rglob("*.nbi"))


def _directory_of_length(root, length):
    """A directory made under ``root`` whose path is ``length`` bytes, whatever ``root``'s length.

    Components of 200 while one more leaves room for a last, then the last of the length that lands there, so
    the directory itself can always be made.
    """
    deep = root
    while len(str(deep)) + 201 < length - 1:
        deep = deep / ("d" * 200)
    deep = deep / ("d" * (length - 1 - len(str(deep))))
    assert len(str(deep)) == length
    deep.mkdir(parents=True)
    return deep


@pytest.mark.parametrize("name", [
    "numbox-0.0.0-py3.12.egg", "numbox-0.0.0-py3-none-any.whl", "my.zip.dir/numbox-0.0.0-py3.12.egg",
])
def test_an_import_from_an_archive_compiles_uncached_with_one_warning_naming_the_remedy(tmp_path, name):
    # numba's cache locators need the source file on disk, so an import from
    # an .egg, .whl or .pyz archive, which Spark's --py-files ships, raised
    # RuntimeError at the first decorated function, naming neither the way to
    # a cache nor the option that turns caching off. The package answers the
    # question once, so there is one warning, not one per function.
    # NUMBA_CACHE_DIR is no way to a cache here: it is set and writable, the
    # warning fires all the same and nothing is written there, so the warning
    # says so instead of offering it. Under a directory whose name holds .zip
    # numba's .zip locator takes the file by that substring and raises
    # ValueError finding no .zip in it; that is the cache's error too.
    (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
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


def test_options_without_a_cache_key_are_asked_for_the_sites_that_cache_under_them(tmp_path):
    # NUMBOX_JIT_OPTIONS without "cache" leaves njit's default, off, but the
    # sqlite virtual-table and table-valued-function callbacks cache unless it
    # says no, and a check that read the missing key as off let the first
    # callback die at numba's set-up from an archive.
    archive = _archive(tmp_path / "numbox-0.0.0-py3.12.egg")
    env = dict(os.environ, PYTHONPATH=str(archive), NUMBA_CACHE_DIR=str(tmp_path / "cache"),
               NUMBOX_JIT_OPTIONS='{"boundscheck": false}')
    child = "import numbox.core.bindings.sqlite.vtable as m; print(m.__file__)"
    run = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr


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


IMPORT_SQLITE_TYPEMAP = "import numbox.core.bindings.sqlite._typemap as m; print(m.__file__)"
IMPORT_LIBM = "import numbox.core.bindings.libm as m; print(m.__file__)"


def test_a_module_that_survives_as_pyc_alone_beside_sourced_ones_takes_the_fallback(tmp_path):
    # numba finds a location by each module's own source, so one module without
    # its .py fails where its neighbours, __init__.py among them, pass. A walk
    # that asked one module per directory passed here, and libm died.
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    libm = site / "numbox" / "core" / "bindings" / "libm.py"
    assert compileall.compile_file(str(libm), quiet=1, legacy=True)
    libm.unlink()
    env = dict(os.environ, PYTHONPATH=str(site), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_LIBM], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and str(site) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "source is not a file on disk" in run.stderr and "Set NUMBA_CACHE_DIR" not in run.stderr


def test_a_pyc_alone_compiled_from_a_tree_still_on_disk_caches_by_that_tree(tmp_path):
    # The sourceless loader keeps the file a .pyc was compiled from on its code,
    # and numba looks that up: here a tree elsewhere, still on disk, so libm's
    # functions cache under NUMBA_CACHE_DIR by it. A probe that asked by the .py
    # beside the .pyc, gone, turned caching off for the package and said the
    # source is not on disk.
    built = tmp_path / "build" / "numbox" / "core" / "bindings" / "libm.py"
    built.parent.mkdir(parents=True)
    shutil.copy(REPO / "numbox" / "core" / "bindings" / "libm.py", built)
    assert compileall.compile_file(str(built), quiet=1, legacy=True)
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    libm = site / "numbox" / "core" / "bindings" / "libm.py"
    libm.unlink()
    shutil.copy(built.with_suffix(".pyc"), libm.with_suffix(".pyc"))
    env = dict(os.environ, PYTHONPATH=str(site), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", IMPORT_LIBM], capture_output=True,
                         text=True, env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and str(site) in run.stdout, run.stderr
    assert any(name.startswith("libm.") for name in _index_files(tmp_path / "cache")), "libm did not cache"


def test_a_directory_whose_modules_survive_as_pyc_alone_takes_the_fallback(tmp_path):
    # The rest of the package keeps its sources, so its directories pass; the
    # one directory numba cannot cache from has no .py to name, and a walk that
    # named directories by their .py files skipped it. The remedy is the one
    # for a source that is not on disk, not NUMBA_CACHE_DIR, which numba reads
    # for a source on disk only.
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    sqlite = site / "numbox" / "core" / "bindings" / "sqlite"
    assert compileall.compile_dir(str(sqlite), quiet=1, legacy=True)
    for source in list(sqlite.glob("*.py")):
        source.unlink()
    env = dict(os.environ, PYTHONPATH=str(site), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_SQLITE_TYPEMAP], capture_output=True,
                         text=True, env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and str(sqlite) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "source is not a file on disk" in run.stderr and "Set NUMBA_CACHE_DIR" not in run.stderr


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


@pytest.mark.parametrize("tree", ["site", "numbox.zip.tree"])
def test_a_pyc_in_a_zip_asks_by_the_file_it_was_compiled_from(tmp_path, tree):
    # zipimport takes a .pyc before the .py beside it and keeps the file it was
    # compiled from on its code, which is what numba looks up for the module's
    # functions; here that file is gone. A listing of the archive's .py members
    # said every directory caches, from numba 0.61 on, and libm's first
    # binding died at numba's set-up. A tree named after the archive, beside
    # it, passed a test by string prefix for the archive's own files, and its
    # .pyc went unasked: numba, seeing ".zip" in the file's path and no part
    # named so, died with its ValueError.
    site = tmp_path / tree
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    libm = site / "numbox" / "core" / "bindings" / "libm.py"
    assert compileall.compile_file(str(libm), quiet=1, legacy=True)
    libm.unlink()
    archive = tmp_path / "numbox.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for member in sorted(path for path in site.rglob("*") if path.suffix in (".py", ".pyc")):
            zipped.write(member, str(member.relative_to(site)))
    shutil.rmtree(site)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_LIBM], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "source is not a file on disk" in run.stderr and "holding its source files" in run.stderr, run.stderr
    if _zip_is_cached():
        # numba quotes the file with %r, which doubles Windows's backslashes.
        assert (repr(str(libm)) if tree == "site" else "No zip file found") in run.stderr, run.stderr


def test_a_stale_pyc_beside_its_source_in_a_zip_is_passed_over_as_zipimport_passes_it(tmp_path):
    # zipimport runs the .py where the .pyc beside it is stale against it, by
    # size or time, so the module's functions are cached from the archive like
    # the rest; a probe that asked by the stale .pyc's compile-time file, gone
    # here, turned caching off for the package and offered a .zip holding the
    # sources, which this is. This pins the outcome; that a .pyc zipimport
    # would run is asked at all is pinned by the sourceless case above, since
    # passing every .pyc over gives this outcome too.
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    libm = site / "numbox" / "core" / "bindings" / "libm.py"
    assert compileall.compile_file(str(libm), quiet=1, legacy=True)
    with libm.open("a") as source:
        source.write("\n# a line after the .pyc was compiled\n")
    archive = tmp_path / "numbox.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for member in sorted(path for path in site.rglob("*") if path.suffix in (".py", ".pyc")):
            zipped.write(member, str(member.relative_to(site)))
    shutil.rmtree(site)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_LIBM], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and run.stdout.strip().endswith("libm.py"), run.stderr
    if _zip_is_cached():
        assert "compiles without a cache" not in run.stderr, run.stderr
    if _zip_is_cached() and os.name != "nt":
        # On Windows numba's user cache directory is the shell's known folder,
        # which no variable moves under the home.
        assert _index_files(home), "libm's functions were not cached from the archive"


def test_a_stale_pyc_whose_source_in_the_zip_does_not_compile_is_passed_over(tmp_path):
    # zipimport, passing the stale .pyc over, compiles the .py beside it, and a
    # syntax error there, uncaught, killed the import of configurations where
    # nothing imports the module; its own import would die on it, as it should.
    archive = _archive(tmp_path / "numbox.zip")
    stray = tmp_path / "stray.py"
    stray.write_text("compiled = True\n")
    assert compileall.compile_file(str(stray), quiet=1, legacy=True)
    with zipfile.ZipFile(archive, "a") as zipped:
        zipped.write(stray.with_suffix(".pyc"), "numbox/stray.pyc")
        zipped.writestr("numbox/stray.py", "def broken(:\n    pass\n")
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    if _zip_is_cached():
        assert "compiles without a cache" not in run.stderr, run.stderr


@pytest.mark.parametrize("damage", ["truncated", "lzma"])
def test_a_stray_pyc_in_a_zip_that_nothing_imports_is_passed_over(tmp_path, damage):
    # A .pyc member of this interpreter's magic that zipimport could not run,
    # truncated, or stored with a compression zipimport does not read, and that
    # no import reaches: the probe's unmarshal of the first died with EOFError
    # at the import of configurations, where main imported, and zipimport's
    # own read of the second with zlib.error, past the exceptions then caught.
    archive = _archive(tmp_path / "numbox.zip")
    with zipfile.ZipFile(archive, "a") as zipped:
        if damage == "truncated":
            zipped.writestr("numbox/stray.pyc", importlib.util.MAGIC_NUMBER + bytes(12) + b"\xe3\x00")
        else:
            data = importlib.util.MAGIC_NUMBER + bytes(12) + marshal.dumps(compile("stray = True\n", "stray.py", "exec"))
            zipped.writestr("numbox/stray.pyc", data, compress_type=zipfile.ZIP_LZMA)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    if _zip_is_cached():
        assert "compiles without a cache" not in run.stderr, run.stderr


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
            assert "so make room there, or make it writable" in run.stderr, run.stderr
            assert "import it from a .zip" not in run.stderr
    finally:
        home.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_a_zip_whose_configurations_runs_from_a_pyc_compiled_from_a_tree_still_on_disk(tmp_path):
    # configurations.pyc in the archive, current, compiled from a tree that is
    # still on disk: zipimport runs it with that tree's file on its code, and a
    # probe that took the package's place from its own code's file walked the
    # tree, found it cacheable, and never asked the archive, whose first
    # function died at numba's first save with the home read-only.
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    configurations = site / "numbox" / "core" / "configurations.py"
    assert compileall.compile_file(str(configurations), quiet=1, legacy=True)
    archive = tmp_path / "numbox.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for member in sorted(path for path in site.rglob("*") if path.suffix in (".py", ".pyc")):
            zipped.write(member, str(member.relative_to(site)))
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
                   NUMBA_CACHE_DIR=str(tmp_path / "cache"))
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_LIBM], capture_output=True, text=True,
                             env=env, cwd=str(tmp_path))
        assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "NUMBA_CACHE_DIR has no effect here" in run.stderr, run.stderr
    finally:
        home.chmod(0o755)


@pytest.mark.skipif(os.name == "nt", reason="numba's user cache directory on Windows is not under HOME")
def test_a_zip_import_whose_user_cache_directory_is_too_long_is_told_so(tmp_path):
    # numba's location for a .zip is under the user's cache directory, and
    # with a component of that too long for the file system the location
    # cannot be made: the warning said to make it writable, which cannot help;
    # the path is the thing, through XDG_CACHE_HOME or HOME. The home itself
    # carries the component, since macOS keeps the directory under
    # ~/Library/Caches and reads no XDG_CACHE_HOME.
    archive = _archive(tmp_path / "numbox.zip")
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(tmp_path / ("c" * 300)),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    for name in ("NUMBOX_JIT_OPTIONS", "XDG_CACHE_HOME"):
        env.pop(name, None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    if _zip_is_cached():
        assert "too long for the file system, so put that directory" in run.stderr, run.stderr
        assert "make room there" not in run.stderr, run.stderr


def test_a_moved_zip_whose_pyc_members_name_its_old_path_compiles_uncached_and_is_told_why(tmp_path):
    # numba reads the source's stamp at decoration, the archive's for a .zip,
    # by the path the module's code names: .pyc members compiled to name the
    # archive, then the archive moved, name a path that is not there, and the
    # import died there with FileNotFoundError past a check that read no stamp.
    (tmp_path / "build").mkdir()
    archive = _archive(tmp_path / "build" / "numbox.zip", ("numbox/core/bindings",))
    moved = tmp_path / "numbox.zip"
    archive.rename(moved)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(moved), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", IMPORT_LIBM], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and str(moved) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    if _zip_is_cached():
        assert f"names {archive}" in run.stderr and "which is not there" in run.stderr, run.stderr


@pytest.mark.skipif(os.name == "nt", reason="a symlink needs a privilege on Windows; the cache directory is not under HOME")
def test_a_zip_import_whose_user_cache_directory_hangs_off_a_dangling_link_is_told_to_make_it(tmp_path):
    # numba's location cannot be made under a link to nowhere, ENOENT, which is
    # the error a moved archive's stamp gives too: the warning blamed the
    # archive's .pyc members, of which this archive has none. The moved archive
    # is the one whose path the module's code lies under. The home is the link,
    # for macOS's ~/Library/Caches as for Linux's ~/.cache.
    archive = _archive(tmp_path / "numbox.zip")
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere", target_is_directory=True)
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(tmp_path / "dangling"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    for name in ("NUMBOX_JIT_OPTIONS", "XDG_CACHE_HOME"):
        env.pop(name, None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    if _zip_is_cached():
        assert "so make room there, or make it writable" in run.stderr, run.stderr
        assert "its .pyc was compiled" not in run.stderr, run.stderr


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
@pytest.mark.parametrize("parent, bytecode", [("", ()), ("container.zip", ()), ("", ("numbox/core/bindings",))],
                         ids=["sources", "under a directory named .zip", "bindings as .pyc naming the archive"])
def test_a_zip_import_whose_location_for_one_directory_stopped_being_writable_takes_the_fallback(
        tmp_path, parent, bytecode):
    # numba caches a .zip per directory of it, each in a location of its own
    # under the user's cache directory, so the directories answer separately
    # there too; a check on configurations.py's location alone passed here,
    # and libm died at its first save. The archive's directories are listed,
    # the archive being the first part of the path named .zip that is one: a
    # listing that took a directory so named above it listed nothing. A
    # directory of .pyc members compiled to name the archive, as compileall -d
    # does, stands in the listing like one of .py members: a listing that let
    # only .py members stand never asked for it.
    (tmp_path / parent).mkdir(exist_ok=True)
    archive = _archive(tmp_path / parent / "numbox.zip", bytecode)
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
    env.pop("NUMBA_CACHE_DIR", None)
    env.pop("NUMBOX_JIT_OPTIONS", None)
    warm = _run(env, tmp_path)
    assert warm.returncode == 0, warm.stderr
    if not _zip_is_cached():
        pytest.skip("numba caches a .zip from 0.61 on")
    # Under XDG_CACHE_HOME on Linux, under Library/Caches on macOS.
    locations = [path for path in home.rglob("bindings_*") if path.is_dir()]
    assert len(locations) == 1, sorted(str(path.relative_to(home)) for path in home.rglob("*") if path.is_dir())
    locations[0].chmod(0o555)
    try:
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        # The directory to make writable is the location that lost it, not the
        # user's cache directory above it, which the warning named.
        assert (f"no file can be written in that directory, {locations[0]} (Permission denied), so make room there, "
                "or make it writable") in run.stderr, run.stderr
    finally:
        locations[0].chmod(0o755)


@pytest.mark.skipif(os.name == "nt", reason="a symlink needs a privilege on Windows")
def test_symlinks_that_cycle_inside_the_package_are_walked_once(tmp_path):
    # Two links pointing up the package, followed with no memory of where the
    # walk had been, gave an exponential number of paths before the file
    # system's link limit, and the import did not finish; the base imported at
    # once. Each real directory is walked once.
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    (site / "numbox" / "core" / "up").symlink_to("..", target_is_directory=True)
    (site / "numbox" / "utils" / "up").symlink_to("..", target_is_directory=True)
    env = dict(os.environ, PYTHONPATH=str(site), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", IMPORT_LIBM], capture_output=True,
                         text=True, env=env, cwd=str(tmp_path), timeout=120)
    assert run.returncode == 0 and str(site) in run.stdout, run.stderr


@needs_a_directory_it_cannot_write
def test_a_symlinked_directory_of_the_package_answers_too(tmp_path):
    # A directory of the package reached through a symlink was not walked, so a
    # read-only one behind the link, with no user cache, passed the check and
    # died at libm's first binding.
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    elsewhere = tmp_path / "elsewhere"
    shutil.move(str(site / "numbox" / "core" / "bindings"), str(elsewhere))
    (site / "numbox" / "core" / "bindings").symlink_to(elsewhere, target_is_directory=True)
    home = tmp_path / "home"
    home.mkdir()
    read_only = [home, elsewhere, *(path for path in elsewhere.rglob("*") if path.is_dir())]
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
    finally:
        for path in read_only:
            path.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_a_frozen_application_is_told_its_user_cache_directory(tmp_path):
    # With sys.frozen set numba caches a source that is not on disk in the
    # user's cache directory, as a .zip; where that directory cannot be written
    # the remedy is to make it so, not the sources or a .zip the archive text
    # offered.
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    assert compileall.compile_dir(str(site), quiet=1, legacy=True)
    for source in list(site.rglob("*.py")):
        source.unlink()
    home = tmp_path / "home"
    home.mkdir()
    frozen = "import sys\nsys.frozen = True\n" + IMPORT_AND_USE
    env = dict(os.environ, PYTHONPATH=str(site), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
    env.pop("NUMBA_CACHE_DIR", None)
    env.pop("NUMBOX_JIT_OPTIONS", None)
    warm = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", frozen], capture_output=True,
                          text=True, env=env, cwd=str(tmp_path))
    assert warm.returncode == 0 and _index_files(home), warm.stderr
    read_only = [home, *(path for path in home.rglob("*") if path.is_dir())]
    for path in read_only:
        path.chmod(0o555)
    try:
        run = subprocess.run([sys.executable, "-W", "always", "-c", frozen], capture_output=True, text=True,
                             env=env, cwd=str(tmp_path))
        assert run.returncode == 0 and str(site) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "frozen application" in run.stderr, run.stderr
        # The frozen error is the no-locator one, which names no directory,
        # so the warning names numba's: the user cache directory under the home.
        named = re.search(r"make that directory, (.+?), writable", run.stderr)
        assert named and Path(named.group(1)).is_relative_to(home), run.stderr
        assert "install numbox with its source files" not in run.stderr
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


@pytest.mark.skipif(sys.platform != "linux", reason="a path of 4096 bytes is Linux's limit")
def test_a_location_too_close_to_the_path_limit_for_numbas_files_compiles_uncached_and_is_told_so(tmp_path):
    # numba's own check of a location makes a temporary file, one without a
    # name on Linux, so a NUMBA_CACHE_DIR whose location for a module fits the
    # path limit while numba's files, of a hundred bytes and more, do not
    # passed it, and the import died at the first save. The probe makes a file
    # named as long as numba's longest for the package, and the warning says
    # what the length's remedy is. The location of the first module asked,
    # the directory's name and a hash of its path under NUMBA_CACHE_DIR, ends
    # 20 bytes short of the limit, whatever the checkout's own path.
    from numba.core.caching import _CacheLocator
    location = _CacheLocator.get_suitable_cache_subpath(str(REPO / "numbox" / "core" / "configurations.py"))
    cache_dir = _directory_of_length(tmp_path, 4096 - 20 - 1 - len(location))
    env = dict(os.environ, PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(cache_dir))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(REPO / "numbox" / "core") in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert "the path is too long for the file system: a shorter NUMBA_CACHE_DIR; or NUMBOX_JIT_OPTIONS" in (
        run.stderr), run.stderr
    assert not _index_files(cache_dir)


@needs_a_directory_it_cannot_write
@pytest.mark.skipif(sys.platform != "linux", reason="a path of 4096 bytes is Linux's limit")
def test_a_read_only_install_whose_user_cache_directory_is_too_long_is_told_to_shorten_that(tmp_path):
    # Where the source's __pycache__ cannot be written and NUMBA_CACHE_DIR is
    # not set, numba caches under the user's cache directory, and a location
    # there too long for numba's files was told "a shorter NUMBA_CACHE_DIR, or
    # none, where it is set, else numbox installed at a shorter path": a
    # variable that was not set and a reinstall that changes nothing. The first
    # module asked has its location there end 20 bytes short of the limit.
    from numba.core.caching import _CacheLocator
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbox", site / "numbox", ignore=shutil.ignore_patterns("__pycache__"))
    location = _CacheLocator.get_suitable_cache_subpath(str(site / "numbox" / "core" / "configurations.py"))
    xdg = _directory_of_length(tmp_path / "xdg", 4096 - 20 - 1 - len(location) - len("/numba"))
    read_only = [path for path in site.rglob("*") if path.is_dir()]
    for path in read_only:
        path.chmod(0o555)
    try:
        env = dict(os.environ, PYTHONPATH=str(site), XDG_CACHE_HOME=str(xdg))
        env.pop("NUMBA_CACHE_DIR", None)
        env.pop("NUMBOX_JIT_OPTIONS", None)
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(site) in run.stdout, run.stderr
        assert run.stderr.count("compiles without a cache") == 1, run.stderr
        assert "at a shorter path, through XDG_CACHE_HOME or HOME" in run.stderr, run.stderr
        assert "a shorter NUMBA_CACHE_DIR" not in run.stderr and "numbox installed" not in run.stderr, run.stderr
        cured = _run(dict(env, XDG_CACHE_HOME=str(tmp_path / "short")), tmp_path, warnings="error")
        assert cured.returncode == 0, cured.stderr
        assert _index_files(tmp_path / "short")
    finally:
        for path in read_only:
            path.chmod(0o755)


# Each case: where NUMBA_CACHE_DIR is, where the source is, and which of numba's
# locations for it is too long; "user" stands for the user's cache directory.
LOCATION_CASES = {
    "NUMBA_CACHE_DIR": ("cache", "site", "NUMBA_CACHE_DIR"),
    "beside the source": (None, "site", "beside"),
    "the user's cache directory": (None, "site", "user"),
    "beside the source, NUMBA_CACHE_DIR passed over": ("cache", "site", "beside"),
    "beside a source under NUMBA_CACHE_DIR": ("cache", "cache/site", "beside"),
    "the user's cache directory under NUMBA_CACHE_DIR": ("user-cache", "site", "user"),
    "beside a source under the user's cache directory": (None, "user-cache/numba/site", "beside"),
}


@pytest.mark.parametrize("named", ["the location", "a file in it"])
@pytest.mark.parametrize("case", list(LOCATION_CASES))
def test_a_location_too_long_is_told_what_shortens_the_one_it_is(tmp_path, monkeypatch, case, named):
    # numba takes a directory under NUMBA_CACHE_DIR where that is set, else the
    # __pycache__ beside the source, else a directory under the user's cache
    # directory, and the error names the one that is too long. One remedy for
    # all three named a variable that was not set and a reinstall that changes
    # nothing where the location was under the user's cache directory, and
    # telling them apart by which directory a location starts with took one for
    # another where they nest: an install under either cache directory, or
    # NUMBA_CACHE_DIR above the user's. The user's cache directory is one under
    # tmp_path, so that every case runs on every platform. numbox's own check
    # names the location; numba's first save names a file in it, and a caller
    # can pass that error on.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    cache_dir, source_dir, which = LOCATION_CASES[case]
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / cache_dir) if cache_dir else "")
    py_file = tmp_path / source_dir / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
    user_cache_dir = tmp_path / "user-cache" / "numba"
    # A location other than NUMBA_CACHE_DIR's where the variable is set means
    # numba passed it over, so the warning names it rather than telling the
    # reader to set what is set.
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    if cache_dir:
        instead = (f", or NUMBA_CACHE_DIR, which is set to {tmp_path / cache_dir} and numba could not use, made a "
                   "writable directory at a short path")
    else:
        instead = ", or NUMBA_CACHE_DIR set to a short path"
    where, cure = {
        "NUMBA_CACHE_DIR": (tmp_path / "cache" / subpath, "a shorter NUMBA_CACHE_DIR"),
        "beside": (py_file.parent / "__pycache__", f"numbduck installed at a shorter path{instead}"),
        "user": (user_cache_dir / subpath,
                 f"the user's cache directory, {user_cache_dir}, at a shorter path{configurations._moved_through()}"
                 f"{instead}"),
    }[which]
    if named == "a file in it":
        where = where / "module.f-1.py312.nbi"
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(where))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence", package="numbduck")
    assert remedy == f"the path is too long for the file system: {cure}; or silence", remedy


def test_a_relative_numba_cache_dir_is_matched_as_the_location_numba_names(tmp_path, monkeypatch):
    # numba joins NUMBA_CACHE_DIR as set to the name it gives the source's
    # directory, so set relative, to the directory the process runs from, the
    # location is relative, and so is the path numba's error, and the
    # package's own, names. The remedy writes the locations absolute, and the
    # path as named matched none of them, so it was told as a location that
    # is none of numba's, with the variable to be made a writable directory
    # at a short path, where a shorter NUMBA_CACHE_DIR shortens it.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "cache")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    location = os.path.join("cache", _CacheLocator.get_suitable_cache_subpath(str(py_file)))
    failure = OSError(errno.ENAMETOOLONG, "File name too long", location)
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == "the path is too long for the file system: a shorter NUMBA_CACHE_DIR; or silence", remedy


@pytest.mark.parametrize("locators, cure", [
    ("", "a shorter NUMBA_CACHE_DIR"),
    ("UserWideCacheLocator", "the user's cache directory, {user_cache_dir}, at a shorter path{moved_through}"),
    ("UserWideCacheLocator,UserProvidedCacheLocator",
     "the user's cache directory, {user_cache_dir}, at a shorter path{moved_through}"),
    ("InTreeCacheLocator,UserProvidedCacheLocator,UserWideCacheLocator", "a shorter NUMBA_CACHE_DIR"),
])
def test_a_location_too_long_that_is_numba_cache_dirs_and_the_users_cache_directorys_is_told_by_the_locator_order(
        tmp_path, monkeypatch, locators, cure):
    # NUMBA_CACHE_DIR set to the user's cache directory makes one path of two
    # of numba's locations, and which locator took it is the order's to say:
    # numba's own order tries the variable first, so a shorter NUMBA_CACHE_DIR
    # shortens it; a list that tries the user's cache directory before the
    # variable, or never reads the variable, took it as the user's cache
    # directory, and the variable's remedy, matched first whatever the order,
    # told the reader to shorten a variable numba did not read.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    user_cache_dir = tmp_path / "user-cache" / "numba"
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(user_cache_dir))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    location = user_cache_dir / _CacheLocator.get_suitable_cache_subpath(str(py_file))
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(location))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    expected = cure.format(user_cache_dir=user_cache_dir, moved_through=configurations._moved_through())
    assert remedy == f"the path is too long for the file system: {expected}; or silence", remedy


@pytest.mark.parametrize("cache_dir_set", [False, True])
@pytest.mark.parametrize("locators, offered", [
    ("", True),
    ("IPythonCacheLocator,UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator", True),
    ("InTreeCacheLocator,UserProvidedCacheLocator", False),
])
def test_a_location_too_long_that_is_none_of_numbas_is_named_with_the_variable_numba_tries_first(
        tmp_path, monkeypatch, locators, offered, cache_dir_set):
    # A caller can pass on an error naming a location that is none of the three
    # numba builds from the file. The remedy names it and offers NUMBA_CACHE_DIR
    # where numba tries the variable before any other location, the IPython
    # locator ahead taking no file on disk; it read "NUMBA_CACHE_DIR set to a
    # short path, which numba takes first", the alternative's clause cut from
    # its "or" and nothing said of the location. Set, the variable is named
    # without the claim that numba passed it over: a location that is none of
    # numba's shows nothing of what numba did with the variable, where the
    # remedy said it was one "numba could not use", as it says of a set
    # variable passed over for a location numba took. With another locator
    # ahead of the variable's, the variable is not offered, and no case here
    # held that: the offer with the position dropped passed every one.
    import numba
    from numbox.core.configurations import cache_remedy
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir) if cache_dir_set else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / "package" / "module.py"
    py_file.parent.mkdir()
    py_file.write_text("")
    elsewhere = tmp_path / "elsewhere" / "cache"
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(elsewhere))
    remedy = cache_remedy(str(py_file), failure, "silence")
    if not offered:
        instead = ""
    elif cache_dir_set:
        instead = f", or NUMBA_CACHE_DIR, which is set to {cache_dir}, made a writable directory at a short path"
    else:
        instead = ", or NUMBA_CACHE_DIR set to a short path"
    assert remedy == (
        f"the path is too long for the file system: that location, {elsewhere}, at a shorter path{instead}; or silence"
    ), remedy


@pytest.mark.parametrize("filename", [None, ""])
@pytest.mark.parametrize("cache_dir", [None, "cache", "user-cache/numba"])
def test_a_location_too_long_that_the_error_does_not_name_is_told_the_locations_numba_could_have_taken(
        tmp_path, monkeypatch, cache_dir, filename):
    # numba's errors and the package's own check name the file refused, but a
    # caller can pass on an ENAMETOOLONG that names nothing. The remedy then
    # cannot tell which location numba took, and it lists the locations numba
    # could have taken for the file, in numba's order, for the reader to put
    # the one that is too long at a shorter path; it read "that location,
    # None, at a shorter path", and matched the empty name against the working
    # directory and its parent, which could have taken one location for
    # another. NUMBA_CACHE_DIR set to the user's cache directory makes one
    # path of two locations, and the list named it twice. An empty name is no
    # name either, and it read "that location, , at a shorter path".
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    cache_dir = tmp_path / cache_dir if cache_dir else None
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir) if cache_dir else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    # The working directory is the source's __pycache__, so an empty name
    # matched against it would take the in-tree location for the one named.
    (py_file.parent / "__pycache__").mkdir()
    monkeypatch.chdir(py_file.parent / "__pycache__")
    subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
    locations = [str(py_file.parent / "__pycache__"), str(tmp_path / "user-cache" / "numba" / subpath)]
    if cache_dir:
        locations.insert(0, str(cache_dir / subpath))
        locations = list(dict.fromkeys(locations))
        instead = f", or NUMBA_CACHE_DIR, which is set to {cache_dir}, made a writable directory at a short path"
    else:
        instead = ", or NUMBA_CACHE_DIR set to a short path"
    failure = OSError(errno.ENAMETOOLONG, "File name too long", filename)
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == (
        f"the path is too long for the file system: the location numba took, one of {', '.join(locations[:-1])} or "
        f"{locations[-1]}, at a shorter path{instead}; or silence"), remedy


@pytest.mark.skipif(numba_version < 62, reason="NUMBA_CACHE_LOCATOR_CLASSES arrived in numba 0.62")
@pytest.mark.parametrize("failure, told", [
    pytest.param(OSError(errno.ENAMETOOLONG, "File name too long"),
                 "the path is too long for the file system: the location numba took, {location}, at a shorter path",
                 id="too long"),
    pytest.param(OSError(errno.EACCES, "Permission denied"),
                 "the location numba took, {location}, where no file can be written (Permission denied): make room "
                 "there, or make it writable", id="unwritable"),
])
def test_an_error_that_names_no_file_under_one_locator_is_told_its_one_location(tmp_path, monkeypatch, failure, told):
    # A list of one locator leaves numba one location to take for the file, so
    # an error that names no file is told that location. The list of the
    # locations numba could have taken read "one of" before the one path.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "UserWideCacheLocator", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(str(py_file))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == f"{told.format(location=location)}; or silence", remedy


# Each case: where NUMBA_CACHE_DIR is, NUMBA_CACHE_LOCATOR_CLASSES, which of
# numba's locations the error names, and the alternative offered after it.
UNWRITABLE_CASES = {
    "NUMBA_CACHE_DIR's": ("cache", "", "NUMBA_CACHE_DIR", ", or NUMBA_CACHE_DIR set to another writable directory"),
    "beside the source, NUMBA_CACHE_DIR unset": (None, "", "beside", ", or NUMBA_CACHE_DIR set to a writable directory"),
    "beside the source, NUMBA_CACHE_DIR passed over": (
        "cache", "", "beside",
        ", or NUMBA_CACHE_DIR, which is set to {cache_dir} and numba could not use, made a writable directory at a "
        "short path"),
    "the user's cache directory, another locator first": (None, "UserWideCacheLocator", "user", ""),
    "none of numba's": (None, "", "none", ", or NUMBA_CACHE_DIR set to a writable directory"),
    "unnamed, NUMBA_CACHE_DIR unset": (None, "", "unnamed", ", or NUMBA_CACHE_DIR set to a writable directory"),
    "unnamed, NUMBA_CACHE_DIR set": (
        "cache", "", "unnamed",
        ", or NUMBA_CACHE_DIR, which is set to {cache_dir}, made a writable directory at a short path"),
}


@pytest.mark.parametrize("error", [(errno.ENOSPC, "No space left on device"), (errno.EACCES, "Permission denied")])
@pytest.mark.parametrize("case", list(UNWRITABLE_CASES))
def test_a_location_numba_took_where_no_file_can_be_written_is_told_the_location_and_the_reason(
        tmp_path, monkeypatch, case, error):
    # numba takes a location its own check could write a temporary file in,
    # and the package's check then writes a named file there, which a full
    # disk, a quota or permissions changed since can refuse. The location is
    # the one numba took, so "NUMBA_CACHE_DIR is set to X, which numba could
    # not use: set it to a writable directory at a short path", the answer for
    # a variable numba passed over, was false on both counts; the remedy names
    # the location and the reason, and offers the variable as the alternative
    # where numba tries it before that location, or set elsewhere where the
    # location is the variable's own. A location that is none of numba's for
    # the file, which a caller can pass on, is named as the error names it,
    # as the too-long remedy names one; "numba caches this file in" claimed it
    # for numba. An error naming no file, a caller's too, is told the
    # locations numba could have taken, as the too-long remedy tells them; it
    # fell past this branch to the no-locator remedy, which dropped the reason
    # and said the variable, set and writable, was one numba could not use.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    cache_dir, locators, which, offer = UNWRITABLE_CASES[case]
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / cache_dir) if cache_dir else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
    location = {
        "NUMBA_CACHE_DIR": tmp_path / "cache" / subpath,
        "beside": py_file.parent / "__pycache__",
        "user": tmp_path / "user-cache" / "numba" / subpath,
        "none": tmp_path / "elsewhere" / "cache",
        "unnamed": None,
    }[which]
    number, reason = error
    if which == "unnamed":
        failure = OSError(number, reason)
        could_have_taken = [py_file.parent / "__pycache__", tmp_path / "user-cache" / "numba" / subpath]
        if cache_dir:
            could_have_taken.insert(0, tmp_path / "cache" / subpath)
        opening = f"the location numba took, one of {', '.join(map(str, could_have_taken[:-1]))} or {could_have_taken[-1]},"
    else:
        failure = OSError(number, reason, str(location))
        opening = f"that location, {location}," if which == "none" else f"numba caches this file in {location},"
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == (
        f"{opening} where no file can be written ({reason}): make room there, or make it "
        f"writable{offer.format(cache_dir=tmp_path / 'cache')}; or silence"), remedy


@pytest.mark.parametrize("placement", [
    "a source on disk", "a .zip member", "a .zip member in an ipykernel directory, naming IPython's location",
    "a .zip member, naming a location that is none of numba's",
])
def test_an_error_with_no_strerror_is_told_its_own_text_as_the_reason(tmp_path, monkeypatch, placement):
    # is_a_cache_error admits any OSError, and a caller can pass on one built
    # from a message alone, with no errno, no strerror and no filename, or one
    # naming a file with no strerror. The reason read "(None)" and the message
    # was lost, for a source on disk; the remedies for a source not on disk
    # give the reason the same way, and nothing held them to it.
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    failure = OSError("Disk quota exceeded on /home")
    user_cache_dir = tmp_path / "user-cache" / "numba"
    not_on_disk = (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        "effect here, because the source is not a file on disk: ")
    if placement == "a source on disk":
        py_file = tmp_path / "site" / "package" / "module.py"
        py_file.parent.mkdir(parents=True)
        py_file.write_text("")
        subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
        expected = (
            f"the location numba took, one of {py_file.parent / '__pycache__'} or {user_cache_dir / subpath}, where no "
            f"file can be written ({failure}): make room there, or make it writable, or NUMBA_CACHE_DIR set to a "
            "writable directory; or silence")
    elif placement == "a .zip member":
        py_file = tmp_path / "bundle.zip" / "package" / "module.py"
        expected = not_on_disk + (
            f"no file can be written in that directory, {user_cache_dir} ({failure}), so make room there, or make it "
            "writable, or silence")
    elif placement.startswith("a .zip member in an ipykernel directory"):
        py_file = tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py"
        failure = OSError(errno.ENOSPC, None, str(ipython_dir / "numba_cache"))
        expected = (
            f"numba caches this file in {ipython_dir / 'numba_cache'}, where no file can be written ({failure}): make "
            "room there, or make it writable; or silence")
    else:
        py_file = tmp_path / "bundle.zip" / "package" / "module.py"
        failure = OSError(errno.ENOSPC, None, str(tmp_path / "elsewhere" / "cache"))
        expected = not_on_disk + (
            f"no file can be written at that location, {tmp_path / 'elsewhere' / 'cache'} ({failure}), so make room "
            "there, or make it writable, or silence")
    assert failure.strerror is None and "None" not in expected.replace(str(failure), "")
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == expected, remedy


@pytest.mark.parametrize("locators, ipython, expected", [
    pytest.param(
        "IPythonCacheLocator,InTreeCacheLocator", "IPython.paths",
        "IPython's cache directory, {ipython_dir}, at a shorter path",
        marks=pytest.mark.skipif(numba_version < 62, reason="NUMBA_CACHE_LOCATOR_CLASSES arrived in 0.62")),
    ("", "IPython.paths",
     "IPython's cache directory, {ipython_dir}, at a shorter path, or NUMBA_CACHE_DIR set to a short path"),
    ("", "IPython.utils.path",
     "IPython's cache directory, {ipython_dir}, at a shorter path, or NUMBA_CACHE_DIR set to a short path"),
    ("", None, "that location, {named}, at a shorter path, or NUMBA_CACHE_DIR set to a short path"),
])
def test_a_cell_file_numbas_ipython_locator_takes_is_told_ipythons_cache_directory(
        tmp_path, monkeypatch, locators, ipython, expected):
    # numba's IPython locator takes a file on disk in an ipykernel directory,
    # a notebook's cell file, and caches it in numba_cache under IPython's
    # cache directory, with no directory of its own per file. That location
    # too long for the file system got "that location, X", the wording for a
    # location that is none of numba's. numba imports IPython for the location
    # when it makes it and catches only OSError there, so without IPython it
    # raises ImportError at decoration, no cache error; an error a caller
    # passes on naming that location is then told it as none of numba's.
    import types
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    ipython_dir = tmp_path / "ipython"
    if ipython:
        # The module numba imports the location from: IPython.paths, or
        # IPython.utils.path in an older IPython, which numba falls back to
        # when IPython.paths does not import.
        paths = types.ModuleType(ipython)
        paths.get_ipython_cache_dir = lambda: str(ipython_dir)
        monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
        monkeypatch.setitem(sys.modules, "IPython.paths", paths if ipython == "IPython.paths" else None)
        monkeypatch.setitem(sys.modules, "IPython.utils", types.ModuleType("IPython.utils"))
        monkeypatch.setitem(sys.modules, "IPython.utils.path", paths if ipython == "IPython.utils.path" else None)
    else:
        monkeypatch.setitem(sys.modules, "IPython", None)
    py_file = tmp_path / "ipykernel_123" / "cell.py"
    py_file.parent.mkdir()
    py_file.write_text("")
    named = ipython_dir / "numba_cache" / "cell-1.py312.nbi"
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(named))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    expected = expected.format(ipython_dir=ipython_dir, named=named)
    assert remedy == f"the path is too long for the file system: {expected}; or silence", remedy


def test_a_zip_member_in_an_ipykernel_directory_is_asked_with_the_probes_source(tmp_path, monkeypatch):
    # numba's IPython locator takes a file in an ipykernel directory, on disk
    # or not, and reads the function's source when it does: from the file, or
    # through the module's loader for a member of a .zip. The probe the check
    # compiles has no module, so for such a member inspect found no source,
    # and the check raised the OSError from reading it, which is no cache
    # error and which the remedy, taking an OSError for a source not on disk
    # as the .zip's, answered with the user's cache directory made writable,
    # a directory numba never tried. The probe's source is given to inspect
    # for the time numba picks the locator, and the question goes on to
    # IPython's cache directory, as numba's does.
    import linecache
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", "def f():\n    pass\n")
    py_file = str(archive / "ipykernel_123" / "cell.py")
    configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert (ipython_dir / "numba_cache").is_dir()
    # The source is the probe's for the locator's choosing alone.
    assert py_file not in linecache.cache


def test_the_check_puts_back_the_lines_linecache_held_for_a_file_not_on_disk(tmp_path, monkeypatch):
    # A traceback through a .zip member, or inspect, leaves the member's lines
    # in the process-wide linecache under its name; the check gives the probe's
    # source to inspect under that name for the time numba picks the locator,
    # and what was there is owed back. Nothing failed with the entry discarded
    # instead, every check so far asking about a file nothing had read.
    import linecache
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", "def cell():\n    return 1\n")
    py_file = str(archive / "ipykernel_123" / "cell.py")
    held = (24, None, ["def cell():\n", "    return 1\n"], py_file)
    monkeypatch.setitem(linecache.cache, py_file, held)
    seen = []
    real = configurations.CompileResultCacheImpl

    def recording(probe):
        seen.append(linecache.getlines(py_file))
        return real(probe)

    monkeypatch.setattr(configurations, "CompileResultCacheImpl", recording)
    configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert seen == [["def _cache_probe():\n", "    pass\n"]], seen
    assert linecache.cache[py_file] is held
    assert (ipython_dir / "numba_cache").is_dir()


@pytest.mark.parametrize("locators, made", [
    pytest.param("", "__pycache__", id="numba's order, the in-tree locator takes it"),
    pytest.param("IPythonCacheLocator,InTreeCacheLocator", "numba_cache", id="the IPython locator first",
                 marks=pytest.mark.skipif(numba_version < 62, reason="NUMBA_CACHE_LOCATOR_CLASSES arrived in 0.62")),
])
def test_a_cell_file_on_disk_is_asked_without_touching_linecache(tmp_path, monkeypatch, locators, made):
    # inspect reads a file on disk itself, so the probe's source is given it
    # for a file not on disk alone. Given for every file, it stood in for the
    # real file's lines in the process-wide linecache while numba picked the
    # locator, and a locator reading another function's source from the file
    # meanwhile, as numba's IPython locator reads the function's own, read the
    # probe's two lines instead. numba's own order tries the in-tree locator
    # before the IPython one, which takes a cell file on disk only listed
    # ahead of it.
    import linecache
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    py_file = tmp_path / "ipykernel_123" / "cell.py"
    py_file.parent.mkdir()
    py_file.write_text("def cell():\n    return 1\n")
    lines = linecache.getlines(str(py_file))
    assert lines == ["def cell():\n", "    return 1\n"]
    seen = []
    real = configurations.CompileResultCacheImpl

    def recording(probe):
        seen.append(linecache.getlines(str(py_file)))
        return real(probe)

    monkeypatch.setattr(configurations, "CompileResultCacheImpl", recording)
    configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert seen == [lines] and linecache.getlines(str(py_file)) == lines, seen
    location = py_file.parent / "__pycache__" if made == "__pycache__" else ipython_dir / "numba_cache"
    assert location.is_dir()


def test_the_probes_source_is_given_under_a_lock_for_a_file_not_on_disk(tmp_path, monkeypatch):
    # Two checks of one file not on disk that overlapped put the probe's
    # source back for each other, and the process-wide linecache then held it
    # for the file for good, an entry with no file to check it against. The
    # source is given under a lock, one check at a time, so that each check
    # puts back what it found: a second check waits at the lock while the
    # first holds it.
    import inspect
    import linecache
    import threading
    import types
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    py_file = str(tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py")
    real = configurations.CompileResultCacheImpl
    inside = {"first": threading.Event(), "second": threading.Event()}
    hold = threading.Event()
    seen = {}

    def holding(probe):
        name = threading.current_thread().name
        seen[name] = (configurations._probe_source.locked(), inspect.getsource(probe))
        inside[name].set()
        if name == "first":
            hold.wait(5)
        return real(probe)

    monkeypatch.setattr(configurations, "CompileResultCacheImpl", holding)
    checks = [threading.Thread(target=configurations.check_cache_location, args=(py_file,), name=name)
              for name in ("first", "second")]
    checks[0].start()
    assert inside["first"].wait(5)
    checks[1].start()
    # The second check waits at the lock while the first holds it.
    assert not inside["second"].wait(0.5)
    hold.set()
    for check in checks:
        check.join(5)
    assert inside["second"].is_set()
    assert seen["first"] == (True, "def _cache_probe():\n    pass\n"), seen
    assert seen["second"] == (True, "def _cache_probe():\n    pass\n"), seen
    assert py_file not in linecache.cache


@pytest.mark.parametrize("refused, told", [
    pytest.param(("cell-1.py312.nbi.tmp.0123456789abcdef", errno.ENAMETOOLONG, "File name too long"),
                 "the path is too long for the file system: IPython's cache directory, {ipython_dir}, at a shorter path",
                 id="too long, a file in it"),
    pytest.param(("", errno.EACCES, "Permission denied"),
                 "numba caches this file in {numba_cache}, where no file can be written (Permission denied): make room "
                 "there, or make it writable", id="unwritable, the location"),
])
def test_a_zip_member_in_an_ipykernel_directory_refused_ipythons_location_is_told_ipythons_cache_directory(
        tmp_path, monkeypatch, refused, told):
    # numba's IPython locator takes a member of a .zip in an ipykernel directory
    # before the .zip locator is reached, and caches it in numba_cache under
    # IPython's cache directory, which the error names, or a file numba writes
    # in it. The remedy, taking every OSError for a source not on disk as the
    # .zip's, told that location as the user's cache directory, with what moves
    # that directory, where it is IPython's and the remedy is the one a cell
    # file on disk gets there, with no NUMBA_CACHE_DIR to offer for a source
    # not on disk.
    import types
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    py_file = str(tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py")
    name, number, strerror = refused
    named = ipython_dir / "numba_cache" / name if name else ipython_dir / "numba_cache"
    remedy = configurations.cache_remedy(py_file, OSError(number, strerror, str(named)), "silence")
    told = told.format(ipython_dir=ipython_dir, numba_cache=ipython_dir / "numba_cache")
    assert remedy == f"{told}; or silence", remedy


def test_a_zip_members_error_is_answered_without_asking_ipython_for_its_cache_directory(tmp_path, monkeypatch):
    # numba's IPython locator takes a file in an ipykernel directory alone, so
    # a .zip member anywhere else is never cached in IPython's cache directory;
    # the remedy asked IPython for that directory all the same, to compare the
    # error's path against it, and IPython, with its own directory unwritable,
    # under a read-only home, warned on every import of the package and left a
    # temporary directory behind. The directory is asked for where numba's
    # IPython locator takes the file and is listed.
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    def never():
        raise AssertionError("IPython was asked for its cache directory")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = never
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    py_file = str(tmp_path / "bundle.zip" / "package" / "module.py")
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(py_file)
    remedy = configurations.cache_remedy(py_file, OSError(errno.EACCES, "Permission denied", str(location)), "silence")
    # numba's .zip locator arrived in 0.61; before it, a .zip member has no
    # location of numba's, and the one the error names is told as named.
    told = (f"no file can be written in that directory, {location} (Permission denied)" if numba_version >= 61 else
            f"no file can be written at that location, {location} (Permission denied)")
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: {told}, so make room there, or make it writable, "
        "or silence"), remedy


@pytest.mark.parametrize("platform, moved_through", [
    ("linux", ", through XDG_CACHE_HOME or HOME"), ("darwin", ", through HOME"), ("win32", ""),
])
def test_the_user_cache_directory_is_told_what_moves_it_on_the_platform(tmp_path, monkeypatch, platform, moved_through):
    # numba's appdirs puts the user's cache directory under XDG_CACHE_HOME or
    # HOME on Linux, under HOME alone on macOS, and asks the system on Windows,
    # so "through XDG_CACHE_HOME or HOME" sent a macOS reader to a variable that
    # moves nothing, for a source on disk and for a .zip alike.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = tmp_path / "package" / "module.py"
    py_file.parent.mkdir()
    py_file.write_text("")
    user_cache_dir = configurations.AppDirs(appname="numba", appauthor=False).user_cache_dir
    location = os.path.join(user_cache_dir, _CacheLocator.get_suitable_cache_subpath(str(py_file)))
    monkeypatch.setattr(sys, "platform", platform)
    on_disk = configurations.cache_remedy(
        str(py_file), OSError(errno.ENAMETOOLONG, "File name too long", location), "silence")
    assert f"the user's cache directory, {user_cache_dir}, at a shorter path{moved_through}, or NUMBA_CACHE_DIR" in (
        on_disk), on_disk
    if numba_version < 61:
        # numba's .zip locator arrived in 0.61; before it, a .zip member has
        # no location of numba's under the user's cache directory.
        return
    zip_member = str(tmp_path / "package.zip" / "package" / "module.py")
    zip_location = os.path.join(user_cache_dir, _CacheLocator.get_suitable_cache_subpath(zip_member))
    zipped = configurations.cache_remedy(
        zip_member, OSError(errno.ENAMETOOLONG, "File name too long", zip_location), "silence")
    assert f"put that directory, {zip_location}, at a shorter path{moved_through}, or silence" in zipped, zipped


@pytest.mark.parametrize("locators, offered", [
    ("InTreeCacheLocator,UserWideCacheLocator", False),
    ("numba.core.caching.UserProvidedCacheLocator, numba.core.caching.InTreeCacheLocator", True),
    ("IPythonCacheLocator,UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator", True),
    ("InTreeCacheLocator,UserProvidedCacheLocator,UserWideCacheLocator", False),
    pytest.param(
        "UserProvidedCacheLocator,InTreeCacheLocatorFsAgnostic,UserWideCacheLocator", True,
        marks=pytest.mark.skipif(numba_version < 62, reason="InTreeCacheLocatorFsAgnostic arrived in numba 0.62")),
    pytest.param(
        "InTreeCacheLocatorFsAgnostic", False,
        marks=pytest.mark.skipif(numba_version < 62, reason="InTreeCacheLocatorFsAgnostic arrived in numba 0.62")),
])
def test_numba_cache_dir_is_offered_only_where_numba_tries_it_before_the_location_it_took(
        tmp_path, monkeypatch, locators, offered):
    # numba 0.62 and later take NUMBA_CACHE_LOCATOR_CLASSES as the locators and
    # their order, each entry a class of numba's caching module by its name or
    # its dotted path. NUMBA_CACHE_DIR is an alternative to the __pycache__
    # numba took only where numba tries the variable before that: a set one
    # was passed over, an unset one would be taken. The IPython and .zip
    # locators take no plain file on disk, so one of them ahead changes
    # nothing, where the first entry's name alone said another locator was
    # first. numba's InTreeCacheLocatorFsAgnostic caches in the __pycache__ as
    # its parent does, and keyed by name alone it was no in-tree locator, so
    # its too-long __pycache__ got "that location" and no cure.
    import numba
    from numbox.core.configurations import cache_remedy
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / "package" / "module.py"
    py_file.parent.mkdir()
    py_file.write_text("")
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(py_file.parent / "__pycache__"))
    remedy = cache_remedy(str(py_file), failure, "silence", package="numbduck")
    instead = (f", or NUMBA_CACHE_DIR, which is set to {tmp_path / 'cache'} and numba could not use, made a writable "
               "directory at a short path") if offered else ""
    assert remedy == f"the path is too long for the file system: numbduck installed at a shorter path{instead}; or silence", (
        remedy)


# Each case: whether NUMBA_CACHE_DIR is set, NUMBA_CACHE_LOCATOR_CLASSES, and
# the remedy before "or silence".
NO_LOCATOR_CASES = {
    "unset": (False, "", "Set NUMBA_CACHE_DIR to a writable directory"),
    "set, passed over": (
        True, "",
        "NUMBA_CACHE_DIR is set to {cache_dir}, which numba could not use: set it to a writable directory at a short "
        "path"),
    "without the user-provided locator": (
        False, "InTreeCacheLocator,UserWideCacheLocator",
        "numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, {locators}, says: make one of those locations writable"),
    "unset, the user-provided locator after another": (
        False, "InTreeCacheLocator,UserProvidedCacheLocator", "Set NUMBA_CACHE_DIR to a writable directory"),
    "set, passed over, the user-provided locator after another": (
        True, "InTreeCacheLocator,UserProvidedCacheLocator",
        "NUMBA_CACHE_DIR is set to {cache_dir}, which numba could not use: set it to a writable directory at a short "
        "path"),
    "only locators that take no plain file on disk": (
        False, "IPythonCacheLocator,ZipCacheLocator",
        "numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, {locators}, says, and none of those locators takes a "
        "source file on disk: list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or "
        "UserWideCacheLocator"),
}


@pytest.mark.parametrize("setting", list(NO_LOCATOR_CASES))
def test_no_locator_for_a_source_on_disk_is_told_what_numba_would_take(tmp_path, monkeypatch, setting):
    # numba raises "no locator available" where it passed every location over.
    # "Set NUMBA_CACHE_DIR to a writable directory" was the answer whatever the
    # setting: to a NUMBA_CACHE_DIR too deep for numba to make its directory in,
    # where a writable one as deep cures nothing, and to a locator list numba
    # 0.62 and later read from NUMBA_CACHE_LOCATOR_CLASSES without the
    # user-provided locator, where numba never reads the variable. With that
    # locator anywhere in the list numba reads the variable once the locators
    # before it have passed, so its place in the list changes nothing here. A
    # list of numba's IPython and .zip locators alone has no locator for a
    # plain file on disk, and "make one of those locations writable" asked for
    # what no listing can give.
    import numba
    from numbox.core.configurations import cache_remedy
    py_file = tmp_path / "package" / "module.py"
    py_file.parent.mkdir()
    py_file.write_text("")
    cache_dir = str(tmp_path / ("d" * 200))
    cache_dir_set, locators, opening = NO_LOCATOR_CASES[setting]
    monkeypatch.setattr(numba.config, "CACHE_DIR", cache_dir if cache_dir_set else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    remedy = cache_remedy(str(py_file), failure, "silence")
    assert remedy == f"{opening.format(cache_dir=cache_dir, locators=locators)}, or silence", remedy


@pytest.mark.skipif(numba_version < 62, reason="NUMBA_CACHE_LOCATOR_CLASSES arrived in numba 0.62")
def test_a_location_too_long_under_the_zip_locator_listed_first_is_the_users_cache_directorys(tmp_path, monkeypatch):
    # numba's .zip locator takes any path with ".zip" in it before every
    # locator after it, and caches one with a part ending in ".zip", a
    # directory of that name included, under the user's cache directory.
    # Listed first it leaves
    # NUMBA_CACHE_DIR unread for such a path, where the remedy, holding that
    # locator to take no file on disk, left it out of the order and offered
    # the variable.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        numba.config, "CACHE_LOCATOR_CLASSES",
        "ZipCacheLocator,UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator", raising=False)
    py_file = tmp_path / "bundle.zip" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    user_cache_dir = tmp_path / "user-cache" / "numba"
    location = user_cache_dir / _CacheLocator.get_suitable_cache_subpath(str(py_file))
    failure = OSError(errno.ENAMETOOLONG, "File name too long", str(location))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == (
        f"the path is too long for the file system: the user's cache directory, {user_cache_dir}, at a shorter path"
        f"{configurations._moved_through()}; or silence"), remedy


from_061 = pytest.mark.skipif(numba_version < 61, reason="numba's .zip locator arrived in 0.61")
from_062 = pytest.mark.skipif(numba_version < 62, reason="NUMBA_CACHE_LOCATOR_CLASSES arrived in numba 0.62")


# Each case: whether NUMBA_CACHE_DIR is set, NUMBA_CACHE_LOCATOR_CLASSES, and
# the remedy before "or silence".
@pytest.mark.parametrize("cache_dir_set, locators, expected", [
    (False, "", "Set NUMBA_CACHE_DIR to a writable directory"),
    pytest.param(
        False, "ZipCacheLocator,InTreeCacheLocator,UserWideCacheLocator",
        'numba\'s .zip locator, which NUMBA_CACHE_LOCATOR_CLASSES puts before every locator for a source file on disk, '
        'takes this file for the ".zip" in its path and finds no archive there: list UserProvidedCacheLocator, with '
        "NUMBA_CACHE_DIR set, InTreeCacheLocator or UserWideCacheLocator before it",
        marks=from_062),
    pytest.param(
        True, "InTreeCacheLocator,ZipCacheLocator,UserProvidedCacheLocator",
        'numba\'s .zip locator takes this file for the ".zip" in its path and finds no archive there, after every '
        "locator NUMBA_CACHE_LOCATOR_CLASSES, {locators}, puts before it passed the file over: make one of those "
        "locations writable, or list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or "
        "UserWideCacheLocator before it",
        marks=from_062),
    pytest.param(
        True, "UserProvidedCacheLocator,ZipCacheLocator,InTreeCacheLocator",
        "NUMBA_CACHE_DIR is set to {cache_dir}, which numba could not use: set it to a writable directory at a short "
        "path",
        marks=from_062),
    pytest.param(
        False, "UserProvidedCacheLocator,ZipCacheLocator,InTreeCacheLocator",
        "Set NUMBA_CACHE_DIR to a writable directory", marks=from_062),
])
def test_a_source_on_disk_with_zip_in_its_path_and_no_archive_is_told_by_the_zip_locators_place(
        tmp_path, monkeypatch, cache_dir_set, locators, expected):
    # numba's .zip locator takes a path with ".zip" in it and raises ValueError
    # where no part of the path ends in .zip. numba tries the locators before
    # it and none after, so the error means the ones before it passed the file
    # over: in numba's own order that is every locator for a file on disk, and
    # the no-locator remedy applies; listed first it takes the file first, and
    # the remedy offered NUMBA_CACHE_DIR, which numba never reaches; listed
    # after one locator for a file on disk and before the user-provided one,
    # it raised before numba read NUMBA_CACHE_DIR, and the remedy, answering
    # the error only where the .zip locator came first, said the variable, set
    # and writable, was one numba could not use. The user-provided locator
    # takes a file only with NUMBA_CACHE_DIR set, so listing it is asked for
    # with the variable set, as the no-locator remedy asks; without that, the
    # reader who listed it first saw the same error again.
    import numba
    from numbox.core.configurations import cache_remedy
    cache_dir = str(tmp_path / "cache")
    monkeypatch.setattr(numba.config, "CACHE_DIR", cache_dir if cache_dir_set else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / "not.zipped" / "module.py"
    py_file.parent.mkdir()
    py_file.write_text("")
    remedy = cache_remedy(str(py_file), ValueError("No zip file found in path"), "silence")
    assert remedy == f"{expected.format(cache_dir=cache_dir, locators=locators)}, or silence", remedy


@from_062
@pytest.mark.parametrize("failure", [
    pytest.param(OSError(errno.ENAMETOOLONG, "File name too long"), id="too long, unnamed"),
    pytest.param(OSError(errno.EACCES, "Permission denied"), id="unwritable, unnamed"),
    pytest.param(OSError(errno.ENAMETOOLONG, "File name too long", "/elsewhere/cache"), id="too long, named"),
])
def test_an_error_under_a_locator_list_with_no_locator_that_takes_the_file_is_told_the_list(
        tmp_path, monkeypatch, failure):
    # A list with no locator that takes a source file on disk leaves numba no
    # location for it, and numba's error is the no-locator one, which is told
    # the list. An OSError a caller passes on under such a list was answered
    # as if numba had taken a location: unnamed, "the location numba took"
    # with no location to list after it, and named, that location at a
    # shorter path, though the list keeps numba from caching the file at all.
    import numba
    from numbox.core.configurations import cache_remedy
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "IPythonCacheLocator,ZipCacheLocator", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    assert cache_remedy(str(py_file), failure, "silence") == (
        "numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, IPythonCacheLocator,ZipCacheLocator, says, and none of "
        "those locators takes a source file on disk: list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, "
        "InTreeCacheLocator or UserWideCacheLocator, or silence"), cache_remedy(str(py_file), failure, "silence")


@from_062
@pytest.mark.parametrize("failure", [
    pytest.param(OSError(errno.ENAMETOOLONG, "File name too long"), id="too long, unnamed"),
    pytest.param(OSError(errno.EACCES, "Permission denied"), id="unwritable, unnamed"),
    pytest.param(OSError(errno.ENAMETOOLONG, "File name too long", "/elsewhere/cache"), id="too long, named"),
])
def test_an_error_under_the_user_provided_locator_alone_with_numba_cache_dir_unset_is_told_to_set_it(
        tmp_path, monkeypatch, failure):
    # numba's user-provided locator takes a file only with NUMBA_CACHE_DIR set,
    # so a list of that locator alone leaves numba no locator while the
    # variable is unset, and numba's error is the no-locator one, told to set
    # the variable. An OSError a caller passes on under that list counted the
    # locator as taking the file, and was told "the location numba took" with
    # no location after it, or the location it named at a shorter path, where
    # numba took none.
    import numba
    from numbox.core.configurations import cache_remedy
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "UserProvidedCacheLocator", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    no_locator = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    told = cache_remedy(str(py_file), no_locator, "silence")
    assert told == "Set NUMBA_CACHE_DIR to a writable directory, or silence", told
    assert cache_remedy(str(py_file), failure, "silence") == told, cache_remedy(str(py_file), failure, "silence")


ON_DISK = ("install numbox with its source files on disk, unpacked from any archive")
LIST_ON_DISK = (", and list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or UserWideCacheLocator "
                "in NUMBA_CACHE_LOCATOR_CLASSES, {locators}, which has none of them")
A_ZIP = ("import it from a .zip holding its source files, which numba 0.61 and later cache in the user's cache directory")
LIST_A_ZIP = " through ZipCacheLocator, once NUMBA_CACHE_LOCATOR_CLASSES, {locators}, lists it"
SET_THE_VARIABLE = (", with NUMBA_CACHE_DIR set, which UserProvidedCacheLocator, the one locator for a file on disk in "
                    "NUMBA_CACHE_LOCATOR_CLASSES, {locators}, takes nothing without")


@from_062
@pytest.mark.parametrize("locators, cache_dir_set, on_disk, a_zip", [
    pytest.param("", False, ON_DISK, A_ZIP, id="numba's order"),
    pytest.param("UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator", False, ON_DISK,
                 A_ZIP + LIST_A_ZIP, id="no .zip locator"),
    pytest.param("ZipCacheLocator", False, ON_DISK + LIST_ON_DISK, A_ZIP, id="no locator for a file on disk"),
    pytest.param("IPythonCacheLocator", False, ON_DISK + LIST_ON_DISK, A_ZIP + LIST_A_ZIP, id="neither"),
    pytest.param("UserProvidedCacheLocator,ZipCacheLocator", False, ON_DISK + SET_THE_VARIABLE, A_ZIP,
                 id="the user-provided locator alone for a file on disk, NUMBA_CACHE_DIR unset"),
    pytest.param("UserProvidedCacheLocator,ZipCacheLocator", True, ON_DISK, A_ZIP,
                 id="the user-provided locator alone for a file on disk, NUMBA_CACHE_DIR set"),
    pytest.param("InTreeCacheLocator,ZipCacheLocator", False, ON_DISK, A_ZIP,
                 id="the in-tree locator alone for a file on disk"),
    pytest.param("UserWideCacheLocator,ZipCacheLocator", False, ON_DISK, A_ZIP,
                 id="the user-wide locator alone for a file on disk"),
])
def test_an_import_from_an_egg_under_a_locator_list_is_offered_each_way_with_the_locator_it_needs(
        tmp_path, monkeypatch, locators, cache_dir_set, on_disk, a_zip):
    # The remedy for an archive numba does not cache offers the source files
    # on disk, which the three locators for a file on disk take, or a .zip
    # holding them, which the .zip locator alone takes. It offered both under
    # any list, so under one without the .zip locator it offered a .zip numba
    # then had no locator for, as under one without a locator for a file on
    # disk it offered an install numba would not cache either; and the
    # user-provided locator, alone of the three in the list, takes nothing
    # without NUMBA_CACHE_DIR, so unset it is asked for with the install,
    # where the in-tree or the user-wide one alone takes a file on disk as
    # it is, and the install is offered as it is.
    import numba
    from numbox.core.configurations import cache_remedy
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(tmp_path / "cache") if cache_dir_set else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / "numbox-0.0.0-py3.12.egg" / "numbox" / "module.py"
    failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    remedy = cache_remedy(str(py_file), failure, "silence")
    assert remedy == (
        "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, "
        f"{on_disk.format(locators=locators)}, or {a_zip.format(locators=locators)}; or silence"), remedy


@pytest.mark.skipif(numba_version < 62, reason="NUMBA_CACHE_LOCATOR_CLASSES arrived in numba 0.62")
@pytest.mark.parametrize("locators, expected", [
    ("", "NUMBA_CACHE_DIR has no effect here, because the source is not a file on disk: to cache, install numbox with "
         "its source files on disk, unpacked from any archive, or import it from a .zip holding its source files, "
         "which numba 0.61 and later cache in the user's cache directory"),
    ("UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator",
     "numba caches a .zip through ZipCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, {locators}, leaves it out: "
     "list it"),
])
def test_a_zip_import_under_a_locator_list_without_the_zip_locator_is_told_to_list_it(
        tmp_path, monkeypatch, locators, expected):
    # numba caches a .zip through its .zip locator alone, so a list that leaves
    # that locator out gives numba no locator for a source in a .zip, and the
    # error is the no-locator one. The archive remedy, which reads the list for
    # a source on disk only, told the reader to import from a .zip holding the
    # source files, which they had done, and never named the list.
    import numba
    from numbox.core.configurations import cache_remedy
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = str(tmp_path / "bundle.zip" / "numbox" / "module.py")
    failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    remedy = cache_remedy(py_file, failure, "silence")
    assert remedy == f"{expected.format(locators=locators)}; or silence", remedy


@pytest.mark.skipif(numba_version < 62, reason="NUMBA_CACHE_LOCATOR_CLASSES arrived in numba 0.62")
@pytest.mark.parametrize("locators, expected", [
    ("", "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
         "effect here, because the source is not a file on disk: make that directory, {user_cache_dir}, writable"),
    ("UserProvidedCacheLocator,InTreeCacheLocator",
     "numba caches a frozen application through UserWideCacheLocator alone, and NUMBA_CACHE_LOCATOR_CLASSES, "
     "{locators}, leaves it out: list it"),
])
def test_a_frozen_application_under_a_locator_list_without_the_user_wide_locator_is_told_to_list_it(
        tmp_path, monkeypatch, locators, expected):
    # numba's user-wide locator alone takes a frozen application, whose source
    # is not on disk, so a list that leaves it out gives numba no locator and
    # the error is the no-locator one. The frozen remedy, reading no list,
    # asked for the user's cache directory to be made writable, which it was
    # and which numba never tried.
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = str(tmp_path / "frozen" / "numbox" / "module.py")
    failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    remedy = configurations.cache_remedy(py_file, failure, "silence")
    expected = expected.format(locators=locators, user_cache_dir=tmp_path / "user-cache" / "numba")
    assert remedy == f"{expected}, or silence", remedy


NONE_OF_NUMBAS_OFF_DISK = {
    "too long": ((errno.ENAMETOOLONG, "File name too long"),
                 "the path is too long for the file system, so put that location, {named}, at a shorter path"),
    "unwritable": ((errno.EACCES, "Permission denied"),
                   "no file can be written at that location, {named} (Permission denied), so make room there, or make it "
                   "writable"),
}


@pytest.mark.parametrize("locators, in_numba_cache, refused", [
    pytest.param("ZipCacheLocator,IPythonCacheLocator", "cell-1.py312.nbi", "too long", marks=from_062,
                 id="the .zip locator first, a file in numba_cache, too long"),
    pytest.param("ZipCacheLocator,IPythonCacheLocator", "", "unwritable", marks=from_062,
                 id="the .zip locator first, numba_cache, unwritable"),
    pytest.param("", "subdir/cell-1.py312.nbi", "too long", id="numba's order, a directory numba never makes in it"),
])
def test_an_error_naming_ipythons_location_numbas_ipython_locator_did_not_take_is_told_the_location_as_named(
        tmp_path, monkeypatch, locators, in_numba_cache, refused):
    # numba's .zip locator takes a file without trying its location, so a
    # list with it before the IPython one leaves that one unreached for a
    # .zip member in an ipykernel directory, and numba caches the member in
    # the user's cache directory; the remedy answered an error a caller passes
    # on naming IPython's numba_cache with IPython's cache directory all the
    # same, a directory numba never used. And numba's IPython locator writes
    # its files in numba_cache itself, with no directory under it, so a path
    # under one is none of numba's, and was told as the .zip's directory to
    # put at a shorter path through what moves the user's cache directory,
    # which moves nothing of it. A location that is none of numba's is told
    # as the error names it, as for a source on disk.
    import types
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    py_file = str(tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py")
    named = ipython_dir / "numba_cache" / in_numba_cache if in_numba_cache else ipython_dir / "numba_cache"
    error, told = NONE_OF_NUMBAS_OFF_DISK[refused]
    remedy = configurations.cache_remedy(py_file, OSError(*error, str(named)), "silence")
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: {told.format(named=named)}, or silence"), remedy


@from_062
def test_a_zip_members_error_naming_the_zips_location_after_ipythons_locator_passed_it_over_is_told_the_location(
        tmp_path, monkeypatch):
    # numba's IPython locator passes a location it cannot make or write over,
    # and the .zip locator listed after it then takes the member and caches
    # it in the user's cache directory: an error naming that location is the
    # .zip's, under numba's own order and under one that lists the IPython
    # locator first alike.
    import types
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "IPythonCacheLocator,ZipCacheLocator", raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    py_file = str(tmp_path / "bundle.zip" / "ipykernel_123" / "cell.py")
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(py_file)
    remedy = configurations.cache_remedy(
        py_file, OSError(errno.EACCES, "Permission denied", str(location / "cell-1.py312.nbi.tmp.0123456789abcdef")),
        "silence")
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: no file can be written in that directory, {location} "
        "(Permission denied), so make room there, or make it writable, or silence"), remedy


@pytest.mark.parametrize("locators, member", [
    pytest.param("", "bundle.zip/package/module.py", id="numba's order, a .zip member in no ipykernel directory",
                 marks=from_061),
    pytest.param("ZipCacheLocator", "bundle.zip/ipykernel_123/cell.py", id="a list without the IPython locator",
                 marks=from_062),
    pytest.param("ZipCacheLocator,IPythonCacheLocator", "bundle.zip/ipykernel_123/cell.py",
                 id="the .zip locator listed first, which takes the member before the IPython one is reached",
                 marks=from_062),
])
def test_a_file_not_on_disk_numbas_ipython_locator_does_not_read_is_asked_without_touching_linecache(
        tmp_path, monkeypatch, locators, member):
    # The probe's source stands in for the file's in the process-wide
    # linecache while numba picks the locator, for numba's IPython locator,
    # which reads the function's source when it takes the file; it stood
    # there for every file not on disk, and a reader in another thread,
    # formatting a traceback through that file meanwhile, got the probe's two
    # lines for it, for a .zip member in no ipykernel directory too, which
    # that locator never takes. It stands there for a file a listed locator of
    # that family takes, and for no other.
    import linecache
    import types
    import numba
    import numba.core.caching
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(numba.core.caching, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    ipython_dir = tmp_path / "ipython"
    paths = types.ModuleType("IPython.paths")
    paths.get_ipython_cache_dir = lambda: str(ipython_dir)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.paths", paths)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(member.split("/", 1)[1], "def f():\n    pass\n")
    py_file = str(tmp_path / member)
    seen = []
    real = configurations.CompileResultCacheImpl

    def recording(probe):
        seen.append((py_file in linecache.cache, linecache.getlines(py_file)))
        return real(probe)

    monkeypatch.setattr(configurations, "CompileResultCacheImpl", recording)
    configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert seen == [(False, [])], seen
    assert (tmp_path / "user-cache" / "numba").is_dir()
    assert not ipython_dir.exists()


@from_062
@pytest.mark.parametrize("refused, told", [
    pytest.param(("__pycache__", errno.ENAMETOOLONG, "File name too long"),
                 "the path is too long for the file system: that location, {named}, at a shorter path",
                 id="too long, beside the source"),
    pytest.param(("__pycache__", errno.EACCES, "Permission denied"),
                 "that location, {named}, where no file can be written (Permission denied): make room there, or make it "
                 "writable", id="unwritable, beside the source"),
    pytest.param(("", errno.ENAMETOOLONG, "File name too long"),
                 "the path is too long for the file system: the user's cache directory, {user_cache_dir}, at a shorter "
                 "path{moved_through}", id="too long, the .zip locator's location"),
])
def test_a_file_on_disk_under_a_zip_directory_with_the_zip_locator_first_is_answered_for_the_locators_numba_reaches(
        tmp_path, monkeypatch, refused, told):
    # numba's .zip locator takes a path with .zip in it, a file on disk under
    # a directory named so included, without trying its location, so numba
    # reaches no locator listed after it. The remedy matched the error's path
    # against the locations of those too, and an error naming the __pycache__
    # beside the source, under a list with the in-tree locator after the .zip
    # one, was told the package installed at a shorter path, for a location
    # numba never used. The path is matched against the locations of the
    # locators numba reaches alone, and any other is told as the error names
    # it.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "ZipCacheLocator,InTreeCacheLocator", raising=False)
    py_file = tmp_path / "d.zip" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    user_cache_dir = tmp_path / "user-cache" / "numba"
    location = user_cache_dir / _CacheLocator.get_suitable_cache_subpath(str(py_file))
    which, number, strerror = refused
    named = py_file.parent / which if which else location
    remedy = configurations.cache_remedy(str(py_file), OSError(number, strerror, str(named)), "silence")
    told = told.format(named=named, user_cache_dir=user_cache_dir, moved_through=configurations._moved_through())
    assert remedy == f"{told}; or silence", remedy


def test_an_empty_path_object_names_the_working_directory_as_pathlib_reads_it(tmp_path, monkeypatch):
    # An error that names no file, or an empty name, is told the locations
    # numba could have taken; a path object has no empty name, pathlib
    # reading an empty string as the working directory, ".", so one built
    # from it names that directory, which is none of numba's, and is told as
    # named, as the string "." is.
    import numba
    import numbox.core.configurations as configurations
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")

    def remedy(name):
        return configurations.cache_remedy(
            str(py_file), OSError(errno.ENAMETOOLONG, "File name too long", name), "silence")

    assert Path("") == Path(".") and os.fspath(Path("")) == "."
    assert remedy(Path("")) == remedy(".") == (
        "the path is too long for the file system: that location, ., at a shorter path, or NUMBA_CACHE_DIR set to a "
        "short path; or silence")
    assert remedy("").startswith("the path is too long for the file system: the location numba took, one of ")
    assert remedy(b"") == remedy("")


@from_062
def test_a_locator_entry_naming_no_class_is_left_to_numba_and_its_error_raised_as_it_was(tmp_path, monkeypatch):
    # numba resolves a dotted entry of NUMBA_CACHE_LOCATOR_CLASSES to whatever
    # the module holds under the name and fails, for want of from_function,
    # at a function it decorates where it reaches that entry; the check
    # read the entry as numba does and asked issubclass of it first, which
    # raised a TypeError naming neither the entry nor the variable, ahead of
    # numba's error. An entry that is no class is no locator here, numba's
    # own error is raised as it was, and the remedy reads the list without it.
    import types
    import numba
    import numbox.core.configurations as configurations
    ours = types.ModuleType("our_locators")

    def not_a_locator_class():
        pass

    ours.not_a_locator_class = not_a_locator_class
    monkeypatch.setitem(sys.modules, "our_locators", ours)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "our_locators.not_a_locator_class,InTreeCacheLocator",
                        raising=False)
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("ipykernel_123/cell.py", "def f():\n    pass\n")
    py_file = str(archive / "ipykernel_123" / "cell.py")
    with pytest.raises(AttributeError, match="from_function") as raised:
        configurations.check_cache_location(py_file, configurations.LONGEST_CACHE_FILE_NAME)
    assert not configurations.is_a_cache_error(raised.value)
    on_disk = tmp_path / "site" / "package" / "module.py"
    on_disk.parent.mkdir(parents=True)
    on_disk.write_text("")
    failure = OSError(errno.EACCES, "Permission denied", str(on_disk.parent / "__pycache__"))
    with_the_entry = configurations.cache_remedy(str(on_disk), failure, "silence")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator", raising=False)
    assert with_the_entry == configurations.cache_remedy(str(on_disk), failure, "silence")
    assert "numba caches this file in" in with_the_entry, with_the_entry


@from_062
@pytest.mark.parametrize("refused", ["CacheLocator", "SourceFileBackedLocatorMixin"])
def test_a_listed_name_numba_refuses_is_left_out_as_numba_leaves_it(tmp_path, monkeypatch, refused):
    # numba 0.62 and later resolve a plain entry of NUMBA_CACHE_LOCATOR_CLASSES
    # as a name of the caching module and refuse any other before a location
    # is tried; the remedy, reading a name with the underscore numba put
    # before its locators until 0.62 too, read the two names of numba's base
    # classes as those classes and answered for a list numba refuses, with
    # one of their locations to make writable. The underscore is tried under
    # 0.60 and 0.61 alone, which read no list.
    import numba
    import numbox.core.configurations as configurations
    from numba.core import caching
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    assert configurations._locators(f"{refused},InTreeCacheLocator") == [caching.InTreeCacheLocator]
    py_file = tmp_path / "site" / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", refused, raising=False)
    with pytest.raises(RuntimeError, match="Unknown cache locator class") as raised:
        configurations.check_cache_location(str(py_file), configurations.LONGEST_CACHE_FILE_NAME)
    assert not configurations.is_a_cache_error(raised.value)
    failure = OSError(errno.EACCES, "Permission denied", str(py_file.parent / "__pycache__"))
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    assert remedy == (
        f"numba looks only where NUMBA_CACHE_LOCATOR_CLASSES, {refused}, says, and none of those locators takes a "
        "source file on disk: list UserProvidedCacheLocator, with NUMBA_CACHE_DIR set, InTreeCacheLocator or "
        "UserWideCacheLocator, or silence"), remedy


A_ZIP_REFUSED = {
    "unwritable": ((errno.EACCES, "Permission denied"),
                   "no file can be written in that directory, {location} (Permission denied), so make room there, or "
                   "make it writable"),
    "too long": ((errno.ENAMETOOLONG, "File name too long"),
                 "the path is too long for the file system, so put that directory, {location}, at a shorter "
                 "path{moved_through}"),
}


@from_062
@pytest.mark.parametrize("refused", list(A_ZIP_REFUSED))
def test_a_zip_import_refused_its_location_under_a_list_without_the_user_wide_locator_is_told_the_location(
        tmp_path, monkeypatch, refused):
    # The remedy that asks for the user-wide locator to be listed is for
    # numba's no-locator error, which a frozen application gets under a list
    # without that locator. A .zip's error names the location numba took
    # through its .zip locator, which such a list can hold, and the remedy is
    # that location made writable or put at a shorter path, as under numba's
    # own order; held to any error under such a list, the frozen remedy would
    # tell a .zip's reader to list a locator numba takes no .zip through.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "InTreeCacheLocator,ZipCacheLocator", raising=False)
    py_file = str(tmp_path / "bundle.zip" / "package" / "module.py")
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(py_file)
    error, told = A_ZIP_REFUSED[refused]
    remedy = configurations.cache_remedy(py_file, OSError(*error, str(location)), "silence")
    told = told.format(location=location, moved_through=configurations._moved_through())
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: {told}, or silence"), remedy


@from_061
@pytest.mark.parametrize("refused", list(A_ZIP_REFUSED))
def test_a_zip_error_naming_a_file_numba_writes_in_its_location_is_told_the_location(tmp_path, monkeypatch, refused):
    # The package's own check names the location numba took for a .zip, a
    # directory of its own under the user's cache directory; numba's first
    # save names the file it was writing there, under a temporary name, and a
    # caller can pass that error on. The remedy quoted the file as the
    # directory to make writable or to put at a shorter path, which the
    # remedy for a source on disk, matching the file's directory too, did not.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = str(tmp_path / "bundle.zip" / "package" / "module.py")
    location = tmp_path / "user-cache" / "numba" / _CacheLocator.get_suitable_cache_subpath(py_file)
    named = location / "module-1.py312.nbi.tmp.0123456789abcdef"
    error, told = A_ZIP_REFUSED[refused]
    remedy = configurations.cache_remedy(py_file, OSError(*error, str(named)), "silence")
    told = told.format(location=location, moved_through=configurations._moved_through())
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        f"effect here, because the source is not a file on disk: {told}, or silence"), remedy


def test_a_file_not_found_error_that_names_no_file_for_a_source_not_on_disk_gets_the_archive_remedy(tmp_path, monkeypatch):
    # numba's errors name the file they are about, and so does the package's
    # own check, but a caller can pass on a FileNotFoundError that names none.
    # The check for a moved archive, which asks whether the source lies under
    # the file the error names, added os.sep to that name before asking, and
    # raised TypeError on None instead of returning a remedy.
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    py_file = str(tmp_path / "site" / "package" / "module.py")
    remedy = configurations.cache_remedy(py_file, FileNotFoundError(errno.ENOENT, "No such file or directory"), "silence")
    assert remedy == (
        "numba caches a .zip, or a frozen application, in the user's cache directory, and NUMBA_CACHE_DIR has no "
        "effect here, because the source is not a file on disk: no file can be written in that directory, "
        f"{tmp_path / 'user-cache' / 'numba'} (No such file or directory), so make room there, or make it writable, "
        "or silence"), remedy


SUBCLASSED_LOCATORS = (
    "from numba.core.caching import UserProvidedCacheLocator, ZipCacheLocator\n"
    "class OurUserProvided(UserProvidedCacheLocator):\n    pass\n"
    "class OurZip(ZipCacheLocator):\n    pass\n"
)

# Each case: NUMBA_CACHE_LOCATOR_CLASSES, whether NUMBA_CACHE_DIR is set, the
# source's directory, which location the error names ("none" for numba's
# no-locator error), and the remedy before "; or silence" or ", or silence".
SUBCLASS_CASES = {
    "the user-provided subclass took NUMBA_CACHE_DIR's location": (
        "our_locators.OurUserProvided,InTreeCacheLocator,UserWideCacheLocator", True, "site", "NUMBA_CACHE_DIR",
        "the path is too long for the file system: a shorter NUMBA_CACHE_DIR"),
    "the .zip subclass took the user's cache directory's location": (
        "our_locators.OurZip,UserProvidedCacheLocator,InTreeCacheLocator,UserWideCacheLocator", True, "bundle.zip",
        "user", "the path is too long for the file system: the user's cache directory, {user_cache_dir}, at a "
        "shorter path{moved_through}"),
    "the user-provided subclass before the in-tree locator that took the location": (
        "our_locators.OurUserProvided,InTreeCacheLocator", True, "site", "beside",
        "the path is too long for the file system: numbox installed at a shorter path, or NUMBA_CACHE_DIR, which "
        "is set to {cache_dir} and numba could not use, made a writable directory at a short path"),
    "no locator, the user-provided subclass passed over": (
        "our_locators.OurUserProvided,InTreeCacheLocator", True, "site", "none",
        "NUMBA_CACHE_DIR is set to {cache_dir}, which numba could not use: set it to a writable directory at a short "
        "path"),
}


@pytest.mark.skipif(numba_version < 62, reason="NUMBA_CACHE_LOCATOR_CLASSES arrived in numba 0.62")
@pytest.mark.parametrize("case", list(SUBCLASS_CASES))
def test_a_subclass_of_one_of_numbas_locators_is_told_as_its_parent_is(tmp_path, monkeypatch, case):
    # NUMBA_CACHE_LOCATOR_CLASSES takes a dotted path to any class, and a
    # subclass of one of numba's locators caches where its parent does; numba
    # takes it as it takes the parent. The remedy knew a subclass of the
    # in-tree locator for one, and matched the user-provided and .zip
    # locators by identity, so their subclasses got "that location, X" with no
    # cure, and the variable the subclass reads counted as unread.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    (tmp_path / "our_locators.py").write_text(SUBCLASSED_LOCATORS)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "our_locators", raising=False)
    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    locators, cache_dir_set, source_dir, which, expected = SUBCLASS_CASES[case]
    cache_dir = tmp_path / "cache"
    user_cache_dir = tmp_path / "user-cache" / "numba"
    monkeypatch.setattr(numba.config, "CACHE_DIR", str(cache_dir) if cache_dir_set else "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", locators, raising=False)
    py_file = tmp_path / source_dir / "package" / "module.py"
    py_file.parent.mkdir(parents=True)
    py_file.write_text("")
    subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
    if which == "none":
        failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
        ending = ", or silence"
    else:
        location = {
            "NUMBA_CACHE_DIR": cache_dir / subpath, "beside": py_file.parent / "__pycache__", "user": user_cache_dir / subpath,
        }[which]
        failure = OSError(errno.ENAMETOOLONG, "File name too long", str(location))
        ending = "; or silence"
    remedy = configurations.cache_remedy(str(py_file), failure, "silence")
    expected = expected.format(cache_dir=cache_dir, user_cache_dir=user_cache_dir,
                               moved_through=configurations._moved_through())
    assert remedy == f"{expected}{ending}", remedy


@pytest.mark.parametrize("given", ["Path", "bytes"])
@pytest.mark.parametrize("placement", ["no locator", "too long", "archive"])
def test_a_source_given_as_a_path_object_or_bytes_gets_the_remedy_its_string_gets(
        tmp_path, monkeypatch, placement, given):
    # check_cache_location takes any path-like through os.fspath, and the
    # remedy took one too until it asked '".zip" in py_file' of it, which a
    # Path cannot answer: TypeError for every source on disk under numba's own
    # order, and AttributeError at 'py_file.startswith' for a source not on disk.
    # A bytes path, which check_cache_location takes as well, raised TypeError
    # at the first 'startswith' asked of it with a str, os.fspath passing bytes
    # through as bytes.
    import numba
    import numbox.core.configurations as configurations

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    if placement == "archive":
        py_file = tmp_path / "bundle.zip" / "package" / "module.py"
        failure = FileNotFoundError(errno.ENOENT, "No such file or directory", str(tmp_path / "bundle.zip"))
    else:
        py_file = tmp_path / "site" / "package" / "module.py"
        py_file.parent.mkdir(parents=True)
        py_file.write_text("")
        if placement == "too long":
            failure = OSError(errno.ENAMETOOLONG, "File name too long", str(py_file.parent / "__pycache__"))
        else:
            failure = RuntimeError(f"cannot cache function '_cache_probe': no locator available for file '{py_file}'")
    as_given = py_file if given == "Path" else os.fsencode(py_file)
    as_a_string = configurations.cache_remedy(str(py_file), failure, "silence")
    assert configurations.cache_remedy(as_given, failure, "silence") == as_a_string


@pytest.mark.parametrize("given", ["Path", "bytes"])
@pytest.mark.parametrize("placement", ["too long", "unwritable", "a .zip's location", "a moved archive"])
def test_an_error_naming_its_file_as_a_path_object_or_bytes_gets_the_remedy_a_string_name_gets(
        tmp_path, monkeypatch, placement, given):
    # An error names the file it refused as the call that raised it was given
    # it, bytes for a call given bytes, or as a caller built it, a path object
    # too, and the remedy read the name as a string: bytes matched none of
    # the locations, which the remedy writes as strings, and were told as a
    # location that is none of numba's, and for a source not on disk the
    # check for a moved archive, adding os.sep to the name, raised TypeError
    # on bytes and on a path object alike.
    import numba
    import numbox.core.configurations as configurations
    from numba.core.caching import _CacheLocator

    class UserCacheUnderTmp:
        def __init__(self, appname, appauthor):
            self.user_cache_dir = str(tmp_path / "user-cache" / "numba")

    monkeypatch.setattr(configurations, "AppDirs", UserCacheUnderTmp)
    monkeypatch.setattr(numba.config, "CACHE_DIR", "")
    monkeypatch.setattr(numba.config, "CACHE_LOCATOR_CLASSES", "", raising=False)
    refused = (errno.EACCES, "Permission denied")
    if placement in ("a .zip's location", "a moved archive"):
        py_file = tmp_path / "bundle.zip" / "package" / "module.py"
        if placement == "a moved archive":
            error, named = (errno.ENOENT, "No such file or directory"), tmp_path / "bundle.zip"
        else:
            subpath = _CacheLocator.get_suitable_cache_subpath(str(py_file))
            error, named = refused, tmp_path / "user-cache" / "numba" / subpath
    else:
        py_file = tmp_path / "site" / "package" / "module.py"
        py_file.parent.mkdir(parents=True)
        py_file.write_text("")
        error = (errno.ENAMETOOLONG, "File name too long") if placement == "too long" else refused
        named = py_file.parent / "__pycache__"
    as_given = named if given == "Path" else os.fsencode(named)
    as_a_string = configurations.cache_remedy(str(py_file), OSError(*error, str(named)), "silence")
    assert configurations.cache_remedy(str(py_file), OSError(*error, as_given), "silence") == as_a_string


@pytest.mark.parametrize("placement", ["too long", "archive"])
def test_the_remedies_that_name_a_package_name_the_one_given(tmp_path, monkeypatch, placement):
    # A package built on numbox puts the question for its own files and takes
    # the remedy from cache_remedy; the two remedies that tell the reader to
    # install a package again named numbox whichever package had asked.
    import numba
    from numbox.core.configurations import cache_remedy
    if placement == "too long":
        monkeypatch.setattr(numba.config, "CACHE_DIR", "")
        py_file = tmp_path / "ducklib.py"
        py_file.write_text("")
        failure = OSError(errno.ENAMETOOLONG, "File name too long", str(tmp_path / "__pycache__"))
        named = "{} installed at a shorter path"
    else:
        py_file = tmp_path / "numbduck-0.0.0-py3-none-any.whl" / "numbduck" / "ducklib.py"
        failure = RuntimeError(f"cannot cache function 'f': no locator available for file '{py_file}'")
        named = "install {} with its source files on disk"
    remedy = cache_remedy(str(py_file), failure, "silence", package="numbduck")
    assert named.format("numbduck") in remedy and "numbox" not in remedy, remedy
    assert named.format("numbox") in cache_remedy(str(py_file), failure, "silence")


def _function_names(tree):
    """Each function's qualified name as numba names its cache files, angle brackets dropped, with its line."""
    def walk(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield ".".join(scope + [child.name]), child.lineno
                yield from walk(child, scope + [child.name, "locals"])
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, scope + [child.name])
            else:
                yield from walk(child, scope)
    return walk(tree, [])


def test_numbas_longest_file_name_for_the_package_is_within_the_probes_bound():
    # The probe reserves LONGEST_CACHE_FILE_NAME bytes in a location for the
    # files numba writes there: the module's stem, the function's qualified
    # name, a line number, the interpreter tag with its abiflags and an index
    # number, written under a 21-byte temporary name. Every function of the
    # package must fit, decorated or not, and so must the builder's generated
    # kernel, anchored to its file under a name of 64 hex characters.
    from numbox.core.configurations import LONGEST_CACHE_FILE_NAME
    tag = f".py{sys.version_info[0]}{sys.version_info[1]}t.99.nbc.tmp.0123456789abcdef"
    names = [f"{path.stem}.{qualname}-{lineno}{tag}"
             for path in sorted((REPO / "numbox").rglob("*.py"))
             for qualname, lineno in _function_names(ast.parse(path.read_text(encoding="utf-8")))]
    names.append(f"builder._make_{'0' * 64}-9999{tag}")
    longest = max(names, key=len)
    assert len(names) > 500
    assert len(longest) <= LONGEST_CACHE_FILE_NAME, longest


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

COMPILE_A_KERNEL = (
    "from numbox.core.variable.variable import Graph\n"
    "from numbox.core.variable.compile_kernel import compile_kernel\n"
    "def f(x):\n"
    "    return x + 1.0\n"
    "graph = Graph({'calc': [{'name': 'y', 'inputs': {'x': 'ext'}, 'formula': f}]}, ['ext'])\n"
    "kernel = compile_kernel(graph, 'calc.y')\n"
    "assert kernel.execute({'ext': {'x': 1.0}})['calc.y'] == 2.0\n"
    "print('executed')\n"
)

BUILD_A_DERIVE = (
    "from numpy import isclose\n"
    "from numbox.core.work.builder import Derived, End, make_graph\n"
    "x = End(name='x', init_value=3.14)\n"
    "def twice(x):\n"
    "    return 2 * x\n"
    "y = Derived(name='y', init_value=0.0, derive=twice, sources=(x,))\n"
    "access = make_graph(y)\n"
    "access.y.calculate()\n"
    "assert isclose(access.y.data, 6.28), access.y.data\n"
    "print('derived')\n"
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
@pytest.mark.parametrize("child, word, warns", [
    (MAKE_A_STRUCTREF, "made", True), (REGISTER_AN_AGGREGATE, "summed", True), (REGISTER_A_TVF, "selected", True),
    (COMPILE_A_KERNEL, "executed", True), (BUILD_A_DERIVE, "derived", False),
], ids=["make_structref", "register_aggregate", "register_tvf", "compile_kernel", "derive"])
def test_a_warm_anchor_in_a_directory_that_stopped_being_writable(tmp_path, child, word, warns):
    # The anchor is on disk from an earlier run, in a NUMBA_CACHE_DIR that can
    # no longer be written. numba then caches the code in the user's cache
    # directory, and a check on the anchor's own directory would have turned
    # caching off where numba had a location. With the user's cache directory
    # unwritable too there is none, and a check that only wrote the anchor
    # would have let the first decorated function die at numba's set-up. The
    # builder's derive falls back without a word, as it did for an anchor it
    # could not write.
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
        if warns:
            assert "compiles without a cache" in uncached.stderr and "Set NUMBA_CACHE_DIR" in uncached.stderr
        else:
            assert "compiles without a cache" not in uncached.stderr, uncached.stderr
    finally:
        for path in read_only:
            path.chmod(0o755)


@pytest.mark.parametrize("child, word, warns", [
    (MAKE_A_STRUCTREF, "made", True), (REGISTER_AN_AGGREGATE, "summed", True), (REGISTER_A_TVF, "selected", True),
    (COMPILE_A_KERNEL, "executed", True), (BUILD_A_DERIVE, "derived", False),
], ids=["make_structref", "register_aggregate", "register_tvf", "compile_kernel", "derive"])
@pytest.mark.parametrize("placement", ["a file", "a component too long"])
def test_generated_code_compiles_uncached_under_a_numba_cache_dir_numba_cannot_use(
        tmp_path, placement, child, word, warns):
    # NUMBA_CACHE_DIR pointing into a file, or with a component longer than the
    # file system allows: the package's own functions cache beside their
    # sources, numba passing such a location over, and the anchor's directory
    # cannot be made. Any error there means no cache here, so the code compiles
    # without one, as it did before the anchors asked numba, and the warning's
    # remedy is the error's: a directory numba can use, or a shorter path.
    if placement == "a file":
        cache_dir = tmp_path / "file"
        cache_dir.write_text("not a directory\n")
        remedy = "Set NUMBA_CACHE_DIR to a writable directory"
    else:
        cache_dir = tmp_path / ("c" * 300)
        # Windows reports the component as a syntax error (WinError 123, EINVAL),
        # naming no length, and the warning offers the directory then.
        remedy = ("Set NUMBA_CACHE_DIR to a writable directory" if os.name == "nt"
                  else "too long for the file system: NUMBA_CACHE_DIR at a shorter path")
    env = dict(os.environ, PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(cache_dir))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode == 0 and word in run.stdout, run.stderr
    if warns:
        assert "compiles without a cache" in run.stderr and remedy in run.stderr, run.stderr
    else:
        assert "compiles without a cache" not in run.stderr, run.stderr


def test_make_graph_under_a_callers_cache_option_falls_back_from_an_archive(tmp_path):
    # make_graph's kernel is anchored to builder.py itself and cached beside it,
    # and a caller's options reach numba as they are: with cache on from an
    # archive the kernel died at numba's set-up, where the package, the
    # structref, the kernel compiler and the derives had fallen back.
    archive = _archive(tmp_path / "numbox-0.0.0-py3.12.egg")
    child = BUILD_A_DERIVE.replace("make_graph(y)", "make_graph(y, jit_options={'cache': True})")
    assert child != BUILD_A_DERIVE
    env = dict(os.environ, PYTHONPATH=str(archive), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode == 0 and "derived" in run.stdout, run.stderr
    assert "the code generated at builder.py" in run.stderr and "source is not a file on disk" in run.stderr
    assert '"cache" off in the jit options this code was given' in run.stderr, run.stderr
    # The derives cache under anchors of their own, written here, and the
    # builder's answer for its file is not theirs.
    assert _index_files(tmp_path / "cache"), "the derive did not cache under NUMBA_CACHE_DIR"


A_STRUCTREF_WITH_A_TYPING_ERROR = MAKE_A_STRUCTREF.replace(
    "Struct = make_structref('Struct', {'value': float32}, TypeClass)\n",
    "def bad(self):\n"
    "    return self.value + 'x'\n"
    "Struct = make_structref('Struct', {'value': float32}, TypeClass, struct_methods={'bad': bad})\n"
    "Struct(2.5).bad()\n",
)


def test_a_typing_error_in_generated_code_quotes_the_source_with_caching_off(tmp_path):
    # numba quotes the offending line from the file the code names, the
    # anchor, which with caching off was no longer written: the message then
    # pointed at a file that was not there, with "source missing", where main
    # showed the line. The anchor is written where it can be, cache or no.
    assert A_STRUCTREF_WITH_A_TYPING_ERROR != MAKE_A_STRUCTREF
    script = tmp_path / "probe.py"
    script.write_text(A_STRUCTREF_WITH_A_TYPING_ERROR)
    env = dict(os.environ, PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(tmp_path / "cache"),
               NUMBOX_JIT_OPTIONS='{"cache": false}')
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode != 0 and "TypingError" in run.stderr, run.stderr
    assert "source missing" not in run.stderr and "return self.value + 'x'" in run.stderr, run.stderr
    assert list((tmp_path / "cache").rglob("*.py")) and not _index_files(tmp_path / "cache")


@pytest.mark.parametrize("child, word", [
    (MAKE_A_STRUCTREF.replace("TypeClass)\n", "TypeClass, jit_options={'cache': True})\n"), "made"),
    (COMPILE_A_KERNEL.replace("'calc.y')", "'calc.y', jit_options={'cache': False}, cache=True)"), "executed"),
], ids=["make_structref with jit_options", "compile_kernel with cache"])
def test_the_anchor_warning_names_the_options_the_caller_gave(tmp_path, child, word):
    # make_structref, compile_kernel and the builder take jit options of the
    # caller's, which NUMBOX_JIT_OPTIONS does not reach, and compile_kernel's
    # cache argument overrides those too; the warning offered the variable
    # alone, here set to turn caching off already, and then the jit options,
    # here off as well.
    assert child not in (MAKE_A_STRUCTREF, COMPILE_A_KERNEL)
    cache_dir = tmp_path / "file"
    cache_dir.write_text("not a directory\n")
    env = dict(os.environ, PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(cache_dir),
               NUMBOX_JIT_OPTIONS='{"cache": false}')
    run = subprocess.run([sys.executable, "-W", "always", "-c", child], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode == 0 and word in run.stdout, run.stderr
    assert "compiles without a cache" in run.stderr, run.stderr
    assert '"cache" off in the jit options this code was given, or in its cache argument where it takes one' in run.stderr


# The type class lives in a module of its own, as the docs ask, so that the
# struct's cache entries load in a second process.
A_TYPE_CLASS = (
    "from numba.core.types import StructRef\n"
    "from numba.experimental.structref import register\n"
    "@register\n"
    "class TypeClass(StructRef):\n"
    "    pass\n"
)

MAKE_A_LONG_NAMED_STRUCTREF = (
    "from numba.core.types import float32\n"
    "from numbox.utils.highlevel import make_structref\n"
    "from numbox.utils.preprocessing import bounded_stem\n"
    "from long_named_type_class import TypeClass\n"
    "def dddddddddddddddddddddddddddddddddddddddd(self):\n"
    "    return self.value * 2\n"
    "def " + "m" * 200 + "(self):\n"
    "    return self.value * 3\n"
    "name = NAME\n"
    "field = 'f' + name[1:]\n"
    "fields = {'value': float32}\n"
    "if bounded_stem(field) != field:\n"
    "    fields[bounded_stem(field)] = float32\n"
    "fields[field] = float32\n"
    "methods = {'d' * 40: dddddddddddddddddddddddddddddddddddddddd, 'm' * 200: " + "m" * 200 + "}\n"
    "Struct = make_structref(name, fields, TypeClass, struct_methods=methods)\n"
    "values = [1.5 * (index + 1) for index in range(len(fields))]\n"
    "struct = Struct(*values)\n"
    "assert [getattr(struct, each) for each in fields] == values and getattr(struct, 'd' * 40)() == 3.0\n"
    "assert getattr(struct, 'm' * 200)() == 4.5\n"
    "assert Struct.__name__ == name and Struct.__qualname__ == name and repr(struct).startswith(name + '(')\n"
    "print('made', len(name))\n"
)


@pytest.mark.parametrize("name", ["S" * 40, "S" * 41, "S" * 150, "S" * 300, "é" * 40, "結" * 100],
                         ids=["40 ascii", "41 ascii", "150 ascii", "300 ascii", "40 accented", "100 cjk"])
def test_a_struct_name_of_any_length_caches(tmp_path, name):
    # numba names a cache file after the anchor's stem and the jitted
    # function's qualname, both of which carried the struct's name, so a
    # name of about 93 characters overflowed the file system's 255 bytes in
    # numba's own files, past the anchor's check. The stems and the generated
    # names are bounded now: as they are up to 40 bytes, a prefix and a digest
    # beyond, and the class takes its full name back once compiled. The file
    # system counts bytes, so a name of 40 accented characters (80 bytes) is
    # bounded, and 100 CJK characters (300 bytes) are cut by whole characters.
    # A field named with the struct's length is bounded in its getter the same
    # way, and a field named with that bounded name, defined before it, keeps
    # its property: the long field's getter took the name and the hand-over
    # deleted the short field's. One method's name is 40 bytes, the most a
    # bounded name can be, so its thunk's files are the longest numba writes
    # for any struct: under the 230 bytes bounded_stem promises, which leave
    # room for numba's temporary name at the write; the other's is 200, which
    # the thunk's and the overload's names must bound.
    (tmp_path / "long_named_type_class.py").write_text(A_TYPE_CLASS)
    script = tmp_path / "make.py"
    script.write_text(MAKE_A_LONG_NAMED_STRUCTREF.replace("NAME", repr(name)), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("NUMBOX_JIT_OPTIONS", None)
    run = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", str(script)],
                         capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and f"made {len(name)}" in run.stdout, run.stderr
    cache_files = [path.name for path in (tmp_path / "cache").rglob("*.nb*")]
    assert cache_files and all(len(each.encode()) < 230 for each in cache_files), cache_files
    indexes = _index_files(tmp_path / "cache")
    again = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", str(script)],
                           capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert again.returncode == 0, again.stderr
    assert _index_files(tmp_path / "cache") == indexes


@pytest.mark.skipif(sys.platform != "linux", reason="a path of 4096 bytes and a name of 255 are Linux's limits")
@pytest.mark.parametrize("directory, remedy", [
    ("HOME", "too long for the file system: NUMBA_CACHE_DIR at a shorter path"),
    ("NUMBA_CACHE_DIR", "the path is too long for the file system: a shorter NUMBA_CACHE_DIR; or NUMBOX_JIT_OPTIONS"),
])
def test_an_anchor_path_too_long_for_the_file_system_compiles_uncached_and_the_warning_says_so(
        tmp_path, directory, remedy):
    # A cache directory deep enough that the anchor's directory fits the path
    # limit and the anchor's own name, of fixed length, does not: the user's
    # cache directory under a deep home, which the warning met with "a shorter
    # NUMBA_CACHE_DIR" where none was set. There is no cache here, and the
    # warning names the length rather than offering a writable directory, which
    # this one is; NUMBA_CACHE_DIR at a short path cures it. A NUMBA_CACHE_DIR
    # is made deep enough for the package's own files instead, whose location
    # under it, a directory named for theirs by its name and a hash of its
    # path, ends 20 bytes short of the limit: numba's own check passes there,
    # with a named temporary file too where the file system makes no unnamed
    # one, and the package's does not, so the package answers first, with its
    # remedy, and the struct compiles under its answer without a word of its
    # own. At the home's depth numba's check failed the location on such a file
    # system and moved on to the tree, and the anchor answered instead.
    if directory == "NUMBA_CACHE_DIR":
        from numba.core.caching import _CacheLocator
        location = _CacheLocator.get_suitable_cache_subpath(str(REPO / "numbox" / "core" / "configurations.py"))
        deep = _directory_of_length(tmp_path, 4096 - 20 - 1 - len(location))
    else:
        deep = _directory_of_length(tmp_path, 4096 - 50)
    env = dict(os.environ, PYTHONPATH=str(REPO))
    for name in ("NUMBOX_JIT_OPTIONS", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
        env.pop(name, None)
    env[directory] = str(deep)
    run = subprocess.run([sys.executable, "-W", "always", "-c", MAKE_A_STRUCTREF], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path))
    assert run.returncode == 0 and "made" in run.stdout, run.stderr
    assert run.stderr.count("compiles without a cache") == 1, run.stderr
    assert remedy in run.stderr, run.stderr
