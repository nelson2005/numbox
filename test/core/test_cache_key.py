"""numbox's part of numba's cache key: the settings, flags, globals and callees it holds, the opt-in decorators, the
fallback to uncached and the churn warning.

The cross-process tests run a probe in child processes against one ``NUMBA_CACHE_DIR`` and read what each child
reports, a cache hit or miss per function and the values it computed, from a JSON file it writes: several settings
flood a child's stdout with numba's own dumps.
"""
import enum
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import types
import warnings

from concurrent.futures import ThreadPoolExecutor

import numba
import numbox
import numpy as np
import pytest

from numba import njit, objmode
from numba.core import config as numba_config
from numba.core.registry import cpu_target
from numba.experimental import jitclass
from numba.extending import intrinsic
from numpy import sum as _npsum

from numbox.core import cache_key
from numbox.core.cache_key import (
    KEYED_SETTINGS, UNKEYED_SETTINGS, CacheChurnWarning, explain_cache_key, keyed_cfunc, keyed_njit,
)
from numbox.core.proxy.proxy import proxy
from numbox.utils.derive_wap import jit_addr_supported


def _child_env(cache_dir, **extra):
    """The test's environment for a child, with numba's defaults for every keyed setting and the package's options."""
    env = {k: v for k, v in os.environ.items() if k not in KEYED_SETTINGS and k != "NUMBOX_JIT_OPTIONS"}
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


def _run(script, env, out, *args):
    r = subprocess.run(
        [sys.executable, str(script), str(out), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    assert r.returncode == 0, f"{script.name} {args} failed:\n{r.stderr[-4000:]}"
    with open(out, encoding="utf-8") as f:
        return json.load(f)


# ---- The classification ----

def test_every_setting_numba_reads_is_classified():
    """A numba release that reads a ``NUMBA_*`` setting nobody has classified fails here, on the matrix cell that
    runs it: the setting may change the code numba emits, and then it belongs in the key."""
    read = set(re.findall(r'"(NUMBA_[A-Z0-9_]+)"', inspect.getsource(numba_config)))
    assert not set(KEYED_SETTINGS) & set(UNKEYED_SETTINGS)
    unclassified = read - set(KEYED_SETTINGS) - set(UNKEYED_SETTINGS)
    assert not unclassified, f"numba {numba.__version__} reads settings nobody has classified: {sorted(unclassified)}"


def test_every_keyed_setting_names_the_config_attribute_numba_reads_it_into():
    source = inspect.getsource(numba_config)
    for env, attribute in KEYED_SETTINGS.items():
        if f'"{env}"' not in source:
            continue
        assert re.search(rf'^\s*{attribute}\s*=\s*_readenv\(\s*"{env}"', source, re.M), (env, attribute)
        assert hasattr(numba_config, attribute), (env, attribute)


# ---- Settings, in code ----

@keyed_njit(cache=True)
def _keyed_for_settings(x):
    return x + 1


@pytest.mark.parametrize("env", sorted(KEYED_SETTINGS))
def test_a_setting_changed_in_code_changes_the_key(env, monkeypatch):
    """The key reads each setting when numba looks the function up, so a change made in code after import counts
    for the next compile; the value numba built its code generator under stays beside it."""
    codegen = cpu_target.target_context.codegen()
    keyed = _keyed_for_settings._cache._numbox_keyed
    before = explain_cache_key(_keyed_for_settings)
    key_before = cache_key._key_parts(_keyed_for_settings.py_func, keyed, codegen)
    monkeypatch.setattr(numba_config, KEYED_SETTINGS[env], "numbox-test-value", raising=False)
    after = explain_cache_key(_keyed_for_settings)
    assert after["settings"][env] == (before["settings"][env][0], "'numbox-test-value'")
    assert cache_key._key_parts(_keyed_for_settings.py_func, keyed, codegen) != key_before


def test_numbas_opt_level_keeps_max_apart_from_three():
    """numba reads ``NUMBA_OPT=max`` into an ``int`` of value 3 that remembers it was ``max``, and builds a different
    code generator for it; the key keeps the two apart."""
    assert cache_key._setting_value(numba_config._OptLevel("max")) == "'max'"
    assert cache_key._setting_value(numba_config._OptLevel(3)) == "3"
    assert cache_key._setting_value(3) == "3"
    assert cache_key._setting_value(True) == cache_key._setting_value(1)


_AT_BUILD = '''
import json
import sys

from numba.core import config

from numbox.core.cache_key import explain_cache_key, keyed_njit


@keyed_njit(cache=True)
def f(x):
    return x + 1


f(1)
config.OPT = 0
with open(sys.argv[1], "w", encoding="utf-8") as out:
    json.dump({"opt": explain_cache_key(f)["settings"]["NUMBA_OPT"]}, out)
'''

_BUILT_BEFORE_NUMBOX = '''
import json
import sys

from numba import njit
from numba.core import config
from numba.core.registry import cpu_target


@njit
def warm(x):
    return x + 1


warm(1)
from numbox.core.cache_key import explain_cache_key, keyed_njit  # noqa: E402


@keyed_njit(cache=True)
def f(x):
    return x + 1


codegen = cpu_target.target_context.codegen()
with open(sys.argv[1], "w", encoding="utf-8") as out:
    json.dump({"recorded": getattr(codegen, "_numbox_settings_at_build", None) is not None,
               "opt": explain_cache_key(f)["settings"]["NUMBA_OPT"]}, out)
'''


def test_the_key_holds_the_setting_numba_built_its_code_generator_under(tmp_path):
    script = tmp_path / "at_build.py"
    script.write_text(_AT_BUILD, encoding="utf-8")
    got = _run(script, _child_env(tmp_path / "cache"), tmp_path / "out.json")
    assert got["opt"] == ["3", "0"]


def test_a_code_generator_built_before_numbox_takes_the_settings_at_numbox_import(tmp_path):
    script = tmp_path / "built_before.py"
    script.write_text(_BUILT_BEFORE_NUMBOX, encoding="utf-8")
    got = _run(script, _child_env(tmp_path / "cache"), tmp_path / "out.json")
    assert got == {"recorded": True, "opt": ["3", "3"]}


# ---- Settings, across processes ----

_SWEEP_MODULE = '''
import numpy as np
from numba import prange

from numbox.core.cache_key import keyed_njit


@keyed_njit(cache=True)
def index(a, i):
    return a[i]


@keyed_njit(cache=True, parallel=True)
def parallel_tuple(a, b, c):
    t = (a, b, c)
    out = np.empty_like(a)
    for i in prange(a.shape[0]):
        out[i] = t[0][i] + t[1][i] + t[2][i]
    return out
'''

_SWEEP_RUN = '''
import json
import sys

import numpy as np
from numba.core import config

import sweep_module
from numbox.core.cache_key import KEYED_SETTINGS, _setting_value

out = {"settings": {env: _setting_value(getattr(config, attr, None)) for env, attr in KEYED_SETTINGS.items()}}
a = np.arange(3.0)
out["index"] = float(sweep_module.index(a, 1))
out["index_hits"] = sum(sweep_module.index.stats.cache_hits.values())
out["index_misses"] = sum(sweep_module.index.stats.cache_misses.values())
if out["settings"]["NUMBA_BOUNDSCHECK"] == "1":
    try:
        sweep_module.index(a, 5)
        out["out_of_bounds"] = "no error"
    except IndexError:
        out["out_of_bounds"] = "IndexError"
if "parallel" in sys.argv[2:]:
    try:
        out["parallel_tuple"] = float(sweep_module.parallel_tuple(a, a, a).sum())
        out["parallel_tuple_hits"] = sum(sweep_module.parallel_tuple.stats.cache_hits.values())
    except Exception as exc:
        out["parallel_tuple"] = type(exc).__name__
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(out, f)
'''

#: The value each keyed setting is flipped to in the sweep, against numba's default. ``NUMBA_ENABLE_AVX`` is flipped
#: against what the first child reports, which depends on the host. ``NUMBA_USE_RVSDG_FRONTEND`` needs numba 0.60 on
#: Python 3.11 with ``numba-rvsdg`` installed, which no matrix cell has, so it is covered in code above.
_FLIPS = {
    "NUMBA_BOUNDSCHECK": "1",
    "NUMBA_OPT": "0",
    "NUMBA_LOOP_VECTORIZE": "0",
    "NUMBA_SLP_VECTORIZE": "1",
    "NUMBA_DEBUGINFO": "1",
    "NUMBA_ENABLE_PROFILING": "1",
    "NUMBA_EXTEND_VARIABLE_LIFETIMES": "1",
    "NUMBA_LLVM_REFPRUNE_PASS": "0",
    "NUMBA_LLVM_REFPRUNE_FLAGS": "per_bb",
    "NUMBA_DEBUG_NRT": "1",
    "NUMBA_DEBUG_NRT_STACK_LIMIT": "5",
    "NUMBA_FULL_TRACEBACKS": "1",
    "NUMBA_DEVELOPER_MODE": "1",
    "NUMBA_DEBUG_JIT": "1",
    "NUMBA_DEBUG_ARRAY_OPT": "1",
    "NUMBA_DEBUG_ARRAY_OPT_RUNTIME": "1",
    "NUMBA_PARFOR_MAX_TUPLE_SIZE": "2",
    "NUMBA_GDB_BINARY": "gdb-of-the-cache-key-test",
    "NUMBA_DUMP_ASSEMBLY": "1",
    "NUMBA_DEBUG": "1",
    "NUMBA_DIFF_IR": "1",
}


def test_the_sweep_flips_every_keyed_setting_but_the_rvsdg_frontend():
    assert set(_FLIPS) | {"NUMBA_ENABLE_AVX", "NUMBA_USE_RVSDG_FRONTEND"} == set(KEYED_SETTINGS)


def test_every_keyed_setting_misses_when_flipped_and_hits_when_flipped_back(tmp_path):
    """Each keyed setting in a child process: compiled and cached under numba's defaults, then flipped, which must
    miss and show the changed behaviour where it shows (a bounds check, a parfor too wide to compile), then flipped
    back on the same cache, which must find the first entry. Each flip runs on a copy of the first cache, so the
    flips run side by side."""
    (tmp_path / "sweep_module.py").write_text(_SWEEP_MODULE, encoding="utf-8")
    script = tmp_path / "sweep_run.py"
    script.write_text(_SWEEP_RUN, encoding="utf-8")
    base = tmp_path / "cache"
    cold = _run(script, _child_env(base), tmp_path / "cold.json", "parallel")
    assert (cold["index"], cold["index_hits"], cold["index_misses"]) == (1.0, 0, 1)
    assert cold["parallel_tuple"] == 9.0
    flips = dict(_FLIPS, NUMBA_ENABLE_AVX="0" if cold["settings"]["NUMBA_ENABLE_AVX"] == "1" else "1")
    source = inspect.getsource(numba_config)
    flips = {env: value for env, value in flips.items() if f'"{env}"' in source}  # those this numba reads

    def flip(env):
        cache = tmp_path / f"cache-{env}"
        shutil.copytree(base, cache)
        args = ("parallel",) if env == "NUMBA_PARFOR_MAX_TUPLE_SIZE" else ()
        flipped = _run(script, _child_env(cache, **{env: flips[env]}), tmp_path / f"{env}-flipped.json", *args)
        back = _run(script, _child_env(cache), tmp_path / f"{env}-back.json", "parallel")
        return env, flipped, back

    with ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 2)) as pool:
        results = {env: (flipped, back) for env, flipped, back in pool.map(flip, sorted(flips))}
    for env, (flipped, back) in results.items():
        assert flipped["settings"][env] != cold["settings"][env], f"{env}={flips[env]} did not take"
        assert (flipped["index"], flipped["index_misses"]) == (1.0, 1), f"{env} flipped was served the default's binary"
        assert (back["index_hits"], back["parallel_tuple_hits"]) == (1, 1), f"{env} flipped back missed"
    assert results["NUMBA_BOUNDSCHECK"][0]["out_of_bounds"] == "IndexError"
    assert results["NUMBA_PARFOR_MAX_TUPLE_SIZE"][0]["parallel_tuple"] == "UnsupportedParforsError"


_ENVIRON_AFTER_IMPORT = '''
import json
import os
import sys

import numpy as np

from numbox.core.cache_key import keyed_njit


@keyed_njit(cache=True)
def index(a, i):
    return a[i]


os.environ["NUMBA_BOUNDSCHECK"] = "1"
try:
    index(np.arange(3.0), 5)
    raised = False
except IndexError:
    raised = True
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump([raised, sum(index.stats.cache_hits.values())], f)
'''


def test_a_setting_put_in_the_environment_after_import_keys_the_lookup_and_the_save_alike(tmp_path):
    """numba reads its NUMBA_* variables again as a compile starts, which is after the cache lookup; the key reads them
    as that compile will, so the entry the first run saves is the one the second run looks up."""
    script = tmp_path / "environ_after_import.py"
    script.write_text(_ENVIRON_AFTER_IMPORT, encoding="utf-8")
    cache = tmp_path / "cache"
    assert _run(script, _child_env(cache), tmp_path / "one.json") == [True, 0]
    assert _run(script, _child_env(cache), tmp_path / "two.json") == [True, 1]


# ---- Globals, callees and module attributes, across processes ----

_GLOBALS_MODULE = '''
import os

import numba
from numba import njit

import globals_cfg
from numbox.core.cache_key import keyed_njit

G = float(os.environ["G_VAL"])
NT = numba.config.NUMBA_NUM_THREADS


@njit
def helper(x):
    return x * G


@keyed_njit(cache=True)
def direct(x):
    return x * G


@keyed_njit(cache=True)
def through_callee(x):
    return helper(x)


@keyed_njit(cache=True)
def module_attribute(x):
    return x * globals_cfg.SCALE


@keyed_njit(cache=True)
def thread_count():
    return NT
'''

_GLOBALS_RUN = '''
import json
import sys

import globals_module as m

out = {}
for name, call in (("direct", lambda: m.direct(1.0)), ("through_callee", lambda: m.through_callee(1.0)),
                   ("module_attribute", lambda: m.module_attribute(1.0)), ("thread_count", m.thread_count)):
    value = call()
    dispatcher = getattr(m, name)
    out[name] = [value, sum(dispatcher.stats.cache_hits.values())]
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(out, f)
'''


def test_a_changed_global_misses_whether_read_directly_through_a_callee_or_as_a_module_attribute(tmp_path):
    """numba's key holds no global's value, so a cached function reading a global that changed between runs is served
    the old value. numbox's key holds each one the function reads, through the functions it calls and the module
    attributes it reads; a thread count frozen at import is one such global."""
    (tmp_path / "globals_module.py").write_text(_GLOBALS_MODULE, encoding="utf-8")
    (tmp_path / "globals_cfg.py").write_text('import os\nSCALE = float(os.environ["SCALE_VAL"])\n', encoding="utf-8")
    script = tmp_path / "globals_run.py"
    script.write_text(_GLOBALS_RUN, encoding="utf-8")
    cache = tmp_path / "cache"

    def run(n, g, scale, threads):
        env = _child_env(cache, G_VAL=g, SCALE_VAL=scale, NUMBA_NUM_THREADS=threads)
        return _run(script, env, tmp_path / f"run{n}.json")

    assert run(1, "2.0", "5.0", "1") == {
        "direct": [2.0, 0], "through_callee": [2.0, 0], "module_attribute": [5.0, 0], "thread_count": [1, 0],
    }
    assert run(2, "3.0", "5.0", "1") == {
        "direct": [3.0, 0], "through_callee": [3.0, 0], "module_attribute": [5.0, 1], "thread_count": [1, 1],
    }
    assert run(3, "2.0", "7.0", "2") == {
        "direct": [2.0, 1], "through_callee": [2.0, 1], "module_attribute": [7.0, 0], "thread_count": [2, 0],
    }


_CALLER = '''
import json
import sys

from callee import step
from numbox.core.cache_key import keyed_njit


@keyed_njit(cache=True)
def caller(x):
    return step(x)


value = caller(1)
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump([value, sum(caller.stats.cache_hits.values())], f)
'''


def test_an_edit_to_a_callee_in_another_file_misses(tmp_path):
    """numba's index is stamped with the caller's own file, so an edit to a function it calls from another file leaves
    the caller's entry standing; numbox's key holds the callee's code."""
    callee = tmp_path / "callee.py"
    script = tmp_path / "caller.py"
    script.write_text(_CALLER, encoding="utf-8")
    cache = tmp_path / "cache"
    callee.write_text("from numba import njit\n\n\n@njit\ndef step(x):\n    return x + 1\n", encoding="utf-8")
    assert _run(script, _child_env(cache), tmp_path / "one.json") == [2, 0]
    assert _run(script, _child_env(cache), tmp_path / "two.json") == [2, 1]
    callee.write_text("from numba import njit\n\n\n@njit\ndef step(x):\n    return x + 2\n", encoding="utf-8")
    assert _run(script, _child_env(cache), tmp_path / "three.json") == [3, 0]


_OWN_CODE_GENERATORS = '''
import os

from llvmlite import ir
from numba import types
from numba.extending import intrinsic, overload

FACTOR = int(os.environ["FACTOR"])
_I64 = ir.IntType(64)


@intrinsic
def times(typingctx, x):
    def codegen(context, builder, signature, args):
        return builder.mul(args[0], ir.Constant(_I64, FACTOR))
    return types.int64(types.int64), codegen


def scale(x):
    return x * FACTOR


@overload(scale)
def ol_scale(x):
    def impl(x):
        return x * FACTOR
    return impl
'''

_CALLS_OWN_CODE_GENERATORS = '''
import json
import sys

from own_code_generators import scale, times
from numbox.core.cache_key import keyed_njit


@keyed_njit(cache=True)
def through_intrinsic(x):
    return times(x)


@keyed_njit(cache=True)
def through_overload(x):
    return scale(x)


with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump({name: [f(10), sum(f.stats.cache_hits.values())]
               for name, f in (("intrinsic", through_intrinsic), ("overload", through_overload))}, fh)
'''


def test_a_value_read_by_a_code_generator_of_ones_own_is_keyed(tmp_path):
    """An ``@intrinsic`` or ``@overload`` from a module no installed package provides has no version to stand in for
    its code generator, so it is followed like any function: a change to a global it reads misses, an LLVM type it
    builds on included, and the same value hits."""
    (tmp_path / "own_code_generators.py").write_text(_OWN_CODE_GENERATORS, encoding="utf-8")
    script = tmp_path / "calls_own_code_generators.py"
    script.write_text(_CALLS_OWN_CODE_GENERATORS, encoding="utf-8")
    cache = tmp_path / "cache"

    def run(n, factor):
        return _run(script, _child_env(cache, FACTOR=factor), tmp_path / f"run{n}.json")

    assert run(1, "2") == {"intrinsic": [20, 0], "overload": [20, 0]}
    assert run(2, "2") == {"intrinsic": [20, 1], "overload": [20, 1]}
    assert run(3, "3") == {"intrinsic": [30, 0], "overload": [30, 0]}


_CALLEES = '''
from numba import njit, vectorize

from callee_consts import K
from numbox.core.cache_key import keyed_njit


@njit(cache=True)
def plain(x):
    return x * K


@vectorize(cache=True)
def vectorized(x):
    return x * K


@keyed_njit(cache=True)
def keyed(x):
    return x * K
'''

_CALLERS = '''
import json
import sys
import warnings

from callees import keyed, plain, vectorized
from numbox.core.cache_key import UncachedWarning, keyed_njit


@keyed_njit(cache=True)
def calls_plain(x):
    return plain(x)


@keyed_njit(cache=True)
def calls_vectorized(x):
    return vectorized(x)


@keyed_njit(cache=True)
def calls_keyed(x):
    return keyed(x)


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    calls_plain(1.0)
    calls_vectorized(1.0)
    out = {"calls_keyed": [calls_keyed(1.0), sum(calls_keyed.stats.cache_hits.values())]}
out["saved"] = [f._cache._cache_file._load_index() != {} for f in (calls_plain, calls_vectorized)]
out["warnings"] = sorted(str(w.message) for w in caught if issubclass(w.category, UncachedWarning))
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(out, fh)
'''


def test_a_callee_cached_under_numbas_key_alone_compiles_its_caller_uncached(tmp_path):
    """numba loads a plain ``njit(cache=True)`` function, or a ``vectorize(cache=True)`` ufunc, from its own cache, which
    a change to what it reads does not miss: a keyed caller would link the old code into a binary saved under a key
    naming the new value, so it compiles uncached with a warning naming the callee, as compile_kernel's kernels do. A
    callee decorated with keyed_njit is keyed itself, so its caller stays cached and sees the change."""
    (tmp_path / "callees.py").write_text(_CALLEES, encoding="utf-8")
    script = tmp_path / "callers.py"
    script.write_text(_CALLERS, encoding="utf-8")
    cache = tmp_path / "cache"
    consts = tmp_path / "callee_consts.py"
    consts.write_text("K = 2.0\n", encoding="utf-8")
    one = _run(script, _child_env(cache), tmp_path / "one.json")
    assert one["calls_keyed"] == [2.0, 0]
    assert one["saved"] == [False, False]
    plain_warning, vectorized_warning = one["warnings"]
    assert ("the global plain read by the function calls_plain is cached under numba's key alone; decorate it with "
            "keyed_njit, or drop its cache=True") in plain_warning
    assert ("the global vectorized read by the function calls_vectorized is a @vectorize ufunc cached under numba's "
            "key alone; drop its cache=True") in vectorized_warning
    assert _run(script, _child_env(cache), tmp_path / "two.json")["calls_keyed"] == [2.0, 1]
    consts.write_text("K = 3.0\n", encoding="utf-8")
    assert _run(script, _child_env(cache), tmp_path / "three.json")["calls_keyed"] == [3.0, 0]


_HELPERS = '''
import numpy as np
from numba import int64
from numba.core import types
from numba.extending import intrinsic, register_jitable

from numbox.utils.highlevel import cres


@register_jitable
def jitable(x):
    return x * {factor}


@intrinsic
def times(typingctx, x):
    def codegen(context, builder, signature, args):
        return builder.mul(args[0], context.get_constant(types.int64, {factor}))
    return types.int64(types.int64), codegen


@cres(int64(int64))
def compiled(x):
    return x * {factor}


dtype = np.dtype("{dtype}")
'''

_READS_HELPERS = '''
import json
import sys

import numpy as np

import helpers
from numbox.core.cache_key import keyed_njit


@keyed_njit(cache=True)
def through_register_jitable(x):
    return helpers.jitable(x)


@keyed_njit(cache=True)
def through_intrinsic(x):
    return helpers.times(x)


@keyed_njit(cache=True)
def through_cres(x):
    return helpers.compiled(x)


@keyed_njit(cache=True)
def through_dtype(x):
    return np.empty(1, helpers.dtype).itemsize + x


out = {}
for name, f in (("register_jitable", through_register_jitable), ("intrinsic", through_intrinsic),
                ("cres", through_cres), ("dtype", through_dtype)):
    out[name] = [f(10), sum(f.stats.cache_hits.values())]
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(out, fh)
'''


def test_an_edit_to_what_a_function_reads_as_a_module_attribute_misses(tmp_path):
    """A ``register_jitable`` function, an ``@intrinsic``, a ``cres`` function and a dtype read as ``helpers.<name>`` go
    into the key as they do imported by name, so an edit to the helper module misses, though the caller's own file,
    which numba's index is stamped with, is unchanged."""
    helpers = tmp_path / "helpers.py"
    script = tmp_path / "reads_helpers.py"
    script.write_text(_READS_HELPERS, encoding="utf-8")
    cache = tmp_path / "cache"
    helpers.write_text(_HELPERS.format(factor=2, dtype="float32"), encoding="utf-8")
    assert _run(script, _child_env(cache), tmp_path / "one.json") == {
        "register_jitable": [20, 0], "intrinsic": [20, 0], "cres": [20, 0], "dtype": [14, 0],
    }
    # numba 0.60 has no jit_addr slot, so cres returns a plain CompileResultWAP there, which numba compiles to its
    # address, a dynamic global: it caches no function that calls one, under its own key or numbox's.
    cres_hit = 1 if jit_addr_supported() else 0
    assert _run(script, _child_env(cache), tmp_path / "two.json") == {
        "register_jitable": [20, 1], "intrinsic": [20, 1], "cres": [20, cres_hit], "dtype": [14, 1],
    }
    helpers.write_text(_HELPERS.format(factor=3, dtype="float64"), encoding="utf-8")
    assert _run(script, _child_env(cache), tmp_path / "three.json") == {
        "register_jitable": [30, 0], "intrinsic": [30, 0], "cres": [30, 0], "dtype": [18, 0],
    }


_NAMEDTUPLE_CFG = '''
import os
from collections import namedtuple

Params = namedtuple("Params", os.environ["FIELDS"])
PARAMS = Params(1.0, 2.0)
'''

_NAMEDTUPLE = '''
import json
import os
import sys
from collections import namedtuple

import namedtuple_cfg
from numbox.core.cache_key import keyed_njit

Params = namedtuple("Params", os.environ["FIELDS"])
PARAMS = Params(1.0, 2.0)


@keyed_njit(cache=True)
def by_name(v):
    return v * PARAMS.alpha


@keyed_njit(cache=True)
def as_module_attribute(v):
    return v * namedtuple_cfg.PARAMS.alpha


out = {name: [f(10.0), sum(f.stats.cache_hits.values())]
       for name, f in (("by_name", by_name), ("as_module_attribute", as_module_attribute))}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(out, fh)
'''


def test_a_namedtuple_whose_field_names_swap_misses(tmp_path):
    """numba compiles ``PARAMS.alpha`` to a position in the tuple, so the same values under the field names in another
    order are another constant; the key holds the class and its field names."""
    (tmp_path / "namedtuple_cfg.py").write_text(_NAMEDTUPLE_CFG, encoding="utf-8")
    script = tmp_path / "namedtuple_run.py"
    script.write_text(_NAMEDTUPLE, encoding="utf-8")
    cache = tmp_path / "cache"

    def run(n, fields):
        return _run(script, _child_env(cache, FIELDS=fields), tmp_path / f"run{n}.json")

    assert run(1, "alpha beta") == {"by_name": [10.0, 0], "as_module_attribute": [10.0, 0]}
    assert run(2, "beta alpha") == {"by_name": [20.0, 0], "as_module_attribute": [20.0, 0]}
    assert run(3, "alpha beta") == {"by_name": [10.0, 1], "as_module_attribute": [10.0, 1]}


_SHAPES = '''
import enum
from collections import namedtuple

from numba import float64
from numba.experimental import jitclass

Point = namedtuple("Point", "{fields}")


class Level(enum.IntEnum):
    LOW = {low}


@jitclass([("v", float64)])
class Counter:
    def __init__(self, v):
        self.v = v

    def scaled(self):
        return self.v * {factor}
'''

_READS_SHAPES = '''
import json
import sys

from shapes import Counter, Level, Point
from numbox.core.cache_key import keyed_njit


@keyed_njit(cache=True)
def through_namedtuple_class(a, b):
    return Point(a, b).x


@keyed_njit(cache=True)
def through_enum_class(x):
    return x * Level.LOW.value


@keyed_njit(cache=True)
def through_jitclass(x):
    return Counter(x).scaled()


@keyed_njit(cache=True, locals={"c": Counter.class_type.instance_type})
def through_a_jitclass_typed_local(x):
    c = Counter(x)
    return c.scaled()


out = {}
for name, f, args in (("namedtuple", through_namedtuple_class, (1.0, 2.0)), ("enum", through_enum_class, (2.0,)),
                      ("jitclass", through_jitclass, (2.0,)), ("typed_local", through_a_jitclass_typed_local, (2.0,))):
    out[name] = [f(*args), sum(f.stats.cache_hits.values())]
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(out, fh)
'''


def test_an_edit_to_a_class_whose_contents_numba_compiles_misses(tmp_path):
    """A namedtuple class's field names, an ``Enum`` class's member values and a jitclass's field types and methods are
    compiled into the code that uses them, so they go into the key with the class and an edit to the module defining
    them misses. A jitclass's numba types are named after an address no other process shares; keyed by what they
    hold, they key alike in every process, so the warm run hits."""
    shapes = tmp_path / "shapes.py"
    script = tmp_path / "reads_shapes.py"
    script.write_text(_READS_SHAPES, encoding="utf-8")
    cache = tmp_path / "cache"
    shapes.write_text(_SHAPES.format(fields="x y", low=2, factor=2.0), encoding="utf-8")
    assert _run(script, _child_env(cache), tmp_path / "one.json") == {
        "namedtuple": [1.0, 0], "enum": [4.0, 0], "jitclass": [4.0, 0], "typed_local": [4.0, 0],
    }
    assert _run(script, _child_env(cache), tmp_path / "two.json") == {
        "namedtuple": [1.0, 1], "enum": [4.0, 1], "jitclass": [4.0, 1], "typed_local": [4.0, 1],
    }
    shapes.write_text(_SHAPES.format(fields="y x", low=3, factor=3.0), encoding="utf-8")
    assert _run(script, _child_env(cache), tmp_path / "three.json") == {
        "namedtuple": [2.0, 0], "enum": [6.0, 0], "jitclass": [6.0, 0], "typed_local": [6.0, 0],
    }


# ---- A value with no stable form ----

_UNKEYABLE = '''
import enum
import json
import os
import sys
import warnings

from numbox.core.cache_key import UncachedWarning, keyed_njit


class Colour(enum.Enum):
    RED = 1


RED = Colour.RED


@keyed_njit(cache=True)
def red():
    return RED.value


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    value = red()
messages = [str(w.message) for w in caught if issubclass(w.category, UncachedWarning)]
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump({"value": value, "warnings": messages, "index": red._cache._cache_file._load_index() != {},
               "misses": sum(red.stats.cache_misses.values())}, f)
'''


def test_a_value_with_no_stable_form_compiles_uncached_with_a_warning(tmp_path):
    """An ``Enum`` member is a constant numba compiles, and none of the kinds the key puts in by name: the function
    compiles in every process, saves nothing, and says why once."""
    script = tmp_path / "unkeyable.py"
    script.write_text(_UNKEYABLE, encoding="utf-8")
    for n in (1, 2):
        got = _run(script, _child_env(tmp_path / "cache"), tmp_path / f"run{n}.json")
        assert got["value"] == 1
        assert got["index"] is False
        assert len(got["warnings"]) == 1
        assert "the global RED read by the function red is of type Colour" in got["warnings"][0]


class _Mode(enum.Enum):
    ON = 2


_ON = _Mode.ON
_modes = types.ModuleType("modes_module")
_modes.CURRENT = _Mode.ON


@keyed_njit(cache=True)
def _reads_an_enum_as_a_module_attribute(x):
    return x * _modes.CURRENT.value


def test_a_value_with_no_stable_form_read_as_a_module_attribute_compiles_uncached():
    """numba compiles ``_modes.CURRENT`` as a constant, as it does an ``Enum`` member read by name, so the key refuses it
    the same way rather than serving one member's binary for another."""
    with pytest.raises(ValueError, match="the module attribute modes_module.CURRENT read by the function "
                                         "_reads_an_enum_as_a_module_attribute is of type _Mode"):
        explain_cache_key(_reads_an_enum_as_a_module_attribute)


@njit
def _reads_an_enum(x):
    return x * _ON.value


_pairs = types.ModuleType("pairs_module")
_pairs.PAIR = (_reads_an_enum, object())


@njit
def _reads_the_pair(x):
    return _pairs.PAIR


_the_same_function = _reads_an_enum


@keyed_njit(cache=True)
def _meets_a_function_twice(x):
    return _reads_the_pair(x), _the_same_function(x)


class _Unhashable:
    __hash__ = None


def _a_plain_function():
    return 0


_tools = types.ModuleType("tools_module")
_tools.scale = 2.0
_tools.shape = _Unhashable()
_tools.ndim = _a_plain_function


@keyed_njit(cache=True)
def _reads_an_array_shape(a):
    return a.shape[0] * a.ndim * _tools.scale


def test_a_module_attribute_numba_could_not_compile_is_passed_over():
    """The function reads ``a.shape`` and ``a.ndim`` off an array and ``_tools.scale`` off the module, so the module's
    attributes of the first two names are among those its code may read: an object numba can neither type nor look
    up, and a plain function numba holds no typing for. Neither could be compiled into the function, so both are
    passed over rather than refusing the key."""
    assert _reads_an_array_shape(np.arange(3.0)) == 6.0
    parts = explain_cache_key(_reads_an_array_shape)
    assert "global tools_module.scale" in parts
    assert "global tools_module.shape" not in parts
    assert "global tools_module.ndim" not in parts


def test_a_module_attribute_passed_over_leaves_no_half_walked_function_behind():
    """``PAIR`` holds an object numba cannot type, so the key passes it over as a name the function may not read from
    the module, by which time the function inside it was half walked, stopped at its ``Enum`` global. Read again by
    name, that function is walked afresh and refuses the key, rather than found already walked."""
    with pytest.raises(ValueError, match="the global _ON read by the function _reads_an_enum is of type _Mode"):
        explain_cache_key(_meets_a_function_twice)


@keyed_njit(cache=True)
def _sums_by_name(a):
    return _npsum(a)


@keyed_njit(cache=True)
def _leaves_for_python(x):
    with objmode(y="int64"):
        y = x + 1
    return y


def test_numpys_array_functions_and_numbas_with_contexts_go_in_by_name():
    """``np.sum`` is no Python function and ``objmode`` is a context object; numba compiles a call of the first through
    a registered template and the second's block from its own code, so imported by name both go in by name and the
    functions stay cached, where an ``UncachedWarning`` would fail this test."""
    assert _sums_by_name(np.arange(4.0)) == 6.0
    assert _leaves_for_python(1) == 2
    assert "global test.core.test_cache_key._npsum" in explain_cache_key(_sums_by_name)
    assert "global test.core.test_cache_key.objmode" in explain_cache_key(_leaves_for_python)


def _make_intrinsic(mode):
    @intrinsic
    def get(typingctx):
        def codegen(context, builder, signature, args):
            return context.get_constant(numba.types.int64, mode.value)
        return numba.types.int64(), codegen
    return get


_GET_ON = _make_intrinsic(_Mode.ON)


@keyed_njit(cache=True)
def _calls_an_intrinsic_closing_over_an_enum():
    return _GET_ON()


@proxy(numba.float64(numba.float64), jit_options={"cache": True})
def _proxied(x):
    return x + 1.0


def test_a_code_generator_numbox_generates_is_keyed_by_numboxs_version():
    """``@proxy`` generates its wrapper's intrinsic in a namespace of its own under this module's name, as numbox's
    code generators all run; it is numbox's code under numbox's version, not this module's own code followed by value,
    which would read the generated namespace's signature and refuse the key, failing this module's import."""
    assert _proxied(1.0) == 2.0
    assert explain_cache_key(_proxied)


@pytest.mark.parametrize("version, walked_as", [("-", "function"), ("own-package==1.0", "intrinsic")])
def test_an_intrinsic_closing_over_a_value_with_no_stable_form_compiles_uncached(version, walked_as, monkeypatch):
    """An intrinsic's code generator reads its closure as it compiles, so a closure cell with no stable form refuses the
    key, as it does read by name, rather than going in as one placeholder whatever its value: whether the intrinsic is
    of code no installed package provides, followed like any function, or a package's, folded beside its version."""
    package_version = cache_key._package_version
    monkeypatch.setattr(
        cache_key, "_package_version", lambda module: version if module == __name__ else package_version(module),
    )
    match = f"the closure cell mode of the {walked_as} _make_intrinsic.<locals>.get is of type _Mode"
    with pytest.raises(ValueError, match=match):
        explain_cache_key(_calls_an_intrinsic_closing_over_an_enum)


_EXTERNAL = numba.types.ExternalFunction("numbox_test_symbol", numba.types.int64(numba.types.float64))


@keyed_njit(cache=True)
def _calls_an_external_function(x):
    return _EXTERNAL(x)


def test_an_external_function_is_keyed_on_its_signature(monkeypatch):
    """numba knows an external function by its symbol and its signature, and the type's name holds the symbol alone."""
    part = "global test.core.test_cache_key._EXTERNAL"
    before = explain_cache_key(_calls_an_external_function)[part]
    retyped = numba.types.ExternalFunction("numbox_test_symbol", numba.types.int64(numba.types.int32))
    monkeypatch.setattr(sys.modules[__name__], "_EXTERNAL", retyped)
    assert explain_cache_key(_calls_an_external_function)[part] != before


@jitclass([("n", numba.int64)])
class _Tally:
    def __init__(self, n):
        self.n = n


@keyed_njit(cache=True, locals={"t": numba.types.Optional(_Tally.class_type.instance_type)})
def _holds_an_optional_jitclass(n):
    t = _Tally(n)
    return t.n


_jitclass_pairs = types.ModuleType("jitclass_pairs_module")
_jitclass_pairs.PAIR = (_Tally, object())


@njit
def _reads_the_jitclass_pair(x):
    return _jitclass_pairs.PAIR


_the_same_jitclass = _Tally


@keyed_njit(cache=True)
def _meets_a_jitclass_twice(x):
    return _reads_the_jitclass_pair(x), _the_same_jitclass(x)


def test_a_module_attribute_passed_over_leaves_no_half_walked_jitclass_behind():
    """``PAIR`` holds an object numba cannot type, so the key passes it over, by which time the jitclass beside it was
    walked into the parts. Read again by name, the jitclass is walked afresh, and its fields and methods are in the
    key, rather than found already walked with its part rolled back."""
    assert "jitclass _Tally" in explain_cache_key(_meets_a_jitclass_twice)


def test_a_numba_type_naming_a_jitclass_by_its_address_compiles_uncached():
    """numba names a jitclass's types after the address of an object of its own, which no other process shares. A
    jitclass type itself goes in by its fields and methods; any other type whose name holds one is refused."""
    with pytest.raises(ValueError, match="which names an object by its address"):
        explain_cache_key(_holds_an_optional_jitclass)


# ---- The churn warning ----

_CHURN = '''
G = 0


def body(x):
    return x + G
'''


def test_a_global_that_changes_every_run_draws_the_churn_warning(tmp_path, monkeypatch):
    """Each new value of a global saves a new entry; past the threshold numbox names the global, once."""
    (tmp_path / "churn_module.py").write_text(_CHURN, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(numba_config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cache_key, "CHURN_THRESHOLD", 3)
    module = importlib.import_module("churn_module")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for value in range(6):
            module.G = value
            assert keyed_njit(cache=True)(module.body)(1) == 1 + value
    churn = [str(w.message) for w in caught if issubclass(w.category, CacheChurnWarning)]
    assert len(churn) == 1, churn
    assert "global churn_module.G (4 values)" in churn[0]
    assert "holds 4 entries" in churn[0]


def test_many_signatures_alone_draw_no_churn_warning(tmp_path, monkeypatch):
    (tmp_path / "signatures_module.py").write_text("def body(x):\n    return x\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(numba_config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cache_key, "CHURN_THRESHOLD", 2)
    module = importlib.import_module("signatures_module")
    f = keyed_njit(cache=True)(module.body)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for argument in (1, 1.0, 1j, True):
            f(argument)
    assert not [w for w in caught if issubclass(w.category, CacheChurnWarning)]


# ---- The opt-in ----

def _plain(x):
    return x + 1


def test_keyed_njit_keys_lazy_and_eager_compiles_and_njit_does_not():
    lazy = keyed_njit(cache=True)(_plain)
    eager = keyed_njit("float64(float64)", cache=True)(_plain)
    called = keyed_njit(_plain, cache=True)
    plain = njit(cache=True)(_plain)
    for keyed in (lazy, eager, called):
        assert keyed._cache._numbox_keyed[0] == "njit"
        assert "code test.core.test_cache_key._plain" in explain_cache_key(keyed)
    with pytest.raises(ValueError, match="has no cache numbox keys"):
        explain_cache_key(plain)
    with pytest.raises(ValueError, match="has no cache numbox keys"):
        explain_cache_key(keyed_njit(_plain))
    assert eager(1.0) == 2.0


def test_keyed_cfunc_is_keyed_apart_from_an_njit_of_the_same_function():
    """numba keys a ``@cfunc`` and an ``@njit`` of one function and signature alike, though their compiled results
    differ; numbox's flags hold the decorator."""
    c = keyed_cfunc("float64(float64)", cache=True)(_plain)
    j = keyed_njit("float64(float64)", cache=True)(_plain)
    assert c._cache._numbox_keyed[0] == "cfunc"
    assert explain_cache_key(c)["flags"].startswith("kind=cfunc;")
    assert explain_cache_key(j)["flags"].startswith("kind=njit;")
    assert c.ctypes(1.0) == 2.0


def test_the_flags_are_in_the_key():
    fast = keyed_njit(cache=True, fastmath=True)(_plain)
    plain = keyed_njit(cache=True)(_plain)
    assert explain_cache_key(fast)["flags"] != explain_cache_key(plain)["flags"]
    assert "'fastmath':True" in explain_cache_key(fast)["flags"]


@keyed_njit(cache=True)
def _sliced(a):
    return a[1:3].sum() + a[::2].sum() + a[..., 1].sum()


def test_folded_slices_and_ellipsis_in_the_code_are_keyed():
    """Python 3.14 folds ``a[1:3]`` into a ``slice`` constant, and ``a[..., 1]`` puts ``Ellipsis`` in a tuple constant
    on every version; the key reads both, where an ``UncachedWarning`` would fail this test."""
    assert _sliced(np.arange(6.0).reshape(3, 2)) == 33.0
    assert "code test.core.test_cache_key._sliced" in explain_cache_key(_sliced)


def test_explain_cache_key_lists_the_parts():
    parts = explain_cache_key(_keyed_for_settings)
    assert parts["numbox"] == numbox.__version__
    assert set(parts["settings"]) == set(KEYED_SETTINGS)
    assert all(len(pair) == 2 for pair in parts["settings"].values())
    assert "code test.core.test_cache_key._keyed_for_settings" in parts


# ---- What numbox compiles ----

_BINDING = '''
import json
import sys

from numbox.core.bindings.libm import cos

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump([cos(0.0), sum(cos.stats.cache_hits.values()), cos._cache._numbox_keyed[0]], f)
'''


def test_a_binding_misses_under_a_changed_numbox_jit_options(tmp_path):
    """The package's options reach every binding; numba's key does not hold them, numbox's does."""
    script = tmp_path / "binding.py"
    script.write_text(_BINDING, encoding="utf-8")
    cache = tmp_path / "cache"
    assert _run(script, _child_env(cache), tmp_path / "one.json") == [1.0, 0, "njit"]
    assert _run(script, _child_env(cache), tmp_path / "two.json") == [1.0, 1, "njit"]
    fast = _child_env(cache, NUMBOX_JIT_OPTIONS='{"cache": true, "fastmath": true}')
    assert _run(script, fast, tmp_path / "three.json") == [1.0, 0, "njit"]
    assert _run(script, _child_env(cache), tmp_path / "four.json") == [1.0, 1, "njit"]


_CRES = '''
import json
import os
import sys

from numba import float64

from numbox.utils.highlevel import cres

G = float(os.environ["G_VAL"])


@cres(float64(float64), cache=True)
def scaled(x):
    return x * G


with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(scaled(2.0), f)
'''


def test_cres_reads_a_changed_global_fresh(tmp_path):
    script = tmp_path / "cres_probe.py"
    script.write_text(_CRES, encoding="utf-8")
    cache = tmp_path / "cache"
    assert _run(script, _child_env(cache, G_VAL="2.0"), tmp_path / "one.json") == 4.0
    assert _run(script, _child_env(cache, G_VAL="3.0"), tmp_path / "two.json") == 6.0


_CRES_OF_A_JITCLASS = '''
import json
import sys

from numba import float64
from numba.experimental import jitclass

from numbox.core.cache_key import keyed_njit
from numbox.utils.highlevel import cres


@jitclass([("v", float64)])
class Counter:
    def __init__(self, v):
        self.v = v


@cres(float64(Counter.class_type.instance_type))
def read(c):
    return c.v


@keyed_njit(cache=True)
def calls_read():
    return read(Counter(2.0))


with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump([calls_read(), sum(calls_read.stats.cache_hits.values())], f)
'''


def test_a_cres_signature_holding_a_jitclass_keys_alike_in_every_process(tmp_path):
    """numba's text for a signature names a jitclass's type after an address no other process shares; the key holds
    each type of the signature as it holds the type itself, so the warm run hits. numba 0.60 caches no caller of a
    cres function at all (see the module-attribute test above)."""
    script = tmp_path / "cres_of_a_jitclass.py"
    script.write_text(_CRES_OF_A_JITCLASS, encoding="utf-8")
    cache = tmp_path / "cache"
    assert _run(script, _child_env(cache), tmp_path / "one.json") == [2.0, 0]
    assert _run(script, _child_env(cache), tmp_path / "two.json") == [2.0, 1 if jit_addr_supported() else 0]


# ---- Installing the key ----

_NUMBOX_CORE_LOADS = (
    "import sys, numbox.core; print(*sorted(m for m in sys.modules if m.partition('.')[0] == 'numbox'))"
)


def test_every_module_numbox_core_loads_imports_first_in_a_fresh_interpreter(tmp_path):
    """numbox.core installs the cache key as it loads, and the cache key imports numbox.utils.fingerprint: a module it
    loads must not need numbox.core while it loads itself, or imported first it meets its own half-loaded self."""
    env = _child_env(tmp_path / "cache")
    listing = subprocess.run([sys.executable, "-c", _NUMBOX_CORE_LOADS], capture_output=True, text=True, env=env)
    assert listing.returncode == 0, listing.stderr[-2000:]
    modules = listing.stdout.split()
    assert "numbox.utils.fingerprint" in modules
    failed = {}
    for name in modules:
        r = subprocess.run([sys.executable, "-c", f"import {name}"], capture_output=True, text=True, env=env)
        if r.returncode:
            failed[name] = r.stderr.strip().splitlines()[-1]
    assert not failed, failed
