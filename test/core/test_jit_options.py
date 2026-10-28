"""The tests cover values of ``NUMBOX_JIT_OPTIONS`` that numbox refuses with an error that names the variable. numbox
reads the variable once, when ``numbox.core.configurations`` is imported, so each test imports that module in a child
process.
"""
import os
import subprocess
import sys

import pytest

IMPORT_CONFIGURATIONS = "import numbox.core.configurations"


@pytest.mark.parametrize("value", ["null", "42", "[]", '"cache"'])
def test_a_value_that_is_not_an_object_is_refused_by_name(tmp_path, value):
    # Without the check, njit would receive valid JSON of another shape as keyword arguments and fail there, or the
    # fallback would fail when it reads "cache". In either case, the error message would name neither the variable
    # nor the shape that the variable takes.
    env = dict(os.environ, NUMBOX_JIT_OPTIONS=value)
    run = subprocess.run([sys.executable, "-c", IMPORT_CONFIGURATIONS], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode != 0
    assert "NUMBOX_JIT_OPTIONS must be a JSON object" in run.stderr, run.stderr


@pytest.mark.parametrize("value", ['{"cache": "false"}', '{"cache": 0}', '{"cache": null}'])
def test_a_cache_value_that_is_not_a_boolean_is_refused_by_name(tmp_path, value):
    # The string "false" is a common quoting error in an environment variable. numba tests the option for truth, and
    # the string "false" is true, so caching would stay on without the check.
    env = dict(os.environ, NUMBOX_JIT_OPTIONS=value)
    run = subprocess.run([sys.executable, "-c", IMPORT_CONFIGURATIONS], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path))
    assert run.returncode != 0
    assert 'NUMBOX_JIT_OPTIONS "cache" must be true or false' in run.stderr, run.stderr
