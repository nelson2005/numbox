#!/usr/bin/env bash
# Run a pytest selection against every CPython in the CI matrix, locally.
#
# The cache/hash campaign keys generated names on code.co_code, which changes
# with every CPython bytecode revision -- so a digest that is stable on the dev
# interpreter can still be wrong on four others. CI is a ~25 minute round trip
# to learn that; this is a few minutes.
#
# Interpreters come from uv (`uv python install 3.10 3.11 3.12 3.13 3.14`).
# Venvs are cached under the scratch root and reused across invocations.
#
# Usage:
#   docs/plans/matrix_check.sh                     # whole suite, every version
#   docs/plans/matrix_check.sh -k make_structref   # pytest args pass through
set -euo pipefail

REPO=/home/erik/projects/numbox
# Point at an isolated `git worktree` to run the matrix while the main checkout is
# being mutated (e.g. by a concurrent audit). PYTHONPATH below puts it ahead of the
# venvs' editable install, which otherwise resolves back to $REPO.
TARGET="${MATRIX_TARGET:-$REPO}"
ENVS="${MATRIX_ENV_ROOT:-/tmp/claude-1000/-home-erik/matrix-venvs}"
VERSIONS="${MATRIX_VERSIONS:-3.10 3.11 3.12 3.13 3.14}"

mkdir -p "$ENVS"
rc=0

for v in $VERSIONS; do
  venv="$ENVS/py${v//./}"
  if [ ! -x "$venv/bin/python" ]; then
    echo "=== provisioning $v ==="
    uv venv --python "$v" "$venv" >/dev/null
    # numba 0.60 predates 3.13; let the resolver pick within the project pin.
    uv pip install --python "$venv/bin/python" -q pytest "numba>=0.60.0,<0.67.0"
    uv pip install --python "$venv/bin/python" -q --no-deps -e "$REPO"
  fi
done

# A stale numba cache keyed by another interpreter would mask exactly the bug
# class this script exists to catch.
"$REPO/venv/bin/python" -c "
import pathlib, shutil
for p in pathlib.Path('$TARGET').rglob('__pycache__'):
    if 'venv' not in str(p):
        shutil.rmtree(p, ignore_errors=True)
shutil.rmtree(pathlib.Path.home()/'.cache'/'numba', ignore_errors=True)
"

for v in $VERSIONS; do
  venv="$ENVS/py${v//./}"
  ver=$("$venv/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
  nb=$("$venv/bin/python" -c 'import numba; print(numba.__version__)')
  printf '=== python %s / numba %s ===\n' "$ver" "$nb"
  resolved=$(cd "$TARGET" && PYTHONPATH="$TARGET" "$venv/bin/python" -c 'import numbox,os;print(os.path.dirname(os.path.dirname(numbox.__file__)))')
  if [ "$resolved" != "$TARGET" ]; then
    echo "  ABORT: numbox resolved to $resolved, not $TARGET"
    exit 2
  fi
  if (cd "$TARGET" && PYTHONPATH="$TARGET" "$venv/bin/python" -m pytest -q -p no:cacheprovider "$@" 2>&1 | tail -4); then
    :
  else
    rc=1
  fi
done

exit "$rc"
