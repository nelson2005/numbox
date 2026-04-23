# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

numbox — a toolbox of low-level utilities for working with numba. Provides type erasure (`Any`), native library bindings (`Bindings`), graph nodes (`Node`), function proxies (`Proxy`), graph calculation (`Variable`), and units of work (`Work`).

## Build & Dev

- Venv: `python3.11 -m venv venv && venv/bin/pip install -e . flake8 pytest`
- Install: `pip install -e .`
- Test: `pytest`
- Lint: `flake8` (max-line-length=127, max-complexity=10)
- Docs: `cd docs && make html` (Sphinx)
- Python: >=3.10 (CI tests 3.10–3.14; local venv pinned to 3.11)
- Key dependency: `numba>=0.60.0,<0.66.0` (matches `pyproject.toml`; use `numba==0.60.0` locally)

## Architecture

### Bindings System (core/bindings/)

The bindings subsystem wraps C library functions for use inside numba `@njit` code. Four layers:

1. **`utils.py`** — loads shared libraries via `ctypes.CDLL` with `RTLD_GLOBAL` so symbols are visible to LLVM
2. **`signatures.py`** — flat dict mapping C function names to numba type signatures (e.g., `"cos": float64(float64)`). Organized by library: `signatures_c`, `signatures_m`, `signatures_sqlite`
3. **`call.py`** — `@numba.extending.intrinsic` that generates LLVM IR to call native functions directly via `llvmlite`
4. **`_math.py`, `_c.py`, `_sqlite.py`** — thin Python wrappers using `@cres(signatures.get("func"), cache=True)`

### Adding a New Binding

1. Add signature to `signatures.py` in the appropriate sub-dict
2. Add wrapper to the corresponding `_*.py` file following this pattern:
```python
@cres(signatures.get("func_name"), cache=True)
def func_name(x):
    return _call_lib_func("func_name", (x,))
```
3. Function names must match the C library names exactly
4. Args passed as tuple literal to `_call_lib_func`

### Core Modules

- **`core/any/`** — type erasure: wraps any value into uniform type
- **`core/bindings/`** — JIT-compatible wrappers for native C libraries
- **`core/proxy/`** — function proxies with specified signatures for JIT caching
- **`core/variable/`** — graph calculation framework with JIT dispatcher
- **`core/work/`** — JIT-compatible units of calculation with dependencies

### Utilities (utils/)

- `highlevel.py` — `cres` decorator (compiles to `CompileResultWAP` with explicit signature)
- `lowlevel.py` — low-level numba helpers
- `clock.py` — JIT-callable `monotonic_ns() -> int64` intrinsic (cross-platform)
- `meminfo.py` — memory info utilities
- `standard.py` — standard utilities
- `timer.py` — timing utilities
- `void_type.py` — void type support

## Key Paths

- `numbox/core/bindings/signatures.py` — all native function type signatures
- `numbox/core/bindings/_math.py` — libm wrappers (33 single-arg + 9 two-arg float64 functions)
- `numbox/core/bindings/_c.py` — libc wrappers
- `numbox/core/bindings/_sqlite.py` — libsqlite3 wrappers
- `numbox/utils/clock.py` — cross-platform monotonic nanosecond clock intrinsic
- `test/core/` — tests for all core modules

## Preferences

Cross-project preferences live in the user's MEMORY.md. Only numbox-specific workflow rules are kept here.

- Always exclude CLAUDE.md and fork-only `numbox_ci.yml` matrix expansions from upstream PRs (use a dedicated branch based on `upstream/main`)
- Never merge local feature branches into main — main must always match `upstream/main` (exception: CLAUDE.md and the fork-only CI matrix additions)
- Feature branches: base off `origin/main` (has CLAUDE.md + fork CI); upstream PR branches: base off `upstream/main` (no CLAUDE.md, stock CI)
- Do all coding work on the feature branch (has CLAUDE.md + fork CI), then cherry-pick to the upstream PR branch when ready

## CI

- **numbox_ci.yml** — lint + test + build on push/PR (matrix: Python 3.10–3.14, ubuntu + ubuntu-arm + windows + macOS; min/max numba versions; pytest --durations=20)
- **docs.yml** — Sphinx docs → GitHub Pages on push to main
- **release.yml** — build + publish to PyPI on release

## Related Projects

- **[numbduck](https://github.com/Goykhman/numbduck)** — adapts DuckDB's C API for use inside numba `@njit` code. Built on numbox's bindings toolkit (`signatures` dict, `@cres`, `_call_lib_func`).
- **[numbarrow](https://github.com/Goykhman/numbarrow)** — bridges PyArrow arrays into numba `@njit` code. Also built on numbox.

## Project Status

- **monotonic_ns clock** — merged 2026-04-13 via fork [nelson2005/numbox#7](https://github.com/nelson2005/numbox/pull/7) / upstream [Goykhman/numbox#8](https://github.com/Goykhman/numbox/pull/8). JIT-callable `monotonic_ns() -> int64` intrinsic in [`numbox/utils/clock.py`](numbox/utils/clock.py).
- **Fork-only CLAUDE.md + CI expansion** — [nelson2005/numbox#5](https://github.com/nelson2005/numbox/pull/5) (this PR). Adds this file and expands the CI matrix with per-Python numba version pins, macOS runner, and `pytest --durations=20`.
