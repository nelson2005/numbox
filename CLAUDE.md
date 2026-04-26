# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

numbox — a toolbox of low-level utilities for working with numba. Provides type erasure (`Any`), native library bindings (`Bindings`), graph nodes (`Node`), function proxies (`Proxy`), graph calculation (`Variable`), and units of work (`Work`).

## Build & Dev

- Venv: `python3.11 -m venv venv && venv/bin/pip install -e . flake8 pytest pytest-cov`
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

- Always exclude CLAUDE.md, `docs/plans/**`, and fork-only `numbox_ci.yml` matrix expansions from upstream PRs (use a dedicated branch based on `upstream/main`)
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
- **numbduck generics promotion** — merged 2026-04-24 via upstream [Goykhman/numbox#9](https://github.com/Goykhman/numbox/pull/9) at [`49a67d5`](https://github.com/Goykhman/numbox/commit/49a67d5); tagged `0.5.9`. Promoted `array_data_p`, `load_lib_path`, `cres_if_available`, [`core/bindings/abi.py`](numbox/core/bindings/abi.py) (struct-by-value helpers + `_is_win` gate), [`utils/meminfo.py`](numbox/utils/meminfo.py) bridge intrinsics, and [`core/vector.py`](numbox/core/vector.py). Fork PR [nelson2005/numbox#6](https://github.com/nelson2005/numbox/pull/6) closed as superseded.

## Follow-ups

- **`Vector` vs `List` benchmark in [`stress_work_runner`](test/stress_work_runner.py).** Goykhman flagged this as an interesting comparison in the [#9 review thread](https://github.com/Goykhman/numbox/pull/9#discussion_r3139779580): [`Work`](numbox/core/work/work.py#L52)/[`Node`](numbox/core/work/node.py#L62) currently use numba's reflected `List` for uniformly-cast inputs, and `Vector`'s contiguous-storage + geometric-growth shape may outperform on push-heavy workloads at large N. Prototype by swapping the `List` site, then compare timings from `stress_work_runner`.
- **Promote SysV x86-64 byval+optnone pattern into numbox.** [`core/bindings/abi.py`](numbox/core/bindings/abi.py) currently covers only the ≤16-byte register-passing path (`_call_lib_func_struct_in`/`_struct_out` raise `TypingError` above 16 bytes) and the attribute-free by-pointer path (`_call_lib_func_byval`). Neither wraps the >16-byte SysV idiom (alloca + store + `byval` arg attribute + `optnone`/`noinline` function attributes) that numbduck currently needs for [`_duckdb_create_decimal`, `_duckdb_create_varint`, and `_duckdb_bind_decimal`](https://github.com/Goykhman/numbduck/blob/feat/use-numbox-generics/numbduck/ducklib.py) (all 24-byte structs). Add a helper — working name `_call_lib_func_byval_large` — so numbduck can drop its local `_is_sysv_x86_64 = not _is_win and platform.machine() in ("x86_64", "AMD64")` bridge. See [llvmlite#300 comment](https://github.com/numba/llvmlite/issues/300#issuecomment-327235846) for the underlying LLVM optimizer concern.
