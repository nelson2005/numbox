# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

numbox — a toolbox of low-level utilities for working with numba. Provides type erasure (`Any`), native library bindings (`Bindings`), graph nodes (`Node`), function proxies (`Proxy`), graph calculation (`Variable`), and units of work (`Work`).

## Build & Dev

- Venv: `python3.12 -m venv venv && venv/bin/pip install -e . flake8 pytest`
- Install: `pip install -e .`
- Test: `pytest`
- Lint: `flake8` (max-line-length=127, max-complexity=10)
- Docs: `cd docs && make html` (Sphinx)
- Python: >=3.10 (CI tests 3.10–3.14; local venv pinned to 3.12)
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

### Bindings: implementation gotchas

These have caught cleanly-reasoned designs more than once. Apply to all new bindings, intrinsics, and platform-aware additions.

**Symbol resolution must use extern refs, not literal addresses.** [`ll.address_of_symbol(name)`](https://llvmlite.readthedocs.io/en/latest/user-guide/binding/modules.html) at lowering time returns the *current process's* runtime address — useful only as a presence check. Baking that int into LLVM IR breaks `cache=True` because ASLR randomizes the address per process and cached objects are meant to survive across runs and machines. The correct pattern, used by [`_call_lib_func`](numbox/core/bindings/call.py) itself: emit an extern declaration with [`get_or_insert_function(builder.module, func_ll_ty, func_name)`](numbox/core/bindings/call.py#L185) and let llvmlite's JIT linker resolve the name at link time. The [literal-address check](numbox/core/bindings/call.py#L76) earlier in the intrinsic is *only* a presence assertion; `func_p_as_int` is never consumed by codegen. The same extern-ref pattern works for data symbols (`@stdout = external global ptr`) and for accessor functions whose return value is per-thread ([`__errno_location`](https://man7.org/linux/man-pages/man3/errno.3.html), `__error`, `_errno`).

**Reuse the existing pointer/string helpers; don't reinvent.** Already in [`numbox/utils/lowlevel.py`](numbox/utils/lowlevel.py):

- [`array_data_p(arr) -> intp`](numbox/utils/lowlevel.py#L297) — numpy array data pointer (signed). Python- and `@njit`-callable.
- [`get_str_from_p_as_int(p) -> unicode_type`](numbox/utils/lowlevel.py#L148) — read NUL-terminated C string at address `p` into a Python `unicode_type`. Capped at [`MAX_STR_LENGTH`](numbox/core/configurations.py) (= `2**31 - 1`; the cap bounds the `carray` view, the loop exits on first NUL). `@njit`-callable.
- [`get_unicode_data_p(s) -> intp`](numbox/utils/lowlevel.py#L174) — pointer to a Python unicode's data payload (null-terminated). `@njit`-callable.

These are the canonical primitives for C-string interop. New bindings should compose them, not reimplement byte loops or pointer casts. **Before designing anything that touches strings, pointers, or buffer ownership, read [`numbox/utils/lowlevel.py`](numbox/utils/lowlevel.py) end-to-end first.**

**Public surface is star-imported.** [`numbox/core/bindings/__init__.py`](numbox/core/bindings/__init__.py) does `from numbox.core.bindings._c import *` (and same for `_math`, `_sqlite`). Anything at top level without a leading underscore is part of the public API. Keep new intrinsics private (`_`-prefixed); keep user-facing wrappers public.

**Platform-variable C types — `long`, `time_t`, `size_t`.** `long` is 64-bit on POSIX (LP64) but 32-bit on Windows x64 (LLP64); `time_t` size varies historically; `size_t` is 64-bit on all current 64-bit CI platforms. A `signatures` entry that uses `int64` for `long` will silently corrupt registers on Windows. Functions affected: `fseek`/`ftell`/`fsetpos`/`fgetpos`, `time`/`clock`, `strtol`/`strtoul`. Either dispatch per platform (option-(ii) style: different symbol or signature per platform) or omit the function from the batch and document as a follow-up. Don't ship a uniform-`int64` signature that's correct on POSIX and wrong on Windows.

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
- **numbduck generics promotion** — merged 2026-04-24 via upstream [Goykhman/numbox#9](https://github.com/Goykhman/numbox/pull/9) at [`49a67d5`](https://github.com/Goykhman/numbox/commit/49a67d5); tagged `0.5.9`. Promoted `array_data_p`, `load_lib_path`, `cres_if_available`, [`core/bindings/abi.py`](numbox/core/bindings/abi.py) (struct-by-value helpers + `_is_win` gate), [`utils/meminfo.py`](numbox/utils/meminfo.py) bridge intrinsics, and [`core/vector/vector.py`](numbox/core/vector/vector.py). Fork PR [nelson2005/numbox#6](https://github.com/nelson2005/numbox/pull/6) closed as superseded.
- **Unified `_call_lib_func` ABI dispatch** — merged 2026-04-27 via fork [nelson2005/numbox#8](https://github.com/nelson2005/numbox/pull/8) / upstream [Goykhman/numbox#10](https://github.com/Goykhman/numbox/pull/10) at [`92c0ca4`](https://github.com/Goykhman/numbox/commit/92c0ca4); tagged [`0.5.10`](https://github.com/Goykhman/numbox/releases/tag/0.5.10). Replaces the `_call_lib_func_struct_in` / `_call_lib_func_struct_out` / `_call_lib_func_args_struct_out` intrinsics with a single platform-aware dispatcher in [`numbox/core/bindings/call.py`](numbox/core/bindings/call.py) keyed off [`_classify`](numbox/core/bindings/abi.py) (scalar / small struct / large struct) × [`_current_platform`](numbox/core/bindings/abi.py) (Windows x64 / SysV x86-64 / AAPCS64). Subsumes the previously-planned `_call_lib_func_byval_large` helper. Unblocks the [numbduck](https://github.com/Goykhman/numbduck) migration off `_is_sysv_x86_64` once a numbduck branch picks up `0.5.10`.
- **SysV/AAPCS64 INT/INT eightbyte repack + >16B `sret` returns** — merged 2026-05-01 via upstream [Goykhman/numbox#11](https://github.com/Goykhman/numbox/pull/11) at [`6bcda41`](https://github.com/Goykhman/numbox/commit/6bcda41); tagged [`0.5.11`](https://github.com/Goykhman/numbox/releases/tag/0.5.11); synced into fork via [nelson2005/numbox#15](https://github.com/nelson2005/numbox/pull/15). Closes previously-open follow-ups for `>16B` struct returns and SysV x86-64 16B INT/INT repack. Two pieces:
  1. [`f04e0ac`](https://github.com/Goykhman/numbox/commit/f04e0ac) — detect 16B structs whose two eightbytes classify pure-INTEGER but whose LLVM type is not canonical `{i64, i64}` (e.g. `{i32, i32, i64}` — `duckdb_interval`) and repack via memory bitcast before the call (mirrored on the return side). Works around [llvmlite#300](https://github.com/numba/llvmlite/issues/300). New helpers in [`abi.py`](numbox/core/bindings/abi.py): `_iter_struct_fields`, `_classify_eightbytes` (SSE-wins per eightbyte, handles boundary-spanning fields), `_is_canonical_int64_pair_layout` (sorts Record fields by offset). Windows-x64 sret path untouched.
  2. [`59b79b5`](https://github.com/Goykhman/numbox/commit/59b79b5) — extend `use_sret` to `_CLASS_STRUCT_LARGE` returns; the alloca + sret-attributed hidden first arg + void return codegen path is already in place from the Windows `<=16B` case and applies uniformly on SysV x86-64, AAPCS64, and Win64. `Tuple`/`UniTuple` LARGE returns work end-to-end on all three; `Record` LARGE returns explicitly rejected (RecordModel uses raw `[N x i8]*` — stack-alloca sret would dangle after `@njit` return, needs NRT-allocated storage; add when a consumer arrives).

  Unblocks numbduck's `duckdb_interval` migration off the local `_build_packed_interval` workaround, and `_duckdb_get_decimal` / `_duckdb_get_varint` migration off hand-rolled alloca-pointer-pass.
- **libc bindings expansion + variadic I/O via `@proxy`** — merged 2026-05-23 via fork [nelson2005/numbox#20](https://github.com/nelson2005/numbox/pull/20) / upstream [Goykhman/numbox#12](https://github.com/Goykhman/numbox/pull/12) at [`9e92917`](https://github.com/Goykhman/numbox/commit/9e92917); tagged `0.5.12`. Adds [stdio handles + buffered I/O](numbox/core/bindings/_stdio.py) (stdout/stderr/stdin via the `__stdoutp`/`__stderrp`/`__stdinp` accessor pattern; fputs/fflush/fputc/fgetc/clearerr/feof/ferror), [thread-safe errno helpers](numbox/core/bindings/_errno.py) (errno_get/errno_set/errno_clear via `__errno_location` / `__error` / `_errno`), [thread-safe strerror](numbox/core/bindings/_strerror.py) (strerror_safe with per-platform symbol selection: `__xpg_strerror_r` on glibc and musl, `strerror_r` elsewhere), and [variadic format I/O](numbox/core/bindings/_fmtio.py) (printf/fprintf/snprintf/sscanf as dual-mode `@overload` + `@intrinsic`; `%n` rejected at typing time including BSD `q` and Microsoft `I32`/`I64` modifiers). Three load-bearing architectural pieces landed alongside:
  1. **`@proxy` consolidates `cres_cacheable`** — [`proxy.py`](numbox/core/proxy/proxy.py) gained `.as_func` (`CompileResultWAP` attached to the dispatcher); every libc binding wrapper uses `@proxy(sig, jit_options={"cache": True})`. `cres_cacheable` + `_addr_global.py` deleted. Anchor mechanism: blank-line-prepend at `inspect.getfile(func)` so `wrapper.co_firstlineno` lands on the user's `@proxy` decorator line (avoids CPython [#122981](https://github.com/python/cpython/issues/122981)).
  2. **`load_at` / `store_at`** in [`numbox/utils/lowlevel.py`](numbox/utils/lowlevel.py) generalize the int32-specific helpers via `context.get_data_type(ty)`. `load_at(p, TypeRef)` reads the LLVM layout from a numba `TypeRef`; `store_at(p, v)` reads from `v`'s numba type.
  3. **Content-addressed cache anchors** for `make_structref` extracted to [`numbox/utils/preprocessing.py`](numbox/utils/preprocessing.py) (anchor root + materialization + orphan sweep with 60s age filter). Works around both CPython [#122981](https://github.com/python/cpython/issues/122981) AND numba `co_consts` cache-key collisions (numba hashes `co_code` but not `co_consts`; pure-numeric-literal body changes don't shift `co_code` because `LOAD_CONST` encodes an index into `co_consts`).

  Follow-up upstream commit [`48d63e3`](https://github.com/Goykhman/numbox/commit/48d63e3) renamed `default_jit_options → jit_options` across the codebase (dict-keyword shape kept; only the symbol renamed).

## Follow-ups

- **`Vector` vs `List` benchmark in [`stress_work_runner`](test/stress_work_runner.py).** Goykhman flagged this as an interesting comparison in the [#9 review thread](https://github.com/Goykhman/numbox/pull/9#discussion_r3139779580): [`Work`](numbox/core/work/work.py#L52)/[`Node`](numbox/core/work/node.py#L62) currently use numba's reflected `List` for uniformly-cast inputs, and `Vector`'s contiguous-storage + geometric-growth shape may outperform on push-heavy workloads at large N. Prototype by swapping the `List` site, then compare timings from `stress_work_runner`.
