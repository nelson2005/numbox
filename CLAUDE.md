# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

numbox — a toolbox of low-level utilities for working with numba. Provides type erasure (`Any`), native library bindings (`Bindings`), graph nodes (`Node`), function proxies (`Proxy`), graph calculation (`Variable`), and units of work (`Work`).

## Build & Dev

- Venv: `python3.12 -m venv venv && venv/bin/pip install -e . flake8 pytest`
- Install: `pip install -e .`
- Test: `pytest`
- Lint: `flake8` (config in `.flake8`: max-line-length=127, default rules, per-file F403/F405 ignore for `test/core/test_bindings.py`)
- Docs: `cd docs && make html` (Sphinx)
- Python: >=3.10 (CI tests 3.10–3.14; local venv pinned to 3.12)
- Key dependency: `numba>=0.60.0,<0.66.0` (matches `pyproject.toml`; local venv has `numba==0.65.1`)

## Architecture

### Bindings System (core/bindings/)

The bindings subsystem wraps C library functions for use inside numba `@njit` code. Four layers:

1. **`utils.py`** — loads shared libraries via `ctypes.CDLL` with `RTLD_GLOBAL` so symbols are visible to LLVM
2. **`signatures.py`** — flat dict mapping C function names to numba type signatures (e.g., `"cos": float64(float64)`). Organized by library: `signatures_c`, `signatures_m`, `signatures_sqlite`
3. **`call.py`** — `@numba.extending.intrinsic` that generates LLVM IR to call native functions directly via `llvmlite`
4. **`_math.py`, `_c.py`, `_sqlite_*.py`** — thin Python wrappers using `@proxy(signatures.get("func"), jit_options={"cache": True})`

### Adding a New Binding

1. Add signature to `signatures.py` in the appropriate sub-dict
2. Add wrapper to the corresponding `_*.py` file following this pattern:
```python
@proxy(signatures.get("func_name"), jit_options={"cache": True})
def func_name(x):
    return _call_lib_func("func_name", (x,))
```
3. Function names must match the C library names exactly
4. Args passed as tuple literal to `_call_lib_func`
5. **Docs:** for a wrapper added to an existing `_*.py` module, the `automodule` directive in `docs/numbox.core.bindings.rst` picks it up automatically — nothing to edit. For a **new** `_*.py` module, OR if you rename / delete an existing module, also update `docs/numbox.core.bindings.rst`: the "Bindings module conventions" family list AND add / remove / rename the per-module `automodule` section under "Modules". Then run `cd docs && /home/erik/projects/numbox/venv/bin/sphinx-build -b html . _build/html` and confirm exit 0 (warning count stable is OK).

### LLVM symbol resolution and macOS

LLVM's JIT linker resolves extern symbols via `llvm::sys::DynamicLibrary::SearchForAddressOfSymbol`. This checks, in order: (1) the `ExplicitSymbols` map (populated by `llvmlite.binding.add_symbol`), (2) handles loaded via `LoadLibraryPermanently`, (3) `dlsym(process_handle, name)` after the process handle is registered (which happens on the first `@njit` compile). The process handle is equivalent to `dlsym(RTLD_DEFAULT, ...)` — it searches all Mach-O images / ELF objects in load order and returns the first match.

On **Linux**, `RTLD_GLOBAL` via `ctypes.CDLL` is sufficient: there is typically one copy of any given library, so `dlsym(RTLD_DEFAULT)` finds the right one after JIT init.

On **macOS**, the system sqlite is in the [dyld shared cache](https://keith.github.io/xcode-man-pages/dyld.1.html), mapped into every process at launch — before any user `dlopen`. `dlsym(RTLD_DEFAULT, "sqlite3_open")` returns the shared-cache address (system sqlite), not the Homebrew or framework-bundled version that Python's `_sqlite3.so` actually uses. `RTLD_GLOBAL` and `load_library_permanently` cannot change this — the shared cache is always first in load order.

The fix: [`numbox/utils/pysqlite_bridge.py`](numbox/utils/pysqlite_bridge.py) reads the correct addresses from `_sqlite3.so` via `ctypes.CDLL(handle).symbol` (which uses `dlsym(handle, name)` — searches the library + its dependencies, respecting two-level namespace) and registers them with `add_symbol`. LLVM checks `ExplicitSymbols` before any `dlsym` fallback, so the correct addresses win.

This applies to **any** library where macOS ships a system copy in the shared cache (sqlite, libz, libxml2, etc.). If a future binding wraps a library that has both a system and a user-installed version on macOS, the same `add_symbol` pattern from `pysqlite_bridge.py` is needed.

### Bindings: implementation notes

**Symbol resolution must use extern refs, not literal addresses.** [`ll.address_of_symbol(name)`](https://llvmlite.readthedocs.io/en/latest/user-guide/binding/modules.html) at lowering time returns the *current process's* runtime address — useful only as a presence check. Baking that int into LLVM IR breaks `cache=True` because ASLR randomizes the address per process and cached objects are meant to survive across runs and machines. The correct pattern, used by [`_call_lib_func`](numbox/core/bindings/call.py) itself: emit an extern declaration with [`get_or_insert_function(builder.module, func_ll_ty, func_name)`](numbox/core/bindings/call.py#L185) and let llvmlite's JIT linker resolve the name at link time. The [literal-address check](numbox/core/bindings/call.py#L76) earlier in the intrinsic is *only* a presence assertion; `func_p_as_int` is never consumed by codegen. The same extern-ref pattern works for data symbols (`@stdout = external global ptr`) and for accessor functions whose return value is per-thread ([`__errno_location`](https://man7.org/linux/man-pages/man3/errno.3.html), `__error`, `_errno`).

**Reuse the existing pointer/string helpers; don't reinvent.** Already in [`numbox/utils/lowlevel.py`](numbox/utils/lowlevel.py):

- [`array_data_p(arr) -> intp`](numbox/utils/lowlevel.py#L297) — numpy array data pointer (signed). Python- and `@njit`-callable.
- [`get_str_from_p_as_int(p) -> unicode_type`](numbox/utils/lowlevel.py#L148) — read NUL-terminated C string at address `p` into a Python `unicode_type`. Capped at [`MAX_STR_LENGTH`](numbox/core/configurations.py) (= `2**31 - 1`; the cap bounds the `carray` view, the loop exits on first NUL). `@njit`-callable.
- [`get_unicode_data_p(s) -> intp`](numbox/utils/lowlevel.py#L174) — pointer to a Python unicode's data payload (null-terminated). `@njit`-callable.

These are the canonical primitives for C-string interop. New bindings should compose them, not reimplement byte loops or pointer casts. **Before designing anything that touches strings, pointers, or buffer ownership, read [`numbox/utils/lowlevel.py`](numbox/utils/lowlevel.py) end-to-end first.**

**Public surface is star-imported.** [`numbox/core/bindings/__init__.py`](numbox/core/bindings/__init__.py) does `from numbox.core.bindings._c import *` (and same for `_math`, `_sqlite_*`). Anything at top level without a leading underscore is part of the public API. Keep new intrinsics private (`_`-prefixed); keep user-facing wrappers public.

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
- `numbox/core/bindings/_sqlite_conn.py` — connection + metadata wrappers; initializes module-level `sqlite3_lib`
- `numbox/core/bindings/_sqlite_stmt.py` — statement lifecycle
- `numbox/core/bindings/_sqlite_bind.py` — parameter binding
- `numbox/core/bindings/_sqlite_column.py` — column accessors
- `numbox/core/bindings/_sqlite_exec.py` — exec + free
- `numbox/core/bindings/_sqlite_blob.py` — BLOB incremental I/O
- `numbox/core/bindings/_sqlite_hooks.py` — callback hooks
- `numbox/core/bindings/_sqlite_constants.py` — SQLite result codes, type codes, flags, destructor sentinels
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
- **SQLite bindings buildout** — Expands `signatures_sqlite` from 3 entries to 60 (57 new) across [_sqlite_conn.py](numbox/core/bindings/_sqlite_conn.py), [_sqlite_stmt.py](numbox/core/bindings/_sqlite_stmt.py), [_sqlite_bind.py](numbox/core/bindings/_sqlite_bind.py), [_sqlite_column.py](numbox/core/bindings/_sqlite_column.py), [_sqlite_exec.py](numbox/core/bindings/_sqlite_exec.py), [_sqlite_blob.py](numbox/core/bindings/_sqlite_blob.py), [_sqlite_hooks.py](numbox/core/bindings/_sqlite_hooks.py), and [_sqlite_constants.py](numbox/core/bindings/_sqlite_constants.py). Adds Windows support via a new generic `_windows_bundled_dll_path` fallback in [utils.py](numbox/core/bindings/utils.py) (locates `sqlite3.dll` in CPython's `<base_prefix>/DLLs/` and conda's `<base_prefix>/Library/bin/`). Refactors `load_lib` into `_resolve_lib_path` + `load_lib_with_handle` so `proxy_if_available` has access to the CDLL handle. Five functions are decorated with `proxy_if_available`: `sqlite3_changes64` / `sqlite3_total_changes64` (SQLite 3.37+; Python 3.10 ships 3.34) and `sqlite3_column_database_name` / `_table_name` / `_origin_name` (compile-time `SQLITE_ENABLE_COLUMN_METADATA`).

  **Four lifetime gotchas worth memorizing:**
  1. **`bind_text` / `bind_blob` destructor** — pass `SQLITE_TRANSIENT` (= -1) for SQLite to copy. `SQLITE_STATIC` (= 0) assumes the buffer outlives the statement.
  2. **`sqlite3_errmsg` lifetime** — SQLite-owned pointer, valid only until the next API call on the same connection. Decode via `get_str_from_p_as_int` immediately.
  3. **`sqlite3_expanded_sql` cleanup** — caller must release with `sqlite3_free()` (`sqlite3_expanded_sql` bound in `_sqlite_stmt.py`; `sqlite3_free` in `_sqlite_exec.py`).
  4. **cfunc lifetime for hook APIs** — registered cfunc must outlive the hook registration. Keep the cfunc at module scope in the caller.

  Callback wiring pattern for `sqlite3_exec` and the six hook APIs:
  ```python
  @cfunc(types.int32(types.voidptr, types.int32, types.intp, types.intp))
  def _row_cb(ctx, ncol, values_pp, names_pp):
      return 0
  sqlite3_exec(db_p, sql_p, _row_cb.address, ctx_p, 0)
  ```

  macOS symbol resolution writeup with CI evidence: [gist](https://gist.github.com/nelson2005/203d078fb0e6cdd3f2ff16a7cce7a77d).
- **SQLite UDF/UDAF/window function bindings (phase 2)** — merged 2026-05-30 via upstream [Goykhman/numbox#17](https://github.com/Goykhman/numbox/pull/17) at [`6243a80`](https://github.com/Goykhman/numbox/commit/6243a80) (fork review [PR #34](https://github.com/nelson2005/numbox/pull/34) closed unmerged — review vehicle only). Adds 34 bindings for scalar/aggregate/window UDFs callable from `@njit` code across three modules: [`_sqlite_value.py`](numbox/core/bindings/_sqlite_value.py) (value accessors for reading UDF args), [`_sqlite_result.py`](numbox/core/bindings/_sqlite_result.py) (result setters), and [`_sqlite_udf.py`](numbox/core/bindings/_sqlite_udf.py) (`sqlite3_create_function_v2` / `sqlite3_create_window_function` / `sqlite3_aggregate_context` / `sqlite3_user_data` / `sqlite3_context_db_handle`), plus subtype constants in `_sqlite_constants.py`. Aggregate/window state lives in the `sqlite3_aggregate_context` slot with the meminfo released in `xFinal` (never `xValue`); the tests demonstrate **both idioms** — a structref-backed aggregate (8-byte slot → meminfo → numba structref via [`numbox/utils/meminfo.py`](numbox/utils/meminfo.py)) and an array-backed window function (16-byte `[meminfo_p, data_p]` slot for an NRT array payload, incref'd via the inlined `_incref_meminfo` intrinsic so `removerefctpass` cannot strip it; same shape as numbduck's array UDAF). `test_udaf_no_meminfo_leak` exercises both paths.
- **SQLite `query_to_array` + vtable pushdown + TVFs (phase 5)** — merged 2026-06-11 via upstream [Goykhman/numbox#22](https://github.com/Goykhman/numbox/pull/22) at [`50df0f4`](https://github.com/Goykhman/numbox/commit/50df0f4) (fork review [PR #46](https://github.com/nelson2005/numbox/pull/46) — review vehicle only, never merged to fork main; full 25-comment maintainer-review disposition recorded in fork issue [#51](https://github.com/nelson2005/numbox/issues/51)). Three new modules — [`_sqlite_query.py`](numbox/core/bindings/_sqlite_query.py) (`query_to_array`: run SQL from `@njit` code and materialize the result as a 2-D float64 numpy array; raises on mid-step errors), [`_sqlite_tvf.py`](numbox/core/bindings/_sqlite_tvf.py) (`register_tvf`: table-valued functions backed by `@cfunc` kernels, with numeric-arg validation and a per-arg `argvIndex` bitmask in `xBestIndex`), [`_sqlite_typemap.py`](numbox/core/bindings/_sqlite_typemap.py) (packed-cell tag/spec map + unaligned UTF-32 stores with UTF-8 validation) — plus a major [`_sqlite_vtable.py`](numbox/core/bindings/_sqlite_vtable.py) expansion (read-only vtables over structured numpy arrays with eq/range constraint pushdown including bool columns via `xBestIndex`, `create_module_v2` with deferred `xDestroy`, `SQLITE_STATIC` for string/BLOB results), and align-1 [`load_unaligned`/`store_unaligned`](numbox/utils/lowlevel.py) helpers with IR-level `align 1` regression tests (`load_at`/`store_at` are UB on misaligned addresses).

  **Platform gotcha worth memorizing:** on macOS-arm64 (py3.14 / numba 0.65.1) raw-pointer result stores (`array_data_p` + `store_unaligned`) were dead-code-eliminated — `query_to_array` returned correct-shape all-zeros on that toolchain only. Fix: write results through `out.view(np.uint8)`, a numba-tracked view the optimizer can't drop, with native byte-order serialization via a typed view of an 8-byte scratch buffer.
- **compile_kernel (`Variable` graph → fused `@njit` kernel)** — upstream [Goykhman/numbox#24](https://github.com/Goykhman/numbox/pull/24) OPEN (supersedes closed [#23](https://github.com/Goykhman/numbox/pull/23); fork review [nelson2005/numbox#52](https://github.com/nelson2005/numbox/pull/52) — review vehicle only; earlier vehicle [#49](https://github.com/nelson2005/numbox/pull/49) closed unmerged). New `numbox/core/variable/compile_kernel.py`: `compile_kernel(graph, required)` compiles a `Variable` graph into one fused `@njit` kernel for the requested variables, alongside `core.work` — no per-node types needed (numba infers every interior type from runtime args; plain-Python formulas auto-wrapped with `njit()`). Reuses `Graph.compile` (topological order + external-variable discovery) and `utils/preprocessing`'s content-addressed anchor (structured fingerprint covering each formula's code, consts, defaults, closure cells, referenced globals/helpers, module, and jit flags; cached kernels reload cross-process). `CompiledKernel` exposes a bare `.kernel` hot path plus a dict-in/out `.execute` mirroring `CompiledGraph.execute`; generated identifiers are keyword- and newline-safe with minimal deterministic suffixes. v1 omits per-node `cacheable` memoization, incremental `recompute`, `None`-as-value formulas, and node-identity load/combine (use `CompiledGraph`/`Work`). Design + plan under `docs/superpowers/{specs,plans}/2026-06-07-compile-kernel*` (fork-only, not on `main`); tests in `test/core/test_compile_kernel.py`.

## Follow-ups

- **`Vector` vs `List` benchmark in [`stress_work_runner`](test/stress_work_runner.py).** Goykhman flagged this as an interesting comparison in the [#9 review thread](https://github.com/Goykhman/numbox/pull/9#discussion_r3139779580): [`Work`](numbox/core/work/work.py#L52)/[`Node`](numbox/core/work/node.py#L62) currently use numba's reflected `List` for uniformly-cast inputs, and `Vector`'s contiguous-storage + geometric-growth shape may outperform on push-heavy workloads at large N. Prototype by swapping the `List` site, then compare timings from `stress_work_runner`.
- **User-defined SQL functions** — `sqlite3_create_function_v2` + the `sqlite3_value_*` / `sqlite3_result_*` API surface. Its own significant buildout; relevant for numbduck-style consumers but separate scope.
- **Backup API** — `sqlite3_backup_init` / `_step` / `_finish` / `_pagecount` / `_remaining`. Six functions plus the progress callback shape.
- **Serialize / deserialize** — `sqlite3_serialize` / `sqlite3_deserialize` for in-memory snapshots.
- **Higher-level structref wrappers** — `Connection` / `Statement` structrefs if a downstream consumer needs ergonomic types.
- **Drop `proxy_if_available` gate on `changes64` / `total_changes64`** once Python 3.10 drops out of the support matrix (Python 3.11+ all ship SQLite 3.37+).
