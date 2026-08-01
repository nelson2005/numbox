# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

numbox — a toolbox of low-level utilities for working with numba. Provides type erasure (`Any`), native library bindings (`Bindings`), graph nodes (`Node`), function proxies (`Proxy`), graph calculation (`Variable`), and units of work (`Work`).

## Build & Dev

- Venv: `python3.12 -m venv venv && venv/bin/pip install -e . flake8 pytest`
- Install: `pip install -e .`
- Test: `pytest`
- Lint: `flake8` (config in `.flake8`: max-line-length=127, default rules)
- Docs: `cd docs && make html` (Sphinx)
- Python: >=3.10 (CI tests 3.10–3.14; local venv pinned to 3.12)
- Key dependency: `numba>=0.60.0,<0.67.0` (matches `pyproject.toml`; local venv has `numba==0.65.1`)

## Architecture

### Bindings System (core/bindings/)

The bindings subsystem wraps C library functions for use inside numba `@njit` code. Four layers:

1. **`utils.py`** — loads shared libraries via `ctypes.CDLL` with `RTLD_GLOBAL` so symbols are visible to LLVM
2. **`signatures.py`** — flat dict mapping C function names to numba type signatures (e.g., `"cos": float64(float64)`). Organized by library: `signatures_c`, `signatures_m`, `signatures_sqlite`
3. **`call.py`** — `@numba.extending.intrinsic` that generates LLVM IR to call native functions directly via `llvmlite`
4. **`libm.py`, `libc.py`, `sqlite/*.py`** — thin Python wrappers using `@proxy(signatures.get("func"), jit_options={"cache": True})`

### Adding a New Binding

1. Add signature to `signatures.py` in the appropriate sub-dict
2. Add the wrapper to the corresponding module (e.g. `libm.py`, `libc.py`, `sqlite/conn.py`) and add its public name to that module's `__all__`, following this pattern:
```python
@proxy(signatures.get("func_name"), jit_options={"cache": True})
def func_name(x):
    return _call_lib_func("func_name", (x,))
```
3. Function names must match the C library names exactly
4. Args passed as tuple literal to `_call_lib_func`
5. **Docs:** for a wrapper added to an existing binding module, the `automodule` directive in `docs/numbox.core.bindings.rst` picks it up automatically — nothing to edit. For a **new** binding module, OR if you rename / delete an existing module, also update `docs/numbox.core.bindings.rst`: the "Bindings module conventions" family list AND add / remove / rename the per-module `automodule` section under "Modules". Then run `cd docs && /home/erik/projects/numbox/venv/bin/sphinx-build -b html . _build/html` and confirm exit 0 (warning count stable is OK).

### LLVM symbol resolution and macOS

LLVM's JIT linker resolves extern symbols via `llvm::sys::DynamicLibrary::SearchForAddressOfSymbol`. This checks, in order: (1) the `ExplicitSymbols` map (populated by `llvmlite.binding.add_symbol`), (2) handles loaded via `LoadLibraryPermanently`, (3) `dlsym(process_handle, name)` after the process handle is registered (which happens on the first `@njit` compile). The process handle is equivalent to `dlsym(RTLD_DEFAULT, ...)` — it searches all Mach-O images / ELF objects in load order and returns the first match.

On **Linux**, `RTLD_GLOBAL` via `ctypes.CDLL` is sufficient: there is typically one copy of any given library, so `dlsym(RTLD_DEFAULT)` finds the right one after JIT init.

On **macOS**, the system sqlite is in the [dyld shared cache](https://keith.github.io/xcode-man-pages/dyld.1.html), mapped into every process at launch — before any user `dlopen`. `dlsym(RTLD_DEFAULT, "sqlite3_open")` returns the shared-cache address (system sqlite), not the Homebrew or framework-bundled version that Python's `_sqlite3.so` actually uses. `RTLD_GLOBAL` and `load_library_permanently` cannot change this — the shared cache is always first in load order.

How [`numbox/utils/pysqlite_bridge.py`](numbox/utils/pysqlite_bridge.py) handles it: it does **not** override symbol resolution. It *detects* the mismatch — `libraries_coordinated()` compares numbox's linked sqlite version (`sqlite3_libversion`) against Python's `sqlite3.sqlite_version` — and `extract_connection_ptr` raises rather than hand a `sqlite3*` from one sqlite to bindings linked against another. The supported fix is to force-load the right copy first: `DYLD_INSERT_LIBRARIES=/path/to/your/libsqlite3.dylib`.

This detect-and-refuse (plus `DYLD_INSERT_LIBRARIES`) guard applies to **any** library macOS ships in the shared cache (sqlite, libz, libxml2, etc.). Note: the `add_symbol`/`ExplicitSymbols` path in the resolution order above *is* used in numbox — [`numbox/core/proxy/proxy.py`](numbox/core/proxy/proxy.py) registers a process-stable alias for each proxied body's cfunc wrapper — but for JIT symbol aliasing, not macOS library coordination.

### Bindings: implementation notes

**Symbol resolution must use extern refs, not literal addresses.** [`ll.address_of_symbol(name)`](https://llvmlite.readthedocs.io/en/latest/user-guide/binding/modules.html) at lowering time returns the *current process's* runtime address — useful only as a presence check. Baking that int into LLVM IR breaks `cache=True` because ASLR randomizes the address per process and cached objects are meant to survive across runs and machines. The correct pattern, used by [`_call_lib_func`](numbox/core/bindings/call.py) itself: emit an extern declaration with [`get_or_insert_function(builder.module, func_ll_ty, func_name)`](numbox/core/bindings/call.py#L185) and let llvmlite's JIT linker resolve the name at link time. The [literal-address check](numbox/core/bindings/call.py#L76) earlier in the intrinsic is *only* a presence assertion; `func_p_as_int` is never consumed by codegen. The same extern-ref pattern works for data symbols (`@stdout = external global ptr`) and for accessor functions whose return value is per-thread ([`__errno_location`](https://man7.org/linux/man-pages/man3/errno.3.html), `__error`, `_errno`).

**Reuse the existing pointer/string helpers; don't reinvent.** Already in [`numbox/utils/lowlevel.py`](numbox/utils/lowlevel.py):

- [`array_data_p(arr) -> intp`](numbox/utils/lowlevel.py#L297) — numpy array data pointer (signed). Python- and `@njit`-callable.
- [`get_str_from_p_as_int(p) -> unicode_type`](numbox/utils/lowlevel.py#L148) — read NUL-terminated C string at address `p` into a Python `unicode_type`. Capped at [`MAX_STR_LENGTH`](numbox/core/configurations.py) (= `2**31 - 1`; the cap bounds the `carray` view, the loop exits on first NUL). `@njit`-callable.
- [`get_unicode_data_p(s) -> intp`](numbox/utils/lowlevel.py#L174) — pointer to a Python unicode's data payload (null-terminated). `@njit`-callable.

These are the canonical primitives for C-string interop. New bindings should compose them, not reimplement byte loops or pointer casts. **Before designing anything that touches strings, pointers, or buffer ownership, read [`numbox/utils/lowlevel.py`](numbox/utils/lowlevel.py) end-to-end first.**

**Public surface is per-module — no star-import.** [`numbox/core/bindings/__init__.py`](numbox/core/bindings/__init__.py) is intentionally empty so that importing one binding does not eagerly compile the whole subsystem (a `cos` import used to drag in all of sqlite — ~12.8 s cold). Import from the specific module: `from numbox.core.bindings.libm import cos`, `from numbox.core.bindings.sqlite.conn import sqlite3_open`. Each module's `__all__` is its public API; keep intrinsics private (`_`-prefixed). Do NOT re-add re-exports to `__init__.py`.

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
- `numbox/core/bindings/libm.py` — libm wrappers (33 single-arg + 9 two-arg float64 functions)
- `numbox/core/bindings/libc.py` — libc wrappers
- `numbox/core/bindings/sqlite/conn.py` — connection + metadata wrappers; initializes module-level `sqlite3_lib`
- `numbox/core/bindings/sqlite/stmt.py` — statement lifecycle
- `numbox/core/bindings/sqlite/bind.py` — parameter binding
- `numbox/core/bindings/sqlite/column.py` — column accessors
- `numbox/core/bindings/sqlite/exec.py` — exec + free
- `numbox/core/bindings/sqlite/blob.py` — BLOB incremental I/O
- `numbox/core/bindings/sqlite/hooks.py` — callback hooks
- `numbox/core/bindings/sqlite/constants.py` — SQLite result codes, type codes, flags, destructor sentinels
- `numbox/core/bindings/sqlite/_typemap.py` — private column-tag/typemap helpers (not part of the public surface)
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

## Gotchas

Hard-won notes that aren't obvious from the code. Feature/merge history lives in the upstream PRs and `git log`.

### SQLite C API lifetimes

- **`bind_text` / `bind_blob` destructor** — pass `SQLITE_TRANSIENT` (= -1) so SQLite copies the buffer; `SQLITE_STATIC` (= 0) assumes it outlives the statement.
- **`sqlite3_errmsg`** — SQLite-owned pointer, valid only until the next API call on that connection; decode via `get_str_from_p_as_int` immediately.
- **`sqlite3_expanded_sql`** — caller must release it with `sqlite3_free()`.
- **Hook / `sqlite3_exec` `@cfunc`** — the registered cfunc must outlive the registration; keep it at module scope. Callback shape:
  ```python
  @cfunc(types.int32(types.voidptr, types.int32, types.intp, types.intp))
  def _row_cb(ctx, ncol, values_pp, names_pp):
      return 0
  sqlite3_exec(db_p, sql_p, _row_cb.address, ctx_p, 0)
  ```
- macOS symbol-resolution writeup with CI evidence: [gist](https://gist.github.com/nelson2005/203d078fb0e6cdd3f2ff16a7cce7a77d) (see also "LLVM symbol resolution and macOS" above).

### Codegen / numba

- **macOS-arm64 dead-code-elimination** (observed on py3.14 / numba 0.65.1): raw-pointer result stores (`array_data_p` + `store_unaligned`) were silently dropped, so `query_to_array` returned correct-shape all-zeros on that toolchain only. Write results through a numba-tracked `out.view(np.uint8)` (native byte-order via a typed view of an 8-byte scratch buffer), never a raw pointer.
- **`@proxy` / `make_structref` cache anchors** ([`numbox/utils/preprocessing.py`](numbox/utils/preprocessing.py)) work around CPython [#122981](https://github.com/python/cpython/issues/122981) (the `@proxy` decorator line must be `co_firstlineno`) **and** numba's `co_consts` cache-key collision (numba hashes `co_code` but not `co_consts`, so pure-numeric-literal body edits don't shift the key).
- **`Record` LARGE struct returns are rejected** by the ABI dispatcher: RecordModel uses a raw `[N x i8]*`, so a stack-alloca `sret` would dangle after the `@njit` return — it needs NRT-allocated storage. `Tuple` / `UniTuple` LARGE returns work.

## Follow-ups

- **`Vector` vs `List` benchmark in [`stress_work_runner`](test/stress_work_runner.py).** Goykhman flagged this as an interesting comparison in the [#9 review thread](https://github.com/Goykhman/numbox/pull/9#discussion_r3139779580): [`Work`](numbox/core/work/work.py#L52)/[`Node`](numbox/core/work/node.py#L62) currently use numba's reflected `List` for uniformly-cast inputs, and `Vector`'s contiguous-storage + geometric-growth shape may outperform on push-heavy workloads at large N. Prototype by swapping the `List` site, then compare timings from `stress_work_runner`.
- **Backup API** — `sqlite3_backup_init` / `_step` / `_finish` / `_pagecount` / `_remaining`. Six functions plus the progress callback shape.
- **Serialize / deserialize** — `sqlite3_serialize` / `sqlite3_deserialize` for in-memory snapshots.
- **Higher-level structref wrappers** — `Connection` / `Statement` structrefs if a downstream consumer needs ergonomic types.
- **Drop `proxy_if_available` gate on `changes64` / `total_changes64`** once Python 3.10 drops out of the support matrix (Python 3.11+ all ship SQLite 3.37+).
- **Bound the Work-method cache for a Dispatcher-typed graph node.** A `make_graph` node whose value is an `@njit` dispatcher (`End(init_value=some_njit_fn)`) makes numba's own `@njit(cache=True)` [`Work`](numbox/core/work/work.py) structref methods (`Work.calculate`, the `ol_calculate`-generated `_calculate_`, etc.) specialize on the Dispatcher-carrying Work type — which numba cannot cross-process-cache — so `work.*.nbc` grows one entry per process (measured 1→2→3). Never a stale/wrong binary (recompile only). The [#73](https://github.com/nelson2005/numbox/issues/73) sibling-site fix closes the numbox-named kernel/derive units but not this, because the methods' `cache` flag is fixed at import and cannot be made per-type without gating each proxy method + overload behind a recursive `_embeds_dispatcher(work_ty)` check (a sizeable, higher-risk change to the core structref). Exotic pattern; left as a deliberate follow-up.
- **Detect a swallowed `Work.derive` exception in tooling.** A derive that raises has its exception discarded at the first-class call boundary: `calculate()` returns normally, `data` is left zero-filled, and `derived` is set anyway, so the wrong value is cached and never recomputed. For `unicode_type` data, reading it back from Python segfaults. Reproduces through every construction path, `make_work`, `ll_make_work`, `make_work_helper(derive_py=...)` and the `End`/`Derived`/`make_graph` builder, so it is not `cres`-specific. The contract is documented as of [Goykhman/numbox#36](https://github.com/Goykhman/numbox/pull/36), but nothing detects a violation. Open question: can numbox catch it itself, rather than relying on users to wrap every derive in `try`/`except`? One candidate is a success-path sentinel in a tuple return, verified to work but it changes each node's `data` type; another is putting the guard in the wrapper `_derive_anchor_cres` ([`builder.py`](numbox/core/work/builder.py)) already generates, which would cover the builder path with no user discipline. Background, repros and the maintainer's agreement to document it are in fork issue [#74](https://github.com/nelson2005/numbox/issues/74). Upstream is a dead end: the swallow is [numba#8246](https://github.com/numba/numba/issues/8246), which numba's own suite pins as expected behaviour in `TestExceptionInFunctionType.test_exception_ignored_in_cfunc`.
- **Decide whether the `@proxy` cache guard should cover CUDA cache loads.** [`_install_cache_alias_guard`](numbox/core/proxy/proxy.py) wraps `CompileResultCacheImpl` and `CodeLibraryCacheImpl`, but `numba.cuda.dispatcher.CUDACacheImpl` subclasses `CacheImpl` **directly** (`mro` is `['CUDACacheImpl', 'CacheImpl', 'object']`), so it is never wrapped and a CUDA kernel cached against a since-renamed `numbox_pxy_*` alias loads unvalidated. Covering it needs a **third** payload extractor, not a reuse of either existing one: `CUDACacheImpl.reduce` returns `kernel._reduce_states()`, a **dict** (`_Kernel._rebuild` takes keyword args), so both `lambda payload: payload[0]` and the identity lambda fail on it. This is a coverage gap, not a broken extraction — the two wrapped paths are verified correct on every numba in the supported range. Established by source and `mro` inspection only; no CUDA load was executed. Either wrap it with a dict-aware extractor, or state in the proxy docs that CUDA caches are out of scope.
