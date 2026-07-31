# libc bindings expansion — design

Target release: **0.5.12**. Date: 2026-05-11.

## 1. Summary

Expand `numbox.core.bindings` to expose three foundational capabilities currently missing, plus a 25-function monomorphic libc batch that they enable:

- **(A) Stdio handles** — `stdout()`, `stderr()`, `stdin()` callable from `@njit` code, with platform-aware extern-symbol references (no literal addresses baked into cached objects).
- **(B) Thread-safe errno access** — `errno_get()`, `errno_set(v)`, plus a unified cross-platform `strerror_safe(errnum, buf, buflen)` that hides the glibc / musl / macOS / Windows symbol-and-arg-order mess.
- **(C) Monomorphic libc batch** — 12 non-variadic stdio, 7 string, 5 memory, 1 env function, all fitting the existing `_call_lib_func` infrastructure.

Explicit non-goals for 0.5.12: variadic `printf` / `fprintf` / `scanf` family (deferred to a dedicated release), `fseek` / `ftell` (require LP64-vs-LLP64 platform dispatch), `setenv` / `unsetenv` (POSIX-only, Windows `_putenv_s` differs), `malloc` / `free` (NRT ownership composition concerns), time functions (`time_t` size variance), `setvbuf` / `setbuf` (require numbox-owned platform-aware mode constants).

## 2. Scope and non-scope

### In scope

| Capability | Functions / surface | Notes |
|---|---|---|
| (A) Stdio handles | `stdout`, `stderr`, `stdin` | Callable, return `intp` (current process's `FILE*`) |
| (B) errno | `errno_get`, `errno_set` | Per-thread, no caching of the pointer |
| (B) Thread-safe strerror | `strerror_safe(errnum, buf, buflen)` | Platform-dispatched to `__xpg_strerror_r` / `strerror_r` / `strerror_s` |
| (C) Stdio non-variadic | `puts`, `fputs`, `fputc`, `putchar`, `fwrite`, `fread`, `fflush`, `fopen`, `fclose`, `feof`, `ferror`, `clearerr` | 12 functions, all fit `_call_lib_func` |
| (C) Strings | `strcmp`, `strncmp`, `strchr`, `strrchr`, `strstr`, `strncpy`, `strerror` | 7 functions; `strerror` documented as not thread-safe |
| (C) Memory | `memcpy`, `memmove`, `memset`, `memcmp`, `memchr` | 5 functions |
| (C) Env | `getenv` | 1 function |

### Out of scope (deferred, with rationale)

| Capability | Why deferred |
|---|---|
| Variadic (`printf`, `fprintf`, `sprintf`, `snprintf`, `scanf`, `sscanf`, `fscanf`, `vsnprintf`) | Requires a new variadic-aware intrinsic with per-platform ABI rules (SysV AL = vector-arg count; AAPCS64 darwin vs aarch64-linux differ; Win64 home space). Its own design cycle. |
| `fseek`, `ftell`, `fsetpos`, `fgetpos` | C `long` is 64-bit on POSIX (LP64), 32-bit on Windows x64 (LLP64). A uniform `int64` signature would silently corrupt registers on Windows. Needs option-(ii)-style dispatch to POSIX `fseeko`/`ftello` vs Windows `_fseeki64`/`_ftelli64`. |
| `setenv`, `unsetenv` | POSIX-only. Windows uses `_putenv_s` with different signature. Same dispatch shape as `strerror_safe`. |
| `malloc`, `free`, `calloc`, `realloc` | Mixing raw `malloc`'d memory with numba's NRT introduces ownership-composition gaps. Add when a specific consumer needs it. |
| `time`, `clock`, `gmtime`, `localtime`, `mktime`, `strftime` | `time_t` size historically variable. `numbox.utils.clock.monotonic_ns` covers time needs; `strftime`/`strptime` are variadic-adjacent. |
| `setvbuf`, `setbuf` | Mode macros `_IOFBF` / `_IOLBF` / `_IONBF` have different numeric values on Windows vs POSIX, requiring numbox-owned platform-aware constants. `fflush` after every write is the documented workaround for 0.5.12. |
| `qsort`, `bsearch`, `atexit`, `signal` | Callback args (`@cfunc` function pointers) — expressible as `intp` today but no documented pattern or test in numbox. Separate exercise. |

## 3. Architecture

### 3.1 Symbol resolution: extern refs, not literal addresses

The single most important constraint. Reiterated in the new "Bindings: implementation gotchas" section of `CLAUDE.md`:

- [`ll.address_of_symbol(name)`](../../numbox/core/bindings/call.py#L76) returns the *current process's* runtime address. ASLR randomizes per-process; cached `@njit` objects are meant to survive across runs and machines. Baking that int into LLVM IR breaks `cache=True`.
- The correct pattern, used by `_call_lib_func` at [`call.py:185`](../../numbox/core/bindings/call.py#L185): `get_or_insert_function(builder.module, func_ll_ty, func_name)` emits an extern declaration by name; llvmlite's JIT linker resolves at link time, per process, ASLR-safe.
- The literal-address check at `call.py:76` is *only* a presence assertion; `func_p_as_int` is never consumed by codegen.

The same extern-ref pattern works for **data symbols** (`@stdout = external global ptr`) and for **accessor functions whose return value is per-thread** (`__errno_location`, `__error`, `_errno`).

All three new intrinsics in this work use this pattern. No literal addresses anywhere in IR.

### 3.2 File layout

Three new files, each owning one intrinsic + its `@cres` wrappers:

```
numbox/core/bindings/
├── _stdio.py        [NEW]  _stdio_handle intrinsic + stdout(), stderr(), stdin()
├── _errno.py        [NEW]  _errno_ptr intrinsic + errno_get(), errno_set(v)
├── _strerror.py     [NEW]  _strerror_safe intrinsic + strerror_safe(errnum, buf, buflen)
├── _c.py            [EXTENDED]  +25 cres wrappers (Section 4.4 batch) + module docstring with caller idioms
├── signatures.py    [EXTENDED]  +25 entries in signatures_c, partitioned by section comments
├── __init__.py      [EXTENDED]  +3 lines re-exporting _stdio, _errno, _strerror
├── call.py          [unchanged]
├── abi.py           [unchanged]
└── utils.py         [unchanged]
```

Rationale: each intrinsic binds one conceptual surface; small files (~30–60 lines each) keep Sphinx automodule blocks granular and review tractable.

### 3.3 Platform dispatch — where it lives

Inside each intrinsic's lowering function. Never at Python import time (would bake host-specific assumptions into the cres wrapper's compiled output).

**Stdio handles** (`_stdio_handle`):

| Platform | LLVM IR shape |
|---|---|
| Linux | `@stdout = external global ptr`; emit `load` + `ptrtoint` |
| macOS | `@__stdoutp = external global ptr`; same |
| Windows | extern decl `__acrt_iob_func(i32) -> ptr`; emit `call __acrt_iob_func(0/1/2)` + `ptrtoint` |

**errno** (`_errno_ptr`):

| Platform | Accessor symbol |
|---|---|
| glibc / musl | `__errno_location` |
| macOS | `__error` |
| Windows | `_errno` |

All three are declared `__attribute__((const))` (or effectively-const on Windows) — LLVM may CSE within a function. Correct because errno is per-thread; one function's calls all see the same thread's errno.

**strerror_safe** (`_strerror_safe`):

| Platform | Symbol | C signature | Arg reorder |
|---|---|---|---|
| glibc | `__xpg_strerror_r` ✅ verified via `nm` | `int(int, char*, size_t)` | no |
| musl | `strerror_r` (per musl docs) | `int(int, char*, size_t)` | no |
| macOS | `strerror_r` (per Apple's `strerror_r(3)`) | `int(int, char*, size_t)` | no |
| Windows | `strerror_s` (per MS docs) | `errno_t(char*, size_t, int)` | **YES** — buf, buflen, errnum |

glibc-vs-musl detection at lowering time: probe `ll.address_of_symbol("__xpg_strerror_r")`; non-None → glibc; None → fall back to `strerror_r` (musl shape).

**Cache portability across distinct libcs — what's tested and what's assumed.** The IR-inspection test in §5 verifies *our half*: when the probe returns None, the emitted LLVM IR contains the `strerror_r` symbol; when non-None, it contains `__xpg_strerror_r`. What it does *not* directly test: that numba's compilation-cache key incorporates IR (or some hash that changes when our symbol choice changes), so that moving a cache across libc implementations triggers recompile rather than a stale-symbol JIT-link failure. That property is a behavior of numba itself, not of numbox; we rely on it as a working assumption. 0.5.12 does not directly exercise cross-libc cache portability (no musl runner — see §6.1 for why). If a downstream consumer reports cache-portability issues across libcs, that's the moment to invest in a musl-numba runner.

## 4. Per-chunk design

### 4.1 Stdio handles (chunk 1)

**Public API.** Three `@cres(intp(), cache=True)`-wrapped functions returning the current process's `FILE*` as `intp`:

```python
@cres(intp(), cache=True)
def stdout(): return _stdio_handle("stdout")

@cres(intp(), cache=True)
def stderr(): return _stdio_handle("stderr")

@cres(intp(), cache=True)
def stdin():  return _stdio_handle("stdin")
```

**Intrinsic.** `_stdio_handle(name_literal_ty)` accepts a literal string `"stdout"` / `"stderr"` / `"stdin"`, branches on `platform.system()` at lowering, emits the platform-correct IR per §3.3.

**Why callable (not constants).** Module-load-time resolution + cache-baked literals would break under ASLR. See §3.1.

**Cross-references.** Used by §4.4 (monomorphic stdio) tests and caller idioms (`fputs("...", stderr())`).

### 4.2 errno (chunk 2)

**Public API.**

```python
@cres(int32(), cache=True)
def errno_get(): ...

@cres(types.void(int32), cache=True)
def errno_set(v): ...
```

**Intrinsic.** `_errno_ptr` emits extern decl of `__errno_location` / `__error` / `_errno`, calls it (no args, returns `int*`), and the wrappers `errno_get`/`errno_set` do the `load i32` / `store i32` at that pointer.

**Per-thread correctness.** Accessor is called at runtime on whatever OS thread is executing — never cached across threads. Numba's `prange` worker pool puts each iteration on some worker; the accessor returns that worker's errno, naturally correct.

**LLVM hoisting is fine within a function.** The accessor is effectively `__attribute__((const))` — LLVM may CSE multiple calls within one function on one thread. Correct because errno belongs to that thread for the function's duration.

**Python-side observation caveat (documented in docstring).** errno set on a `@njit(parallel=True)` worker is not readable from the Python caller after return — different OS thread. Collect any errno-derived state into a return value before exiting the parallel region.

**Out of scope.** errno-code constants (`ENOENT`, `EACCES`, etc.): Python's `errno` module already exposes them platform-correctly. Callers compare `errno_get()` to `errno.ENOENT` Python-side or close over the int as an `@njit` constant. Shipping a numbox table would duplicate Python's surface and add a maintenance liability.

### 4.3 strerror_safe (chunk 3)

**Public API.**

```python
@cres(int32(int32, intp, intp), cache=True)
def strerror_safe(errnum, buf, buflen): ...
```

POSIX-shaped: returns `0` on success, positive errno (`ERANGE` on short buffer, `EINVAL` on unknown errnum). Writes NUL-terminated message into `buf`.

**Intrinsic.** `_strerror_safe` does the platform dispatch table from §3.3 at lowering, emits the extern decl with the C-correct signature (note: Windows reorders the args at the *call site*, not in the C declaration), and the call.

**Caller idiom** (uses existing primitives from `numbox/utils/lowlevel.py`):

```python
import numpy as np
from numbox.core.bindings import strerror_safe, errno_get
from numbox.utils.lowlevel import array_data_p, get_str_from_p_as_int

@njit
def describe():
    buf = np.zeros(128, dtype=np.uint8)
    buf_p = array_data_p(buf)
    e = errno_get()
    if strerror_safe(e, buf_p, buf.size) != 0:
        return ""
    return get_str_from_p_as_int(buf_p)
```

**Alpine shell-only CI job** (§6) validates the musl symbol-layout assumption: `__xpg_strerror_r` absent, `strerror_r` present.

**Monkeypatched IR-inspection test on glibc CI** (§5) validates the selection logic: patch `ll.address_of_symbol` to return None for `__xpg_strerror_r`, invoke the intrinsic, assert the emitted LLVM IR contains a call to `strerror_r` (the fallback name), not `__xpg_strerror_r`. Direct IR inspection, no end-to-end call — explicitly avoiding the ABI-meaning trap (on glibc, `strerror_r` is the GNU form, not the POSIX form; calling it under POSIX-shaped IR would corrupt return-value reads).

### 4.4 Monomorphic libc batch (chunks 4–7)

All fit `_call_lib_func` directly. No new intrinsics. Signature dict additions + `@cres` wrappers in `_c.py`.

**Stdio non-variadic — 12 functions (chunk 4):**

```
puts(const char*) -> int
fputs(const char*, FILE*) -> int
fputc(int, FILE*) -> int
putchar(int) -> int
fwrite(const void*, size_t, size_t, FILE*) -> size_t
fread(void*, size_t, size_t, FILE*) -> size_t
fflush(FILE*) -> int
fopen(const char *path, const char *mode) -> FILE*
fclose(FILE*) -> int
feof(FILE*) -> int
ferror(FILE*) -> int
clearerr(FILE*) -> void
```

Sizes: `size_t` is 64-bit on all current 64-bit CI platforms — signed `intp` is safe. `int` is 32-bit everywhere we care about — `int32`.

**Strings — 7 functions (chunk 5):**

```
strcmp(const char*, const char*) -> int
strncmp(const char*, const char*, size_t) -> int
strchr(const char*, int) -> char*
strrchr(const char*, int) -> char*
strstr(const char*, const char*) -> char*
strncpy(char*, const char*, size_t) -> char*
strerror(int) -> char*    [docstring: not thread-safe; use strerror_safe for thread-safe]
```

**Memory — 5 functions (chunk 6):**

```
memcpy(void*, const void*, size_t) -> void*
memmove(void*, const void*, size_t) -> void*
memset(void*, int, size_t) -> void*
memcmp(const void*, const void*, size_t) -> int
memchr(const void*, int, size_t) -> void*
```

**Env — 1 function (chunk 7):**

```
getenv(const char*) -> char*
```

**`signatures_c` partitioning.** Single dict (one lookup site in `_call_lib_func`), but visual structure via section-comment dividers:

```python
signatures_c = {
    # === existing (rand/srand/strlen/lldiv) ===
    "rand": int32(),
    ...
    # === stdio (non-variadic) ===
    "puts": int32(intp),
    ...
    # === strings ===
    "strcmp": int32(intp, intp),
    ...
    # === memory ===
    "memcpy": intp(intp, intp, intp),
    ...
    # === env ===
    "getenv": intp(intp),
}
```

**Caller idioms (go in `_c.py` module docstring):**

```python
# Log to stderr from @njit
@njit
def log_to_stderr(msg):
    p = get_unicode_data_p(msg)
    fputs(p, stderr())
    fflush(stderr())

# Append to a file from @njit
@njit
def append_to_file(path, msg):
    path_p = get_unicode_data_p(path)
    mode_p = get_unicode_data_p("ab")
    fp = fopen(path_p, mode_p)
    if fp == 0:
        return -1
    msg_p = get_unicode_data_p(msg)
    n = fwrite(msg_p, 1, strlen(msg_p), fp)
    fclose(fp)
    return n

# Buffer compare via numpy (assumes both arrays contiguous)
@njit
def buffer_equal(a, b):
    if a.nbytes != b.nbytes:
        return False
    return memcmp(array_data_p(a), array_data_p(b), a.nbytes) == 0
```

## 5. Tests

### 5.1 New test files

| File | Scope |
|---|---|
| `test/core/test_stdio_handles.py` | `stdout()`, `stderr()`, `stdin()` non-zero; round-trip `fputs("x\n", stderr()); fflush(stderr())` captured via `capsys` |
| `test/core/test_errno.py` | Round-trip set/get; real failure via `fopen` (depends on chunk 4 ordering); two-Python-thread no-contamination test; `@njit(parallel=True)` prange iteration-correctness test |
| `test/core/test_strerror_safe.py` | Round-trip for `ENOENT`; short-buffer behavior; concurrent-threads test (two threads writing to distinct buffers); **monkeypatched symbol-probe IR-inspection test** (force fallback path on glibc, inspect emitted IR for `strerror_r` symbol name) |

### 5.2 Extended `test/core/test_bindings.py`

Adds five new test functions (in addition to keeping existing `test_c` / `test_sqlite` unchanged):

- `test_c_stdio` — write/read roundtrip to `tmp_path` via fopen/fwrite/fclose/fopen/fread/fclose; assert byte-equality.
- `test_c_strings` — strcmp/strncmp/strchr/strrchr/strstr/strncpy on known inputs, pointer args via `get_unicode_data_p` / `array_data_p`.
- `test_c_memory` — memcpy correctness, memmove overlapping-region correctness, memset known byte fill, memcmp sign-of-return, memchr offset finding.
- `test_c_env` — `getenv("PATH")` nonzero; `getenv("NUMBOX_NONEXISTENT_XYZZY")` returns 0.
- `test_c_strerror` — `strerror(errno.ENOENT)` nonzero pointer; `get_str_from_p_as_int` produces non-empty result. Threading test deferred to `test_strerror_safe.py`.

### 5.3 IR-inspection test for strerror_safe symbol probe

The most delicate test. Approach: bypass the cres wrapper, invoke the intrinsic's codegen directly with a monkeypatched `ll.address_of_symbol`, capture the emitted LLVM IR, assert the chosen symbol name is correct.

The "force end-to-end call with monkeypatched probe" alternative is structurally broken on glibc: the symbol `strerror_r` on glibc IS the GNU form (returns `char*`), not the POSIX form (returns `int`). Calling glibc's GNU `strerror_r` under our POSIX-shaped LLVM IR reads the wrong-typed return register and may not even write to the caller's buffer. We can't simulate musl semantics on glibc, only musl symbol-selection logic.

What the Alpine shell-only job (§6) covers separately: the assumption that musl exposes `strerror_r` (POSIX form) and not `__xpg_strerror_r`. If musl ever adds `__xpg_strerror_r`, the Alpine job fails and we know to revisit.

## 6. CI changes

### 6.1 Alpine shell-only job (new entry in `numbox_ci.yml`)

```yaml
- name: musl symbol layout
  if: matrix.alpine
  shell: sh
  run: |
    apk add --no-cache binutils
    set -e
    test -e /lib/ld-musl-x86_64.so.1 || (echo "no musl on this image"; exit 1)
    LIBC=/lib/ld-musl-x86_64.so.1
    nm -D "$LIBC" | grep -E '^[0-9a-f]+ T strerror_r$' || (echo "musl strerror_r missing"; exit 1)
    if nm -D "$LIBC" | grep -q '__xpg_strerror_r'; then
      echo "musl now exports __xpg_strerror_r — revisit strerror_safe probe"; exit 1
    fi
    echo "musl libc has expected layout"
```

(Exact YAML structure to be finalized during chunk 3 — likely a new matrix `include` entry with a container, run only the verification step.)

Reasoning: numba does not ship musl wheels, so a full pytest on Alpine would require a multi-minute source build per CI run. The shell-only check validates the *underlying assumption* of the symbol probe at near-zero cost, paired with the IR-inspection test that runs on every other CI matrix entry.

### 6.2 paths-ignore: no change

`numbox_ci.yml` already has `paths-ignore: ['**.md', 'docs/**']`. This work modifies code (`numbox/**`) and `.rst` docs. Code changes trigger CI; later docs-only follow-ups skip CI — consistent with current behavior.

## 7. Docs changes

Extend `docs/numbox.core.bindings.rst` (existing file). No new RST.

### 7.1 New narrative section

Title: **"Stdio handles, errno, and thread-safe strerror"**. Inserted between existing "ABI dispatch" and "Modules". Covers:

- Why stdio handles are callable functions, not constants (extern-ref pattern; ASLR + cache=True).
- Why errno access is per-thread (accessor must be called at use site, not cached).
- The `strerror_safe` platform mapping table.
- Caller idioms (`log_to_stderr`, `append_to_file`, `describe_errno`, `buffer_equal`).

### 7.2 New automodule blocks

Three additions mirroring existing pattern:

```rst
numbox.core.bindings._stdio
---------------------------

.. automodule:: numbox.core.bindings._stdio
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._errno
---------------------------

.. automodule:: numbox.core.bindings._errno
   :members:
   :show-inheritance:
   :undoc-members:

numbox.core.bindings._strerror
------------------------------

.. automodule:: numbox.core.bindings._strerror
   :members:
   :show-inheritance:
   :undoc-members:
```

## 8. Implementation discipline: TDD

Each chunk in §9 follows red → green → refactor:

1. Write the test(s) for the chunk's surface. Run them. Confirm failure (red).
2. Write the minimal implementation that makes them pass. Run them. Confirm pass (green).
3. Refactor for clarity if useful, with tests still passing.

Chunks are sized so each red→green cycle is bounded to ~30–90 minutes of focused work.

Apply the `superpowers-extended-cc:test-driven-development` skill at chunk start. Skipping at the implementer's peril — the per-thread errno tests and the IR-inspection test for strerror_safe are exactly the kinds of regressions that "I'll add tests after" leaves uncovered.

## 9. Implementation order — eight chunks

Each chunk = one commit (or a small contiguous run of red→green→refactor commits, squashed at the end).

| # | Chunk | Why this order |
|---|---|---|
| 1 | **Stdio handles** (`stdout`, `stderr`, `stdin`) | Establishes new-intrinsic file pattern (`_stdio.py`); exercises extern-global LLVM ref. Foundational and small. |
| 2 | **errno** (`errno_get`, `errno_set`) | Same extern-ref pattern but for an accessor-returned pointer; introduces per-thread semantics. Independent of #1. |
| 3 | **strerror_safe** + monkeypatched IR-inspection test + Alpine shell-only CI job | Builds on errno conceptually (paired error-message API). Introduces platform-dispatch-inside-intrinsic. |
| 4 | **Monomorphic stdio** (12 functions) | First chunk to exercise `stderr()` end-to-end via `fputs`. Cross-references chunk 1. |
| 5 | **Monomorphic strings** (7 functions, including `strerror` with caveat) | Independent. |
| 6 | **Monomorphic memory** (5 functions) | Independent. |
| 7 | **Monomorphic env** (`getenv`) | Trivial; one entry. |
| 8 | **Sphinx docs narrative section + automodule blocks for `_stdio` / `_errno` / `_strerror`** | Last because wrapper docstrings must be final; narrative synthesizes the whole surface. |

Why 1 → 2 → 3 → 4 (not 4 → 1 → 2 → 3): chunk 4's tests want to use `stderr()` and `errno_get()` (e.g. asserting `fopen` of a nonexistent path sets ENOENT). Ordering 1–3 first means chunk 4's tests reference real, working primitives, not stubs.

Each chunk produces its own commit on the feature branch. Eight chunks → eight commits, each red→green→refactor-clean. Makes review tractable and bisect-friendly if regression surfaces.

## 10. Branch and PR plan

### 10.1 Pre-implementation state (as of 2026-05-11)

- `main` has the 0.5.11 status update (merged from PR #16's first commit).
- PR #16 (`docs/claude-md-0.5.11-status` branch) has the 0.5.11 status update plus the new "Bindings: implementation gotchas" section in CLAUDE.md. Awaiting merge.
- This spec doc lives on a new branch `feature/libc-bindings-expansion`, based off the post-PR-#16 `main`.

### 10.2 Implementation flow

1. Implementation proceeds on `feature/libc-bindings-expansion` after PR #16 merges (so the gotchas guide each chunk).
2. 8 chunks as 8 commits per §9.
3. Open **fork PR** to `nelson2005/numbox:main` first. Wait for fork CI green (full matrix: ubuntu / ubuntu-arm / windows / macOS + the new Alpine shell-only job) and fork bot reviews (minimax / Copilot / gemini).
4. After fork PR approval and merge, cherry-pick to a clean `upstream-pr/libc-bindings-expansion` branch based on `upstream/main`. Exclude from upstream:
   - `CLAUDE.md`
   - `docs/plans/**` (this spec doc, the tasks.json)
   - The Alpine shell-only CI matrix entry (fork-only)
5. Open upstream PR to `Goykhman/numbox:main`. Address review.
6. After upstream merge, sync into fork as a separate small PR (pattern matches PRs #9, #15).
7. Tag **0.5.12** on the upstream merge commit.

### 10.3 Project Status entry (for CLAUDE.md after release)

```markdown
- **libc bindings expansion** — merged YYYY-MM-DD via fork [#XX](...) / upstream [#YY](...) at [`COMMIT`](...); tagged [`0.5.12`](...). Adds:
  - Stdio handles: `stdout()`, `stderr()`, `stdin()` callable from `@njit`.
  - Thread-safe errno: `errno_get`, `errno_set`, `strerror_safe`.
  - Monomorphic libc batch: 12 stdio + 7 string + 5 memory + 1 env functions.
  - Alpine shell-only CI job verifying musl symbol-layout assumption.
```

## 11. Pre-flight checks (implementer)

Before chunk 1:

1. Read [`numbox/utils/lowlevel.py`](../../numbox/utils/lowlevel.py) end-to-end. The libc bindings compose `array_data_p`, `get_str_from_p_as_int`, `get_unicode_data_p` extensively.
2. Read [`numbox/core/bindings/call.py`](../../numbox/core/bindings/call.py) — specifically the `_call_lib_func` codegen and the `get_or_insert_function` extern-ref pattern at line 185.
3. Run baseline `venv/bin/pytest --durations=20` on `main` to confirm a clean starting point.
4. Verify `venv/bin/python -c "import numba; print(numba.__version__)"` matches the pinned local version (0.60.0 per `pyproject.toml` local default).
5. Clean numba cache + `__pycache__` before the first test run:
   ```bash
   venv/bin/python -c "import shutil, pathlib; shutil.rmtree(pathlib.Path('~/.cache/numba').expanduser(), ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
   ```

## 12. Open follow-ups

Deferred from 0.5.12 with rationale (already enumerated in §2 non-scope). Each will get its own focused design when prioritized:

- **Variadic intrinsic** (`printf`, `fprintf`, `sprintf`, `snprintf`, `scanf`, `sscanf`, `fscanf`, `vsnprintf`). Likely 0.5.13.
- **File seeking** (`fseek`, `ftell`, `fsetpos`, `fgetpos` — POSIX `fseeko`/`ftello` vs Windows `_fseeki64`/`_ftelli64`).
- **Environment write** (`setenv`/`unsetenv` POSIX vs `_putenv_s` Windows).
- **Stream buffering** (`setvbuf`/`setbuf` + numbox-owned `IOFBF`/`IOLBF`/`IONBF` constants).
- **Time** (`time`, `clock`, `gmtime`, `localtime`, `mktime`, `strftime`/`strptime`).
- **Memory allocation** (`malloc`, `free`, `calloc`, `realloc` — when a consumer with clear ownership semantics needs them).
- **Callback args** (`qsort`, `bsearch`, `atexit`, `signal` — documented pattern + tests for `@cfunc`-as-arg).
- **`Record` LARGE returns in `_call_lib_func`** (already an existing open follow-up from 0.5.11; orthogonal but adjacent).
