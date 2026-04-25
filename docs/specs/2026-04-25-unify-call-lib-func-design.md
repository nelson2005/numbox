# Unify `_call_lib_func` — design (2026-04-25)

## Motivation

[`numbox/core/bindings/abi.py`](../../numbox/core/bindings/abi.py) currently provides three intrinsics that callers must pick between based on what the C function does to its arguments and return value:

- [`_call_lib_func_struct_in`](../../numbox/core/bindings/abi.py#L91) — pass a ≤16-byte struct (by value on SysV x86-64 / AAPCS64, by pointer on Windows x64). Raises `TypingError` for >16-byte structs.
- [`_call_lib_func_struct_out`](../../numbox/core/bindings/abi.py#L126) — return a ≤16-byte struct (direct on SysV / AAPCS64, via `sret` on Windows x64). Raises `TypingError` for >16-byte structs.
- [`_call_lib_func_args_struct_out`](../../numbox/core/bindings/abi.py#L168) — same return-side gating as `_struct_out`, with multiple scalar args.

There is no helper for the >16-byte struct-IN case. Numbduck has three call sites that do this by hand — `_duckdb_create_decimal`, `_duckdb_create_varint`, `_duckdb_bind_decimal` — each with its own custom `@intrinsic` that mirrors `_emit_byval_call` and adds the SysV-x86-64-only `byval` arg attribute plus `optnone` + `noinline` function attributes ([rationale](https://github.com/numba/llvmlite/issues/300#issuecomment-327235846)). They also carry a local `_is_sysv_x86_64 = not _is_win and platform.machine() in ("x86_64", "AMD64")` flag.

Adding a fourth dedicated intrinsic (`_call_lib_func_byval_large`) closes the gap but doubles down on a structural problem: the caller has to pick the right helper, and the choice is driven by ABI details (struct size, platform) that the type system already knows. This spec instead **unifies** the byval/struct-in/struct-out/args-struct-out paths into the existing primary `_call_lib_func` intrinsic in [`numbox/core/bindings/call.py`](../../numbox/core/bindings/call.py), so the caller writes one thing for any C `func(T)` shape regardless of size or platform.

## Scope

**In:**
- Extend `_call_lib_func(func_name, args_tuple)` to be ABI-aware: per-arg classification (scalar vs struct ≤16B vs struct >16B), per-platform dispatch (SysV x86-64 vs Windows x64 vs AAPCS64), and the SysV >16B `byval` + `optnone` + `noinline` idiom.
- Extend the same intrinsic's return-side handling: scalar returns direct, ≤16B struct returns direct on register-passing ABIs and via `sret` on Windows x64.
- Delete `_call_lib_func_struct_in`, `_call_lib_func_struct_out`, `_call_lib_func_args_struct_out`. No deprecation shims.
- Update [`test/core/test_abi.py`](../../test/core/test_abi.py) to reflect the deletions and exercise the unified path.

**Out:**
- `_call_lib_func_byval(func_name, struct)` — kept as-is. Different semantic: the C signature is literally `func(T*)` (e.g. [`duckdb_fetch_chunk(duckdb_result *)`](https://github.com/Goykhman/numbduck/blob/feat/use-numbox-generics/numbduck/ducklib.py#L1074)), not `func(T)` lowered to a byval pointer by the ABI. The numba type system can't disambiguate `T` from `T*` without a precise signature dict, and that refactor is heavier than the unification benefit justifies.
- Numbduck migration. Once this lands and is cut into a release, numbduck's three custom intrinsics can be dropped and the local `_is_sysv_x86_64` flag retired, but that is a numbduck PR, not part of this spec.
- Struct returns >16 bytes. No current consumer in numbox or numbduck. The unified `_call_lib_func` raises `TypingError` if encountered.

## API change

**Before** (current shape, requires caller to pick):

```python
from numbox.core.bindings.abi import (
    _call_lib_func_struct_in, _call_lib_func_struct_out,
    _call_lib_func_args_struct_out,
)

# C: int func(T)  with sizeof(T) <= 16
result = _call_lib_func_struct_in("create_thing", thing_value)

# C: T func(int, int)
thing = _call_lib_func_args_struct_out("make_thing", (a, b))
```

**After** (single intrinsic, no caller-side ABI choice):

```python
from numbox.core.bindings.call import _call_lib_func

# C: int func(T)        — any size, any platform
result = _call_lib_func("create_thing", (thing_value,))

# C: T func(int, int)   — return ABI handled
thing = _call_lib_func("make_thing", (a, b))

# C: int func(int, int, T) with sizeof(T) > 16
result = _call_lib_func("bind_thing", (handle, idx, big_thing))
```

The caller picks `_call_lib_func` for any C signature of form `func(T)` (return + args by value at the C level). They pick `_call_lib_func_byval` only for C signatures of form `func(T*)` — the choice is driven by what `duckdb.h` (or whatever C header) actually declares, not by ABI internals.

## ABI dispatch table

Per arg, classified by numba type and platform:

| numba type | sizeof | SysV x86-64 | Windows x64 | AAPCS64 |
|---|---|---|---|---|
| scalar (any non-struct) | n/a | direct | direct | direct |
| `Record` / `BaseTuple` | ≤ 16 | by value | alloca+store+pointer | by value |
| `Record` / `BaseTuple` | > 16 | alloca+store+pointer with `byval` attr; enclosing function gets `optnone` + `noinline` | alloca+store+pointer | alloca+store+pointer |

Per return type, classified by `signatures[func_name].return_type`:

| numba type | sizeof | SysV x86-64 | Windows x64 | AAPCS64 |
|---|---|---|---|---|
| scalar | n/a | direct | direct | direct |
| `Record` / `BaseTuple` | ≤ 16 | direct | `sret` | direct |
| `Record` / `BaseTuple` | > 16 | `TypingError` | `TypingError` | `TypingError` |

`BaseTuple` is the numba abstract base for `Tuple`, `UniTuple`, and `NamedTuple`; all three share `.types` and are size-computable via the existing [`_struct_bytes`](../../numbox/core/bindings/abi.py#L39) helper.

The SysV >16B path is the new one. Everything else is the existing `_struct_in` / `_struct_out` / `_args_struct_out` logic merged into one decision table.

## Module organization

`abi.py` keeps:
- [`_struct_bytes(ty, fn_name)`](../../numbox/core/bindings/abi.py#L39) — struct-shape size helper. Used by `_call_lib_func` and any future custom intrinsics.
- [`_emit_byval_call(builder, context, arg, arg_ll_ty, ret_type, func_name)`](../../numbox/core/bindings/abi.py#L59) — codegen helper for "alloca+store+call-via-pointer". Used by `_call_lib_func` and by `_call_lib_func_byval`.
- [`_call_lib_func_byval(func_name, struct_arg)`](../../numbox/core/bindings/abi.py#L68) — kept; different C semantic.
- A small private platform classifier — `_PLATFORM_SYSV_X86_64`, `_PLATFORM_WIN`, `_PLATFORM_AAPCS64` constants and a `_current_platform()` function. Replaces the bare `_is_win` flag, which was insufficient (it conflated AAPCS64 with SysV x86-64). Used internally by the dispatch logic.

`abi.py` loses:
- `_call_lib_func_struct_in`
- `_call_lib_func_struct_out`
- `_call_lib_func_args_struct_out`

`call.py` is where `_call_lib_func` lives and grows. It imports `_emit_byval_call`, `_struct_bytes`, and the platform classifier from `abi.py`. The dispatch decision tree is implemented in `call.py`'s `codegen` closure; ABI primitives stay in `abi.py`.

## Tests

[`test/core/test_abi.py`](../../test/core/test_abi.py):

**Kept:**
- `test_struct_bytes_supports_all_struct_types` — verbatim.
- `test_struct_bytes_rejects_non_struct_type` — verbatim.

**Updated:**
- `test_abi_imports` — assert the three retired names are gone (`hasattr(abi, "_call_lib_func_struct_in") is False`, etc.) and the survivors (`_emit_byval_call`, `_call_lib_func_byval`, `_struct_bytes`, the platform classifier) remain.
- `test_call_lib_func_args_struct_out_lldiv` — renamed to `test_call_lib_func_lldiv_via_unified` and rewritten to call `_call_lib_func("lldiv", (10, 3))` directly. Same `lldiv` libc function, same end-to-end value validation (`quot == 3`, `rem == 1`), but exercises the return-side ABI through the unified intrinsic.

**Added:**
- `test_call_lib_func_scalar_args_unchanged` — `_call_lib_func("cos", (0.0,))` returns `1.0`. Regression guard for the existing math/c/sqlite callers (current behavior).
- `test_call_lib_func_byval_attribute_in_ir_for_large_struct` — IR-inspection. Build a 24-byte `Record` arg, compile a JIT function that calls `_call_lib_func("dummy_24b_in", (rec,))` against a synthetic signature, capture the LLVM IR via numba's `inspect_llvm()`, assert: `byval` attribute present on the struct arg, `optnone` and `noinline` on the enclosing function. Skipped on Windows and AAPCS64 (those platforms do not add the attributes).
- `test_call_lib_func_no_byval_attribute_for_small_struct` — symmetric IR check that ≤16B structs do NOT carry `byval` on SysV x86-64. Skipped on Windows (different path).

End-to-end >16B by-value remains covered by numbduck's `test/test_ducklib.py` integration tests, which exercise real DuckDB C-API entry points (`duckdb_create_decimal` etc.). Numbox itself doesn't ship a >16B-by-value test library.

## Branch and commit plan

**Branch:** `feat/unify-call-lib-func` off `origin/main` (already created — has CLAUDE.md + the fork CI matrix from PR #7).

**Commits:**

1. **Extend `_call_lib_func` to be ABI-aware.** Touches `call.py` only. Adds the per-arg classification, platform dispatch, byval+optnone gate on SysV >16B, sret return path on Windows x64. The three legacy intrinsics in `abi.py` still exist and still work; nothing else changes. Ships green.
2. **Delete redundant ABI intrinsics.** Removes `_call_lib_func_struct_in`, `_call_lib_func_struct_out`, `_call_lib_func_args_struct_out` from `abi.py`. Updates `test_abi.py` import-assertion test and renames/rewrites the lldiv test to use `_call_lib_func`. Ships green only after commit (1).
3. **Add ABI dispatch tests.** The three new tests (scalar args regression, byval-on-large IR check, no-byval-on-small IR check). Ships green.

Splitting (1) from (2) keeps each commit small and bisect-friendly: a regression in unification can be isolated from a regression in the deletion / rename.

After this lands and gets cut into a numbox release, a follow-up numbduck PR retires the local `_is_sysv_x86_64` flag and migrates the three byval+optnone call sites to the unified `_call_lib_func`. That follow-up is tracked in numbduck's CLAUDE.md, not in this spec.

## Open questions

None. Design is fully specified.
