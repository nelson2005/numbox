# numbox deep review — 2026-06-20

- **Date:** 2026-06-20
- **Branch:** `review/numbox-2026-06-20`
- **Commit under review:** `ece98ce`
- **Review units:** 119 targets × dimension lenses → 119 reviews + 119 verifies (119/119 each)
- **Findings:** 124 raised → **118 confirmed**, **1 uncertain**, **5 refuted** (verification culled the false positives)

## Tally (confirmed)

| Severity | Count |
|----------|-------|
| critical | 0 |
| high | 4 |
| medium | 29 |
| low | 85 |

| Dimension | Confirmed |
|-----------|-----------|
| MEM (memory / ABI / refcount) | 6 |
| COR (correctness) | 28 |
| SEC (security) | 3 |
| DES (design / simplification) | 55 |
| TST (tests & docs) | 26 |

The four `high` findings are three distinct root causes (two surface under both MEM and COR): the
signed-char C-string reader, the tuple-alignment ABI model, and the swallowed xFilter exception.

---

## MEM — Memory / ABI / refcount

### MEM-1 (high) — `get_str_from_p_as_int` reads bytes as signed char; non-ASCII C strings raise

[`numbox/utils/lowlevel.py:249-260`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/lowlevel.py#L249-L260)

The canonical NUL-terminated-C-string reader builds its byte view with `dtype=char`, and numba's
`char` is **signed int8**. Any byte `0x80-0xFF` (every UTF-8 lead/continuation byte, every Latin-1 high
byte) reads back negative, and `chr(<negative>)` raises `ValueError: chr() arg not in range(0x10ffff)`
inside `@njit`. Triggers on **any** C string containing a non-ASCII byte. This is the documented primitive
CLAUDE.md directs callers to use for `sqlite3_errmsg`, file paths, `strerror`, and exec error messages —
all UTF-8, all able to carry non-ASCII (table/column names). A non-ASCII error message turns a normal
C-API read into an njit exception.

- **Evidence:** `mem_view = carray(void_p, shape=(MAX_STR_LENGTH,), dtype=char)` then `s += chr(char_as_code_p)`. Verified this session: `numba.core.types.char` is `int8`, `.signed == True`; feeding the UTF-8 bytes `0xC3 0xA9 0x41 0x00` raised the `ValueError`. The only tests use pure-ASCII, so the path is untested.
- **Fix:** view the buffer through an **unsigned** 8-bit dtype (`uint8`) so each element is `0..255` before `chr()`. Note even unsigned, byte-at-a-time `chr` over UTF-8 yields per-byte Latin-1 codepoints, not the decoded string; if faithful UTF-8 is intended, collect the bytes and `.decode("utf-8")`. At minimum fix the signedness so high bytes don't raise.
- **Confidence:** high. **Findings:** MEM-lowlevel-1, COR-lowlevel-1.

### MEM-2 (high) — BaseTuple ABI size/offset model ignores alignment padding

[`numbox/core/bindings/abi.py:95-96, 120-126`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/abi.py#L95-L96)

For a `BaseTuple`, `_struct_bytes` returns `sum(t.bitwidth)//8` and `_iter_struct_fields` lays fields
back-to-back with no inter-field padding. numba lowers tuples through `StructModel` to a **non-packed**
`ir.LiteralStructType`, which LLVM lays out with natural alignment (padding inserted), and `call.py`'s
codegen passes exactly that padded value type. So the classifier's size/offsets diverge from the layout
actually used at the call boundary:

- `(int32, double, int32)` bit-packs to 16 → classified `STRUCT_SMALL` (by-value register path), but the
  real `{i32, double, i32}` aligns the double to offset 8 and totals **24 bytes** → should be `STRUCT_LARGE`
  (by pointer/sret). Passing a 24-byte aggregate through the ≤16-byte register-coercion path on SysV
  x86-64 / AAPCS64 is an ABI mismatch: silent wrong result or stack corruption.
- Even within genuinely-16-byte tuples, the padding error shifts which eightbyte a float occupies, so the
  INTEGER/SSE eightbyte classification (hence the INT/INT repack decision) can be wrong — e.g.
  `(int32, double)` is modelled as both-SSE, but `{i32, double}` puts the double at offset 8 → (INTEGER, SSE).

- **Evidence:** [`abi.py:96`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/abi.py#L96) bit-packs; the Record branch by contrast uses real `ty.size`/`fld.offset`. [`call.py:131/145`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/call.py#L131) use `context.get_value_type(...)` (the padded `LiteralStructType`) for the alloca/store/coercion. [`test/core/test_abi.py:226-227`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/test/core/test_abi.py#L226-L227) even pins `_struct_bytes(NamedTuple([int32, int64])) == 12`, whereas `{i32, i64}` is 16 bytes under natural alignment.
- **Fix:** derive tuple size/offsets from the actual data-model layout (query the LLVM struct element offsets / ABI size from `context.get_value_type(ty)`) or replicate natural-alignment layout (align each field up to its size, pad total to max-field alignment) so tuples match Records. Add regressions for `(int32, double, int32)` [LARGE], `(int32, int64)` [size 16 not 12], `(int32, double)` [eightbytes (INTEGER, SSE)].
- **Confidence:** medium. **Findings:** MEM-abi-1, COR-abi-1.

### MEM-3 (high) — User fn exception at the xFilter cfunc boundary becomes a silent empty result

[`numbox/core/bindings/_sqlite_tvf.py:456-463`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_tvf.py#L456-L463)

`_tvf_xfilter` has no try/except and unconditionally returns `SQLITE_OK`. When the user's `fn` (or an NRT
allocation inside `_tvf_xfilter_impl`) raises, numba's `@cfunc` boundary swallows the exception, prints to
stderr, and returns the int32 zero default (== `SQLITE_OK`). Because the impl zeroes the cursor's
`mi_p/data_p/n_rows/row_stride` **before** calling `fn`, the cursor is left empty and `_tvf_xeof` returns
1 immediately. So `SELECT * FROM f(...)` completes "successfully" with zero rows — indistinguishable from
a legitimately-empty TVF. Triggers on any raising `fn` (bad arg, allocation failure, a bug in `fn`) and on
a NOMEM in the njit impl.

- **Evidence:** the cfunc's own comment (lines 458-461) acknowledges the swallow but claims "there is no handle to call `sqlite3_result_error`". True for `sqlite3_result_error`, but xFilter signals failure via its **own return code** — a non-`SQLITE_OK` return aborts the statement with an error; the sibling cfuncs (xConnect:168, xOpen:252, xColumn:347) already return `SQLITE_ERROR` on `except Exception`.
- **Fix:** wrap the impl call in a **bare try/except** (not try/finally — numba 0.65.1 RERAISEs on finally) and return `SQLITE_ERROR` on exception; optionally set the vtab's `zErrMsg` via `sqlite3_mprintf` for a message. This both surfaces the error to SQLite and guarantees the cfunc boundary is never crossed by an in-flight exception, matching the catch-only pattern used by every other TVF cfunc.
- **Confidence:** high. **Findings:** MEM-sqlite_tvf-1, COR-sqlite_tvf-1.

### MEM-4 (medium) — `extract_connection_ptr` returns a borrowed `sqlite3*` with no documented keep-alive

[`numbox/utils/pysqlite_bridge.py:110-164`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/pysqlite_bridge.py#L110-L164)

`extract_connection_ptr` reads the live `db` handle out of the Python `sqlite3.Connection` and returns it
as a plain int. That pointer is owned by `conn`: it is freed on `conn.close()` or GC. The returned int
holds no reference to `conn`, and the docstring's Returns/Raises sections never state the lifetime contract.
A caller that extracts the pointer, drops `conn` (or closes it), then passes the int into the `@njit`
bindings gets a use-after-free / segfault on a freed `sqlite3*` — a silent, hard-to-diagnose crash.

- **Evidence:** `db_ptr = _PysqliteConnection.from_address(id(conn)).db` (156) → `return db_ptr` (164). The Returns block says only "`int` — `sqlite3*` as a Python int"; Raises never mentions lifetime. The tests keep `conn` alive in try/finally for the whole pointer-use window, confirming the undocumented invariant.
- **Fix:** document the keep-alive invariant in the Returns section — the returned `sqlite3*` is **borrowed** from `conn`, valid only while `conn` is alive and open; the caller must retain a reference to `conn` for the entire lifetime of any `@njit` use, and must not use it after `conn.close()`. (Cannot bake a reference into a bare int — mirror the `SQLITE_STATIC`-style ownership note already used for `bind_text`/`bind_blob`.)
- **Confidence:** high. **Findings:** MEM-pysqlite_bridge-1.

### MEM-5 (low) — xBestIndex `bound_mask` uses a 64-bit shift undefined for ≥64 hidden args

[`numbox/core/bindings/_sqlite_tvf.py:192-202`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_tvf.py#L192-L202)

`bound_mask` is uint64 and the all-bound target is `(uint64(1) << uint64(n_hidden)) - 1`. A 64-bit shift
by ≥64 is undefined at LLVM (poison/0). For `n_hidden == 64` the target accidentally still equals
`UINT64_MAX`, but for ≥65 both the per-arg bit and the target shift out of range, so xBestIndex wrongly
rejects/accepts plans. `n_hidden = len(arg_tags)` is user-controlled with no upper-bound guard. Unreachable
in practice (a TVF with ≥64 hidden scalar args is absurd) but latent UB.

- **Fix:** add a guard in `_build_tvf_descriptor`/`register_tvf` rejecting `len(arg_tags) > 63` (or use a count-based all-bound test). **Confidence:** medium. **Findings:** MEM-sqlite_tvf-2, COR-sqlite_tvf-2.

### MEM-6 (low) — `_struct_bytes`/`_iter_struct_fields` raise a raw `AttributeError` on a nested-tuple field

[`numbox/core/bindings/abi.py:96, 122-124`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/abi.py#L96)

The BaseTuple branches dereference `.bitwidth` on every field unconditionally. A tuple containing a
struct-shaped field (Tuple-of-Tuple / Tuple-of-Record) raises a raw `AttributeError` instead of the clean
caller-naming `TypingError` the surrounding code emits for the not-struct case. Nested-aggregate tuple
signatures aren't currently a supported shape, so this is robustness, not a live failure.

- **Fix:** guard the tuple-element loop to raise the same `TypingError` when a field lacks `.bitwidth`, or document that tuple bindings must be flat scalar tuples. **Confidence:** high. **Findings:** MEM-abi-2.

---

## COR — Correctness

> COR-lowlevel-1 (high, signed-char reader) and COR-abi-1 (medium, tuple alignment) are the COR faces of
> MEM-1 and MEM-2 above; they are not repeated here.

### COR-1 (medium) — printf-family: no format-spec vs argument cross-validation in the `@njit` path

[`numbox/core/bindings/_fmtio.py:210-227, 271-301`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_fmtio.py#L210-L227) (and the SEC face at `316-317, 547-548`)

The codegen layer validates each arg is a printf-able scalar but never cross-checks the conversion
specifiers in the literal format string against arg count or per-arg type. `printf("%d\n", 3.5)` is
accepted: the float64 travels in an SSE register while libc reads a GP integer register for `%d` →
silent wrong result. `printf("%d %d\n", 5)` reads an undefined varargs slot (garbage / crash);
`printf("%s", 5)` passes the type check but libc dereferences the int `5` as a `char*`. Pure-Python mode
raises cleanly on the same calls (`%` enforces agreement), so the two modes diverge: one errors, the
other miscompiles. The format string and arg types are both available at typing time, so this is
statically detectable. (See SEC-1 for the information-disclosure framing.)

- **Evidence:** `_validate_writer_arg_type` only does the per-arg `isinstance(..., (Float, Integer, Boolean, UnicodeType))` check; `fmt_str` is in hand (`extract_literal_str`) but only passed to `_reject_percent_n_or_raise` — nothing counts/maps specifiers. The `%n` regex already parses specifiers, so a specifier walk is feasible.
- **Fix:** at typing time, scan the literal format's conversion specifiers and require (a) specifier count == `len(args_ty)` and (b) each specifier class matches the corresponding arg type (`%d/%i/%x/%u`→Integer/Boolean; `%f/%e/%g`→Float; `%s`→UnicodeType or intp; `%p`→intp), raising `TypingError` on mismatch. At minimum enforce the count check.
- **Confidence:** high. **Findings:** COR-libc_fmtio-2, SEC-libc_fmtio-1.

### COR-2 (low) — `recompute` silently discards a supplied Variables-source override downstream of another change (precedence undocumented)

[`numbox/core/variable/variable.py:386-402`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/variable.py#L386-L402)

When one `recompute()` call supplies explicit new values for A and B where B (a Variables/formula node) transitively depends on A, B's supplied value is silently overwritten by recomputation from A (proven by the repro below).

**Spec gap, not a contract violation.** The `recompute` docstring (lines 382-383) says only that `changed` *may carry* new values for variables "coming from either `External` or `Variables` source" — it specifies no precedence (no "honor"/"preserve"/"override" wording). So whether silently discarding the supplied value is *wrong* is an undocumented API decision, not a broken promise. It is at least surprising, and it makes the `Variables`-source half of `changed` inert for any node downstream of another change. **Downgraded medium -> low** after re-reading the docstring.

- **Evidence:** line 397 assigns each changed var's value and adds it to `changed_vars`. `_collect_affected` is seeded with **all** changed_vars; popping A reaches B's CompiledNode via `dependents.get(A)`, so B lands in `affected`. Lines 400-401 then null every affected node (nulling B's just-assigned override) and `_calculate` recomputes B from A's new value. No guard excludes directly-changed nodes.
- **Fix (a decision + doc, plus an optional guard):** define `recompute`'s precedence for a supplied `Variables`-source value and **document it in the docstring** (override-wins is the natural choice). If override-wins, also exclude directly-changed nodes from the null+recompute pass (`[n for n in affected_nodes if n.variable not in changed_vars]`) so the supplied value survives.
- **Confidence:** high that the behavior occurs; its *wrongness* is the undocumented API decision above (not a fact). **Findings:** COR-variable-1 (the verifier confirmed the mechanism, not the original "wrong result" framing).
- **Repro:** [`repro/cor2_recompute_override_repro.py`](repro/cor2_recompute_override_repro.py) — `test_recompute_honors_variables_override` is red against the reviewed code (`b=20`). Its expected `999` encodes the **assumed** override-wins semantics, not a documented guarantee, so it is a characterization test of the chosen behavior; lift/adjust into `test/core/test_variable.py` once precedence is decided.

### COR-2b (low, post-review) — `CompiledKernel.recompute` shares COR-2's silent interior-override drop; its docstring is also inaccurate

[`numbox/core/variable/compile_kernel.py:705-730`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/compile_kernel.py#L705-L730)

Found during post-review validation (NOT part of the automated 124). `CompiledKernel.recompute` reuses [`_collect_affected(changed_vars)`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/compile_kernel.py#L711) — the same helper behind COR-2 — so it has the **same behavior**: an explicit interior override (`{"vars_": {"b": 999}}`) is **honored in isolation but silently discarded when the overridden node is downstream of another co-changed input** (`{"ext": {"a": 2}, "vars_": {"b": 999}}` -> `b=20, c=21`, the `999` dropped). Verified empirically (repro below).

- **Not a computation bug:** deterministic, internally consistent (`b=10*a`), no corruption/UB. The defect is the *silent* drop of an explicit caller instruction, dependent on what else is in the batch — the same under-specified semantics as COR-2. The code even *warns* when an update has no effect ([`compile_kernel.py:539`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/compile_kernel.py#L539)), so silently allowing this no-op override is inconsistent with its own behavior.
- **Separate doc-accuracy bug (compile_kernel only):** the docstring ([lines 673-676](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/compile_kernel.py#L673-L676)) states unconditionally that an overridden node's "own formula is *not* re-run" — **false** in the co-change case. Fix the docstring (or the behavior) so they agree.
- **Shared root cause with COR-2:** both flow through `_collect_affected(changed_vars)`; decide precedence once (honor / recompute-wins / error — but do not silently drop) and apply it in both `recompute`s.
- **Repro:** [`repro/cor2b_compile_kernel_recompute_override_repro.py`](repro/cor2b_compile_kernel_recompute_override_repro.py).

### COR-3 (medium) — Qualified-name `rsplit('.', 1)` mis-parses any name containing a dot

[`numbox/core/variable/variable.py:94-105, 494, 579`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/variable.py#L94-L105)

Qualified names are `f"{source}.{name}"` and decomposed with `rsplit(QUAL_SEP, 1)`. A name (or source)
containing `.` is not round-trip-invertible: `'a.b'` under source `'s'` → qual `'s.a.b'` → decomposes to
source `'s.a'`, name `'b'`. Result: `KeyError 'Unknown source s.a'`, or — if a source `'s.a'` with var `'b'`
exists — a silently wrong lookup to the wrong Variable. No validation rejects dots in names.

- **Evidence:** `make_qual_name` (105) and the two `rsplit` sites (494, 579); names flow in unvalidated from `Variables.__init__` (216) and `External.__getitem__` (134-137).
- **Fix:** validate at construction that neither source nor name contains `QUAL_SEP` (clear error), or carry `(source, name)` tuples through `_topological_order`/recompute/explain instead of re-splitting a flat string. Document `.` as reserved if validating.
- **Confidence:** medium. **Findings:** COR-variable-2.

### COR-4 (medium) — `make_structref` method-param assertion rejects legitimate signatures

[`numbox/utils/highlevel.py:123-126`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/highlevel.py#L123-L126)

Two related defects in the same `re.findall` + `assert` header-validation block:

1. **repr-vs-source mismatch** — `params_str` is built via `repr()` of each default (`f'{k}={v.default!r}'`,
   [`standard.py:46`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/standard.py#L46)) while `params_str_` is the raw source captured by the header regex. `assert params_str
   == params_str_` then fires for any default whose source spelling differs from `repr()`: `n=1_000`
   (`1_000` vs `1000`), `n=0x10` (`0x10` vs `16`), `s="a, b"` (double vs single quotes).
2. **regex can't span a nested `)`** — the capture `([^)]*)` stops at the first `)`. A signature with a
   nested `)` — `def m(self, x=factory())` or `def m(self, f: Callable[[int], int])` — is mis-captured/
   truncated, so the `assert len(method_header) == 1` and the text-equality assert break.

Either way, valid user input is rejected with an opaque `AssertionError`.

- **Fix:** stop re-deriving the param list from source via regex; use `inspect.signature` / `ast` on the method object (already available — `make_params_strings` calls `inspect.signature`) as the single source of truth and drop the regex cross-check, or compare on a normalized AST form.
- **Confidence:** high. **Findings:** COR-highlevel-1, COR-highlevel-2.

### COR-5 (medium) — `@proxy` cache-anchor mis-lands for functions in the first 12 lines of a file

[`numbox/core/proxy/proxy.py:87-93`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/proxy/proxy.py#L87-L93)

The anchor that makes the generated wrapper's `@njit` land exactly on the user's `@proxy` line silently
fails when `func.__code__.co_firstlineno < njit_lineno_in_txt` (== 13): `prepend = max(0, co_firstlineno -
13) = 0`, so `@njit` compiles at file line 13 instead of the real decorator line. That re-opens the exact
`inspect.findsource` backward-scan / `TokenError` hazard the mechanism is documented to prevent. No in-repo
binding triggers it (all `@proxy` sites sit below line 13 due to import headers), but it is reachable by any
downstream file (numbduck/numbarrow) placing a `@proxy` function near the top of a module.

- **Evidence:** `njit_lineno_in_txt` reconstructs to 13 (confirmed by re-running the generator over `code_txt`); the docstring/`proxy.rst` state the wrapper's `@njit` MUST land at `co_firstlineno`, which the clamp violates for early functions.
- **Fix:** detect and reject the under-anchored case (raise telling the user to move the function below the threshold), or restructure `code_txt` so the `@njit` line is line 1; at minimum assert the alignment so the failure is visible rather than a silent mis-anchor.
- **Confidence:** high. **Findings:** COR-proxy-1.

### COR-6 (medium) — Explicit `ty` not folded into the builder kernel fingerprint

[`numbox/core/work/builder.py:73-77, 106-116, 170-184`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/builder.py#L73-L77)

The content-addressed kernel name `_make_<hash>` is derived only from the generated body text and
`derive_hashes`. An End/Derived node's explicit `ty` enters codegen only as the bare global identifier
`{name}_ty`, so the body text is identical regardless of the type object's value. Two graphs sharing node
names/structure/derive-source/init-values but differing only in an explicit `ty` produce the same
`make_name` and the same runtime arg types (init values type identically). If numba's disk-cache key does
not incorporate the referenced module-global type object's value, the second graph can hit a stale cached
kernel that builds Work nodes of the first graph's declared data type — silently wrong `data` field type.

- **Fix:** fold each node's declared type (`get_ty(spec)`, mangled via `itanium_mangler.mangle_type_or_value` as `highlevel.hash_type` already does) into `hash_str` before computing `hash_`, so a type-only change shifts `make_name`.
- **Confidence:** low. **Findings:** COR-work_builder-1.

### COR-7 (medium) — `explain()` raises `AttributeError` on derive nodes not built via `builder.make_graph`

[`numbox/core/work/explain.py:8-10`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/explain.py#L8-L10)

`get_func_code` does `_derive_funcs.get(derive_func_p_).__name__`, but `_derive_funcs` is populated **only**
by `builder._derived_cres`. A Work derive node constructed directly via `make_work` / `ll_make_work` never
registers, so `.get(...)` returns `None` and `None.__name__` raises `AttributeError` instead of a meaningful
error or correct output.

- **Fix:** guard the lookup — raise a clear "explain() requires a builder.make_graph graph" error, or fall back to the node name without source; document the limitation.
- **Confidence:** medium. **Findings:** COR-work_print_explain-1.

### COR-8 (medium) — `get_input` bounds check ignores negative indices

[`numbox/core/work/node.py:81-88`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/node.py#L81-L88) and [`numbox/core/work/work.py:356-364`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/work.py#L356-L364)

`ol_get_input` only checks `if i >= num_inputs`. A negative `i` (e.g. `-1`) passes that guard and reaches
`self.inputs[i]`; numba's typed/reflected List honors Python negative indexing, so `get_input(-1)` silently
returns the **last** input instead of raising the intended out-of-range `NumbaError`. The caller asked for
an invalid id and gets a wrong node back. Both `node.py` and `work.py` have the identical one-sided guard.

- **Fix:** change both guards to `if i < 0 or i >= num_inputs:` so the existing `NumbaError` covers the full invalid range. (A regression test must call `get_input(-1)` and assert it raises.)
- **Confidence:** medium. **Findings:** COR-work_node-1, COR-work-1.

### COR-9 (low) — `_python_fmt_compat` strips the length modifier from a `%%`-escaped sequence

[`numbox/core/bindings/_fmtio.py:154-161`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_fmtio.py#L154-L161)

In pure-Python mode, a format with a `%%`-escape followed by a length-modifier+conversion (e.g. `"%%ld"`,
`"value is 50%%ld"`) has its length modifier wrongly stripped: `_python_fmt_compat('%%ld')` → `'%%d'` →
`'%d'`, whereas libc emits the literal `'%ld'`. Breaks the module's "same source in either mode" invariant.

- **Evidence:** `_LENGTH_MODIFIER_RE.sub(...)` runs with no prior `%%`-stripping; the `%n` checker at line 252 already does `replace('%%', '\x00\x00')` first — the length-modifier path omits that step.
- **Fix:** mirror the `%n` checker — sentinel-replace `%%` before the sub, restore after; or anchor the regex so it can't start on the second `%` of an escaped pair. **Confidence:** high. **Findings:** COR-libc_fmtio-1.

### COR-10 (low) — `sqlite3_trace_v2` `uMask` declared `int32` for a C `unsigned int`

[`numbox/core/bindings/signatures.py:173`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/signatures.py#L173)

Type-fidelity inconsistency, not a wrong-result bug: the defined trace masks all fit in positive int32 and
a 32-bit register carries the same bits either way. Latent only if a future caller passes a mask with bit
31 set via the int32 path. Sibling unsigned params (`sqlite3_value_subtype`, `sqlite3_result_subtype`)
already use `uint32`.

- **Fix:** `"sqlite3_trace_v2": int32(intp, uint32, intp, intp)`. **Confidence:** high. **Findings:** COR-sqlite_hooks-1.

### COR-11 (low) — `get_jit_options` accepts any valid JSON, not just an object

[`numbox/core/configurations.py:5-19`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/configurations.py#L5-L19)

If `NUMBOX_JIT_OPTIONS` is valid JSON but not an object (`null`, `true`, `42`, `[]`, `"cache"`),
`json.loads` succeeds and `get_jit_options` returns a non-dict that is bound to module-level `jit_options`
and dict-splatted as `@njit(**jit_options)` across dozens of sites → `TypeError: argument after ** must be
a mapping` at import time, far from the env-var cause.

- **Fix:** after `json.loads`, validate `isinstance(as_json, dict)` and raise `ValueError` (matching the existing `JSONDecodeError → ValueError` style) so misuse fails at config-parse time. **Confidence:** high. **Findings:** COR-configurations-1, DES-configurations-1.

### COR-12 (low) — Reserved-name guard checks the wrapper name but not the intrinsic name

[`numbox/core/proxy/proxy.py:80-81`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/proxy/proxy.py#L80-L81)

The exec'd source defines both the intrinsic `_{name}` and the wrapper `{name}`; the pre-exec guard only
checks the wrapper name, so a pre-existing collision on the intrinsic name is silently shadowed in the
fresh `ns` (no module mutation, only the diagnostic `ValueError` is lost).

- **Fix:** extend the guard to also test the intrinsic name (`make_proxy_name(func_proxy_name)`). **Confidence:** high. **Findings:** COR-proxy-2.

### COR-13 (low) — `make_node` recurses before memoizing → `RecursionError` on a cyclic registry

[`numbox/core/variable/node.py:22-31`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/node.py#L22-L31)

`_make` writes `made[key]` only **after** recursing into all inputs, so a cyclic registry (A↔B) never
terminates and raises `RecursionError` instead of the clean "Cycle detected" error that
`Graph._topological_order` produces via its `visiting` set. `make_node` is reachable directly in tests
without going through `Graph.compile`, so the two entry points disagree on the same malformed graph.

- **Fix:** insert a sentinel into `made[key]` before recursing and detect re-entry to raise a cycle error, or mirror the `visiting`-set guard. **Confidence:** high. **Findings:** COR-variable_node_utils-1.

### COR-14 (low) — Vector `getitem`/`setitem` index the capacity-sized buffer, not logical size

[`numbox/core/vector/vector.py:104-117`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/vector/vector.py#L104-L117)

`v[i]` returns `v.buf[i]` with no comparison against `v.size`; for `size <= i < capacity` it returns an
uninitialized slot, and `i >= capacity` is an unchecked OOB access. The existing test deliberately sets
`v.size` by hand, so the raw-buffer idiom appears intended.

- **Fix:** document on the overloads that the index is unchecked against size (valid only for `0 <= i < size`, caller owns the bound), or add an `i < v.size` IndexError check if logical-vector semantics are intended. **Confidence:** low. **Findings:** COR-vector-1.

### COR-15 (low) — Derive fingerprint hashes only `getsource`, missing referenced globals/closures

[`numbox/core/work/builder.py:108`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/builder.py#L108)

`derive_hashes.append(sha256(getsource(derive_func)...))` fingerprints a derive by source text only; a
referenced global/closure whose value changes without a text change doesn't shift `hash_`. Mostly benign —
the derive is compiled separately and passed as a runtime arg, so numba's own derive cache governs the
body; the residual risk is the builder anchor not invalidating.

- **Fix:** route derive fingerprinting through a structured code+consts+closure+globals helper (if `numbox/utils/fingerprint.py` provides one), else document that pure-literal derive edits require a cache clear. **Confidence:** low. **Findings:** COR-work_builder-2.

### COR-16 (low) — `all_inputs_names`/`all_end_nodes` dedup on name only and have no cycle guard

[`numbox/core/work/node.py:101-148`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/node.py#L101-L148)

Dedup keys purely on the string `name`, so two structurally distinct same-named input nodes collapse to one
(losing one). The recursive helpers re-descend into every input unconditionally; termination relies on the
undocumented DAG + unique-name precondition, so a cycle recurses to stack exhaustion. Wide diamonds re-walk
shared subtrees (bounded redundancy).

- **Fix:** document the unique-name + acyclic precondition, or guard re-descent with a visited set keyed on node **identity** so a cycle yields a defined error and shared subtrees are walked once. **Confidence:** medium. **Findings:** COR-work_node-2.

### COR-17 (low) — `clock_gettime` return value ignored; failure yields a silent 0 ns

[`numbox/utils/clock.py:93-104`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/clock.py#L93-L104)

The POSIX `monotonic_ns` codegen discards the i32 return of `clock_gettime(CLOCK_MONOTONIC, &ts)`. The
timespec is pre-zeroed, so a `-1` return (theoretical for CLOCK_MONOTONIC) yields `0` ns silently,
violating the contract on the failure path.

- **Fix:** branch on the i32 return and trap/raise on non-zero, or document this as best-effort. **Confidence:** medium. **Findings:** COR-utils_misc-2.

### COR-18 (low) — `digest()` concatenates per-function fingerprints with no delimiter

[`numbox/utils/digest.py:42-52`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/digest.py#L42-L52)

Each fn's fingerprint is folded into the same sha256 stream back-to-back with no separator/length prefix, so
in principle two distinct `(subject, fns)` inputs with identical concatenated bytes collide on the cache key
(stale kernel → wrong result). Implausible in practice (structured `func(<module>:<qualname>;...` prefixes,
fixed-arity role-ordered fn lists).

- **Fix:** fold a fixed delimiter + count before each fn (`h.update(f"fn{i}\x00".encode())`). **Confidence:** low. **Findings:** COR-fingerprint_digest-2.

### COR-19 (low) — `_canon_value` lacks a cycle key for self-referential plain containers

[`numbox/utils/fingerprint.py:55-61`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/fingerprint.py#L55-L61)

A self-referential built-in container reachable from a default/closure/global drives `_canon_value` into
unbounded recursion → `RecursionError`. **Not** a wrong-result bug: both consumers (`compile_kernel` and
`digest`) catch `RecursionError` and degrade safely.

- **Fix (optional):** add an id-based cycle guard for tuple/list/set/dict in `_canon_value`, mirroring the function guard at lines 92-94. **Confidence:** high. **Findings:** COR-fingerprint_digest-1.

### COR-20 (low) — `libraries_coordinated()` tests version-string equality, not library identity

[`numbox/utils/pysqlite_bridge.py:93-107`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/pysqlite_bridge.py#L93-L107)

Returns True whenever the two libsqlite3 builds report the same version string, even if numbox's bindings
and Python's `sqlite3` are bound to two physically different binaries of that version. Contract/heuristic
gap — the opaque-handle ABI is stable across same-major versions, so this is not a demonstrated wrong-result
path.

- **Fix:** soften the docstring to state it's a version-string check (strong proxy, not proof), or strengthen with `sqlite3_libversion_number()` / a resolved-symbol pointer-identity check. **Confidence:** low. **Findings:** COR-pysqlite_bridge-1.

### COR-21 (low) — Default-vs-empty test uses `==` instead of `is` against `inspect._empty`

[`numbox/utils/standard.py:45-47`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/standard.py#L45-L47)

`v.default == inspect._empty` compares for equality, not identity. For a default whose `__eq__` returns a
non-bool against the sentinel (e.g. `def f(x, y=np.zeros(3))`), the comparison returns a numpy array and the
surrounding ternary raises `ValueError: truth value of an array is ambiguous` at codegen-string assembly,
pre-empting the visible repr/exec failure the docstring promises. Affects both `make_params_strings`
consumers (the `cres`/`standard` path and the `make_structref` struct-methods path).

- **Fix:** use identity — `v.default is inspect._empty`. **Confidence:** high. **Findings:** COR-utils_misc-1, COR-highlevel-3.

---

## SEC — Security

### SEC-1 (medium) — Format conversion specifiers never checked against arg count or per-arg type

[`numbox/core/bindings/_fmtio.py:210-227, 316-317, 547-548`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_fmtio.py#L210-L227)

The security face of COR-1. printf/fprintf/snprintf validate only that each variadic arg is some scalar;
they never parse the format's conversion specifiers, so neither the count nor the kinds are matched.
`printf("%d %d", x)` (one arg, two conversions) makes libc read a second variadic slot that was never
pushed → garbage register/stack value formatted out (**information disclosure**) or crash. `printf("%s", 5)`
passes the type check (Integer is allowed) but libc dereferences the int `5` as a `char*`
(crash / OOB disclosure). Format string is a compile-time literal and arg types are known at typing time —
statically detectable yet undetected. The module docstring even claims "the same constraint a C compiler
operates under when emitting a format-checked printf call" — but `-Wformat` checks count and per-specifier
type; this binding does not.

- **Fix:** parse the literal format's specifiers at typing time and require count == `len(args_ty)` plus per-specifier class checks against arg types, delivering the C-compiler-grade checking the docstring promises and closing the garbage-read / pointer-deref hazards.
- **Confidence:** high. **Findings:** SEC-libc_fmtio-1, COR-libc_fmtio-2.

### SEC-2 (low) — `sscanf` permits unbounded `%s` / `%[` conversions into caller buffers

[`numbox/core/bindings/_fmtio.py:387-415, 667-686`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_fmtio.py#L387-L415)

`sscanf` validates only that output args are intp pointers; it does not require a max field width on `%s`/
`%[`. `sscanf(buf_p, "%s", out)` over an attacker-influenced buffer writes every non-whitespace byte plus a
NUL into `out` with no length cap → buffer overflow / memory corruption (classic C `sscanf` overflow).
Downgraded to low: the format is a developer-written compile-time literal (the developer can write `"%63s"`)
and the binding is an intentionally thin C-ABI surface where the caller owns buffer and format.

- **Fix (defense-in-depth):** reject width-less `%s`/`%[` in the sscanf typing function, or at minimum document the overflow hazard and recommend an explicit max field width. **Confidence:** high. **Findings:** SEC-libc_fmtio-2.

### SEC-3 (low) — Generic Windows fallback loads `find_library()` result with `winmode=0`

[`numbox/core/bindings/utils.py:109-163`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/utils.py#L109-L163)

For a non-`c`/`m`/`sqlite3` Windows library with no bundled DLL, `_resolve_lib_path` falls through to
`find_library(name)` (may return a bare filename) and `_load_lib_with_handle` calls `CDLL(path, winmode=0)`
(LoadLibraryEx flags=0 → default DLL search order for a bare name) → DLL-planting/hijack surface if such a
binding is ever added. **No library numbox currently loads reaches this branch** (`c`/`m` go through
`find_msvcrt`; `sqlite3` is forced to the absolute bundled path and explicitly never uses `find_library`).
Latent / defense-in-depth only.

- **Fix:** when/if a binding uses the generic `find_library` path, resolve to an absolute path before `CDLL`, or pass `winmode=LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`, following the absolute-bundled-path pattern already used for sqlite3. **Confidence:** medium. **Findings:** SEC-bindings_utils-1.

---

## DES — Design / simplification

### DES-medium

**DES-m1 — UTF-8 decode state machine duplicated** ([`_sqlite_query.py:32-71`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_query.py#L32-L71)). `_put_unicode` reimplements
the entire UTF-8 validation/decode machine that already exists in `_sqlite_typemap.utf8_to_utf32` (byte-for-
byte identical 1/2/3/4-byte forms, overlong/surrogate/range checks, U+FFFD fallback); the copies differ only
in the emit step (tracked-array write vs raw-pointer `store_unaligned`). Two copies of a security-sensitive
decoder = latent divergence. **Fix:** factor the decode loop so only the emit step varies (write callback /
shared decoder into a tracked uint32 scratch). **Conf:** high. **Finding:** DES-sqlite_query-1.

**DES-m2 — 14-branch tag→sqlite3_result ladder duplicated across three cfunc bodies** ([`_sqlite_tvf.py:308-343`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_tvf.py#L308-L343)).
`_tvf_xcolumn` is a near-verbatim copy of `_sqlite_vtable._xcolumn` (same 14 tag branches, loads, casts,
NUL-trim + utf32→utf8), differing only by `_SQLITE_TRANSIENT` vs `SQLITE_STATIC`; the per-tag load logic is
duplicated a third time in `_cell_value_f64`/`_cell_value_i64`. Adding a dtype tag/width fix requires lockstep
edits. **Fix:** factor a single `_emit_cell(...)` helper into `_sqlite_typemap.py` taking the destructor
sentinel as an argument. **Conf:** high. **Finding:** DES-sqlite_tvf-1.

**DES-m3 — Case-B eager segment loop duplicates `_discover_and_run`'s orchestration** ([`compile_kernel.py:906-934`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/compile_kernel.py#L906-L934)).
The eager build loop reimplements the `for kind, run_nodes in runs:` orchestration already in
`_discover_and_run` (lines 450-485); the author flags it inline. Differences: declared types vs runtime
`typeof`, and `seg_sigs` folding. **Fix:** extract a `_build_plan(...)` helper parameterized by how each jit
run's compile-types are obtained and whether `seg_sigs` is folded; call from both sites. **Conf:** high.
**Finding:** DES-compile_kernel-1.

**DES-m4 — Four-way duplication of the registry-cached codegen skeleton** ([`work.py:219-239,287-304,336-353,402-421`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/work.py#L219-L239)).
`ol_calculate`/`ol_load`/`ol_combine`/`ol_make_inputs_vector` repeat the identical lookup→ns→ensure-getters
→compile→exec→cache skeleton; only the code-builder, registry, exec name, and ns extras vary. **Fix:** extract
one `_codegen_method(num_sources, registry, make_code, exec_name, ns_extras=None)` helper. **Conf:** high.
**Finding:** DES-work-3.

**DES-m5 — Asymmetric, partially-undocumented `removerefctpass` defense in `meminfo.py`** ([`meminfo.py:77-111`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/meminfo.py#L77-L111)).
`_release_meminfo` calls `NRT_MemInfo_release` directly with a multi-line note explaining why a plain decref
would be stripped; `_incref_meminfo` has the identical `void(intp)` shape but uses `context.nrt.incref` with
**no** note on why that incref survives the same pass. A maintainer can't tell if the incref is safe-by-design
or a latent strip-vulnerability, and `_sqlite_tvf.py` already relies on the inlined incref surviving. **Fix:**
add a one-paragraph docstring to `_incref_meminfo` stating why the incref is not stripped (or switch to a
direct `NRT_incref` for symmetry). **Conf:** medium. **Finding:** DES-meminfo-1.

**DES-m6 — `timer.py` mutates the root logger at import and logs routine timing at WARNING** ([`timer.py:5-21`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/timer.py#L5-L21)).
Importing `numbox.utils.timer` calls `logging.basicConfig(level=WARNING)` at module top level (a library
permanently reconfiguring the process-wide root logger), and emits per-call timing via `logger.warning(...)`
so routine info surfaces as warnings. **Fix:** drop the module-level `basicConfig` (let the app own root
config) and emit via `logger.info(...)`. Only in-repo callers read `timer.times`, not the log level.
**Conf:** high. **Finding:** DES-utils_misc-1. *(Paired with TST-tst_utils_core-6 — no test for Timer — under
the same `timer.py` cluster.)*

### DES-low

These are localized cleanups (dead params/state, duplication, naming, single-use wrappers). Grouped here for
brevity; each is non-behavioral. Files and the one-line action:

- **Dead state / dead params:** [`work.py:169,211-216`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/work.py#L169) write-only `_source_getter_registry` (DES-work-1);
  [`builder.py:124-132`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/builder.py#L124-L132) dead `registry` param on `_infer_end_and_derived_nodes` (DES-work_builder-1);
  [`builder.py:25-26,50-53`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/builder.py#L25-L26) `registry` field stored-never-read on End/Derived (DES-work_builder-2);
  [`_kernel_partition.py:168-197`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/_kernel_partition.py#L168-L197) dead `external` param on `segment_liveness` (DES-kernel_partition-1);
  [`combine_utils.py:42-51`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/combine_utils.py#L42-L51) unused `field_ind` enumerate index (DES-work_combine_loader_lowlevel-2);
  [`compile_kernel.py:195-196`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/compile_kernel.py#L195-L196) `in_names` computed only for a source comment (DES-compile_kernel-3).
- **Duplication to factor:** `lowlevel.py:58-61,…` verbatim pointer-type TypingError guard ×4 →
  `_require_intp_pointer` helper (DES-lowlevel-1); [`lowlevel.py:56-152`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/lowlevel.py#L56-L152) aligned/unaligned load+store codegen
  differs only by `align=1` — keep the public 4-way split, factor the inner codegen (DES-lowlevel-2);
  [`_kernel_partition.py:168-228`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/_kernel_partition.py#L168-L228) `segment_liveness`/`cone_liveness` share live-in logic (DES-kernel_partition-2);
  [`_sqlite_query.py:113-126`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_query.py#L113-L126) `_TAG_S`/`_TAG_BLOB` differ only by accessor (DES-sqlite_query-2);
  [`_fmtio.py:237-259,504-518`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_fmtio.py#L237-L259) `%n`-rejection ×2 (DES-libc_fmtio-1); [`_fmtio.py:401-405,677-681`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_fmtio.py#L401-L405) sscanf
  arg-type check inlined ×4 (DES-libc_fmtio-2); `_sqlite_udf_helpers.py` codegen scaffolding shared with
  `_sqlite_tvf.py` (DES-sqlite_udf-2), repeated try/except+`sqlite3_result_error` idiom (DES-sqlite_udf-3),
  identical init+finalize branches (DES-sqlite_udf-1); [`_sqlite_tvf.py:146-289`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_tvf.py#L146-L289) xConnect/xDisconnect/xRowid/
  xEof boilerplate re-implemented vs the vtable module (DES-sqlite_tvf-3); `call.py:176-177,…` `{i64,i64}`
  literal struct reconstructed ×3 (DES-call-1); [`bindings/utils.py:159-164`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/utils.py#L159-L164) per-platform CDLL block ×2
  (DES-bindings_utils-1); [`any_type.py:63-66,77-82`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/any/any_type.py#L63-L66) payload-wrap idiom ×2 (DES-any-1); [`vector.py:120-141`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/vector/vector.py#L120-L141)
  grow-and-copy block ×2 (DES-vector-1); [`work/node.py:101-148`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/node.py#L101-L148) traversal body duplicated module-helper vs
  overload (DES-work_node-1); [`work.py:367-374,454-465`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/work.py#L367-L374) `get_inputs_names`/`depends_on` copy-pasted Work↔Node
  (DES-work-4); [`compile_kernel.py:454-462,917-919`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/compile_kernel.py#L454-L462) live-in computed two ways (DES-compile_kernel-2),
  `:464` duplicated sort-key lambda (DES-compile_kernel-4); [`pysqlite_bridge.py:106,148`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/pysqlite_bridge.py#L106) version-string
  computed twice (DES-pysqlite_bridge-1); [`_errno.py:35`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_errno.py#L35) inconsistent size_t/intp construction across
  siblings (DES-libc_stdio_strerror_errno-1).
- **Single-use wrapper / API hygiene:** [`_sqlite_vtable.py:730-739`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_vtable.py#L730-L739) `_VTableHandle` is a redundant single-use
  wrapper around the already-keep-alive `_BuiltDescriptor` (DES-sqlite_vtable-1), and its `__init__` takes
  variadic `*objs` but is always called with one (DES-sqlite_vtable-2); [`print_tree.py:15-55`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/print_tree.py#L15-L55) internal helpers
  lack a leading underscore and `make_graph` shadows the public `builder.make_graph` (DES-work_print_explain-1);
  [`explain.py:24-25`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/explain.py#L24-L25) bare magic `[1]` index into `work.derive` (DES-work_print_explain-2);
  [`combine_utils.py:57-66`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/combine_utils.py#L57-L66) overload name contradicts behavior (copy-paste from loader_utils)
  (DES-work_combine_loader_lowlevel-1); [`proxy.py:48-49`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/proxy/proxy.py#L48-L49) self-referential `dict.update` obscures
  `inline='always'` (DES-proxy-1); `_math.py` wrappers `round`/`pow` shadow builtins in the star-imported
  public namespace (DES-libc_math-1); [`_fmtio.py:682-685,461-467`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_fmtio.py#L682-L685) sscanf threads an unused
  `get_unicode_data_p` arg (DES-libc_fmtio-3); [`_strerror.py:95-114`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_strerror.py#L95-L114) test-only IR-probe helper lives in the
  production module (DES-libc_stdio_strerror_errno-2); [`_sqlite_tvf.py:85,403`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_tvf.py#L85) `_TVF_DESC_DTYPE.itemsize`
  written never read (DES-sqlite_tvf-2); [`_sqlite_typemap.py:66-105`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_typemap.py#L66-L105) public `utf8_to_utf32` has no production
  caller (only its `_sqlite_query` twin is live) (DES-sqlite_typemap_constants-1); [`abi.py:155-160`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/abi.py#L155-L160) docstring
  attributes orchestration to the wrong caller, `:165-184` loop var `size` shadows the outer `size`
  (DES-abi-1, DES-abi-2); [`highlevel.py:68-70,242-247`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/utils/highlevel.py#L68-L70) `hash_type`/`prune_type` are unused public surface,
  `:84` identity list-comprehension, `:219-234` six redundant exec-ns entries (DES-highlevel-1/3/2);
  [`work.py:346,414`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/work/work.py#L346) redundant ns overrides re-inject module-level names (DES-work-2), `:460-465`
  assert-after-def placement inconsistent with the Node sibling (DES-work-5); [`variable/utils.py:1-7`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/utils.py#L1-L7)
  module docstring omits one helper concern (DES-variable_node_utils-1); [`variable.py:391-396`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/variable/variable.py#L391-L396) O(N) linear
  scan over `ordered_nodes` (DES-variable-2), `:478-479` dead defensive string-normalization branch
  (DES-variable-1).
- **Config object-ness:** [`configurations.py:5-19`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/configurations.py#L5-L19) validates JSON syntax but not dict-ness (DES-configurations-1)
  — folded into COR-11.
- **Docs grouping:** `docs/numbox.utils.rst` missing automodule for `clock`/`standard`/`void_type`
  (DES-utils_misc-2) — folded into the docs task with TST-tst_utils_core-3.

---

## TST — Tests & docs

### TST-medium

**TST-m1 — register_tvf string/unicode/blob output columns are never tested** ([`_sqlite_tvf.py:330-343`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_tvf.py#L330-L343) plus
the scratch path `232-235,394,408`). Every `register_tvf` test uses an int/float-only `out_dtype`, so
`_tvf_xcolumn`'s `_TAG_S`/`_TAG_U`/`_TAG_BLOB` branches and the per-cursor scratch (sizing, TRANSIENT) never
execute. A wrong offset/`width//4`/scratch-size/TRANSIENT-vs-STATIC choice in the tvf string path would ship
undetected even though the analogous `register_table` path is exhaustively tested. **Fix:** add a
`register_tvf` test with `U`/`S`/BLOB columns (interior-NUL + multi-byte/emoji to exercise scratch sizing and
TRANSIENT), asserting round-trip through `sqlite3_column_text`/`_blob`/`_bytes`. **Conf:** high.
**Finding:** TST-tst_sqlite-1.

**TST-m2 — Any.reset old-payload release is not refcount-asserted** (`test/core/test_any_type.py:test_10`).
`ol_reset` overwrites `self.p` on every reset; correctness depends on numba's structref setattr decref'ing
the old field exactly once. `test_10` resets thrice but never asserts the previously-held content's refcount
dropped, so a leak (no decref) or double-free (extra decref) on overwrite would pass. **Fix:** bind a
structref, reset Any onto it, capture `get_nrt_refcount`, reset onto a different value, assert the first
refcount dropped by exactly 1. **Conf:** high. **Finding:** TST-tst_utils_core-4.

**TST-m3 — Windows DLL search-order resolution has no test** ([`bindings/utils.py:49-112`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/utils.py#L49-L112)). `_resolve_lib_path`
implements the security-relevant rule that `sqlite3` MUST return the CPython-bundled DLL and NEVER fall back
to `find_library` (a PATH-resident third-party `sqlite3.dll` writes to NULL in `sqlite3_open` from a foreign
process). No test asserts the bundled path is preferred or that the sqlite3 fallback is refused, so a
regression re-enabling the fallback would pass CI and only crash on a runner with AWS CLI on PATH. **Fix:**
add a pure-logic test monkeypatching `utils.platform_ = 'Windows'` and the candidate-existence check,
asserting (a) `_resolve_lib_path('sqlite3')` returns the bundled path and `None` (not `find_library`) when
absent, and (b) for a non-sqlite3 name the bundled path beats `find_library`. **Conf:** high.
**Finding:** TST-tst_bindings_abi_libc-2.

**TST-m4 — No musl runner; strerror_safe musl path covered only by a glibc IR probe** ([`numbox_ci.yml:25,33-86`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/.github/workflows/numbox_ci.yml#L25)).
`_strerror.py` depends on a musl-specific runtime invariant (`__xpg_strerror_r` weak-aliases musl's POSIX
`strerror_r`). The only test is `skipif != Linux` and is an IR-text inspection that monkeypatches
`address_of_symbol` — it never runs through musl's actual `strerror_r`, and the matrix has no Alpine
container, so musl behavior is never run end-to-end. **Fix:** add a single Alpine (musl) CI job running the
`test_strerror_safe.py` runtime tests + the all-families subprocess probe. **Conf:** high.
**Finding:** TST-tst_bindings_abi_libc-4. *(This also supplies the `musl_symbol_check` canary that
TST-m5 / the `_strerror` docstring advertises.)*

**TST-m5 — `_strerror` docstring asserts a CI canary that does not exist** ([`_strerror.py:24-26`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_strerror.py#L24-L26)). The module
docstring claims the fallback is "verified by the Alpine musl_symbol_check CI canary"; no such canary exists
(grep of `.github/` for musl/alpine/xpg/symbol_check returns nothing; matrix is all-glibc). A reader trusting
it believes the musl invariant is CI-guarded when it is not. **Fix:** either add the Alpine/musl canary (an
`nm -D` check that both `strerror_r` and `__xpg_strerror_r` resolve in a musl container — same job as TST-m4)
or soften the docstring to match the rst's "proposed, not-yet-implemented" wording. **Conf:** high.
**Finding:** TST-tst_bindings_abi_libc-1.

**TST-m6 — Sphinx build is not gated on PRs and tolerates all warnings** ([`docs.yml:3-5,28-29`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/.github/workflows/docs.yml#L3-L5)). `docs.yml`
runs `sphinx-build` only `on: push: branches: [main]` and without `-W`/`--keep-going`, so a PR that breaks
the build or introduces dangling xrefs/broken automodule targets (the doc findings below) is never checked
before merge and warnings never fail CI. `doc-codeblock-flake8.yml` only lints `code-block:: python`, missing
literal `::` examples and rst-structural drift. **Fix:** add a `pull_request` docs job (build, skip the
gh-pages deploy) and add `-W --keep-going` so broken automodule targets and dangling xrefs fail the build.
**Conf:** high. **Finding:** TST-tst_docs_build-5.

**TST-m7 — pysqlite_bridge docs claim a non-existent RTLD_GLOBAL import-preload** ([`docs/numbox.utils.rst:210-217`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/docs/numbox.utils.rst#L210-L217)).
The "Import order matters on macOS" note says importing `pysqlite_bridge` preloads libsqlite3 with
`RTLD_GLOBAL` so subsequent JIT compiles resolve sqlite3 symbols to the same library — no such preload exists.
A user following this guidance gets no symbol-resolution effect and may ship a broken macOS setup. The actual
fix is the `add_symbol` path in the module docstring. **Fix:** delete/rewrite the paragraph to match the
`add_symbol` mechanism. **Conf:** high. **Finding:** TST-tst_utils_core-1.

**TST-m8 — Doc says Py_DEBUG builds are unsupported, but the code supports them** ([`docs/numbox.utils.rst:219-221`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/docs/numbox.utils.rst#L219-L221)).
utils.rst says "Py_DEBUG builds … are not supported", but `_pyobject_head_fields()` detects them via
`sys.gettotalrefcount` and prepends the two extra `_ob_next`/`_ob_prev` pointers so the db offset is correct.
The doc misrepresents a supported config — a debug-build user might avoid a working path, or a maintainer
might delete the working detection. **Fix:** update the doc to state debug builds ARE handled; add a test
asserting `_pyobject_head_fields()` includes `_ob_next`/`_ob_prev` under a simulated debug build to pin it.
**Conf:** high. **Finding:** TST-tst_utils_core-2.

**TST-m9 — `_sqlite_typemap` module entirely missing from sphinx docs** ([`docs/numbox.core.bindings.rst:19-31,448-651`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/docs/numbox.core.bindings.rst#L19-L31)).
`_sqlite_typemap.py` exposes public API (`utf32_to_utf8`, `utf8_to_utf32`, `tags_buf_t`) but has no
automodule directive and is absent from the "Bindings module conventions" family list — the lone omission
among the `_sqlite_*`/`_*` modules, and a documented-convention violation per CLAUDE.md "Adding a New
Binding" step 5. The correctness-sensitive UTF transcoding helpers are undocumented. **Fix:** add it to the
conventions family list and a per-module automodule section. **Conf:** high. **Findings:** TST-tst_docs_build-1,
TST-tst_sqlite-2.

**TST-m10 — work.rst broken xref and dead graph-manager example** ([`docs/numbox.core.work.rst:341,524-526,542`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/docs/numbox.core.work.rst#L341)).
(a) `:class:`numbox.core.utils.void_type.VoidType`` points at a non-existent path — VoidType lives at
`numbox.utils.void_type` (single-token `core.` typo). (b) The graph-manager `::` example imports three
symbols that exist nowhere (`default_jit_options`, `_make_work`, and the `ol_make_work` xref): copy-paste
hits ImportError, the xref dangles. **Fix:** correct the xref to `numbox.utils.void_type.VoidType`; update
the example to `from numbox.core.configurations import jit_options`, replace `_make_work` with `make_work`
(or `ll_make_work`), and fix the line-542 xref to a real symbol. **Conf:** high.
**Findings:** TST-tst_docs_build-3, TST-tst_docs_build-2.

### TST-low

Localized doc-drift and untested-surface gaps (grouped):

- **Doc drift:** [`docs/numbox.core.variable.rst:129-133`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/docs/numbox.core.variable.rst#L129-L133) references a nonexistent `.variables` attribute on
  the Variables namespace (TST-tst_variable_work-1); [`docs/numbox.core.work.rst:473,480,199,408-412`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/docs/numbox.core.work.rst#L473) public
  modules referenced by `:func:` xrefs have no automodule (undocumented API + dangling xrefs)
  (TST-tst_docs_build-4); `docs/numbox.utils.rst` omits automodule for `clock`/`meminfo`/`standard`/
  `void_type` (TST-tst_utils_core-3, overlapping DES-utils_misc-2).
- **Untested public surface:** [`_c.py:50-144`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_c.py#L50-L144) `puts`/`fputc`/`putchar`/`feof`/`ferror`/`clearerr` have no test
  (TST-tst_bindings_abi_libc-3); [`bindings/utils.py:13-46`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/utils.py#L13-L46) `intp_ll_type`/`extract_literal_str` edge cases
  untested (TST-tst_bindings_abi_libc-6); [`_fmtio.py:81-85`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_fmtio.py#L81-L85) documented `%ld` Win64 truncation footgun has no
  asserting test (TST-tst_bindings_abi_libc-5); `timer.py:Timer` has no unit test (TST-tst_utils_core-6,
  paired with DES-utils_misc-1).
- **Untested edge cases:** [`_sqlite_query.py:111-126`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_query.py#L111-L126) oversized-cell truncation (`nbytes >= width`) for
  TEXT/BLOB/U is untested (TST-tst_sqlite-3, paired with the `_sqlite_query` cluster);
  [`_sqlite_typemap.py:78-105`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_typemap.py#L78-L105) truncated multi-byte sequence at end-of-buffer untested (TST-tst_sqlite-4);
  `test_sqlite_constants.py` asserts only the 8 index-constraint ops, leaving result/type/flag/destructor
  values unguarded (TST-tst_sqlite-5); [`test_lowlevel.py:165-171`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/test/utils/test_lowlevel.py#L165-L171) `get_str_from_p_as_int` tested on one
  ASCII string only — empty-string and multibyte edges untested (TST-tst_utils_core-5, the test gap behind
  MEM-1/COR-lowlevel-1); `test_digest.py` `_Unfingerprintable` fallback for a real non-codeless FunctionType
  untested (TST-tst_utils_core-7); `test_params_jitability.py` no test for `_evaluate` seeding of a declared
  kernel with a demoted (Case B) node (TST-tst_variable_work-3) nor for recompute interior-override on a
  declared/eager kernel (TST-tst_variable_work-2).

---

## Human triage

One finding could not be confirmed or refuted by reading source alone:

### DES-sqlite_query-3 (low, uncertain) — `_copy_bytes` hand-rolls a byte loop where a tracked slice assignment would do

[`numbox/core/bindings/_sqlite_query.py:25-29`](https://github.com/nelson2005/numbox/blob/ece98cec16f27c6d0e8ea5d985e591252e2d7c89/numbox/core/bindings/_sqlite_query.py#L25-L29)

`_copy_bytes` is an explicit `for k in range(nbytes): dst[off+k] = src[k]`. The finding claims a numba slice
assignment `dst[off:off+nbytes] = src[:nbytes]` is *also* a tracked array write — so it would preserve the
macOS-arm64 DCE-avoidance property the whole module is built around — and is simpler.

**Why uncertain:** the simplification's safety depends on whether numba's slice-assignment lowering survives
the macOS-arm64 LLVM optimizer's dead-store elimination the same way the element-by-element write does. That
is a runtime property of a specific toolchain (the DCE bug is documented as macOS-arm64 / py3.14 / numba
0.65.1 only) and **cannot be determined from reading source**. The finding's own recommendation says "verify
the slice-assign survives the macOS-arm64 optimizer before relying on it."

**What a human must check:** on macOS-arm64 (py3.14 / numba 0.65.1, the toolchain where the DCE bug was
observed), rewrite `_copy_bytes` to the slice assignment, run the `query_to_array` round-trip tests, and
confirm the result is not all-zeros (i.e. the writes were not eliminated). If they survive, adopt the
slice-assign; otherwise keep the explicit loop. Do **not** apply this as a blind refactor.

---

## Coverage

- **Targets / units:** 119 review units (target × lens). All **119 reviews** and all **119 verifies** are
  present on disk — no unit is missing its findings or verdict file. `coverage` in `_aggregate.json`:
  `review_units=119, reviews_done=119, verifies_done=119`.
- **Findings:** 124 raised → 118 confirmed, 1 uncertain, **5 refuted**. Verification was working: it culled
  five false positives (`DES-sqlite_vtable-3`, `DES-libc_c-1`, `COR-work_print_explain-2`, `COR-any-1`,
  `MEM-pysqlite_bridge-2`), which are excluded from the body and counted here only.
- **Clusters:** 17 multi-finding clusters (same file + ~line) were merged into single report entries / fix
  tasks; every contributing finding id is cited. Severity tallies count each cluster once (no double-count).
