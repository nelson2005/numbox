# Dimension MEM — Memory / ABI / refcount safety (highest-value dimension)

This is the highest-value dimension for numbox: it threads C ABIs, raw pointers, NRT refcounts, and
the numba/LLVM JIT boundary, where mistakes are silent and catastrophic. Audit your target for:

## What to hunt

- **NRT refcount balance.** Every `incref`/`borrow` must be matched by a `decref`/`release` on every
  path (including early returns and error paths). Look for: meminfo leaked when a structref/array is
  borrowed but not released; double-decref; refs created and dropped on the floor. `borrow_structref`
  internally does `_incref_meminfo` + `_deref_structref_raw_ptr`; the lifecycle helpers are
  `export_meminfo` (init), `borrow_structref` (update/combine/finalize), `release_meminfo` (destroy).
- **`@cfunc` exception boundary (two independent failure modes).** When an exception is raised inside
  a `@cfunc` body handed to a C library: (1) numba **swallows** it, prints to stderr, and returns the
  type's ZERO default — the C caller is never told, so you get a **silent wrong result**; (2) an NRT
  ref held **live across a separate call that raises** leaks one meminfo (the post-call decref is
  skipped on unwind). Cure is a **bare `try/except`** inside the body (NOT `try/finally` — that
  `RERAISE`s and asserts on numba 0.65.1), plus signalling via the callback's own channel
  (`sqlite3_result_error[_code]` for context callbacks; abort/veto return code for return-code
  callbacks; nothing for void callbacks — there, catch only to avoid the leak). Flag any `@cfunc`
  body that can raise across a call without this guard.
- **cfunc / JIT-code lifetime vs data lifetime.** numba JIT code (cfunc/njit) lives for the process
  and is NOT reclaimed on GC; dispatchers are globally pinned. So a "keep-alive handle" whose only job
  is to retain a registered cfunc pointer is **unnecessary** (don't flag its absence as a bug; don't
  praise its presence as required). BUT a handle that owns **data the C library reads later** (e.g. a
  vtable descriptor + backing numpy arrays passed as `pClientData`) is genuinely load-bearing —
  dropping it is a real use-after-free. Distinguish these precisely.
- **Struct-by-value ABI.** Win64 (1/2/4/8-byte structs in registers, else by pointer), SysV x86-64
  (eightbyte classification, SSE-wins-per-eightbyte, boundary-spanning fields, 16B INT/INT repack for
  llvmlite#300), AAPCS64. `>16B` returns via `sret` (stack-alloca hidden-arg) — but `Record` LARGE
  returns are rejected (RecordModel uses raw `[N x i8]*`; stack-alloca sret would dangle). Check
  classification edge cases and that the right path is chosen per platform.
- **Pointer / buffer correctness.** `array_data_p` (signed intp), unaligned access: `load_unaligned`/
  `store_unaligned` are align-1 and required for misaligned addresses — using aligned `load_at`/
  `store_at` on a misaligned pointer is **UB**. Check pointer signedness, bounds, lifetime of pointers
  into Python objects (unicode data payload, `_sqlite3.so` symbol addresses), and the macOS dead-code-
  elimination footgun (raw-pointer result stores get dropped on macOS-arm64 unless written through a
  numba-tracked view like `out.view(np.uint8)`).
- **Symbol resolution / `cache=True` correctness.** Baking a literal runtime address
  (`ll.address_of_symbol`) into IR breaks `cache=True` (ASLR randomizes per process). The correct
  pattern is an extern declaration resolved at link time. On macOS the dyld shared cache wins over
  `RTLD_GLOBAL`; `add_symbol` (ExplicitSymbols) is the only override.
- **NUL / fixed-width buffer bridges.** Faithful numpy->SQLite contract is **trim trailing NULs only,
  preserve interior** (BLOB and TEXT carry exact byte length; first-NUL truncation silently loses
  data). `sqlite3_result_double` coerces **NaN -> SQL NULL** (±inf pass through) — a float result
  path cannot uphold a "no NULL" invariant.
- **`removerefctpass` dependence.** Code that relies on numba's `removerefctpass` symmetric stripping
  (e.g. inlined `_incref_meminfo` so a manual incref survives) — note that this pass is **removed in
  numba main (queued for 0.66)**; it still exists in the pinned 0.65.1. Flag load-bearing dependence
  as a forward-compat risk (medium), not a current bug.

## Known-correct patterns — do NOT report these as bugs (unless you can show the rationale fails here)

- Zeroing the `sqlite3_vtab` base (pModule/nRef/zErrMsg) in xCreate/xConnect — the SQLite **core**
  owns and populates those; setting them is wrong. A "pModule not set -> null deref" claim is a known
  **false positive**.
- Absence of a keep-alive handle for a registered cfunc (see lifetime note above).

These are point-in-time domain notes — **verify the current code's actual behavior** before relying on
any of them; the code may have changed. Use them to know where to look, not what to conclude.
