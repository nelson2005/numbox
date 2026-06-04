# SQLite phase-5 review-fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every confirmed finding from the 2026-06-04 deep-dive review of the phase-5 SQLite work (query_to_array + vtable pushdown / TVF / xDestroy) — silent-wrong-result and UB edge cases — with a regression test for each, on the local-only branch `feat/sqlite-query-vtable-phase5` before it is pushed/PR'd.

**Architecture:** Surgical edits to the seven phase-5 production modules plus one new helper (`store_unaligned`) in `numbox/utils/lowlevel.py`. The throughline of every bug is *untested input shapes* — large-magnitude integers, packed/misaligned dtypes, and non-contiguous / non-numeric inputs — so every task is test-first (TDD): write the failing test that exercises the untested shape, then make it pass. No behavior changes for the already-passing shapes.

**Tech Stack:** numba 0.65.1 (matrix 0.60.0–0.65.1), numpy 2.4.5, SQLite C-API via numbox bindings, pytest.

**Branch:** `feat/sqlite-query-vtable-phase5` (already checked out; local-only). Do all work here; cherry-pick to an `upstream-pr/` branch is a later, separate step outside this plan.

---

## Running tests

Per the project rule "clean caches before every pytest run", precede **every** pytest invocation with this cache clear (numba's `co_code`/`cvar` cache can mask layout/cache regressions). Shorthand `<CLEAN>` below means:

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil,pathlib,os;[shutil.rmtree(p,ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')];nb=pathlib.Path(os.path.expanduser('~/.cache/numba'));(shutil.rmtree(nb,ignore_errors=True) if nb.exists() else None)"
```

Always use the venv interpreter: `/home/erik/projects/numbox/venv/bin/pytest` and `/home/erik/projects/numbox/venv/bin/flake8`. Never bare `pytest`/`python`. Use `git -C /home/erik/projects/numbox` for git; never `cd`.

## Design decisions (settled — rationale for the executor)

- **H1 + M4 unified → exact integer-domain pushdown.** Rather than dropping range pushdown on int64/uint64 (which would gut the feature for the default integer width), the cursor compares integer columns in the **int64 domain** (decode the bound via `sqlite3_value_int64`, read the cell as int64). uint64 is read as `int64(uint64)` — the same wrap `_xcolumn` already surfaces to SQLite — so the cursor pre-filter and SQLite's `omit=0` re-check agree (fixes M4 too). Float columns keep the float64 path (exact for f32/f64).
- **M2 → `store_unaligned`.** Add an `align=1` store mirroring the existing `load_unaligned`; the read path is already unaligned, only the write path was aligned.
- **M3 → store logical pointer + stride.** Mirror `register_table`: pin via `structref_meminfo` (keep-alive) but address rows via `array_data_p(result)` + `result.strides[0]`. Handles offset/strided/Fortran 1-D returns without `np.ascontiguousarray` (whose numba record-array support is uncertain).
- **L3 → document the contract** (a wrong *dtype* return cannot be checked at codegen time and the `@cfunc` boundary would swallow a runtime assert anyway); M3 already makes offset/strided returns correct.

---

### Task 0: Add `store_unaligned` to lowlevel.py

**Goal:** Provide an `align=1` store primitive mirroring the existing `load_unaligned`, so structured-dtype field writes at misaligned offsets are well-defined.

**Files:**
- Modify: `numbox/utils/lowlevel.py` (add `_store_unaligned` intrinsic + `store_unaligned` njit wrapper next to `store_at`, ~lines 82–107)
- Test: `test/core/test_load_unaligned.py` (append)

**Acceptance Criteria:**
- [ ] `store_unaligned(p, v)` writes `v` at raw pointer `p` using `builder.store(v, ptr, align=1)`
- [ ] A value written by `store_unaligned` at a deliberately misaligned address round-trips via `load_unaligned`
- [ ] flake8 clean

**Verify:** `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_load_unaligned.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing test** — append to `test/core/test_load_unaligned.py`:

```python
def test_store_unaligned_roundtrip():
    import numpy as np
    from numba import njit
    from numba.core.types import int64, float64
    from numbox.utils.lowlevel import store_unaligned, load_unaligned, array_data_p

    @njit
    def rw(buf_p):
        # write an i8 and an f8 at byte offsets 1 and 9 (both misaligned)
        store_unaligned(buf_p + 1, int64(0x0102030405060708))
        store_unaligned(buf_p + 9, float64(3.5))
        a = load_unaligned(buf_p + 1, int64)
        b = load_unaligned(buf_p + 9, float64)
        return a, b

    buf = np.zeros(32, dtype=np.uint8)
    a, b = rw(array_data_p(buf))
    assert a == 0x0102030405060708
    assert b == 3.5
```

- [ ] **Step 2: Run to confirm it fails** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_load_unaligned.py::test_store_unaligned_roundtrip -q` → FAIL (`ImportError: cannot import name 'store_unaligned'`)

- [ ] **Step 3: Implement** — in `numbox/utils/lowlevel.py`, immediately after the `store_at` njit wrapper (after line 106), add:

```python
@intrinsic
def _store_unaligned(typingctx: Context, p_ty, v_ty):
    if unliteral(p_ty) not in (intp, uintp):
        raise TypingError(
            f"store_unaligned: pointer argument must be intp or uintp, got {p_ty}"
        )
    sig = void(p_ty, v_ty)

    def codegen(context: BaseContext, builder, signature, args):
        p, v = args
        ty_ll = context.get_data_type(v_ty)
        ptr = builder.inttoptr(p, ty_ll.as_pointer())
        builder.store(v, ptr, align=1)
    return sig, codegen


@njit(**jit_options)
def store_unaligned(p, v):
    """Like :func:`store_at` but emits an ``align=1`` store, legal on a misaligned
    address (e.g. a packed structured-dtype field) where ``store_at`` is UB."""
    return _store_unaligned(p, v)
```

- [ ] **Step 4: Run to confirm it passes** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_load_unaligned.py -q` → PASS

- [ ] **Step 5: flake8 + commit**

```bash
/home/erik/projects/numbox/venv/bin/flake8 numbox/utils/lowlevel.py test/core/test_load_unaligned.py --max-line-length=127
git -C /home/erik/projects/numbox add numbox/utils/lowlevel.py test/core/test_load_unaligned.py
git -C /home/erik/projects/numbox commit -m "feat(lowlevel): add store_unaligned (align=1 store) mirroring load_unaligned"
```

---

### Task 1: query_to_array — surface mid-step SQLite errors (M1)

**Goal:** `query_to_array` raises on a `sqlite3_step` error mid-iteration instead of silently returning a truncated array.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_query.py` (`_query_core` return value + step loop ~88–117; `query_to_array` ~143–150; imports line 11)
- Test: `test/core/test_sqlite_query.py` (append)

**Acceptance Criteria:**
- [ ] `_query_core` returns `(array, terminal_rc)` where `terminal_rc` is the step code that ended the loop
- [ ] `query_to_array` raises `RuntimeError` (via `_raise_rc`) when the terminal rc is not `SQLITE_DONE`
- [ ] A clean query still returns its array; an empty result returns a length-0 array
- [ ] A query that errors after emitting ≥1 row raises rather than truncating

**Verify:** `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_query.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `test/core/test_sqlite_query.py`:

```python
def test_query_empty_result():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(i INTEGER)")
    with c_string("SELECT i FROM t") as sql:
        out = query_to_array(db, sql, np.dtype([("i", "i8")]))
    assert out.shape == (0,)
    sqlite3_close(db)


def test_query_step_error_raises():
    # A read-only vtable whose xColumn raises mid-scan makes sqlite3_step return
    # SQLITE_ERROR after the first row; query_to_array must raise, not truncate.
    from numbox.core.bindings import register_table
    db = _open_mem()
    # Force a step-time failure with a deterministic builtin: abort() raises
    # SQLITE_ERROR at step time, after prepare succeeds.
    _exec(db, "CREATE TABLE t(i INTEGER)")
    _exec(db, "INSERT INTO t VALUES (1), (2), (3)")
    with c_string("SELECT abort() FROM t") as sql:
        with pytest.raises(RuntimeError) as excinfo:
            query_to_array(db, sql, np.dtype([("i", "i8")]))
    assert "query_to_array failed" in str(excinfo.value)
    sqlite3_close(db)
```

> Note: `abort()` is not a builtin SQLite function. Use a guaranteed step-time error instead: a `CABS`-free path. Replace the SQL in `test_query_step_error_raises` with a query that errors at step time — `"SELECT i FROM t WHERE i = (SELECT i FROM t)"` raises `SQLITE_ERROR` ("more than one row returned by a subquery") at step time after prepare succeeds. Verify the chosen SQL actually fails at *step* (not prepare) on the venv sqlite before finalizing the test; if prepare catches it, fall back to registering a numbox vtable whose `_xcolumn` is made to raise, or to a `randomblob`/`zeroblob` size-overflow (`SELECT zeroblob(2000000000)||zeroblob(2000000000) FROM t`) which raises `SQLITE_TOOBIG` at step.

- [ ] **Step 2: Run to confirm `test_query_step_error_raises` fails** (returns truncated array, no raise) — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_query.py::test_query_step_error_raises -q` → FAIL

- [ ] **Step 3: Implement** — in `numbox/core/bindings/_sqlite_query.py`:

Change the import on line 11 to add `SQLITE_DONE`:
```python
from numbox.core.bindings._sqlite_constants import SQLITE_ROW, SQLITE_NULL, SQLITE_OK, SQLITE_DONE
```

Rewrite `_query_core`'s loop and return to capture the terminal rc (replace the `while ...` loop header and the `return res` tail):
```python
@njit(**jit_options)
def _query_core(stmt, ncols, offsets, tags, widths, itemsize, dt):
    """Step ``stmt`` to exhaustion, materialising rows into an NRT buffer that
    grows geometrically, then trim to the exact length. Returns ``(array, rc)``
    where ``rc`` is the terminal step return code (SQLITE_DONE on success)."""
    cap = 16
    out = np.empty(cap, dt)
    n = 0
    rc = sqlite3_step(stmt)
    while rc == SQLITE_ROW:
        if n == cap:
            cap = cap * 2
            new = np.empty(cap, dt)
            old_bytes = carray(_cast_int_to_void_p(array_data_p(out)), (n * itemsize,), dtype=np.uint8)
            new_bytes = carray(_cast_int_to_void_p(array_data_p(new)), (n * itemsize,), dtype=np.uint8)
            for b in range(n * itemsize):
                new_bytes[b] = old_bytes[b]
            out = new
        base = array_data_p(out) + n * itemsize
        row = carray(_cast_int_to_void_p(base), (itemsize,), dtype=np.uint8)
        for b in range(itemsize):
            row[b] = 0
        for j in range(ncols):
            _store_cell(base, offsets[j], tags[j], widths[j], stmt, j)
        n += 1
        rc = sqlite3_step(stmt)
    res = np.empty(n, dt)
    src_bytes = carray(_cast_int_to_void_p(array_data_p(out)), (n * itemsize,), dtype=np.uint8)
    res_bytes = carray(_cast_int_to_void_p(array_data_p(res)), (n * itemsize,), dtype=np.uint8)
    for b in range(n * itemsize):
        res_bytes[b] = src_bytes[b]
    return res, rc
```

Update `query_to_array`'s try-block to consume the tuple and raise on a non-DONE terminal rc (replace lines 143–150):
```python
    try:
        ncols = sqlite3_column_count(stmt.value)
        if ncols != len(names):
            raise ValueError("dtype has %d fields but query returns %d columns" % (len(names), ncols))
        res, last_rc = _query_core(stmt.value, ncols, offsets, tags, widths,
                                   int(dtype.itemsize), numba.from_dtype(dtype))
        if last_rc != SQLITE_DONE:
            _raise_rc(db, last_rc)
        return res
    finally:
        sqlite3_finalize(stmt.value)
```

- [ ] **Step 4: Run to confirm pass** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_query.py -q` → PASS

- [ ] **Step 5: flake8 + commit**

```bash
/home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_sqlite_query.py test/core/test_sqlite_query.py --max-line-length=127
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_query.py test/core/test_sqlite_query.py
git -C /home/erik/projects/numbox commit -m "fix(sqlite-query): raise on mid-iteration step error instead of silently truncating"
```

---

### Task 2: query_to_array — unaligned field stores + column-accessor order + docstring (M2a, nits)

**Goal:** `_store_cell` writes structured-dtype fields with `store_unaligned` (well-defined for packed dtypes), calls the format accessor before `column_bytes`, and the docstring states `sql_p` is a char* pointer.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_query.py` (`_store_cell` ~28–85; imports line 23; `query_to_array` docstring ~128–131)
- Test: `test/core/test_sqlite_query.py` (append)

**Acceptance Criteria:**
- [ ] `_store_cell`'s multi-byte numeric and float-NaN writes use `store_unaligned`
- [ ] For `_TAG_S`/`_TAG_BLOB`, `sqlite3_column_text`/`sqlite3_column_blob` is evaluated before `sqlite3_column_bytes`
- [ ] A packed/misaligned structured dtype (e.g. `[('flag','i1'),('x','f8'),('s','U4')]`) round-trips correctly
- [ ] A BLOB column round-trips into an `'S'` field
- [ ] Docstring documents `sql_p` as a pointer to NUL-terminated SQL text

**Verify:** `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_query.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `test/core/test_sqlite_query.py`:

```python
def test_query_packed_misaligned_dtype():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(flag INTEGER, x REAL, s TEXT)")
    _exec(db, "INSERT INTO t VALUES (1, 2.5, 'ok'), (0, -3.25, 'hi')")
    dt = np.dtype([("flag", "i1"), ("x", "f8"), ("s", "U4")])  # x at offset 1, s at offset 9
    assert not dt.isalignedstruct
    with c_string("SELECT flag, x, s FROM t") as sql:
        out = query_to_array(db, sql, dt)
    assert list(out["flag"]) == [1, 0]
    assert list(out["x"]) == [2.5, -3.25]
    assert list(out["s"]) == ["ok", "hi"]
    sqlite3_close(db)


def test_query_blob_into_S_field():
    db = _open_mem()
    _exec(db, "CREATE TABLE t(b BLOB)")
    _exec(db, "INSERT INTO t VALUES (x'00ff10'), (x'4142')")
    with c_string("SELECT b FROM t") as sql:
        out = query_to_array(db, sql, np.dtype([("b", "S3")]))
    # interior bytes preserved up to field width; trailing NUL pad trimmed on read
    assert out["b"][0] == b"\x00\xff\x10"
    assert out["b"][1] == b"AB"
    sqlite3_close(db)
```

- [ ] **Step 2: Run to confirm `test_query_packed_misaligned_dtype` fails** on a strict-alignment build, or at minimum that the new dtype path is exercised. On x86 it may pass even when buggy (aligned store tolerated) — note this in the commit and rely on the ubuntu-arm CI leg. Run: `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_query.py::test_query_packed_misaligned_dtype test/core/test_sqlite_query.py::test_query_blob_into_S_field -q`

- [ ] **Step 3: Implement** — in `numbox/core/bindings/_sqlite_query.py`:

Add `store_unaligned` to the lowlevel import (line 23):
```python
from numbox.utils.lowlevel import _cast_int_to_void_p, array_data_p, store_unaligned
```

Replace every `store_at(addr, ...)` call in `_store_cell` with `store_unaligned(addr, ...)` (the NULL-NaN writes and all numeric writes, lines 40–65). The byte-wise `_TAG_S`/`_TAG_BLOB`/`_TAG_U` paths already use `uint8`/`utf8_to_utf32` and need no store change here.

Reorder the `_TAG_S` and `_TAG_BLOB` branches so the format accessor is called before `column_bytes` (replace lines 68–85):
```python
    elif tag == _TAG_S:
        src_p = sqlite3_column_text(stmt, j)
        nbytes = sqlite3_column_bytes(stmt, j)
        src = carray(_cast_int_to_void_p(src_p), (nbytes,), dtype=np.uint8)
        dst = carray(_cast_int_to_void_p(addr), (width,), dtype=np.uint8)
        n = nbytes if nbytes < width else width
        for b in range(n):
            dst[b] = src[b]
        for b in range(n, width):
            dst[b] = 0
    elif tag == _TAG_BLOB:
        src_p = sqlite3_column_blob(stmt, j)
        nbytes = sqlite3_column_bytes(stmt, j)
        src = carray(_cast_int_to_void_p(src_p), (nbytes,), dtype=np.uint8)
        dst = carray(_cast_int_to_void_p(addr), (width,), dtype=np.uint8)
        n = nbytes if nbytes < width else width
        for b in range(n):
            dst[b] = src[b]
        for b in range(n, width):
            dst[b] = 0
```

Update the `query_to_array` docstring (lines 128–131) to:
```python
    """Run the NUL-terminated SQL text at pointer ``sql_p`` on ``db`` and return
    its rows as a 1-D numpy structured array of ``dtype`` (one field per result
    column, by position). ``sql_p`` is a char* pointer (e.g. from
    ``numbox.utils.cstrings.c_string`` or ``get_unicode_data_p``), not a Python
    str. NULL -> NaN (float) / 0 (int) / empty (text/blob)."""
```

- [ ] **Step 4: Run to confirm pass** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_query.py -q` → PASS

- [ ] **Step 5: flake8 + commit**

```bash
/home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_sqlite_query.py test/core/test_sqlite_query.py --max-line-length=127
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_query.py test/core/test_sqlite_query.py
git -C /home/erik/projects/numbox commit -m "fix(sqlite-query): unaligned field stores for packed dtypes; column accessor before column_bytes; docstring"
```

---

### Task 3: typemap — unaligned UTF-32 writes + decode validation (M2b, L2)

**Goal:** `utf8_to_utf32` writes code points with `store_unaligned` (safe for misaligned `'U'` fields) and validates continuation bytes / rejects surrogate & overlong encodings (→ U+FFFD).

**Files:**
- Modify: `numbox/core/bindings/_sqlite_typemap.py` (`utf8_to_utf32` ~64–95; imports line 13)
- Test: `test/core/test_sqlite_query.py` (append — the decode helper lives there)

**Acceptance Criteria:**
- [ ] `utf8_to_utf32` writes each `uint32` code point via `store_unaligned(dst + 4*k, uint32(cp))` and zero-pads via `store_unaligned`
- [ ] A `'U'` field at a misaligned offset round-trips (covered by Task 2's packed-dtype test once this lands)
- [ ] Malformed UTF-8 (bad continuation byte, surrogate, overlong) decodes to U+FFFD (0xFFFD) rather than a wrong code point
- [ ] Existing `utf8_to_utf32` tests still pass

**Verify:** `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_query.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `test/core/test_sqlite_query.py` (reuses the module's `_decode` helper):

```python
def test_utf8_to_utf32_bad_continuation_is_replacement():
    # 0xE0 starts a 3-byte sequence but the next byte 0x20 is not a continuation
    n, dst = _decode(b"\xe0\x20\x41", 4)
    assert int(dst[0]) == 0xFFFD


def test_utf8_to_utf32_surrogate_is_replacement():
    # CESU-8-style encoding of a high surrogate U+D800 (0xED 0xA0 0x80) is illegal
    n, dst = _decode(b"\xed\xa0\x80", 4)
    assert int(dst[0]) == 0xFFFD


def test_utf8_to_utf32_overlong_is_replacement():
    # overlong 2-byte encoding of '/' (0xC0 0xAF) is illegal
    n, dst = _decode(b"\xc0\xaf", 4)
    assert int(dst[0]) == 0xFFFD
```

- [ ] **Step 2: Run to confirm failure** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_query.py -k "bad_continuation or surrogate or overlong" -q` → FAIL

- [ ] **Step 3: Implement** — in `numbox/core/bindings/_sqlite_typemap.py`:

Add `store_unaligned` to the lowlevel import (line 13):
```python
from numbox.utils.lowlevel import _cast_int_to_void_p, load_unaligned, store_unaligned
```

Replace `utf8_to_utf32` (lines 64–95) with a version that uses `store_unaligned`, validates continuation bytes, and rejects surrogate/overlong:
```python
@njit(**jit_options)
def utf8_to_utf32(src, nbytes, dst, width_cp):
    """Decode the UTF-8 bytes at ``src`` (length ``nbytes``) into up to
    ``width_cp`` little-endian uint32 code points at ``dst``; NUL-pad the
    remainder. Malformed input (bad continuation byte, surrogate, overlong
    encoding, out-of-range) decodes to U+FFFD. Returns the number of code points
    written. ``dst`` may be misaligned (writes are align=1)."""
    inp = carray(_cast_int_to_void_p(src), (nbytes,), dtype=np.uint8)
    for k in range(width_cp):
        store_unaligned(dst + 4 * k, uint32(0))
    i = 0
    k = 0
    while i < nbytes and k < width_cp:
        b0 = uint32(inp[i])
        if b0 < 0x80:
            cp = b0
            i += 1
        elif b0 >> 5 == 0x6 and i + 1 < nbytes and (inp[i + 1] >> 6) == 0x2:
            cp = ((b0 & 0x1F) << 6) | (uint32(inp[i + 1]) & 0x3F)
            if cp < 0x80:
                cp = 0xFFFD
            i += 2
        elif b0 >> 4 == 0xE and i + 2 < nbytes and (inp[i + 1] >> 6) == 0x2 and (inp[i + 2] >> 6) == 0x2:
            cp = ((b0 & 0x0F) << 12) | ((uint32(inp[i + 1]) & 0x3F) << 6) | (uint32(inp[i + 2]) & 0x3F)
            if cp < 0x800 or (0xD800 <= cp <= 0xDFFF):
                cp = 0xFFFD
            i += 3
        elif (b0 >> 3 == 0x1E and i + 3 < nbytes and (inp[i + 1] >> 6) == 0x2
              and (inp[i + 2] >> 6) == 0x2 and (inp[i + 3] >> 6) == 0x2):
            cp = (((b0 & 0x07) << 18) | ((uint32(inp[i + 1]) & 0x3F) << 12)
                  | ((uint32(inp[i + 2]) & 0x3F) << 6) | (uint32(inp[i + 3]) & 0x3F))
            if cp < 0x10000 or cp > 0x10FFFF:
                cp = 0xFFFD
            i += 4
        else:
            cp = 0xFFFD
            i += 1
        store_unaligned(dst + 4 * k, uint32(cp))
        k += 1
    return k
```

- [ ] **Step 4: Run to confirm pass** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_query.py -q` → PASS (existing utf8 tests + new ones)

- [ ] **Step 5: flake8 + commit**

```bash
/home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_sqlite_typemap.py test/core/test_sqlite_query.py --max-line-length=127
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_typemap.py test/core/test_sqlite_query.py
git -C /home/erik/projects/numbox commit -m "fix(sqlite-typemap): unaligned UTF-32 stores + validate UTF-8 continuation/surrogate/overlong"
```

---

### Task 4: vtable — exact integer-domain pushdown (H1 + M4)

**Goal:** The read-only vtable's range/EQ pushdown compares integer columns in the int64 domain (no float64 precision loss; uint64 read as the same wrapped int64 `_xcolumn` surfaces), so large-magnitude integer filters return correct rows.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_vtable.py` (`_PRED_DTYPE` ~114; imports ~29–34; add `_cell_value_i64`; `_row_matches` ~405–428; `_xfilter` ~439–466)
- Test: `test/core/test_sqlite_vtable.py` (append)

**Acceptance Criteria:**
- [ ] `_PRED_DTYPE` carries `is_int` (i4), `ival` (i8), and `fval` (f8)
- [ ] `_xfilter` decodes integer-tagged columns via `sqlite3_value_int64` and float-tagged via `sqlite3_value_double`, setting `is_int`
- [ ] `_cell_value_i64` returns each integer cell as int64 (uint64 wrapped, matching `_xcolumn`)
- [ ] `_row_matches` compares integer columns exactly (int64) and float columns as float64
- [ ] An int64 column with a value at 2**53+1 returns correctly under a `> 2**53` range query (parity with full scan)
- [ ] A uint64 column with a value ≥ 2**63 filters consistently with what `_xcolumn` surfaces
- [ ] All existing pushdown tests still pass

**Verify:** `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_vtable.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `test/core/test_sqlite_vtable.py`:

```python
def test_pushdown_int64_above_2_53_range():
    db = _open_memory()
    base = 1 << 53
    vals = [base, base + 1, base + 2, base + 3]
    a = np.array([[v] for v in vals], dtype=np.int64)
    h = register_table(db, "t", a, columns=["c"])  # noqa: F841
    for op in (">", ">=", "<", "<="):
        got = sorted(_select_col0(db, "SELECT c FROM t WHERE c %s %d" % (op, base + 1)))
        exp = sorted(v for v in vals if eval("v %s (base + 1)" % op))
        assert got == exp, (op, got, exp)
    # the exact row that float64 collapse used to drop:
    assert _select_col0(db, "SELECT c FROM t WHERE c > %d" % base) == sorted(vals[1:])
    sqlite3_close(db)


def test_pushdown_uint64_high_magnitude_consistent():
    db = _open_memory()
    vals = [(1 << 63), (1 << 63) + 5, (1 << 63) + 1]
    a = np.array([[v] for v in vals], dtype=np.uint64)
    h = register_table(db, "t", a, columns=["c"])  # noqa: F841
    # xColumn surfaces uint64 as wrapped int64; the cursor must agree, so a
    # pushdown query returns exactly what a full scan + SQLite re-check returns.
    pushed = sorted(_select_col0(db, "SELECT c FROM t WHERE c > %d" % ((1 << 63) + 0)))
    allrows = sorted(_select_col0(db, "SELECT c FROM t"))
    full = sorted(v for v in allrows if v > ((1 << 63) + 0))
    assert pushed == full, (pushed, full)
    sqlite3_close(db)
```

> Note: the `eval` in the first test computes the Python-int expected set; `base` is in scope. If the project lint forbids `eval`, expand the four comparisons explicitly (`{">" : lambda v: v > base+1, ...}` as in `test_pushdown_range_matches_fullscan`). Prefer the explicit-dict form to match the existing test style and avoid `eval`.

- [ ] **Step 2: Run to confirm `test_pushdown_int64_above_2_53_range` fails** (the `> base` row at 2**53+1 is dropped) — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_vtable.py::test_pushdown_int64_above_2_53_range -q` → FAIL

- [ ] **Step 3: Implement** — in `numbox/core/bindings/_sqlite_vtable.py`:

Add `sqlite3_value_int64` to the bindings import (the block at lines 29–34):
```python
from numbox.core.bindings import (
    sqlite3_errmsg, sqlite3_free, sqlite3_malloc,
    sqlite3_result_int64, sqlite3_result_double,
    sqlite3_result_text, sqlite3_result_blob, sqlite3_result_error,
    sqlite3_value_double, sqlite3_value_int64,
)
```

Widen `_PRED_DTYPE` (line 114) to carry both representations + the discriminator:
```python
_PRED_DTYPE = np.dtype([("col", "i4"), ("op", "i4"), ("is_int", "i4"),
                        ("ival", "i8"), ("fval", "f8")], align=True)
_PRED_SIZE = _PRED_DTYPE.itemsize
```

Add `_cell_value_i64` immediately after `_cell_value` (after line 402). It mirrors `_cell_value`'s addressing but returns int64 (uint64 wrapped to match `_xcolumn`):
```python
@njit(**jit_options)
def _cell_value_i64(d, rowid, col):
    ncols = d[0].ncols
    base = d[0].data_base
    row_stride = d[0].row_stride
    offsets = carray(_cast_int_to_void_p(d[0].col_offsets), (ncols,), dtype=np.int64)
    tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=np.int32)
    addr = base + rowid * row_stride + offsets[col]
    tag = tags[col]
    if tag == _TAG_I8:
        return int64(load_unaligned(addr, int8))
    elif tag == _TAG_I16:
        return int64(load_unaligned(addr, int16))
    elif tag == _TAG_I32:
        return int64(load_unaligned(addr, int32))
    elif tag == _TAG_I64:
        return load_unaligned(addr, int64)
    elif tag == _TAG_U8:
        return int64(load_unaligned(addr, uint8))
    elif tag == _TAG_U16:
        return int64(load_unaligned(addr, uint16))
    elif tag == _TAG_U32:
        return int64(load_unaligned(addr, uint32))
    elif tag == _TAG_U64:
        return int64(load_unaligned(addr, uint64))
    return int64(0)
```

Add an integer-tag predicate next to `_is_numeric_tag` (after line 258):
```python
@njit(**jit_options)
def _is_int_tag(tag):
    return tag <= _TAG_U64
```

Rewrite `_row_matches` (lines 405–428) to branch on `is_int`:
```python
@njit(**jit_options)
def _row_matches(cur):
    c = carray(_cast_int_to_void_p(cur), (1,), dtype=_CUR_DTYPE)
    if c[0].n_pred == 0:
        return True
    d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_DESC_DTYPE)
    preds = carray(_cast_int_to_void_p(c[0].pred_p), (c[0].n_pred,), dtype=_PRED_DTYPE)
    for k in range(c[0].n_pred):
        op = preds[k].op
        if preds[k].is_int != 0:
            cv = _cell_value_i64(d, c[0].rowid, preds[k].col)
            rv = preds[k].ival
        else:
            cv = _cell_value(d, c[0].rowid, preds[k].col)
            rv = preds[k].fval
        if op == SQLITE_INDEX_CONSTRAINT_EQ:
            ok = cv == rv
        elif op == SQLITE_INDEX_CONSTRAINT_GT:
            ok = cv > rv
        elif op == SQLITE_INDEX_CONSTRAINT_GE:
            ok = cv >= rv
        elif op == SQLITE_INDEX_CONSTRAINT_LT:
            ok = cv < rv
        else:
            ok = cv <= rv
        if not ok:
            return False
    return True
```

In `_xfilter` (lines 439–466), look up each bound column's tag and decode in its native domain. Replace the per-slot fill loop (the block that currently does `preds[k].col = ...; preds[k].op = ...; preds[k].val = sqlite3_value_double(vals[k])`) with:
```python
            d = carray(_cast_int_to_void_p(c[0].descriptor), (1,), dtype=_DESC_DTYPE)
            ncols = d[0].ncols
            col_tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=np.int32)
            for k in range(argc):
                col = spec[2 * k]
                preds[k].col = col
                preds[k].op = spec[2 * k + 1]
                if _is_int_tag(col_tags[col]):
                    preds[k].is_int = 1
                    preds[k].ival = sqlite3_value_int64(vals[k])
                    preds[k].fval = 0.0
                else:
                    preds[k].is_int = 0
                    preds[k].ival = 0
                    preds[k].fval = sqlite3_value_double(vals[k])
```

- [ ] **Step 4: Run to confirm pass** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_vtable.py -q` → PASS

- [ ] **Step 5: flake8 + commit**

```bash
/home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_sqlite_vtable.py test/core/test_sqlite_vtable.py --max-line-length=127
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_vtable.py test/core/test_sqlite_vtable.py
git -C /home/erik/projects/numbox commit -m "fix(sqlite-vtable): compare integer pushdown predicates in int64 domain (no float64 precision loss; uint64 consistent with xColumn)"
```

---

### Task 5: vtable — guard the idxStr allocation; document the omit invariant (L1, nit)

**Goal:** `_xbestindex` cannot leak its `sqlite3_malloc`'d `idxStr` if a `carray`/decode raises, and the `omit=0` correctness dependency is documented.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_vtable.py` (`_xbestindex` ~268–311)
- Test: `test/core/test_sqlite_vtable.py` (existing pushdown tests are the regression guard; no new behavior to assert beyond "still correct")

**Acceptance Criteria:**
- [ ] `_xbestindex` body is wrapped so any exception frees `idx_p` and returns `SQLITE_ERROR`
- [ ] `idx_p` is zeroed in the local after being handed to SQLite (`needToFreeIdxStr=1`) so the handler never double-frees SQLite-owned memory
- [ ] A code comment at `usage[i].omit = 0` records that omit MUST stay 0 (correctness of integer/float pushdown depends on SQLite re-checking surfaced rows)
- [ ] All existing pushdown tests still pass

**Verify:** `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_vtable.py -q` → all pass

**Steps:**

- [ ] **Step 1: Implement** — in `numbox/core/bindings/_sqlite_vtable.py`, rewrite `_xbestindex` (lines 268–311) wrapping the allocating body in try/except and zeroing `idx_p` after handoff. Keep the existing logic; only add the guard, the post-handoff zero, and the omit comment:

```python
@cfunc(types.int32(types.intp, types.intp), cache=_CACHE)
def _xbestindex(vtab, idx_info):
    idx_p = 0
    try:
        v = carray(_cast_int_to_void_p(vtab), (1,), dtype=_VTAB_DTYPE)
        d = carray(_cast_int_to_void_p(v[0].descriptor), (1,), dtype=_DESC_DTYPE)
        ii = carray(_cast_int_to_void_p(idx_info), (1,), dtype=_IDX_INFO_DTYPE)
        ncols = d[0].ncols
        tags = carray(_cast_int_to_void_p(d[0].col_tags), (ncols,), dtype=np.int32)
        n_constraint = ii[0].nConstraint
        cons = carray(_cast_int_to_void_p(ii[0].aConstraint), (n_constraint,), dtype=_CONSTRAINT_DTYPE)
        usage = carray(_cast_int_to_void_p(ii[0].aConstraintUsage), (n_constraint,), dtype=_USAGE_DTYPE)

        idx_p = sqlite3_malloc(int32(n_constraint * 8)) if n_constraint > 0 else 0
        if n_constraint > 0 and idx_p == 0:
            return SQLITE_NOMEM
        spec = carray(_cast_int_to_void_p(idx_p), (2 * n_constraint,), dtype=np.int32)

        nbound = 0
        for i in range(n_constraint):
            col = cons[i].iColumn
            op = cons[i].op
            if cons[i].usable != 0 and _is_supported_op(op) and 0 <= col < ncols and _is_numeric_tag(tags[col]):
                usage[i].argvIndex = int32(nbound + 1)
                # omit MUST stay 0: SQLite re-checks every surfaced row, which is
                # the correctness net for the cursor's pruning. _row_matches now
                # compares exactly (int64 for integer columns), so pruning is
                # precise -- but keep omit=0 so any future predicate widening
                # cannot silently leak/drop rows.
                usage[i].omit = 0
                spec[2 * nbound] = int32(col)
                spec[2 * nbound + 1] = int32(op)
                nbound += 1

        ii[0].idxNum = int32(nbound)
        if nbound > 0:
            ii[0].idxStr = idx_p
            ii[0].needToFreeIdxStr = int32(1)
            idx_p = 0  # SQLite owns it now; the except handler must not free it
        else:
            sqlite3_free(idx_p)
            idx_p = 0

        nrows = d[0].nrows
        ii[0].estimatedRows = nrows if nbound == 0 else nrows // (nbound + 1) + 1
        ii[0].estimatedCost = float64(nrows)
        return SQLITE_OK
    except Exception:
        sqlite3_free(idx_p)
        return SQLITE_ERROR
```

- [ ] **Step 2: Run to confirm pass** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_vtable.py -q` → PASS

- [ ] **Step 3: flake8 + commit**

```bash
/home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_sqlite_vtable.py --max-line-length=127
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_vtable.py
git -C /home/erik/projects/numbox commit -m "fix(sqlite-vtable): guard xBestIndex idxStr against leak on exception; document omit=0 invariant"
```

---

### Task 6: tvf — stride/offset-correct result addressing; document return contract (M3, L3)

**Goal:** A TVF whose `fn` returns a sliced / strided / offset 1-D structured array is read correctly (logical data pointer + real row stride), not silently misread; the `register_tvf` docstring states the return contract.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_tvf.py` (`_TVF_CUR_DTYPE` ~90–94; `_XFILTER_SRC` ~110–130; `_tvf_xcolumn` addr math ~303; `register_tvf` docstring ~417–434)
- Test: `test/core/test_sqlite_tvf.py` (append)

**Acceptance Criteria:**
- [ ] `_TVF_CUR_DTYPE` has a `row_stride` (i8) field
- [ ] `_tvf_xfilter_impl` pins via `structref_meminfo` (keep-alive) but stores `data_p = array_data_p(result)` and `row_stride = result.strides[0]`
- [ ] `_tvf_xcolumn` addresses rows as `data_p + rowid * row_stride + offsets[j]`
- [ ] A `fn` returning `out[1:]` (offset) and `out[::2]` (strided) yield correct rows
- [ ] A `fn` returning a 0-row array yields no rows (NULL/empty guard intact)
- [ ] `register_tvf` docstring states `fn` must return a 1-D structured array of `out_dtype`

**Verify:** `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_tvf.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `test/core/test_sqlite_tvf.py`:

```python
@njit
def _series_sliced(start, stop):
    # returns a non-contiguous / offset view, not a fresh contiguous array
    out = np.empty((stop - start) + 1, _OUT)
    for i in range((stop - start) + 1):
        out[i].n = (start - 1) + i
    return out[1:]  # offset slice: logical start != allocation base


@njit
def _series_strided(start, stop):
    out = np.empty(2 * (stop - start), _OUT)
    for i in range(2 * (stop - start)):
        out[i].n = -1
    for i in range(stop - start):
        out[2 * i].n = start + i
    return out[::2]  # stride = 2 * itemsize


@njit
def _series_empty(start, stop):
    return np.empty(0, _OUT)


def test_tvf_offset_slice_return():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series_sliced)
    _, rows = _select_int(db, "SELECT n FROM series(2, 5)")
    assert [x[0] for x in rows] == [2, 3, 4]
    sqlite3_close(db.value)
    del h


def test_tvf_strided_return():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series_strided)
    _, rows = _select_int(db, "SELECT n FROM series(2, 5)")
    assert [x[0] for x in rows] == [2, 3, 4]
    sqlite3_close(db.value)
    del h


def test_tvf_empty_return():
    db = _open()
    h = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series_empty)
    _, rows = _select_int(db, "SELECT n FROM series(2, 5)")
    assert rows == []
    sqlite3_close(db.value)
    del h
```

- [ ] **Step 2: Run to confirm the slice/strided tests fail** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_tvf.py -k "offset_slice or strided" -q` → FAIL (wrong rows)

- [ ] **Step 3: Implement** — in `numbox/core/bindings/_sqlite_tvf.py`:

Add `row_stride` to `_TVF_CUR_DTYPE` (lines 90–93):
```python
_TVF_CUR_DTYPE = np.dtype([
    ("base", _SQLITE3_VTAB_CURSOR_DTYPE), ("descriptor", "i8"), ("rowid", "i8"),
    ("mi_p", "i8"), ("data_p", "i8"), ("n_rows", "i8"), ("row_stride", "i8"), ("scratch_p", "i8"),
], align=True)
_TVF_CUR_SIZE = _TVF_CUR_DTYPE.itemsize
```

Update the pin block in `_XFILTER_SRC` (lines 124–129) to store the logical pointer + stride while still pinning via the meminfo:
```python
    result = {fn_call}
    mi_p, _base = structref_meminfo(result)
    _incref_meminfo(mi_p)
    c[0].mi_p = mi_p
    c[0].data_p = array_data_p(result)
    c[0].n_rows = result.shape[0]
    c[0].row_stride = result.strides[0]
```

> `array_data_p` is already imported in `_sqlite_tvf.py` (line 65–67). Confirm it is in the `# noqa: F401` import group; it is referenced by the generated source so it must stay imported.

Initialize `row_stride` in `_tvf_xopen` alongside the other cursor fields (the block at lines 235–242), adding:
```python
            c[0].row_stride = 0
```
(place it next to `c[0].n_rows = 0`). Also zero it in `_tvf_xclose`'s reset and in `_tvf_xfilter_impl`'s top-of-call reset (lines 114–118) — add `c[0].row_stride = 0` wherever `c[0].n_rows = 0` is set on the release/reset paths.

Update `_tvf_xcolumn`'s address math (line 303) to use the stored stride:
```python
            row_stride = d_unused = 0  # placeholder removed below
```
Replace line 303 exactly:
```python
            addr = data_p + rowid * c[0].row_stride + offsets[j]
```
(Read `c` is already in scope in `_tvf_xcolumn`; `data_p`, `rowid`, `offsets`, `itemsize` are computed just above — keep `itemsize` for any other use but address rows via `c[0].row_stride`.)

Update the `register_tvf` docstring (lines 417–434) — add a sentence to the existing docstring body:
```python
    ``fn`` must return a 1-D numpy structured array whose dtype is ``out_dtype``;
    a slice/strided/offset view is handled (the row stride is honored), but a
    return whose *dtype* differs from ``out_dtype`` is read through ``out_dtype``'s
    layout and yields undefined values.
```

- [ ] **Step 4: Run to confirm pass** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_tvf.py -q` → PASS

- [ ] **Step 5: flake8 + commit**

```bash
/home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_sqlite_tvf.py test/core/test_sqlite_tvf.py --max-line-length=127
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_tvf.py test/core/test_sqlite_tvf.py
git -C /home/erik/projects/numbox commit -m "fix(sqlite-tvf): honor result row stride/offset in xColumn; document fn return contract"
```

---

### Task 7: tvf — validate non-numeric arg_types; cover fn-raises behavior (M5, tests_docs-2)

**Goal:** `register_tvf` rejects non-numeric `arg_types` instead of silently decoding a string/bytes argument as a garbage float; the documented "user fn raises → 0 rows" behavior gets a test.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_tvf.py` (`_build_tvf_descriptor` ~368–398)
- Test: `test/core/test_sqlite_tvf.py` (append)

**Acceptance Criteria:**
- [ ] `register_tvf` raises `TypeError` if any element of `arg_types` is not an integer or float scalar dtype
- [ ] A `fn` that raises yields zero rows (no crash), confirming the deliberate `@cfunc`-boundary swallow
- [ ] Existing numeric-arg TVF tests still pass

**Verify:** `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_tvf.py -q` → all pass

**Steps:**

- [ ] **Step 1: Write the failing tests** — append to `test/core/test_sqlite_tvf.py`:

```python
def test_tvf_non_numeric_arg_type_raises():
    db = _open()
    with pytest.raises(TypeError):
        register_tvf(db.value, "f", (np.dtype("U4"),), _OUT, _series)
    sqlite3_close(db.value)


@njit
def _raises(start, stop):
    out = np.empty(stop - start, _OUT)
    # deliberately index out of bounds to raise inside the user fn
    out[stop - start].n = 0
    return out


def test_tvf_user_fn_raises_yields_no_rows():
    db = _open()
    h = register_tvf(db.value, "boom", (np.int64, np.int64), _OUT, _raises)
    _, rows = _select_int(db, "SELECT n FROM boom(2, 5)")
    assert rows == []
    sqlite3_close(db.value)
    del h
```

- [ ] **Step 2: Run to confirm `test_tvf_non_numeric_arg_type_raises` fails** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_tvf.py::test_tvf_non_numeric_arg_type_raises -q` → FAIL (no TypeError)

- [ ] **Step 3: Implement** — in `numbox/core/bindings/_sqlite_tvf.py`, after `arg_tags` is computed in `_build_tvf_descriptor` (after line 378), add the guard:

```python
    if any(t not in _INT_TAGS and t not in _FLOAT_TAGS for t in arg_tags):
        raise TypeError(
            "register_tvf arg_types must be integer or float scalar dtypes; "
            "string/bytes hidden args are not supported")
```

(`_INT_TAGS` and `_FLOAT_TAGS` are already module-level frozensets, lines 96–98.)

- [ ] **Step 4: Run to confirm pass** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_tvf.py -q` → PASS

- [ ] **Step 5: flake8 + commit**

```bash
/home/erik/projects/numbox/venv/bin/flake8 numbox/core/bindings/_sqlite_tvf.py test/core/test_sqlite_tvf.py --max-line-length=127
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_tvf.py test/core/test_sqlite_tvf.py
git -C /home/erik/projects/numbox commit -m "fix(sqlite-tvf): reject non-numeric arg_types; cover user-fn-raises behavior"
```

---

### Task 8: Coverage tests — two-TVF coexistence, cross-process cache, xDestroy deferred close

**Goal:** Close the test-coverage gaps the review flagged that aren't tied to a code fix: two *different* TVF registrations coexisting in one process, cross-process (cold→warm) cache reuse for query_to_array and TVF, and xDestroy NOT firing while a statement is unfinalized (BUSY close).

**Files:**
- Test: `test/core/test_sqlite_tvf.py` (append), `test/core/test_sqlite_query.py` (append), `test/core/test_sqlite_vtable.py` (append)

**Acceptance Criteria:**
- [ ] Two TVFs with different `fn` AND different `out_dtype` registered in one connection each return their own rows/columns
- [ ] A subprocess that imports numbox and runs query_to_array + a TVF twice (cold then warm cache) succeeds both runs (mirrors the existing `test_xprocess_cache_no_growth` style)
- [ ] Closing a connection with an unfinalized TVF statement (`sqlite3_close` returning BUSY) does NOT pop the registry; finalizing then closing does
- [ ] All pass

**Verify:** `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_tvf.py test/core/test_sqlite_query.py test/core/test_sqlite_vtable.py -q` → all pass

**Steps:**

- [ ] **Step 1: Two-TVF coexistence** — append to `test/core/test_sqlite_tvf.py`:

```python
def test_tvf_two_distinct_registrations_same_process():
    db = _open()
    h1 = register_tvf(db.value, "series", (np.int64, np.int64), _OUT, _series)
    h2 = register_tvf(db.value, "scaled", (np.int64, np.int64, np.float64), _OUT2, _scaled)
    _, r1 = _select_int(db, "SELECT n FROM series(2, 5)")
    stmt = c_int64(0)
    with c_string("SELECT n, v FROM scaled(0, 3, 2.5)") as p:
        sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    r2 = []
    while sqlite3_step(stmt.value) == 100:
        r2.append((sqlite3_column_int64(stmt.value, 0), sqlite3_column_double(stmt.value, 1)))
    sqlite3_finalize(stmt.value)
    assert [x[0] for x in r1] == [2, 3, 4]
    assert r2 == [(0, 0.0), (1, 2.5), (2, 5.0)]
    sqlite3_close(db.value)
    del h1, h2
```

- [ ] **Step 2: Cross-process cache for query_to_array** — append to `test/core/test_sqlite_query.py`. Model it on the existing `test_xprocess_cache_no_growth` in `test_sqlite_vtable.py` (read that test first for the exact `subprocess` + `textwrap.dedent` + env idiom). The subprocess body should: open `:memory:`, create+populate a table, call `query_to_array` twice, assert results; run it once (cold) then again (warm) and assert both exit 0.

```python
def test_query_xprocess_cache(tmp_path):
    import subprocess, sys, textwrap, os
    prog = textwrap.dedent('''
        from ctypes import addressof, c_int64
        import numpy as np
        from numbox.core.bindings import sqlite3_open, sqlite3_close, query_to_array, sqlite3_exec
        from numbox.utils.cstrings import c_string
        db = c_int64(0)
        with c_string(":memory:") as p:
            assert sqlite3_open(p, addressof(db)) == 0
        with c_string("CREATE TABLE t(i INTEGER, x REAL)") as p:
            assert sqlite3_exec(db.value, p, 0, 0, 0) == 0
        with c_string("INSERT INTO t VALUES (1,1.5),(2,2.5)") as p:
            assert sqlite3_exec(db.value, p, 0, 0, 0) == 0
        dt = np.dtype([("i","i8"),("x","f8")])
        with c_string("SELECT i, x FROM t ORDER BY i") as s:
            out = query_to_array(db.value, s, dt)
        assert list(out["i"]) == [1, 2]
        sqlite3_close(db.value)
        print("OK")
    ''')
    env = dict(os.environ)
    for _ in range(2):  # cold then warm
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        assert "OK" in r.stdout
```

- [ ] **Step 3: xDestroy not fired on BUSY close** — append to `test/core/test_sqlite_vtable.py` (uses `register_tvf`, `_SQLITE_ROW`, and `v._REGISTRY` like the existing `test_xdestroy_tvf_pops_registry_on_close`):

```python
def test_xdestroy_deferred_while_statement_open():
    from numbox.core.bindings import _sqlite_vtable as v
    from numbox.core.bindings import register_tvf
    from numba import njit
    out = np.dtype([("n", "i8")])

    @njit
    def _series(start, stop):
        o = np.empty(stop - start, out)
        for i in range(stop - start):
            o[i].n = start + i
        return o

    db = c_int64(0)
    with c_string(":memory:") as p:
        sqlite3_open(p, addressof(db))
    h = register_tvf(db.value, "series", (np.int64, np.int64), out, _series)
    stmt = c_int64(0)
    with c_string("SELECT n FROM series(2, 5)") as p:
        sqlite3_prepare_v2(db.value, p, -1, addressof(stmt), 0)
    sqlite3_step(stmt.value)  # leave the statement open (not finalized)
    n_before = len(v._REGISTRY)
    rc = sqlite3_close(db.value)  # sqlite3_close returns SQLITE_BUSY with an open stmt
    assert rc != 0  # BUSY: close refused, xDestroy NOT fired
    assert len(v._REGISTRY) == n_before  # registry entry still present
    sqlite3_finalize(stmt.value)
    assert sqlite3_close(db.value) == 0  # now it closes and fires xDestroy
    assert len(v._REGISTRY) == n_before - 1
    del h
```

> Verify on the venv sqlite that `sqlite3_close` (v1) indeed returns BUSY with an open statement; if the binding maps to `sqlite3_close_v2` (zombie close, returns OK and defers), adjust the assertions to check the registry is popped only after finalize. Read which symbol `sqlite3_close` binds in `numbox/core/bindings/_sqlite_conn.py` before finalizing the assertions.

- [ ] **Step 4: Run all three files** — `<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest test/core/test_sqlite_tvf.py test/core/test_sqlite_query.py test/core/test_sqlite_vtable.py -q` → PASS

- [ ] **Step 5: flake8 + commit**

```bash
/home/erik/projects/numbox/venv/bin/flake8 test/core/test_sqlite_tvf.py test/core/test_sqlite_query.py test/core/test_sqlite_vtable.py --max-line-length=127
git -C /home/erik/projects/numbox add test/core/test_sqlite_tvf.py test/core/test_sqlite_query.py test/core/test_sqlite_vtable.py
git -C /home/erik/projects/numbox commit -m "test(sqlite): cover two-TVF coexistence, cross-process cache, deferred xDestroy"
```

---

### Task 9: Full CI-gate verification + docs check

**Goal:** Confirm the whole change set passes every CI gate locally (the review's pass criteria) and the docs are consistent.

**Files:**
- Verify only: `docs/numbox.core.bindings.rst` (no new modules were added, so likely no change — confirm the automodule list is still accurate)

**Acceptance Criteria:**
- [ ] `flake8 . --count --show-source --statistics` → 0
- [ ] Full `pytest --durations=20` → all pass (no regressions; new tests included)
- [ ] `sphinx-build` → exit 0, no NEW warnings referencing the changed modules
- [ ] `doc-codeblock-flake8` (extract_codeblocks with venv flake8 on PATH) → clean
- [ ] `lychee` over changed `.rst/.md/.py` → clean except the known pre-existing musl timeout

**Verify:** the command block below, each exit 0 (lychee may report the 1 known musl timeout; that is acceptable and pre-existing)

**Steps:**

- [ ] **Step 1: flake8 (whole tree, CI invocation)**
```bash
/home/erik/projects/numbox/venv/bin/flake8 . --count --show-source --statistics
```

- [ ] **Step 2: Full suite, cold cache**
```bash
<CLEAN> && /home/erik/projects/numbox/venv/bin/pytest --durations=20 -q
```
Expected: all pass / 2 skipped (same skips as baseline), no failures.

- [ ] **Step 3: Sphinx (clean build)**
```bash
rm -rf /home/erik/projects/numbox/docs/_build && /home/erik/projects/numbox/venv/bin/sphinx-build -b html /home/erik/projects/numbox/docs /home/erik/projects/numbox/docs/_build/html
```
Expected: `build succeeded`; confirm no NEW warnings cite `_sqlite_query`/`_sqlite_tvf`/`_sqlite_typemap`/`_sqlite_vtable`.

- [ ] **Step 4: doc-codeblock-flake8**
```bash
PATH="/home/erik/projects/numbox/venv/bin:$PATH" /home/erik/projects/numbox/venv/bin/python /home/erik/projects/numbox/.github/scripts/extract_codeblocks.py /home/erik/projects/numbox/docs README.md
```
Expected: exit 0.

- [ ] **Step 5: lychee over changed files**
```bash
git -C /home/erik/projects/numbox diff --name-only --diff-filter=AM origin/main...HEAD | grep -E '\.(rst|md|py)$' | grep -vE '^\.github/' | xargs -r lychee --no-progress --max-retries 2 --timeout 20 --accept 200,206,429 --exclude '^mailto:'
```
Expected: clean except the known `git.musl-libc.org/.../strerror_r.c` timeout (pre-existing, not introduced here).

- [ ] **Step 6: Mark plan complete** — update `docs/superpowers/plans/2026-06-04-sqlite-phase5-review-fixes.md.tasks.json` statuses to `completed`, commit the plan + tasks.json:
```bash
git -C /home/erik/projects/numbox add docs/superpowers/plans/2026-06-04-sqlite-phase5-review-fixes.md docs/superpowers/plans/2026-06-04-sqlite-phase5-review-fixes.md.tasks.json
git -C /home/erik/projects/numbox commit -m "docs(plan): phase-5 review-fix plan + task tracker"
```

---

## Self-Review (completed by plan author)

- **Finding coverage:** H1→T4; M1→T1; M2→T0+T2+T3; M3→T6; M4→T4; M5→T7; L1→T5; L2→T3; L3→T6 (doc); nits (sql_p docstring→T2, omit comment→T5, column_bytes order→T2)→covered; test gaps (large-int/uint64→T4, packed dtype→T2, misaligned 'U'→T3, TVF slice/strided/0-row→T6, string-arg/fn-raises→T7, two-TVF/cross-process/deferred-xDestroy→T8). Every confirmed finding maps to a task.
- **Type consistency:** `_PRED_DTYPE` field names (`col/op/is_int/ival/fval`) are used identically in T4's `_xfilter` and `_row_matches`; `row_stride` is added to `_TVF_CUR_DTYPE` in T6 and read in the same task's `_tvf_xcolumn`; `store_unaligned` defined in T0 is imported in T2/T3.
- **Known soft spots flagged for the executor (not placeholders — verification steps):** the exact step-error-raising SQL (T1), whether `sqlite3_close` binds v1/v2 (T8), and `eval`-free expected-set construction (T4) each carry an inline verification note to settle against the live venv before finalizing the test.
