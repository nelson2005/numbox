# SQLite UDAF Registration Helper — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `register_aggregate` / `register_window` helpers that generate the SQLite UDAF/window callbacks (xStep/xInverse/xValue/xFinal) — the `aggregate_context` + meminfo lifecycle — so callers write only their `@njit` `init`/`step`/`finalize` (+ `inverse`/`value`) state logic.

**Architecture:** Mechanism **B** (selected by the spec's S1–S4 spike re-review). Per-UDAF callback source is generated with the state type and user functions **baked in as module globals** (so calls inline), written to a **content-addressed anchor file** under numba's cache dir via `numbox.utils.preprocessing` (reusing the `make_structref` recipe), and the `@njit(cache=True)` impls cache across processes. The anchor's content hash folds a **cloudpickle of the user functions' code objects** (co_consts-sensitive) so editing a body — including a numeric literal — invalidates correctly. Thin per-UDAF `@cfunc` trampolines forward into the cached impls; a keep-alive handle holds them.

**Tech Stack:** numba 0.65.1 (`njit`/`cfunc`, structref, `numba.core.serialize.cloudpickle`), numbox phase-2 sqlite bindings + `meminfo` bridge + `preprocessing.py` anchors, pytest.

**Spec:** `docs/superpowers/specs/2026-05-29-sqlite-udaf-registration-helper-design.md`

**Branch / sequencing:** Implement on `feat/sqlite-udaf-helper` (off `feat/sqlite-udf`, which carries the unmerged phase-2 modules). Upstream-PR strategy (wait for #17 vs rebase) is decided separately at integration time.

---

## File structure

- **Create** `numbox/core/bindings/_sqlite_udf_helpers.py` — the helper: source templates, digest, codegen+anchor+exec, `register_aggregate`, `register_window`, `_UDAFHandle`. One responsibility: turn user state-logic functions into registered SQLite UDAFs.
- **Modify** `numbox/core/bindings/__init__.py` — add the star-import so `register_aggregate`/`register_window` join the public bindings API.
- **Create** `test/core/test_sqlite_udf_helpers.py` — parity, leak, no-collision, deterministic, cross-process cache, invalidation tests.
- **Modify** `docs/numbox.core.bindings.rst` — `automodule` section + family-list entry for the new module (mandatory docs per CLAUDE.md).

---

### Task 1: Aggregate registration helper (`register_aggregate`) + codegen/anchor/digest/handle

**Goal:** A working `register_aggregate(db, name, n_arg, state_type, init, step, finalize, *, deterministic=False)` that registers a structref-backed aggregate UDAF whose result matches the hand-written phase-2 pattern, including the empty-group path.

**Files:**
- Create: `numbox/core/bindings/_sqlite_udf_helpers.py`
- Test: `test/core/test_sqlite_udf_helpers.py`

**Acceptance Criteria:**
- [ ] `register_aggregate` registers a `SUM` UDAF; `SELECT my_sum(v) FROM t` over `[1..5]` yields `15`.
- [ ] An empty table yields `0` (empty-group path: `aggregate_context` returns NULL → finalize a fresh `init()` state, no release).
- [ ] `state_type` that is not a `StructRef` instance raises `TypeError` at registration.
- [ ] The returned handle keeps the `cfunc`s + user fns alive (registration survives a `gc.collect()` before the query).

**Verify:** `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_udf_helpers.py -k "aggregate" -v` → 2 passed

**Steps:**

- [ ] **Step 1: Clean caches (project rule), then write the failing tests**

Run first (project rule — clean numba + pycache before any pytest run):
```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree(pathlib.Path.home()/'.cache'/'numba', ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]; print('cleared')"
```

Create `test/core/test_sqlite_udf_helpers.py`:
```python
"""Tests for the structref-backed SQLite UDAF/window registration helpers."""
import gc
from ctypes import addressof, c_int64

import numpy as np
from numba import carray, cfunc, njit, types
from numba.core import types as nb_types
from numba.experimental import structref

from numbox.core.bindings._sqlite_conn import sqlite3_close, sqlite3_open
from numbox.core.bindings._sqlite_constants import SQLITE_OK, SQLITE_UTF8
from numbox.core.bindings._sqlite_exec import sqlite3_exec
from numbox.core.bindings._sqlite_result import sqlite3_result_int, sqlite3_result_int64
from numbox.core.bindings._sqlite_udf import (
    sqlite3_create_function_v2,
    sqlite3_user_data,
)
from numbox.core.bindings._sqlite_udf_helpers import register_aggregate
from numbox.core.bindings._sqlite_value import sqlite3_value_int64
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p


# --- state type (module-level => importable, stable __module__) ---
@structref.register
class SumStateType(nb_types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)


class SumState(structref.StructRefProxy):
    def __new__(cls, total):
        return structref.StructRefProxy.__new__(cls, total)


structref.define_proxy(SumState, SumStateType, ["total"])
sum_state_type = SumStateType([("total", nb_types.int64)])


@njit
def sum_init():
    return SumState(nb_types.int64(0))


@njit
def sum_step(state, ctx, argc, argv_pp):
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total += sqlite3_value_int64(args[0])


@njit
def sum_finalize(state, ctx):
    sqlite3_result_int64(ctx, state.total)


# --- test plumbing ---
def _open_memory():
    db_p = c_int64(0)
    with c_string(":memory:") as name_p:
        assert sqlite3_open(name_p, addressof(db_p)) == SQLITE_OK
    return db_p.value


def _make_table(db, values):
    with c_string("CREATE TABLE t(v INTEGER)") as p:
        assert sqlite3_exec(db, p, 0, 0, 0) == SQLITE_OK
    for v in values:
        with c_string("INSERT INTO t VALUES (%d)" % v) as p:
            assert sqlite3_exec(db, p, 0, 0, 0) == SQLITE_OK


def _read1_int64(db, select_sql):
    """Route a scalar SELECT through a capture UDF that stores arg0 into a
    numpy buffer; returns (value, keepalive)."""
    buf = np.zeros(1, dtype=np.int64)

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def cap_cb(ctx, argc, argv):
        ud = sqlite3_user_data(ctx)
        args = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
        o = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
        o[0] = sqlite3_value_int64(args[0])
        sqlite3_result_int(ctx, 0)

    with c_string("__cap") as cp:
        assert sqlite3_create_function_v2(
            db, cp, 1, SQLITE_UTF8, buf.ctypes.data, cap_cb.address, 0, 0, 0) == SQLITE_OK
    with c_string(select_sql) as sp:
        assert sqlite3_exec(db, sp, 0, 0, 0) == SQLITE_OK
    return int(buf[0]), cap_cb


def test_aggregate_sum():
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    h = register_aggregate(db, "my_sum", 1, sum_state_type,
                           sum_init, sum_step, sum_finalize)
    gc.collect()  # handle must keep callbacks alive
    val, _cap = _read1_int64(db, "SELECT __cap(my_sum(v)) FROM t")
    sqlite3_close(db)
    assert val == 15
    assert h is not None


def test_aggregate_empty_group():
    db = _open_memory()
    _make_table(db, [])
    h = register_aggregate(db, "my_sum", 1, sum_state_type,
                           sum_init, sum_step, sum_finalize)
    val, _cap = _read1_int64(db, "SELECT __cap(my_sum(v)) FROM t")
    sqlite3_close(db)
    assert val == 0
    assert h is not None


def test_aggregate_bad_state_type():
    import pytest
    db = _open_memory()
    with pytest.raises(TypeError):
        register_aggregate(db, "bad", 1, object(), sum_init, sum_step, sum_finalize)
    sqlite3_close(db)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_udf_helpers.py -k aggregate -v`
Expected: FAIL — `ModuleNotFoundError: numbox.core.bindings._sqlite_udf_helpers`.

- [ ] **Step 3: Write the helper module (aggregate path)**

Create `numbox/core/bindings/_sqlite_udf_helpers.py`:
```python
"""Higher-level registration helpers for structref-backed SQLite UDAFs.

``register_aggregate`` / ``register_window`` generate the SQLite callback
functions (xStep/xInverse/xValue/xFinal) that perform the per-group state
lifecycle -- ``sqlite3_aggregate_context`` allocation + NULL guard, the single
intp slot, init-on-first-step via ``export_meminfo``, ``borrow_structref``, and
the release-in-xFinal-but-NOT-xValue rule -- so callers write only their
``@njit`` ``init``/``step``/``finalize`` (and ``inverse``/``value`` for windows)
state logic.

Mechanism (see docs/superpowers/specs/2026-05-29-sqlite-udaf-registration-helper-design.md):
per-UDAF callback source is generated with the state type and the user functions
baked in as module globals (so the calls inline), written to a content-addressed
anchor file under numba's cache dir (reusing numbox.utils.preprocessing), and the
``@njit(cache=True)`` impls cache across processes. The anchor's content hash
folds a cloudpickle of the user functions' code objects so editing a body --
including a numeric literal -- invalidates correctly.

**Caller requirements.** The state-type class and the ``@njit`` functions MUST
live in an importable module (stable ``__module__``, not ``__main__``); this is a
precondition for numba caching of any generated code that references them.
"""
import ctypes
import hashlib
from inspect import getmodule

import numba
from numba import cfunc, types
from numba.core.serialize import cloudpickle
from numba.core.types import StructRef

import numbox
from numbox.core.bindings._sqlite_conn import sqlite3_errmsg
from numbox.core.bindings._sqlite_constants import (
    SQLITE_DETERMINISTIC,
    SQLITE_OK,
    SQLITE_UTF8,
)
from numbox.core.bindings._sqlite_udf import (
    sqlite3_create_function_v2,
    sqlite3_create_window_function,
)
from numbox.utils.cstrings import c_string
from numbox.utils.preprocessing import (
    _anchor_path,
    _materialize_anchor,
    _orphan_anchor_sweep,
)

# These names are referenced by the GENERATED anchor source; importing them here
# puts them in this module's __dict__, which seeds the exec namespace below.
import numpy as np  # noqa: F401
from numba import carray, njit  # noqa: F401
from numbox.core.bindings._sqlite_udf import sqlite3_aggregate_context  # noqa: F401
from numbox.utils.lowlevel import _cast_int_to_void_p  # noqa: F401
from numbox.utils.meminfo import (  # noqa: F401
    borrow_structref,
    export_meminfo,
    release_meminfo,
)

__all__ = ["register_aggregate", "register_window"]

_ANCHOR_SUBDIR = "numbox-sqlite-udaf"
_orphan_anchor_sweep(_ANCHOR_SUBDIR)


def _file_anchor():
    """Identity handle whose module __dict__ (this module's globals, incl.
    ``__name__``) seeds the generated code's exec namespace. Mirrors
    ``numbox.utils.highlevel.make_structref``; the ``__name__`` is required or
    the warm cache reload aborts in numba's Environment rebuild."""
    raise NotImplementedError


# Generated-source templates. Baked global names: _state_type, _init, _step,
# _finalize, _inverse, _value. Each impl is @njit(cache=True) so it caches
# cross-process; the user fns inline because they are module globals here.
_XSTEP_SRC = '''
@njit(cache=True)
def _xstep_impl(ctx, argc, argv_pp):
    agg = sqlite3_aggregate_context(ctx, 8)
    if agg == 0:
        return
    slot = carray(_cast_int_to_void_p(agg), (1,), dtype=np.intp)
    if slot[0] == 0:
        slot[0] = export_meminfo(_init())
    _step(borrow_structref(_state_type, slot[0]), ctx, argc, argv_pp)
'''

_XFINAL_SRC = '''
@njit(cache=True)
def _xfinal_impl(ctx):
    agg = sqlite3_aggregate_context(ctx, 0)
    if agg == 0:
        _finalize(_init(), ctx)
        return
    slot = carray(_cast_int_to_void_p(agg), (1,), dtype=np.intp)
    if slot[0] == 0:
        _finalize(_init(), ctx)
        return
    _finalize(borrow_structref(_state_type, slot[0]), ctx)
    release_meminfo(slot[0])
'''


class _UDAFHandle:
    """Keeps the generated ``cfunc`` callbacks (and the user functions they bake
    in) alive. SQLite holds raw pointers to the cfuncs; if this handle is
    garbage-collected the pointers dangle and the next call segfaults. **The
    caller MUST retain the returned handle for as long as the UDAF is used.**"""
    __slots__ = ("_keep",)

    def __init__(self, *objs):
        self._keep = objs


def _validate_state_type(state_type):
    if not isinstance(state_type, StructRef):
        raise TypeError(
            "state_type must be a numba StructRef instance "
            "(e.g. MyStateType([('x', int64)])), got %r" % (state_type,))


def _digest(state_type, fns):
    """Content hash that invalidates when the state type, the user functions
    (co_consts-sensitive, via cloudpickle of the code object -- NOT bare
    co_code), or the numba/numbox versions change."""
    h = hashlib.sha256()
    h.update(repr(state_type).encode("utf-8"))
    h.update(numba.__version__.encode("utf-8"))
    h.update((numbox.__version__ or "").encode("utf-8"))
    for fn in fns:
        py = getattr(fn, "py_func", fn)
        h.update(cloudpickle.dumps(py.__code__))
    return h.hexdigest()[:16]


def _compile_callbacks(stem, srcs, state_type, fns):
    """Generate + content-address-anchor + exec the @njit(cache=True) impls.

    ``fns`` maps generated global names (``_init``, ``_step``, ...) to user
    callables. Returns the exec namespace (contains ``_xstep_impl`` etc.)."""
    digest = _digest(state_type, list(fns.values()))
    code_txt = "# udaf-digest: %s\n%s" % (digest, "".join(srcs))
    ns = {**getmodule(_file_anchor).__dict__, "_state_type": state_type, **fns}
    anchor = _anchor_path(_ANCHOR_SUBDIR, stem, code_txt)
    _materialize_anchor(anchor, code_txt)
    code = compile(code_txt, str(anchor), mode="exec")
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns


def _raise_rc(db, name, rc):
    msg_p = sqlite3_errmsg(db)
    detail = ""
    if msg_p:
        detail = ": " + ctypes.cast(
            msg_p, ctypes.c_char_p).value.decode("utf-8", "replace")
    raise RuntimeError(
        "sqlite3 UDAF registration failed for %r (rc=%d)%s" % (name, rc, detail))


def _stem(prefix, name):
    return prefix + "".join(c if c.isalnum() else "_" for c in name)


def register_aggregate(db, name, n_arg, state_type, init, step, finalize,
                       *, deterministic=False):
    """Register a structref-backed aggregate UDAF.

    :param db: connection pointer (intp), as returned by ``sqlite3_open``.
    :param name: SQL function name (str); the C-string lifetime is handled here.
    :param n_arg: argument count, or -1 for variadic.
    :param state_type: the numba structref *instance* type for per-group state.
    :param init: ``@njit`` ``() -> state`` returning a fresh state.
    :param step: ``@njit`` ``(state, ctx, argc, argv_pp)`` updating state.
    :param finalize: ``@njit`` ``(state, ctx)`` writing the result.
    :param deterministic: OR-in ``SQLITE_DETERMINISTIC``.
    :returns: a keep-alive handle the caller MUST retain (see ``_UDAFHandle``).
    """
    _validate_state_type(state_type)
    ns = _compile_callbacks(
        _stem("udaf_", name), [_XSTEP_SRC, _XFINAL_SRC], state_type,
        {"_init": init, "_step": step, "_finalize": finalize})
    xstep_impl = ns["_xstep_impl"]
    xfinal_impl = ns["_xfinal_impl"]

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def step_cb(ctx, argc, argv):
        xstep_impl(ctx, argc, argv)

    @cfunc(types.void(types.intp))
    def final_cb(ctx):
        xfinal_impl(ctx)

    flags = SQLITE_UTF8 | (SQLITE_DETERMINISTIC if deterministic else 0)
    with c_string(name) as name_p:
        rc = sqlite3_create_function_v2(
            db, name_p, n_arg, flags, 0, 0,
            step_cb.address, final_cb.address, 0)
    if rc != SQLITE_OK:
        _raise_rc(db, name, rc)
    return _UDAFHandle(step_cb, final_cb, xstep_impl, xfinal_impl,
                       init, step, finalize)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_udf_helpers.py -k aggregate -v`
Expected: PASS (`test_aggregate_sum`, `test_aggregate_empty_group`, `test_aggregate_bad_state_type`).

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_udf_helpers.py test/core/test_sqlite_udf_helpers.py
git -C /home/erik/projects/numbox commit -m "feat(udaf): register_aggregate helper (content-addressed codegen + meminfo lifecycle)"
```

---

### Task 2: Window registration helper (`register_window`)

**Goal:** Add `register_window(db, name, n_arg, state_type, init, step, inverse, value, finalize, *, deterministic=False)`, owning the `xInverse`/`xValue` callbacks and the release-in-`xFinal`-but-not-`xValue` rule.

**Files:**
- Modify: `numbox/core/bindings/_sqlite_udf_helpers.py` (add `_XINVERSE_SRC`, `_XVALUE_SRC`, `register_window`)
- Test: `test/core/test_sqlite_udf_helpers.py` (add window test)

**Acceptance Criteria:**
- [ ] A running-sum window (`ROWS BETWEEN 1 PRECEDING AND CURRENT ROW`) over `[1..5]` yields `[1, 3, 5, 7, 9]`.
- [ ] `xValue` reads the running result without releasing; `xFinal` releases once (covered by the leak test in Task 3).

**Verify:** `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_udf_helpers.py -k window -v` → 1 passed

**Steps:**

- [ ] **Step 1: Write the failing window test**

Append to `test/core/test_sqlite_udf_helpers.py`:
```python
from numbox.core.bindings._sqlite_udf_helpers import register_window  # noqa: E402


@njit
def w_inverse(state, ctx, argc, argv_pp):
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total -= sqlite3_value_int64(args[0])


@njit
def w_value(state, ctx):
    sqlite3_result_int64(ctx, state.total)


def _read_window(db, select_sql, nrows):
    meta = np.zeros(nrows + 1, dtype=np.int64)  # meta[0]=count, meta[1:]=values

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def wcap_cb(ctx, argc, argv):
        ud = sqlite3_user_data(ctx)
        m = carray(_cast_int_to_void_p(ud), (nrows + 1,), dtype=np.int64)
        i = m[0]
        args = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
        m[1 + i] = sqlite3_value_int64(args[0])
        m[0] = i + 1
        sqlite3_result_int(ctx, 0)

    with c_string("__wcap") as cp:
        assert sqlite3_create_function_v2(
            db, cp, 1, SQLITE_UTF8, meta.ctypes.data, wcap_cb.address, 0, 0, 0) == SQLITE_OK
    with c_string(select_sql) as sp:
        assert sqlite3_exec(db, sp, 0, 0, 0) == SQLITE_OK
    return [int(meta[1 + i]) for i in range(int(meta[0]))], wcap_cb


def test_window_running_sum():
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    h = register_window(db, "my_wsum", 1, sum_state_type,
                        sum_init, sum_step, w_inverse, w_value, sum_finalize)
    sql = ("SELECT __wcap(my_wsum(v) OVER "
           "(ORDER BY v ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)) "
           "FROM t ORDER BY v")
    vals, _cap = _read_window(db, sql, 5)
    sqlite3_close(db)
    assert vals == [1, 3, 5, 7, 9]
    assert h is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_udf_helpers.py -k window -v`
Expected: FAIL — `ImportError: cannot import name 'register_window'`.

- [ ] **Step 3: Add window templates + `register_window`**

In `numbox/core/bindings/_sqlite_udf_helpers.py`, add after `_XFINAL_SRC`:
```python
_XINVERSE_SRC = '''
@njit(cache=True)
def _xinverse_impl(ctx, argc, argv_pp):
    agg = sqlite3_aggregate_context(ctx, 0)
    if agg == 0:
        return
    slot = carray(_cast_int_to_void_p(agg), (1,), dtype=np.intp)
    if slot[0] == 0:
        return
    _inverse(borrow_structref(_state_type, slot[0]), ctx, argc, argv_pp)
'''

_XVALUE_SRC = '''
@njit(cache=True)
def _xvalue_impl(ctx):
    agg = sqlite3_aggregate_context(ctx, 0)
    if agg == 0:
        _value(_init(), ctx)
        return
    slot = carray(_cast_int_to_void_p(agg), (1,), dtype=np.intp)
    if slot[0] == 0:
        _value(_init(), ctx)
        return
    _value(borrow_structref(_state_type, slot[0]), ctx)
'''
```

Add at the end of the module:
```python
def register_window(db, name, n_arg, state_type, init, step, inverse, value,
                    finalize, *, deterministic=False):
    """Register a structref-backed window UDAF.

    Same as :func:`register_aggregate` plus ``inverse(state, ctx, argc,
    argv_pp)`` (un-applies a row; state already exists) and ``value(state,
    ctx)`` (emits the running result WITHOUT releasing). Only ``xFinal``
    releases the meminfo.
    """
    _validate_state_type(state_type)
    ns = _compile_callbacks(
        _stem("wudaf_", name),
        [_XSTEP_SRC, _XINVERSE_SRC, _XVALUE_SRC, _XFINAL_SRC], state_type,
        {"_init": init, "_step": step, "_inverse": inverse,
         "_value": value, "_finalize": finalize})
    xstep_impl = ns["_xstep_impl"]
    xinverse_impl = ns["_xinverse_impl"]
    xvalue_impl = ns["_xvalue_impl"]
    xfinal_impl = ns["_xfinal_impl"]

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def step_cb(ctx, argc, argv):
        xstep_impl(ctx, argc, argv)

    @cfunc(types.void(types.intp, types.int32, types.intp))
    def inverse_cb(ctx, argc, argv):
        xinverse_impl(ctx, argc, argv)

    @cfunc(types.void(types.intp))
    def value_cb(ctx):
        xvalue_impl(ctx)

    @cfunc(types.void(types.intp))
    def final_cb(ctx):
        xfinal_impl(ctx)

    flags = SQLITE_UTF8 | (SQLITE_DETERMINISTIC if deterministic else 0)
    with c_string(name) as name_p:
        rc = sqlite3_create_window_function(
            db, name_p, n_arg, flags, 0,
            step_cb.address, final_cb.address,
            value_cb.address, inverse_cb.address, 0)
    if rc != SQLITE_OK:
        _raise_rc(db, name, rc)
    return _UDAFHandle(step_cb, inverse_cb, value_cb, final_cb,
                       xstep_impl, xinverse_impl, xvalue_impl, xfinal_impl,
                       init, step, inverse, value, finalize)
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_udf_helpers.py -k window -v`
Expected: PASS (`test_window_running_sum`).

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/_sqlite_udf_helpers.py test/core/test_sqlite_udf_helpers.py
git -C /home/erik/projects/numbox commit -m "feat(udaf): register_window helper (xInverse/xValue, release-in-xFinal-only)"
```

---

### Task 3: Memory-safety & correctness guards

**Goal:** Lock in the lifecycle's memory safety and per-UDAF independence: meminfo balance, no cross-UDAF contamination, and `deterministic=True`.

**Files:**
- Test: `test/core/test_sqlite_udf_helpers.py` (add three tests)

**Acceptance Criteria:**
- [ ] Over many aggregate+window runs, `mi_alloc == mi_free` (the `export`/`release` balance and release-in-`xFinal`-only rule hold).
- [ ] Two distinct aggregates (`sum` and `sum-of-doubles`) sharing `sum_state_type` give correct independent results (`15` and `30`).
- [ ] `deterministic=True` registers cleanly and computes correctly.

**Verify:** `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_udf_helpers.py -k "leak or distinct or deterministic" -v` → 3 passed

**Steps:**

- [ ] **Step 1: Write the failing tests**

Append to `test/core/test_sqlite_udf_helpers.py`:
```python
@njit
def sum2_step(state, ctx, argc, argv_pp):  # distinct body => distinct anchor
    args = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
    state.total += 2 * sqlite3_value_int64(args[0])


def test_two_distinct_aggregates_no_collision():
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    h1 = register_aggregate(db, "my_sum", 1, sum_state_type,
                            sum_init, sum_step, sum_finalize)
    h2 = register_aggregate(db, "my_sum2", 1, sum_state_type,
                            sum_init, sum2_step, sum_finalize)
    v1, _c1 = _read1_int64(db, "SELECT __cap(my_sum(v)) FROM t")
    v2, _c2 = _read1_int64(db, "SELECT __cap(my_sum2(v)) FROM t")
    sqlite3_close(db)
    assert (v1, v2) == (15, 30)
    assert h1 is not None and h2 is not None


def test_deterministic_flag():
    db = _open_memory()
    _make_table(db, [1, 2, 3, 4, 5])
    h = register_aggregate(db, "my_sum_det", 1, sum_state_type,
                           sum_init, sum_step, sum_finalize, deterministic=True)
    val, _cap = _read1_int64(db, "SELECT __cap(my_sum_det(v)) FROM t")
    sqlite3_close(db)
    assert val == 15
    assert h is not None


def test_no_meminfo_leak():
    """The helper-generated lifecycle must preserve export/release balance."""
    from numba.core.runtime import nrt
    _nrt = nrt._nrt
    if not hasattr(_nrt, "memsys_enable_stats"):
        import pytest
        pytest.skip("NRT allocation stats unavailable")
    _nrt.memsys_enable_stats()
    test_aggregate_sum()          # warm up JIT / one-time allocs
    test_window_running_sum()
    before = nrt.rtsys.get_allocation_stats()
    for _ in range(10):
        test_aggregate_sum()
        test_window_running_sum()
    after = nrt.rtsys.get_allocation_stats()
    allocated = after.mi_alloc - before.mi_alloc
    freed = after.mi_free - before.mi_free
    assert allocated == freed, "meminfo imbalance: %d alloc, %d free" % (allocated, freed)
```

- [ ] **Step 2: Run to verify status**

Run: `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_udf_helpers.py -k "leak or distinct or deterministic" -v`
Expected: these exercise existing Task 1/2 code; if any FAIL it indicates a real lifecycle bug (e.g., missing `release_meminfo` ⇒ leak test fails; shared-anchor collision ⇒ distinct test gives wrong value). Fix in `_sqlite_udf_helpers.py` until they PASS. (No new product code is expected if Tasks 1–2 are correct.)

- [ ] **Step 3: Commit**

```bash
git -C /home/erik/projects/numbox add test/core/test_sqlite_udf_helpers.py
git -C /home/erik/projects/numbox commit -m "test(udaf): meminfo-leak, no-collision, and deterministic guards"
```

---

### Task 4: Cross-process caching + invalidation guards

**Goal:** Prove the load-bearing B claims hold end-to-end: the generated impls warm-HIT the cross-process cache (no unbounded `.nbc` growth), and editing a user function's numeric literal invalidates correctly (no stale machine code).

**Files:**
- Test: `test/core/test_sqlite_udf_helpers.py` (add two subprocess-based tests + a generated driver)

**Acceptance Criteria:**
- [ ] A warm second process does **not** add a new impl `.nbc` (stable count) — guards against the rejected C failure mode (unbounded growth).
- [ ] Editing `step` so only a numeric literal changes (`*2` → `*3`) makes a fresh process compute the **new** result (not a stale cached one) — guards the co_consts-sensitive digest.

**Verify:** `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_udf_helpers.py -k "xprocess or invalidation" -v` → 2 passed

**Steps:**

- [ ] **Step 1: Write the failing subprocess tests**

Append to `test/core/test_sqlite_udf_helpers.py`:
```python
import os                                            # noqa: E402
import subprocess                                    # noqa: E402
import sys                                           # noqa: E402
import textwrap                                      # noqa: E402


# A self-contained driver: defines a state type + sum UDAF in an importable
# module, registers it, runs SELECT sum(v), prints the result. {MULT} is the
# step multiplier (1 normally; flipped to prove invalidation).
_DRIVER = textwrap.dedent('''
    from ctypes import addressof, c_int64
    import numpy as np
    from numba import carray, cfunc, njit, types
    from numba.core import types as nb_types
    from numba.experimental import structref
    from numbox.core.bindings._sqlite_conn import sqlite3_close, sqlite3_open
    from numbox.core.bindings._sqlite_constants import SQLITE_OK, SQLITE_UTF8
    from numbox.core.bindings._sqlite_exec import sqlite3_exec
    from numbox.core.bindings._sqlite_result import sqlite3_result_int, sqlite3_result_int64
    from numbox.core.bindings._sqlite_udf import sqlite3_create_function_v2, sqlite3_user_data
    from numbox.core.bindings._sqlite_udf_helpers import register_aggregate
    from numbox.core.bindings._sqlite_value import sqlite3_value_int64
    from numbox.utils.cstrings import c_string
    from numbox.utils.lowlevel import _cast_int_to_void_p

    @structref.register
    class StT(nb_types.StructRef):
        def preprocess_fields(self, fields):
            return tuple((n, nb_types.unliteral(t)) for n, t in fields)
    class St(structref.StructRefProxy):
        def __new__(cls, total):
            return structref.StructRefProxy.__new__(cls, total)
    structref.define_proxy(St, StT, ["total"])
    st = StT([("total", nb_types.int64)])

    @njit
    def s_init():
        return St(nb_types.int64(0))
    @njit
    def s_step(state, ctx, argc, argv_pp):
        a = carray(_cast_int_to_void_p(argv_pp), (argc,), dtype=np.intp)
        state.total += {MULT} * sqlite3_value_int64(a[0])
    @njit
    def s_fin(state, ctx):
        sqlite3_result_int64(ctx, state.total)

    db_p = c_int64(0)
    with c_string(":memory:") as n:
        sqlite3_open(n, addressof(db_p))
    db = db_p.value
    with c_string("CREATE TABLE t(v INTEGER)") as p:
        sqlite3_exec(db, p, 0, 0, 0)
    for v in (1, 2, 3, 4, 5):
        with c_string("INSERT INTO t VALUES (%d)" % v) as p:
            sqlite3_exec(db, p, 0, 0, 0)
    h = register_aggregate(db, "f", 1, st, s_init, s_step, s_fin)
    buf = np.zeros(1, dtype=np.int64)
    @cfunc(types.void(types.intp, types.int32, types.intp))
    def cap(ctx, argc, argv):
        ud = sqlite3_user_data(ctx)
        a = carray(_cast_int_to_void_p(argv), (argc,), dtype=np.intp)
        o = carray(_cast_int_to_void_p(ud), (1,), dtype=np.int64)
        o[0] = sqlite3_value_int64(a[0]); sqlite3_result_int(ctx, 0)
    with c_string("cap") as cp:
        sqlite3_create_function_v2(db, cp, 1, SQLITE_UTF8, buf.ctypes.data, cap.address, 0, 0, 0)
    with c_string("SELECT cap(f(v)) FROM t") as sp:
        sqlite3_exec(db, sp, 0, 0, 0)
    print("RESULT", int(buf[0]))
''')


def _run_driver(tmp_path, cache_dir, mult):
    script = tmp_path / ("drv_%d.py" % mult)
    script.write_text(_DRIVER.replace("{MULT}", str(mult)))
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache_dir))
    out = subprocess.run([sys.executable, str(script)], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][0]
    return int(line.split()[1])


def _count_nbc(cache_dir):
    return sum(1 for _ in cache_dir.rglob("*.nbc"))


def test_xprocess_cache_no_growth(tmp_path):
    cache = tmp_path / "nbcache"
    cache.mkdir()
    assert _run_driver(tmp_path, cache, 1) == 15      # cold: compiles + writes cache
    n_cold = _count_nbc(cache)
    assert n_cold > 0
    assert _run_driver(tmp_path, cache, 1) == 15      # warm: must reuse, not append
    assert _count_nbc(cache) == n_cold, "warm run grew the cache (C failure mode)"


def test_invalidation_on_literal_edit(tmp_path):
    cache = tmp_path / "nbcache"
    cache.mkdir()
    assert _run_driver(tmp_path, cache, 1) == 15       # step: += 1*v  => 15
    assert _run_driver(tmp_path, cache, 3) == 45       # step: += 3*v  => 45 (not stale 15)
```

- [ ] **Step 2: Run to verify status**

Run: `/home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_udf_helpers.py -k "xprocess or invalidation" -v`
Expected: PASS if Task 1's digest uses the co_consts-sensitive `cloudpickle.dumps(py.__code__)`. If `test_invalidation_on_literal_edit` FAILS (returns 15 for `mult=3`), the digest is wrong (likely reverted to `co_code`) — fix `_digest` in `_sqlite_udf_helpers.py`.

- [ ] **Step 3: Commit**

```bash
git -C /home/erik/projects/numbox add test/core/test_sqlite_udf_helpers.py
git -C /home/erik/projects/numbox commit -m "test(udaf): cross-process cache reuse + co_consts invalidation guards"
```

---

### Task 5: Public API wiring + docs

**Goal:** Expose `register_aggregate`/`register_window` on the public bindings API and document the new module; full lint + test + docs build green.

**Files:**
- Modify: `numbox/core/bindings/__init__.py`
- Modify: `docs/numbox.core.bindings.rst`

**Acceptance Criteria:**
- [ ] `from numbox.core.bindings import register_aggregate, register_window` works.
- [ ] `docs/numbox.core.bindings.rst` has an `automodule` section for `_sqlite_udf_helpers` and a family-list entry.
- [ ] `flake8 --max-line-length=127 numbox/core/bindings/_sqlite_udf_helpers.py test/core/test_sqlite_udf_helpers.py` is clean.
- [ ] Sphinx build exits 0.

**Verify:** `cd /home/erik/projects/numbox/docs && /home/erik/projects/numbox/venv/bin/sphinx-build -b html . _build/html` → exit 0; and `/home/erik/projects/numbox/venv/bin/python -c "from numbox.core.bindings import register_aggregate, register_window; print('ok')"` → `ok`

**Steps:**

- [ ] **Step 1: Wire the star-import**

In `numbox/core/bindings/__init__.py`, add after the `_sqlite_udf` line (line 13):
```python
from numbox.core.bindings._sqlite_udf_helpers import *  # noqa: F401, F403
```

- [ ] **Step 2: Verify import**

Run: `/home/erik/projects/numbox/venv/bin/python -c "from numbox.core.bindings import register_aggregate, register_window; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Add the docs section**

In `docs/numbox.core.bindings.rst`, mirror the existing per-module `automodule` blocks (e.g. the `_sqlite_udf` one) and add, in the "Modules" area:
```rst
SQLite UDAF registration helpers
--------------------------------

.. automodule:: numbox.core.bindings._sqlite_udf_helpers
   :members:
   :undoc-members:
   :show-inheritance:
```
Also add `_sqlite_udf_helpers` to the "Bindings module conventions" family list in the same file (follow the exact format of the neighbouring entries — read the file first and match it).

- [ ] **Step 4: Lint + docs build + full suite**

Run:
```bash
/home/erik/projects/numbox/venv/bin/python -m flake8 --max-line-length=127 numbox/core/bindings/_sqlite_udf_helpers.py test/core/test_sqlite_udf_helpers.py
cd /home/erik/projects/numbox/docs && /home/erik/projects/numbox/venv/bin/sphinx-build -b html . _build/html
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; shutil.rmtree(pathlib.Path.home()/'.cache'/'numba', ignore_errors=True); [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]; print('cleared')"
cd /home/erik/projects/numbox && /home/erik/projects/numbox/venv/bin/python -m pytest test/core/test_sqlite_udf_helpers.py -v --durations=20
```
Expected: flake8 clean (no output); sphinx exit 0; pytest all PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/bindings/__init__.py docs/numbox.core.bindings.rst
git -C /home/erik/projects/numbox commit -m "feat(udaf): export register_aggregate/register_window + docs"
```

---

## Notes for the implementer

- **Always use the venv python** (`/home/erik/projects/numbox/venv/bin/python`) for every python/pytest/flake8/sphinx call — never bare `python`.
- **Clean caches before each pytest run** (numba `~/.cache/numba` + `__pycache__`) per project rule, via the python one-liner shown in Task 1 Step 1 (never `find -exec rm`).
- **The digest MUST stay co_consts-sensitive** (`cloudpickle.dumps(py.__code__)`). Reverting to `co_code` reintroduces the stale-cache bug Task 4 guards against (verified in spec spike S3).
- **Do not seed the exec namespace with a bare `{}`** — it must carry `__name__` (we use `getmodule(_file_anchor).__dict__`), or warm cache reload crashes (spec spike S1).
- The state-type class and user `@njit` functions used by callers must live in an importable module (the tests define them at module scope for this reason).
