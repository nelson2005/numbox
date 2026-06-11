# compile_kernel Review-Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the verified findings of the 2026-06-11 deep-dive review of `compile_kernel` (fork PR #49), foremost the content-addressed cache identity that can serve silently wrong results under the default `cache=True`.

**Architecture:** All work lands on `feat/variable-compile-kernel` in `numbox/core/variable/compile_kernel.py` plus its tests, the benchmark, and the rst docs. The centerpiece replaces source-text hashing (`_safe_getsource`) with a structured *fingerprint* of each formula (code-object bytes/consts/names, default values, closure-cell values, referenced-global values, defining module, compile flags); anything un-fingerprintable downgrades that kernel to uncached (never wrong, just not reused). Around it: honest `cache`-flag precedence, anchors written only when caching, a one-line gc fix, eager input validation, and docs that tell the truth.

**Tech Stack:** Python 3.12 / numba 0.65.1 / numpy; pytest; sphinx. Venv interpreter: `/home/erik/projects/numbox/venv/bin/python` (always absolute, never bare `python`/`pytest`).

**Review evidence:** `/home/erik/reviews/numbox-pr49-deep-dive/report.md` (+ `findings/F-XX.json`). Finding IDs cited per task. Out of scope: cherry-picking to `upstream-pr/variable-compile-kernel` / upstream PR #23 (separate, consent-gated step), and any change to `numbox/core/variable/variable.py` or `numbox/utils/preprocessing.py` (upstream-owned substrate — every fix here stays inside the feature's own files).

---

## Conventions (read once, used by every task)

- **Repo root:** `/home/erik/projects/numbox`; branch `feat/variable-compile-kernel`. Never use `cd`; use absolute paths and `git -C /home/erik/projects/numbox`.
- **`<clean-caches>`** in Verify lines stands for this exact command (run before every pytest invocation):

```bash
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
```

- **`<pytest>`** stands for `/home/erik/projects/numbox/venv/bin/python -m pytest`.
- **Flake8 (both rule sets must pass before any commit):**

```bash
/home/erik/projects/numbox/venv/bin/python -m flake8 /home/erik/projects/numbox/numbox /home/erik/projects/numbox/test
/home/erik/projects/numbox/venv/bin/python -m flake8 --select=E9,F63,F7,F82,F401 /home/erik/projects/numbox/numbox /home/erik/projects/numbox/test
```

- Commit messages follow the branch's existing style (`compile_kernel: <what>`), one commit per task, no AI attribution, no Co-Authored-By.
- Tests that need cache isolation use the existing subprocess pattern from `test/core/test_compile_kernel.py:173` (`env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "nbcache")}`).

---

## File structure

| File | Role in this plan |
|---|---|
| `numbox/core/variable/compile_kernel.py` | All production changes (Tasks 0–6). Grows ~140 lines (fingerprint machinery replaces `_safe_getsource`); stays a single focused module per house style. |
| `test/core/test_compile_kernel.py` | New/adapted tests for every task; two existing `_safe_getsource` tests (lines 130, 388) are rewritten in Task 3. |
| `test/compile_kernel_benchmark.py` | Task 8 only (per-user tmp dir). |
| `docs/numbox.core.variable.rst` | Task 9 (cache semantics, fallback, scale envelope; section starts at line 220). |
| `test/auxiliary_utils.py` | Read-only — Task 7 reuses `assert_njit_cache_survives_subprocess_roundtrip` (line 78). |

---

### Task 0: Release the kernel exec namespace (F-06)

**Goal:** A `CompiledKernel` whose owner drops it becomes garbage-collectable even after a real JIT compile — `return ns.pop(name)` removes the dispatcher's self-reference from its own `__globals__`.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py:177`
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] `weakref.ref(ck.kernel)` is dead after `del ck` + `gc.collect()` ×3, for a kernel that was **executed** (real compile, `cache=False`)
- [ ] Multi-signature compilation still works after the pop (float call then int call on the same kernel)
- [ ] Full feature suite still green

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -v` → all pass incl. `test_kernel_dispatcher_collectable_after_release`

**Steps:**

- [ ] **Step 1: Write the failing test** (append to `test/core/test_compile_kernel.py`)

```python
def test_kernel_dispatcher_collectable_after_release():
    import gc
    import weakref

    def f(x):
        return x * 2.0

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    ck = compile_kernel(g, "calc.y", cache=False)
    assert ck.execute({"ext": {"x": 3.0}}) == {"calc.y": 6.0}
    assert ck.kernel(4) == (8.0,)
    ref = weakref.ref(ck.kernel)
    del ck
    for _ in range(3):
        gc.collect()
    assert ref() is None
```

(The `ck.execute(...)` float call and the `ck.kernel(4)` int call together exercise a second-signature compile before release — the case the `ns.pop` fix must keep working.)

- [ ] **Step 2: Run it — expect FAIL** (`assert ref() is None` fails; the executed dispatcher is pinned by the exec-namespace cycle, see `findings/F-06.adjudication.json`)

Run: `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py::test_kernel_dispatcher_collectable_after_release -v`
Expected: FAIL on the weakref assertion

- [ ] **Step 3: Apply the fix** in `_compile` (line 177)

```python
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns.pop(name)
```

(The kernel never references its own global name — second-signature compiles read only `f_*`/`njit`/`_kernel_jit_options` from `ns`; verified empirically during the review.)

- [ ] **Step 4: Re-run the test and the full feature file — expect PASS**
- [ ] **Step 5: Flake8 (both rule sets), then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: pop kernel from its exec namespace so released kernels are collectable"
```

---

### Task 1: Honest cache semantics and lazy anchors (F-08, F-09)

**Goal:** `cache` follows a documented precedence (explicit param > `jit_options["cache"]` > `NUMBOX_JIT_OPTIONS` > `True`), anchors are written only when actually caching, and an unwritable cache dir degrades to uncached compilation with a warning instead of crashing.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (`compile_kernel` signature, `_compile` lines 159–177, imports)
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] `compile_kernel(..., cache=False)` writes **zero** files under `NUMBA_CACHE_DIR` (no anchor, no `.nbc`/`.nbi`)
- [ ] `NUMBOX_JIT_OPTIONS='{"cache": false}'` (env) and `jit_options={"cache": False}` (param) each disable caching when `cache` is not passed
- [ ] Explicit `cache=True` overrides `jit_options={"cache": False}`
- [ ] With a read-only `NUMBA_CACHE_DIR`, `compile_kernel` succeeds uncached and emits a `UserWarning` (POSIX-only test)
- [ ] Existing cross-process cache tests still green

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -v -k "cache_precedence or cache_false or readonly_cache"` → 5 new tests pass; then full file green

**Steps:**

- [ ] **Step 1: Write the failing tests** (append; subprocess pattern because `NUMBA_CACHE_DIR`/`NUMBOX_JIT_OPTIONS` are read at import)

```python
_CACHE_PROBE = """
    import pathlib
    import sys
    from numbox.core.variable.variable import Graph
    from numbox.core.variable.compile_kernel import compile_kernel

    def f(x):
        return x + 1.0

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    kwargs = eval(sys.argv[1])
    ck = compile_kernel(g, "calc.y", **kwargs)
    print(ck.execute({"ext": {"x": 1.0}})["calc.y"])
"""


def _run_cache_probe(tmp_path, kwargs_src, extra_env=None):
    f = tmp_path / "probe.py"
    f.write_text(textwrap.dedent(_CACHE_PROBE))
    cache_dir = tmp_path / "nbcache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir), **(extra_env or {})}
    p = subprocess.run(
        [sys.executable, str(f), kwargs_src],
        capture_output=True, text=True, env=env,
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "2.0"
    files = [q for q in cache_dir.rglob("*") if q.is_file()] if cache_dir.exists() else []
    return files, p.stderr


def test_cache_false_writes_nothing(tmp_path):
    files, _ = _run_cache_probe(tmp_path, "{'cache': False}")
    assert files == []


def test_cache_precedence_env_knob(tmp_path):
    files, _ = _run_cache_probe(
        tmp_path, "{}", extra_env={"NUMBOX_JIT_OPTIONS": '{"cache": false}'})
    assert files == []


def test_cache_precedence_jit_options(tmp_path):
    files, _ = _run_cache_probe(tmp_path, "{'jit_options': {'cache': False}}")
    assert files == []


def test_cache_precedence_param_wins(tmp_path):
    files, _ = _run_cache_probe(
        tmp_path, "{'cache': True, 'jit_options': {'cache': False}}")
    assert files != []


@pytest.mark.skipif(sys.platform == "win32", reason="chmod-based read-only dir is POSIX-only")
def test_readonly_cache_dir_degrades_gracefully(tmp_path):
    cache_dir = tmp_path / "nbcache"
    cache_dir.mkdir()
    cache_dir.chmod(0o500)
    try:
        files, stderr = _run_cache_probe(tmp_path, "{}")
        assert files == []
        assert "cache directory unusable" in stderr
    finally:
        cache_dir.chmod(0o700)
```

- [ ] **Step 2: Run — expect FAIL** (today: `cache=False` still writes an anchor; env/jit_options knobs are overridden; read-only dir raises `PermissionError`)

- [ ] **Step 3: Implement.** Change the public signature (line 215) to `cache=None`:

```python
def compile_kernel(graph, required, *, jit_options=None, cache=None):
```

Rewrite `_compile` (keep the hashing exactly as-is for now — Task 3 replaces it):

```python
def _compile(source, bindings, jit_options, cache):
    """Content-addressed compile of the kernel source into an @njit dispatcher."""
    formula_src = "\n".join(_safe_getsource(f) for f in bindings.values())
    hash_text = source + "\n# formulas:\n" + formula_src
    digest = hashlib.sha256(hash_text.encode("utf-8")).hexdigest()[:16]
    name = f"_kernel_{digest}"
    opts = {**_default_jit_options, **(jit_options or {})}
    if cache is not None:
        opts["cache"] = cache
    opts.setdefault("cache", True)
    final_src = "@njit(**_kernel_jit_options)\n" + source.replace(
        "def _kernel(", f"def {name}(", 1
    )
    anchor = _anchor_root(_ANCHOR_SUBDIR) / f"_kernel_{digest}.py"
    if opts["cache"]:
        try:
            anchor.parent.mkdir(parents=True, exist_ok=True)
            _materialize_anchor(anchor, final_src)
        except OSError as e:
            warnings.warn(
                f"compile_kernel: cache directory unusable ({e}); "
                f"compiling without an on-disk cache"
            )
            opts["cache"] = False
    code = compile(final_src, str(anchor), "exec")
    # __name__ must be an importable module so numba can rebuild the cached
    # overload's environment in another process (importlib.import_module needs
    # a real name, not None); mirrors make_graph / make_structref.
    ns = {**bindings, "njit": njit, "_kernel_jit_options": opts, "__name__": __name__}
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns.pop(name)
```

Imports: add `import warnings`; replace the `_anchor_path` import with `_anchor_root`:

```python
from numbox.utils.preprocessing import (
    _anchor_root, _materialize_anchor, _orphan_anchor_sweep,
)
```

(`compile()` accepts a filename whose file does not exist — when not caching, the anchor path serves only as the traceback filename. The digest and anchor filename are unchanged from `_anchor_path`'s scheme, so warm caches survive this task.)

- [ ] **Step 4: Run the 5 new tests + full feature file — expect PASS** (incl. `test_compile_cache_survives_fresh_process`)
- [ ] **Step 5: Flake8 (both rule sets), then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: cache precedence (param > jit_options > env), anchors only when caching, graceful unwritable-cache fallback"
```

---

### Task 2: Formula fingerprint machinery (F-01..F-05 core)

**Goal:** A pure, unit-tested `_formula_fingerprint(formula) -> (text, cacheable)` that injectively captures everything numba freezes into a compiled artifact: code-object state, default values, closure-cell values, referenced-global values (recursing into helper functions), defining module, and dispatcher `targetoptions` — with a per-object fallback (`cacheable=False`) for anything it cannot canonicalize.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (new private helpers; `_safe_getsource` stays untouched until Task 3)
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] Two different lambdas defined on one source line fingerprint differently
- [ ] Changing a module-level global, a helper function's body, or a default-arg value changes the fingerprint
- [ ] Two 2000-element arrays differing only at index 500 (closure cells) fingerprint differently
- [ ] Set-valued cells fingerprint identically regardless of insertion order; module globals canonicalize by name; mutually-recursive helpers terminate
- [ ] Un-canonicalizable values (object with custom/raising `__repr__` in a cell or global) → `cacheable=False`
- [ ] cres/`CompileResultWAP` formulas → `cacheable=False`

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -v -k fingerprint` → all new unit tests pass

**Steps:**

- [ ] **Step 1: Write the failing unit tests** (append)

```python
def test_fingerprint_same_line_lambdas_distinct():
    from numbox.core.variable.compile_kernel import _formula_fingerprint
    f10, f1000 = (lambda y: y * 10.0), (lambda y: y * 1000.0)
    fp_a, ok_a = _formula_fingerprint(f10)
    fp_b, ok_b = _formula_fingerprint(f1000)
    assert ok_a and ok_b
    assert fp_a != fp_b


def test_fingerprint_covers_globals_helpers_defaults():
    from numbox.core.variable.compile_kernel import _formula_fingerprint
    src = textwrap.dedent("""
        SCALE = {scale}
        def helper(v):
            return v {op} 1.0
        def f(x, m={default}):
            return helper(x) * SCALE * m
    """)
    variants = {}
    for key, (scale, op, default) in {
        "base": ("2.0", "+", "1.0"),
        "global": ("3.0", "+", "1.0"),
        "helper": ("2.0", "-", "1.0"),
        "default": ("2.0", "+", "5.0"),
    }.items():
        ns = {}
        exec(compile(src.format(scale=scale, op=op, default=default), f"<fp-{key}>", "exec"), ns)
        fp, ok = _formula_fingerprint(ns["f"])
        assert ok
        variants[key] = fp
    assert len(set(variants.values())) == 4


def test_fingerprint_large_array_cells_distinct():
    from numbox.core.variable.compile_kernel import _formula_fingerprint
    def factory(a):
        return lambda x: x + a[500]
    a1, a2 = np.zeros(2000), np.zeros(2000)
    a2[500] = 1.0
    assert repr(a1) == repr(a2)          # the old scheme's blind spot
    fp1, _ = _formula_fingerprint(factory(a1))
    fp2, _ = _formula_fingerprint(factory(a2))
    assert fp1 != fp2


def test_fingerprint_set_cells_order_stable():
    from numbox.core.variable.compile_kernel import _formula_fingerprint
    def factory(s):
        return lambda x: x if x in s else -x
    s1 = {"alpha", "beta", "gamma"}
    s2 = set(reversed(sorted(s1)))
    assert _formula_fingerprint(factory(s1))[0] == _formula_fingerprint(factory(s2))[0]


def test_fingerprint_recursive_helpers_terminate():
    from numbox.core.variable.compile_kernel import _formula_fingerprint
    ns = {}
    exec(textwrap.dedent("""
        def even(n):
            return n == 0 or odd(n - 1)
        def odd(n):
            return n != 0 and even(n - 1)
        def f(x):
            return x if even(int(x)) else -x
    """), ns)
    fp, ok = _formula_fingerprint(ns["f"])
    assert ok and "recursive(" in fp


def test_fingerprint_fallback_paths():
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    class Boom:
        def __repr__(self):
            raise RuntimeError("no repr")

    def factory(cfg):
        return lambda x: x

    fp, ok = _formula_fingerprint(factory(Boom()))
    assert not ok and " @" in fp

    @cres(float64(float64))
    def wap(x):
        return x * 3.0

    fp2, ok2 = _formula_fingerprint(wap)
    assert not ok2
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: cannot import name '_formula_fingerprint'`)

- [ ] **Step 3: Implement** (insert after `_wrap_formula`; add imports `import numpy as np`, `from types import CodeType, FunctionType, ModuleType`)

```python
class _Unfingerprintable(Exception):
    """A value the cache digest cannot canonicalize; the kernel goes uncached."""


def _canon_value(value, seen):
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return repr(value)
    if isinstance(value, np.ndarray):
        data = np.ascontiguousarray(value)
        raw = hashlib.sha256(data.tobytes()).hexdigest()
        return f"ndarray({data.dtype.str};{data.shape};{raw})"
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__}[" + ",".join(_canon_value(v, seen) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}[" + ",".join(sorted(_canon_value(v, seen) for v in value)) + "]"
    if isinstance(value, dict):
        items = sorted((_canon_value(k, seen), _canon_value(v, seen)) for k, v in value.items())
        return "dict[" + ",".join(f"{k}:{v}" for k, v in items) + "]"
    if isinstance(value, ModuleType):
        return f"module({value.__name__})"
    if isinstance(value, Dispatcher):
        topts = _canon_value(dict(getattr(value, "targetoptions", {}) or {}), seen)
        return f"dispatcher({_fingerprint_function(value.py_func, seen)};{topts})"
    if isinstance(value, FunctionType):
        return f"function({_fingerprint_function(value, seen)})"
    raise _Unfingerprintable(type(value).__name__)


def _fingerprint_codeobj(code, seen):
    consts = ",".join(
        _fingerprint_codeobj(c, seen) if isinstance(c, CodeType) else _canon_value(c, seen)
        for c in code.co_consts
    )
    return (
        f"code({code.co_code.hex()};flags={code.co_flags};argc={code.co_argcount};"
        f"kwonly={code.co_kwonlyargcount};names={','.join(code.co_names)};consts=[{consts}])"
    )


def _referenced_global_names(code):
    names = set(code.co_names)
    for c in code.co_consts:
        if isinstance(c, CodeType):
            names |= _referenced_global_names(c)
    return names


def _fingerprint_function(func, seen):
    if id(func) in seen:
        return f"recursive({func.__qualname__})"
    seen = seen | {id(func)}
    code = func.__code__
    cells = []
    for name, cell in zip(code.co_freevars, func.__closure__ or ()):
        try:
            contents = cell.cell_contents
        except ValueError as e:
            raise _Unfingerprintable("empty closure cell") from e
        cells.append(f"{name}={_canon_value(contents, seen)}")
    hashed_globals = []
    for name in sorted(_referenced_global_names(code)):
        if name in func.__globals__:
            hashed_globals.append(f"{name}={_canon_value(func.__globals__[name], seen)}")
    return (
        f"func({func.__module__}:{func.__qualname__};{_fingerprint_codeobj(code, seen)};"
        f"defaults={_canon_value(func.__defaults__ or (), seen)};"
        f"kwdefaults={_canon_value(func.__kwdefaults__ or {}, seen)};"
        f"closure=[{';'.join(cells)}];globals=[{';'.join(hashed_globals)}])"
    )


def _formula_fingerprint(formula):
    """Behavioral identity of a formula for the cache digest.

    Returns ``(text, cacheable)``. The text covers every value channel
    numba freezes into a compiled artifact: code-object bytecode/consts/
    names, default-argument values, closure-cell values, the values of
    referenced module-level globals (recursing into helper functions and
    dispatchers, with cycle protection), the defining module, and
    dispatcher targetoptions. Builtins resolve outside ``__globals__``
    and are deliberately not hashed. Any value with no canonical form
    makes the formula un-fingerprintable: the returned text is then a
    per-object placeholder and ``cacheable`` is False, so the kernel is
    compiled without an on-disk cache -- never reused, never wrong.
    """
    target = getattr(formula, "py_func", None)
    if target is None:
        target = getattr(formula, "__wrapped__", None)
    if target is None:
        target = formula
    if not isinstance(target, FunctionType):
        return f"{repr(formula)} @{id(formula)}", False
    extra = ""
    if isinstance(formula, Dispatcher):
        extra = ";targetoptions=" + _canon_value(dict(formula.targetoptions or {}), set())
    try:
        return _fingerprint_function(target, set()) + extra, True
    except _Unfingerprintable:
        return f"{repr(formula)} @{id(formula)}", False
```

- [ ] **Step 4: Run the fingerprint tests — expect PASS**; full feature file still green (nothing is wired yet)
- [ ] **Step 5: Flake8 (both rule sets), then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: structured formula fingerprint (code, defaults, cells, globals, flags) with uncacheable fallback"
```

---

### Task 3: Wire the fingerprint into the cache digest (F-01..F-05, F-07, F-16 end-to-end)

**Goal:** `_compile` hashes fingerprints plus the kernel's effective compile flags instead of `getsource` text; un-fingerprintable formulas force the kernel uncached (no anchor, no numba cache, no warning noise); the stale/collision vectors from the review are demonstrably closed.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (`_compile`; delete `_safe_getsource`; module docstring line 7 note)
- Test: `test/core/test_compile_kernel.py` (new regression battery; rewrite the two `_safe_getsource` tests at lines 130 and 388)

**Acceptance Criteria:**
- [ ] Editing only a module-level constant between two subprocess runs sharing one `NUMBA_CACHE_DIR` changes the result (no stale hit)
- [ ] Two kernels over 2000-element arrays differing at index 500 return different results (same process, `cache=True`)
- [ ] Two same-line lambda kernels return different results
- [ ] `jit_options={"error_model": "python"}` vs default produce distinct cached kernels (python model raises `ZeroDivisionError`; default returns `inf`)
- [ ] A cres-formula kernel writes zero anchor files and zero numba cache files, emits no `NumbaWarning`, and still computes correctly
- [ ] A formula closing over an un-canonicalizable object compiles, runs correctly, and is uncached
- [ ] Existing cross-process tests (`test_compile_cache_survives_fresh_process`, `test_cache_no_skeleton_collision`) still green

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -v` → full file green incl. the new `-k digest` battery

**Steps:**

- [ ] **Step 1: Write the failing regression battery** (append)

```python
_DIGEST_PROBE = """
    import sys
    sys.path.insert(0, {moddir!r})
    import formulas_mod
    from numbox.core.variable.variable import Graph
    from numbox.core.variable.compile_kernel import compile_kernel

    g = Graph({{"calc": [{{"name": "y", "inputs": {{"x": "ext"}}, "formula": formulas_mod.f}}]}}, ["ext"])
    ck = compile_kernel(g, "calc.y")
    print(ck.execute({{"ext": {{"x": 10.0}}}})["calc.y"])
"""


def test_digest_global_change_invalidates_cache(tmp_path):
    moddir = tmp_path / "mods"
    moddir.mkdir()
    mod = moddir / "formulas_mod.py"
    runner = tmp_path / "run.py"
    runner.write_text(textwrap.dedent(_DIGEST_PROBE.format(moddir=str(moddir))))
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "nbcache")}

    mod.write_text("SCALE = 2.0\ndef f(x):\n    return x * SCALE\n")
    p1 = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True, env=env)
    assert p1.returncode == 0, p1.stderr
    assert p1.stdout.strip() == "20.0"

    mod.write_text("SCALE = 3.0\ndef f(x):\n    return x * SCALE\n")
    p2 = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True, env=env)
    assert p2.returncode == 0, p2.stderr
    assert p2.stdout.strip() == "30.0"


def test_digest_large_array_closure_no_collision(tmp_path):
    def factory(a):
        return lambda x: x + a[500]

    results = []
    for fill in (0.0, 1.0):
        a = np.zeros(2000)
        a[500] = fill
        g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": factory(a)}]}, ["ext"])
        results.append(compile_kernel(g, "calc.y").execute({"ext": {"x": 1.0}})["calc.y"])
    assert results == [1.0, 2.0]


def test_digest_same_line_lambdas_no_collision():
    f10, f1000 = (lambda y: y * 10.0), (lambda y: y * 1000.0)
    results = []
    for f in (f10, f1000):
        g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
        results.append(compile_kernel(g, "calc.y").execute({"ext": {"x": 1.0}})["calc.y"])
    assert results == [10.0, 1000.0]


def test_digest_includes_jit_flags():
    def f(x):
        return 1.0 / x

    def build():
        return Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])

    ck_default = compile_kernel(build(), "calc.y")
    assert ck_default.execute({"ext": {"x": 0.0}})["calc.y"] == np.inf
    ck_python = compile_kernel(build(), "calc.y", jit_options={"error_model": "python"})
    with pytest.raises(ZeroDivisionError):
        ck_python.execute({"ext": {"x": 0.0}})


def test_digest_cres_kernel_uncached_and_quiet(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent("""
        import warnings
        from numba.core.types import float64
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel
        from numbox.utils.highlevel import cres

        @cres(float64(float64))
        def f(x):
            return x - 1.0

        g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ck = compile_kernel(g, "calc.y")
        print(ck.execute({"ext": {"x": 3.0}})["calc.y"])
    """))
    cache_dir = tmp_path / "nbcache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}
    p = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "2.0"
    files = [q for q in cache_dir.rglob("*") if q.is_file()] if cache_dir.exists() else []
    assert files == []


def test_digest_unfingerprintable_cell_runs_uncached():
    class Cfg:
        def __repr__(self):
            raise RuntimeError("no repr")
    cfg = Cfg()
    cfg.v = 7.0

    def factory(c):
        captured = c.v
        return lambda x: x + captured

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": factory(cfg)}]}, ["ext"])
    assert compile_kernel(g, "calc.y").execute({"ext": {"x": 1.0}})["calc.y"] == 8.0
```

(Note `factory` captures `c.v` — a float cell — so this exercises a *fingerprintable* closure; to exercise the fallback, also keep a variant capturing `c` itself:)

```python
def test_digest_object_cell_falls_back_uncached(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent("""
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        class Cfg:
            v = 7.0

        cfg = Cfg()

        def factory(c):
            return lambda x: x + c.v

        g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": factory(cfg)}]}, ["ext"])
        print(compile_kernel(g, "calc.y").execute({"ext": {"x": 1.0}})["calc.y"])
    """))
    cache_dir = tmp_path / "nbcache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}
    p = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "8.0"
    files = [q for q in cache_dir.rglob("*") if q.is_file()] if cache_dir.exists() else []
    assert files == []
```

- [ ] **Step 2: Run — expect FAIL** on: global-change (stale `20.0`), large-array (`[1.0, 1.0]`), same-line (`[10.0, 10.0]`), jit-flags (no `ZeroDivisionError`), cres (anchor file present + `NumbaWarning` under `-W error`), object-cell (anchor present)

- [ ] **Step 3: Rewire `_compile`** — replace the first two lines of the body and the cache gate:

```python
def _compile(source, bindings, jit_options, cache):
    """Content-addressed compile of the kernel source into an @njit dispatcher."""
    fingerprints = []
    cacheable = True
    for fg, formula in bindings.items():
        fp, ok = _formula_fingerprint(formula)
        fingerprints.append(f"{fg}: {fp}")
        cacheable = cacheable and ok
    opts = {**_default_jit_options, **(jit_options or {})}
    if cache is not None:
        opts["cache"] = cache
    opts.setdefault("cache", True)
    flags = {k: v for k, v in opts.items() if k != "cache"}
    hash_text = (
        "ck-digest-v2\n" + source
        + "\n# formulas:\n" + "\n".join(fingerprints)
        + "\n# flags: " + _canon_value(flags, set())
    )
    if not cacheable:
        opts["cache"] = False
    digest = hashlib.sha256(hash_text.encode("utf-8")).hexdigest()[:16]
    name = f"_kernel_{digest}"
    final_src = "@njit(**_kernel_jit_options)\n" + source.replace(
        "def _kernel(", f"def {name}(", 1
    )
    anchor = _anchor_root(_ANCHOR_SUBDIR) / f"_kernel_{digest}.py"
    if opts["cache"]:
        try:
            anchor.parent.mkdir(parents=True, exist_ok=True)
            _materialize_anchor(anchor, final_src)
        except OSError as e:
            warnings.warn(
                f"compile_kernel: cache directory unusable ({e}); "
                f"compiling without an on-disk cache"
            )
            opts["cache"] = False
    code = compile(final_src, str(anchor), "exec")
    # __name__ must be an importable module so numba can rebuild the cached
    # overload's environment in another process (importlib.import_module needs
    # a real name, not None); mirrors make_graph / make_structref.
    ns = {**bindings, "njit": njit, "_kernel_jit_options": opts, "__name__": __name__}
    exec(code, ns)  # nosec B102 - JIT codegen of internal source
    return ns.pop(name)
```

Then **delete `_safe_getsource`** (lines 75–103) and its now-unused `from inspect import getsource` import. Update the module docstring's cache sentence to describe the fingerprint scheme.

- [ ] **Step 4: Rewrite the two `_safe_getsource` tests** preserving their intent:
  - `test_safe_getsource_named_function_and_cres` (line 130) → `test_fingerprint_named_function_and_cres`: a named function fingerprints with `cacheable=True` and the text contains its qualname; a cres WAP returns `cacheable=False` and must not raise.
  - `test_safe_getsource_repr_fallback_is_per_object` (line 388) → `test_fingerprint_fallback_is_per_object`: two distinct un-sourceable objects (the existing `_Konst` instances) yield different fallback texts (the ` @<id>` suffix) and both `cacheable=False`.

```python
def test_fingerprint_named_function_and_cres():
    from numbox.core.variable.compile_kernel import _formula_fingerprint

    def named(x):
        return x + 1.0

    fp, ok = _formula_fingerprint(named)
    assert ok and "named" in fp

    @cres(float64(float64))
    def wap(x):
        return x * 3.0

    fp2, ok2 = _formula_fingerprint(wap)
    assert not ok2 and " @" in fp2
```

(Adapt the per-object test by reusing the file's existing `_Konst` fixtures from line 388's body; assert the two fingerprints differ and both flags are `False`.)

- [ ] **Step 5: Run the full feature file — expect PASS**, including both pre-existing cross-process cache tests
- [ ] **Step 6: Flake8 (both rule sets), then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: digest v2 - fingerprint formulas and compile flags; unfingerprintable formulas compile uncached without anchors"
```

---

### Task 4: Wider formula acceptance + eager signature checks (F-12, F-24, F-13)

**Goal:** `@vectorize` DUFuncs and `@cfunc` CFuncs pass through unwrapped; non-callable formulas and arity mismatches fail at `compile_kernel()` time with the variable's qual_name in the message; kw-only formulas no longer reach numba's bare `IndexError`.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (`_wrap_formula`, `_generate_body`, imports)
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] A `@vectorize` DUFunc formula and a `@cfunc` CFunc formula both compile and compute correctly in a fused kernel
- [ ] `formula="lambda y: y"` (a string) raises `TypeError` naming the variable's qual_name at compile time
- [ ] A 2-input node with a 1-arg formula raises `ValueError` naming the qual_name at compile time; same for a kw-only formula
- [ ] A `*args` formula still compiles; cres formulas skip the check (lazy contract intact — `test_non_jittable_formula_fails_at_first_call_not_compile` stays green)

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -v -k "dufunc or cfunc_formula or not_callable or arity"` then full file

**Steps:**

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_dufunc_and_cfunc_formulas_accepted():
    from numba import vectorize, cfunc

    d = vectorize(lambda a: a + 0.5)
    c = cfunc(float64(float64))(lambda a: a * 2.0)
    g = Graph({"calc": [
        {"name": "u", "inputs": {"x": "ext"}, "formula": d},
        {"name": "v", "inputs": {"u": "calc"}, "formula": c},
    ]}, ["ext"])
    out = compile_kernel(g, "calc.v").execute({"ext": {"x": 1.0}})
    assert out == {"calc.v": 3.0}


def test_not_callable_formula_rejected_eagerly():
    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": "lambda y: y"}]}, ["ext"])
    with pytest.raises(TypeError, match=r"calc\.y.*not callable"):
        compile_kernel(g, "calc.y")


def test_arity_mismatch_rejected_eagerly():
    def one_arg(x):
        return x

    g = Graph({"calc": [{"name": "y", "inputs": {"a": "ext", "b": "ext"}, "formula": one_arg}]}, ["ext"])
    with pytest.raises(ValueError, match=r"calc\.y.*2 declared input"):
        compile_kernel(g, "calc.y")


def test_kwonly_formula_rejected_eagerly():
    def kw_only(*, y):
        return y

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": kw_only}]}, ["ext"])
    with pytest.raises(ValueError, match=r"calc\.y"):
        compile_kernel(g, "calc.y")


def test_varargs_formula_still_accepted():
    def star(*vals):
        return vals[0] + vals[1]

    g = Graph({"calc": [{"name": "y", "inputs": {"a": "ext", "b": "ext"}, "formula": star}]}, ["ext"])
    out = compile_kernel(g, "calc.y").execute({"ext": {"a": 1.0, "b": 2.0}})
    assert out == {"calc.y": 3.0}
```

- [ ] **Step 2: Run — expect FAIL** (DUFunc/CFunc raise `TypeError` from `njit()`; string formula fails late with an untyped-global error; arity/kw-only fail only at first call)

- [ ] **Step 3: Implement.** Imports:

```python
import inspect
from numba.core.ccallback import CFunc
from numba.np.ufunc.dufunc import DUFunc
```

`_wrap_formula`:

```python
def _wrap_formula(formula):
    """Return an njit-callable for `formula`; plain-Python callables are njit-wrapped."""
    if isinstance(formula, (Dispatcher, CompileResultWAP, DUFunc, CFunc)):
        return formula
    if not callable(formula):
        raise TypeError(f"formula {formula!r} is not callable")
    return njit(formula)
```

New eager check (insert above `_generate_body`):

```python
def _check_formula_arity(formula, n_inputs, qual_name):
    target = getattr(formula, "py_func", None) or getattr(formula, "__wrapped__", None) or formula
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return
    try:
        sig.bind(*range(n_inputs))
    except TypeError as e:
        raise ValueError(
            f"{qual_name!r}: formula signature {sig} cannot accept its "
            f"{n_inputs} declared input(s) passed positionally ({e})"
        ) from None
```

In `_generate_body`, replace the binding line (`bindings[fg] = _wrap_formula(var.formula)`) with:

```python
        try:
            bindings[fg] = _wrap_formula(var.formula)
        except TypeError as e:
            raise TypeError(f"{var.qual_name()!r}: {e}") from e
        _check_formula_arity(var.formula, len(node.inputs), var.qual_name())
```

(`inspect.signature` raises `TypeError` on `CompileResultWAP`, so cres formulas skip the arity check and keep the documented lazy contract. The fingerprint from Task 2 already resolves DUFunc/CFunc via `__wrapped__`, so they cache normally.)

- [ ] **Step 4: Run new tests + full file — expect PASS**
- [ ] **Step 5: Flake8 (both rule sets), then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: accept DUFunc/CFunc formulas; eager callable and arity validation with qual_name context"
```

---

### Task 5: Validate `required` and guard the External footguns (F-11, F-10, F-17)

**Goal:** Malformed `required` entries fail fast with the offending entry named; a `required` name that only exists because External auto-created it during this compile warns loudly; an External variable carrying a formula is rejected (CompiledGraph would compute it, the kernel would not).

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (`compile_kernel`, `_generate_body`, imports)
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] Non-string entry → `TypeError` naming the entry; dot-less entry → `ValueError` naming the entry
- [ ] Unknown name in a `Variables` namespace → `ValueError` (wrapping the substrate `KeyError`) mentioning the name
- [ ] A typo'd External qual_name still compiles (substrate contract) but emits a `UserWarning` matching `did not exist before compilation`
- [ ] An External variable with a non-None formula → `ValueError` naming the qual_name
- [ ] All existing tests green (the empty-`required` `ValueError` is unchanged)

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -v -k "required_valid or external_typo or external_formula"` then full file

**Steps:**

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_required_validation_messages():
    def f(x):
        return x

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    with pytest.raises(TypeError, match=r"required entries.*42"):
        compile_kernel(g, ["calc.y", 42])
    with pytest.raises(ValueError, match=r"'caly'.*not qualified"):
        compile_kernel(g, "caly")
    with pytest.raises(ValueError, match=r"cannot be resolved.*nope"):
        compile_kernel(g, "calc.nope")


def test_external_typo_warns_but_compiles():
    def f(x):
        return x

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    with pytest.warns(UserWarning, match="did not exist before compilation"):
        ck = compile_kernel(g, ["ext.tpyo"])
    assert ck.execute({"ext": {"tpyo": 5.0}}) == {"ext.tpyo": 5.0}


def test_external_variable_with_formula_rejected():
    def f(x):
        return x

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    g.external["ext"].update("x", Variable(name="x", source="ext", formula=lambda: 1.0))
    with pytest.raises(ValueError, match=r"ext\.x.*external but carries a formula"):
        compile_kernel(g, "calc.y")
```

- [ ] **Step 2: Run — expect FAIL** (today: `AttributeError`/raw `ValueError`/`KeyError`; no warning; formula-bearing external silently becomes an input)

- [ ] **Step 3: Implement.** Imports: extend the existing `variable` import line to include `QUAL_SEP`; the warning needs the `warnings` import added in Task 1. Rewrite the body of `compile_kernel`:

```python
def compile_kernel(graph, required, *, jit_options=None, cache=None):
    """Compile `graph` into a fused @njit kernel for the `required` variables."""
    required = [required] if isinstance(required, str) else list(required)
    required = list(dict.fromkeys(required))  # dedupe, preserve first-seen order
    for entry in required:
        if not isinstance(entry, str):
            raise TypeError(f"required entries must be qualified-name strings; got {entry!r}")
        if QUAL_SEP not in entry:
            raise ValueError(
                f"required entry {entry!r} is not qualified (expected 'source{QUAL_SEP}name')"
            )
    pre_existing = {src: set(ns.keys()) for src, ns in graph.external.items()}
    try:
        compiled = graph.compile(required)
    except KeyError as e:
        raise ValueError(f"required name cannot be resolved in the graph: {e}") from e
    for entry in required:
        src, _, name = entry.rpartition(QUAL_SEP)
        if src in pre_existing and name not in pre_existing[src] and name in graph.external[src]:
            warnings.warn(
                f"required entry {entry!r} did not exist before compilation; External "
                f"namespaces create variables on first lookup, so a typo silently "
                f"becomes a new kernel input",
                stacklevel=2,
            )
    idents = _assign_identifiers([n.variable for n in compiled.ordered_nodes])
    source, bindings, params, outputs = _generate_body(compiled, required, idents)
    kernel = _compile(source, bindings, jit_options, cache)
    identifiers = {v.qual_name(): ident for v, ident in idents.items()}
    return CompiledKernel(kernel, params, outputs, source, identifiers)
```

In `_generate_body`, right after `ext_sorted` is built (line 122):

```python
    for var in ext_sorted:
        if var.formula is not None:
            raise ValueError(
                f"{var.qual_name()!r} is external but carries a formula; CompiledGraph "
                f"computes such a variable while a fused kernel treats it as a plain "
                f"input. Move it into a Variables namespace or drop the formula."
            )
```

(Note: `graph.compile` memoizes per `required` tuple, so the typo warning fires on the first compile of a given set — that is when the auto-creation happens; the test compiles a fresh set and sees the warning.)

- [ ] **Step 4: Run new tests + full file — expect PASS**
- [ ] **Step 5: Flake8 (both rule sets), then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: validate required entries, warn on auto-created External names, reject formula-bearing externals"
```

---

### Task 6: Contextual RecursionError for deep graphs (F-37)

**Goal:** A graph deeper than Python's recursion limit fails with an error that names the cause and the remedy instead of a bare `RecursionError` traceback.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (the `graph.compile` call introduced in Task 5; `import sys`)
- Test: `test/core/test_compile_kernel.py`

**Acceptance Criteria:**
- [ ] A chain deeper than the current recursion limit raises `RecursionError` whose message contains `setrecursionlimit` and the current limit
- [ ] The test runs in well under a second (the failure is pre-numba)

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py::test_deep_chain_recursion_error_is_contextual -v` → PASS

**Steps:**

- [ ] **Step 1: Write the failing test** (append)

```python
def test_deep_chain_recursion_error_is_contextual():
    depth = sys.getrecursionlimit() + 100

    def step(x):
        return x

    specs = [{"name": "n0", "inputs": {"x": "ext"}, "formula": step}]
    specs += [
        {"name": f"n{i}", "inputs": {f"n{i - 1}": "calc"}, "formula": step}
        for i in range(1, depth)
    ]
    g = Graph({"calc": specs}, ["ext"])
    with pytest.raises(RecursionError, match="setrecursionlimit"):
        compile_kernel(g, f"calc.n{depth - 1}")
```

- [ ] **Step 2: Run — expect FAIL** (bare `RecursionError: maximum recursion depth exceeded`, no match)

- [ ] **Step 3: Implement.** Add `import sys` to the module imports; extend the `try` from Task 5:

```python
    try:
        compiled = graph.compile(required)
    except KeyError as e:
        raise ValueError(f"required name cannot be resolved in the graph: {e}") from e
    except RecursionError:
        raise RecursionError(
            f"graph dependency depth exceeds Python's recursion limit "
            f"({sys.getrecursionlimit()}); the traversal needs roughly one stack frame "
            f"per chained node - raise sys.setrecursionlimit(...) before compile_kernel"
        ) from None
```

- [ ] **Step 4: Run the test + full file — expect PASS**
- [ ] **Step 5: Flake8 (both rule sets), then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: contextual RecursionError naming the limit and remedy for deep graphs"
```

---

### Task 7: Save-side cache assertion + coverage gaps (F-15, F-22)

**Goal:** The suite fails if kernel disk-cache *saving* silently regresses, and the untested end-to-end shapes (external-only kernel, mixed external+interior outputs) are exercised.

**Files:**
- Test: `test/core/test_compile_kernel.py` only

**Acceptance Criteria:**
- [ ] A cold→warm subprocess roundtrip asserts `.nbc`/`.nbi` files exist after the cold run and are byte-stable (mtimes unchanged) across the warm run — via `assert_njit_cache_survives_subprocess_roundtrip`
- [ ] `compile_kernel(g, "ext.x")` (external-only output, empty kernel body) compiles end-to-end and echoes its input
- [ ] `required=["calc.y", "ext.x"]` (mixed) returns both, in requested order

**Verify:** `<clean-caches> && <pytest> /home/erik/projects/numbox/test/core/test_compile_kernel.py -v -k "save_side or external_only or mixed_outputs"` → 3 new tests pass

**Steps:**

- [ ] **Step 1: Write the tests** (append; the import mirrors `test/core/test_proxy.py:15`)

```python
from test.auxiliary_utils import assert_njit_cache_survives_subprocess_roundtrip


def test_compile_kernel_cache_save_side(tmp_path):
    assert_njit_cache_survives_subprocess_roundtrip(
        tmp_path,
        """
        from numbox.core.variable.variable import Graph
        from numbox.core.variable.compile_kernel import compile_kernel

        def f(x):
            return x * 2.0

        def h(y):
            return y + 1.0

        g = Graph({"calc": [
            {"name": "y", "inputs": {"x": "ext"}, "formula": f},
            {"name": "z", "inputs": {"y": "calc"}, "formula": h},
        ]}, ["ext"])
        ck = compile_kernel(g, "calc.z")
        print(ck.execute({"ext": {"x": 3.0}})["calc.z"])
        """,
        ["7.0"],
    )


def test_external_only_output_end_to_end():
    g = Graph({"calc": []}, ["ext"])
    with pytest.warns(UserWarning, match="did not exist before compilation"):
        ck = compile_kernel(g, "ext.x")
    assert ck.params == ["ext.x"]
    assert ck.execute({"ext": {"x": 5.5}}) == {"ext.x": 5.5}


def test_mixed_outputs_end_to_end():
    def f(x):
        return x * 10.0

    g = Graph({"calc": [{"name": "y", "inputs": {"x": "ext"}, "formula": f}]}, ["ext"])
    ck = compile_kernel(g, ["calc.y", "ext.x"])
    assert ck.outputs == ["calc.y", "ext.x"]
    assert ck.execute({"ext": {"x": 2.0}}) == {"calc.y": 20.0, "ext.x": 2.0}
```

(The helper's contract — cold run populates `.nbc`/`.nbi`, warm run leaves paths and mtimes untouched — is documented at `test/auxiliary_utils.py:78`. `test_external_only_output_end_to_end` wraps the compile in `pytest.warns` because `ext.x` is genuinely auto-created on first lookup — Task 5's warning is correct and expected there.)

- [ ] **Step 2: Run — the two end-to-end tests should PASS immediately (they close coverage, not bugs); the save-side test must PASS too.** If the save-side test fails, that is a real find — debug before proceeding (do not weaken the assertion).
- [ ] **Step 3: Mutation check (manual, no commit):** temporarily change `opts.setdefault("cache", True)` to `opts.setdefault("cache", False)` in `_compile`, re-run `test_compile_kernel_cache_save_side`, confirm it FAILS (no cache files), revert. This proves the test is non-vacuous.
- [ ] **Step 4: Flake8 (both rule sets), then commit**

```bash
git -C /home/erik/projects/numbox add test/core/test_compile_kernel.py
git -C /home/erik/projects/numbox commit -m "compile_kernel: assert cache save-side via subprocess roundtrip; cover external-only and mixed-output kernels"
```

---

### Task 8: Benchmark per-user tmp module dir (F-23)

**Goal:** The benchmark's generated formula module lives in a per-user, mode-0700 directory so shared-host runs neither EPERM-crash on another user's file nor exec attacker-writable code.

**Files:**
- Modify: `test/compile_kernel_benchmark.py` (`load_formulas`, ~line 176; imports)

**Acceptance Criteria:**
- [ ] The module path is `<tmp>/ck_bench_<username>/_ck_bench_formulas_<profile>_<n>.py`, directory mode 0700, POSIX ownership verified
- [ ] A small benchmark run completes and the module file appears under the per-user dir
- [ ] mtime-stability behavior is preserved (unchanged content is not rewritten)

**Verify:** `set -euo pipefail; /home/erik/projects/numbox/venv/bin/python /home/erik/projects/numbox/test/compile_kernel_benchmark.py --nodes 8 --size 100 --repeats 2 --profile cheap` → completes; `ls /tmp/ck_bench_$(whoami)/` shows the formulas module

**Steps:**

- [ ] **Step 1: Implement.** Add `import getpass` to the imports. Insert above `load_formulas`:

```python
def _formulas_dir():
    d = pathlib.Path(tempfile.gettempdir()) / f"ck_bench_{getpass.getuser()}"
    d.mkdir(mode=0o700, exist_ok=True)
    if hasattr(os, "getuid") and d.stat().st_uid != os.getuid():
        raise RuntimeError(f"{d} exists but is not owned by the current user")
    return d
```

In `load_formulas`, replace

```python
    path = pathlib.Path(tempfile.gettempdir()) / f"{mod_name}.py"
```

with

```python
    path = _formulas_dir() / f"{mod_name}.py"
```

- [ ] **Step 2: Run the small benchmark (Verify command above); confirm the module file location and that a second identical run does not rewrite it** (compare `stat -c %Y` before/after)
- [ ] **Step 3: Flake8 (both rule sets), then commit**

```bash
git -C /home/erik/projects/numbox add test/compile_kernel_benchmark.py
git -C /home/erik/projects/numbox commit -m "compile_kernel benchmark: per-user 0700 tmp dir for the generated formulas module"
```

---

### Task 9: Truthful documentation (F-14 + envelope/cache docs)

**Goal:** The public docstring and the rst tell the truth about: the eager/lazy error split, `required` semantics, cache precedence and what the digest covers, the un-fingerprintable fallback, anchor side effects, and the measured scale envelope.

**Files:**
- Modify: `numbox/core/variable/compile_kernel.py` (`compile_kernel` docstring)
- Modify: `docs/numbox.core.variable.rst` (compile_kernel section, lines 220–275)

**Acceptance Criteria:**
- [ ] Docstring documents: `required` (str|list, order fixes `.outputs`), `jit_options` merge + `cache` precedence, eager vs lazy errors, fingerprint coverage + fallback-means-uncached, anchor side effects only when caching
- [ ] rst gains a "Caching" paragraph (digest coverage, precedence, fallback) and a "Practical limits" paragraph (depth ≈ `sys.getrecursionlimit()`, ~20 ms / ~1 MiB per node cold-compile guidance from the review's measurements)
- [ ] `sphinx-build` exits 0 with no new warnings vs `origin/main`; doc code blocks flake8-clean

**Verify:**

```bash
set -euo pipefail
/home/erik/projects/numbox/venv/bin/sphinx-build -b html /home/erik/projects/numbox/docs /tmp/sphinx-ckfix 2>&1 | tail -3
/home/erik/projects/numbox/venv/bin/python /home/erik/projects/numbox/.github/scripts/extract_codeblocks.py
```

→ sphinx exit 0; extracted blocks flake8-clean (run the repo's doc-codeblock workflow steps as in `.github/workflows/doc-codeblock-flake8.yml`)

**Steps:**

- [ ] **Step 1: Replace the one-line docstring** of `compile_kernel` with:

```python
    """Compile `graph` into a fused @njit kernel for the `required` variables.

    :param graph: a `Graph`; its dependency structure and formulas are fused
        into one straight-line @njit function (see `CompiledKernel`).
    :param required: qualified name or list of qualified names. Order is
        preserved (first occurrence wins) and fixes the order of
        `CompiledKernel.outputs` / the kernel's return tuple.
    :param jit_options: merged over numbox's defaults
        (`NUMBOX_JIT_OPTIONS` env) and passed to @njit. All options except
        `cache` participate in the content-addressed digest.
    :param cache: tri-state. `None` (default) defers to
        `jit_options["cache"]`, then the `NUMBOX_JIT_OPTIONS` env default,
        then `True`. An explicit `True`/`False` wins over both.

    Error timing: structural problems raise here (unknown or malformed
    `required` entries, non-callable formulas, arity mismatches against the
    declared inputs, formula-bearing external variables, graphs deeper than
    the recursion limit); numba typing problems surface at the kernel's
    first call (auto-njit of plain-Python formulas is lazy).

    Caching: the kernel digest fingerprints each formula's bytecode,
    constants, default values, closure-cell values, referenced module-level
    globals (including helper functions, recursively), defining module, and
    the effective jit flags. A formula whose state cannot be fingerprinted
    (e.g. cres/CompileResultWAP objects, values with no canonical form)
    downgrades that one kernel to cache=False: always recompiled, never
    stale. When caching is enabled, a content-addressed anchor `.py` file is
    written under numba's cache directory; with caching off (or the cache
    dir unwritable, which warns and degrades) nothing is written.
    """
```

- [ ] **Step 2: Extend the rst section** (after the v1-limitations paragraph around line 246) with two prose paragraphs — adapt wording freely, keep claims tied to these facts:

```rst
**Caching.** The fused kernel is cached on disk, content-addressed by a
fingerprint of the generated kernel source, every formula's behavioral
state (bytecode, constants, default values, closure-cell values, referenced
module-level globals including helpers, defining module), and the effective
jit flags. Changing any of these recompiles instead of reusing a stale
binary. Formulas whose state cannot be fingerprinted (for example
``cres``-compiled callables) make that one kernel uncached: always
recompiled per process, never wrong. The ``cache`` keyword is tri-state:
``None`` defers to ``jit_options["cache"]`` and the ``NUMBOX_JIT_OPTIONS``
environment default; an explicit ``True``/``False`` wins.

**Practical limits.** Graph traversal is recursive: dependency chains
deeper than roughly ``sys.getrecursionlimit()`` raise a ``RecursionError``
that names the remedy (raise the limit before compiling). Cold compilation
of the fused kernel costs on the order of 20 ms and ~1 MiB of memory per
formula node (numba 0.65, CPython 3.12); graphs beyond a few thousand
nodes compile increasingly slowly and are better split or evaluated via
:class:`numbox.core.variable.variable.CompiledGraph`.
```

- [ ] **Step 3: Build sphinx + doc-codeblock checks (Verify above); compare the warning list against `origin/main`'s build** (`git -C /home/erik/projects/numbox stash` not needed — build origin/main's docs once in `/tmp` if a baseline is wanted; the review established the baseline is 64 pre-existing warnings)
- [ ] **Step 4: Flake8 (both rule sets — rst python blocks too via the extract script), then commit**

```bash
git -C /home/erik/projects/numbox add numbox/core/variable/compile_kernel.py docs/numbox.core.variable.rst
git -C /home/erik/projects/numbox commit -m "compile_kernel: document cache precedence, digest coverage, error timing, and practical limits"
```

---

### Task 10: Full local CI gate, push, fork CI

**Goal:** Everything `.github/workflows/` checks passes locally; then (with explicit user approval) push to `origin/feat/variable-compile-kernel` and watch fork CI green.

**Files:** none modified (gate + push)

**Acceptance Criteria:**
- [ ] Full test suite green (`test/`), with caches cleaned first
- [ ] Both flake8 rule sets clean repo-wide
- [ ] Sphinx exit 0, no new warnings; doc-codeblock-flake8 clean; lychee clean on changed files
- [ ] User explicitly approved the push in this session
- [ ] Fork `numbox_ci` matrix green on the pushed head (judge only `status==completed`)

**Verify:** the gate commands below, then `gh run watch <run-id> --exit-status`

**Steps:**

- [ ] **Step 1: Full local gate**

```bash
set -euo pipefail
/home/erik/projects/numbox/venv/bin/python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('/home/erik/projects/numbox').rglob('__pycache__')]; shutil.rmtree(pathlib.Path.home() / '.cache' / 'numba', ignore_errors=True)"
/home/erik/projects/numbox/venv/bin/python -m pytest /home/erik/projects/numbox/test --durations=20
/home/erik/projects/numbox/venv/bin/python -m flake8 /home/erik/projects/numbox/numbox /home/erik/projects/numbox/test
/home/erik/projects/numbox/venv/bin/python -m flake8 --select=E9,F63,F7,F82,F401 /home/erik/projects/numbox/numbox /home/erik/projects/numbox/test
/home/erik/projects/numbox/venv/bin/sphinx-build -b html /home/erik/projects/numbox/docs /tmp/sphinx-gate 2>&1 | tail -3
```

Plus the doc-codeblock and lychee steps exactly as their workflow files define them (`.github/workflows/doc-codeblock-flake8.yml`, `.github/workflows/link-check.yml` — run lychee over the full changed-file set vs `origin/main`, not just edited docs).

- [ ] **Step 2: Commit the plan documents themselves** (this file + `.tasks.json`) if not yet committed:

```bash
git -C /home/erik/projects/numbox add docs/superpowers/plans/2026-06-11-compile-kernel-review-fixes.md docs/superpowers/plans/2026-06-11-compile-kernel-review-fixes.md.tasks.json
git -C /home/erik/projects/numbox commit -m "docs(plan): compile_kernel review-fixes plan + tasks"
```

- [ ] **Step 3: ASK THE USER for push approval** (explicit, this session — do not push without it). On approval:

```bash
git -C /home/erik/projects/numbox push origin feat/variable-compile-kernel
```

- [ ] **Step 4: Watch fork CI** — find the run for the pushed SHA, then block on it:

```bash
gh run list --repo nelson2005/numbox --branch feat/variable-compile-kernel --workflow numbox_ci --limit 1
gh run watch <run-id> --repo nelson2005/numbox --exit-status
```

Report the matrix result (expect 27/27) and the PR link: https://github.com/nelson2005/numbox/pull/49

- [ ] **Step 5: Update this plan's `.tasks.json`** statuses to `completed` and commit if anything changed.

(Upstream PR #23 cherry-pick is intentionally NOT part of this plan — it requires separate explicit consent per the standing rule, and Goykhman's review of #23 is still pending.)

---

## Self-review notes

- Every report recommendation maps to a task: digest→T2/T3, ns.pop→T0, cache semantics→T1, tests→T7 (+batteries in T1/T3/T4/T5/T6), errors/docs→T4/T5/T9, DUFunc/CFunc→T4, scale→T6/T9, benchmark tmp→T8, External guard→T5, cres anchors→T3. F-25 (set reprs) is fixed by T2's sorted set canonicalization; F-26 (re-njit per call) is consciously deferred — re-njit cost is dwarfed by kernel compile and memoizing dispatchers would add cross-kernel state for no measured win (YAGNI).
- Tasks 1 and 3 both edit `_compile`: T1 lands precedence+lazy-anchor on the old hash so each commit is green; T3 swaps only the digest lines. The shown T3 code includes T1's final form — they are consistent by construction.
- Naming consistency check: `_formula_fingerprint`/`_canon_value`/`_fingerprint_function`/`_fingerprint_codeobj`/`_referenced_global_names`/`_Unfingerprintable`/`_check_formula_arity`/`_formulas_dir` are used with these exact names throughout.
