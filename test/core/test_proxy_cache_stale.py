"""Cross-process regression tests for a stale ``@proxy`` cfunc alias in a warm numba cache.

``@proxy`` publishes the cfunc wrapper of each proxied body under a process-stable alias whose name folds a fingerprint
of that body, so an edit to the body renames the alias. A ``cache=True`` caller in another file is keyed only on its
own source, because numba's key does not distinguish callees. After such an edit, the caller therefore still hits the
cache and loads an object that references a symbol this process never registered. RuntimeDyld resolves the externals
of an object in one batch, so that single missing name zeroes *every* external relocation in the object. The process
then crashes inside the CPython argument-unpacking wrapper before any user code runs, with a bare segfault and an empty
stderr. The engine keeps that failure, so the next cached object loaded afterwards, even a sound one, aborts the
process instead and names the alias that no longer exists.
Both failure modes are hazards for anyone who edits a proxied binding without first clearing the cache.

The target behavior is that the loader detects the unregistered alias, discards that one entry, warns, and recompiles.
Each scenario therefore also carries a second binding that is never edited, with its own caller in the same file and
the same cache directory. That caller must still load from the cache and must cause no warning, so a fix that disables
caching or invalidates the whole directory cannot pass. Separate tests, which never trigger the crash, verify the
premise: warm entries really load from the cache, and the edit re-keys one binding and only one.

A caller that reaches the ``.as_func`` of a binding as a compile-time constant is the complementary case, and the
module covers it too. Such a caller links the machine code of the body into its own object instead of referencing the
alias. numba is willing to cache the caller only because of that linking, and the linking also leaves nothing for the
prefix scan of the guard to find. After an edit to the body, the caller then runs the old binary silently and
indefinitely, and the same happens to a binding that became absent rather than changed. Those tests characterize that
outcome rather than verify a fix, and their docstrings describe how a change to it would show in the test.
"""
import os
import subprocess
import sys
import textwrap

import pytest

from numbox.core.proxy.proxy import _undefined_symbols

_EDITED = "proxied_binding.py"
_HEALED = {"call_scale": "compiled", "call_scale2": "compiled", "call_offset": "served", "plain": "served"}
_OLD_BODY = {"call_scale": "11.0", "call_scale2": "12.0", "call_offset": "108.0", "plain": "12.0"}
_NEW_BODY = {"call_scale": "16.0", "call_scale2": "17.0", "call_offset": "108.0", "plain": "12.0"}


def _write_binding(path, name, expr):
    """Write a one-function proxied binding module.

    A rewrite advances the mtime explicitly rather than trusting the clock. The new body is the same length as the
    old one, and both numba's source stamp and CPython's pyc validation compare ``(st_mtime, st_size)``. If the mtime
    granularity is coarse, a rewrite within the same second is therefore indistinguishable from the original, and the
    edit would never be seen. The padding above the decorator is deliberate too. The cache anchor requires the
    decorator line to be the function's ``co_firstlineno``, and that line must not precede the decorator line of the
    generated wrapper.
    """
    before = path.stat() if path.exists() else None
    path.write_text(textwrap.dedent(f'''
        """A proxied binding. The alias published for this body folds the body's fingerprint, so any edit below
        renames it and leaves every cached object that referenced the old name pointing at nothing.
        """
        from numba import types
        from numbox.core.proxy.proxy import proxy

        SIG = types.float64(types.float64)


        @proxy(SIG, jit_options={{"cache": True}})
        def {name}(x):
            return {expr}
    '''), encoding="utf-8")
    if before is None:
        return
    os.utime(path, (before.st_mtime + 10,) * 2)
    after = path.stat()
    if (after.st_mtime, after.st_size) == (before.st_mtime, before.st_size):
        raise RuntimeError(
            f"{path.name} kept its (st_mtime, st_size) across the rewrite, so the edit is invisible to numba and "
            "the scenario would quietly prove nothing"
        )


def _write_scenario(tmp_path):
    """Write the scenario to disk and return the probe script.

    The two bindings are in separate files, so the control is untouched at file level as well: neither its alias nor
    its source stamp can change when the other body is edited. The three callers share one file to make the control as
    tight as possible: same module, same cache directory, same run. Two callers of the edited body, rather than one,
    verify that *every* stale entry is discarded, not just the first one that the loader reaches. ``plain``, which is
    cached but references no alias, is the control for the other direction. It takes the early exit of the validator
    and must still load from the cache.
    """
    _write_binding(tmp_path / _EDITED, "scale", "x * 2.0")
    _write_binding(tmp_path / "proxied_binding_stable.py", "offset", "x + 100.0")
    (tmp_path / "cached_callers.py").write_text(textwrap.dedent('''
        """Cached callers of both bindings, in a file the edit never touches, so their numba cache keys still match
        afterwards and the warm entries are served."""
        from numba import njit

        from proxied_binding import scale
        from proxied_binding_stable import offset


        @njit(cache=True)
        def call_scale(x):
            return scale(x) + 1.0


        @njit(cache=True)
        def call_scale2(x):
            return scale(x) + 2.0


        @njit(cache=True)
        def call_offset(x):
            return offset(x) + 3.0


        @njit(cache=True)
        def plain(x):
            return x + 7.0
    '''), encoding="utf-8")

    probe = tmp_path / "stale_alias_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        import warnings

        from numba.core.types import float64

        import proxied_binding as edited
        import proxied_binding_stable as stable

        print("ALIAS_edited", edited.scale._numbox_proxy_alias, flush=True)
        print("ALIAS_stable", stable.offset._numbox_proxy_alias, flush=True)
        print("BINDINGS_IMPORTED", flush=True)

        # The discard warning is caught here rather than read off stderr: an inherited -W setting would otherwise
        # decide whether it is printed, swallowed, or fatal, and the default filter shows a given warning once.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from cached_callers import call_offset, call_scale, call_scale2, plain
            fns = {"call_scale": call_scale, "call_scale2": call_scale2, "call_offset": call_offset, "plain": plain}
            print("CALLERS_IMPORTED", flush=True)
            if os.environ["PRECOMPILE"] == "1":
                for name, fn in fns.items():
                    fn.compile((float64,))
                    print("COMPILED", name, flush=True)
            for name, fn in fns.items():
                print(f"RESULT_{name}", fn(5.0), flush=True)

        for name, fn in fns.items():
            hits = sum(fn.stats.cache_hits.values())
            misses = sum(fn.stats.cache_misses.values())
            state = "served" if hits and not misses else "compiled" if misses and not hits else f"{hits}/{misses}"
            print(f"STATS_{name}", state, flush=True)
        print("WARNINGS", " | ".join(f"{w.category.__name__}: {w.message}".replace("\\n", " ") for w in caught))
        print("DONE ok", flush=True)
    '''), encoding="utf-8")
    return probe


def _edit_body(tmp_path):
    """Rewrite the body of the edited binding, which is the only trigger: same file, same length, different alias."""
    _write_binding(tmp_path / _EDITED, "scale", "x * 3.0")


def _probe_env(tmp_path, cache_name, precompile=False):
    """Return the environment for one probe run.

    ``NUMBA_CACHE_DIR`` also relocates numbox's own anchor files, so a per-test directory keeps the whole scenario out
    of the developer's shared cache. The parent's ``sys.path`` is forwarded because the probe must import numbox from
    the location where this test found it. ``tmp_path`` is named explicitly rather than relying on the script
    directory, which ``PYTHONSAFEPATH`` suppresses. Bytecode writing is disabled because a pyc validated on
    ``(st_mtime, st_size)`` is the other way in which a same-length edit can go unnoticed. The inherited warning and
    numba settings are dropped, because each of them can make the scenario vacuous. They decide whether an unrelated
    warning ends the probe, and whether anything is compiled. ``NUMBA_OPT`` also decides whether numba is willing to
    cache the callers: at ``NUMBA_OPT=0``, the proxy call is not folded to a plain extern reference.
    ``NUMBOX_PROXY_CACHE_STRICT`` is dropped for the same reason. An exported strict setting would turn every repair
    scenario below into a hard error, so a strict run sets the variable explicitly rather than inheriting it.
    """
    env = dict(os.environ)
    for hostile in ("PYTHONWARNINGS", "PYTHONDEVMODE", "NUMBA_DISABLE_JIT", "NUMBA_OPT", "NUMBOX_PROXY_CACHE_STRICT"):
        env.pop(hostile, None)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / cache_name)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path)] + sys.path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PRECOMPILE"] = "1" if precompile else "0"
    return env


def _run_probe(probe, env):
    """Run a probe to completion and return the whole result.

    These scenarios expect the child process to crash, so the caller, not this helper, decides which return code is
    acceptable. The timeout is the only defense against a crashed child that never exits, because after a fault a
    crash reporter can keep the process from exiting."""
    return subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=300,
    )


def _report(r):
    """Return the full transcript, so that a failure shows which crash case occurred, or that neither occurred."""
    return f"rc={r.returncode}\n--- stdout ---\n{r.stdout}--- stderr ---\n{r.stderr}"


def _fields(r):
    """Return the ``KEY value`` lines of the probe. ``DONE`` is absent whenever the child crashed during its run."""
    fields = dict(line.split(" ", 1) for line in r.stdout.splitlines() if " " in line)
    assert fields.get("DONE") == "ok", f"the probe did not run to completion\n{_report(r)}"
    return fields


def _warm(probe, env):
    """Warm the cache and return the probe's fields.

    A failure here means a broken scenario rather than a failure of the behavior under test, so the helper raises it
    as an error. An assertion would be reported as a behavioral failure, and a broken harness must not be mistaken for
    one.
    """
    r = _run_probe(probe, env)
    if r.returncode != 0 or "DONE ok" not in r.stdout:
        raise RuntimeError(f"warming the cache failed\n{_report(r)}")
    return _fields(r)


def _stats(fields, names=_HEALED):
    return {name: fields[f"STATS_{name}"] for name in names}


def _results(fields, names=_HEALED):
    return {name: fields[f"RESULT_{name}"] for name in names}


def test_unedited_rerun_is_served_from_cache(tmp_path):
    """The tests below prove nothing unless a warm entry really loads from the cache, because a caller that
    recompiled anyway could never receive a stale object. The test makes one cold run, and then an identical run
    against the same cache directory."""
    probe = _write_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")

    cold = _warm(probe, env)
    assert _stats(cold) == dict.fromkeys(_HEALED, "compiled"), _stats(cold)
    assert _results(cold) == _OLD_BODY, _results(cold)

    warm = _warm(probe, env)
    assert _stats(warm) == dict.fromkeys(_HEALED, "served"), f"an unchanged re-run recompiled: {_stats(warm)}"
    assert _results(warm) == _results(cold), _results(warm)


def test_editing_a_proxied_body_renames_only_that_binding_alias(tmp_path):
    """The edit re-keys exactly one binding, so it makes one caller stale and leaves the control valid.

    The edited body runs against a *fresh* cache directory, so that this test avoids the crash. The test measures the
    result of the new body, and which aliases changed, without a stale entry in the cache.
    """
    probe = _write_scenario(tmp_path)
    before = _warm(probe, _probe_env(tmp_path, "nbcache"))

    _edit_body(tmp_path)
    after = _warm(probe, _probe_env(tmp_path, "nbcache_after_edit"))

    assert after["ALIAS_edited"] != before["ALIAS_edited"], "the edited body kept its alias"
    assert after["ALIAS_stable"] == before["ALIAS_stable"], "the untouched body was re-keyed"
    assert _results(after) == _NEW_BODY, _results(after)


@pytest.mark.parametrize("precompile", [False, True], ids=["call", "compile"])
def test_stale_proxy_alias_is_discarded_and_recompiled(tmp_path, precompile):
    """A warm caller reached after an edit to the proxied body must recompile, not crash.

    Without the guard, the first call segfaults. The one missing alias zeroes every external relocation in the loaded
    object, so control never reaches the compiled body. stdout stops right after the callers are imported, and stderr
    is empty, so the failure gives no diagnostic. If the test compiles the callers explicitly instead, the failure
    moves. The stale load itself is silent, because no code has entered the object yet. The next cached load of any
    object then aborts the process and names the retired alias, because the execution engine keeps the first failure.

    With the guard, both callers of the edited body are discarded and recompiled against it, while the caller of the
    untouched binding still loads from the cache. A warning announces the discard and names the alias that this
    process never registered. The last run verifies that the repair is a one-time recompile. An implementation that
    refused to load those entries without replacing them would make every later process compile them in full again.
    """
    probe = _write_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache", precompile=precompile)
    stale_alias = _warm(probe, env)["ALIAS_edited"]

    _edit_body(tmp_path)
    r = _run_probe(probe, env)
    if r.returncode == 0 and _fields(r)["STATS_call_offset"] != "served":
        raise RuntimeError(f"the edit re-keyed the callers too, so no stale entry was ever loaded\n{_report(r)}")
    assert r.returncode == 0, f"a cache entry referencing an unregistered proxy alias reached the engine\n{_report(r)}"
    fields = _fields(r)
    assert _results(fields) == _NEW_BODY, _results(fields)
    assert _stats(fields) == _HEALED, f"the wrong entries were discarded: {_stats(fields)}"
    assert "StaleProxyCacheWarning" in fields["WARNINGS"], f"the discard was not announced: {fields['WARNINGS']}"
    assert stale_alias in fields["WARNINGS"], f"no warning named {stale_alias}: {fields['WARNINGS']}"

    healed = _fields(_run_probe(probe, env))
    assert _stats(healed) == dict.fromkeys(_HEALED, "served"), f"the recompile was not written back: {_stats(healed)}"
    assert "StaleProxyCacheWarning" not in healed["WARNINGS"], f"the heal repeats forever: {healed['WARNINGS']}"


_CONST_FNS = ("const_scale", "disp_scale", "const_offset", "disp_offset")
_CONST_OLD_BODY = {"const_scale": "11.0", "disp_scale": "12.0", "const_offset": "108.0", "disp_offset": "109.0",
                   "direct": "10.0"}
_CONST_NEW_BODY = {"const_scale": "16.0", "disp_scale": "17.0", "const_offset": "108.0", "disp_offset": "109.0",
                   "direct": "15.0"}


def _write_const_caller_scenario(tmp_path):
    """Write the same two bindings, reached by both routes, and return the probe script.

    A ``const_*`` caller reads a binding's ``.as_func`` as a module-level constant. The machine code of the body is
    therefore linked into the caller's own object, and no alias is emitted for it. A ``disp_*`` caller calls the
    dispatcher, which emits the alias that the guard above looks for. Both routes are applied to the edited binding and
    to the untouched one, so the callers of the second binding remain the control that the module docstring requires.
    An implementation that repaired the const route by declining to cache it would therefore appear as ``const_offset``
    recompiling. The signatures are explicit, so every caller compiles at import, inside the warning capture of the
    probe.
    """
    _write_binding(tmp_path / _EDITED, "scale", "x * 2.0")
    _write_binding(tmp_path / "proxied_binding_stable.py", "offset", "x + 100.0")
    (tmp_path / "const_callers.py").write_text(textwrap.dedent('''
        """Cached callers of both bindings by both routes, in a file the edit never touches."""
        from numba import njit
        from numba.core.types import float64

        from proxied_binding import scale
        from proxied_binding_stable import offset

        SCALE_AS_FUNC = scale.as_func
        OFFSET_AS_FUNC = offset.as_func


        @njit(float64(float64), cache=True)
        def const_scale(x):
            return SCALE_AS_FUNC(x) + 1.0


        @njit(float64(float64), cache=True)
        def disp_scale(x):
            return scale(x) + 2.0


        @njit(float64(float64), cache=True)
        def const_offset(x):
            return OFFSET_AS_FUNC(x) + 3.0


        @njit(float64(float64), cache=True)
        def disp_offset(x):
            return offset(x) + 4.0
    '''), encoding="utf-8")

    probe = tmp_path / "const_caller_probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        import pathlib
        import warnings

        import proxied_binding as edited

        print("ALIAS_edited", edited.scale._numbox_proxy_alias, flush=True)

        # Whether numba refused to cache a caller is reported as a warning at its compile, so the import has to
        # happen under the capture, as does every call: the refusal is re-emitted on each run.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import const_callers
            fns = {name: getattr(const_callers, name)
                   for name in ("const_scale", "disp_scale", "const_offset", "disp_offset")}
            for name, fn in fns.items():
                print(f"RESULT_{name}", fn(5.0), flush=True)
            print("RESULT_direct", edited.scale(5.0), flush=True)

        # Counting the files a caller actually left behind is the only way to tell "cached" from "refused, and
        # recompiled quietly enough to look the same": both report a miss on every run.
        cache_dir = pathlib.Path(os.environ["NUMBA_CACHE_DIR"])
        for name, fn in fns.items():
            hits = sum(fn.stats.cache_hits.values())
            misses = sum(fn.stats.cache_misses.values())
            state = "served" if hits and not misses else "compiled" if misses and not hits else f"{hits}/{misses}"
            print(f"STATS_{name}", state, flush=True)
            for kind in ("nbi", "nbc"):
                written = sum(1 for _ in cache_dir.rglob(f"const_callers.{name}-*.{kind}"))
                print(f"{kind.upper()}_{name}", written, flush=True)
        print("WARNINGS", " | ".join(f"{w.category.__name__}: {w.message}".replace("\\n", " ") for w in caught))
        print("DONE ok", flush=True)
    '''), encoding="utf-8")
    return probe


def test_a_const_reference_caller_becomes_cacheable(tmp_path):
    """A ``cache=True`` caller that reaches a proxied binding's ``.as_func`` as a compile-time constant is cacheable.

    Lowering the constant declares the entry point of the body as a symbol and links its library in. The per-process
    address globals that such a reference would otherwise carry are therefore eliminated before numba scans the
    module. Without that, numba declines the entry and issues a warning on every run: ``Cannot cache compiled
    function "const_scale" as it uses dynamic globals (such as ctypes pointers and large global arrays)``. numba then
    leaves no file of that caller in the cache directory, so the second process compiles the caller from scratch
    again.

    ``disp_offset`` loads warm in any case, so an implementation that achieved this by loosening caching in general
    would fail here rather than pass.
    """
    probe = _write_const_caller_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")

    cold = _warm(probe, env)
    assert _results(cold, _CONST_OLD_BODY) == _CONST_OLD_BODY, _results(cold, _CONST_OLD_BODY)
    assert cold["STATS_const_scale"] == "compiled", f"a cold run served something: {_stats(cold, _CONST_FNS)}"

    warm = _warm(probe, env)
    assert warm["STATS_disp_offset"] == "served", f"the control caller recompiled: {_stats(warm, _CONST_FNS)}"
    refused = 'Cannot cache compiled function "const_scale"'

    assert warm["STATS_const_scale"] == "served", f"the const caller did not cache: {_stats(warm, _CONST_FNS)}"
    assert refused not in cold["WARNINGS"], f"numba declined the const caller: {cold['WARNINGS']}"
    assert refused not in warm["WARNINGS"], f"numba declined the const caller: {warm['WARNINGS']}"
    assert int(cold["NBI_const_scale"]) >= 1 and int(cold["NBC_const_scale"]) >= 1, (
        f"nothing was written for the const caller: {cold['NBI_const_scale']}/{cold['NBC_const_scale']}"
    )
    assert warm["STATS_const_offset"] == "served", (
        f"the second const caller did not cache, so only one binding's const route is pinned: "
        f"{_stats(warm, _CONST_FNS)}"
    )


def test_a_const_reference_caller_serves_a_stale_body_after_an_edit(tmp_path):
    """Characterization: the cacheability above comes at the cost of a stale binary that the alias guard cannot see.

    A const reference links the machine code of the proxied body directly into the caller's object. Nothing in the
    caller therefore mentions the ``numbox_pxy_`` alias, and the guard takes its early exit every time. Suppose the
    body is edited without clearing the cache. The caller then keeps returning the old number, while the same
    process reaches the same binding through two other routes and gets the new one. The only
    ``StaleProxyCacheWarning`` names the dispatcher-path caller that was repaired correctly, never the caller that
    returns wrong numbers. The run after that issues no warning and still returns the wrong number.

    The test is written to fail when that changes. If the guard is ever extended to cover a body linked in rather
    than referenced by alias, ``const_scale`` will be repaired and the first assertion below will fail. That failure
    reports that the hazard is gone, not a regression. ``const_offset`` and ``disp_offset`` meanwhile enforce the
    control rule of the module: a repair achieved by refusing to load the whole directory would show on them first.
    """
    probe = _write_const_caller_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")
    assert _results(_warm(probe, env), _CONST_OLD_BODY) == _CONST_OLD_BODY

    _edit_body(tmp_path)
    fields = _fields(_run_probe(probe, env))

    assert fields["RESULT_const_scale"] == _CONST_OLD_BODY["const_scale"], (
        f"the const caller healed -- if the guard now covers a linked-in body this test has done its job: "
        f"{_results(fields, _CONST_NEW_BODY)}"
    )
    assert fields["STATS_const_scale"] == "served", f"the old number came from somewhere else: {fields}"
    assert fields["RESULT_disp_scale"] == _CONST_NEW_BODY["disp_scale"], (
        f"the dispatcher-path caller did not heal: {fields['RESULT_disp_scale']}"
    )
    assert fields["RESULT_direct"] == _CONST_NEW_BODY["direct"], (
        f"calling the binding itself did not reach the new body: {fields['RESULT_direct']}"
    )
    assert fields["STATS_const_offset"] == "served" and fields["STATS_disp_offset"] == "served", (
        f"the untouched binding's callers were invalidated too: {_stats(fields, _CONST_FNS)}"
    )
    assert "const_callers.disp_scale-" in fields["WARNINGS"], (
        f"the one entry that was discarded went unannounced: {fields['WARNINGS']}"
    )
    assert "const_callers.const_scale-" not in fields["WARNINGS"], (
        f"the caller serving the old number was named after all: {fields['WARNINGS']}"
    )

    again = _fields(_run_probe(probe, env))
    assert again["RESULT_const_scale"] == _CONST_OLD_BODY["const_scale"], (
        f"the stale entry healed itself on a later run: {again['RESULT_const_scale']}"
    )
    assert "StaleProxyCacheWarning" not in again["WARNINGS"], (
        f"every later run is silent about the entry it is still serving: {again['WARNINGS']}"
    )


def _rewrite(path, old, new):
    """Replace a body expression and advance the source stamp, as ``_write_binding`` does, so that the edit is seen."""
    before = path.stat()
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    os.utime(path, (before.st_mtime + 10,) * 2)


def _write_two_proxy_scenario(tmp_path):
    """Write one cached caller of two *different* proxied bodies in separate files, so its object imports two aliases.

    Each binding is below the minimum line of the cache anchor. An edit to both bodies renames both aliases, so the
    single caller object references two names that this process never registered. A warning that named only the
    first stale alias it found, rather than all of them, would undercount in this case.
    """
    (tmp_path / "bind_a.py").write_text(textwrap.dedent('''
        """binding a."""
        from numba import types
        from numbox.core.proxy.proxy import proxy

        SIG = types.float64(types.float64)


        @proxy(SIG, jit_options={"cache": True})
        def scale(x):
            return x * 2.0
    '''), encoding="utf-8")
    (tmp_path / "bind_b.py").write_text(textwrap.dedent('''
        """binding b."""
        from numba import types
        from numbox.core.proxy.proxy import proxy

        SIG = types.float64(types.float64)


        @proxy(SIG, jit_options={"cache": True})
        def shift(x):
            return x + 10.0
    '''), encoding="utf-8")
    (tmp_path / "two_caller.py").write_text(textwrap.dedent('''
        """One cached caller of both proxies, in a file the edits never touch."""
        from numba import njit

        from bind_a import scale
        from bind_b import shift


        @njit(cache=True)
        def call_both(x):
            return scale(x) + shift(x) + 1.0
    '''), encoding="utf-8")
    probe = tmp_path / "two_probe.py"
    probe.write_text(textwrap.dedent('''
        import warnings

        import bind_a
        import bind_b

        print("ALIAS_a", bind_a.scale._numbox_proxy_alias, flush=True)
        print("ALIAS_b", bind_b.shift._numbox_proxy_alias, flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from two_caller import call_both
            print("RESULT", call_both(5.0), flush=True)
        stale = [w for w in caught if w.category.__name__ == "StaleProxyCacheWarning"]
        print("N_STALE_WARNINGS", len(stale), flush=True)
        for w in stale:
            print("WARN", str(w.message).replace(chr(10), " "), flush=True)
        print("DONE ok", flush=True)
    '''), encoding="utf-8")
    return probe


def test_a_multi_alias_object_names_every_stale_alias_no_undercount(tmp_path):
    """The discard warning for one cached object that imports two stale aliases must name *both* aliases.

    The test guards against a diagnostic that undercounts, in which an object that references several retired
    aliases receives a warning that names only some of them. The guard returns the whole set, and the caller joins
    it. A single warning for the one stale object here must therefore carry both retired names. An implementation
    that stopped at the first stale symbol would name one alias and miss the other. The test asserts nothing numeric
    about compilation, and it captures the retired aliases from the output of the warm run itself.
    """
    probe = _write_two_proxy_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")

    warm = _fields(_run_probe(probe, env))
    old_a, old_b = warm["ALIAS_a"], warm["ALIAS_b"]

    for name, old, new in (("bind_a.py", "x * 2.0", "x * 3.0"), ("bind_b.py", "x + 10.0", "x + 20.0")):
        _rewrite(tmp_path / name, old, new)

    fields = _fields(_run_probe(probe, env))
    assert fields["ALIAS_a"] != old_a and fields["ALIAS_b"] != old_b, "an edited body kept its alias"
    assert fields["N_STALE_WARNINGS"] == "1", (
        f"the single stale object should draw exactly one warning, got {fields['N_STALE_WARNINGS']}"
    )
    warned = fields["WARN"]
    for retired in (old_a, old_b):
        assert retired in warned, f"the warning undercounted -- it did not name {retired}: {warned}"


def test_the_discard_can_be_escalated_to_an_error(tmp_path):
    """Silent repair is the default, not the only option: a caller who prefers an error can escalate the warning.

    The warning is a named class reachable from the public module path, so a filter can select it without catching
    every ``RuntimeWarning``. Escalation aborts before the repair, which leaves the stale entry on disk for inspection
    rather than replacing it.
    """
    probe = _write_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")
    _warm(probe, env)

    strict = tmp_path / "strict_probe.py"
    strict.write_text(textwrap.dedent('''
        import warnings

        from numbox.core.proxy.proxy import StaleProxyCacheWarning

        warnings.filterwarnings("error", category=StaleProxyCacheWarning)
        from cached_callers import call_scale
        print("RESULT", call_scale(5.0), flush=True)
    '''), encoding="utf-8")

    _edit_body(tmp_path)
    r = _run_probe(strict, env)
    assert r.returncode != 0, f"the stale entry was healed despite the filter\n{_report(r)}"
    assert "StaleProxyCacheWarning" in r.stderr, f"escalated to something unnamed\n{_report(r)}"


def _write_optional_binding(tmp_path):
    """Write a binding that exists only in a process in which its C symbol exists.

    The two tests below share this helper, and they reach the same absent binding by two routes. On the dispatcher
    route the guard finds the alias. On the compile-time constant route there is no alias to find. The comparison is
    meaningful only if the binding on either side of it is identical, so the helper writes it once.
    """
    (tmp_path / "fakelib.py").write_text(textwrap.dedent('''
        """Stands in for a C library whose symbol is present in one process and gone in the next."""
        import os


        class _Lib:
            pass


        lib = _Lib()
        if os.environ["SYMBOL_PRESENT"] == "1":
            lib.scale = object()
    '''), encoding="utf-8")
    (tmp_path / "optional_binding.py").write_text(textwrap.dedent('''
        """A binding that exists only when its C symbol does."""
        from numba import types
        from numbox.core.proxy.proxy import proxy_if_available

        from fakelib import lib

        SIG = types.float64(types.float64)


        @proxy_if_available(lib, SIG, jit_options={"cache": True})
        def scale(x):
            return x * 2.0
    '''), encoding="utf-8")


def test_a_binding_that_disappeared_is_discarded_rather_than_called(tmp_path):
    """A ``proxy_if_available`` binding that is absent leaves its alias resolvable, and that is the dangerous case.

    The absent path registers a trap under the alias that a warm caller embedded, so the symbol resolves and the
    object appears loadable. Calling the object is the worst available outcome. The trap raises inside a ``@cfunc``,
    numba discards the exception at the C boundary and returns zero, and the caller computes with the zero and exits
    successfully. If the guard discards the entry instead, the caller reaches the same typing error that a cold cache
    gives, and this test verifies that outcome.
    """
    _write_optional_binding(tmp_path)
    (tmp_path / "optional_caller.py").write_text(textwrap.dedent('''
        from numba import njit

        from optional_binding import scale


        @njit(cache=True)
        def call_scale(x):
            return scale(x) + 1.0
    '''), encoding="utf-8")
    probe = tmp_path / "optional_probe.py"
    probe.write_text(textwrap.dedent('''
        import warnings

        import optional_binding  # noqa: F401

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from optional_caller import call_scale
            try:
                print("RESULT", call_scale(5.0), flush=True)
            except Exception as exc:
                print("RAISED", type(exc).__name__, flush=True)
        print("WARNINGS", " | ".join(f"{w.category.__name__}" for w in caught))
        print("DONE ok", flush=True)
    '''), encoding="utf-8")

    env = _probe_env(tmp_path, "nbcache")
    env["SYMBOL_PRESENT"] = "1"
    assert _warm(probe, env)["RESULT"] == "11.0"

    env["SYMBOL_PRESENT"] = "0"
    fields = _fields(_run_probe(probe, env))
    assert "RESULT" not in fields, f"the vanished binding was called and returned {fields.get('RESULT')}"
    assert fields["RAISED"] == "TypingError", f"failed in an unexpected way: {fields['RAISED']}"
    assert "StaleProxyCacheWarning" in fields["WARNINGS"], f"the discard was not announced: {fields['WARNINGS']}"


def test_a_const_reference_caller_calls_a_binding_that_disappeared(tmp_path):
    """Characterization: the invariant of the test above does not hold on the const route.

    Suppose the same absent binding is reached as a compile-time constant rather than through its dispatcher. The
    machine code of the body is then linked into the caller's own object, so that object names no ``numbox_pxy_``
    alias. The guard takes its early exit, and the warm entry loads in a process in which the binding does not
    exist. The caller returns the number of the absent binding, while the identical source against a cold cache
    raises the typing error that the test above verifies. Warm and cold runs therefore disagree about whether the
    binding can be called. Strict mode does not help either, because the documented way to learn that an entry went
    stale never detects this one.

    This test records a known gap, not a desired outcome, so a passing run is not an endorsement of it. The defect is
    in the constant lowering that links the body in, and the fix is left to separate work. If that lowering ever
    changes to discard such a caller, the warm run will raise as the cold one does, and the assertions below will
    fail. That failure reports that the gap is closed, not a regression.

    The test is useful because a reader may add this shape deliberately, in the belief that it is defensive. An
    unguarded ``scale.as_func`` raises ``AttributeError`` at import if the binding is absent. A caller defined inside
    the ``hasattr`` statement that the proxy docstring shows is left undefined in that case. Both forms are safe.
    Demoting that guard to a conditional expression turns a visible import failure into a silent wrong answer. A body
    that calls a real C symbol is worse still, because it crashes with a bare segfault and an empty stderr. The body
    here therefore calls no C symbol: the deterministic form of the same gap is the one that a test can assert on.
    """
    _write_optional_binding(tmp_path)
    (tmp_path / "const_optional_caller.py").write_text(textwrap.dedent('''
        """A cached caller reading the binding's ``.as_func`` through a guard demoted to a conditional expression,
        so the module imports and the caller is defined even where the binding is not."""
        from numba import njit

        from optional_binding import scale

        SCALE_AS_FUNC = scale.as_func if hasattr(scale, "as_func") else None


        @njit(cache=True)
        def const_scale(x):
            return SCALE_AS_FUNC(x) + 1.0
    '''), encoding="utf-8")
    probe = tmp_path / "const_optional_probe.py"
    probe.write_text(textwrap.dedent('''
        import warnings

        import optional_binding  # noqa: F401

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from const_optional_caller import const_scale
            try:
                print("RESULT", const_scale(5.0), flush=True)
            except Exception as exc:
                print("RAISED", type(exc).__name__, flush=True)
        hits = sum(const_scale.stats.cache_hits.values())
        misses = sum(const_scale.stats.cache_misses.values())
        print("STATS", "served" if hits and not misses else "compiled" if misses and not hits else f"{hits}/{misses}")
        print("WARNINGS", " | ".join(f"{w.category.__name__}" for w in caught))
        print("DONE ok", flush=True)
    '''), encoding="utf-8")

    env = _probe_env(tmp_path, "nbcache")
    env["SYMBOL_PRESENT"] = "1"
    assert _warm(probe, env)["RESULT"] == "11.0"

    env["SYMBOL_PRESENT"] = "0"
    warm = _fields(_run_probe(probe, env))

    cold_env = _probe_env(tmp_path, "nbcache_absent_cold")
    cold_env["SYMBOL_PRESENT"] = "0"
    cold = _fields(_run_probe(probe, cold_env))
    assert cold.get("RAISED") == "TypingError", f"a cold cache did not reach the clean typing error: {cold}"

    assert warm.get("RESULT") == "11.0", f"the warm run stopped serving the vanished binding: {warm}"
    assert warm["STATS"] == "served", f"that number was recompiled rather than loaded from the cache: {warm}"
    assert "StaleProxyCacheWarning" not in warm["WARNINGS"], f"the guard saw the const route after all: {warm}"

    strict_env = dict(env)
    strict_env["NUMBOX_PROXY_CACHE_STRICT"] = "1"
    strict = _run_probe(probe, strict_env)
    assert strict.returncode == 0, f"strict mode caught the entry the default missed\n{_report(strict)}"
    assert _fields(strict)["RESULT"] == "11.0", f"strict mode changed the answer without stopping\n{_report(strict)}"


def test_the_mach_o_reader_strips_the_leading_underscore(tmp_path):
    """The scenarios above can reach the Mach-O reader only on macOS, so this test drives the reader directly against
    a real Mach-O object emitted by the same LLVM that numba compiles with.

    The test checks the underscore that Mach-O puts in front of every C symbol. A reader that forgets to strip it
    compares ``_numbox_pxy_...`` against the prefix, matches nothing, and accepts every stale object. That outcome is
    indistinguishable from a run with nothing to do, on the one platform on which no one can observe the failure.
    """
    emit = tmp_path / "emit_macho.py"
    emit.write_text(textwrap.dedent('''
        import platform
        import sys

        from llvmlite import binding as ll
        from llvmlite import ir

        arch = {"aarch64": "arm64", "AMD64": "x86_64"}.get(platform.machine(), platform.machine())
        triple = f"{arch}-apple-darwin"
        try:
            ll.initialize_all_targets()
            ll.initialize_all_asmprinters()
            target = ll.Target.from_triple(triple)
        except Exception as exc:
            print("SKIP", triple, exc)
            sys.exit(0)

        mod = ir.Module(name="fixture")
        mod.triple = triple
        fnty = ir.FunctionType(ir.DoubleType(), [ir.DoubleType()])
        callee = ir.Function(mod, fnty, name="numbox_pxy_probe_0123456789abcdef")
        fn = ir.Function(mod, fnty, name="caller")
        builder = ir.IRBuilder(fn.append_basic_block("entry"))
        builder.ret(builder.call(callee, [fn.args[0]]))

        tm = target.create_target_machine(reloc="static", codemodel="large")
        obj = tm.emit_object(ll.parse_assembly(str(mod)))
        with open(sys.argv[1], "wb") as f:
            f.write(obj)
        print("OK", len(obj))
    '''), encoding="utf-8")

    obj_path = tmp_path / "fixture.o"
    r = subprocess.run([sys.executable, str(emit), str(obj_path)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    if r.stdout.startswith("SKIP"):
        pytest.skip(f"llvmlite cannot emit Mach-O here: {r.stdout.strip()}")

    obj = obj_path.read_bytes()
    assert obj[:4] == b"\xcf\xfa\xed\xfe", f"not a 64-bit little-endian Mach-O: {obj[:4]!r}"
    found = _undefined_symbols(obj)
    assert "numbox_pxy_probe_0123456789abcdef" in found, f"the aliased import was not found: {sorted(found)}"
    assert "_numbox_pxy_probe_0123456789abcdef" not in found, "the leading underscore was left on"


def test_the_reader_recognises_the_objects_numba_actually_caches(tmp_path):
    """The reader must be able to parse whatever container numba caches on this platform.

    Every other test here reaches the reader only through a stale entry, so each of them proves that the guard works
    by observing that a crash does *not* happen. That leaves the most basic premise of the guard unverified: that the
    reader can parse the format of the objects that numba passes to it. A reader that recognizes nothing fails
    silently and completely. It returns no symbols, finds no stale alias, and passes every object exactly as if the
    guard were absent. The only evidence is then a segfault on whichever platform stopped matching, and nothing names
    the cause.

    The platform does not show which branch matches. The ELF reader parses the objects on the Windows jobs, because
    LLVM rewrites a COFF target to ELF for the JIT, so a ``get_object_format`` result of COFF does not describe the
    bytes. The test asserts the outcome rather than that reasoning, with a warm re-run in which the reader records
    each object that it receives. If the premise ever stops holding, the failure is then a named failure here rather
    than a crash somewhere else. The test needs no stale entry and no edit, so it remains valid even if the crash
    scenario itself ever stops reproducing.
    """
    probe = _write_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")
    _warm(probe, env)

    reader = tmp_path / "reader_probe.py"
    reader.write_text(textwrap.dedent('''
        """Re-run the warm callers with the format reader recording every object it is asked about."""
        import numbox.core.proxy.proxy as proxy_mod

        seen = []
        read_symbols = proxy_mod._undefined_symbols


        def recording(object_code):
            found = read_symbols(object_code)
            seen.append((object_code[:4].hex(), any(s.startswith(proxy_mod._ALIAS_PREFIX) for s in found)))
            return found


        proxy_mod._undefined_symbols = recording

        from cached_callers import call_scale
        print("RESULT", call_scale(5.0), flush=True)
        print("INSPECTED", len(seen), flush=True)
        print("WITH_ALIAS", sum(1 for _magic, found_alias in seen if found_alias), flush=True)
        print("MAGIC", ",".join(sorted({magic for magic, _found_alias in seen})) or "-", flush=True)
        print("DONE ok", flush=True)
    '''), encoding="utf-8")

    fields = _fields(_run_probe(reader, env))
    assert fields["RESULT"] == _OLD_BODY["call_scale"], f"the warm caller did not run: {fields['RESULT']}"
    assert int(fields["INSPECTED"]) > 0, (
        "no alias-bearing object reached the reader, so nothing here was tested: the callers were recompiled "
        "rather than served, or the guard is not installed"
    )
    assert int(fields["WITH_ALIAS"]) > 0, (
        f"the reader parsed no alias out of any object numba cached here, so it passes every stale entry "
        f"untouched: leading magic bytes {fields['MAGIC']}"
    )


def test_strict_mode_raises_on_a_stale_alias_before_healing(tmp_path):
    """``NUMBOX_PROXY_CACHE_STRICT`` turns the silent self-repair into a hard error that leaves the entry unchanged.

    The default is to discard the stale entry, warn, and recompile in place. A caller who is debugging the
    misbehavior of a cache wants the opposite: the run stops at the first stale entry, names it, and leaves it on
    disk for inspection. The strict run must therefore fail with ``StaleProxyCacheError`` named on stderr, rather
    than exit cleanly. A second run with the setting off then proves that the strict run did *not* recompile. That
    run still finds the same entry stale and repairs it. If the strict run had recompiled, this run would load the
    new body silently, with no warning.
    """
    probe = _write_scenario(tmp_path)
    _warm(probe, _probe_env(tmp_path, "nbcache"))
    _edit_body(tmp_path)

    strict_env = _probe_env(tmp_path, "nbcache")
    strict_env["NUMBOX_PROXY_CACHE_STRICT"] = "1"
    r = _run_probe(probe, strict_env)
    assert r.returncode != 0, f"strict mode healed the stale entry instead of aborting\n{_report(r)}"
    assert "StaleProxyCacheError" in r.stderr, f"the abort was not the named strict error\n{_report(r)}"

    healed = _run_probe(probe, _probe_env(tmp_path, "nbcache"))
    assert healed.returncode == 0, f"the entry strict left behind did not heal with the knob off\n{_report(healed)}"
    fields = _fields(healed)
    assert _results(fields) == _NEW_BODY, _results(fields)
    assert "StaleProxyCacheWarning" in fields["WARNINGS"], (
        f"strict recompiled the entry after all -- nothing was left stale to heal: {fields['WARNINGS']}"
    )


def test_strict_mode_is_quiet_on_a_healthy_warm_cache(tmp_path):
    """Strict mode must not raise an error on a healthy warm cache.

    On an unedited warm cache, every proxy caller imports a resolvable alias. The prefix-present-but-no-import check
    therefore never triggers, and no error is raised. The run loads the cached bodies exactly as the default would,
    and neither strict error name appears on stderr."""
    probe = _write_scenario(tmp_path)
    _warm(probe, _probe_env(tmp_path, "nbcache"))

    strict_env = _probe_env(tmp_path, "nbcache")
    strict_env["NUMBOX_PROXY_CACHE_STRICT"] = "1"
    r = _run_probe(probe, strict_env)
    assert r.returncode == 0, f"strict mode falsely rejected a healthy warm cache\n{_report(r)}"
    fields = _fields(r)
    assert _stats(fields) == dict.fromkeys(_HEALED, "served"), f"a healthy strict run recompiled: {_stats(fields)}"
    assert _results(fields) == _OLD_BODY, _results(fields)
    assert "Error" not in r.stderr, f"strict emitted an error on a healthy cache\n{_report(r)}"


def test_strict_mode_raises_when_a_payload_cannot_be_read(tmp_path, monkeypatch):
    """The abstention path, a payload whose shape the reader cannot unpack, fails open by default and raises when the
    strict setting is on. This test is the counterpart of
    ``test_an_unrecognised_cache_payload_is_handed_on_untouched``: the same payloads that are passed on unchanged with
    the setting off must abort with the named strict error when it is on."""
    from numbox.core.proxy.proxy import UnvalidatedProxyCacheError, _guarded_rebuild

    class _Impl:
        filename_base = "unreadable"

    def original(self, target_context, payload):
        return "loaded"

    rebuild = _guarded_rebuild(original, lambda payload: payload[0])
    monkeypatch.setenv("NUMBOX_PROXY_CACHE_STRICT", "1")
    for payload in ({"keyword": "style"}, object(), ()):
        with pytest.raises(UnvalidatedProxyCacheError):
            rebuild(_Impl(), None, payload)


def test_strict_mode_still_loads_a_well_formed_object_with_no_stale_alias(tmp_path, monkeypatch):
    """A cleanly read object with the prefix but no matching imported alias loads, whether strict mode is on or off.

    Such an object cannot be distinguished from a healthy, entirely non-proxy function that merely embeds an
    alias-shaped string constant (see ``test_..._embedding_the_prefix`` below). Strict mode therefore does not raise
    on it: only a payload that *raises* when read is a validation gap. CI instead detects a reader that cannot parse
    a real numba object, through ``test_the_reader_recognises_the_objects_numba_actually_caches``.
    """
    from numbox.core.proxy.proxy import _guarded_rebuild

    class _Impl:
        filename_base = "prefix-without-import"

    def original(self, target_context, payload):
        return "loaded"

    # The blob carries the alias prefix, so the fast path does not short-circuit. The prefix is data in a blob that is
    # neither ELF nor Mach-O, so the reader parses it cleanly and finds no imported alias.
    blob = b"numbox_pxy_scale_deadbeefdeadbeef" + b"\x00" * 64
    payload = ("libname", "object", (blob,))
    rebuild = _guarded_rebuild(original, lambda p: p)

    assert rebuild(_Impl(), None, payload) == "loaded", "the default must load such an object untouched"
    monkeypatch.setenv("NUMBOX_PROXY_CACHE_STRICT", "1")
    assert rebuild(_Impl(), None, payload) == "loaded", "strict must not abort on a cleanly-read prefix-bearing object"


def test_strict_mode_does_not_abort_on_a_healthy_object_embedding_the_prefix(tmp_path):
    """This test covers the case above end to end. A real cached function with no proxy involvement carries the alias
    prefix as a runtime string constant, and it must load under strict mode without a false abort.

    The guard runs only in a process that imports numbox, so the runner imports it. A person who debugs the proxy
    cache is exactly the user who enables strict mode. Such a user is also the most likely to have jitted code that
    handles ``numbox_pxy_`` strings. The function returns the literal, so that the literal is emitted into the
    object rather than folded away.
    """
    (tmp_path / "prefix_literal_mod.py").write_text(textwrap.dedent('''
        from numba import njit


        @njit(cache=True)
        def has_literal(i):
            label = "numbox_pxy_scale_deadbeefdeadbeef"
            return label[i]
    '''), encoding="utf-8")
    runner = tmp_path / "prefix_literal_run.py"
    runner.write_text(textwrap.dedent('''
        import numbox.core.proxy.proxy  # noqa: F401 -- importing numbox is what installs the cache guard
        from prefix_literal_mod import has_literal
        print("RESULT", has_literal(0), flush=True)
        print("DONE ok", flush=True)
    '''), encoding="utf-8")

    env = _probe_env(tmp_path, "nbcache")
    assert _warm(runner, env)["RESULT"] == "n"

    strict_env = dict(env)
    strict_env["NUMBOX_PROXY_CACHE_STRICT"] = "1"
    r = _run_probe(runner, strict_env)
    assert r.returncode == 0, f"strict mode false-aborted on a healthy non-proxy cache\n{_report(r)}"
    assert _fields(r)["RESULT"] == "n", _report(r)


@pytest.mark.parametrize("value,expected", [
    (None, False), ("", False), ("0", False), ("false", False), ("FALSE", False), ("no", False),
    ("off", False), ("OFF", False), (" 0 ", False),
    ("1", True), ("true", True), ("yes", True), ("on", True), ("2", True), ("anything", True),
])
def test_strict_cache_mode_reads_the_env_truthily(value, expected, monkeypatch):
    """The values that turn the setting off are a documented contract, on which a user relies to disable it per run
    (``=0``). The test checks each value. Dropping an element of the falsey set would turn
    ``export NUMBOX_PROXY_CACHE_STRICT=0`` into hard cache-load errors. This test then fails, whereas a suite that
    only ever sets the variable to ``1`` would still pass."""
    from numbox.core.configurations import _strict_cache_mode

    if value is None:
        monkeypatch.delenv("NUMBOX_PROXY_CACHE_STRICT", raising=False)
    else:
        monkeypatch.setenv("NUMBOX_PROXY_CACHE_STRICT", value)
    assert _strict_cache_mode() is expected


def test_an_unrecognised_cache_payload_is_handed_on_untouched(tmp_path):
    """The shape of the payload is the assumption most likely to change under this guard. If the guard cannot read a
    shape, it must skip the validation and still perform the load. Every numba version in the supported range passes
    a tuple here."""
    from numbox.core.proxy.proxy import _guarded_rebuild

    class _Impl:
        filename_base = "unreadable"

    def original(self, target_context, payload):
        return "loaded"

    rebuild = _guarded_rebuild(original, lambda payload: payload[0])
    for payload in ({"keyword": "style"}, object(), ()):
        assert rebuild(_Impl(), None, payload) == "loaded", f"{type(payload).__name__} payload broke the load"


def test_a_guvectorize_caller_heals_too(tmp_path):
    """A gufunc reaches the cache through a second implementation, whose payload is shaped differently.

    ``@njit`` stores the serialized library inside a compile result, while the wrapper library of a gufunc is the
    payload itself. The two are therefore unwrapped differently, and only this test exercises the second
    implementation. A gufunc also compiles eagerly at import rather than on first call, which is why the warning
    capture has to wrap the import.
    """
    _write_binding(tmp_path / _EDITED, "scale", "x * 2.0")
    (tmp_path / "gufunc_caller.py").write_text(textwrap.dedent('''
        """A cached gufunc calling the proxied body, in a file the edit never touches."""
        from numba import guvectorize
        from numba.core.types import float64

        from proxied_binding import scale


        @guvectorize([(float64[:], float64[:])], "(n)->(n)", cache=True, nopython=True)
        def gu_scale(x, out):
            for i in range(x.shape[0]):
                out[i] = scale(x[i]) + 1.0
    '''), encoding="utf-8")
    probe = tmp_path / "gufunc_probe.py"
    probe.write_text(textwrap.dedent('''
        import warnings

        import numpy as np

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from gufunc_caller import gu_scale
            result = gu_scale(np.array([5.0]))[0]
        print("RESULT_gu_scale", result, flush=True)
        print("WARNINGS", " | ".join(f"{w.category.__name__}: {w.message}".replace("\\n", " ") for w in caught))
        print("DONE ok", flush=True)
    '''), encoding="utf-8")

    env = _probe_env(tmp_path, "nbcache")
    assert _warm(probe, env)["RESULT_gu_scale"] == "11.0"

    _edit_body(tmp_path)
    fields = _fields(_run_probe(probe, env))
    assert fields["RESULT_gu_scale"] == "16.0", f"the gufunc served a stale body: {fields['RESULT_gu_scale']}"
    assert "StaleProxyCacheWarning" in fields["WARNINGS"], f"the discard was not announced: {fields['WARNINGS']}"
