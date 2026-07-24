"""Cross-process regression tests for a stale ``@proxy`` cfunc alias in a warm numba cache.

``@proxy`` publishes each proxied body's cfunc wrapper under a process-stable alias whose name folds a fingerprint of
that body, so editing the body renames the alias. A ``cache=True`` caller living in another file keys only on its own
source -- numba's key says nothing about its callees -- so after such an edit the caller still cache-hits and loads an
object referencing a symbol this process never registered. RuntimeDyld resolves an object's externals in one batch, so
that single missing name zeroes *every* external relocation in the object, and the process dies inside the CPython
argument-unpacking wrapper before any user code runs: a bare segfault with an empty stderr. The engine keeps that
failure, so the next cached object loaded afterwards -- a sound one will do -- aborts the process instead, naming the
alias that no longer exists.
Both shapes are landmines for anyone who edits a proxied binding without first clearing the cache (fork issue #73).

The target behaviour is that the loader notices the unregistered alias, discards that one entry, warns, and recompiles.
Each scenario therefore also carries a second binding that is never edited, with its own caller in the same file and
the same cache directory; that caller must still be served from cache and draw no warning, so a fix that switches
caching off or invalidates the whole directory cannot pass. The premise -- warm entries really are served, and the edit
re-keys one binding and only one -- is pinned separately, by tests that never go near the crash.
"""
import os
import subprocess
import sys
import textwrap

import pytest

_EDITED = "proxied_binding.py"
_HEALED = {"call_scale": "compiled", "call_scale2": "compiled", "call_offset": "served", "plain": "served"}
_OLD_BODY = {"call_scale": "11.0", "call_scale2": "12.0", "call_offset": "108.0", "plain": "12.0"}
_NEW_BODY = {"call_scale": "16.0", "call_scale2": "17.0", "call_offset": "108.0", "plain": "12.0"}


def _write_binding(path, name, expr):
    """Write a one-function proxied binding module.

    A rewrite bumps mtime forward by hand rather than trusting the clock: the new body is the same length as the old
    one, and both numba's source stamp and CPython's pyc validation compare ``(st_mtime, st_size)``, so where mtime
    granularity is coarse a same-second rewrite is indistinguishable from the original and the edit would never be
    seen. The padding above the decorator is deliberate too -- the cache anchor requires the decorator line to be the
    function's ``co_firstlineno``, and it may not sit above the generated wrapper's own decorator line.
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
    """Lay the scenario out on disk and hand back the probe script.

    The two bindings get separate files so the control is untouched at file level as well: nothing about it, alias or
    source stamp, can move when the other body is edited. The three callers share one file to make the control as
    tight as possible: same module, same cache directory, same run. Two callers of the edited body rather than one is
    what pins that *every* stale entry is discarded, not just the first the loader reaches, and ``plain`` -- cached,
    but referencing no alias at all -- is the control for the other direction: it takes the validator's early exit,
    and must survive untouched.
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
    """Rewrite the edited binding's body, which is the whole trigger: same file, same length, different alias."""
    _write_binding(tmp_path / _EDITED, "scale", "x * 3.0")


def _probe_env(tmp_path, cache_name, precompile=False):
    """Environment for one probe run.

    ``NUMBA_CACHE_DIR`` also relocates numbox's own anchor files, so a per-test directory keeps the whole scenario out
    of the developer's shared cache. The parent's ``sys.path`` is forwarded because the probe imports numbox from
    wherever this test found it, and ``tmp_path`` is named explicitly rather than relying on the script directory,
    which ``PYTHONSAFEPATH`` suppresses. Bytecode writing is off because a pyc validated on ``(st_mtime, st_size)``
    is the other way a same-length edit can go unnoticed. The inherited warning and numba knobs are dropped because
    each of them can empty the scenario out: they decide whether an unrelated warning ends the probe, whether
    anything is compiled at all, and -- at ``NUMBA_OPT=0``, where the proxy call is not folded to a plain extern
    reference -- whether numba is willing to cache the callers in the first place.
    """
    env = dict(os.environ)
    for hostile in ("PYTHONWARNINGS", "PYTHONDEVMODE", "NUMBA_DISABLE_JIT", "NUMBA_OPT"):
        env.pop(hostile, None)
    env["NUMBA_CACHE_DIR"] = str(tmp_path / cache_name)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path)] + sys.path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PRECOMPILE"] = "1" if precompile else "0"
    return env


def _run_probe(probe, env):
    """Run a probe to completion and hand back the whole result: these scenarios expect the child to die, so the
    caller, not this helper, decides what an acceptable return code is. The timeout is the only defence against a
    crashed child that never exits -- a fault can park a process behind a crash reporter."""
    return subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=300,
    )


def _report(r):
    """Full transcript, so a failure shows which crash shape happened -- or that neither did."""
    return f"rc={r.returncode}\n--- stdout ---\n{r.stdout}--- stderr ---\n{r.stderr}"


def _fields(r):
    """The probe's ``KEY value`` lines. ``DONE`` is absent whenever the child died mid-run."""
    fields = dict(line.split(" ", 1) for line in r.stdout.splitlines() if " " in line)
    assert fields.get("DONE") == "ok", f"the probe did not run to completion\n{_report(r)}"
    return fields


def _warm(probe, env):
    """Warm the cache and return the probe's fields.

    A failure here is a broken scenario rather than the behaviour under test, so it is raised as an error: an
    assertion would be reported as a behavioural failure, and a dead harness must not be mistaken for one.
    """
    r = _run_probe(probe, env)
    if r.returncode != 0 or "DONE ok" not in r.stdout:
        raise RuntimeError(f"warming the cache failed\n{_report(r)}")
    return _fields(r)


def _stats(fields):
    return {name: fields[f"STATS_{name}"] for name in _HEALED}


def _results(fields):
    return {name: fields[f"RESULT_{name}"] for name in _HEALED}


def test_unedited_rerun_is_served_from_cache(tmp_path):
    """Nothing below proves anything unless a warm entry really is served: a caller that recompiled anyway could
    never be handed a stale object. One cold run, then an identical one against the same cache directory."""
    probe = _write_scenario(tmp_path)
    env = _probe_env(tmp_path, "nbcache")

    cold = _warm(probe, env)
    assert _stats(cold) == dict.fromkeys(_HEALED, "compiled"), _stats(cold)
    assert _results(cold) == _OLD_BODY, _results(cold)

    warm = _warm(probe, env)
    assert _stats(warm) == dict.fromkeys(_HEALED, "served"), f"an unchanged re-run recompiled: {_stats(warm)}"
    assert _results(warm) == _results(cold), _results(warm)


def test_editing_a_proxied_body_renames_only_that_binding_alias(tmp_path):
    """The edit re-keys exactly one binding, which is what makes one caller stale and leaves the control valid.

    The edited body is run against a *fresh* cache directory so this stays clear of the crash: it answers what the new
    body is worth, and which aliases moved, without a stale entry in the way.
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
    """Reaching a warm caller after the proxied body was edited must recompile it, not crash.

    Unfixed, calling it segfaults on the first call: the one missing alias zeroed every external relocation in the
    loaded object, so control never reaches the compiled body -- stdout stops dead after the callers are imported and
    stderr is empty, which is as undiagnosable as a failure gets. Compiling the callers explicitly instead moves the
    failure: the stale load itself is silent, because nothing has jumped into the object yet, and the next cached load
    of any object aborts the process outright naming the retired alias, because the execution engine keeps the first
    failure.

    Fixed, both callers of the edited body are discarded and recompiled against it while the caller of the untouched
    binding is still served, and the discard is announced by a warning that names the alias nobody registered. The
    last run pins the heal as a one-time recompile: an implementation that refused to serve those entries without
    replacing them would leave every later process paying full compilation again.
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


def test_the_discard_can_be_escalated_to_an_error(tmp_path):
    """Healing silently is the default, not the only option: a caller who would rather know can escalate.

    The warning is a named class reachable from the public module path, so a filter can single it out without
    catching every ``RuntimeWarning``. Escalation aborts before the heal, which leaves the stale entry on disk for
    inspection rather than replacing it.
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


def test_a_binding_that_disappeared_is_discarded_rather_than_called(tmp_path):
    """A ``proxy_if_available`` binding that is gone leaves its alias resolvable, and that is the dangerous case.

    The absent path registers a trap under the alias a warm caller baked in, so the symbol resolves and the object
    looks loadable. Calling it is the worst available outcome: the trap raises inside a ``@cfunc``, numba swallows
    that at the C boundary and returns zero, and the caller computes on the zero and exits successfully. Discarding
    the entry instead reaches the same clean typing error a cold cache gives, which is what this pins.
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


def test_an_unrecognised_cache_payload_is_handed_on_untouched(tmp_path):
    """The payload's shape is the assumption most likely to change under this guard, so a shape it cannot read
    must cost only the validation -- never the load. Every numba in the supported range passes a tuple here."""
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

    ``@njit`` stores the serialized library inside a compile result; a gufunc's wrapper library is the payload, so
    the two are unwrapped differently and only this exercises the second one. It also compiles eagerly at import
    rather than on first call, which is why the warning capture has to wrap the import.
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
